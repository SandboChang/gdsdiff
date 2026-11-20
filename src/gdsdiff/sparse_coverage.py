"""Exact hole correction when the two layouts have different record layouts."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .config import LayerSpec
from .exact_diff_markdown import PolygonInventoryDelta, _canonical_inventory_points, _inventory_delta_for_indices
from .exact_oracle import (
    DiffFragment,
    ExactOracleResult,
    LayerExactDiff,
    _ALIGNED_SPARSE_MAX_CHANGED,
    _BooleanEvidenceError,
    _checked_gdstk_boolean,
    _fragment_sort_key,
    _independent_bbox_components,
    _layer_origin,
    _to_gdstk_polygons,
)
from .geometry import PolygonBuffer, polygon_bbox, polygon_twice_area
from .spatial import BBox

_MAX_CANDIDATES = 2 * _ALIGNED_SPARSE_MAX_CHANGED
_MAX_BBOX_CHECKS = 1_000_000_000
_MAX_CONTEXT_RECORDS = 20_000
_MAX_CONTEXT_VERTICES = 1_000_000
_MAX_RECOVERY_RECORDS = 2048
_VERTICES_PER_BATCH = 262_144
_KEY_FIELDS = ("left", "bottom", "right", "top", "vertices", "flags")
_KEY_DTYPE = np.dtype([(field, np.int64) for field in _KEY_FIELDS])
_COVERAGE_SCOPE = "full-layout-verified-context-reduction"


class _SparseCoverageDeclined(ValueError):
    pass


class _SparseCandidateMismatch(_SparseCoverageDeclined):
    pass


@dataclass(frozen=True)
class _VerifiedSparseCoverage:
    exact_result: ExactOracleResult
    inventory: PolygonInventoryDelta
    proof_s: float
    boolean_s: float
    old_retained: int
    new_retained: int
    partitioned_directions: int
    reflected_components: int

    def to_json(self) -> dict[str, object]:
        return {
            "coverage_scope": _COVERAGE_SCOPE,
            "exterior_proof": "exact-integer-record-multiset",
            "old_retained_polygon_count": self.old_retained,
            "new_retained_polygon_count": self.new_retained,
            "partitioned_direction_count": self.partitioned_directions,
            "reflected_component_count": self.reflected_components,
        }


def _record_delta_bounds(old: PolygonBuffer, new: PolygonBuffer, delta) -> dict[LayerSpec, set[BBox]]:
    if not 0 < delta.record_count <= _MAX_CANDIDATES:
        raise _SparseCoverageDeclined("record delta exceeds sparse work limits")
    bounds: dict[LayerSpec, set[BBox]] = defaultdict(set)
    for row in range(delta.record_count):
        direction = int(delta.directions[row])
        if direction not in (1, 2):
            raise _SparseCandidateMismatch("invalid record delta direction")
        buffer = old if direction == 1 else new
        index = int(delta.representative_indices[row])
        if not 0 <= index < buffer.polygon_count:
            raise _SparseCandidateMismatch("invalid record delta index")
        layer = buffer.layer_table.layers[int(buffer.layer_ids[index])]
        bounds[layer].add(tuple(int(value) for value in buffer.bboxes[index]))
    return bounds


def _run_verified_sparse_exact_oracle(
    old: PolygonBuffer,
    new: PolygonBuffer,
    *,
    candidate_bounds: dict[LayerSpec, set[BBox]] | None = None,
) -> _VerifiedSparseCoverage:
    proof_start = perf_counter()
    layer_ids = old.layer_table.index_by_layer, new.layer_table.index_by_layer
    layers = sorted(layer_ids[0].keys() | layer_ids[1].keys())
    if candidate_bounds is not None and candidate_bounds.keys() - set(layers):
        raise _SparseCandidateMismatch("record delta contains an unselected layer")

    retained = {}
    bbox_checks = context_vertices = candidate_count = 0
    for layer in layers:
        old_indices = _layer_indices(old, layer_ids[0].get(layer))
        new_indices = _layer_indices(new, layer_ids[1].get(layer))
        if candidate_bounds is None:
            # An independent CPU audit must not reuse GPU candidates or input buffers.
            seeds = _different_record_bounds(old, old_indices, new, new_indices)
        else:
            seeds = candidate_bounds.get(layer, set())
        candidate_count += len(seeds)
        bbox_checks += len(seeds) * (len(old_indices) + len(new_indices))
        if candidate_count > _MAX_CANDIDATES or bbox_checks > _MAX_BBOX_CHECKS:
            raise _SparseCoverageDeclined("candidate or context search exceeds sparse work limits")
        old_keep = _touching_mask(old, old_indices, seeds)
        new_keep = _touching_mask(new, new_indices, seeds)
        if candidate_bounds is not None and _different_record_bounds(
            old, old_indices[~old_keep], new, new_indices[~new_keep],
        ):
            raise _SparseCandidateMismatch("records outside GPU candidate bounds differ")

        old_context, new_context = old_indices[old_keep], new_indices[new_keep]
        if len(old_context) + len(new_context) > _MAX_CONTEXT_RECORDS:
            raise _SparseCoverageDeclined("retained polygon count exceeds sparse work limits")
        for buffer, indices in ((old, old_context), (new, new_context)):
            if buffer.flags[indices].any():
                raise _SparseCoverageDeclined("retained context contains flagged polygons")
            context_vertices += int(_record_lengths(buffer, indices).sum())
        if context_vertices > _MAX_CONTEXT_VERTICES:
            raise _SparseCoverageDeclined("retained vertex count exceeds sparse work limits")
        inventory = _inventory_delta_for_indices(old, old_context, new, new_context)
        actual_bounds = {record.bbox for record in inventory.removed + inventory.added}
        # An unmatched record that extends beyond the candidates could be
        # masked by context we omitted, even when the exterior records match.
        if not actual_bounds <= seeds:
            raise _SparseCandidateMismatch("an unmatched polygon is not fully covered by a candidate")
        retained[layer] = old_context, new_context, inventory

    inventories = tuple(items[2] for items in retained.values())
    inventory = PolygonInventoryDelta(
        removed=tuple(record for layer_inventory in inventories for record in layer_inventory.removed),
        added=tuple(record for layer_inventory in inventories for record in layer_inventory.added),
    )
    if max(len(inventory.removed), len(inventory.added)) > _ALIGNED_SPARSE_MAX_CHANGED:
        raise _SparseCoverageDeclined("verified record changes exceed sparse work limits")
    proof_s = perf_counter() - proof_start
    boolean_start = perf_counter()
    diffs = []
    partitioned = reflected = 0
    for layer, (old_indices, new_indices, layer_inventory) in retained.items():
        if not layer_inventory.changed_count:
            diffs.append(LayerExactDiff(layer))
            continue
        a = tuple(_record_points(old, index) for index in old_indices)
        b = tuple(_record_points(new, index) for index in new_indices)
        removed, partitioned_a, reflected_a = _directional_difference(layer, a, b)
        added, partitioned_b, reflected_b = _directional_difference(layer, b, a)
        partitioned += partitioned_a + partitioned_b
        reflected += reflected_a + reflected_b
        if _fragment_vertices(removed).keys() & _fragment_vertices(added).keys():
            raise _BooleanEvidenceError(f"gdstk {layer.layer}:{layer.datatype} emitted a loop in both directions")
        diffs.append(LayerExactDiff(layer, removed, added, tuple(sorted(removed + added, key=_fragment_sort_key))))

    return _VerifiedSparseCoverage(
        ExactOracleResult(tuple(diffs)), inventory, proof_s, perf_counter() - boolean_start,
        sum(len(items[0]) for items in retained.values()), sum(len(items[1]) for items in retained.values()),
        partitioned, reflected,
    )


def _touching_mask(buffer: PolygonBuffer, indices: np.ndarray, seeds: set[BBox]) -> np.ndarray:
    keep = np.zeros(len(indices), dtype=bool)
    if seeds and len(indices):
        x0, y0, x1, y1 = np.ascontiguousarray(buffer.bboxes[indices].T)
        for left, bottom, right, top in seeds:
            keep |= (x0 <= right) & (x1 >= left) & (y0 <= top) & (y1 >= bottom)
    return keep


def _layer_indices(buffer: PolygonBuffer, layer_id: int | None) -> np.ndarray:
    return np.empty(0, dtype=np.intp) if layer_id is None else np.flatnonzero(buffer.layer_ids == layer_id)


def _record_lengths(buffer: PolygonBuffer, indices: np.ndarray) -> np.ndarray:
    return (buffer.polygon_offsets[indices + 1] - buffer.polygon_offsets[indices]).astype(np.intp)


def _record_keys(buffer: PolygonBuffer, indices: np.ndarray) -> np.ndarray:
    keys = np.empty(len(indices), dtype=_KEY_DTYPE)
    for field, values in zip(_KEY_FIELDS[:4], buffer.bboxes[indices].T, strict=True):
        keys[field] = values
    keys["vertices"] = _record_lengths(buffer, indices)
    keys["flags"] = buffer.flags[indices]
    return keys


def _key_groups(sorted_keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not len(sorted_keys):
        return sorted_keys, np.empty(0, dtype=np.intp)
    starts = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1]
    return sorted_keys[starts], np.diff(starts, append=len(sorted_keys))


def _different_record_bounds(
    old: PolygonBuffer, old_indices: np.ndarray, new: PolygonBuffer, new_indices: np.ndarray,
) -> set[BBox]:
    """Find unequal record groups. Metadata narrows the join but cannot prove equality."""
    left, right = _record_keys(old, old_indices), _record_keys(new, new_indices)
    bounds: set[BBox] = set()
    if not np.array_equal(left, right):
        def order(keys):
            return np.lexsort(tuple(keys[field] for field in reversed(_KEY_FIELDS)))

        old_order, new_order = order(left), order(right)
        left, right = left[old_order], right[new_order]
        old_indices, new_indices = old_indices[old_order], new_indices[new_order]
        if not np.array_equal(left, right):
            old_keys, old_counts = _key_groups(left)
            new_keys, new_counts = _key_groups(right)
            _, a, b = np.intersect1d(old_keys, new_keys, assume_unique=True, return_indices=True)
            counts_equal = old_counts[a] == new_counts[b]
            old_matched = np.zeros(len(old_keys), dtype=bool)
            new_matched = np.zeros(len(new_keys), dtype=bool)
            old_matched[a[counts_equal]], new_matched[b[counts_equal]] = True, True
            if int(old_counts[~old_matched].sum() + new_counts[~new_matched].sum()) > _MAX_CANDIDATES:
                raise _SparseCoverageDeclined("too many unpaired records for sparse proof")
            for unmatched in (old_keys[~old_matched], new_keys[~new_matched]):
                bounds.update(tuple(int(k[field]) for field in _KEY_FIELDS[:4]) for k in unmatched)
            old_keep, new_keep = np.repeat(old_matched, old_counts), np.repeat(new_matched, new_counts)
            left = left[old_keep]
            old_indices, new_indices = old_indices[old_keep], new_indices[new_keep]

    old_extra, new_extra = Counter(), Counter()
    unequal_count = 0
    for a, b in _unequal_vertex_pairs(old, old_indices, new, new_indices, left["vertices"]):
        unequal_count += len(a)
        if unequal_count > _MAX_CANDIDATES:
            raise _SparseCoverageDeclined("too many unpaired vertex sequences for sparse proof")
        for buffer, indices, counter in ((old, a, old_extra), (new, b, new_extra)):
            for index in indices:
                points = _record_points(buffer, index)
                flags = int(buffer.flags[index])
                bbox = tuple(int(value) for value in buffer.bboxes[index])
                counter[(bbox, flags, points if flags else _canonical_inventory_points(points))] += 1
    bounds.update(key[0] for key in (old_extra - new_extra).keys() | (new_extra - old_extra).keys())
    return bounds


def _unequal_vertex_pairs(old, old_indices, new, new_indices, lengths):
    if not len(lengths):
        return
    if np.any(lengths <= 0):
        raise _SparseCoverageDeclined("sparse proof requires nonempty polygon records")
    ends = np.cumsum(lengths, dtype=np.intp)
    start = offset = 0
    while start < len(ends):
        stop = max(start + 1, int(np.searchsorted(ends, offset + _VERTICES_PER_BATCH, side="right")))
        counts = lengths[start:stop]
        starts = ends[start:stop] - offset - counts
        size = int(ends[stop - 1]) - offset
        a, b = old_indices[start:stop], new_indices[start:stop]
        x = _vertex_batch(old, a, counts, starts, size)
        y = _vertex_batch(new, b, counts, starts, size)
        unequal = x[:, 0] != y[:, 0]
        unequal |= x[:, 1] != y[:, 1]
        if unequal.any():
            records = np.logical_or.reduceat(unequal, starts)
            yield a[records], b[records]
        start, offset = stop, offset + size


def _vertex_batch(buffer, indices, counts, starts, size):
    offsets = buffer.polygon_offsets[indices].astype(np.intp)
    if offsets[-1] - offsets[0] == starts[-1] and np.all(offsets - offsets[0] == starts):
        return buffer.vertices_xy[offsets[0]:offsets[0] + size]
    flat = np.repeat(offsets - starts, counts) + np.arange(size, dtype=np.intp)
    return buffer.vertices_xy[flat]


def _record_points(buffer, index):
    return tuple((int(x), int(y)) for x, y in buffer.polygon_vertices(index))


def _fragment_vertices(fragments):
    return Counter(_canonical_inventory_points(fragment.points) for fragment in fragments)


def _directional_difference(layer, subject, clip):
    try:
        return _difference(layer, subject, clip), 0, 0
    except _BooleanEvidenceError:
        # Closed contact groups are disjoint, so combining their results needs no Boolean.
        if len(subject) + len(clip) > _MAX_RECOVERY_RECORDS:
            raise _BooleanEvidenceError("gdstk numerical recovery exceeds the sparse partition budget") from None
        groups = _independent_bbox_components(subject, clip)
        fragments = []
        reflected = 0
        for a, b in groups:
            if len(groups) > 1:
                try:
                    fragments.extend(_difference(layer, a, b))
                    continue
                except _BooleanEvidenceError:
                    pass
            first = _difference(layer, a, b, reflect=(1, -1))
            second = _difference(layer, a, b, reflect=(-1, 1))
            if _fragment_vertices(first) != _fragment_vertices(second):
                raise _BooleanEvidenceError(f"gdstk {layer.layer}:{layer.datatype} reflected vertex loops disagree")
            fragments.extend(first)
            reflected += 1
        return tuple(sorted(fragments, key=_fragment_sort_key)), 1, reflected


def _difference(layer, subject, clip, *, reflect=(1, 1)):
    sx, sy = reflect
    if reflect != (1, 1):
        subject = tuple(tuple((sx * x, sy * y) for x, y in points) for points in subject)
        clip = tuple(tuple((sx * x, sy * y) for x, y in points) for points in clip)
    origin = _layer_origin(subject, clip)
    ox, oy = origin
    if any(x - ox > (1 << 52) or y - oy > (1 << 52) for points in subject + clip for x, y in points):
        raise _SparseCoverageDeclined("retained coordinates exceed the exact integer range for gdstk")
    polygons = _checked_gdstk_boolean(
        _to_gdstk_polygons(subject, layer, origin), _to_gdstk_polygons(clip, layer, origin), "not", precision=1.0,
    )
    fragments = []
    for polygon in polygons:
        xy = polygon.points
        if not np.isfinite(xy).all() or not np.array_equal(xy, np.rint(xy)):
            raise _BooleanEvidenceError(f"gdstk {layer.layer}:{layer.datatype} returned off-grid vertices")
        points = tuple((sx * (int(x) + ox), sy * (int(y) + oy)) for x, y in xy)
        area = polygon_twice_area(points)
        if not area:
            raise _BooleanEvidenceError(f"gdstk {layer.layer}:{layer.datatype} returned a degenerate loop")
        fragments.append(DiffFragment(points, polygon_bbox(points), area))
    return tuple(sorted(fragments, key=_fragment_sort_key))
