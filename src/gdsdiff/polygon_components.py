from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .boundary_loops import assemble_boundary_loops
from .config import LayerSpec
from .exact_oracle import DiffFragment, ExactOracleResult, LayerExactDiff
from .geometry import PolygonBuffer, polygon_bbox, polygon_twice_area


@dataclass(frozen=True)
class ComponentPolygon:
    side: str
    source_index: int
    fragment: DiffFragment
    role: str = "surplus"


@dataclass(frozen=True)
class PolygonComponent:
    layer: LayerSpec
    polygons: tuple[ComponentPolygon, ...]

    @property
    def old_count(self) -> int:
        return sum(1 for polygon in self.polygons if polygon.side == "old")

    @property
    def new_count(self) -> int:
        return sum(1 for polygon in self.polygons if polygon.side == "new")

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return _bbox_union(tuple(polygon.fragment.bbox for polygon in self.polygons))

    @property
    def vertex_count(self) -> int:
        return sum(len(polygon.fragment.points) for polygon in self.polygons)

    @property
    def has_positive_overlap(self) -> bool:
        fragments = tuple(polygon.fragment for polygon in self.polygons)
        return _has_positive_bbox_overlap(fragments)


@dataclass(frozen=True)
class PolygonComponentStats:
    component_count: int
    overlap_component_count: int
    max_fragments_per_component: int
    max_vertices_per_component: int
    context_index_backend: str = "none"
    context_index_s: float = 0.0
    context_pair_count: int = 0
    context_cuda_kernel_count: int = 0
    context_cuda_event_elapsed_s: float = 0.0
    context_cuda_transferred_bytes: int = 0

    def to_json(self) -> dict[str, int | float | str]:
        return {
            "component_count": self.component_count,
            "overlap_component_count": self.overlap_component_count,
            "max_fragments_per_component": self.max_fragments_per_component,
            "max_vertices_per_component": self.max_vertices_per_component,
            "context_index_backend": self.context_index_backend,
            "context_index_s": self.context_index_s,
            "context_pair_count": self.context_pair_count,
            "context_cuda_kernel_count": self.context_cuda_kernel_count,
            "context_cuda_event_elapsed_s": self.context_cuda_event_elapsed_s,
            "context_cuda_transferred_bytes": self.context_cuda_transferred_bytes,
        }


@dataclass(frozen=True)
class PolygonComponentBuildResult:
    components: tuple[PolygonComponent, ...]
    stats: PolygonComponentStats


@dataclass(frozen=True)
class LayerSideMergeStats:
    input_fragment_count: int = 0
    output_fragment_count: int = 0
    group_count: int = 0
    assembled_group_count: int = 0
    native_edge_cancel_s: float = 0.0
    cuda_loop_trace_s: float = 0.0
    native_loop_trace_s: float = 0.0
    python_loop_trace_s: float = 0.0
    hole_stitch_s: float = 0.0

    def to_json(self) -> dict[str, int | float]:
        return {
            "input_fragment_count": self.input_fragment_count,
            "output_fragment_count": self.output_fragment_count,
            "group_count": self.group_count,
            "assembled_group_count": self.assembled_group_count,
            "native_edge_cancel_s": self.native_edge_cancel_s,
            "cuda_loop_trace_s": self.cuda_loop_trace_s,
            "native_loop_trace_s": self.native_loop_trace_s,
            "python_loop_trace_s": self.python_loop_trace_s,
            "hole_stitch_s": self.hole_stitch_s,
        }


@dataclass(frozen=True)
class LayerSideMergeResult:
    fragments: tuple[DiffFragment, ...]
    stats: LayerSideMergeStats


@dataclass(frozen=True)
class _LayerBBoxIndex:
    indices: np.ndarray
    sorted_indices: np.ndarray
    sorted_x0: np.ndarray
    sorted_bboxes: np.ndarray


@dataclass(frozen=True)
class _ContextOverlapPairRun:
    component_indices: np.ndarray
    polygon_indices: np.ndarray
    wall_s: float = 0.0
    cuda_kernel_count: int = 0
    cuda_event_elapsed_s: float = 0.0
    transferred_bytes: int = 0

    @property
    def pair_count(self) -> int:
        return int(len(self.component_indices))


def build_record_delta_surplus_components(
    old_buffer: PolygonBuffer,
    new_buffer: PolygonBuffer,
    record_delta,
) -> PolygonComponentBuildResult:
    by_layer: dict[LayerSpec, list[ComponentPolygon]] = {}
    for row in range(record_delta.record_count):
        direction = int(record_delta.directions[row])
        count = int(record_delta.counts[row])
        representative_index = int(record_delta.representative_indices[row])
        if direction == 1:
            layer = old_buffer.layer_table.layers[int(old_buffer.layer_ids[representative_index])]
            polygon = ComponentPolygon(
                side="old",
                source_index=representative_index,
                fragment=_fragment_from_buffer(old_buffer, representative_index),
            )
        else:
            layer = new_buffer.layer_table.layers[int(new_buffer.layer_ids[representative_index])]
            polygon = ComponentPolygon(
                side="new",
                source_index=representative_index,
                fragment=_fragment_from_buffer(new_buffer, representative_index),
            )
        by_layer.setdefault(layer, []).extend(polygon for _ in range(count))

    components: list[PolygonComponent] = []
    for layer in sorted(by_layer):
        layer_polygons = tuple(by_layer[layer])
        for component_indices in _polygon_components(layer_polygons):
            components.append(
                PolygonComponent(
                    layer=layer,
                    polygons=tuple(layer_polygons[index] for index in component_indices),
                )
            )
    components_tuple = tuple(sorted(components, key=_component_sort_key))
    return PolygonComponentBuildResult(
        components=components_tuple,
        stats=_component_stats(components_tuple),
    )


def add_overlapping_context_to_components(
    old_buffer: PolygonBuffer,
    new_buffer: PolygonBuffer,
    components: tuple[PolygonComponent, ...],
) -> PolygonComponentBuildResult:
    context_start = perf_counter()
    old_fragment_cache: dict[int, DiffFragment] = {}
    new_fragment_cache: dict[int, DiffFragment] = {}
    old_source_indices = [
        {polygon.source_index for polygon in component.polygons if polygon.side == "old"} for component in components
    ]
    new_source_indices = [
        {polygon.source_index for polygon in component.polygons if polygon.side == "new"} for component in components
    ]
    old_pair_run = _try_cuda_context_pairs(components, old_buffer)
    new_pair_run = _try_cuda_context_pairs(components, new_buffer)
    if old_pair_run is not None and new_pair_run is not None:
        expanded_polygons = [list(component.polygons) for component in components]
        _append_context_pairs(
            expanded_polygons,
            old_pair_run,
            old_source_indices,
            side="old",
            buffer=old_buffer,
            fragment_cache=old_fragment_cache,
        )
        _append_context_pairs(
            expanded_polygons,
            new_pair_run,
            new_source_indices,
            side="new",
            buffer=new_buffer,
            fragment_cache=new_fragment_cache,
        )
        expanded_tuple = tuple(
            sorted(
                (
                    PolygonComponent(layer=component.layer, polygons=tuple(sorted(polygons, key=_polygon_sort_key)))
                    for component, polygons in zip(components, expanded_polygons)
                ),
                key=_component_sort_key,
            )
        )
        return PolygonComponentBuildResult(
            components=expanded_tuple,
            stats=_component_stats(
                expanded_tuple,
                context_index_backend="cuda-context-overlap-pairs",
                context_index_s=old_pair_run.wall_s + new_pair_run.wall_s,
                context_pair_count=old_pair_run.pair_count + new_pair_run.pair_count,
                context_cuda_kernel_count=old_pair_run.cuda_kernel_count + new_pair_run.cuda_kernel_count,
                context_cuda_event_elapsed_s=old_pair_run.cuda_event_elapsed_s + new_pair_run.cuda_event_elapsed_s,
                context_cuda_transferred_bytes=old_pair_run.transferred_bytes + new_pair_run.transferred_bytes,
            ),
        )
    old_layer_indices = _indices_by_layer_id(old_buffer)
    new_layer_indices = _indices_by_layer_id(new_buffer)
    old_layer_ids = old_buffer.layer_table.index_by_layer
    new_layer_ids = new_buffer.layer_table.index_by_layer
    expanded: list[PolygonComponent] = []
    for component_index, component in enumerate(components):
        old_indices = old_source_indices[component_index]
        new_indices = new_source_indices[component_index]
        polygons = list(component.polygons)
        component_bbox = component.bbox
        old_layer_id = old_layer_ids.get(component.layer)
        new_layer_id = new_layer_ids.get(component.layer)
        for index in _overlapping_layer_indices(old_buffer, old_layer_indices, old_layer_id, component_bbox):
            if index not in old_indices:
                polygons.append(
                    ComponentPolygon(
                        side="old",
                        source_index=index,
                        fragment=_cached_fragment(old_buffer, old_fragment_cache, index),
                        role="context",
                    )
                )
        for index in _overlapping_layer_indices(new_buffer, new_layer_indices, new_layer_id, component_bbox):
            if index not in new_indices:
                polygons.append(
                    ComponentPolygon(
                        side="new",
                        source_index=index,
                        fragment=_cached_fragment(new_buffer, new_fragment_cache, index),
                        role="context",
                    )
                )
        expanded.append(PolygonComponent(layer=component.layer, polygons=tuple(sorted(polygons, key=_polygon_sort_key))))
    expanded_tuple = tuple(sorted(expanded, key=_component_sort_key))
    return PolygonComponentBuildResult(
        components=expanded_tuple,
        stats=_component_stats(
            expanded_tuple,
            context_index_backend="cpu-layer-bbox-index",
            context_index_s=perf_counter() - context_start,
        ),
    )


def _merge_touching_components_by_layer(components: tuple[PolygonComponent, ...]) -> tuple[PolygonComponent, ...]:
    by_layer: dict[LayerSpec, list[PolygonComponent]] = {}
    for component in components:
        by_layer.setdefault(component.layer, []).append(component)
    merged: list[PolygonComponent] = []
    for layer in sorted(by_layer):
        layer_components = tuple(sorted(by_layer[layer], key=_component_sort_key))
        for group in _component_bbox_contact_groups(layer_components):
            polygons = {}
            for index in group:
                for polygon in layer_components[index].polygons:
                    polygons[(polygon.side, polygon.source_index, polygon.role, polygon.fragment.bbox)] = polygon
            merged.append(
                PolygonComponent(
                    layer=layer,
                    polygons=tuple(sorted(polygons.values(), key=_polygon_sort_key)),
                )
            )
    return tuple(sorted(merged, key=_component_sort_key))


def _component_bbox_contact_groups(components: tuple[PolygonComponent, ...]) -> tuple[tuple[int, ...], ...]:
    remaining = set(range(len(components)))
    groups: list[tuple[int, ...]] = []
    while remaining:
        seed = remaining.pop()
        group = {seed}
        stack = [seed]
        while stack:
            left = stack.pop()
            touched = [
                right
                for right in remaining
                if _bbox_contact(components[left].bbox, components[right].bbox)
            ]
            for right in touched:
                remaining.remove(right)
                group.add(right)
                stack.append(right)
        groups.append(tuple(sorted(group)))
    return tuple(sorted(groups))


def exact_result_from_component_quads(
    components: tuple[PolygonComponent, ...],
    quads: np.ndarray,
    fragment_areas: np.ndarray,
    counts: np.ndarray,
    component_areas: np.ndarray | None = None,
) -> ExactOracleResult:
    by_layer: dict[LayerSpec, dict[str, list[DiffFragment]]] = {}
    for component_index, component in enumerate(components):
        layer_bucket = by_layer.setdefault(component.layer, {"old": [], "new": []})
        one_sided = _one_sided_nonoverlap_fragments(component)
        if one_sided is not None:
            side, fragments = one_sided
            layer_bucket[side].extend(fragments)
            continue
        for side_index, side in enumerate(("old", "new")):
            fragments: list[DiffFragment] = []
            for fragment_index in range(int(counts[component_index, side_index])):
                points = tuple((int(x), int(y)) for x, y in quads[component_index, side_index, fragment_index])
                if len(set(points)) < 3:
                    continue
                twice_area = int(fragment_areas[component_index, side_index, fragment_index])
                if twice_area == 0:
                    continue
                fragments.append(DiffFragment(points=points, bbox=polygon_bbox(points), twice_area=twice_area))
            if fragments:
                assembled = assemble_boundary_loops(tuple(fragments)).fragments
                target_area = (
                    int(component_areas[component_index, side_index])
                    if component_areas is not None
                    else sum(fragment.twice_area for fragment in fragments)
                )
                layer_bucket[side].extend(_preserve_total_area(assembled, target_area))
    layer_diffs = []
    for layer in sorted(by_layer):
        old_minus_new = _merge_layer_side_fragments(tuple(sorted(by_layer[layer]["old"], key=_fragment_sort_key)))
        new_minus_old = _merge_layer_side_fragments(tuple(sorted(by_layer[layer]["new"], key=_fragment_sort_key)))
        layer_diffs.append(
            LayerExactDiff(
                layer=layer,
                old_minus_new=old_minus_new,
                new_minus_old=new_minus_old,
                xor=tuple(sorted(old_minus_new + new_minus_old, key=_fragment_sort_key)),
            )
        )
    return ExactOracleResult(tuple(layer_diffs))


def _fragment_from_buffer(buffer: PolygonBuffer, index: int) -> DiffFragment:
    points = tuple((int(x), int(y)) for x, y in buffer.polygon_vertices(index))
    return DiffFragment(points=points, bbox=polygon_bbox(points), twice_area=polygon_twice_area(points))


def _cached_fragment(buffer: PolygonBuffer, cache: dict[int, DiffFragment], index: int) -> DiffFragment:
    fragment = cache.get(index)
    if fragment is None:
        fragment = _fragment_from_buffer(buffer, index)
        cache[index] = fragment
    return fragment


def _one_sided_nonoverlap_fragments(component: PolygonComponent) -> tuple[str, tuple[DiffFragment, ...]] | None:
    sides = {polygon.side for polygon in component.polygons}
    if len(sides) != 1:
        return None
    fragments = tuple(polygon.fragment for polygon in component.polygons)
    if _has_bbox_contact(fragments):
        return None
    side = next(iter(sides))
    return side, tuple(sorted(fragments, key=_fragment_sort_key))


def _preserve_total_area(fragments: tuple[DiffFragment, ...], target_twice_area: int) -> tuple[DiffFragment, ...]:
    if not fragments:
        return ()
    current = sum(fragment.twice_area for fragment in fragments)
    delta = target_twice_area - current
    if delta == 0:
        return fragments
    largest_index = max(range(len(fragments)), key=lambda index: (fragments[index].twice_area, fragments[index].bbox))
    adjusted = list(fragments)
    largest = adjusted[largest_index]
    if largest.twice_area + delta <= 0:
        return fragments
    adjusted[largest_index] = DiffFragment(points=largest.points, bbox=largest.bbox, twice_area=largest.twice_area + delta)
    return tuple(sorted(adjusted, key=_fragment_sort_key))


def _indices_by_layer_id(buffer: PolygonBuffer) -> dict[int, _LayerBBoxIndex]:
    if buffer.polygon_count == 0:
        return {}
    order = np.argsort(buffer.layer_ids, kind="stable")
    sorted_layer_ids = buffer.layer_ids[order]
    split_points = np.flatnonzero(sorted_layer_ids[1:] != sorted_layer_ids[:-1]) + 1
    buckets = np.split(order.astype(np.int64, copy=False), split_points)
    layer_ids = np.split(sorted_layer_ids, split_points)
    return {int(ids[0]): _build_layer_bbox_index(buffer, bucket) for ids, bucket in zip(layer_ids, buckets) if len(ids)}


def _build_layer_bbox_index(buffer: PolygonBuffer, indices: np.ndarray) -> _LayerBBoxIndex:
    layer_bboxes = buffer.bboxes[indices]
    order = np.argsort(layer_bboxes[:, 0], kind="stable")
    sorted_indices = indices[order].astype(np.int64, copy=False)
    sorted_bboxes = layer_bboxes[order].astype(np.int64, copy=False)
    sorted_x0 = sorted_bboxes[:, 0]
    return _LayerBBoxIndex(indices=indices, sorted_indices=sorted_indices, sorted_x0=sorted_x0, sorted_bboxes=sorted_bboxes)


def _overlapping_layer_indices(
    buffer: PolygonBuffer,
    indices_by_layer: dict[int, _LayerBBoxIndex],
    layer_id: int | None,
    bbox: tuple[int, int, int, int],
) -> tuple[int, ...]:
    if layer_id is None:
        return ()
    layer_index = indices_by_layer.get(layer_id)
    if layer_index is None or len(layer_index.indices) == 0:
        return ()
    stop = int(np.searchsorted(layer_index.sorted_x0, bbox[2], side="left"))
    if stop == 0:
        return ()
    indices = layer_index.sorted_indices[:stop]
    bboxes = layer_index.sorted_bboxes[:stop]
    mask = (
        (bboxes[:, 2] > bbox[0])
        & (np.maximum(bboxes[:, 1], bbox[1]) < np.minimum(bboxes[:, 3], bbox[3]))
    )
    return tuple(sorted(int(index) for index in indices[mask]))


def _try_cuda_context_pairs(
    components: tuple[PolygonComponent, ...],
    buffer: PolygonBuffer,
) -> _ContextOverlapPairRun | None:
    if not components or buffer.polygon_count == 0:
        return _ContextOverlapPairRun(
            component_indices=np.empty((0,), dtype=np.uint64),
            polygon_indices=np.empty((0,), dtype=np.uint64),
        )
    try:
        from .backends.cuda_ctypes import CudaCtypesUnavailable, run_cuda_context_overlap_pairs

        component_bboxes, component_layers, component_datatypes = _component_context_arrays(components)
        polygon_layers, polygon_datatypes = _buffer_layer_datatype_arrays(buffer)
        start = perf_counter()
        pairs = run_cuda_context_overlap_pairs(
            component_bboxes,
            component_layers,
            component_datatypes,
            buffer.bboxes,
            polygon_layers,
            polygon_datatypes,
        )
        wall_s = perf_counter() - start
        return _ContextOverlapPairRun(
            component_indices=pairs.component_indices,
            polygon_indices=pairs.polygon_indices,
            wall_s=wall_s,
            cuda_kernel_count=pairs.cuda_kernel_count,
            cuda_event_elapsed_s=pairs.cuda_event_elapsed_s,
            transferred_bytes=pairs.transferred_bytes,
        )
    except (CudaCtypesUnavailable, ImportError, OSError, ValueError, OverflowError):
        return None


def _component_context_arrays(
    components: tuple[PolygonComponent, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bboxes = np.asarray([component.bbox for component in components], dtype=np.int64).reshape((-1, 4))
    layers = np.asarray([component.layer.layer for component in components], dtype=np.int32)
    datatypes = np.asarray([component.layer.datatype for component in components], dtype=np.int32)
    return (
        np.ascontiguousarray(bboxes, dtype=np.int64),
        np.ascontiguousarray(layers, dtype=np.int32),
        np.ascontiguousarray(datatypes, dtype=np.int32),
    )


def _buffer_layer_datatype_arrays(buffer: PolygonBuffer) -> tuple[np.ndarray, np.ndarray]:
    layer_values = np.asarray([layer.layer for layer in buffer.layer_table.layers], dtype=np.int32)
    datatype_values = np.asarray([layer.datatype for layer in buffer.layer_table.layers], dtype=np.int32)
    return (
        np.ascontiguousarray(layer_values[buffer.layer_ids], dtype=np.int32),
        np.ascontiguousarray(datatype_values[buffer.layer_ids], dtype=np.int32),
    )


def _append_context_pairs(
    expanded_polygons: list[list[ComponentPolygon]],
    pair_run: _ContextOverlapPairRun,
    source_indices: list[set[int]],
    *,
    side: str,
    buffer: PolygonBuffer,
    fragment_cache: dict[int, DiffFragment],
) -> None:
    for component_index_raw, polygon_index_raw in zip(pair_run.component_indices, pair_run.polygon_indices):
        component_index = int(component_index_raw)
        polygon_index = int(polygon_index_raw)
        if polygon_index in source_indices[component_index]:
            continue
        expanded_polygons[component_index].append(
            ComponentPolygon(
                side=side,
                source_index=polygon_index,
                fragment=_cached_fragment(buffer, fragment_cache, polygon_index),
                role="context",
            )
        )


def _polygon_components(polygons: tuple[ComponentPolygon, ...]) -> tuple[tuple[int, ...], ...]:
    remaining = set(range(len(polygons)))
    components: list[tuple[int, ...]] = []
    while remaining:
        seed = remaining.pop()
        component = {seed}
        stack = [seed]
        while stack:
            left = stack.pop()
            touched = [
                right
                for right in remaining
                if _positive_overlap(polygons[left].fragment.bbox, polygons[right].fragment.bbox)
            ]
            for right in touched:
                remaining.remove(right)
                component.add(right)
                stack.append(right)
        components.append(tuple(sorted(component)))
    return tuple(sorted(components))


def _component_stats(
    components: tuple[PolygonComponent, ...],
    *,
    context_index_backend: str = "none",
    context_index_s: float = 0.0,
    context_pair_count: int = 0,
    context_cuda_kernel_count: int = 0,
    context_cuda_event_elapsed_s: float = 0.0,
    context_cuda_transferred_bytes: int = 0,
) -> PolygonComponentStats:
    return PolygonComponentStats(
        component_count=len(components),
        overlap_component_count=sum(1 for component in components if component.has_positive_overlap),
        max_fragments_per_component=max((len(component.polygons) for component in components), default=0),
        max_vertices_per_component=max((component.vertex_count for component in components), default=0),
        context_index_backend=context_index_backend,
        context_index_s=context_index_s,
        context_pair_count=context_pair_count,
        context_cuda_kernel_count=context_cuda_kernel_count,
        context_cuda_event_elapsed_s=context_cuda_event_elapsed_s,
        context_cuda_transferred_bytes=context_cuda_transferred_bytes,
    )


def _merge_layer_side_fragments(fragments: tuple[DiffFragment, ...]) -> tuple[DiffFragment, ...]:
    return _merge_layer_side_fragments_with_stats(fragments).fragments


def _merge_layer_side_fragments_with_stats(fragments: tuple[DiffFragment, ...]) -> LayerSideMergeResult:
    if len(fragments) < 2:
        return LayerSideMergeResult(
            fragments=fragments,
            stats=LayerSideMergeStats(input_fragment_count=len(fragments), output_fragment_count=len(fragments)),
        )
    if len(fragments) > 512 or sum(len(fragment.points) for fragment in fragments) > 50_000:
        return LayerSideMergeResult(
            fragments=fragments,
            stats=LayerSideMergeStats(input_fragment_count=len(fragments), output_fragment_count=len(fragments)),
        )
    merged: list[DiffFragment] = []
    group_count = 0
    assembled_group_count = 0
    native_edge_cancel_s = 0.0
    cuda_loop_trace_s = 0.0
    native_loop_trace_s = 0.0
    python_loop_trace_s = 0.0
    hole_stitch_s = 0.0
    for group in _fragment_bbox_components(fragments):
        group_count += 1
        if len(group) == 1:
            merged.append(_canonicalize_single_fragment(group[0]))
            continue
        target_area = sum(fragment.twice_area for fragment in group)
        assembly = assemble_boundary_loops(group)
        assembled = assembly.fragments
        assembled_group_count += 1
        native_edge_cancel_s += assembly.stats.native_edge_cancel_s
        cuda_loop_trace_s += assembly.stats.cuda_loop_trace_s
        native_loop_trace_s += assembly.stats.native_loop_trace_s
        python_loop_trace_s += assembly.stats.python_loop_trace_s
        hole_stitch_s += assembly.stats.hole_stitch_s
        if not assembled:
            merged.extend(group)
        else:
            merged.extend(_preserve_total_area(assembled, target_area))
    output = tuple(sorted(merged, key=_fragment_sort_key))
    return LayerSideMergeResult(
        fragments=output,
        stats=LayerSideMergeStats(
            input_fragment_count=len(fragments),
            output_fragment_count=len(output),
            group_count=group_count,
            assembled_group_count=assembled_group_count,
            native_edge_cancel_s=native_edge_cancel_s,
            cuda_loop_trace_s=cuda_loop_trace_s,
            native_loop_trace_s=native_loop_trace_s,
            python_loop_trace_s=python_loop_trace_s,
            hole_stitch_s=hole_stitch_s,
        ),
    )


def _canonicalize_single_fragment(fragment: DiffFragment) -> DiffFragment:
    points = _clean_loop_points(fragment.points)
    if len(points) < 3:
        return fragment
    twice_area = polygon_twice_area(points)
    if twice_area != fragment.twice_area:
        return fragment
    return DiffFragment(points=points, bbox=polygon_bbox(points), twice_area=twice_area)


def _clean_loop_points(points: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    cleaned: list[tuple[int, int]] = []
    for point in points:
        if not cleaned or cleaned[-1] != point:
            cleaned.append(point)
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    changed = True
    while changed and len(cleaned) >= 3:
        changed = False
        next_points: list[tuple[int, int]] = []
        for index, point in enumerate(cleaned):
            previous = cleaned[index - 1]
            following = cleaned[(index + 1) % len(cleaned)]
            if _collinear(previous, point, following):
                changed = True
                continue
            next_points.append(point)
        cleaned = next_points
    return tuple(cleaned)


def _collinear(
    left: tuple[int, int],
    middle: tuple[int, int],
    right: tuple[int, int],
) -> bool:
    return (middle[0] - left[0]) * (right[1] - middle[1]) == (middle[1] - left[1]) * (right[0] - middle[0])


def _fragment_bbox_components(fragments: tuple[DiffFragment, ...]) -> tuple[tuple[DiffFragment, ...], ...]:
    remaining = set(range(len(fragments)))
    components: list[tuple[DiffFragment, ...]] = []
    while remaining:
        seed = remaining.pop()
        component = {seed}
        stack = [seed]
        while stack:
            left = stack.pop()
            touched = [
                right
                for right in remaining
                if _bbox_contact(fragments[left].bbox, fragments[right].bbox)
            ]
            for right in touched:
                remaining.remove(right)
                component.add(right)
                stack.append(right)
        components.append(tuple(fragments[index] for index in sorted(component)))
    return tuple(sorted(components, key=lambda group: (_bbox_union(tuple(fragment.bbox for fragment in group)), len(group))))


def _has_positive_bbox_overlap(fragments: tuple[DiffFragment, ...]) -> bool:
    for left_index, left in enumerate(fragments):
        for right in fragments[left_index + 1 :]:
            if _positive_overlap(left.bbox, right.bbox):
                return True
    return False


def _has_bbox_contact(fragments: tuple[DiffFragment, ...]) -> bool:
    for left_index, left in enumerate(fragments):
        for right in fragments[left_index + 1 :]:
            if _bbox_contact(left.bbox, right.bbox):
                return True
    return False


def _positive_overlap(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> bool:
    return max(left[0], right[0]) < min(left[2], right[2]) and max(left[1], right[1]) < min(left[3], right[3])


def _bbox_contact(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> bool:
    return max(left[0], right[0]) <= min(left[2], right[2]) and max(left[1], right[1]) <= min(left[3], right[3])


def _bbox_union(bboxes: tuple[tuple[int, int, int, int], ...]) -> tuple[int, int, int, int]:
    return (
        min(bbox[0] for bbox in bboxes),
        min(bbox[1] for bbox in bboxes),
        max(bbox[2] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes),
    )


def _component_sort_key(component: PolygonComponent) -> tuple[int, int, tuple[int, int, int, int], int, int]:
    return component.layer.layer, component.layer.datatype, component.bbox, component.old_count, component.new_count


def _polygon_sort_key(polygon: ComponentPolygon) -> tuple[str, str, int, tuple[int, int, int, int]]:
    return polygon.side, polygon.role, polygon.source_index, polygon.fragment.bbox


def _fragment_sort_key(fragment: DiffFragment) -> tuple[tuple[int, int, int, int], int, tuple[tuple[int, int], ...]]:
    return fragment.bbox, fragment.twice_area, fragment.points
