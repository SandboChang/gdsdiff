from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from time import perf_counter

from .canonicalize import intern_polygons
from .config import LayerSpec
from .exact_oracle import DiffFragment, ExactOracleResult, LayerExactDiff
from .geometry import PolygonBuffer, polygon_bbox, polygon_twice_area
from .polygon_components import build_record_delta_surplus_components


@dataclass(frozen=True)
class SurplusCoverageStats:
    old_surplus_count: int
    new_surplus_count: int
    changed_layer_count: int
    conservative_overlap_layer_count: int
    component_count: int
    overlap_component_count: int
    max_fragments_per_component: int
    max_vertices_per_component: int
    coverage_exact_proven: bool
    canonicalize_s: float
    record_emit_s: float

    def to_json(self) -> dict[str, object]:
        return {
            "old_surplus_count": self.old_surplus_count,
            "new_surplus_count": self.new_surplus_count,
            "changed_layer_count": self.changed_layer_count,
            "conservative_overlap_layer_count": self.conservative_overlap_layer_count,
            "component_count": self.component_count,
            "overlap_component_count": self.overlap_component_count,
            "max_fragments_per_component": self.max_fragments_per_component,
            "max_vertices_per_component": self.max_vertices_per_component,
            "coverage_exact_proven": self.coverage_exact_proven,
            "canonicalize_s": self.canonicalize_s,
            "record_emit_s": self.record_emit_s,
        }


@dataclass(frozen=True)
class SurplusCoverageResult:
    exact_result: ExactOracleResult
    stats: SurplusCoverageStats


def run_canonical_surplus_coverage_probe(old_buffer: PolygonBuffer, new_buffer: PolygonBuffer) -> SurplusCoverageResult:
    """Emit directional component records from canonical source-polygon surplus.

    This is a coverage probe, not the final arbitrary-polygon Boolean engine. It is exact only
    when the surplus polygons are conservatively proven independent by positive bbox overlap.
    """

    canonicalize_start = perf_counter()
    old_intern = intern_polygons(old_buffer)
    new_intern = intern_polygons(new_buffer)
    canonicalize_s = perf_counter() - canonicalize_start

    old_by_key = _records_by_key(old_buffer, old_intern.records)
    new_by_key = _records_by_key(new_buffer, new_intern.records)

    emit_start = perf_counter()
    old_surplus: dict[LayerSpec, list[DiffFragment]] = defaultdict(list)
    new_surplus: dict[LayerSpec, list[DiffFragment]] = defaultdict(list)
    for key in sorted(set(old_by_key) | set(new_by_key), key=_key_sort):
        old_records = old_by_key.get(key, ())
        new_records = new_by_key.get(key, ())
        delta = len(old_records) - len(new_records)
        if delta > 0:
            layer = key[0]
            old_surplus[layer].extend(_fragment_from_buffer(old_buffer, record.polygon_index) for record in old_records[:delta])
        elif delta < 0:
            layer = key[0]
            new_surplus[layer].extend(_fragment_from_buffer(new_buffer, record.polygon_index) for record in new_records[: -delta])

    layers = tuple(sorted(set(old_surplus) | set(new_surplus)))
    layer_diffs = tuple(
        LayerExactDiff(
            layer=layer,
            old_minus_new=tuple(sorted(old_surplus.get(layer, ()), key=_fragment_sort_key)),
            new_minus_old=tuple(sorted(new_surplus.get(layer, ()), key=_fragment_sort_key)),
            xor=tuple(
                sorted(
                    tuple(old_surplus.get(layer, ())) + tuple(new_surplus.get(layer, ())),
                    key=_fragment_sort_key,
                )
            ),
        )
        for layer in layers
    )
    overlap_layers = tuple(
        layer
        for layer in layers
        if _has_positive_bbox_overlap(tuple(old_surplus.get(layer, ())) + tuple(new_surplus.get(layer, ())))
    )
    component_stats = _component_stats(old_surplus, new_surplus)
    record_emit_s = perf_counter() - emit_start
    return SurplusCoverageResult(
        exact_result=ExactOracleResult(layer_diffs),
        stats=SurplusCoverageStats(
            old_surplus_count=sum(len(fragments) for fragments in old_surplus.values()),
            new_surplus_count=sum(len(fragments) for fragments in new_surplus.values()),
            changed_layer_count=len(layers),
            conservative_overlap_layer_count=len(overlap_layers),
            component_count=component_stats["component_count"],
            overlap_component_count=component_stats["overlap_component_count"],
            max_fragments_per_component=component_stats["max_fragments_per_component"],
            max_vertices_per_component=component_stats["max_vertices_per_component"],
            coverage_exact_proven=len(overlap_layers) == 0,
            canonicalize_s=canonicalize_s,
            record_emit_s=record_emit_s,
        ),
    )


def run_record_delta_surplus_coverage_probe(old_buffer: PolygonBuffer, new_buffer: PolygonBuffer, record_delta) -> SurplusCoverageResult:
    """Emit source-surplus component records from GPU record-delta rows.

    The CUDA record-delta path uses canonical orientation-insensitive hashes and returns a
    representative source polygon index for each changed key. This removes the full CPU
    canonicalization pass, but changed coverage is not marked exact until the true polygon
    Boolean engine exists.
    """

    emit_start = perf_counter()
    component_result = build_record_delta_surplus_components(old_buffer, new_buffer, record_delta)
    old_surplus: dict[LayerSpec, list[DiffFragment]] = defaultdict(list)
    new_surplus: dict[LayerSpec, list[DiffFragment]] = defaultdict(list)
    for component in component_result.components:
        for polygon in component.polygons:
            if polygon.side == "old":
                old_surplus[component.layer].append(polygon.fragment)
            else:
                new_surplus[component.layer].append(polygon.fragment)

    layers = tuple(sorted(set(old_surplus) | set(new_surplus)))
    layer_diffs = tuple(
        LayerExactDiff(
            layer=layer,
            old_minus_new=tuple(sorted(old_surplus.get(layer, ()), key=_fragment_sort_key)),
            new_minus_old=tuple(sorted(new_surplus.get(layer, ()), key=_fragment_sort_key)),
            xor=tuple(
                sorted(
                    tuple(old_surplus.get(layer, ())) + tuple(new_surplus.get(layer, ())),
                    key=_fragment_sort_key,
                )
            ),
        )
        for layer in layers
    )
    overlap_layers = tuple(
        layer
        for layer in layers
        if _has_positive_bbox_overlap(tuple(old_surplus.get(layer, ())) + tuple(new_surplus.get(layer, ())))
    )
    record_emit_s = perf_counter() - emit_start
    changed_count = sum(len(fragments) for fragments in old_surplus.values()) + sum(
        len(fragments) for fragments in new_surplus.values()
    )
    return SurplusCoverageResult(
        exact_result=ExactOracleResult(layer_diffs),
        stats=SurplusCoverageStats(
            old_surplus_count=sum(len(fragments) for fragments in old_surplus.values()),
            new_surplus_count=sum(len(fragments) for fragments in new_surplus.values()),
            changed_layer_count=len(layers),
            conservative_overlap_layer_count=len(overlap_layers),
            component_count=component_result.stats.component_count,
            overlap_component_count=component_result.stats.overlap_component_count,
            max_fragments_per_component=component_result.stats.max_fragments_per_component,
            max_vertices_per_component=component_result.stats.max_vertices_per_component,
            coverage_exact_proven=changed_count == 0,
            canonicalize_s=0.0,
            record_emit_s=record_emit_s,
        ),
    )
def _records_by_key(buffer: PolygonBuffer, records) -> dict[tuple[LayerSpec, bytes], tuple[object, ...]]:
    by_key: dict[tuple[LayerSpec, bytes], list[object]] = defaultdict(list)
    for record in records:
        layer = buffer.layer_table.layers[int(buffer.layer_ids[record.polygon_index])]
        by_key[(layer, record.blob)].append(record)
    return {key: tuple(records) for key, records in by_key.items()}


def _fragment_from_buffer(buffer: PolygonBuffer, index: int) -> DiffFragment:
    points = tuple((int(x), int(y)) for x, y in buffer.polygon_vertices(index))
    return DiffFragment(points=points, bbox=polygon_bbox(points), twice_area=polygon_twice_area(points))


def _has_positive_bbox_overlap(fragments: tuple[DiffFragment, ...]) -> bool:
    for left_index, left in enumerate(fragments):
        for right in fragments[left_index + 1 :]:
            if _positive_overlap(left.bbox, right.bbox):
                return True
    return False


def _component_stats(
    old_surplus: dict[LayerSpec, list[DiffFragment]],
    new_surplus: dict[LayerSpec, list[DiffFragment]],
) -> dict[str, int]:
    component_count = 0
    overlap_component_count = 0
    max_fragments = 0
    max_vertices = 0
    for layer in sorted(set(old_surplus) | set(new_surplus)):
        fragments = tuple(old_surplus.get(layer, ())) + tuple(new_surplus.get(layer, ()))
        if not fragments:
            continue
        components = _fragment_components(fragments)
        component_count += len(components)
        for component in components:
            max_fragments = max(max_fragments, len(component))
            max_vertices = max(max_vertices, sum(len(fragments[index].points) for index in component))
            if _has_positive_bbox_overlap(tuple(fragments[index] for index in component)):
                overlap_component_count += 1
    return {
        "component_count": component_count,
        "overlap_component_count": overlap_component_count,
        "max_fragments_per_component": max_fragments,
        "max_vertices_per_component": max_vertices,
    }


def _fragment_components(fragments: tuple[DiffFragment, ...]) -> tuple[tuple[int, ...], ...]:
    remaining = set(range(len(fragments)))
    components: list[tuple[int, ...]] = []
    while remaining:
        seed = remaining.pop()
        component = {seed}
        stack = [seed]
        while stack:
            left = stack.pop()
            touched = [right for right in remaining if _positive_overlap(fragments[left].bbox, fragments[right].bbox)]
            for right in touched:
                remaining.remove(right)
                component.add(right)
                stack.append(right)
        components.append(tuple(sorted(component)))
    return tuple(sorted(components))


def _positive_overlap(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> bool:
    return max(left[0], right[0]) < min(left[2], right[2]) and max(left[1], right[1]) < min(left[3], right[3])


def _key_sort(key: tuple[LayerSpec, bytes]) -> tuple[int, int, bytes]:
    layer, blob = key
    return layer.layer, layer.datatype, blob


def _fragment_sort_key(fragment: DiffFragment) -> tuple[tuple[int, int, int, int], int, tuple[tuple[int, int], ...]]:
    return fragment.bbox, fragment.twice_area, fragment.points
