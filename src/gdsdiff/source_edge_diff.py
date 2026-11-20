from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import hypot, isfinite
from typing import Callable, Sequence

import numpy as np

from .exact_oracle import DiffFragment, ExactOracleResult, LayerExactDiff
from .geometry import polygon_bbox, polygon_twice_area
from .polygon_components import (
    LayerSideMergeStats,
    PolygonComponent,
    _component_sort_key,
    _fragment_sort_key,
    _merge_layer_side_fragments_with_stats,
)

Point = tuple[int, int]
FloatPoint = tuple[float, float]
FloatSegment = tuple[FloatPoint, FloatPoint]


@dataclass(frozen=True)
class SourceEdgeDiffStats:
    component_count: int
    boundary_edge_count: int
    loop_count: int
    dropped_zero_area_loop_count: int
    gpu_loop_component_count: int = 0
    gpu_loop_edge_count: int = 0
    cuda_loop_component_count: int = 0
    cuda_loop_edge_count: int = 0
    metal_loop_component_count: int = 0
    metal_loop_edge_count: int = 0
    native_loop_component_count: int = 0
    native_loop_edge_count: int = 0
    gpu_boundary_component_count: int = 0
    gpu_boundary_edge_count: int = 0
    gpu_split_free_component_count: int = 0
    gpu_mixed_boundary_component_count: int = 0
    gpu_mixed_boundary_edge_count: int = 0
    layer_merge_input_fragment_count: int = 0
    layer_merge_output_fragment_count: int = 0
    layer_merge_group_count: int = 0
    layer_merge_assembled_group_count: int = 0
    layer_merge_cuda_edge_cancel_s: float = 0.0
    layer_merge_cuda_edge_cancel_kernel_count: int = 0
    layer_merge_cuda_edge_cancel_event_elapsed_s: float = 0.0
    layer_merge_cuda_edge_cancel_transferred_bytes: int = 0
    layer_merge_native_edge_cancel_s: float = 0.0
    layer_merge_cuda_loop_trace_s: float = 0.0
    layer_merge_metal_loop_trace_s: float = 0.0
    layer_merge_metal_loop_kernel_count: int = 0
    layer_merge_metal_loop_event_elapsed_s: float = 0.0
    layer_merge_metal_loop_transferred_bytes: int = 0
    layer_merge_native_loop_trace_s: float = 0.0
    layer_merge_python_loop_trace_s: float = 0.0
    layer_merge_hole_stitch_s: float = 0.0
    nested_direction_fragment_component_count: int = 0

    def to_json(self) -> dict[str, int | float]:
        return {
            "component_count": self.component_count,
            "boundary_edge_count": self.boundary_edge_count,
            "loop_count": self.loop_count,
            "dropped_zero_area_loop_count": self.dropped_zero_area_loop_count,
            "gpu_loop_component_count": self.gpu_loop_component_count,
            "gpu_loop_edge_count": self.gpu_loop_edge_count,
            "cuda_loop_component_count": self.cuda_loop_component_count,
            "cuda_loop_edge_count": self.cuda_loop_edge_count,
            "metal_loop_component_count": self.metal_loop_component_count,
            "metal_loop_edge_count": self.metal_loop_edge_count,
            "native_loop_component_count": self.native_loop_component_count,
            "native_loop_edge_count": self.native_loop_edge_count,
            "gpu_boundary_component_count": self.gpu_boundary_component_count,
            "gpu_boundary_edge_count": self.gpu_boundary_edge_count,
            "gpu_split_free_component_count": self.gpu_split_free_component_count,
            "gpu_mixed_boundary_component_count": self.gpu_mixed_boundary_component_count,
            "gpu_mixed_boundary_edge_count": self.gpu_mixed_boundary_edge_count,
            "layer_merge_input_fragment_count": self.layer_merge_input_fragment_count,
            "layer_merge_output_fragment_count": self.layer_merge_output_fragment_count,
            "layer_merge_group_count": self.layer_merge_group_count,
            "layer_merge_assembled_group_count": self.layer_merge_assembled_group_count,
            "layer_merge_cuda_edge_cancel_s": self.layer_merge_cuda_edge_cancel_s,
            "layer_merge_cuda_edge_cancel_kernel_count": self.layer_merge_cuda_edge_cancel_kernel_count,
            "layer_merge_cuda_edge_cancel_event_elapsed_s": self.layer_merge_cuda_edge_cancel_event_elapsed_s,
            "layer_merge_cuda_edge_cancel_transferred_bytes": self.layer_merge_cuda_edge_cancel_transferred_bytes,
            "layer_merge_native_edge_cancel_s": self.layer_merge_native_edge_cancel_s,
            "layer_merge_cuda_loop_trace_s": self.layer_merge_cuda_loop_trace_s,
            "layer_merge_metal_loop_trace_s": self.layer_merge_metal_loop_trace_s,
            "layer_merge_metal_loop_kernel_count": self.layer_merge_metal_loop_kernel_count,
            "layer_merge_metal_loop_event_elapsed_s": self.layer_merge_metal_loop_event_elapsed_s,
            "layer_merge_metal_loop_transferred_bytes": self.layer_merge_metal_loop_transferred_bytes,
            "layer_merge_native_loop_trace_s": self.layer_merge_native_loop_trace_s,
            "layer_merge_python_loop_trace_s": self.layer_merge_python_loop_trace_s,
            "layer_merge_hole_stitch_s": self.layer_merge_hole_stitch_s,
            "nested_direction_fragment_component_count": self.nested_direction_fragment_component_count,
        }


@dataclass(frozen=True)
class SourceEdgeDiffResult:
    exact_result: ExactOracleResult
    stats: SourceEdgeDiffStats


@dataclass(frozen=True)
class _SegmentRecord:
    side: str
    polygon_index: int
    start: FloatPoint
    stop: FloatPoint


def exact_result_from_source_edges(
    components: tuple[PolygonComponent, ...],
    *,
    exact_on_nested: Callable[[], ExactOracleResult | None] | None = None,
) -> SourceEdgeDiffResult | ExactOracleResult:
    source_components = _merge_contacting_source_components(components)
    by_layer: dict[object, dict[str, list[DiffFragment]]] = {}
    edge_count = 0
    loop_count = 0
    dropped_count = 0
    gpu_component_count = 0
    gpu_edge_count = 0
    gpu_split_free_count = 0
    gpu_mixed_component_count = 0
    gpu_mixed_edge_count = 0
    native_loop_component_count = 0
    native_loop_edge_count = 0
    gpu_loop_component_count = 0
    gpu_loop_edge_count = 0
    cuda_loop_component_count = 0
    cuda_loop_edge_count = 0
    metal_loop_component_count = 0
    metal_loop_edge_count = 0
    nested_direction_fragment_component_count = 0
    layer_merge_stats = LayerSideMergeStats()
    side_cache: dict[tuple[str, tuple[object, ...]], tuple[tuple[DiffFragment, ...], _ComponentSideStats]] = {}
    for component in source_components:
        layer_bucket = by_layer.setdefault(component.layer, {"old": [], "new": []})
        old_minus, old_stats = _cached_component_side_diff(component, direction="old", cache=side_cache)
        new_minus, new_stats = _cached_component_side_diff(component, direction="new", cache=side_cache)
        # Only replace coverage when this is the same trigger that would
        # otherwise discard the assembled result and run hole correction.
        if exact_on_nested is not None and (
            old_stats.nested_direction_fragment_component_count or new_stats.nested_direction_fragment_component_count
        ):
            exact = exact_on_nested()
            if exact is not None:
                return exact
            exact_on_nested = None
        layer_bucket["old"].extend(old_minus)
        layer_bucket["new"].extend(new_minus)
        edge_count += old_stats.boundary_edge_count + new_stats.boundary_edge_count
        loop_count += old_stats.loop_count + new_stats.loop_count
        dropped_count += old_stats.dropped_zero_area_loop_count + new_stats.dropped_zero_area_loop_count
        gpu_component_count += old_stats.gpu_boundary_component_count + new_stats.gpu_boundary_component_count
        gpu_edge_count += old_stats.gpu_boundary_edge_count + new_stats.gpu_boundary_edge_count
        gpu_split_free_count += old_stats.gpu_split_free_component_count + new_stats.gpu_split_free_component_count
        gpu_mixed_component_count += old_stats.gpu_mixed_boundary_component_count + new_stats.gpu_mixed_boundary_component_count
        gpu_mixed_edge_count += old_stats.gpu_mixed_boundary_edge_count + new_stats.gpu_mixed_boundary_edge_count
        native_loop_component_count += old_stats.native_loop_component_count + new_stats.native_loop_component_count
        native_loop_edge_count += old_stats.native_loop_edge_count + new_stats.native_loop_edge_count
        gpu_loop_component_count += old_stats.gpu_loop_component_count + new_stats.gpu_loop_component_count
        gpu_loop_edge_count += old_stats.gpu_loop_edge_count + new_stats.gpu_loop_edge_count
        cuda_loop_component_count += old_stats.cuda_loop_component_count + new_stats.cuda_loop_component_count
        cuda_loop_edge_count += old_stats.cuda_loop_edge_count + new_stats.cuda_loop_edge_count
        metal_loop_component_count += old_stats.metal_loop_component_count + new_stats.metal_loop_component_count
        metal_loop_edge_count += old_stats.metal_loop_edge_count + new_stats.metal_loop_edge_count
        nested_direction_fragment_component_count += (
            old_stats.nested_direction_fragment_component_count
            + new_stats.nested_direction_fragment_component_count
        )

    layer_diffs = []
    for layer in sorted(by_layer):
        old_merge = _merge_layer_side_fragments_with_stats(tuple(sorted(by_layer[layer]["old"], key=_fragment_sort_key)))
        new_merge = _merge_layer_side_fragments_with_stats(tuple(sorted(by_layer[layer]["new"], key=_fragment_sort_key)))
        layer_merge_stats = _add_layer_merge_stats(layer_merge_stats, old_merge.stats)
        layer_merge_stats = _add_layer_merge_stats(layer_merge_stats, new_merge.stats)
        old_minus_new = old_merge.fragments
        new_minus_old = new_merge.fragments
        layer_diffs.append(
            LayerExactDiff(
                layer=layer,
                old_minus_new=old_minus_new,
                new_minus_old=new_minus_old,
                xor=tuple(sorted(old_minus_new + new_minus_old, key=_fragment_sort_key)),
            )
        )
    return SourceEdgeDiffResult(
        exact_result=ExactOracleResult(tuple(layer_diffs)),
        stats=SourceEdgeDiffStats(
            component_count=len(source_components),
            boundary_edge_count=edge_count,
            loop_count=loop_count,
            dropped_zero_area_loop_count=dropped_count,
            gpu_loop_component_count=gpu_loop_component_count,
            gpu_loop_edge_count=gpu_loop_edge_count,
            cuda_loop_component_count=cuda_loop_component_count,
            cuda_loop_edge_count=cuda_loop_edge_count,
            metal_loop_component_count=metal_loop_component_count,
            metal_loop_edge_count=metal_loop_edge_count,
            native_loop_component_count=native_loop_component_count,
            native_loop_edge_count=native_loop_edge_count,
            gpu_boundary_component_count=gpu_component_count,
            gpu_boundary_edge_count=gpu_edge_count,
            gpu_split_free_component_count=gpu_split_free_count,
            gpu_mixed_boundary_component_count=gpu_mixed_component_count,
            gpu_mixed_boundary_edge_count=gpu_mixed_edge_count,
            layer_merge_input_fragment_count=layer_merge_stats.input_fragment_count,
            layer_merge_output_fragment_count=layer_merge_stats.output_fragment_count,
            layer_merge_group_count=layer_merge_stats.group_count,
            layer_merge_assembled_group_count=layer_merge_stats.assembled_group_count,
            layer_merge_cuda_edge_cancel_s=layer_merge_stats.cuda_edge_cancel_s,
            layer_merge_cuda_edge_cancel_kernel_count=layer_merge_stats.cuda_edge_cancel_kernel_count,
            layer_merge_cuda_edge_cancel_event_elapsed_s=layer_merge_stats.cuda_edge_cancel_event_elapsed_s,
            layer_merge_cuda_edge_cancel_transferred_bytes=layer_merge_stats.cuda_edge_cancel_transferred_bytes,
            layer_merge_native_edge_cancel_s=layer_merge_stats.native_edge_cancel_s,
            layer_merge_cuda_loop_trace_s=layer_merge_stats.cuda_loop_trace_s,
            layer_merge_metal_loop_trace_s=layer_merge_stats.metal_loop_trace_s,
            layer_merge_metal_loop_kernel_count=layer_merge_stats.metal_loop_kernel_count,
            layer_merge_metal_loop_event_elapsed_s=layer_merge_stats.metal_loop_event_elapsed_s,
            layer_merge_metal_loop_transferred_bytes=layer_merge_stats.metal_loop_transferred_bytes,
            layer_merge_native_loop_trace_s=layer_merge_stats.native_loop_trace_s,
            layer_merge_python_loop_trace_s=layer_merge_stats.python_loop_trace_s,
            layer_merge_hole_stitch_s=layer_merge_stats.hole_stitch_s,
            nested_direction_fragment_component_count=nested_direction_fragment_component_count,
        ),
    )


def _add_layer_merge_stats(left: LayerSideMergeStats, right: LayerSideMergeStats) -> LayerSideMergeStats:
    return LayerSideMergeStats(
        input_fragment_count=left.input_fragment_count + right.input_fragment_count,
        output_fragment_count=left.output_fragment_count + right.output_fragment_count,
        group_count=left.group_count + right.group_count,
        assembled_group_count=left.assembled_group_count + right.assembled_group_count,
        cuda_edge_cancel_s=left.cuda_edge_cancel_s + right.cuda_edge_cancel_s,
        cuda_edge_cancel_kernel_count=left.cuda_edge_cancel_kernel_count + right.cuda_edge_cancel_kernel_count,
        cuda_edge_cancel_event_elapsed_s=left.cuda_edge_cancel_event_elapsed_s + right.cuda_edge_cancel_event_elapsed_s,
        cuda_edge_cancel_transferred_bytes=left.cuda_edge_cancel_transferred_bytes + right.cuda_edge_cancel_transferred_bytes,
        native_edge_cancel_s=left.native_edge_cancel_s + right.native_edge_cancel_s,
        cuda_loop_trace_s=left.cuda_loop_trace_s + right.cuda_loop_trace_s,
        metal_loop_trace_s=left.metal_loop_trace_s + right.metal_loop_trace_s,
        metal_loop_kernel_count=left.metal_loop_kernel_count + right.metal_loop_kernel_count,
        metal_loop_event_elapsed_s=left.metal_loop_event_elapsed_s + right.metal_loop_event_elapsed_s,
        metal_loop_transferred_bytes=left.metal_loop_transferred_bytes + right.metal_loop_transferred_bytes,
        native_loop_trace_s=left.native_loop_trace_s + right.native_loop_trace_s,
        python_loop_trace_s=left.python_loop_trace_s + right.python_loop_trace_s,
        hole_stitch_s=left.hole_stitch_s + right.hole_stitch_s,
    )


def _cached_component_side_diff(
    component: PolygonComponent,
    *,
    direction: str,
    cache: dict[tuple[str, tuple[object, ...]], tuple[tuple[DiffFragment, ...], _ComponentSideStats]],
) -> tuple[tuple[DiffFragment, ...], _ComponentSideStats]:
    origin = component.bbox[:2]
    key = (direction, _normalized_component_key(component, origin))
    cached = cache.get(key)
    if cached is None:
        fragments, stats = _component_side_diff(component, direction=direction)
        cached = (_translate_fragments(fragments, (-origin[0], -origin[1])), stats)
        cache[key] = cached
    fragments, stats = cached
    return _translate_fragments(fragments, origin), stats


def _merge_contacting_source_components(components: tuple[PolygonComponent, ...]) -> tuple[PolygonComponent, ...]:
    by_layer: dict[object, list[PolygonComponent]] = {}
    for component in components:
        by_layer.setdefault(component.layer, []).append(component)

    merged: list[PolygonComponent] = []
    for layer in sorted(by_layer):
        layer_components = tuple(sorted(by_layer[layer], key=_component_sort_key))
        for indices in _component_bbox_contact_groups(layer_components):
            polygons = {}
            for index in indices:
                for polygon in layer_components[index].polygons:
                    polygons[(polygon.side, polygon.fragment.bbox, polygon.fragment.twice_area, polygon.fragment.points)] = polygon
            merged.append(
                PolygonComponent(
                    layer=layer,
                    polygons=tuple(sorted(polygons.values(), key=lambda polygon: (polygon.side, polygon.source_index, polygon.role, polygon.fragment.bbox))),
                )
            )
    return tuple(sorted(merged, key=_component_sort_key))


def _component_bbox_contact_groups(components: tuple[PolygonComponent, ...]) -> tuple[tuple[int, ...], ...]:
    bboxes = tuple(component.bbox for component in components)
    remaining = set(range(len(components)))
    groups: list[tuple[int, ...]] = []
    while remaining:
        seed = remaining.pop()
        group = {seed}
        stack = [seed]
        while stack:
            left = stack.pop()
            touched = [right for right in remaining if _bbox_contact(bboxes[left], bboxes[right])]
            for right in touched:
                remaining.remove(right)
                group.add(right)
                stack.append(right)
        groups.append(tuple(sorted(group)))
    return tuple(sorted(groups))


def _bbox_contact(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> bool:
    return max(left[0], right[0]) <= min(left[2], right[2]) and max(left[1], right[1]) <= min(left[3], right[3])


def _normalized_component_key(component: PolygonComponent, origin: Point) -> tuple[object, ...]:
    ox, oy = origin
    records = []
    for polygon in component.polygons:
        points = tuple((point[0] - ox, point[1] - oy) for point in polygon.fragment.points)
        records.append((polygon.side, points))
    return tuple(sorted(records))


def _translate_fragments(fragments: tuple[DiffFragment, ...], delta: Point) -> tuple[DiffFragment, ...]:
    dx, dy = delta
    if dx == 0 and dy == 0:
        return fragments
    out = []
    for fragment in fragments:
        points = tuple((point[0] + dx, point[1] + dy) for point in fragment.points)
        bbox = (
            fragment.bbox[0] + dx,
            fragment.bbox[1] + dy,
            fragment.bbox[2] + dx,
            fragment.bbox[3] + dy,
        )
        out.append(DiffFragment(points=points, bbox=bbox, twice_area=fragment.twice_area))
    return tuple(out)


@dataclass(frozen=True)
class _ComponentSideStats:
    boundary_edge_count: int
    loop_count: int
    dropped_zero_area_loop_count: int
    gpu_loop_component_count: int = 0
    gpu_loop_edge_count: int = 0
    cuda_loop_component_count: int = 0
    cuda_loop_edge_count: int = 0
    metal_loop_component_count: int = 0
    metal_loop_edge_count: int = 0
    native_loop_component_count: int = 0
    native_loop_edge_count: int = 0
    gpu_boundary_component_count: int = 0
    gpu_boundary_edge_count: int = 0
    gpu_split_free_component_count: int = 0
    gpu_mixed_boundary_component_count: int = 0
    gpu_mixed_boundary_edge_count: int = 0
    nested_direction_fragment_component_count: int = 0


def _component_side_diff(component: PolygonComponent, *, direction: str) -> tuple[tuple[DiffFragment, ...], _ComponentSideStats]:
    if direction not in {"old", "new"}:
        raise ValueError(f"unknown diff direction: {direction!r}")
    target_fragments = tuple(polygon.fragment for polygon in component.polygons if polygon.side == direction)
    opposite = tuple(polygon for polygon in component.polygons if polygon.side != direction)
    if len(target_fragments) == 1 and not opposite:
        return target_fragments, _ComponentSideStats(0, 1, 0)
    if target_fragments and not opposite:
        return _one_sided_union_diff(target_fragments)
    polygons_by_side = {
        "old": tuple(polygon.fragment.points for polygon in component.polygons if polygon.side == "old"),
        "new": tuple(polygon.fragment.points for polygon in component.polygons if polygon.side == "new"),
    }
    bboxes_by_side = {
        side: tuple(polygon_bbox(points) for points in polygons)
        for side, polygons in polygons_by_side.items()
    }
    if not polygons_by_side[direction]:
        return (), _ComponentSideStats(0, 0, 0)
    segments = _build_segments(polygons_by_side)
    if not segments:
        return (), _ComponentSideStats(0, 0, 0)
    splits = _segment_splits(segments)
    boundary_edges, used_gpu = _maybe_cuda_boundary_edges_for_direction(
        segments,
        splits,
        polygons_by_side,
        bboxes_by_side,
        direction=direction,
    )
    fragments, dropped, loop_backend = _fragments_from_boundary_edges(boundary_edges)
    return (
        tuple(sorted(fragments, key=_fragment_sort_key)),
        _ComponentSideStats(
            boundary_edge_count=len(boundary_edges),
            loop_count=len(fragments),
            dropped_zero_area_loop_count=dropped,
            gpu_loop_component_count=1 if loop_backend in {"cuda", "metal"} else 0,
            gpu_loop_edge_count=len(boundary_edges) if loop_backend in {"cuda", "metal"} else 0,
            cuda_loop_component_count=1 if loop_backend == "cuda" else 0,
            cuda_loop_edge_count=len(boundary_edges) if loop_backend == "cuda" else 0,
            metal_loop_component_count=1 if loop_backend == "metal" else 0,
            metal_loop_edge_count=len(boundary_edges) if loop_backend == "metal" else 0,
            native_loop_component_count=1 if loop_backend == "native" else 0,
            native_loop_edge_count=len(boundary_edges) if loop_backend == "native" else 0,
            gpu_boundary_component_count=1 if used_gpu else 0,
            gpu_boundary_edge_count=len(boundary_edges) if used_gpu else 0,
            gpu_mixed_boundary_component_count=1 if used_gpu else 0,
            gpu_mixed_boundary_edge_count=len(boundary_edges) if used_gpu else 0,
            nested_direction_fragment_component_count=1 if _fragments_have_nested_bboxes(fragments) else 0,
        ),
    )


def _one_sided_union_diff(target_fragments: tuple[DiffFragment, ...]) -> tuple[tuple[DiffFragment, ...], _ComponentSideStats]:
    polygons = tuple(fragment.points for fragment in target_fragments)
    bboxes = tuple(fragment.bbox for fragment in target_fragments)
    segments = _build_segments({"target": polygons})
    if not segments:
        return (), _ComponentSideStats(0, 0, 0)
    gpu_split_free = False
    boundary_edges, used_gpu = _maybe_cuda_unsplit_boundary_edges_for_one_sided_union(segments, polygons, bboxes)
    if boundary_edges is not None:
        gpu_split_free = True
    else:
        splits = _segment_splits(segments)
        boundary_edges, used_gpu = _maybe_cuda_boundary_edges_for_one_sided_union(segments, splits, polygons, bboxes)
    fragments, dropped, loop_backend = _fragments_from_boundary_edges(boundary_edges)
    return (
        tuple(sorted(fragments, key=_fragment_sort_key)),
        _ComponentSideStats(
            boundary_edge_count=len(boundary_edges),
            loop_count=len(fragments),
            dropped_zero_area_loop_count=dropped,
            gpu_loop_component_count=1 if loop_backend in {"cuda", "metal"} else 0,
            gpu_loop_edge_count=len(boundary_edges) if loop_backend in {"cuda", "metal"} else 0,
            cuda_loop_component_count=1 if loop_backend == "cuda" else 0,
            cuda_loop_edge_count=len(boundary_edges) if loop_backend == "cuda" else 0,
            metal_loop_component_count=1 if loop_backend == "metal" else 0,
            metal_loop_edge_count=len(boundary_edges) if loop_backend == "metal" else 0,
            native_loop_component_count=1 if loop_backend == "native" else 0,
            native_loop_edge_count=len(boundary_edges) if loop_backend == "native" else 0,
            gpu_boundary_component_count=1 if used_gpu else 0,
            gpu_boundary_edge_count=len(boundary_edges) if used_gpu else 0,
            gpu_split_free_component_count=1 if gpu_split_free else 0,
            nested_direction_fragment_component_count=1 if _fragments_have_nested_bboxes(fragments) else 0,
        ),
    )


def _fragments_have_nested_bboxes(fragments: Sequence[DiffFragment]) -> bool:
    bboxes = tuple(fragment.bbox for fragment in fragments)
    for index, outer in enumerate(bboxes):
        for inner_index, inner in enumerate(bboxes):
            if index != inner_index and _bbox_strictly_contains(outer, inner):
                return True
    return False


def _bbox_strictly_contains(
    outer: tuple[int, int, int, int],
    inner: tuple[int, int, int, int],
) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
        and outer != inner
    )


def _maybe_cuda_unsplit_boundary_edges_for_one_sided_union(
    segments: tuple[_SegmentRecord, ...],
    polygons: tuple[tuple[Point, ...], ...],
    bboxes: tuple[tuple[int, int, int, int], ...],
) -> tuple[tuple[FloatSegment, ...] | None, bool]:
    if len(segments) < 256 or len(polygons) < 4:
        return None, False
    try:
        edges, safe = _cuda_unsplit_boundary_edges_for_one_sided_union(segments, polygons, bboxes)
    except Exception:
        return None, False
    if not safe:
        return None, False
    return edges, True


def _cuda_unsplit_boundary_edges_for_one_sided_union(
    segments: tuple[_SegmentRecord, ...],
    polygons: tuple[tuple[Point, ...], ...],
    bboxes: tuple[tuple[int, int, int, int], ...],
) -> tuple[tuple[FloatSegment, ...], bool]:
    from .backends.cuda_ctypes import run_cuda_one_sided_boundary_edges_unsplit_if_safe

    vertices, offsets, signed_areas, segment_points, segment_polygon_indices = _one_sided_cuda_arrays(
        segments,
        polygons,
    )
    edges, safe = run_cuda_one_sided_boundary_edges_unsplit_if_safe(
        vertices,
        offsets,
        np.asarray(bboxes, dtype=np.int64),
        signed_areas,
        segment_points,
        segment_polygon_indices,
    )
    return tuple(sorted(((_point_key((edge[0][0], edge[0][1])), _point_key((edge[1][0], edge[1][1]))) for edge in edges))), safe


def _maybe_cuda_boundary_edges_for_one_sided_union(
    segments: tuple[_SegmentRecord, ...],
    splits: tuple[tuple[float, ...], ...],
    polygons: tuple[tuple[Point, ...], ...],
    bboxes: tuple[tuple[int, int, int, int], ...],
) -> tuple[tuple[FloatSegment, ...], bool]:
    piece_count = sum(1 for split in splits for left, right in zip(split, split[1:]) if right - left > 1e-10)
    if piece_count < 256 or len(polygons) < 4:
        return _boundary_edges_for_one_sided_union(segments, splits, polygons, bboxes), False
    try:
        return _cuda_boundary_edges_for_one_sided_union(segments, splits, polygons, bboxes), True
    except Exception:
        pass
    # Keep Metal boundary sampling as a direct backend experiment for now. The
    # production source-edge path uses the CPU reference sampler for parity on
    # compact, overlapping undercut geometry.
    return _boundary_edges_for_one_sided_union(segments, splits, polygons, bboxes), False


def _maybe_cuda_boundary_edges_for_direction(
    segments: tuple[_SegmentRecord, ...],
    splits: tuple[tuple[float, ...], ...],
    polygons_by_side: dict[str, tuple[tuple[Point, ...], ...]],
    bboxes_by_side: dict[str, tuple[tuple[int, int, int, int], ...]],
    *,
    direction: str,
) -> tuple[tuple[FloatSegment, ...], bool]:
    piece_count = sum(1 for split in splits for left, right in zip(split, split[1:]) if right - left > 1e-10)
    if piece_count < 128 or not polygons_by_side["old"] or not polygons_by_side["new"]:
        return _boundary_edges_for_direction(segments, splits, polygons_by_side, bboxes_by_side, direction=direction), False
    try:
        return _cuda_boundary_edges_for_direction(segments, splits, polygons_by_side, bboxes_by_side, direction=direction), True
    except Exception:
        pass
    # Keep Metal boundary sampling as a direct backend experiment for now. The
    # production source-edge path uses the CPU reference sampler for parity on
    # compact, overlapping undercut geometry.
    return _boundary_edges_for_direction(segments, splits, polygons_by_side, bboxes_by_side, direction=direction), False


def _cuda_boundary_edges_for_direction(
    segments: tuple[_SegmentRecord, ...],
    splits: tuple[tuple[float, ...], ...],
    polygons_by_side: dict[str, tuple[tuple[Point, ...], ...]],
    bboxes_by_side: dict[str, tuple[tuple[int, int, int, int], ...]],
    *,
    direction: str,
) -> tuple[FloatSegment, ...]:
    from .backends.cuda_ctypes import run_cuda_mixed_boundary_edges

    vertices, offsets, bboxes, polygon_sides = _mixed_cuda_polygon_arrays(polygons_by_side, bboxes_by_side)
    piece_rows: list[tuple[float, float, float, float]] = []
    for segment_index, segment in enumerate(segments):
        for left_t, right_t in zip(splits[segment_index], splits[segment_index][1:]):
            if right_t - left_t <= 1e-10:
                continue
            start = _point_at(segment, left_t)
            stop = _point_at(segment, right_t)
            piece_rows.append((start[0], start[1], stop[0], stop[1]))
    if not piece_rows:
        return ()
    edges = run_cuda_mixed_boundary_edges(
        vertices,
        offsets,
        bboxes,
        polygon_sides,
        np.asarray(piece_rows, dtype=np.float64),
        direction=direction,
    )
    return tuple(sorted(((_point_key((edge[0][0], edge[0][1])), _point_key((edge[1][0], edge[1][1]))) for edge in edges)))


def _metal_boundary_edges_for_direction(
    segments: tuple[_SegmentRecord, ...],
    splits: tuple[tuple[float, ...], ...],
    polygons_by_side: dict[str, tuple[tuple[Point, ...], ...]],
    bboxes_by_side: dict[str, tuple[tuple[int, int, int, int], ...]],
    *,
    direction: str,
) -> tuple[FloatSegment, ...]:
    from .backends.metal_ctypes import run_metal_mixed_boundary_edges

    vertices, offsets, bboxes, polygon_sides = _mixed_cuda_polygon_arrays(polygons_by_side, bboxes_by_side)
    piece_rows: list[tuple[float, float, float, float]] = []
    for segment_index, segment in enumerate(segments):
        for left_t, right_t in zip(splits[segment_index], splits[segment_index][1:]):
            if right_t - left_t <= 1e-10:
                continue
            start = _point_at(segment, left_t)
            stop = _point_at(segment, right_t)
            piece_rows.append((start[0], start[1], stop[0], stop[1]))
    if not piece_rows:
        return ()
    edges = run_metal_mixed_boundary_edges(
        vertices,
        offsets,
        bboxes,
        polygon_sides,
        np.asarray(piece_rows, dtype=np.float64),
        direction=direction,
    )
    return tuple(sorted(((_point_key((edge[0][0], edge[0][1])), _point_key((edge[1][0], edge[1][1]))) for edge in edges)))


def _mixed_cuda_polygon_arrays(
    polygons_by_side: dict[str, tuple[tuple[Point, ...], ...]],
    bboxes_by_side: dict[str, tuple[tuple[int, int, int, int], ...]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vertex_rows: list[Point] = []
    offsets = [0]
    bboxes: list[tuple[int, int, int, int]] = []
    sides: list[int] = []
    for side, side_id in (("old", 1), ("new", 2)):
        for polygon, bbox in zip(polygons_by_side[side], bboxes_by_side[side]):
            vertex_rows.extend(polygon)
            offsets.append(len(vertex_rows))
            bboxes.append(bbox)
            sides.append(side_id)
    return (
        np.asarray(vertex_rows, dtype=np.int64),
        np.asarray(offsets, dtype=np.uint64),
        np.asarray(bboxes, dtype=np.int64),
        np.asarray(sides, dtype=np.uint8),
    )


def _cuda_boundary_edges_for_one_sided_union(
    segments: tuple[_SegmentRecord, ...],
    splits: tuple[tuple[float, ...], ...],
    polygons: tuple[tuple[Point, ...], ...],
    bboxes: tuple[tuple[int, int, int, int], ...],
) -> tuple[FloatSegment, ...]:
    from .backends.cuda_ctypes import run_cuda_one_sided_boundary_edges

    piece_rows: list[tuple[float, float, float, float]] = []
    piece_polygon_indices: list[int] = []
    for segment_index, segment in enumerate(segments):
        for left_t, right_t in zip(splits[segment_index], splits[segment_index][1:]):
            if right_t - left_t <= 1e-10:
                continue
            start = _point_at(segment, left_t)
            stop = _point_at(segment, right_t)
            piece_rows.append((start[0], start[1], stop[0], stop[1]))
            piece_polygon_indices.append(segment.polygon_index)
    if not piece_rows:
        return ()
    vertices, offsets, signed_areas, _segment_points, _segment_polygon_indices = _one_sided_cuda_arrays(
        (),
        polygons,
    )
    edges = run_cuda_one_sided_boundary_edges(
        vertices,
        offsets,
        np.asarray(bboxes, dtype=np.int64),
        signed_areas,
        np.asarray(piece_rows, dtype=np.float64),
        np.asarray(piece_polygon_indices, dtype=np.uint32),
    )
    return tuple(sorted(((_point_key((edge[0][0], edge[0][1])), _point_key((edge[1][0], edge[1][1]))) for edge in edges)))


def _metal_boundary_edges_for_one_sided_union(
    segments: tuple[_SegmentRecord, ...],
    splits: tuple[tuple[float, ...], ...],
    polygons: tuple[tuple[Point, ...], ...],
    bboxes: tuple[tuple[int, int, int, int], ...],
) -> tuple[FloatSegment, ...]:
    from .backends.metal_ctypes import run_metal_one_sided_boundary_edges

    piece_rows: list[tuple[float, float, float, float]] = []
    piece_polygon_indices: list[int] = []
    for segment_index, segment in enumerate(segments):
        for left_t, right_t in zip(splits[segment_index], splits[segment_index][1:]):
            if right_t - left_t <= 1e-10:
                continue
            start = _point_at(segment, left_t)
            stop = _point_at(segment, right_t)
            piece_rows.append((start[0], start[1], stop[0], stop[1]))
            piece_polygon_indices.append(segment.polygon_index)
    if not piece_rows:
        return ()
    vertices, offsets, signed_areas, _segment_points, _segment_polygon_indices = _one_sided_cuda_arrays(
        (),
        polygons,
    )
    edges = run_metal_one_sided_boundary_edges(
        vertices,
        offsets,
        np.asarray(bboxes, dtype=np.int64),
        signed_areas,
        np.asarray(piece_rows, dtype=np.float64),
        np.asarray(piece_polygon_indices, dtype=np.uint32),
    )
    return tuple(sorted(((_point_key((edge[0][0], edge[0][1])), _point_key((edge[1][0], edge[1][1]))) for edge in edges)))


def _one_sided_cuda_arrays(
    segments: tuple[_SegmentRecord, ...],
    polygons: tuple[tuple[Point, ...], ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vertex_rows: list[Point] = []
    offsets = [0]
    signed_areas: list[int] = []
    for polygon in polygons:
        vertex_rows.extend(polygon)
        offsets.append(len(vertex_rows))
        signed_areas.append(_signed_twice_area(polygon))
    segment_rows = tuple((segment.start[0], segment.start[1], segment.stop[0], segment.stop[1]) for segment in segments)
    segment_polygon_indices = tuple(segment.polygon_index for segment in segments)
    return (
        np.asarray(vertex_rows, dtype=np.int64),
        np.asarray(offsets, dtype=np.uint64),
        np.asarray(signed_areas, dtype=np.int64),
        np.asarray(segment_rows, dtype=np.float64).reshape((-1, 4)),
        np.asarray(segment_polygon_indices, dtype=np.uint32),
    )


def _fragments_from_boundary_edges(boundary_edges: tuple[FloatSegment, ...]) -> tuple[list[DiffFragment], int, str]:
    integer_edges = _integer_edges_array_if_exact(boundary_edges)
    if integer_edges is not None:
        cuda = _try_cuda_fragments_from_integer_boundary_edges(integer_edges)
        if cuda is not None:
            return list(cuda.fragments), cuda.dropped_zero_area_loop_count, "cuda"
        metal = _try_metal_fragments_from_integer_boundary_edges(integer_edges)
        if metal is not None:
            return list(metal.fragments), metal.dropped_zero_area_loop_count, "metal"
        native = _try_native_fragments_from_integer_boundary_edges(integer_edges)
        if native is not None:
            return list(native.fragments), native.dropped_zero_area_loop_count, "native"
    loops = _trace_loops(boundary_edges)
    fragments: list[DiffFragment] = []
    dropped = 0
    for loop in loops:
        points = _clean_integer_loop(_snap_loop(loop))
        if len(points) < 3:
            dropped += 1
            continue
        twice_area = polygon_twice_area(points)
        if twice_area == 0:
            dropped += 1
            continue
        fragments.append(DiffFragment(points=points, bbox=polygon_bbox(points), twice_area=twice_area))
    return fragments, dropped, "python"


def _integer_edges_array_if_exact(boundary_edges: tuple[FloatSegment, ...]) -> np.ndarray | None:
    rows: list[tuple[int, int, int, int]] = []
    for start, stop in boundary_edges:
        snapped = (_snap_coordinate(start[0]), _snap_coordinate(start[1]), _snap_coordinate(stop[0]), _snap_coordinate(stop[1]))
        if snapped[0] is None or snapped[1] is None or snapped[2] is None or snapped[3] is None:
            return None
        rows.append((snapped[0], snapped[1], snapped[2], snapped[3]))
    return np.asarray(rows, dtype=np.int64)


def _try_cuda_fragments_from_integer_boundary_edges(edges: np.ndarray):
    if len(edges) > 4096:
        return None
    try:
        from .backends.cuda_ctypes import run_cuda_boundary_loops_from_edges

        return run_cuda_boundary_loops_from_edges(edges)
    except Exception:
        return None


def _try_metal_fragments_from_integer_boundary_edges(edges: np.ndarray):
    # The Metal loop tracer is kept as a direct backend experiment, but source-edge
    # evidence needs the native/Python loop assemblers for parity on compact
    # overlapping undercut geometry.
    return None


def _try_native_fragments_from_integer_boundary_edges(edges: np.ndarray):
    if len(edges) < 64:
        return None
    try:
        from .backends.boundary_loop_ctypes import native_boundary_loops_from_edges

        return native_boundary_loops_from_edges(edges)
    except Exception:
        return None


def _snap_coordinate(value: float) -> int | None:
    if not isfinite(value):
        return None
    rounded = round(value)
    if abs(value - rounded) > 1e-9:
        return None
    return int(rounded)


def _build_segments(polygons_by_side: dict[str, tuple[tuple[Point, ...], ...]]) -> tuple[_SegmentRecord, ...]:
    segments: list[_SegmentRecord] = []
    for side, polygons in polygons_by_side.items():
        for polygon_index, points in enumerate(polygons):
            for index, point in enumerate(points):
                next_point = points[(index + 1) % len(points)]
                if point == next_point:
                    continue
                segments.append(
                    _SegmentRecord(
                        side=side,
                        polygon_index=polygon_index,
                        start=(float(point[0]), float(point[1])),
                        stop=(float(next_point[0]), float(next_point[1])),
                    )
                )
    return tuple(segments)


def _segment_splits(segments: tuple[_SegmentRecord, ...]) -> tuple[tuple[float, ...], ...]:
    splits: list[set[float]] = [{0.0, 1.0} for _ in segments]
    bboxes = tuple(_segment_bbox(segment) for segment in segments)
    ordered_indices = tuple(sorted(range(len(segments)), key=lambda index: (bboxes[index][0], bboxes[index][1], index)))
    for left_order_index, left_index in enumerate(ordered_indices):
        left = segments[left_index]
        left_bbox = bboxes[left_index]
        left_dx = left.stop[0] - left.start[0]
        left_dy = left.stop[1] - left.start[1]
        for right_index in ordered_indices[left_order_index + 1 :]:
            right_bbox = bboxes[right_index]
            if right_bbox[0] > left_bbox[2]:
                break
            if not _float_bbox_contact(left_bbox, right_bbox):
                continue
            right = segments[right_index]
            right_dx = right.stop[0] - right.start[0]
            right_dy = right.stop[1] - right.start[1]
            denom = _cross(left_dx, left_dy, right_dx, right_dy)
            cx = right.start[0] - left.start[0]
            cy = right.start[1] - left.start[1]
            if abs(denom) <= 1e-9:
                if abs(_cross(cx, cy, left_dx, left_dy)) <= 1e-6:
                    _add_collinear_splits(splits, left_index, right_index, left, right)
                continue
            left_t = _cross(cx, cy, right_dx, right_dy) / denom
            right_t = _cross(cx, cy, left_dx, left_dy) / denom
            if _within_unit_interval(left_t) and _within_unit_interval(right_t):
                _add_split(splits[left_index], left_t)
                _add_split(splits[right_index], right_t)
    return tuple(tuple(sorted(values)) for values in splits)


def _segment_bbox(segment: _SegmentRecord) -> tuple[float, float, float, float]:
    return (
        min(segment.start[0], segment.stop[0]),
        min(segment.start[1], segment.stop[1]),
        max(segment.start[0], segment.stop[0]),
        max(segment.start[1], segment.stop[1]),
    )


def _float_bbox_contact(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> bool:
    return max(left[0], right[0]) <= min(left[2], right[2]) and max(left[1], right[1]) <= min(left[3], right[3])


def _add_collinear_splits(
    splits: list[set[float]],
    left_index: int,
    right_index: int,
    left: _SegmentRecord,
    right: _SegmentRecord,
) -> None:
    left_dx = left.stop[0] - left.start[0]
    left_dy = left.stop[1] - left.start[1]
    right_dx = right.stop[0] - right.start[0]
    right_dy = right.stop[1] - right.start[1]
    left_len2 = left_dx * left_dx + left_dy * left_dy
    right_len2 = right_dx * right_dx + right_dy * right_dy
    if left_len2:
        for point in (right.start, right.stop):
            _add_split(splits[left_index], ((point[0] - left.start[0]) * left_dx + (point[1] - left.start[1]) * left_dy) / left_len2)
    if right_len2:
        for point in (left.start, left.stop):
            _add_split(splits[right_index], ((point[0] - right.start[0]) * right_dx + (point[1] - right.start[1]) * right_dy) / right_len2)


def _boundary_edges_for_direction(
    segments: tuple[_SegmentRecord, ...],
    splits: tuple[tuple[float, ...], ...],
    polygons_by_side: dict[str, tuple[tuple[Point, ...], ...]],
    bboxes_by_side: dict[str, tuple[tuple[int, int, int, int], ...]],
    *,
    direction: str,
) -> tuple[FloatSegment, ...]:
    counts: dict[FloatSegment, int] = defaultdict(int)
    for segment_index, segment in enumerate(segments):
        dx = segment.stop[0] - segment.start[0]
        dy = segment.stop[1] - segment.start[1]
        length = hypot(dx, dy)
        if length <= 1e-12:
            continue
        normal = (-dy / length, dx / length)
        for left_t, right_t in zip(splits[segment_index], splits[segment_index][1:]):
            if right_t - left_t <= 1e-10:
                continue
            start = _point_at(segment, left_t)
            stop = _point_at(segment, right_t)
            midpoint = ((start[0] + stop[0]) * 0.5, (start[1] + stop[1]) * 0.5)
            left_sample = (midpoint[0] + normal[0] * 1e-5, midpoint[1] + normal[1] * 1e-5)
            right_sample = (midpoint[0] - normal[0] * 1e-5, midpoint[1] - normal[1] * 1e-5)
            left_inside = _target_inside(polygons_by_side, bboxes_by_side, left_sample, direction)
            right_inside = _target_inside(polygons_by_side, bboxes_by_side, right_sample, direction)
            if left_inside and not right_inside:
                _add_boundary_edge(counts, (_point_key(start), _point_key(stop)))
            elif right_inside and not left_inside:
                _add_boundary_edge(counts, (_point_key(stop), _point_key(start)))
    edges: list[FloatSegment] = []
    for edge, count in counts.items():
        edges.extend(edge for _ in range(count))
    return tuple(sorted(edges))


def _boundary_edges_for_one_sided_union(
    segments: tuple[_SegmentRecord, ...],
    splits: tuple[tuple[float, ...], ...],
    polygons: tuple[tuple[Point, ...], ...],
    bboxes: tuple[tuple[int, int, int, int], ...],
) -> tuple[FloatSegment, ...]:
    signed_areas = tuple(_signed_twice_area(polygon) for polygon in polygons)
    counts: dict[FloatSegment, int] = defaultdict(int)
    for segment_index, segment in enumerate(segments):
        signed_area = signed_areas[segment.polygon_index]
        if signed_area == 0:
            continue
        dx = segment.stop[0] - segment.start[0]
        dy = segment.stop[1] - segment.start[1]
        length = hypot(dx, dy)
        if length <= 1e-12:
            continue
        normal = (-dy / length, dx / length)
        for left_t, right_t in zip(splits[segment_index], splits[segment_index][1:]):
            if right_t - left_t <= 1e-10:
                continue
            start = _point_at(segment, left_t)
            stop = _point_at(segment, right_t)
            midpoint = ((start[0] + stop[0]) * 0.5, (start[1] + stop[1]) * 0.5)
            if signed_area > 0:
                outside_sample = (midpoint[0] - normal[0] * 1e-5, midpoint[1] - normal[1] * 1e-5)
                if not _union_contains(polygons, bboxes, outside_sample):
                    _add_boundary_edge(counts, (_point_key(start), _point_key(stop)))
            else:
                outside_sample = (midpoint[0] + normal[0] * 1e-5, midpoint[1] + normal[1] * 1e-5)
                if not _union_contains(polygons, bboxes, outside_sample):
                    _add_boundary_edge(counts, (_point_key(stop), _point_key(start)))
    edges: list[FloatSegment] = []
    for edge, count in counts.items():
        edges.extend(edge for _ in range(count))
    return tuple(sorted(edges))


def _add_boundary_edge(counts: dict[FloatSegment, int], edge: FloatSegment) -> None:
    reverse = (edge[1], edge[0])
    if counts.get(reverse, 0):
        counts[reverse] -= 1
        if counts[reverse] == 0:
            del counts[reverse]
    else:
        counts[edge] += 1


def _trace_loops(edges: tuple[FloatSegment, ...]) -> tuple[tuple[FloatPoint, ...], ...]:
    outgoing: dict[FloatPoint, list[FloatPoint]] = defaultdict(list)
    for start, stop in edges:
        outgoing[start].append(stop)
    for targets in outgoing.values():
        targets.sort()

    remaining: set[FloatSegment] = set(edges)
    loops: list[tuple[FloatPoint, ...]] = []
    while remaining:
        start, stop = min(remaining)
        remaining.remove((start, stop))
        loop = [start, stop]
        previous = start
        current = stop
        while current != start:
            candidates = [point for point in outgoing.get(current, ()) if (current, point) in remaining]
            if not candidates:
                break
            in_vec = (current[0] - previous[0], current[1] - previous[1])
            next_point = min(
                candidates,
                key=lambda point: (_turn_key(in_vec, (point[0] - current[0], point[1] - current[1])), point),
            )
            remaining.remove((current, next_point))
            loop.append(next_point)
            previous, current = current, next_point
        if loop[-1] == start:
            loop.pop()
        loops.append(tuple(loop))
    return tuple(sorted(loops))


def _target_inside(
    polygons_by_side: dict[str, tuple[tuple[Point, ...], ...]],
    bboxes_by_side: dict[str, tuple[tuple[int, int, int, int], ...]],
    point: FloatPoint,
    direction: str,
) -> bool:
    old_inside = _union_contains(polygons_by_side["old"], bboxes_by_side["old"], point)
    new_inside = _union_contains(polygons_by_side["new"], bboxes_by_side["new"], point)
    if direction == "old":
        return old_inside and not new_inside
    return new_inside and not old_inside


def _union_contains(
    polygons: tuple[tuple[Point, ...], ...],
    bboxes: tuple[tuple[int, int, int, int], ...],
    point: FloatPoint,
) -> bool:
    return any(
        _point_in_bbox(point, bbox) and _point_in_polygon(point, polygon)
        for polygon, bbox in zip(polygons, bboxes)
    )


def _point_in_bbox(point: FloatPoint, bbox: tuple[int, int, int, int]) -> bool:
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def _point_in_polygon(point: FloatPoint, polygon: tuple[Point, ...]) -> bool:
    x, y = point
    inside = False
    for index, start in enumerate(polygon):
        stop = polygon[(index + 1) % len(polygon)]
        x0, y0 = float(start[0]), float(start[1])
        x1, y1 = float(stop[0]), float(stop[1])
        if (y0 > y) != (y1 > y):
            x_intersection = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if x_intersection > x:
                inside = not inside
    return inside


def _signed_twice_area(points: tuple[Point, ...]) -> int:
    area = 0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        area += point[0] * next_point[1] - next_point[0] * point[1]
    return area


def _snap_loop(loop: tuple[FloatPoint, ...]) -> tuple[Point, ...]:
    points: list[Point] = []
    for point in loop:
        if not isfinite(point[0]) or not isfinite(point[1]):
            continue
        snapped = (round(point[0]), round(point[1]))
        if not points or points[-1] != snapped:
            points.append(snapped)
    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
    return tuple(points)


def _clean_integer_loop(points: tuple[Point, ...]) -> tuple[Point, ...]:
    cleaned = list(points)
    changed = True
    while changed and len(cleaned) >= 3:
        changed = False
        next_points: list[Point] = []
        for index, point in enumerate(cleaned):
            previous = cleaned[index - 1]
            following = cleaned[(index + 1) % len(cleaned)]
            if _collinear(previous, point, following):
                changed = True
                continue
            next_points.append(point)
        cleaned = next_points
    return tuple(cleaned)


def _point_at(segment: _SegmentRecord, t: float) -> FloatPoint:
    return (
        segment.start[0] + (segment.stop[0] - segment.start[0]) * t,
        segment.start[1] + (segment.stop[1] - segment.start[1]) * t,
    )


def _point_key(point: FloatPoint) -> FloatPoint:
    return (round(point[0], 9), round(point[1], 9))


def _add_split(splits: set[float], value: float) -> None:
    if _within_unit_interval(value):
        splits.add(min(1.0, max(0.0, value)))


def _within_unit_interval(value: float) -> bool:
    return -1e-9 <= value <= 1.0 + 1e-9


def _cross(left_x: float, left_y: float, right_x: float, right_y: float) -> float:
    return left_x * right_y - left_y * right_x


def _turn_key(in_vec: FloatPoint, out_vec: FloatPoint) -> tuple[int, float, float]:
    cross = in_vec[0] * out_vec[1] - in_vec[1] * out_vec[0]
    dot = in_vec[0] * out_vec[0] + in_vec[1] * out_vec[1]
    if cross < 0:
        turn_class = 0
    elif cross == 0 and dot > 0:
        turn_class = 1
    elif cross > 0:
        turn_class = 2
    else:
        turn_class = 3
    return turn_class, -abs(cross), -dot


def _collinear(left: Point, middle: Point, right: Point) -> bool:
    return (middle[0] - left[0]) * (right[1] - middle[1]) == (middle[1] - left[1]) * (right[0] - middle[0])
