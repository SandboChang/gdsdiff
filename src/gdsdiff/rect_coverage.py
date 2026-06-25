from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .config import LayerSpec
from .exact_oracle import DiffFragment, ExactOracleResult, LayerExactDiff
from .geometry import PolygonBuffer, polygon_bbox, polygon_twice_area
from .polygon_components import _fragment_sort_key, _merge_layer_side_fragments


class RectCoverageUnsupported(RuntimeError):
    pass


Rect = tuple[int, int, int, int]


@dataclass(frozen=True)
class RectCoverageStats:
    component_count: int
    cuda_task_count: int
    old_rect_count: int
    new_rect_count: int
    dirty_rect_count: int
    max_old_rects_per_component: int
    max_new_rects_per_component: int
    transferred_bytes: int
    cuda_kernel_count: int
    h2d_upload_s: float
    gpu_xor_s: float
    gpu_record_emit_s: float

    def to_json(self) -> dict[str, object]:
        return {
            "component_count": self.component_count,
            "cuda_task_count": self.cuda_task_count,
            "old_rect_count": self.old_rect_count,
            "new_rect_count": self.new_rect_count,
            "dirty_rect_count": self.dirty_rect_count,
            "max_old_rects_per_component": self.max_old_rects_per_component,
            "max_new_rects_per_component": self.max_new_rects_per_component,
            "transferred_bytes": self.transferred_bytes,
            "cuda_kernel_count": self.cuda_kernel_count,
            "h2d_upload_s": self.h2d_upload_s,
            "gpu_xor_s": self.gpu_xor_s,
            "gpu_record_emit_s": self.gpu_record_emit_s,
        }


@dataclass(frozen=True)
class RectCoverageResult:
    exact_result: ExactOracleResult
    stats: RectCoverageStats


@dataclass(frozen=True)
class _RectComponent:
    layer: LayerSpec
    old_rects: tuple[Rect, ...]
    new_rects: tuple[Rect, ...]
    bbox: Rect


def run_cuda_rectangle_coverage_diff(
    old_buffer: PolygonBuffer,
    new_buffer: PolygonBuffer,
    *,
    max_rects_per_component: int = 16,
    max_fragments_per_component: int = 4096,
) -> RectCoverageResult:
    old_by_layer = _rectangles_by_layer(old_buffer)
    new_by_layer = _rectangles_by_layer(new_buffer)
    layers = tuple(sorted(set(old_by_layer) | set(new_by_layer)))
    components: list[_RectComponent] = []
    dirty_rect_count = 0
    for layer in layers:
        old_rects = tuple(sorted(set(old_by_layer.get(layer, ()))))
        new_rects = tuple(sorted(set(new_by_layer.get(layer, ()))))
        dirty = tuple(sorted((set(old_rects) ^ set(new_rects))))
        dirty_rect_count += len(dirty)
        components.extend(_dirty_components(layer, old_rects, new_rects, dirty))

    if not components:
        return RectCoverageResult(
            exact_result=ExactOracleResult(tuple(LayerExactDiff(layer=layer) for layer in layers)),
            stats=RectCoverageStats(
                component_count=0,
                cuda_task_count=0,
                old_rect_count=sum(len(rects) for rects in old_by_layer.values()),
                new_rect_count=sum(len(rects) for rects in new_by_layer.values()),
                dirty_rect_count=0,
                max_old_rects_per_component=0,
                max_new_rects_per_component=0,
                transferred_bytes=0,
                cuda_kernel_count=0,
                h2d_upload_s=0.0,
                gpu_xor_s=0.0,
                gpu_record_emit_s=0.0,
            ),
        )

    max_old = max(len(component.old_rects) for component in components)
    max_new = max(len(component.new_rects) for component in components)
    max_rects = max(max_old, max_new)
    if max_rects > max_rects_per_component:
        raise RectCoverageUnsupported(
            f"rectangle coverage component exceeds CUDA rectangle limit: {max_rects} > {max_rects_per_component}"
        )

    from .backends.cuda_ctypes import CudaCtypesUnavailable, run_cuda_rect_set_exact_fragments

    old_array = np.zeros((len(components), max_rects, 4), dtype=np.int64)
    new_array = np.zeros((len(components), max_rects, 4), dtype=np.int64)
    old_counts = np.zeros((len(components),), dtype=np.uint8)
    new_counts = np.zeros((len(components),), dtype=np.uint8)
    for row, component in enumerate(components):
        old_counts[row] = len(component.old_rects)
        new_counts[row] = len(component.new_rects)
        for col, rect in enumerate(component.old_rects):
            old_array[row, col] = rect
        for col, rect in enumerate(component.new_rects):
            new_array[row, col] = rect

    transferred_bytes = int(old_array.nbytes + new_array.nbytes + old_counts.nbytes + new_counts.nbytes)
    start = perf_counter()
    try:
        rects, counts, _areas = run_cuda_rect_set_exact_fragments(
            old_array,
            new_array,
            old_counts,
            new_counts,
            max_fragments=max_fragments_per_component,
        )
    except CudaCtypesUnavailable as exc:
        raise RectCoverageUnsupported(str(exc)) from exc
    gpu_s = perf_counter() - start

    diffs_by_layer: dict[LayerSpec, list[LayerExactDiff]] = {layer: [] for layer in layers}
    emit_start = perf_counter()
    for row, component in enumerate(components):
        old_fragments = tuple(_fragment_from_rect(tuple(int(v) for v in rects[row, 0, index])) for index in range(int(counts[row, 0])))
        new_fragments = tuple(_fragment_from_rect(tuple(int(v) for v in rects[row, 1, index])) for index in range(int(counts[row, 1])))
        diffs_by_layer[component.layer].append(
            LayerExactDiff(
                layer=component.layer,
                old_minus_new=old_fragments,
                new_minus_old=new_fragments,
                xor=old_fragments + new_fragments,
            )
        )
    emit_s = perf_counter() - emit_start
    layer_diffs = tuple(_merge_layer_diffs(layer, diffs_by_layer[layer]) for layer in layers)
    return RectCoverageResult(
        exact_result=ExactOracleResult(layer_diffs),
        stats=RectCoverageStats(
            component_count=len(components),
            cuda_task_count=len(components),
            old_rect_count=sum(len(rects) for rects in old_by_layer.values()),
            new_rect_count=sum(len(rects) for rects in new_by_layer.values()),
            dirty_rect_count=dirty_rect_count,
            max_old_rects_per_component=max_old,
            max_new_rects_per_component=max_new,
            transferred_bytes=transferred_bytes,
            cuda_kernel_count=1,
            h2d_upload_s=0.0,
            gpu_xor_s=gpu_s,
            gpu_record_emit_s=emit_s,
        ),
    )


def _rectangles_by_layer(buffer: PolygonBuffer) -> dict[LayerSpec, tuple[Rect, ...]]:
    by_layer: dict[LayerSpec, list[Rect]] = {}
    for index, layer_id in enumerate(buffer.layer_ids):
        if int(buffer.flags[index]) != 0:
            continue
        layer = buffer.layer_table.layers[int(layer_id)]
        rect = _as_rect(tuple((int(x), int(y)) for x, y in buffer.polygon_vertices(index)))
        if rect is None:
            raise RectCoverageUnsupported("non-rectangle polygon present in rectangle coverage path")
        by_layer.setdefault(layer, []).append(rect)
    return {layer: tuple(rects) for layer, rects in by_layer.items()}


def _as_rect(points: tuple[tuple[int, int], ...]) -> Rect | None:
    if len(points) != 4:
        return None
    xs = sorted({point[0] for point in points})
    ys = sorted({point[1] for point in points})
    if len(xs) != 2 or len(ys) != 2:
        return None
    if set(points) != {(xs[0], ys[0]), (xs[0], ys[1]), (xs[1], ys[0]), (xs[1], ys[1])}:
        return None
    if xs[0] >= xs[1] or ys[0] >= ys[1]:
        return None
    return xs[0], ys[0], xs[1], ys[1]


def _dirty_components(layer: LayerSpec, old_rects: tuple[Rect, ...], new_rects: tuple[Rect, ...], dirty: tuple[Rect, ...]) -> list[_RectComponent]:
    if not dirty:
        return []
    candidates = tuple(sorted(set(old_rects) | set(new_rects)))
    components: list[set[Rect]] = [{rect} for rect in dirty]
    changed = True
    while changed:
        changed = False
        for component in components:
            bbox = _bbox_union(tuple(component))
            for rect in candidates:
                if rect not in component and _positive_overlap(bbox, rect):
                    component.add(rect)
                    changed = True
        merged: list[set[Rect]] = []
        while components:
            component = components.pop(0)
            merged_any = False
            for other in merged:
                if _positive_overlap(_bbox_union(tuple(component)), _bbox_union(tuple(other))):
                    other.update(component)
                    merged_any = True
                    changed = True
                    break
            if not merged_any:
                merged.append(component)
        components = merged
    out = []
    old_set = set(old_rects)
    new_set = set(new_rects)
    for component in components:
        old_component = tuple(sorted(rect for rect in component if rect in old_set))
        new_component = tuple(sorted(rect for rect in component if rect in new_set))
        out.append(_RectComponent(layer=layer, old_rects=old_component, new_rects=new_component, bbox=_bbox_union(tuple(component))))
    return sorted(out, key=lambda item: (item.layer, item.bbox))


def _positive_overlap(left: Rect, right: Rect) -> bool:
    return max(left[0], right[0]) < min(left[2], right[2]) and max(left[1], right[1]) < min(left[3], right[3])


def _bbox_union(rects: tuple[Rect, ...]) -> Rect:
    return (
        min(rect[0] for rect in rects),
        min(rect[1] for rect in rects),
        max(rect[2] for rect in rects),
        max(rect[3] for rect in rects),
    )


def _fragment_from_rect(rect: Rect) -> DiffFragment:
    x0, y0, x1, y1 = rect
    points = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    return DiffFragment(points=points, bbox=polygon_bbox(points), twice_area=polygon_twice_area(points))


def _merge_layer_diffs(layer: LayerSpec, diffs: list[LayerExactDiff]) -> LayerExactDiff:
    old_minus = tuple(fragment for diff in diffs for fragment in diff.old_minus_new)
    new_minus = tuple(fragment for diff in diffs for fragment in diff.new_minus_old)
    merged_old_minus = _merge_layer_side_fragments(tuple(sorted(old_minus, key=_fragment_sort_key)))
    merged_new_minus = _merge_layer_side_fragments(tuple(sorted(new_minus, key=_fragment_sort_key)))
    return LayerExactDiff(
        layer=layer,
        old_minus_new=merged_old_minus,
        new_minus_old=merged_new_minus,
        xor=tuple(sorted(merged_old_minus + merged_new_minus, key=_fragment_sort_key)),
    )
