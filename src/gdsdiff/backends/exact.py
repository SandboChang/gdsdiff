from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..exact_oracle import DiffFragment, ExactLayerTask, LayerExactDiff, _exact_layer_diff_task
from ..geometry import polygon_bbox, polygon_twice_area
from .base import BackendCapabilities


class GdstkExactComponentBackend:
    """Reference exact-component executor.

    This backend intentionally uses the existing `gdstk` exact task
    implementation. It is not an accelerator, but it defines the result
    contract that CUDA and Metal exact-component executors must match.
    """

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(name="gdstk-exact-component", deterministic=True, requires_gpu=False)

    def compare_components(self, tasks: tuple[ExactLayerTask, ...]) -> tuple[LayerExactDiff, ...]:
        return tuple(_exact_layer_diff_task(task) for task in tasks)


class UnsupportedExactComponent(RuntimeError):
    """Raised when an exact-component backend cannot prove a task exactly."""


@dataclass(frozen=True)
class RectExactStats:
    accelerated_tasks: int
    fallback_tasks: int
    unsupported_tasks: int
    identical_skipped_tasks: int = 0
    cpu_rect_set_tasks: int = 0
    cuda_rect_set_tasks: int = 0


class ManhattanRectExactComponentBackend:
    """Exact component executor for single axis-aligned rectangle tasks.

    This is a small, deterministic reference for the first accelerator kernel
    target. It handles one old rectangle versus one new rectangle, or a
    rectangle present on only one side. Anything more complex is unsupported
    unless `fallback_to_gdstk=True` is set.
    """

    def __init__(self, *, fallback_to_gdstk: bool = False) -> None:
        self.fallback_to_gdstk = fallback_to_gdstk
        self.stats = RectExactStats(accelerated_tasks=0, fallback_tasks=0, unsupported_tasks=0)

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(name="manhattan-rect-exact-component", deterministic=True, requires_gpu=False)

    def compare_components(self, tasks: tuple[ExactLayerTask, ...]) -> tuple[LayerExactDiff, ...]:
        diffs: list[LayerExactDiff] = []
        accelerated = 0
        fallback = 0
        unsupported = 0
        rect_set = 0
        for task in tasks:
            try:
                diffs.append(_compare_rect_task(task))
                accelerated += 1
            except UnsupportedExactComponent:
                try:
                    diffs.append(_compare_rect_set_task(task))
                    rect_set += 1
                    continue
                except UnsupportedExactComponent:
                    pass
                unsupported += 1
                if not self.fallback_to_gdstk:
                    self.stats = RectExactStats(accelerated, fallback, unsupported, cpu_rect_set_tasks=rect_set)
                    raise
                diffs.append(_exact_layer_diff_task(task))
                fallback += 1
        self.stats = RectExactStats(accelerated, fallback, unsupported, cpu_rect_set_tasks=rect_set)
        return tuple(diffs)


class CudaManhattanRectExactComponentBackend:
    """CUDA-assisted exact component executor for single rectangle tasks.

    CUDA emits the directional rectangle difference pieces for supported
    components. Python converts those piece rectangles into `DiffFragment`
    objects and accepts them only when their totals match the CUDA-reported
    areas.
    """

    def __init__(self, *, fallback_to_gdstk: bool = False) -> None:
        self.fallback_to_gdstk = fallback_to_gdstk
        self.stats = RectExactStats(accelerated_tasks=0, fallback_tasks=0, unsupported_tasks=0)

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(name="cuda-manhattan-rect-exact-component", deterministic=True, requires_gpu=True)

    def compare_components(self, tasks: tuple[ExactLayerTask, ...]) -> tuple[LayerExactDiff, ...]:
        from .cuda_ctypes import run_cuda_rect_set_exact_fragments

        diffs: list[LayerExactDiff | None] = [None] * len(tasks)
        cuda_set_items: list[tuple[int, ExactLayerTask, tuple[tuple[int, int, int, int], ...], tuple[tuple[int, int, int, int], ...]]] = []
        fallback = 0
        unsupported = 0
        skipped = 0
        rect_set = 0
        cuda_rect_set = 0
        for index, task in enumerate(tasks):
            try:
                old_rect, new_rect = _task_rect_pair(task)
            except UnsupportedExactComponent:
                rect_sets = _task_rect_sets(task)
                if rect_sets is not None:
                    cuda_set_items.append((index, task, rect_sets[0], rect_sets[1]))
                    continue
                try:
                    diffs[index] = _compare_rect_set_task(task)
                    rect_set += 1
                    continue
                except UnsupportedExactComponent:
                    pass
                unsupported += 1
                if not self.fallback_to_gdstk:
                    self.stats = RectExactStats(0, fallback, unsupported, skipped, rect_set, cuda_rect_set)
                    raise
                diffs[index] = _exact_layer_diff_task(task)
                fallback += 1
                continue
            if old_rect is not None and old_rect == new_rect:
                diffs[index] = LayerExactDiff(layer=task.layer)
                skipped += 1
                continue
            old_rects = () if old_rect is None else (old_rect,)
            new_rects = () if new_rect is None else (new_rect,)
            cuda_set_items.append((index, task, old_rects, new_rects))
        if cuda_set_items:
            max_rects = max(max(len(old_rects), len(new_rects)) for _index, _task, old_rects, new_rects in cuda_set_items)
            if max_rects > 16:
                for index, task, _old_rects, _new_rects in cuda_set_items:
                    try:
                        diffs[index] = _compare_rect_set_task(task)
                        rect_set += 1
                    except UnsupportedExactComponent:
                        unsupported += 1
                        if not self.fallback_to_gdstk:
                            self.stats = RectExactStats(0, fallback, unsupported, skipped, rect_set, cuda_rect_set)
                            raise
                        diffs[index] = _exact_layer_diff_task(task)
                        fallback += 1
            else:
                old_rects_array = np.zeros((len(cuda_set_items), max_rects, 4), dtype=np.int64)
                new_rects_array = np.zeros((len(cuda_set_items), max_rects, 4), dtype=np.int64)
                old_counts = np.zeros((len(cuda_set_items),), dtype=np.uint8)
                new_counts = np.zeros((len(cuda_set_items),), dtype=np.uint8)
                for row, (_index, _task, old_rects, new_rects) in enumerate(cuda_set_items):
                    old_counts[row] = len(old_rects)
                    new_counts[row] = len(new_rects)
                    for col, rect in enumerate(old_rects):
                        old_rects_array[row, col] = rect
                    for col, rect in enumerate(new_rects):
                        new_rects_array[row, col] = rect
                try:
                    cuda_rects, cuda_counts, cuda_areas = run_cuda_rect_set_exact_fragments(
                        old_rects_array,
                        new_rects_array,
                        old_counts,
                        new_counts,
                    )
                except Exception:
                    cuda_rects = cuda_counts = cuda_areas = None
                if cuda_rects is None or cuda_counts is None or cuda_areas is None:
                    for index, task, _old_rects, _new_rects in cuda_set_items:
                        try:
                            diffs[index] = _compare_rect_set_task(task)
                            rect_set += 1
                        except UnsupportedExactComponent:
                            unsupported += 1
                            if not self.fallback_to_gdstk:
                                self.stats = RectExactStats(0, fallback, unsupported, skipped, rect_set, cuda_rect_set)
                                raise
                            diffs[index] = _exact_layer_diff_task(task)
                            fallback += 1
                else:
                    for row, (index, task, _old_rects, _new_rects) in enumerate(cuda_set_items):
                        old_count = int(cuda_counts[row, 0])
                        new_count = int(cuda_counts[row, 1])
                        old_fragments = tuple(
                            _fragment_from_rect(tuple(int(v) for v in cuda_rects[row, 0, piece]))
                            for piece in range(old_count)
                        )
                        new_fragments = tuple(
                            _fragment_from_rect(tuple(int(v) for v in cuda_rects[row, 1, piece]))
                            for piece in range(new_count)
                        )
                        diff = LayerExactDiff(
                            layer=task.layer,
                            old_minus_new=old_fragments,
                            new_minus_old=new_fragments,
                            xor=old_fragments + new_fragments,
                        )
                        expected = np.asarray(
                            [
                                diff.old_minus_new_twice_area,
                                diff.new_minus_old_twice_area,
                                diff.xor_twice_area,
                            ],
                            dtype=np.int64,
                        )
                        if not np.array_equal(cuda_areas[row], expected):
                            raise UnsupportedExactComponent(
                                f"CUDA rectangle set exact area mismatch for task {index}: "
                                f"cuda={cuda_areas[row].tolist()} expected={expected.tolist()}"
                            )
                        diffs[index] = diff
                        cuda_rect_set += 1
        self.stats = RectExactStats(0, fallback, unsupported, skipped, rect_set, cuda_rect_set)
        return tuple(diff for diff in diffs if diff is not None)


def _compare_rect_task(task: ExactLayerTask) -> LayerExactDiff:
    if task.clip_bbox is not None:
        raise UnsupportedExactComponent("rectangle exact backend does not support clipped stripe tasks")
    old_rects = tuple(_as_rect(points) for points in (task.old_points or ()))
    new_rects = tuple(_as_rect(points) for points in (task.new_points or ()))
    if any(rect is None for rect in old_rects + new_rects):
        raise UnsupportedExactComponent("component contains non-rectangle polygon")
    if len(old_rects) > 1 or len(new_rects) > 1:
        raise UnsupportedExactComponent("component contains multiple polygons on one side")
    old_rect = old_rects[0] if old_rects else None
    new_rect = new_rects[0] if new_rects else None
    if old_rect is None and new_rect is None:
        return LayerExactDiff(layer=task.layer)
    old_minus = _rect_minus_rect(old_rect, new_rect) if old_rect is not None else ()
    new_minus = _rect_minus_rect(new_rect, old_rect) if new_rect is not None else ()
    old_fragments = tuple(_fragment_from_rect(rect) for rect in old_minus)
    new_fragments = tuple(_fragment_from_rect(rect) for rect in new_minus)
    return LayerExactDiff(
        layer=task.layer,
        old_minus_new=old_fragments,
        new_minus_old=new_fragments,
        xor=old_fragments + new_fragments,
    )


def _compare_rect_set_task(task: ExactLayerTask) -> LayerExactDiff:
    if task.clip_bbox is not None:
        raise UnsupportedExactComponent("rectangle set exact backend does not support clipped stripe tasks")
    old_rects = tuple(_as_rect(points) for points in (task.old_points or ()))
    new_rects = tuple(_as_rect(points) for points in (task.new_points or ()))
    if any(rect is None for rect in old_rects + new_rects):
        raise UnsupportedExactComponent("rectangle set contains non-rectangle polygon")
    old_rect_tuple = tuple(rect for rect in old_rects if rect is not None)
    new_rect_tuple = tuple(rect for rect in new_rects if rect is not None)
    if len(old_rect_tuple) <= 1 and len(new_rect_tuple) <= 1:
        raise UnsupportedExactComponent("single rectangle task is handled by the rectangle pair path")
    old_minus = _rect_set_minus_rect_set(old_rect_tuple, new_rect_tuple)
    new_minus = _rect_set_minus_rect_set(new_rect_tuple, old_rect_tuple)
    old_fragments = tuple(_fragment_from_rect(rect) for rect in old_minus)
    new_fragments = tuple(_fragment_from_rect(rect) for rect in new_minus)
    return LayerExactDiff(
        layer=task.layer,
        old_minus_new=old_fragments,
        new_minus_old=new_fragments,
        xor=old_fragments + new_fragments,
    )


def _task_rect_pair(task: ExactLayerTask) -> tuple[tuple[int, int, int, int] | None, tuple[int, int, int, int] | None]:
    if task.clip_bbox is not None:
        raise UnsupportedExactComponent("rectangle exact backend does not support clipped stripe tasks")
    old_rects = tuple(_as_rect(points) for points in (task.old_points or ()))
    new_rects = tuple(_as_rect(points) for points in (task.new_points or ()))
    if any(rect is None for rect in old_rects + new_rects):
        raise UnsupportedExactComponent("component contains non-rectangle polygon")
    if len(old_rects) > 1 or len(new_rects) > 1:
        raise UnsupportedExactComponent("component contains multiple polygons on one side")
    old_rect = old_rects[0] if old_rects else None
    new_rect = new_rects[0] if new_rects else None
    if old_rect is None and new_rect is None:
        return None, None
    return old_rect, new_rect


def _task_rect_sets(
    task: ExactLayerTask,
) -> tuple[tuple[tuple[int, int, int, int], ...], tuple[tuple[int, int, int, int], ...]] | None:
    if task.clip_bbox is not None:
        return None
    old_rects = tuple(_as_rect(points) for points in (task.old_points or ()))
    new_rects = tuple(_as_rect(points) for points in (task.new_points or ()))
    if any(rect is None for rect in old_rects + new_rects):
        return None
    old_rect_tuple = tuple(rect for rect in old_rects if rect is not None)
    new_rect_tuple = tuple(rect for rect in new_rects if rect is not None)
    if len(old_rect_tuple) <= 1 and len(new_rect_tuple) <= 1:
        return None
    return old_rect_tuple, new_rect_tuple


def _as_rect(points: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int] | None:
    if len(points) != 4:
        return None
    xs = sorted({int(x) for x, _ in points})
    ys = sorted({int(y) for _, y in points})
    if len(xs) != 2 or len(ys) != 2:
        return None
    expected = {(xs[0], ys[0]), (xs[0], ys[1]), (xs[1], ys[0]), (xs[1], ys[1])}
    if set(points) != expected:
        return None
    if xs[0] == xs[1] or ys[0] == ys[1]:
        return None
    return xs[0], ys[0], xs[1], ys[1]


def _rect_minus_rect(
    rect: tuple[int, int, int, int],
    cutter: tuple[int, int, int, int] | None,
) -> tuple[tuple[int, int, int, int], ...]:
    if cutter is None:
        return (rect,)
    ax0, ay0, ax1, ay1 = rect
    bx0, by0, bx1, by1 = cutter
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix0 >= ix1 or iy0 >= iy1:
        return (rect,)
    pieces = (
        (ax0, ay0, ix0, ay1),
        (ix1, ay0, ax1, ay1),
        (ix0, ay0, ix1, iy0),
        (ix0, iy1, ix1, ay1),
    )
    return tuple(piece for piece in pieces if _rect_area(piece) > 0)


def _rect_area(rect: tuple[int, int, int, int]) -> int:
    x0, y0, x1, y1 = rect
    return max(0, x1 - x0) * max(0, y1 - y0)


def _rect_set_minus_rect_set(
    source: tuple[tuple[int, int, int, int], ...],
    cutters: tuple[tuple[int, int, int, int], ...],
) -> tuple[tuple[int, int, int, int], ...]:
    if not source:
        return ()
    xs = sorted({x for rect in source + cutters for x in (rect[0], rect[2])})
    ys = sorted({y for rect in source + cutters for y in (rect[1], rect[3])})
    if len(xs) < 2 or len(ys) < 2:
        return ()
    cells = []
    for x0, x1 in zip(xs, xs[1:]):
        if x0 == x1:
            continue
        for y0, y1 in zip(ys, ys[1:]):
            if y0 == y1:
                continue
            inside_source = any(_rect_contains_cell(rect, x0, y0, x1, y1) for rect in source)
            if not inside_source:
                continue
            inside_cutter = any(_rect_contains_cell(rect, x0, y0, x1, y1) for rect in cutters)
            if not inside_cutter:
                cells.append((x0, y0, x1, y1))
    return _merge_rect_cells(tuple(cells))


def _rect_contains_cell(rect: tuple[int, int, int, int], x0: int, y0: int, x1: int, y1: int) -> bool:
    return rect[0] <= x0 and x1 <= rect[2] and rect[1] <= y0 and y1 <= rect[3]


def _merge_rect_cells(cells: tuple[tuple[int, int, int, int], ...]) -> tuple[tuple[int, int, int, int], ...]:
    if not cells:
        return ()
    current = sorted(cells)
    changed = True
    while changed:
        changed = False
        used = [False] * len(current)
        merged: list[tuple[int, int, int, int]] = []
        for i, left in enumerate(current):
            if used[i]:
                continue
            candidate = left
            for j in range(i + 1, len(current)):
                if used[j]:
                    continue
                combined = _try_merge_rect_pair(candidate, current[j])
                if combined is not None:
                    candidate = combined
                    used[j] = True
                    changed = True
            used[i] = True
            merged.append(candidate)
        current = sorted(merged)
    return tuple(current)


def _try_merge_rect_pair(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    if left[1] == right[1] and left[3] == right[3] and (left[2] == right[0] or right[2] == left[0]):
        return min(left[0], right[0]), left[1], max(left[2], right[2]), left[3]
    if left[0] == right[0] and left[2] == right[2] and (left[3] == right[1] or right[3] == left[1]):
        return left[0], min(left[1], right[1]), left[2], max(left[3], right[3])
    return None


def _fragment_from_rect(rect: tuple[int, int, int, int]) -> DiffFragment:
    x0, y0, x1, y1 = rect
    points = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    return DiffFragment(points=points, bbox=polygon_bbox(points), twice_area=polygon_twice_area(points))
