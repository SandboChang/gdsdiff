from __future__ import annotations

import platform

import numpy as np

from ..exact_oracle import ExactLayerTask, LayerExactDiff
from .base import BackendCapabilities
from .exact import (
    RectExactStats,
    UnsupportedExactComponent,
    _compare_rect_or_rect_set_task,
    _fragment_from_rect,
    _task_rect_pair,
    _task_rect_sets,
)
from ..exact_oracle import _exact_layer_diff_task


class MetalExactUnavailable(RuntimeError):
    pass


class MetalManhattanRectExactComponentBackend:
    """Metal exact-component backend for bounded Manhattan rectangle tasks.

    The comparator's exact-component contract is intentionally shared by CUDA
    and Metal. This class dispatches compatible rectangle-pair and rectangle-set
    tasks to the Apple Silicon Metal ctypes runtime, with optional gdstk fallback
    for unsupported or unavailable cases.
    """

    def __init__(self, *, fallback_to_gdstk: bool = False) -> None:
        self.fallback_to_gdstk = fallback_to_gdstk

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(name="metal-manhattan-rect-exact-component", deterministic=True, requires_gpu=True)

    def compare_components(self, tasks: tuple[ExactLayerTask, ...]) -> tuple[LayerExactDiff, ...]:
        if platform.system() != "Darwin" or platform.machine() not in {"arm64", "aarch64"}:
            raise MetalExactUnavailable("Metal exact backend requires macOS on Apple Silicon")
        from .metal_ctypes import MetalCtypesUnavailable, run_metal_rect_set_exact_fragments

        diffs: list[LayerExactDiff | None] = [None] * len(tasks)
        metal_set_items: list[tuple[int, ExactLayerTask, tuple[tuple[int, int, int, int], ...], tuple[tuple[int, int, int, int], ...]]] = []
        fallback = 0
        unsupported = 0
        skipped = 0
        rect_set = 0
        metal_rect_set = 0
        for index, task in enumerate(tasks):
            try:
                old_rect, new_rect = _task_rect_pair(task)
            except UnsupportedExactComponent:
                rect_sets = _task_rect_sets(task)
                if rect_sets is not None:
                    metal_set_items.append((index, task, rect_sets[0], rect_sets[1]))
                    continue
                try:
                    diffs[index] = _compare_rect_or_rect_set_task(task)
                    rect_set += 1
                    continue
                except UnsupportedExactComponent:
                    pass
                unsupported += 1
                if not self.fallback_to_gdstk:
                    self.stats = RectExactStats(
                        0,
                        fallback,
                        unsupported,
                        skipped,
                        cpu_rect_set_tasks=rect_set,
                        metal_rect_set_tasks=metal_rect_set,
                    )
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
            metal_set_items.append((index, task, old_rects, new_rects))
        if metal_set_items:
            max_rects = max(max(len(old_rects), len(new_rects)) for _index, _task, old_rects, new_rects in metal_set_items)
            if max_rects > 16:
                for index, task, _old_rects, _new_rects in metal_set_items:
                    try:
                        diffs[index] = _compare_rect_or_rect_set_task(task)
                        rect_set += 1
                    except UnsupportedExactComponent:
                        unsupported += 1
                        if not self.fallback_to_gdstk:
                            self.stats = RectExactStats(
                                0,
                                fallback,
                                unsupported,
                                skipped,
                                cpu_rect_set_tasks=rect_set,
                                metal_rect_set_tasks=metal_rect_set,
                            )
                            raise
                        diffs[index] = _exact_layer_diff_task(task)
                        fallback += 1
            else:
                old_rects_array = np.zeros((len(metal_set_items), max_rects, 4), dtype=np.int64)
                new_rects_array = np.zeros((len(metal_set_items), max_rects, 4), dtype=np.int64)
                old_counts = np.zeros((len(metal_set_items),), dtype=np.uint8)
                new_counts = np.zeros((len(metal_set_items),), dtype=np.uint8)
                for row, (_index, _task, old_rects, new_rects) in enumerate(metal_set_items):
                    old_counts[row] = len(old_rects)
                    new_counts[row] = len(new_rects)
                    for col, rect in enumerate(old_rects):
                        old_rects_array[row, col] = rect
                    for col, rect in enumerate(new_rects):
                        new_rects_array[row, col] = rect
                try:
                    metal_rects, metal_counts, metal_areas = run_metal_rect_set_exact_fragments(
                        old_rects_array,
                        new_rects_array,
                        old_counts,
                        new_counts,
                    )
                except MetalCtypesUnavailable as exc:
                    if not self.fallback_to_gdstk:
                        self.stats = RectExactStats(
                            0,
                            fallback,
                            unsupported,
                            skipped,
                            cpu_rect_set_tasks=rect_set,
                            metal_rect_set_tasks=metal_rect_set,
                        )
                        raise MetalExactUnavailable(str(exc)) from exc
                    metal_rects = metal_counts = metal_areas = None
                if metal_rects is None or metal_counts is None or metal_areas is None:
                    for index, task, _old_rects, _new_rects in metal_set_items:
                        try:
                            diffs[index] = _compare_rect_or_rect_set_task(task)
                            rect_set += 1
                        except UnsupportedExactComponent:
                            unsupported += 1
                            if not self.fallback_to_gdstk:
                                self.stats = RectExactStats(
                                    0,
                                    fallback,
                                    unsupported,
                                    skipped,
                                    cpu_rect_set_tasks=rect_set,
                                    metal_rect_set_tasks=metal_rect_set,
                                )
                                raise
                            diffs[index] = _exact_layer_diff_task(task)
                            fallback += 1
                else:
                    for row, (index, task, _old_rects, _new_rects) in enumerate(metal_set_items):
                        old_count = int(metal_counts[row, 0])
                        new_count = int(metal_counts[row, 1])
                        old_fragments = tuple(
                            _fragment_from_rect(tuple(int(v) for v in metal_rects[row, 0, piece]))
                            for piece in range(old_count)
                        )
                        new_fragments = tuple(
                            _fragment_from_rect(tuple(int(v) for v in metal_rects[row, 1, piece]))
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
                        if not np.array_equal(metal_areas[row], expected):
                            raise UnsupportedExactComponent(
                                f"Metal rectangle set exact area mismatch for task {index}: "
                                f"metal={metal_areas[row].tolist()} expected={expected.tolist()}"
                            )
                        diffs[index] = diff
                        metal_rect_set += 1
        self.stats = RectExactStats(
            0,
            fallback,
            unsupported,
            skipped,
            cpu_rect_set_tasks=rect_set,
            metal_rect_set_tasks=metal_rect_set,
        )
        return tuple(diff for diff in diffs if diff is not None)
