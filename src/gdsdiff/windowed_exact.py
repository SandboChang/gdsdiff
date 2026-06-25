from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass, field

import gdstk

from .config import LayerSpec
from .exact_oracle import DiffFragment, ExactOracleResult, LayerExactDiff, run_full_exact_oracle
from .geometry import PolygonBuffer, polygon_bbox, polygon_twice_area
from .grid import snap_to_grid
from .spatial import BBox, SpatialIndex
from .windows import CandidateWindow

_WORKER_OLD_BUFFER = None
_WORKER_NEW_BUFFER = None
_WORKER_PRECISION = 1.0


@dataclass(frozen=True)
class WindowedExactFailure:
    window: CandidateWindow
    reason: str

    def to_json(self) -> dict[str, object]:
        return {"window": self.window.to_json(), "reason": self.reason}


@dataclass(frozen=True)
class WindowedExactResult:
    layers: tuple[LayerExactDiff, ...]
    escalated_layers: tuple[LayerSpec, ...] = field(default_factory=tuple)
    failures: tuple[WindowedExactFailure, ...] = field(default_factory=tuple)

    @property
    def equivalent(self) -> bool:
        return all(layer.xor_twice_area == 0 for layer in self.layers)

    @property
    def exact_result(self) -> ExactOracleResult:
        return ExactOracleResult(self.layers)


def run_windowed_exact_oracle(
    old_buffer: PolygonBuffer,
    new_buffer: PolygonBuffer,
    windows: tuple[CandidateWindow, ...] | list[CandidateWindow],
    *,
    old_index: SpatialIndex | None = None,
    new_index: SpatialIndex | None = None,
    precision: float = 1.0,
    max_polygons_per_window: int = 200_000,
) -> WindowedExactResult:
    old_index = old_index or SpatialIndex.from_buffer(old_buffer)
    new_index = new_index or SpatialIndex.from_buffer(new_buffer)

    layers_from_windows = tuple(sorted({window.layer for window in windows}))
    direction_points: dict[LayerSpec, dict[str, list[tuple[tuple[int, int], ...]]]] = {
        layer: {"old_minus_new": [], "new_minus_old": []} for layer in layers_from_windows
    }
    escalated_layers: set[LayerSpec] = set()
    failures: list[WindowedExactFailure] = []
    window_tasks = []

    for window in windows:
        old_indices = old_index.query(window.layer, window.verification_bbox)
        new_indices = new_index.query(window.layer, window.verification_bbox)
        window_tasks.append((window, old_indices, new_indices, max_polygons_per_window))

    task_results = _run_window_tasks(old_buffer, new_buffer, window_tasks, precision=precision)
    for window, old_minus, new_minus, failure_reason in task_results:
        if failure_reason is not None:
            escalated_layers.add(window.layer)
            failures.append(WindowedExactFailure(window, failure_reason))

    for window, old_minus, new_minus, failure_reason in task_results:
        if failure_reason is not None or window.layer in escalated_layers:
            continue
        direction_points[window.layer]["old_minus_new"].extend(old_minus)
        direction_points[window.layer]["new_minus_old"].extend(new_minus)

    full_layer_diffs = _full_diffs_by_layer(old_buffer, new_buffer, precision=precision) if escalated_layers else {}
    layer_diffs: list[LayerExactDiff] = []
    for layer in layers_from_windows:
        if layer in escalated_layers:
            layer_diffs.append(
                full_layer_diffs.get(
                    layer,
                    LayerExactDiff(layer=layer),
                )
            )
            continue
        old_minus = _union_fragments(direction_points[layer]["old_minus_new"], layer, precision=precision)
        new_minus = _union_fragments(direction_points[layer]["new_minus_old"], layer, precision=precision)
        xor = _union_fragments([fragment.points for fragment in old_minus] + [fragment.points for fragment in new_minus], layer, precision=precision)
        layer_diffs.append(
            LayerExactDiff(
                layer=layer,
                old_minus_new=old_minus,
                new_minus_old=new_minus,
                xor=xor,
            )
        )

    return WindowedExactResult(
        layers=tuple(sorted(layer_diffs, key=lambda diff: diff.layer)),
        escalated_layers=tuple(sorted(escalated_layers)),
        failures=tuple(failures),
    )


def _run_window_tasks(
    old_buffer: PolygonBuffer,
    new_buffer: PolygonBuffer,
    window_tasks,
    *,
    precision: float,
):
    if not window_tasks:
        return ()
    worker_count = _resolve_window_worker_count(len(window_tasks))
    if worker_count <= 1:
        return tuple(_compute_window_task((old_buffer, new_buffer, precision, *task)) for task in window_tasks)

    global _WORKER_OLD_BUFFER, _WORKER_NEW_BUFFER, _WORKER_PRECISION
    _WORKER_OLD_BUFFER = old_buffer
    _WORKER_NEW_BUFFER = new_buffer
    _WORKER_PRECISION = precision
    try:
        with mp.get_context("fork").Pool(processes=worker_count) as pool:
            return tuple(pool.map(_compute_window_task_worker, window_tasks, chunksize=1))
    finally:
        _WORKER_OLD_BUFFER = None
        _WORKER_NEW_BUFFER = None


def _resolve_window_worker_count(window_count: int) -> int:
    if window_count <= 1:
        return 1
    return max(1, min(os.cpu_count() or 1, window_count))


def _compute_window_task_worker(task):
    if _WORKER_OLD_BUFFER is None or _WORKER_NEW_BUFFER is None:
        raise RuntimeError("windowed exact worker globals are not initialized")
    return _compute_window_task((_WORKER_OLD_BUFFER, _WORKER_NEW_BUFFER, _WORKER_PRECISION, *task))


def _compute_window_task(task):
    old_buffer, new_buffer, precision, window, old_indices, new_indices, max_polygons_per_window = task
    if len(old_indices) + len(new_indices) > max_polygons_per_window:
        return window, (), (), "window polygon threshold exceeded"
    try:
        old_minus, new_minus = _compute_window_directional_fragments(
            old_buffer,
            old_indices,
            new_buffer,
            new_indices,
            window,
            precision=precision,
        )
    except (RuntimeError, ValueError, TypeError) as exc:
        return window, (), (), f"gdstk boolean failed: {exc}"
    return window, old_minus, new_minus, None


def _compute_window_directional_fragments(
    old_buffer: PolygonBuffer,
    old_indices: tuple[int, ...],
    new_buffer: PolygonBuffer,
    new_indices: tuple[int, ...],
    window: CandidateWindow,
    *,
    precision: float,
) -> tuple[tuple[tuple[tuple[int, int], ...], ...], tuple[tuple[tuple[int, int], ...], ...]]:
    origin = window.verification_bbox[0], window.verification_bbox[1]
    old_polys = _buffer_indices_to_gdstk(old_buffer, old_indices, window.layer, origin)
    new_polys = _buffer_indices_to_gdstk(new_buffer, new_indices, window.layer, origin)
    verification_rect = _bbox_rectangle(window.verification_bbox, window.layer, origin)
    core_rect = _bbox_rectangle(window.core_bbox, window.layer, origin)

    old_clipped = gdstk.boolean(old_polys, [verification_rect], "and", precision=precision) or []
    new_clipped = gdstk.boolean(new_polys, [verification_rect], "and", precision=precision) or []
    old_minus = gdstk.boolean(old_clipped, new_clipped, "not", precision=precision) or []
    new_minus = gdstk.boolean(new_clipped, old_clipped, "not", precision=precision) or []
    old_core = gdstk.boolean(old_minus, [core_rect], "and", precision=precision) or []
    new_core = gdstk.boolean(new_minus, [core_rect], "and", precision=precision) or []
    return _global_points_from_gdstk(old_core, origin), _global_points_from_gdstk(new_core, origin)


def _buffer_indices_to_gdstk(
    buffer: PolygonBuffer,
    indices: tuple[int, ...],
    layer: LayerSpec,
    origin: tuple[int, int],
) -> list[gdstk.Polygon]:
    ox, oy = origin
    polygons = []
    for index in indices:
        points = tuple((int(x) - ox, int(y) - oy) for x, y in buffer.polygon_vertices(index))
        if len(points) >= 3:
            polygons.append(gdstk.Polygon(points, layer=layer.layer, datatype=layer.datatype))
    return polygons


def _bbox_rectangle(bbox: BBox, layer: LayerSpec, origin: tuple[int, int]) -> gdstk.Polygon:
    ox, oy = origin
    x0, y0, x1, y1 = bbox
    return gdstk.rectangle(
        (x0 - ox, y0 - oy),
        (x1 + 1 - ox, y1 + 1 - oy),
        layer=layer.layer,
        datatype=layer.datatype,
    )


def _global_points_from_gdstk(polygons: list[gdstk.Polygon], origin: tuple[int, int]) -> tuple[tuple[tuple[int, int], ...], ...]:
    ox, oy = origin
    result = []
    for polygon in polygons:
        points = tuple((snap_to_grid(point[0], 1) + ox, snap_to_grid(point[1], 1) + oy) for point in polygon.points)
        if len(points) >= 3:
            result.append(points)
    return tuple(result)


def _union_fragments(
    polygons: list[tuple[tuple[int, int], ...]],
    layer: LayerSpec,
    *,
    precision: float,
) -> tuple[DiffFragment, ...]:
    if not polygons:
        return ()
    origin = _points_origin(polygons)
    ox, oy = origin
    local = [gdstk.Polygon([(x - ox, y - oy) for x, y in points], layer=layer.layer, datatype=layer.datatype) for points in polygons]
    merged = gdstk.boolean(local, [], "or", precision=precision) or []
    return _fragments_from_gdstk(merged, origin)


def _fragments_from_gdstk(polygons: list[gdstk.Polygon], origin: tuple[int, int]) -> tuple[DiffFragment, ...]:
    ox, oy = origin
    fragments = []
    for polygon in polygons:
        points = tuple((snap_to_grid(point[0], 1) + ox, snap_to_grid(point[1], 1) + oy) for point in polygon.points)
        if len(points) < 3:
            continue
        fragments.append(DiffFragment(points=points, bbox=polygon_bbox(points), twice_area=polygon_twice_area(points)))
    return tuple(sorted(fragments, key=lambda fragment: (fragment.bbox, fragment.points)))


def _points_origin(polygons: list[tuple[tuple[int, int], ...]]) -> tuple[int, int]:
    return min(point[0] for polygon in polygons for point in polygon), min(point[1] for polygon in polygons for point in polygon)


def _full_diffs_by_layer(
    old_buffer: PolygonBuffer,
    new_buffer: PolygonBuffer,
    *,
    precision: float,
) -> dict[LayerSpec, LayerExactDiff]:
    return {layer_diff.layer: layer_diff for layer_diff in run_full_exact_oracle(old_buffer, new_buffer, precision=precision).layers}
