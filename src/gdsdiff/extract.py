from __future__ import annotations

import os
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import LayerSpec, parse_layer_selector
from ._environment import get_environment
from .gds_metadata import decimal_from_number
from .geometry import LayerTable, PolygonBuffer, normalize_polygon_points, polygon_bbox, polygon_flags
from .grid import DEFAULT_COORDINATE_SAFETY_LIMIT, snap_to_grid
from .inventory import select_top_cell
from .multiprocessing_support import fork_context
from .semantics import PathPolicy

_WORKER_RAW_POLYGONS = None
_WORKER_SELECTED_LAYERS = None
_WORKER_GRID = None
_WORKER_UNIT = None
_WORKER_FAST_SCALE = None


@dataclass(frozen=True)
class ExtractionStats:
    raw_polygon_count: int
    kept_polygon_count: int
    vertex_count: int
    max_snap_error_dbu: Decimal

    def to_json(self) -> dict[str, object]:
        return {
            "raw_polygon_count": self.raw_polygon_count,
            "kept_polygon_count": self.kept_polygon_count,
            "vertex_count": self.vertex_count,
            "max_snap_error_dbu": str(self.max_snap_error_dbu),
        }


@dataclass(frozen=True)
class ExtractionResult:
    path: Path
    top_name: str
    buffer: PolygonBuffer
    stats: ExtractionStats


def extract_gds_polygons(
    path: str | Path,
    *,
    top_name: str | None = None,
    layers: str | LayerSpec | Iterable[str | tuple[int, int] | LayerSpec] = "all",
    meters_per_dbu: Decimal | str | float | None = None,
    path_policy: PathPolicy | str = PathPolicy.GDSTK,
) -> ExtractionResult:
    import gdstk

    path = Path(path)
    path_policy = PathPolicy.coerce(path_policy)
    if path_policy == PathPolicy.REPORT_ONLY:
        raise ValueError("report-only path policy cannot produce a passing extraction")
    lib = gdstk.read_gds(str(path))
    top = select_top_cell(lib, top_name=top_name)
    if path_policy == PathPolicy.REJECT and top.get_paths():
        raise ValueError("selected geometry contains paths and path_policy='reject'")
    grid = decimal_from_number(meters_per_dbu) if meters_per_dbu is not None else decimal_from_number(lib.precision)
    unit = decimal_from_number(lib.unit)
    raw_polygons = top.get_polygons(apply_repetitions=True, include_paths=True)
    selected_layers = _selected_layer_set(layers, raw_polygons)
    layer_table = LayerTable(tuple(sorted(selected_layers)))
    layer_index = layer_table.index_by_layer

    records, max_error = _extract_polygon_records(raw_polygons, selected_layers, grid, unit)

    all_vertices: list[tuple[int, int]] = []
    offsets: list[int] = [0]
    layer_ids: list[int] = []
    bboxes: list[tuple[int, int, int, int]] = []
    flags: list[int] = []
    source_ids: list[int] = []

    for source_id, spec, normalized, bbox, polygon_flag in records:
        all_vertices.extend(normalized)
        offsets.append(len(all_vertices))
        layer_ids.append(layer_index[spec])
        bboxes.append(bbox)
        flags.append(polygon_flag)
        source_ids.append(source_id)

    buffer = PolygonBuffer(
        vertices_xy=np.asarray(all_vertices, dtype=np.int64).reshape((-1, 2)),
        polygon_offsets=np.asarray(offsets, dtype=np.uint64),
        layer_ids=np.asarray(layer_ids, dtype=np.uint32),
        bboxes=np.asarray(bboxes, dtype=np.int64).reshape((-1, 4)),
        flags=np.asarray(flags, dtype=np.uint32),
        source_ids=np.asarray(source_ids, dtype=np.uint64),
        layer_table=layer_table,
    )
    return ExtractionResult(
        path=path,
        top_name=top.name,
        buffer=buffer,
        stats=ExtractionStats(
            raw_polygon_count=len(raw_polygons),
            kept_polygon_count=buffer.polygon_count,
            vertex_count=buffer.vertex_count,
            max_snap_error_dbu=max_error,
        ),
    )


def _selected_layer_set(layers, raw_polygons: list[gdstk.Polygon]) -> set[LayerSpec]:
    if layers == "all":
        return {LayerSpec(polygon.layer, polygon.datatype) for polygon in raw_polygons}
    parsed = parse_layer_selector(layers)
    if parsed == "all":
        return {LayerSpec(polygon.layer, polygon.datatype) for polygon in raw_polygons}
    return set(parsed)


def _extract_polygon_records(
    raw_polygons: list[gdstk.Polygon],
    selected_layers: set[LayerSpec],
    grid: Decimal,
    unit: Decimal,
) -> tuple[
    list[tuple[int, LayerSpec, tuple[tuple[int, int], ...], tuple[int, int, int, int], int]],
    Decimal,
]:
    fast_scale = _fast_integer_snap_scale(unit, grid)
    if fast_scale is not None:
        return _extract_polygon_records_fast(raw_polygons, selected_layers, fast_scale)

    worker_count = _resolve_extract_worker_count(len(raw_polygons))
    if worker_count <= 1:
        return _extract_polygon_records_serial(raw_polygons, selected_layers, grid, unit)

    chunk_count = max(worker_count, worker_count * 8)
    chunk_size = max(1, (len(raw_polygons) + chunk_count - 1) // chunk_count)
    chunks = tuple((start, min(start + chunk_size, len(raw_polygons))) for start in range(0, len(raw_polygons), chunk_size))
    context = fork_context()
    if context is None:
        return _extract_polygon_records_serial(raw_polygons, selected_layers, grid, unit)

    global _WORKER_RAW_POLYGONS, _WORKER_SELECTED_LAYERS, _WORKER_GRID, _WORKER_UNIT
    _WORKER_RAW_POLYGONS = raw_polygons
    _WORKER_SELECTED_LAYERS = selected_layers
    _WORKER_GRID = grid
    _WORKER_UNIT = unit
    try:
        try:
            with context.Pool(processes=worker_count) as pool:
                chunk_results = pool.map(_extract_polygon_records_worker, chunks, chunksize=1)
        except OSError:
            return _extract_polygon_records_serial(raw_polygons, selected_layers, grid, unit)
    finally:
        _WORKER_RAW_POLYGONS = None
        _WORKER_SELECTED_LAYERS = None
        _WORKER_GRID = None
        _WORKER_UNIT = None

    records = []
    max_error = Decimal(0)
    for chunk_records, chunk_max_error in chunk_results:
        records.extend(chunk_records)
        max_error = max(max_error, chunk_max_error)
    records.sort(key=lambda record: record[0])
    return records, max_error


def _fast_integer_snap_scale(unit: Decimal, grid: Decimal) -> int | None:
    if get_environment("GDSDIFF_EXTRACT_DECIMAL", "GDS_GPU_COMPARE_EXTRACT_DECIMAL") == "1":
        return None
    try:
        ratio = Fraction(unit) / Fraction(grid)
    except (ArithmeticError, ValueError):
        return None
    if ratio.denominator != 1:
        return None
    scale = ratio.numerator
    if scale <= 0:
        return None
    return scale


def _extract_polygon_records_fast(
    raw_polygons: list[gdstk.Polygon],
    selected_layers: set[LayerSpec],
    scale: int,
) -> tuple[
    list[tuple[int, LayerSpec, tuple[tuple[int, int], ...], tuple[int, int, int, int], int]],
    Decimal,
]:
    worker_count = _resolve_extract_worker_count(len(raw_polygons))
    if worker_count <= 1:
        return _extract_polygon_records_fast_from_range(raw_polygons, selected_layers, scale, 0, len(raw_polygons))

    chunk_count = max(worker_count, worker_count * 8)
    chunk_size = max(1, (len(raw_polygons) + chunk_count - 1) // chunk_count)
    chunks = tuple((start, min(start + chunk_size, len(raw_polygons))) for start in range(0, len(raw_polygons), chunk_size))
    context = fork_context()
    if context is None:
        return _extract_polygon_records_fast_from_range(raw_polygons, selected_layers, scale, 0, len(raw_polygons))

    global _WORKER_RAW_POLYGONS, _WORKER_SELECTED_LAYERS, _WORKER_FAST_SCALE
    _WORKER_RAW_POLYGONS = raw_polygons
    _WORKER_SELECTED_LAYERS = selected_layers
    _WORKER_FAST_SCALE = scale
    try:
        try:
            with context.Pool(processes=worker_count) as pool:
                chunk_results = pool.map(_extract_polygon_records_fast_worker, chunks, chunksize=1)
        except OSError:
            return _extract_polygon_records_fast_from_range(raw_polygons, selected_layers, scale, 0, len(raw_polygons))
    finally:
        _WORKER_RAW_POLYGONS = None
        _WORKER_SELECTED_LAYERS = None
        _WORKER_FAST_SCALE = None

    records = []
    max_error = Decimal(0)
    for chunk_records, chunk_max_error in chunk_results:
        records.extend(chunk_records)
        max_error = max(max_error, chunk_max_error)
    records.sort(key=lambda record: record[0])
    return records, max_error


def _extract_polygon_records_fast_worker(
    chunk: tuple[int, int],
) -> tuple[
    list[tuple[int, LayerSpec, tuple[tuple[int, int], ...], tuple[int, int, int, int], int]],
    Decimal,
]:
    if _WORKER_RAW_POLYGONS is None or _WORKER_SELECTED_LAYERS is None or _WORKER_FAST_SCALE is None:
        raise RuntimeError("fast extraction worker globals are not initialized")
    start, stop = chunk
    return _extract_polygon_records_fast_from_range(
        _WORKER_RAW_POLYGONS,
        _WORKER_SELECTED_LAYERS,
        _WORKER_FAST_SCALE,
        start,
        stop,
    )


def _extract_polygon_records_fast_from_range(
    raw_polygons: list[gdstk.Polygon],
    selected_layers: set[LayerSpec],
    scale: int,
    start: int,
    stop: int,
) -> tuple[
    list[tuple[int, LayerSpec, tuple[tuple[int, int], ...], tuple[int, int, int, int], int]],
    Decimal,
]:
    records = []
    max_error = 0.0
    float_scale = float(scale)
    for source_id in range(start, stop):
        polygon = raw_polygons[source_id]
        spec = LayerSpec(polygon.layer, polygon.datatype)
        if spec not in selected_layers:
            continue
        points = np.asarray(polygon.points, dtype=np.float64)
        if points.size == 0:
            continue
        scaled = points * float_scale
        snapped = _round_half_away_from_zero(scaled)
        if snapped.size and np.max(np.abs(snapped)) > DEFAULT_COORDINATE_SAFETY_LIMIT:
            raise OverflowError("coordinate exceeds configured safety limit")
        max_error = max(max_error, float(np.max(np.abs(scaled - snapped))))
        snapped_points = tuple((int(x), int(y)) for x, y in snapped.astype(np.int64, copy=False))
        normalized = normalize_polygon_points(snapped_points)
        if not normalized:
            continue
        records.append((source_id, spec, normalized, polygon_bbox(normalized), polygon_flags(normalized)))
    return records, Decimal(str(max_error))


def _round_half_away_from_zero(values: np.ndarray) -> np.ndarray:
    return np.copysign(np.floor(np.abs(values) + 0.5), values)


def _resolve_extract_worker_count(raw_polygon_count: int) -> int:
    if raw_polygon_count < _parallel_min_polygons():
        return 1
    return max(1, os.cpu_count() or 1)


def _parallel_min_polygons() -> int:
    try:
        value = int(
            get_environment(
                "GDSDIFF_PARALLEL_MIN_POLYGONS",
                "GDS_GPU_COMPARE_PARALLEL_MIN_POLYGONS",
                "",
            )
            or ""
        )
    except ValueError:
        return 30_000
    return value if value > 0 else 30_000


def _extract_polygon_records_serial(
    raw_polygons: list[gdstk.Polygon],
    selected_layers: set[LayerSpec],
    grid: Decimal,
    unit: Decimal,
) -> tuple[
    list[tuple[int, LayerSpec, tuple[tuple[int, int], ...], tuple[int, int, int, int], int]],
    Decimal,
]:
    return _extract_polygon_records_from_range(raw_polygons, selected_layers, grid, unit, 0, len(raw_polygons))


def _extract_polygon_records_worker(
    chunk: tuple[int, int],
) -> tuple[
    list[tuple[int, LayerSpec, tuple[tuple[int, int], ...], tuple[int, int, int, int], int]],
    Decimal,
]:
    if (
        _WORKER_RAW_POLYGONS is None
        or _WORKER_SELECTED_LAYERS is None
        or _WORKER_GRID is None
        or _WORKER_UNIT is None
    ):
        raise RuntimeError("extraction worker globals are not initialized")
    start, stop = chunk
    return _extract_polygon_records_from_range(
        _WORKER_RAW_POLYGONS,
        _WORKER_SELECTED_LAYERS,
        _WORKER_GRID,
        _WORKER_UNIT,
        start,
        stop,
    )


def _extract_polygon_records_from_range(
    raw_polygons: list[gdstk.Polygon],
    selected_layers: set[LayerSpec],
    grid: Decimal,
    unit: Decimal,
    start: int,
    stop: int,
) -> tuple[
    list[tuple[int, LayerSpec, tuple[tuple[int, int], ...], tuple[int, int, int, int], int]],
    Decimal,
]:
    records = []
    max_error = Decimal(0)
    for source_id in range(start, stop):
        polygon = raw_polygons[source_id]
        spec = LayerSpec(polygon.layer, polygon.datatype)
        if spec not in selected_layers:
            continue
        snapped_points: list[tuple[int, int]] = []
        for point in polygon.points:
            x_physical = decimal_from_number(float(point[0])) * unit
            y_physical = decimal_from_number(float(point[1])) * unit
            x = snap_to_grid(x_physical, grid)
            y = snap_to_grid(y_physical, grid)
            max_error = max(max_error, abs(x_physical / grid - Decimal(x)), abs(y_physical / grid - Decimal(y)))
            snapped_points.append((x, y))
        normalized = normalize_polygon_points(snapped_points)
        if not normalized:
            continue
        records.append((source_id, spec, normalized, polygon_bbox(normalized), polygon_flags(normalized)))
    return records, max_error
