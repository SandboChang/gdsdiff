from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Iterable

import gdstk
import numpy as np

from .config import LayerSpec, parse_layer_selector
from .gds_metadata import decimal_from_number
from .geometry import LayerTable, PolygonBuffer, normalize_polygon_points, polygon_bbox, polygon_flags
from .grid import snap_to_grid
from .inventory import select_top_cell
from .spatial import BBox, bboxes_intersect


@dataclass(frozen=True)
class WindowedHierarchyStats:
    visited_cell_count: int
    considered_reference_instance_count: int
    pruned_reference_instance_count: int
    emitted_polygon_count: int
    raw_local_polygon_count: int
    max_snap_error_dbu: Decimal

    def to_json(self) -> dict[str, object]:
        return {
            "visited_cell_count": self.visited_cell_count,
            "considered_reference_instance_count": self.considered_reference_instance_count,
            "pruned_reference_instance_count": self.pruned_reference_instance_count,
            "emitted_polygon_count": self.emitted_polygon_count,
            "raw_local_polygon_count": self.raw_local_polygon_count,
            "max_snap_error_dbu": str(self.max_snap_error_dbu),
        }


@dataclass(frozen=True)
class WindowedHierarchyExtractionResult:
    top_name: str
    window_bbox: BBox
    buffer: PolygonBuffer
    stats: WindowedHierarchyStats


@dataclass(frozen=True)
class _Transform:
    a: float = 1.0
    b: float = 0.0
    c: float = 0.0
    d: float = 1.0
    tx: float = 0.0
    ty: float = 0.0

    def apply(self, x: float, y: float) -> tuple[float, float]:
        return self.a * x + self.b * y + self.tx, self.c * x + self.d * y + self.ty

    def compose(self, child: "_Transform") -> "_Transform":
        return _Transform(
            a=self.a * child.a + self.b * child.c,
            b=self.a * child.b + self.b * child.d,
            c=self.c * child.a + self.d * child.c,
            d=self.c * child.b + self.d * child.d,
            tx=self.a * child.tx + self.b * child.ty + self.tx,
            ty=self.c * child.tx + self.d * child.ty + self.ty,
        )


def extract_hierarchy_window_from_gds(
    path: str | Path,
    *,
    window_bbox: BBox,
    top_name: str | None = None,
    layers: str | LayerSpec | Iterable[str | tuple[int, int] | LayerSpec] = "all",
    meters_per_dbu: Decimal | str | float | None = None,
) -> WindowedHierarchyExtractionResult:
    lib = gdstk.read_gds(str(path))
    grid = decimal_from_number(meters_per_dbu) if meters_per_dbu is not None else decimal_from_number(lib.precision)
    return extract_hierarchy_window(lib, window_bbox=window_bbox, top_name=top_name, layers=layers, meters_per_dbu=grid)


def extract_hierarchy_window(
    lib: gdstk.Library,
    *,
    window_bbox: BBox,
    top_name: str | None = None,
    layers: str | LayerSpec | Iterable[str | tuple[int, int] | LayerSpec] = "all",
    meters_per_dbu: Decimal | str | float | None = None,
) -> WindowedHierarchyExtractionResult:
    top = select_top_cell(lib, top_name=top_name)
    grid = decimal_from_number(meters_per_dbu) if meters_per_dbu is not None else decimal_from_number(lib.precision)
    unit = decimal_from_number(lib.unit)
    selected_layers = _selected_layers_or_none(layers)
    extractor = _WindowExtractor(unit=unit, grid=grid, selected_layers=selected_layers, window_bbox=window_bbox)
    extractor.traverse_cell(top, _Transform(), set())
    return WindowedHierarchyExtractionResult(
        top_name=top.name,
        window_bbox=window_bbox,
        buffer=extractor.to_buffer(),
        stats=WindowedHierarchyStats(
            visited_cell_count=extractor.visited_cell_count,
            considered_reference_instance_count=extractor.considered_reference_instance_count,
            pruned_reference_instance_count=extractor.pruned_reference_instance_count,
            emitted_polygon_count=len(extractor.polygons),
            raw_local_polygon_count=extractor.raw_local_polygon_count,
            max_snap_error_dbu=extractor.max_snap_error,
        ),
    )


class _WindowExtractor:
    def __init__(self, *, unit: Decimal, grid: Decimal, selected_layers: set[LayerSpec] | None, window_bbox: BBox) -> None:
        self.unit = unit
        self.grid = grid
        self.selected_layers = selected_layers
        self.window_bbox = window_bbox
        self.polygons: list[tuple[LayerSpec, tuple[tuple[int, int], ...]]] = []
        self.max_snap_error = Decimal(0)
        self.visited_cell_count = 0
        self.considered_reference_instance_count = 0
        self.pruned_reference_instance_count = 0
        self.raw_local_polygon_count = 0

    def traverse_cell(self, cell: gdstk.Cell, transform: _Transform, visiting: set[int]) -> None:
        key = id(cell)
        if key in visiting:
            raise ValueError(f"hierarchy cycle detected at cell {cell.name!r}")
        visiting.add(key)
        self.visited_cell_count += 1
        for polygon in cell.get_polygons(depth=0, include_paths=True):
            self.raw_local_polygon_count += 1
            spec = LayerSpec(polygon.layer, polygon.datatype)
            if self.selected_layers is not None and spec not in self.selected_layers:
                continue
            points = tuple(transform.apply(float(point[0]), float(point[1])) for point in polygon.points)
            if not _float_bbox_intersects_window(points, self.window_bbox, self.unit, self.grid):
                continue
            snapped = self._snap_points(points)
            if len(snapped) >= 3 and bboxes_intersect(polygon_bbox(snapped), self.window_bbox):
                self.polygons.append((spec, snapped))
        for reference in cell.references:
            child = _reference_cell(reference)
            if id(child) in visiting:
                raise ValueError(f"hierarchy cycle detected at cell {child.name!r}")
            child_bbox = child.bounding_box()
            if child_bbox is None:
                continue
            for offset in _reference_offsets(reference):
                self.considered_reference_instance_count += 1
                reference_transform = _reference_transform(reference, offset)
                child_transform = transform.compose(reference_transform)
                transformed_bbox = _transform_bbox(child_bbox, child_transform, self.unit, self.grid)
                if not bboxes_intersect(transformed_bbox, self.window_bbox):
                    self.pruned_reference_instance_count += 1
                    continue
                self.traverse_cell(child, child_transform, visiting)
        visiting.remove(key)

    def _snap_points(self, points: tuple[tuple[float, float], ...]) -> tuple[tuple[int, int], ...]:
        snapped_points: list[tuple[int, int]] = []
        for x, y in points:
            x_physical = decimal_from_number(x) * self.unit
            y_physical = decimal_from_number(y) * self.unit
            sx = snap_to_grid(x_physical, self.grid)
            sy = snap_to_grid(y_physical, self.grid)
            self.max_snap_error = max(
                self.max_snap_error,
                abs(x_physical / self.grid - Decimal(sx)),
                abs(y_physical / self.grid - Decimal(sy)),
            )
            snapped_points.append((sx, sy))
        return normalize_polygon_points(snapped_points)

    def to_buffer(self) -> PolygonBuffer:
        layer_table = LayerTable(tuple(sorted({layer for layer, _ in self.polygons})))
        layer_index = layer_table.index_by_layer
        vertices: list[tuple[int, int]] = []
        offsets = [0]
        layer_ids = []
        bboxes = []
        flags = []
        source_ids = []
        for source_id, (layer, points) in enumerate(self.polygons):
            vertices.extend(points)
            offsets.append(len(vertices))
            layer_ids.append(layer_index[layer])
            bboxes.append(polygon_bbox(points))
            flags.append(polygon_flags(points))
            source_ids.append(source_id)
        return PolygonBuffer(
            vertices_xy=np.asarray(vertices, dtype=np.int64).reshape((-1, 2)),
            polygon_offsets=np.asarray(offsets, dtype=np.uint64),
            layer_ids=np.asarray(layer_ids, dtype=np.uint32),
            bboxes=np.asarray(bboxes, dtype=np.int64).reshape((-1, 4)),
            flags=np.asarray(flags, dtype=np.uint32),
            source_ids=np.asarray(source_ids, dtype=np.uint64),
            layer_table=layer_table,
        )


def _reference_cell(reference: gdstk.Reference) -> gdstk.Cell:
    if reference.cell is None:
        raise ValueError(f"unresolved reference to {reference.cell_name!r}")
    return reference.cell


def _reference_offsets(reference: gdstk.Reference) -> tuple[tuple[float, float], ...]:
    repetition = reference.repetition
    if repetition is None or repetition.size <= 1:
        return ((0.0, 0.0),)
    return tuple((float(x), float(y)) for x, y in repetition.get_offsets())


def _reference_transform(reference: gdstk.Reference, offset: tuple[float, float]) -> _Transform:
    rotation = float(reference.rotation or 0.0)
    magnification = float(reference.magnification if reference.magnification is not None else 1.0)
    cos_r = math.cos(rotation)
    sin_r = math.sin(rotation)
    if reference.x_reflection:
        a = magnification * cos_r
        b = magnification * sin_r
        c = magnification * sin_r
        d = -magnification * cos_r
    else:
        a = magnification * cos_r
        b = -magnification * sin_r
        c = magnification * sin_r
        d = magnification * cos_r
    return _Transform(
        a=a,
        b=b,
        c=c,
        d=d,
        tx=float(reference.origin[0]) + offset[0],
        ty=float(reference.origin[1]) + offset[1],
    )


def _transform_bbox(bbox, transform: _Transform, unit: Decimal, grid: Decimal) -> BBox:
    corners = (
        transform.apply(float(bbox[0][0]), float(bbox[0][1])),
        transform.apply(float(bbox[0][0]), float(bbox[1][1])),
        transform.apply(float(bbox[1][0]), float(bbox[0][1])),
        transform.apply(float(bbox[1][0]), float(bbox[1][1])),
    )
    snapped = [(snap_to_grid(decimal_from_number(x) * unit, grid), snap_to_grid(decimal_from_number(y) * unit, grid)) for x, y in corners]
    return polygon_bbox(tuple(snapped))


def _float_bbox_intersects_window(points: tuple[tuple[float, float], ...], window: BBox, unit: Decimal, grid: Decimal) -> bool:
    if not points:
        return False
    snapped = [(snap_to_grid(decimal_from_number(x) * unit, grid), snap_to_grid(decimal_from_number(y) * unit, grid)) for x, y in points]
    return bboxes_intersect(polygon_bbox(tuple(snapped)), window)


def _selected_layers_or_none(layers) -> set[LayerSpec] | None:
    parsed = parse_layer_selector(layers)
    if parsed == "all":
        return None
    return set(parsed)
