from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import LayerSpec

FLAG_FEWER_THAN_THREE_POINTS = 1 << 0
FLAG_ZERO_AREA = 1 << 1


@dataclass(frozen=True)
class LayerTable:
    layers: tuple[LayerSpec, ...]

    def __post_init__(self) -> None:
        if len(set(self.layers)) != len(self.layers):
            raise ValueError("LayerTable cannot contain duplicate layers")

    @property
    def index_by_layer(self) -> dict[LayerSpec, int]:
        return {layer: index for index, layer in enumerate(self.layers)}

    def to_json(self) -> list[dict[str, int]]:
        return [layer.to_json() for layer in self.layers]


@dataclass(frozen=True)
class PolygonBuffer:
    vertices_xy: np.ndarray
    polygon_offsets: np.ndarray
    layer_ids: np.ndarray
    bboxes: np.ndarray
    flags: np.ndarray
    source_ids: np.ndarray
    layer_table: LayerTable

    def __post_init__(self) -> None:
        if self.vertices_xy.dtype != np.int64 or self.vertices_xy.ndim != 2 or self.vertices_xy.shape[1] != 2:
            raise TypeError("vertices_xy must be an int64 array with shape (N, 2)")
        if self.polygon_offsets.dtype != np.uint64 or self.polygon_offsets.ndim != 1:
            raise TypeError("polygon_offsets must be a uint64 vector")
        if self.layer_ids.dtype != np.uint32 or self.layer_ids.ndim != 1:
            raise TypeError("layer_ids must be a uint32 vector")
        if self.bboxes.dtype != np.int64 or self.bboxes.shape != (self.polygon_count, 4):
            raise TypeError("bboxes must be an int64 array with shape (N_polygons, 4)")
        if self.flags.dtype != np.uint32 or self.flags.shape != (self.polygon_count,):
            raise TypeError("flags must be a uint32 vector with one entry per polygon")
        if self.source_ids.dtype != np.uint64 or self.source_ids.shape != (self.polygon_count,):
            raise TypeError("source_ids must be a uint64 vector with one entry per polygon")
        if len(self.polygon_offsets) != self.polygon_count + 1:
            raise ValueError("polygon_offsets length must be polygon_count + 1")
        if len(self.polygon_offsets) and int(self.polygon_offsets[0]) != 0:
            raise ValueError("polygon_offsets must start at zero")
        if len(self.polygon_offsets) and int(self.polygon_offsets[-1]) != self.vertex_count:
            raise ValueError("final polygon offset must equal vertex count")

    @property
    def polygon_count(self) -> int:
        return len(self.layer_ids)

    @property
    def vertex_count(self) -> int:
        return len(self.vertices_xy)

    def polygon_vertices(self, index: int) -> np.ndarray:
        start = int(self.polygon_offsets[index])
        end = int(self.polygon_offsets[index + 1])
        return self.vertices_xy[start:end]


def normalize_polygon_points(points: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    if len(points) > 1 and points[0] == points[-1]:
        points = points[:-1]
    normalized: list[tuple[int, int]] = []
    for point in points:
        if not normalized or normalized[-1] != point:
            normalized.append(point)
    if len(normalized) > 1 and normalized[0] == normalized[-1]:
        normalized.pop()
    return tuple(normalized)


def polygon_bbox(points: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def polygon_twice_area(points: tuple[tuple[int, int], ...]) -> int:
    if len(points) < 3:
        return 0
    area = 0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        area += point[0] * next_point[1] - next_point[0] * point[1]
    return abs(area)


def polygon_flags(points: tuple[tuple[int, int], ...]) -> int:
    flags = 0
    if len(set(points)) < 3:
        flags |= FLAG_FEWER_THAN_THREE_POINTS
    if polygon_twice_area(points) == 0:
        flags |= FLAG_ZERO_AREA
    return flags
