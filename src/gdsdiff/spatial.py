from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import LayerSpec
from .geometry import PolygonBuffer

BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class SpatialIndex:
    entries_by_layer: dict[LayerSpec, tuple[tuple[BBox, int], ...]]
    bbox_arrays_by_layer: dict[LayerSpec, np.ndarray] | None = None
    index_arrays_by_layer: dict[LayerSpec, np.ndarray] | None = None

    @classmethod
    def from_buffer(cls, buffer: PolygonBuffer) -> "SpatialIndex":
        entries: dict[LayerSpec, list[tuple[BBox, int]]] = {}
        for index in range(buffer.polygon_count):
            layer = buffer.layer_table.layers[int(buffer.layer_ids[index])]
            bbox = tuple(int(value) for value in buffer.bboxes[index])
            entries.setdefault(layer, []).append((bbox, index))
        sorted_entries = {layer: tuple(sorted(items, key=lambda item: (item[0], item[1]))) for layer, items in entries.items()}
        bbox_arrays = {
            layer: np.asarray([item[0] for item in items], dtype=np.int64).reshape((-1, 4))
            for layer, items in sorted_entries.items()
        }
        index_arrays = {
            layer: np.asarray([item[1] for item in items], dtype=np.int64)
            for layer, items in sorted_entries.items()
        }
        return cls(sorted_entries, bbox_arrays_by_layer=bbox_arrays, index_arrays_by_layer=index_arrays)

    def query(self, layer: LayerSpec, bbox: BBox) -> tuple[int, ...]:
        if self.bbox_arrays_by_layer is not None and self.index_arrays_by_layer is not None:
            bboxes = self.bbox_arrays_by_layer.get(layer)
            indices = self.index_arrays_by_layer.get(layer)
            if bboxes is None or indices is None or len(bboxes) == 0:
                return ()
            mask = (
                (bboxes[:, 0] <= bbox[2])
                & (bboxes[:, 2] >= bbox[0])
                & (bboxes[:, 1] <= bbox[3])
                & (bboxes[:, 3] >= bbox[1])
            )
            return tuple(int(index) for index in indices[np.nonzero(mask)[0]])
        return tuple(index for candidate_bbox, index in self.entries_by_layer.get(layer, ()) if bboxes_intersect(candidate_bbox, bbox))


def bboxes_intersect(left: BBox, right: BBox) -> bool:
    return left[0] <= right[2] and left[2] >= right[0] and left[1] <= right[3] and left[3] >= right[1]


def bbox_union(bboxes: list[BBox] | tuple[BBox, ...]) -> BBox:
    if not bboxes:
        raise ValueError("bbox_union requires at least one bbox")
    return (
        min(bbox[0] for bbox in bboxes),
        min(bbox[1] for bbox in bboxes),
        max(bbox[2] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes),
    )


def expand_bbox(bbox: BBox, amount: int) -> BBox:
    if amount < 0:
        raise ValueError("expansion amount must be nonnegative")
    return bbox[0] - amount, bbox[1] - amount, bbox[2] + amount, bbox[3] + amount
