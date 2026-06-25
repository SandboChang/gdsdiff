from __future__ import annotations

from dataclasses import dataclass, field

from .config import LayerSpec
from .geometry import PolygonBuffer


OLD_SIDE = 1
NEW_SIDE = 2
BOTH_SIDES = OLD_SIDE | NEW_SIDE


@dataclass(frozen=True, order=True)
class TileKey:
    layer: LayerSpec
    tx: int
    ty: int

    def to_json(self) -> dict[str, int]:
        return {
            "layer": self.layer.layer,
            "datatype": self.layer.datatype,
            "tx": self.tx,
            "ty": self.ty,
        }


@dataclass(frozen=True)
class TileConfig:
    tile_size: int
    max_tiles_per_polygon: int = 4096

    def __post_init__(self) -> None:
        if self.tile_size <= 0:
            raise ValueError("tile_size must be positive")
        if self.max_tiles_per_polygon <= 0:
            raise ValueError("max_tiles_per_polygon must be positive")


@dataclass(frozen=True)
class OversizedPolygon:
    layer: LayerSpec
    canonical_id: int
    side_mask: int
    tile_bbox: tuple[int, int, int, int]

    def to_json(self) -> dict[str, object]:
        return {
            "layer": self.layer.to_json(),
            "canonical_id": self.canonical_id,
            "side_mask": self.side_mask,
            "tile_bbox": list(self.tile_bbox),
        }


@dataclass(frozen=True)
class CpuTilePrefilterResult:
    candidate_tiles: tuple[TileKey, ...]
    oversized_candidates: tuple[OversizedPolygon, ...] = field(default_factory=tuple)
    assignment_count: int = 0
    collapsed_pair_count: int = 0
    oversized_matched_count: int = 0

    @property
    def candidate_tile_count(self) -> int:
        return len(self.candidate_tiles)

    def to_json(self) -> dict[str, object]:
        return {
            "candidate_tiles": [tile.to_json() for tile in self.candidate_tiles],
            "candidate_tile_count": len(self.candidate_tiles),
            "oversized_candidates": [candidate.to_json() for candidate in self.oversized_candidates],
            "assignment_count": self.assignment_count,
            "collapsed_pair_count": self.collapsed_pair_count,
            "oversized_matched_count": self.oversized_matched_count,
        }


def floor_div(value: int, divisor: int) -> int:
    if divisor <= 0:
        raise ValueError("divisor must be positive")
    return value // divisor


def bbox_tile_range(bbox: tuple[int, int, int, int], tile_size: int) -> tuple[int, int, int, int]:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    x0, y0, x1, y1 = bbox
    return (
        floor_div(x0, tile_size),
        floor_div(y0, tile_size),
        floor_div(x1, tile_size),
        floor_div(y1, tile_size),
    )


def run_cpu_tile_prefilter(
    old_buffer: PolygonBuffer,
    old_canonical_ids,
    new_buffer: PolygonBuffer,
    new_canonical_ids,
    config: TileConfig,
) -> CpuTilePrefilterResult:
    pair_side_masks: dict[tuple[TileKey, int], int] = {}
    oversized_side_masks: dict[tuple[LayerSpec, int, tuple[int, int, int, int]], int] = {}
    assignment_count = 0
    oversized_matched_count = 0

    for buffer, ids, side in ((old_buffer, old_canonical_ids, OLD_SIDE), (new_buffer, new_canonical_ids, NEW_SIDE)):
        if len(ids) != buffer.polygon_count:
            raise ValueError("canonical id vector length must match polygon count")
        for index in range(buffer.polygon_count):
            layer = buffer.layer_table.layers[int(buffer.layer_ids[index])]
            canonical_id = int(ids[index])
            tile_bbox = bbox_tile_range(tuple(int(value) for value in buffer.bboxes[index]), config.tile_size)
            tx0, ty0, tx1, ty1 = tile_bbox
            span_x = tx1 - tx0 + 1
            span_y = ty1 - ty0 + 1
            if span_x <= 0 or span_y <= 0:
                raise ValueError("invalid tile span")
            count = span_x * span_y
            if count > config.max_tiles_per_polygon:
                key = (layer, canonical_id, tile_bbox)
                oversized_side_masks[key] = oversized_side_masks.get(key, 0) | side
                continue
            for tx in range(tx0, tx1 + 1):
                for ty in range(ty0, ty1 + 1):
                    key = (TileKey(layer, tx, ty), canonical_id)
                    pair_side_masks[key] = pair_side_masks.get(key, 0) | side
                    assignment_count += 1

    candidate_tiles = {
        tile_key
        for (tile_key, _canonical_id), side_mask in pair_side_masks.items()
        if side_mask != BOTH_SIDES
    }
    oversized_candidates = []
    for (layer, canonical_id, tile_bbox), side_mask in oversized_side_masks.items():
        if side_mask == BOTH_SIDES:
            oversized_matched_count += 1
        else:
            oversized_candidates.append(
                OversizedPolygon(layer=layer, canonical_id=canonical_id, side_mask=side_mask, tile_bbox=tile_bbox)
            )

    return CpuTilePrefilterResult(
        candidate_tiles=tuple(sorted(candidate_tiles)),
        oversized_candidates=tuple(sorted(oversized_candidates, key=lambda item: (item.layer, item.canonical_id, item.tile_bbox))),
        assignment_count=assignment_count,
        collapsed_pair_count=len(pair_side_masks),
        oversized_matched_count=oversized_matched_count,
    )
