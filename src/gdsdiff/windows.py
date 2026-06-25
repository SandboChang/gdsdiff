from __future__ import annotations

from dataclasses import dataclass

from .config import LayerSpec
from .spatial import BBox, SpatialIndex, expand_bbox
from .tiling import CpuTilePrefilterResult, TileConfig, TileKey


@dataclass(frozen=True)
class CandidateWindow:
    layer: LayerSpec
    core_bbox: BBox
    verification_bbox: BBox
    source_candidate_count: int
    estimated_old_polygons: int
    estimated_new_polygons: int
    reason: str = "tile mismatch"

    def to_json(self) -> dict[str, object]:
        return {
            "layer": self.layer.to_json(),
            "core_bbox": list(self.core_bbox),
            "verification_bbox": list(self.verification_bbox),
            "source_candidate_count": self.source_candidate_count,
            "estimated_old_polygons": self.estimated_old_polygons,
            "estimated_new_polygons": self.estimated_new_polygons,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class WindowConfig:
    halo_dbu: int = 2
    max_tiles_per_window: int = 4096

    def __post_init__(self) -> None:
        if self.halo_dbu < 0:
            raise ValueError("halo_dbu must be nonnegative")
        if self.max_tiles_per_window <= 0:
            raise ValueError("max_tiles_per_window must be positive")


def build_candidate_windows(
    prefilter: CpuTilePrefilterResult,
    tile_config: TileConfig,
    old_index: SpatialIndex,
    new_index: SpatialIndex,
    window_config: WindowConfig | None = None,
) -> tuple[CandidateWindow, ...]:
    window_config = window_config or WindowConfig()
    windows: list[CandidateWindow] = []
    windows.extend(
        _windows_from_tiles(
            prefilter.candidate_tiles,
            tile_config,
            old_index,
            new_index,
            window_config,
        )
    )
    for oversized in prefilter.oversized_candidates:
        tx0, ty0, tx1, ty1 = oversized.tile_bbox
        core = _tile_rect_to_bbox(tx0, ty0, tx1, ty1, tile_config.tile_size)
        windows.append(
            _make_window(
                oversized.layer,
                core,
                source_count=(tx1 - tx0 + 1) * (ty1 - ty0 + 1),
                old_index=old_index,
                new_index=new_index,
                window_config=window_config,
                reason="oversized polygon",
            )
        )
    return tuple(sorted(windows, key=lambda window: (window.layer, window.core_bbox, window.reason)))


def _windows_from_tiles(
    tiles: tuple[TileKey, ...],
    tile_config: TileConfig,
    old_index: SpatialIndex,
    new_index: SpatialIndex,
    window_config: WindowConfig,
) -> list[CandidateWindow]:
    grouped: dict[LayerSpec, list[TileKey]] = {}
    for tile in tiles:
        grouped.setdefault(tile.layer, []).append(tile)
    windows: list[CandidateWindow] = []
    for layer, layer_tiles in grouped.items():
        rects = _merge_tiles_to_rectangles(layer_tiles, window_config.max_tiles_per_window)
        for tx0, ty0, tx1, ty1, count in rects:
            core = _tile_rect_to_bbox(tx0, ty0, tx1, ty1, tile_config.tile_size)
            windows.append(
                _make_window(
                    layer,
                    core,
                    source_count=count,
                    old_index=old_index,
                    new_index=new_index,
                    window_config=window_config,
                    reason="tile mismatch",
                )
            )
    return windows


def _merge_tiles_to_rectangles(tiles: list[TileKey], max_tiles_per_window: int) -> list[tuple[int, int, int, int, int]]:
    rows: dict[int, list[int]] = {}
    for tile in tiles:
        rows.setdefault(tile.ty, []).append(tile.tx)
    row_runs: list[tuple[int, int, int]] = []
    for ty, txs in rows.items():
        for x0, x1 in _runs(sorted(set(txs))):
            for split_x0, split_x1 in _split_run(x0, x1, max_tiles_per_window):
                row_runs.append((ty, split_x0, split_x1))
    row_runs.sort()
    rectangles: list[tuple[int, int, int, int, int]] = []
    active: dict[tuple[int, int], tuple[int, int, int, int, int]] = {}
    for ty, x0, x1 in row_runs:
        key = (x0, x1)
        prev = active.pop(key, None)
        width = x1 - x0 + 1
        if prev is not None and prev[3] + 1 == ty and prev[4] + width <= max_tiles_per_window:
            active[key] = (x0, prev[1], x1, ty, prev[4] + width)
        else:
            if prev is not None:
                rectangles.append(prev)
            active[key] = (x0, ty, x1, ty, width)
    rectangles.extend(active.values())
    return sorted(rectangles)


def _runs(values: list[int]) -> list[tuple[int, int]]:
    if not values:
        return []
    result = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        result.append((start, previous))
        start = previous = value
    result.append((start, previous))
    return result


def _split_run(x0: int, x1: int, max_width: int) -> list[tuple[int, int]]:
    if max_width <= 0:
        raise ValueError("max_width must be positive")
    result = []
    current = x0
    while current <= x1:
        end = min(x1, current + max_width - 1)
        result.append((current, end))
        current = end + 1
    return result


def _tile_rect_to_bbox(tx0: int, ty0: int, tx1: int, ty1: int, tile_size: int) -> BBox:
    return tx0 * tile_size, ty0 * tile_size, (tx1 + 1) * tile_size - 1, (ty1 + 1) * tile_size - 1


def _make_window(
    layer: LayerSpec,
    core: BBox,
    *,
    source_count: int,
    old_index: SpatialIndex,
    new_index: SpatialIndex,
    window_config: WindowConfig,
    reason: str,
) -> CandidateWindow:
    verification = expand_bbox(core, window_config.halo_dbu)
    return CandidateWindow(
        layer=layer,
        core_bbox=core,
        verification_bbox=verification,
        source_candidate_count=source_count,
        estimated_old_polygons=len(old_index.query(layer, verification)),
        estimated_new_polygons=len(new_index.query(layer, verification)),
        reason=reason,
    )
