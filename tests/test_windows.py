from __future__ import annotations

import numpy as np

from gdsdiff import (
    CpuTilePrefilterResult,
    LayerSpec,
    OversizedPolygon,
    SpatialIndex,
    TileConfig,
    TileKey,
    WindowConfig,
    build_candidate_windows,
)
from gdsdiff.geometry import LayerTable, PolygonBuffer


def _buffer(polygons, layer=LayerSpec(1, 0)):
    vertices = []
    offsets = [0]
    for polygon in polygons:
        vertices.extend(polygon)
        offsets.append(len(vertices))
    bboxes = []
    for polygon in polygons:
        bboxes.append([min(x for x, _ in polygon), min(y for _, y in polygon), max(x for x, _ in polygon), max(y for _, y in polygon)])
    return PolygonBuffer(
        vertices_xy=np.asarray(vertices, dtype=np.int64).reshape((-1, 2)),
        polygon_offsets=np.asarray(offsets, dtype=np.uint64),
        layer_ids=np.zeros(len(polygons), dtype=np.uint32),
        bboxes=np.asarray(bboxes, dtype=np.int64).reshape((-1, 4)),
        flags=np.zeros(len(polygons), dtype=np.uint32),
        source_ids=np.arange(len(polygons), dtype=np.uint64),
        layer_table=LayerTable((layer,)),
    )


def _rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def test_sparse_candidate_tiles_create_separate_windows():
    layer = LayerSpec(1, 0)
    prefilter = CpuTilePrefilterResult(
        candidate_tiles=(TileKey(layer, 0, 0), TileKey(layer, 3, 0)),
        assignment_count=2,
        collapsed_pair_count=2,
    )
    index = SpatialIndex.from_buffer(_buffer([_rect(0, 0, 9, 9), _rect(30, 0, 39, 9)]))

    windows = build_candidate_windows(prefilter, TileConfig(tile_size=10), index, index, WindowConfig(halo_dbu=2))

    assert [window.core_bbox for window in windows] == [(0, 0, 9, 9), (30, 0, 39, 9)]
    assert [window.verification_bbox for window in windows] == [(-2, -2, 11, 11), (28, -2, 41, 11)]


def test_adjacent_tiles_merge_horizontally_and_vertically():
    layer = LayerSpec(1, 0)
    prefilter = CpuTilePrefilterResult(
        candidate_tiles=(
            TileKey(layer, 0, 0),
            TileKey(layer, 1, 0),
            TileKey(layer, 0, 1),
            TileKey(layer, 1, 1),
        )
    )
    index = SpatialIndex.from_buffer(_buffer([_rect(0, 0, 19, 19)]))

    windows = build_candidate_windows(prefilter, TileConfig(tile_size=10), index, index, WindowConfig(halo_dbu=0))

    assert len(windows) == 1
    assert windows[0].core_bbox == (0, 0, 19, 19)
    assert windows[0].source_candidate_count == 4


def test_negative_tile_window_bbox():
    layer = LayerSpec(1, 0)
    prefilter = CpuTilePrefilterResult(candidate_tiles=(TileKey(layer, -2, -1),))
    index = SpatialIndex.from_buffer(_buffer([_rect(-20, -10, -11, -1)]))

    windows = build_candidate_windows(prefilter, TileConfig(tile_size=10), index, index, WindowConfig(halo_dbu=0))

    assert windows[0].core_bbox == (-20, -10, -11, -1)


def test_window_split_by_max_tiles_per_window():
    layer = LayerSpec(1, 0)
    prefilter = CpuTilePrefilterResult(
        candidate_tiles=tuple(TileKey(layer, x, 0) for x in range(4))
    )
    index = SpatialIndex.from_buffer(_buffer([_rect(0, 0, 39, 9)]))

    windows = build_candidate_windows(
        prefilter,
        TileConfig(tile_size=10),
        index,
        index,
        WindowConfig(halo_dbu=0, max_tiles_per_window=2),
    )

    assert [window.core_bbox for window in windows] == [(0, 0, 19, 9), (20, 0, 39, 9)]


def test_oversized_candidate_rectangle_becomes_window():
    layer = LayerSpec(1, 0)
    prefilter = CpuTilePrefilterResult(
        candidate_tiles=(),
        oversized_candidates=(OversizedPolygon(layer, canonical_id=3, side_mask=1, tile_bbox=(0, 0, 2, 1)),),
    )
    index = SpatialIndex.from_buffer(_buffer([_rect(0, 0, 29, 19)]))

    windows = build_candidate_windows(prefilter, TileConfig(tile_size=10), index, index, WindowConfig(halo_dbu=1))

    assert len(windows) == 1
    assert windows[0].core_bbox == (0, 0, 29, 19)
    assert windows[0].verification_bbox == (-1, -1, 30, 20)
    assert windows[0].reason == "oversized polygon"
    assert windows[0].estimated_old_polygons == 1
