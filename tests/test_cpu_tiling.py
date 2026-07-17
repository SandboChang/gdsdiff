from __future__ import annotations

import numpy as np
import pytest

from gdsdiff import (
    LayerSpec,
    TileConfig,
    TileKey,
    bbox_tile_range,
    floor_div,
    run_cpu_tile_prefilter,
)
from gdsdiff.backends.cpu import CpuTileBackend
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


def test_signed_floor_division_and_bbox_tile_range():
    assert floor_div(-1, 10) == -1
    assert floor_div(-10, 10) == -1
    assert floor_div(-11, 10) == -2
    assert bbox_tile_range((-11, -1, 10, 20), 10) == (-2, -1, 1, 2)


def test_identical_sets_produce_no_candidates():
    old = _buffer([_rect(0, 0, 9, 9)])
    new = _buffer([_rect(0, 0, 9, 9)])

    result = run_cpu_tile_prefilter(old, [7], new, [7], TileConfig(tile_size=10))

    assert result.candidate_tiles == ()
    assert result.assignment_count == 2
    assert result.collapsed_pair_count == 1


def test_changed_polygon_marks_old_and_new_bbox_tiles():
    old = _buffer([_rect(0, 0, 9, 9)])
    new = _buffer([_rect(20, 0, 29, 9)])

    result = run_cpu_tile_prefilter(old, [1], new, [2], TileConfig(tile_size=10))

    assert result.candidate_tiles == (TileKey(LayerSpec(1, 0), 0, 0), TileKey(LayerSpec(1, 0), 2, 0))


def test_duplicate_multiplicity_is_nonsemantic_for_prefilter():
    old = _buffer([_rect(0, 0, 9, 9), _rect(0, 0, 9, 9)])
    new = _buffer([_rect(0, 0, 9, 9)])

    result = run_cpu_tile_prefilter(old, [5, 5], new, [5], TileConfig(tile_size=10))

    assert result.candidate_tiles == ()
    assert result.assignment_count == 3
    assert result.collapsed_pair_count == 1


def test_tile_boundary_is_conservative():
    old = _buffer([_rect(0, 0, 10, 10)])
    new = _buffer([_rect(0, 0, 10, 10)])

    result = run_cpu_tile_prefilter(old, [1], new, [1], TileConfig(tile_size=10))

    assert result.candidate_tiles == ()
    assert result.assignment_count == 8
    assert result.collapsed_pair_count == 4


def test_negative_coordinate_candidate_tile():
    old = _buffer([_rect(-11, -11, -1, -1)])
    new = _buffer([])

    result = run_cpu_tile_prefilter(old, [1], new, [], TileConfig(tile_size=10))

    assert result.candidate_tiles == (
        TileKey(LayerSpec(1, 0), -2, -2),
        TileKey(LayerSpec(1, 0), -2, -1),
        TileKey(LayerSpec(1, 0), -1, -2),
        TileKey(LayerSpec(1, 0), -1, -1),
    )


def test_oversized_matching_polygon_is_not_candidate():
    old = _buffer([_rect(0, 0, 100, 100)])
    new = _buffer([_rect(0, 0, 100, 100)])

    result = run_cpu_tile_prefilter(old, [9], new, [9], TileConfig(tile_size=10, max_tiles_per_polygon=4))

    assert result.candidate_tiles == ()
    assert result.oversized_candidates == ()
    assert result.oversized_matched_count == 1


def test_oversized_unmatched_polygon_becomes_oversized_candidate():
    old = _buffer([_rect(0, 0, 100, 100)])
    new = _buffer([])

    result = run_cpu_tile_prefilter(old, [9], new, [], TileConfig(tile_size=10, max_tiles_per_polygon=4))

    assert result.candidate_tiles == ()
    assert len(result.oversized_candidates) == 1
    candidate = result.oversized_candidates[0]
    assert candidate.side_mask == 1
    assert candidate.tile_bbox == (0, 0, 10, 10)


def test_cpu_backend_wrapper_matches_function():
    old = _buffer([_rect(0, 0, 9, 9)])
    new = _buffer([_rect(20, 0, 29, 9)])
    config = TileConfig(tile_size=10)

    direct = run_cpu_tile_prefilter(old, [1], new, [2], config)
    backend = CpuTileBackend().compare_tile_contributors(old, [1], new, [2], config)

    assert CpuTileBackend().capabilities().name == "cpu"
    assert backend == direct


def test_rejects_canonical_id_length_mismatch():
    with pytest.raises(ValueError, match="canonical id vector"):
        run_cpu_tile_prefilter(_buffer([_rect(0, 0, 1, 1)]), [], _buffer([]), [], TileConfig(tile_size=10))
