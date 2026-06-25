from __future__ import annotations

import numpy as np

from gds_tools_tk.gpu_compare import canonicalize_polygon, intern_polygons
from gds_tools_tk.gpu_compare.geometry import LayerTable, PolygonBuffer
from gds_tools_tk.gpu_compare.config import LayerSpec


def _buffer(polygons, flags=None):
    vertices = []
    offsets = [0]
    for polygon in polygons:
        vertices.extend(polygon)
        offsets.append(len(vertices))
    if flags is None:
        flags = [0] * len(polygons)
    return PolygonBuffer(
        vertices_xy=np.asarray(vertices, dtype=np.int64).reshape((-1, 2)),
        polygon_offsets=np.asarray(offsets, dtype=np.uint64),
        layer_ids=np.zeros(len(polygons), dtype=np.uint32),
        bboxes=np.asarray([[min(x for x, _ in p), min(y for _, y in p), max(x for x, _ in p), max(y for _, y in p)] for p in polygons], dtype=np.int64).reshape((-1, 4)),
        flags=np.asarray(flags, dtype=np.uint32),
        source_ids=np.arange(len(polygons), dtype=np.uint64),
        layer_table=LayerTable((LayerSpec(1, 0),)),
    )


def test_valid_polygon_cyclic_shift_and_reverse_have_same_blob():
    square = np.asarray([(0, 0), (10, 0), (10, 10), (0, 10)], dtype=np.int64)
    shifted = np.asarray([(10, 10), (0, 10), (0, 0), (10, 0)], dtype=np.int64)
    reversed_square = np.asarray([(0, 10), (10, 10), (10, 0), (0, 0)], dtype=np.int64)

    assert canonicalize_polygon(square)[0] == canonicalize_polygon(shifted)[0]
    assert canonicalize_polygon(square)[0] == canonicalize_polygon(reversed_square)[0]


def test_invalid_polygon_preserves_raw_order_conservatively():
    line = np.asarray([(0, 0), (10, 0), (20, 0)], dtype=np.int64)
    reversed_line = np.asarray([(20, 0), (10, 0), (0, 0)], dtype=np.int64)

    assert canonicalize_polygon(line, valid=False)[0] != canonicalize_polygon(reversed_line, valid=False)[0]


def test_intern_polygons_assigns_deterministic_ids_and_duplicate_count():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    shifted_same = [(10, 10), (0, 10), (0, 0), (10, 0)]
    other = [(20, 20), (30, 20), (30, 30), (20, 30)]
    result = intern_polygons(_buffer([square, shifted_same, other]))

    assert result.canonical_ids[0] == result.canonical_ids[1]
    assert result.canonical_ids[2] != result.canonical_ids[0]
    assert result.duplicate_occurrence_count == 1
    assert len(result.unique_blobs) == 2


def test_forced_hash_collision_does_not_merge_different_blobs():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    other = [(20, 20), (30, 20), (30, 30), (20, 30)]
    result = intern_polygons(_buffer([square, other]), force_hash_collision=True)

    assert result.canonical_ids[0] != result.canonical_ids[1]
    assert result.hash_collision_bucket_count == 1
    assert len(result.unique_blobs) == 2


def test_invalid_flag_uses_conservative_raw_blob():
    line = [(0, 0), (10, 0), (20, 0)]
    reversed_line = [(20, 0), (10, 0), (0, 0)]
    result = intern_polygons(_buffer([line, reversed_line], flags=[1, 1]))

    assert result.canonical_ids[0] != result.canonical_ids[1]
