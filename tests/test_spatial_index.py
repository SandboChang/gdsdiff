from __future__ import annotations

import numpy as np

from gds_tools_tk.gdsdiff import LayerSpec, SpatialIndex, bboxes_intersect
from gds_tools_tk.gdsdiff.geometry import LayerTable, PolygonBuffer


def _buffer():
    polygons = [
        [(0, 0), (10, 0), (10, 10), (0, 10)],
        [(20, 0), (30, 0), (30, 10), (20, 10)],
    ]
    vertices = [point for polygon in polygons for point in polygon]
    return PolygonBuffer(
        vertices_xy=np.asarray(vertices, dtype=np.int64),
        polygon_offsets=np.asarray([0, 4, 8], dtype=np.uint64),
        layer_ids=np.asarray([0, 0], dtype=np.uint32),
        bboxes=np.asarray([[0, 0, 10, 10], [20, 0, 30, 10]], dtype=np.int64),
        flags=np.zeros(2, dtype=np.uint32),
        source_ids=np.arange(2, dtype=np.uint64),
        layer_table=LayerTable((LayerSpec(1, 0),)),
    )


def test_bboxes_intersect_inclusive_edges():
    assert bboxes_intersect((0, 0, 10, 10), (10, 10, 20, 20))
    assert not bboxes_intersect((0, 0, 10, 10), (11, 0, 20, 10))


def test_spatial_index_queries_by_layer_and_bbox():
    index = SpatialIndex.from_buffer(_buffer())

    assert index.query(LayerSpec(1, 0), (5, 5, 6, 6)) == (0,)
    assert index.query(LayerSpec(1, 0), (9, 0, 21, 10)) == (0, 1)
    assert index.query(LayerSpec(2, 0), (0, 0, 100, 100)) == ()
