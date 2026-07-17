from __future__ import annotations

import numpy as np
import gdstk

from gdsdiff import LayerSpec, extract_gds_polygons
from gdsdiff.geometry import FLAG_FEWER_THAN_THREE_POINTS, FLAG_ZERO_AREA


def test_extract_simple_rectangle_to_integer_buffer(tmp_path):
    path = tmp_path / "rect.gds"
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell("top")
    cell.add(gdstk.rectangle((0, 0), (1, 2), layer=1, datatype=2))
    lib.write_gds(path)

    result = extract_gds_polygons(path, meters_per_dbu="1e-9")
    buffer = result.buffer

    assert result.top_name == "top"
    assert buffer.layer_table.layers == (LayerSpec(1, 2),)
    assert buffer.polygon_count == 1
    assert buffer.vertex_count == 4
    assert buffer.polygon_offsets.tolist() == [0, 4]
    assert buffer.bboxes.tolist() == [[0, 0, 1000, 2000]]
    assert set(map(tuple, buffer.polygon_vertices(0))) == {(0, 0), (1000, 0), (1000, 2000), (0, 2000)}


def test_extract_negative_coordinates_and_layer_filter(tmp_path):
    path = tmp_path / "filtered.gds"
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell("top")
    cell.add(gdstk.rectangle((-1, -2), (1, 2), layer=1, datatype=0))
    cell.add(gdstk.rectangle((10, 10), (11, 11), layer=2, datatype=0))
    lib.write_gds(path)

    result = extract_gds_polygons(path, layers="1:0", meters_per_dbu="1e-9")

    assert result.stats.raw_polygon_count == 2
    assert result.stats.kept_polygon_count == 1
    assert result.buffer.layer_table.layers == (LayerSpec(1, 0),)
    assert result.buffer.bboxes.tolist() == [[-1000, -2000, 1000, 2000]]


def test_extract_expands_references(tmp_path):
    path = tmp_path / "refs.gds"
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    child = lib.new_cell("child")
    child.add(gdstk.rectangle((0, 0), (1, 1), layer=1, datatype=0))
    top = lib.new_cell("top")
    top.add(gdstk.Reference(child, (10, 20)))
    top.add(gdstk.Reference(child, (30, 40)))
    lib.write_gds(path)

    result = extract_gds_polygons(path, meters_per_dbu="1e-9")

    assert result.buffer.polygon_count == 2
    assert sorted(map(tuple, result.buffer.bboxes.tolist())) == [
        (10000, 20000, 11000, 21000),
        (30000, 40000, 31000, 41000),
    ]


def test_extract_removes_repeated_closing_and_consecutive_duplicate_points(tmp_path):
    path = tmp_path / "duplicates.gds"
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell("top")
    cell.add(gdstk.Polygon([(0, 0), (1, 0), (1, 0), (1, 1), (0, 1), (0, 0)], layer=1))
    lib.write_gds(path)

    result = extract_gds_polygons(path, meters_per_dbu="1e-9")

    assert result.buffer.vertex_count == 4
    assert result.buffer.flags.tolist() == [0]


def test_extract_flags_degenerate_polygon(tmp_path):
    path = tmp_path / "degenerate.gds"
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell("top")
    cell.add(gdstk.Polygon([(0, 0), (1, 0), (2, 0)], layer=1))
    lib.write_gds(path)

    result = extract_gds_polygons(path, meters_per_dbu="1e-9")

    assert result.buffer.flags[0] & FLAG_ZERO_AREA
    assert not (result.buffer.flags[0] & FLAG_FEWER_THAN_THREE_POINTS)


def test_extract_empty_selected_layer_has_consistent_empty_buffers(tmp_path):
    path = tmp_path / "empty.gds"
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell("top")
    cell.add(gdstk.rectangle((0, 0), (1, 1), layer=1))
    lib.write_gds(path)

    result = extract_gds_polygons(path, layers="9:0", meters_per_dbu="1e-9")

    assert result.buffer.polygon_count == 0
    assert result.buffer.vertex_count == 0
    assert result.buffer.vertices_xy.shape == (0, 2)
    assert result.buffer.bboxes.shape == (0, 4)
    assert np.array_equal(result.buffer.polygon_offsets, np.asarray([0], dtype=np.uint64))


def test_fast_integer_extraction_matches_decimal_path(tmp_path, monkeypatch):
    path = tmp_path / "fast_extract.gds"
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell("top")
    cell.add(gdstk.Polygon([(-1.25, -0.5), (2.0, -0.5), (2.0, 1.75), (-1.25, 1.75)], layer=1, datatype=0))
    cell.add(gdstk.Polygon([(10.125, 3.25), (11.5, 3.25), (11.5, 4.75), (10.125, 4.75)], layer=2, datatype=3))
    lib.write_gds(path)

    monkeypatch.delenv("GDS_GPU_COMPARE_EXTRACT_DECIMAL", raising=False)
    fast = extract_gds_polygons(path, meters_per_dbu="1e-9").buffer

    monkeypatch.setenv("GDS_GPU_COMPARE_EXTRACT_DECIMAL", "1")
    decimal = extract_gds_polygons(path, meters_per_dbu="1e-9").buffer

    assert fast.layer_table.layers == decimal.layer_table.layers
    assert np.array_equal(fast.vertices_xy, decimal.vertices_xy)
    assert np.array_equal(fast.polygon_offsets, decimal.polygon_offsets)
    assert np.array_equal(fast.layer_ids, decimal.layer_ids)
    assert np.array_equal(fast.bboxes, decimal.bboxes)
    assert np.array_equal(fast.flags, decimal.flags)
