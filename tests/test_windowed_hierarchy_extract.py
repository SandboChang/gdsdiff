from __future__ import annotations

import gdstk
import numpy as np

from gds_tools_tk.gpu_compare import LayerSpec, run_full_exact_oracle
from gds_tools_tk.gpu_compare.geometry import LayerTable, PolygonBuffer, normalize_polygon_points, polygon_bbox, polygon_flags
from gds_tools_tk.gpu_compare.hierarchy_extract import extract_hierarchy_window
from gds_tools_tk.gpu_compare.spatial import bboxes_intersect


def _rect(x0, y0, x1, y1, *, layer=1, datatype=0):
    return gdstk.rectangle((x0, y0), (x1, y1), layer=layer, datatype=datatype)


def _buffer(polygons):
    layers = tuple(sorted({LayerSpec(polygon.layer, polygon.datatype) for polygon in polygons}))
    table = LayerTable(layers)
    layer_index = table.index_by_layer
    vertices = []
    offsets = [0]
    layer_ids = []
    bboxes = []
    flags = []
    source_ids = []
    for source_id, polygon in enumerate(polygons):
        points = normalize_polygon_points([(round(point[0]), round(point[1])) for point in polygon.points])
        if not points:
            continue
        vertices.extend(points)
        offsets.append(len(vertices))
        layer = LayerSpec(polygon.layer, polygon.datatype)
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
        layer_table=table,
    )


def _full_window_buffer(top, window):
    polygons = []
    for polygon in top.get_polygons(apply_repetitions=True, include_paths=True):
        points = normalize_polygon_points([(round(point[0]), round(point[1])) for point in polygon.points])
        if points and bboxes_intersect(polygon_bbox(points), window):
            polygons.append(polygon)
    return _buffer(polygons)


def _assert_window_matches_full(lib, window, *, top_name=None, layers="all"):
    top = next(cell for cell in lib.cells if cell.name == (top_name or "top"))
    windowed = extract_hierarchy_window(lib, window_bbox=window, top_name=top_name, layers=layers, meters_per_dbu=1e-6)
    full = _full_window_buffer(top, window)
    result = run_full_exact_oracle(windowed.buffer, full)
    assert result.equivalent
    return windowed, full


def test_window_extraction_matches_full_flatten_for_direct_and_reference_geometry():
    lib = gdstk.Library(unit=1e-6, precision=1e-6)
    child = lib.new_cell("child")
    child.add(_rect(0, 0, 10, 10))
    top = lib.new_cell("top")
    top.add(_rect(-20, 0, -10, 10, layer=2))
    top.add(gdstk.Reference(child, (100, 0)))

    windowed, full = _assert_window_matches_full(lib, (95, -5, 115, 15))

    assert windowed.buffer.polygon_count == full.polygon_count == 1
    assert windowed.stats.pruned_reference_instance_count == 0


def test_nested_references_match_full_flatten():
    lib = gdstk.Library(unit=1e-6, precision=1e-6)
    leaf = lib.new_cell("leaf")
    leaf.add(_rect(0, 0, 10, 10))
    mid = lib.new_cell("mid")
    mid.add(gdstk.Reference(leaf, (20, 0)))
    top = lib.new_cell("top")
    top.add(gdstk.Reference(mid, (100, 0)))

    windowed, full = _assert_window_matches_full(lib, (115, -5, 135, 15))

    assert windowed.buffer.polygon_count == full.polygon_count == 1
    assert windowed.stats.visited_cell_count == 3


def test_large_array_small_window_reduces_polygon_creation():
    lib = gdstk.Library(unit=1e-6, precision=1e-6)
    child = lib.new_cell("chip")
    child.add(_rect(0, 0, 5, 5))
    top = lib.new_cell("top")
    top.add(gdstk.Reference(child, (0, 0), columns=50, rows=20, spacing=(10, 10)))

    windowed, full = _assert_window_matches_full(lib, (95, 45, 106, 56))

    assert full.polygon_count == 4
    assert windowed.buffer.polygon_count == 4
    assert windowed.stats.considered_reference_instance_count == 1000
    assert windowed.stats.pruned_reference_instance_count == 996
    assert windowed.stats.emitted_polygon_count < 1000


def test_window_near_array_boundary_matches_full_flatten():
    lib = gdstk.Library(unit=1e-6, precision=1e-6)
    child = lib.new_cell("chip")
    child.add(_rect(0, 0, 5, 5))
    top = lib.new_cell("top")
    top.add(gdstk.Reference(child, (0, 0), columns=3, rows=2, spacing=(10, 10)))

    windowed, full = _assert_window_matches_full(lib, (9, -1, 16, 6))

    assert windowed.buffer.polygon_count == full.polygon_count == 1


def test_rotated_reference_bbox_is_conservative_and_matches_full():
    lib = gdstk.Library(unit=1e-6, precision=1e-6)
    child = lib.new_cell("rot")
    child.add(_rect(0, 0, 10, 20))
    top = lib.new_cell("top")
    top.add(gdstk.Reference(child, (100, 100), rotation=0.5))

    windowed, full = _assert_window_matches_full(lib, (85, 95, 110, 125))

    assert windowed.buffer.polygon_count == full.polygon_count == 1


def test_layer_selection_filters_output():
    lib = gdstk.Library(unit=1e-6, precision=1e-6)
    child = lib.new_cell("child")
    child.add(_rect(0, 0, 10, 10, layer=1))
    child.add(_rect(20, 0, 30, 10, layer=2))
    top = lib.new_cell("top")
    top.add(gdstk.Reference(child, (0, 0)))

    windowed = extract_hierarchy_window(lib, window_bbox=(-5, -5, 35, 15), layers=(LayerSpec(2, 0),), meters_per_dbu=1e-6)

    assert windowed.buffer.polygon_count == 1
    assert windowed.buffer.layer_table.layers == (LayerSpec(2, 0),)


def test_cycle_detection_in_windowed_extraction():
    lib = gdstk.Library(unit=1e-6, precision=1e-6)
    top = lib.new_cell("top")
    top.add(gdstk.Reference(top))

    try:
        extract_hierarchy_window(lib, window_bbox=(-1, -1, 1, 1), top_name="top", meters_per_dbu=1e-6)
    except ValueError as exc:
        assert "hierarchy cycle" in str(exc)
    else:
        raise AssertionError("expected hierarchy cycle failure")
