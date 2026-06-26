from __future__ import annotations

from collections import Counter
from pathlib import Path

import gdstk
import pytest

from gds_tools_tk.gdsdiff.backends.polygon_extract_ctypes import (
    PolygonExtractCtypesUnavailable,
    extract_gds_polygons_native,
)
from gds_tools_tk.gdsdiff.config import LayerSpec
from gds_tools_tk.gdsdiff.extract import extract_gds_polygons
from gds_tools_tk.gdsdiff.geometry import FLAG_FEWER_THAN_THREE_POINTS, FLAG_ZERO_AREA


def _polygon_records(extraction):
    buffer = extraction.buffer
    records = []
    for index, layer_id in enumerate(buffer.layer_ids):
        layer = buffer.layer_table.layers[int(layer_id)]
        points = tuple((int(x), int(y)) for x, y in buffer.polygon_vertices(index))
        records.append((layer.layer, layer.datatype, points))
    return Counter(records)


def test_native_polygon_extract_matches_flat_gdstk_extraction(tmp_path):
    path = tmp_path / "flat.gds"
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    top = lib.new_cell("top")
    top.add(gdstk.rectangle((0, 0), (1, 2), layer=1, datatype=0))
    top.add(gdstk.Polygon([(0, 0), (2, 1), (1, 3)], layer=2, datatype=5))
    lib.write_gds(path)

    native = extract_gds_polygons_native(path)
    reference = extract_gds_polygons(path)

    assert native.buffer.polygon_count == reference.buffer.polygon_count == 2
    assert native.buffer.vertex_count == reference.buffer.vertex_count
    assert _polygon_records(native) == _polygon_records(reference)


def test_native_polygon_extract_matches_translated_sref_and_aref(tmp_path):
    path = tmp_path / "hier.gds"
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    child = lib.new_cell("child")
    child.add(gdstk.rectangle((0, 0), (1, 1), layer=4, datatype=0))
    top = lib.new_cell("top")
    top.add(gdstk.Reference(child, (10, 0)))
    top.add(gdstk.Reference(child, (30, 0), columns=2, rows=2, spacing=(10, 20)))
    top.add(gdstk.rectangle((0, 0), (1, 1), layer=5, datatype=1))
    lib.write_gds(path)

    native = extract_gds_polygons_native(path)
    reference = extract_gds_polygons(path)

    assert native.buffer.polygon_count == reference.buffer.polygon_count == 6
    assert _polygon_records(native) == _polygon_records(reference)


def test_native_polygon_extract_filters_layers(tmp_path):
    path = tmp_path / "layers.gds"
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    top = lib.new_cell("top")
    top.add(gdstk.rectangle((0, 0), (1, 1), layer=1, datatype=0))
    top.add(gdstk.rectangle((2, 0), (3, 1), layer=2, datatype=0))
    lib.write_gds(path)

    native = extract_gds_polygons_native(path, layers=(LayerSpec(2, 0),))

    assert native.buffer.polygon_count == 1
    assert native.buffer.layer_table.layers == (LayerSpec(2, 0),)


def test_native_polygon_extract_preserves_degenerate_polygon_records(tmp_path):
    path = tmp_path / "degenerate.gds"
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    top = lib.new_cell("top")
    top.add(gdstk.Polygon([(2, 0), (3, 0), (4, 0)], layer=9, datatype=1))
    lib.write_gds(path)

    native = extract_gds_polygons_native(path)
    reference = extract_gds_polygons(path)

    assert native.buffer.polygon_count == reference.buffer.polygon_count == 1
    assert _polygon_records(native) == _polygon_records(reference)
    assert [int(flag) for flag in native.buffer.flags] == [int(flag) for flag in reference.buffer.flags]
    assert int(native.buffer.flags[0]) & FLAG_ZERO_AREA
    assert not (int(native.buffer.flags[0]) & FLAG_FEWER_THAN_THREE_POINTS)


def test_native_polygon_extract_matches_chip_fixture_record_counts():
    path = Path("references/testGDS/chip/chip_before.gds")
    if not path.exists():
        pytest.skip("chip fixture is not present")

    native = extract_gds_polygons_native(path, top_name="tdJTWPA_device")
    reference = extract_gds_polygons(path, top_name="tdJTWPA_device")

    assert native.buffer.polygon_count == reference.buffer.polygon_count
    assert native.buffer.vertex_count == reference.buffer.vertex_count
    assert Counter(int(flag) for flag in native.buffer.flags) == Counter(int(flag) for flag in reference.buffer.flags)


def test_native_polygon_extract_fails_closed_on_transformed_references(tmp_path):
    path = tmp_path / "rotated_ref.gds"
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    child = lib.new_cell("child")
    child.add(gdstk.rectangle((0, 0), (1, 1), layer=1, datatype=0))
    top = lib.new_cell("top")
    top.add(gdstk.Reference(child, (0, 0), rotation=0.5))
    lib.write_gds(path)

    with pytest.raises(PolygonExtractCtypesUnavailable, match="transformed references"):
        extract_gds_polygons_native(path)
