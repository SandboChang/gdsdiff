from __future__ import annotations

import math
import os
import struct
from collections import Counter
from pathlib import Path

import gdstk
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX on-demand native helpers are unavailable on Windows")

from gdsdiff.backends import polygon_extract_ctypes
from gdsdiff.backends.polygon_extract_ctypes import (
    PolygonExtractCtypesUnavailable,
    extract_gds_polygons_native,
)
from gdsdiff.config import LayerSpec
from gdsdiff.extract import extract_gds_polygons
from gdsdiff.geometry import FLAG_FEWER_THAN_THREE_POINTS, FLAG_ZERO_AREA


def _polygon_records(extraction):
    buffer = extraction.buffer
    records = []
    for index, layer_id in enumerate(buffer.layer_ids):
        layer = buffer.layer_table.layers[int(layer_id)]
        points = tuple((int(x), int(y)) for x, y in buffer.polygon_vertices(index))
        records.append((layer.layer, layer.datatype, points))
    return Counter(records)


def _native_hashes(extraction):
    return Counter(map(tuple, extraction.buffer.polygon_hashes.tolist()))


def test_native_polygon_extract_matches_nested_orthogonal_references(tmp_path):
    path = tmp_path / "nested.gds"
    flat_path = tmp_path / "flat.gds"
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    child = lib.new_cell("child")
    child.add(gdstk.Polygon([(-2, 1), (3, 1), (1, 4)], layer=1, datatype=2))
    child.add(gdstk.rectangle((1, 2), (2, 4), layer=3))
    child.add(gdstk.Polygon([(0, 0), (0, 1), (0, 2)], layer=4))
    nested = lib.new_cell("nested")
    nested.add(gdstk.Reference(child, (-4, 6), rotation=math.pi / 2))
    nested.add(gdstk.Reference(child, (3, -6), columns=2, rows=3, spacing=(8, -9)))
    top = lib.new_cell("top")
    top.add(gdstk.Reference(nested, (20, -15), rotation=3 * math.pi / 2))
    top.add(gdstk.Reference(child, (-25, 17), rotation=-math.pi / 2))
    top.add(gdstk.Reference(child, (11, 12), rotation=math.pi))
    lib.write_gds(path)

    native = extract_gds_polygons_native(path, top_name="top")
    reference = extract_gds_polygons(path, top_name="top")
    assert _polygon_records(native) == _polygon_records(reference)
    assert Counter(zip(map(tuple, native.buffer.bboxes), native.buffer.flags)) == Counter(
        zip(map(tuple, reference.buffer.bboxes), reference.buffer.flags)
    )
    flat = gdstk.Library(unit=lib.unit, precision=lib.precision)
    flat.new_cell("top").add(*gdstk.read_gds(path).top_level()[0].get_polygons())
    flat.write_gds(flat_path)
    assert _native_hashes(native) == _native_hashes(extract_gds_polygons_native(flat_path))


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
    assert native.oracle_reuse_eligible is True
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
    assert native.oracle_reuse_eligible is False
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


def test_native_packing_is_ordered_across_workers_and_layer_filters(tmp_path, monkeypatch):
    path = tmp_path / "repeated.gds"
    lib = gdstk.Library(unit=1e-6, precision=1e-6)
    child = lib.new_cell("child")
    layers = [(13, 3), (2, 4), (2, 1)]
    for i in range(8197):
        layer, datatype = layers[i % len(layers)]
        child.add(gdstk.Polygon([(3 * i, 0), (3 * i + 2, 0), (3 * i + 1, 2)], layer=layer, datatype=datatype))
    top = lib.new_cell("top")
    top.add(gdstk.rectangle((0, 0), (1, 1), layer=21))
    top.add(gdstk.Reference(child, (40, -5), rotation=math.pi / 2), gdstk.Reference(child, (-10, 8)))
    lib.new_cell("unreachable").add(gdstk.rectangle((0, 0), (2, 2), layer=99))
    lib.write_gds(path)

    monkeypatch.setattr(polygon_extract_ctypes, "_workers_per_input", lambda: 1)
    serial = extract_gds_polygons_native(path, top_name="top").buffer
    assert serial.layer_table.layers == (LayerSpec(2, 1), LayerSpec(2, 4), LayerSpec(13, 3), LayerSpec(21, 0))
    np.testing.assert_array_equal(serial.source_ids, np.arange(2 * 8197 + 1))
    np.testing.assert_array_equal(serial.polygon_vertices(1), [[40, -5], [40, -3], [38, -4]])
    np.testing.assert_array_equal(serial.polygon_vertices(8198), [[-10, 8], [-8, 8], [-9, 10]])

    monkeypatch.setattr(polygon_extract_ctypes, "_workers_per_input", lambda: 4)
    parallel = extract_gds_polygons_native(path, top_name="top", compute_hashes=False).buffer
    assert parallel.layer_table == serial.layer_table
    assert parallel.polygon_hashes is None
    for field in ("vertices_xy", "polygon_offsets", "bboxes", "layer_ids", "flags", "source_ids"):
        np.testing.assert_array_equal(getattr(parallel, field), getattr(serial, field))

    selected = extract_gds_polygons_native(path, top_name="top", layers="13:3,2:1").buffer
    keep = np.flatnonzero(np.isin(serial.layer_ids, [0, 2]))
    assert selected.layer_table.layers == (LayerSpec(2, 1), LayerSpec(13, 3))
    np.testing.assert_array_equal(selected.layer_ids, serial.layer_ids[keep] // 2)
    np.testing.assert_array_equal(selected.polygon_offsets, np.arange(len(keep) + 1) * 3)
    np.testing.assert_array_equal(selected.vertices_xy, np.vstack([serial.polygon_vertices(i) for i in keep]))
    for field in ("bboxes", "flags", "source_ids", "polygon_hashes"):
        np.testing.assert_array_equal(getattr(selected, field), getattr(serial, field)[keep])

    empty = extract_gds_polygons_native(path, top_name="top", layers="65538:1").buffer
    assert empty.polygon_count == empty.vertex_count == 0
    assert empty.layer_table.layers == ()
    assert empty.polygon_offsets.tolist() == [0]


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


def test_native_normalization_preserves_duplicates_and_wide_area_rules(tmp_path):
    path = tmp_path / "normalized.gds"
    lib = gdstk.Library(unit=1e-9, precision=1e-9)
    top = lib.new_cell("top")
    top.add(gdstk.Polygon([(0, 0), (0, 0), (2, 2), (0, 0), (2, 2), (0, 0)]))
    top.add(gdstk.rectangle((-2_000_000_000, -2_000_000_000), (2_000_000_000, 2_000_000_000)))
    lib.write_gds(path)

    native = extract_gds_polygons_native(path)
    reference = extract_gds_polygons(path)
    assert _polygon_records(native) == _polygon_records(reference)
    assert native.buffer.flags.tolist() == [FLAG_FEWER_THAN_THREE_POINTS | FLAG_ZERO_AREA, 0]
    np.testing.assert_array_equal(native.buffer.polygon_vertices(0), [[0, 0], [2, 2], [0, 0], [2, 2]])


def test_native_polygon_extract_matches_chip_fixture_record_counts():
    path = Path("references/testGDS/chip/chip_before.gds")
    if not path.exists():
        pytest.skip("chip fixture is not present")

    native = extract_gds_polygons_native(path, top_name="tdJTWPA_device")
    reference = extract_gds_polygons(path, top_name="tdJTWPA_device")

    assert native.buffer.polygon_count == reference.buffer.polygon_count
    assert native.buffer.vertex_count == reference.buffer.vertex_count
    assert Counter(int(flag) for flag in native.buffer.flags) == Counter(int(flag) for flag in reference.buffer.flags)


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"rotation": 0.5}, "nonorthogonal reference ANGLE"),
        ({"x_reflection": True}, "nonzero reference STRANS"),
        ({"magnification": 2}, "non-unit reference MAG"),
        ({"rotation": math.pi / 2, "columns": 2, "rows": 2, "spacing": (2, 2)}, "rotated AREF"),
    ],
)
def test_native_polygon_extract_fails_closed_on_unsupported_reference_transforms(tmp_path, kwargs, reason):
    path = tmp_path / "rotated_ref.gds"
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    child = lib.new_cell("child")
    child.add(gdstk.rectangle((0, 0), (1, 1), layer=1, datatype=0))
    top = lib.new_cell("top")
    top.add(gdstk.Reference(child, (0, 0), **kwargs))
    lib.write_gds(path)

    with pytest.raises(PolygonExtractCtypesUnavailable, match=reason):
        extract_gds_polygons_native(path)


@pytest.mark.parametrize(
    ("record", "reason"),
    [
        (bytes.fromhex("00061a010002"), "nonzero reference STRANS"),
        (bytes.fromhex("00061a020000"), "invalid STRANS record type or length"),
        (bytes.fromhex("000a1c05425a00000000"), "invalid ANGLE record type or length"),
        (bytes.fromhex("000c1c05425a000000000001"), "nonorthogonal reference ANGLE"),
    ],
)
def test_native_polygon_extract_checks_raw_transform_records(tmp_path, record, reason):
    path = tmp_path / "malformed.gds"
    lib = gdstk.Library()
    child = lib.new_cell("child")
    child.add(gdstk.rectangle((0, 0), (1, 2)))
    lib.new_cell("top").add(gdstk.Reference(child))
    lib.write_gds(path)

    data = path.read_bytes()
    offset = 0
    while True:
        length, kind, _ = struct.unpack_from(">HBB", data, offset)
        offset += length
        if kind == 0x12:  # SNAME; insert transform immediately after it.
            break
    path.write_bytes(data[:offset] + record + data[offset:])

    with pytest.raises(PolygonExtractCtypesUnavailable, match=reason):
        extract_gds_polygons_native(path)


def test_native_polygon_extract_rejects_fractional_array_pitch_under_rotation(tmp_path):
    path = tmp_path / "fractional_pitch.gds"
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    child = lib.new_cell("child")
    child.add(gdstk.rectangle((0, 0), (1, 2)))
    array = lib.new_cell("array")
    array.add(gdstk.Reference(child, columns=2, rows=1, spacing=(0.0005, 0)))
    lib.new_cell("top").add(gdstk.Reference(array, rotation=math.pi / 2))
    lib.write_gds(path)

    with pytest.raises(PolygonExtractCtypesUnavailable, match="nonintegral AREF pitch"):
        extract_gds_polygons_native(path)
