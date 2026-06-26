from __future__ import annotations

import gdstk

from gds_tools_tk.gdsdiff.backends.geometry_ctypes import compare_gds_geometry_native, summarize_gds_geometry_native
from gds_tools_tk.gdsdiff.config import LayerSpec
from gds_tools_tk.gdsdiff.extract import extract_gds_polygons


def _gdstk_layer_summary(path, *, top_name=None):
    extraction = extract_gds_polygons(path, top_name=top_name)
    summary = {}
    for index, layer_id in enumerate(extraction.buffer.layer_ids):
        layer = extraction.buffer.layer_table.layers[int(layer_id)]
        bbox = tuple(int(value) for value in extraction.buffer.bboxes[index])
        count, existing = summary.get(layer, (0, None))
        if existing is None:
            merged = bbox
        else:
            merged = (
                min(existing[0], bbox[0]),
                min(existing[1], bbox[1]),
                max(existing[2], bbox[2]),
                max(existing[3], bbox[3]),
            )
        summary[layer] = (count + 1, merged)
    return extraction.buffer.polygon_count, summary


def test_native_geometry_summary_matches_gdstk_flat_layers(tmp_path):
    path = tmp_path / "flat.gds"
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    top = lib.new_cell("top")
    top.add(gdstk.rectangle((0, 0), (1, 2), layer=1, datatype=0))
    top.add(gdstk.rectangle((3, 4), (5, 6), layer=1, datatype=0))
    top.add(gdstk.rectangle((-1, -2), (0, 0), layer=2, datatype=7))
    lib.write_gds(path)

    native = summarize_gds_geometry_native(path)
    gdstk_count, gdstk_layers = _gdstk_layer_summary(path)

    assert native.flattened_shape_count == gdstk_count == 3
    assert {item.layer: (item.count, item.bbox) for item in native.layers} == gdstk_layers
    assert native.geometry_hash != 0


def test_native_geometry_summary_matches_gdstk_translated_references(tmp_path):
    path = tmp_path / "hier.gds"
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    child = lib.new_cell("child")
    child.add(gdstk.rectangle((0, 0), (1, 1), layer=4, datatype=0))
    top = lib.new_cell("top")
    top.add(gdstk.Reference(child, (10, 0)))
    top.add(gdstk.Reference(child, (20, 5)))
    top.add(gdstk.rectangle((0, 0), (1, 1), layer=5, datatype=1))
    lib.write_gds(path)

    native = summarize_gds_geometry_native(path)
    gdstk_count, gdstk_layers = _gdstk_layer_summary(path)

    assert native.flattened_shape_count == gdstk_count == 3
    assert {item.layer: (item.count, item.bbox) for item in native.layers} == gdstk_layers
    assert LayerSpec(4, 0) in gdstk_layers


def test_native_geometry_counts_only_skips_layer_details(tmp_path):
    path = tmp_path / "flat_counts.gds"
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    top = lib.new_cell("top")
    top.add(gdstk.rectangle((0, 0), (1, 2), layer=1, datatype=0))
    top.add(gdstk.rectangle((3, 4), (5, 6), layer=2, datatype=0))
    lib.write_gds(path)

    native = summarize_gds_geometry_native(path, counts_only=True)

    assert native.flattened_shape_count == 2
    assert native.layers == ()
    assert native.geometry_hash == 0


def test_native_geometry_pair_compare_stops_after_count_mismatch(tmp_path):
    old_path = tmp_path / "old.gds"
    new_path = tmp_path / "new.gds"
    old = gdstk.Library(unit=1e-6, precision=1e-9)
    new = gdstk.Library(unit=1e-6, precision=1e-9)
    old_top = old.new_cell("top")
    new_top = new.new_cell("top")
    old_top.add(gdstk.rectangle((0, 0), (1, 1), layer=1))
    new_top.add(gdstk.rectangle((0, 0), (1, 1), layer=1))
    new_top.add(gdstk.rectangle((2, 0), (3, 1), layer=1))
    old.write_gds(old_path)
    new.write_gds(new_path)

    comparison = compare_gds_geometry_native(old_path, new_path)

    assert comparison.summaries_match is False
    assert comparison.detail_computed is False
    assert comparison.old_summary.flattened_shape_count == 1
    assert comparison.new_summary.flattened_shape_count == 2
    assert comparison.old_summary.layers == ()
    assert comparison.new_summary.layers == ()
