from __future__ import annotations

import gdstk
import pytest

from gdsdiff import LayerSpec, inventory_gds, resolve_selected_layers


def _write_simple(path, *, cell_name="top", layer=(1, 0)):
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell(cell_name)
    cell.add(gdstk.rectangle((0, 0), (10, 10), layer=layer[0], datatype=layer[1]))
    lib.write_gds(path)


def test_inventory_selects_single_top_and_layers(tmp_path):
    path = tmp_path / "single.gds"
    _write_simple(path, layer=(2, 3))

    inventory = inventory_gds(path)

    assert inventory.selected_top == "top"
    assert inventory.metadata.top_names == ("top",)
    assert inventory.layers == (LayerSpec(2, 3),)
    assert inventory.records.polygons == 1
    assert inventory.records.cells == 1


def test_inventory_rejects_multiple_tops_without_selection(tmp_path):
    path = tmp_path / "multi.gds"
    lib = gdstk.Library()
    lib.new_cell("top_a").add(gdstk.rectangle((0, 0), (1, 1), layer=1))
    lib.new_cell("top_b").add(gdstk.rectangle((0, 0), (1, 1), layer=2))
    lib.write_gds(path)

    with pytest.raises(ValueError, match="exactly one top-level"):
        inventory_gds(path)

    selected = inventory_gds(path, top_name="top_b")
    assert selected.selected_top == "top_b"
    assert selected.layers == (LayerSpec(2, 0),)


def test_inventory_rejects_missing_named_top(tmp_path):
    path = tmp_path / "single.gds"
    _write_simple(path)

    with pytest.raises(ValueError, match="not found"):
        inventory_gds(path, top_name="missing")


def test_inventory_counts_reachable_references_labels_and_paths(tmp_path):
    path = tmp_path / "nested.gds"
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    child = lib.new_cell("child")
    child.add(gdstk.rectangle((0, 0), (10, 10), layer=1, datatype=0))
    child.add(gdstk.FlexPath([(0, 20), (20, 20)], 2, layer=7, datatype=1))
    child.add(gdstk.Label("ignored", (0, 0), layer=9, texttype=4))
    top = lib.new_cell("top")
    top.add(gdstk.Reference(child, (100, 100)))
    lib.write_gds(path)

    inventory = inventory_gds(path)

    assert inventory.selected_top == "top"
    assert inventory.reachable_cells == ("child", "top")
    assert LayerSpec(1, 0) in inventory.layers
    assert LayerSpec(7, 1) in inventory.layers
    assert inventory.records.cells == 2
    # In gdstk 1.0.0, this GDS round trip polygonizes the FlexPath record.
    assert inventory.records.polygons == 2
    assert inventory.records.paths == 0
    assert inventory.records.labels == 1
    assert inventory.records.references == 1
    assert inventory.ignored_record_counts["labels"] == 1


def test_inventory_matches_flattened_layers_for_nested_arrays(tmp_path):
    path = tmp_path / "arrays.gds"
    lib = gdstk.Library()
    leaf = lib.new_cell("leaf")
    leaf.add(gdstk.rectangle((0, 0), (1, 1), layer=7, datatype=3))
    leaf.add(gdstk.FlexPath([(0, 1), (1, 1)], 0.5, simple_path=True, layer=2, datatype=5))
    leaf.add(gdstk.Label("not geometry", (0, 0), layer=99, texttype=4))
    block = lib.new_cell("block")
    block.add(gdstk.Reference(leaf, columns=3, rows=4, spacing=(2, 2)))
    top = lib.new_cell("top")
    top.add(gdstk.Reference(block, (20, 25), magnification=2, x_reflection=True))
    top.add(gdstk.Reference(block, columns=2, rows=3, spacing=(30, 30)))
    lib.new_cell("unrelated").add(gdstk.rectangle((0, 0), (1, 1), layer=100))
    lib.write_gds(path)

    loaded_top = next(cell for cell in gdstk.read_gds(path).cells if cell.name == "top")
    flattened = {LayerSpec(p.layer, p.datatype) for p in loaded_top.get_polygons()}
    inventory = inventory_gds(path, top_name="top")

    assert inventory.layers == tuple(sorted(flattened)) == (LayerSpec(2, 5), LayerSpec(7, 3))
    assert inventory.reachable_cells == ("block", "leaf", "top")
    assert inventory.records.to_json() == {"polygons": 1, "paths": 1, "labels": 1, "references": 3, "cells": 3}
    assert inventory.metadata.cell_count == 4


def test_inventory_handles_deep_hierarchy(tmp_path):
    path = tmp_path / "deep.gds"
    lib = gdstk.Library()
    leaf = lib.new_cell("c0")
    leaf.add(gdstk.rectangle((0, 0), (1, 1), layer=4))
    child = leaf
    for i in range(1, 1200):
        parent = lib.new_cell(f"c{i}")
        parent.add(gdstk.Reference(child))
        child = parent
    lib.write_gds(path)

    inventory = inventory_gds(path)

    assert inventory.selected_top == "c1199"
    assert inventory.layers == (LayerSpec(4, 0),)
    assert inventory.records.cells == 1200
    assert inventory.records.references == 1199


def test_resolve_selected_layers_all_and_presence(tmp_path):
    old_path = tmp_path / "old.gds"
    new_path = tmp_path / "new.gds"
    _write_simple(old_path, layer=(1, 0))
    _write_simple(new_path, layer=(2, 0))
    old_inventory = inventory_gds(old_path)
    new_inventory = inventory_gds(new_path)

    selected = resolve_selected_layers(old_inventory, new_inventory, "all")

    assert [entry.to_json() for entry in selected] == [
        {"layer": 1, "datatype": 0, "present_old": True, "present_new": False},
        {"layer": 2, "datatype": 0, "present_old": False, "present_new": True},
    ]


def test_resolve_selected_layers_rejects_absent_both_by_default(tmp_path):
    old_path = tmp_path / "old.gds"
    new_path = tmp_path / "new.gds"
    _write_simple(old_path, layer=(1, 0))
    _write_simple(new_path, layer=(1, 0))
    old_inventory = inventory_gds(old_path)
    new_inventory = inventory_gds(new_path)

    with pytest.raises(ValueError, match="absent in both"):
        resolve_selected_layers(old_inventory, new_inventory, "9:0")

    selected = resolve_selected_layers(old_inventory, new_inventory, "9:0", allow_empty_selected_layer=True)
    assert selected[0].present_old is False
    assert selected[0].present_new is False
