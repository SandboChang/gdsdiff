from __future__ import annotations

import pytest
import gdstk

from scgds.gdsdiff import canonical_cell_identity, compare_hierarchy


def _rect(x0, y0, x1, y1, *, layer=1, datatype=0):
    return gdstk.rectangle((x0, y0), (x1, y1), layer=layer, datatype=datatype)


def _lib_with_references(child_name="child", top_name="top", origins=((0, 0), (20, 0))):
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    child = lib.new_cell(child_name)
    child.add(_rect(0, 0, 10, 10))
    top = lib.new_cell(top_name)
    for origin in origins:
        top.add(gdstk.Reference(child, origin))
    return lib


def test_same_hierarchy_is_identical():
    old = _lib_with_references()
    new = _lib_with_references()

    result = compare_hierarchy(old, new)

    assert result.identical
    assert result.changed_references == ()
    assert result.matched_subtree_count >= 1


def test_renamed_cells_keep_same_identity():
    old = _lib_with_references(child_name="old_child", top_name="old_top")
    new = _lib_with_references(child_name="new_child", top_name="new_top")

    result = compare_hierarchy(old, new)

    assert result.identical
    assert result.old_top == "old_top"
    assert result.new_top == "new_top"


def test_reordered_references_keep_same_identity():
    old = _lib_with_references(origins=((0, 0), (20, 0), (40, 0)))
    new = _lib_with_references(origins=((40, 0), (0, 0), (20, 0)))

    result = compare_hierarchy(old, new)

    assert result.identical


def test_same_coverage_different_hierarchy_is_not_skip_match():
    old = gdstk.Library(unit=1e-6, precision=1e-9)
    old_top = old.new_cell("top")
    old_top.add(_rect(0, 0, 10, 10))
    new = _lib_with_references(origins=((0, 0),))

    result = compare_hierarchy(old, new)

    assert not result.identical


def test_changed_shared_child_marks_reference_regions():
    old = _lib_with_references()
    new = _lib_with_references()
    new["child"].add(_rect(12, 0, 14, 10))

    result = compare_hierarchy(old, new)

    assert not result.identical
    assert [change.kind for change in result.changed_references] == ["changed", "changed"]
    assert all(change.old_child_digest != change.new_child_digest for change in result.changed_references)


def test_moved_reference_marks_removed_and_added_regions():
    old = _lib_with_references(origins=((0, 0),))
    new = _lib_with_references(origins=((5, 0),))

    result = compare_hierarchy(old, new)

    assert not result.identical
    assert sorted(change.kind for change in result.changed_references) == ["added", "removed"]


def test_changed_repetition_is_not_identical():
    old = gdstk.Library(unit=1e-6, precision=1e-9)
    old_child = old.new_cell("child")
    old_child.add(_rect(0, 0, 10, 10))
    old_top = old.new_cell("top")
    old_top.add(gdstk.Reference(old_child, (0, 0), columns=3, rows=1, spacing=(20, 0)))

    new = gdstk.Library(unit=1e-6, precision=1e-9)
    new_child = new.new_cell("child")
    new_child.add(_rect(0, 0, 10, 10))
    new_top = new.new_cell("top")
    new_top.add(gdstk.Reference(new_child, (0, 0), columns=2, rows=1, spacing=(20, 0)))

    result = compare_hierarchy(old, new)

    assert not result.identical
    assert sorted(change.kind for change in result.changed_references) == ["added", "removed"]


def test_nested_transform_identity_ignores_names():
    old = gdstk.Library(unit=1e-6, precision=1e-9)
    old_leaf = old.new_cell("old_leaf")
    old_leaf.add(_rect(0, 0, 10, 10))
    old_mid = old.new_cell("old_mid")
    old_mid.add(gdstk.Reference(old_leaf, (2, 3), rotation=0.5, magnification=2, x_reflection=True))
    old_top = old.new_cell("old_top")
    old_top.add(gdstk.Reference(old_mid, (100, 0)))

    new = gdstk.Library(unit=1e-6, precision=1e-9)
    new_leaf = new.new_cell("new_leaf")
    new_leaf.add(_rect(0, 0, 10, 10))
    new_mid = new.new_cell("new_mid")
    new_mid.add(gdstk.Reference(new_leaf, (2, 3), rotation=0.5, magnification=2, x_reflection=True))
    new_top = new.new_cell("new_top")
    new_top.add(gdstk.Reference(new_mid, (100, 0)))

    assert compare_hierarchy(old, new).identical


def test_cycle_detection():
    lib = gdstk.Library()
    cell = lib.new_cell("loop")
    cell.add(gdstk.Reference(cell))

    with pytest.raises(ValueError, match="hierarchy cycle"):
        canonical_cell_identity(cell)
