from __future__ import annotations

import gdstk

from gds_tools_tk.gdsdiff.backends.structural_ctypes import summarize_gds_structure_native
from gds_tools_tk.gdsdiff.structural_guard import summarize_gds_structure


def test_native_structural_summary_matches_python_and_gdstk(tmp_path):
    path = tmp_path / "hier.gds"
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    child = lib.new_cell("child")
    child.add(gdstk.rectangle((0, 0), (1, 1), layer=1))
    top = lib.new_cell("top")
    top.add(gdstk.Reference(child, (0, 0)))
    top.add(gdstk.Reference(child, (10, 0)))
    top.add(gdstk.rectangle((0, 10), (1, 11), layer=2))
    lib.write_gds(path)

    native = summarize_gds_structure_native(path)
    python = summarize_gds_structure(path)

    assert native.flattened_shape_count == python.flattened_shape_count == 3
    assert native.cell_count == python.cell_count == 2
    assert native.reachable_cell_count == python.reachable_cell_count == 2
    assert native.structural_hash != 0
    assert len(native.to_json()["structural_hash"]) == 16
