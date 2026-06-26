from __future__ import annotations

import gdstk
import pytest

from gds_tools_tk.gdsdiff.backends.cuda_ctypes import (
    cuda_ctypes_available,
    run_cuda_polygon_record_delta,
)
from gds_tools_tk.gdsdiff.backends.polygon_extract_ctypes import extract_gds_polygons_native


def test_cuda_polygon_record_delta_cancels_identical_records_and_reports_directional_deltas(tmp_path):
    if not cuda_ctypes_available():
        pytest.skip("CUDA ctypes backend is not available")

    old_path = tmp_path / "old.gds"
    new_path = tmp_path / "new.gds"
    old_lib = gdstk.Library(unit=1e-6, precision=1e-9)
    old_top = old_lib.new_cell("top")
    old_top.add(gdstk.rectangle((0, 0), (1, 1), layer=1, datatype=0))
    old_top.add(gdstk.rectangle((2, 0), (3, 1), layer=1, datatype=0))
    old_lib.write_gds(old_path)

    new_lib = gdstk.Library(unit=1e-6, precision=1e-9)
    new_top = new_lib.new_cell("top")
    new_top.add(gdstk.rectangle((0, 0), (1, 1), layer=1, datatype=0))
    new_top.add(gdstk.rectangle((4, 0), (5, 1), layer=1, datatype=0))
    new_lib.write_gds(new_path)

    old_extraction = extract_gds_polygons_native(old_path)
    new_extraction = extract_gds_polygons_native(new_path)
    delta = run_cuda_polygon_record_delta(old_extraction.buffer, new_extraction.buffer)

    assert delta.cuda_kernel_count == 2
    assert delta.cuda_event_elapsed_s > 0
    assert delta.matched_record_count == 1
    assert delta.record_count == 2
    assert tuple(int(value) for value in delta.representative_indices) == (1, 1)
    rows = {
        (
            int(delta.directions[index]),
            int(delta.layers[index]),
            int(delta.datatypes[index]),
            tuple(int(value) for value in delta.bboxes[index]),
            int(delta.counts[index]),
        )
        for index in range(delta.record_count)
    }
    assert rows == {
        (1, 1, 0, (2000, 0, 3000, 1000), 1),
        (2, 1, 0, (4000, 0, 5000, 1000), 1),
    }


def test_cuda_polygon_record_delta_canonicalizes_vertex_order(tmp_path):
    if not cuda_ctypes_available():
        pytest.skip("CUDA ctypes backend is not available")

    old_path = tmp_path / "old.gds"
    new_path = tmp_path / "new.gds"
    old_lib = gdstk.Library(unit=1e-6, precision=1e-9)
    old_top = old_lib.new_cell("top")
    old_top.add(gdstk.Polygon([(0, 0), (2, 0), (2, 1), (0, 1)], layer=2, datatype=0))
    old_lib.write_gds(old_path)

    new_lib = gdstk.Library(unit=1e-6, precision=1e-9)
    new_top = new_lib.new_cell("top")
    new_top.add(gdstk.Polygon([(2, 1), (2, 0), (0, 0), (0, 1)], layer=2, datatype=0))
    new_lib.write_gds(new_path)

    old_extraction = extract_gds_polygons_native(old_path)
    new_extraction = extract_gds_polygons_native(new_path)
    delta = run_cuda_polygon_record_delta(old_extraction.buffer, new_extraction.buffer)

    assert delta.record_count == 0
    assert delta.matched_record_count == 1
