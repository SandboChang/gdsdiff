from __future__ import annotations

from pathlib import Path

import gdstk
import pytest

from gds_tools_tk.gpu_compare.backends.cuda_ctypes import cuda_ctypes_available
from gds_tools_tk.gpu_compare.backends.polygon_extract_ctypes import extract_gds_polygons_native
from gds_tools_tk.gpu_compare.config import LayerSpec
from gds_tools_tk.gpu_compare.exact_oracle import run_full_exact_oracle
from gds_tools_tk.gpu_compare.extract import extract_gds_polygons
from gds_tools_tk.gpu_compare.rect_coverage import RectCoverageUnsupported, run_cuda_rectangle_coverage_diff


def test_cuda_rectangle_coverage_diff_emits_directional_fragments(tmp_path):
    if not cuda_ctypes_available():
        pytest.skip("CUDA ctypes backend is not available")

    old_path = tmp_path / "old.gds"
    new_path = tmp_path / "new.gds"
    old_lib = gdstk.Library(unit=1e-6, precision=1e-9)
    old_top = old_lib.new_cell("top")
    old_top.add(gdstk.rectangle((0, 0), (4, 4), layer=1, datatype=0))
    old_lib.write_gds(old_path)

    new_lib = gdstk.Library(unit=1e-6, precision=1e-9)
    new_top = new_lib.new_cell("top")
    new_top.add(gdstk.rectangle((2, 0), (6, 4), layer=1, datatype=0))
    new_lib.write_gds(new_path)

    old_native = extract_gds_polygons_native(old_path)
    new_native = extract_gds_polygons_native(new_path)
    result = run_cuda_rectangle_coverage_diff(old_native.buffer, new_native.buffer)

    diff = result.exact_result.layers[0]
    assert diff.layer == LayerSpec(1, 0)
    assert [fragment.bbox for fragment in diff.old_minus_new] == [(0, 0, 2000, 4000)]
    assert [fragment.bbox for fragment in diff.new_minus_old] == [(4000, 0, 6000, 4000)]
    assert diff.old_minus_new_twice_area == 16_000_000
    assert diff.new_minus_old_twice_area == 16_000_000
    assert result.stats.cuda_task_count == 1
    assert result.stats.cuda_kernel_count == 1


def test_cuda_rectangle_coverage_diff_rejects_non_rectangles(tmp_path):
    path = tmp_path / "tri.gds"
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    top = lib.new_cell("top")
    top.add(gdstk.Polygon([(0, 0), (1, 0), (0, 1)], layer=1, datatype=0))
    lib.write_gds(path)
    extraction = extract_gds_polygons_native(path)

    with pytest.raises(RectCoverageUnsupported, match="non-rectangle"):
        run_cuda_rectangle_coverage_diff(extraction.buffer, extraction.buffer)


@pytest.mark.parametrize(
    ("name", "old_path", "new_path", "top"),
    [
        (
            "medium",
            Path("references/testGDS/medium/medium_before.gds"),
            Path("references/testGDS/medium/medium_after.gds"),
            "medium_compare_top",
        ),
        (
            "equal_count",
            Path("references/testGDS/equal_count/equal_count_before.gds"),
            Path("references/testGDS/equal_count/equal_count_after.gds"),
            "equal_count_top",
        ),
    ],
)
def test_cuda_rectangle_coverage_matches_gdstk_oracle_area_for_rectangle_fixtures(name, old_path, new_path, top):
    if not cuda_ctypes_available():
        pytest.skip("CUDA ctypes backend is not available")
    if not old_path.exists() or not new_path.exists():
        pytest.skip(f"{name} fixture is not present")

    old_native = extract_gds_polygons_native(old_path, top_name=top)
    new_native = extract_gds_polygons_native(new_path, top_name=top)
    gpu_result = run_cuda_rectangle_coverage_diff(old_native.buffer, new_native.buffer)

    old_reference = extract_gds_polygons(old_path, top_name=top)
    new_reference = extract_gds_polygons(new_path, top_name=top)
    cpu_result = run_full_exact_oracle(old_reference.buffer, new_reference.buffer)

    gpu_by_layer = {diff.layer: diff for diff in gpu_result.exact_result.layers}
    cpu_by_layer = {diff.layer: diff for diff in cpu_result.layers}
    assert set(gpu_by_layer) == set(cpu_by_layer)
    for layer in cpu_by_layer:
        assert gpu_by_layer[layer].old_minus_new_twice_area == cpu_by_layer[layer].old_minus_new_twice_area
        assert gpu_by_layer[layer].new_minus_old_twice_area == cpu_by_layer[layer].new_minus_old_twice_area
        assert gpu_by_layer[layer].xor_twice_area == cpu_by_layer[layer].xor_twice_area
