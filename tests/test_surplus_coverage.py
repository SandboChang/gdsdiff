from __future__ import annotations

from pathlib import Path

import gdstk
import pytest

from gdsdiff.backends.cuda_ctypes import cuda_ctypes_available, run_cuda_polygon_record_delta
from gdsdiff.backends.polygon_extract_ctypes import extract_gds_polygons_native
from gdsdiff.config import LayerSpec
from gdsdiff.exact_oracle import run_full_exact_oracle
from gdsdiff.extract import extract_gds_polygons
from gdsdiff.surplus_coverage import (
    run_canonical_surplus_coverage_probe,
    run_record_delta_surplus_coverage_probe,
)


def test_canonical_surplus_probe_proves_independent_triangle_removal(tmp_path):
    old_path = tmp_path / "old.gds"
    new_path = tmp_path / "new.gds"

    old_lib = gdstk.Library(unit=1e-6, precision=1e-9)
    old_top = old_lib.new_cell("top")
    old_top.add(gdstk.Polygon([(0, 0), (3, 0), (0, 2)], layer=2, datatype=1))
    old_lib.write_gds(old_path)

    new_lib = gdstk.Library(unit=1e-6, precision=1e-9)
    new_lib.new_cell("top")
    new_lib.write_gds(new_path)

    old_native = extract_gds_polygons_native(old_path)
    new_native = extract_gds_polygons_native(new_path)
    result = run_canonical_surplus_coverage_probe(old_native.buffer, new_native.buffer)

    assert result.stats.coverage_exact_proven
    assert result.stats.old_surplus_count == 1
    assert result.stats.new_surplus_count == 0
    assert result.stats.component_count == 1
    assert result.stats.overlap_component_count == 0
    assert result.stats.max_fragments_per_component == 1
    assert result.exact_result.layers[0].layer == LayerSpec(2, 1)
    assert result.exact_result.layers[0].old_minus_new_twice_area == 6_000_000
    assert result.exact_result.layers[0].new_minus_old_twice_area == 0

    old_reference = extract_gds_polygons(old_path)
    new_reference = extract_gds_polygons(new_path)
    cpu_result = run_full_exact_oracle(old_reference.buffer, new_reference.buffer)
    assert result.exact_result.layers[0].old_minus_new_twice_area == cpu_result.layers[0].old_minus_new_twice_area


def test_canonical_surplus_probe_does_not_prove_overlapping_surplus(tmp_path):
    old_path = tmp_path / "old.gds"
    new_path = tmp_path / "new.gds"

    old_lib = gdstk.Library(unit=1e-6, precision=1e-9)
    old_top = old_lib.new_cell("top")
    old_top.add(gdstk.rectangle((0, 0), (4, 4), layer=1, datatype=0))
    old_top.add(gdstk.rectangle((2, 0), (6, 4), layer=1, datatype=0))
    old_lib.write_gds(old_path)

    new_lib = gdstk.Library(unit=1e-6, precision=1e-9)
    new_lib.new_cell("top")
    new_lib.write_gds(new_path)

    old_native = extract_gds_polygons_native(old_path)
    new_native = extract_gds_polygons_native(new_path)
    result = run_canonical_surplus_coverage_probe(old_native.buffer, new_native.buffer)

    assert not result.stats.coverage_exact_proven
    assert result.stats.conservative_overlap_layer_count == 1
    assert result.stats.component_count == 1
    assert result.stats.overlap_component_count == 1
    assert result.stats.max_fragments_per_component == 2
    assert result.exact_result.layers[0].old_minus_new_twice_area == 64_000_000

    old_reference = extract_gds_polygons(old_path)
    new_reference = extract_gds_polygons(new_path)
    cpu_result = run_full_exact_oracle(old_reference.buffer, new_reference.buffer)
    assert result.exact_result.layers[0].old_minus_new_twice_area > cpu_result.layers[0].old_minus_new_twice_area


def test_record_delta_surplus_probe_uses_gpu_representative_indices(tmp_path):
    if not cuda_ctypes_available():
        pytest.skip("CUDA ctypes backend is not available")

    old_path = tmp_path / "old.gds"
    new_path = tmp_path / "new.gds"

    old_lib = gdstk.Library(unit=1e-6, precision=1e-9)
    old_top = old_lib.new_cell("top")
    old_top.add(gdstk.Polygon([(0, 0), (3, 0), (0, 2)], layer=2, datatype=1))
    old_lib.write_gds(old_path)

    new_lib = gdstk.Library(unit=1e-6, precision=1e-9)
    new_lib.new_cell("top")
    new_lib.write_gds(new_path)

    old_native = extract_gds_polygons_native(old_path)
    new_native = extract_gds_polygons_native(new_path)
    record_delta = run_cuda_polygon_record_delta(old_native.buffer, new_native.buffer)
    result = run_record_delta_surplus_coverage_probe(old_native.buffer, new_native.buffer, record_delta)

    assert result.stats.old_surplus_count == 1
    assert result.stats.new_surplus_count == 0
    assert not result.stats.coverage_exact_proven
    assert result.stats.component_count == 1
    assert result.stats.max_vertices_per_component == 3
    assert result.exact_result.layers[0].old_minus_new_twice_area == 6_000_000


def test_canonical_surplus_probe_current_nonrectangle_fixtures_are_not_final_exact():
    old_path = Path("references/testGDS/block/block_before.gds")
    new_path = Path("references/testGDS/block/block_after.gds")
    if not old_path.exists() or not new_path.exists():
        pytest.skip("block fixture is not present")

    old_native = extract_gds_polygons_native(old_path, top_name="tdJTWPA_block")
    new_native = extract_gds_polygons_native(new_path, top_name="tdJTWPA_block")
    result = run_canonical_surplus_coverage_probe(old_native.buffer, new_native.buffer)

    assert result.stats.old_surplus_count > 0
    assert result.stats.new_surplus_count > 0
    assert not result.stats.coverage_exact_proven
