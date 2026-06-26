from __future__ import annotations

from pathlib import Path

import pytest

from gds_tools_tk.gdsdiff import LayerSpec, extract_gds_polygons, format_twice_area, run_full_exact_oracle

from .fixtures import fixture_cases, write_fixture_pair


@pytest.mark.parametrize("case", fixture_cases(), ids=lambda case: case.name)
def test_generated_fixture_corpus_matches_oracle_expectations(tmp_path, case):
    old_path, new_path = write_fixture_pair(case, tmp_path)
    old = extract_gds_polygons(old_path, meters_per_dbu="1e-6").buffer
    new = extract_gds_polygons(new_path, meters_per_dbu="1e-6").buffer

    result = run_full_exact_oracle(old, new)

    assert result.equivalent is case.expected_equivalent
    by_layer = {layer_diff.layer: layer_diff for layer_diff in result.layers}
    for layer_key, expected in case.expected_layer_areas.items():
        layer = by_layer[LayerSpec(*layer_key)]
        old_minus, new_minus, xor = expected
        assert format_twice_area(layer.old_minus_new_twice_area) == old_minus
        assert format_twice_area(layer.new_minus_old_twice_area) == new_minus
        assert format_twice_area(layer.xor_twice_area) == xor


def test_chip2_repository_fixture_runs_through_extraction_and_oracle():
    old_path = "references/testGDS/chip/chip2_before.gds"
    new_path = "references/testGDS/chip/chip2_after.gds"
    if not Path(old_path).exists() or not Path(new_path).exists():
        pytest.skip("local reference GDS fixtures are not present")

    old = extract_gds_polygons(old_path, meters_per_dbu="1e-9").buffer
    new = extract_gds_polygons(new_path, meters_per_dbu="1e-9").buffer

    result = run_full_exact_oracle(old, new)

    assert not result.equivalent
    changed_layers = {layer.layer for layer in result.layers if layer.xor_twice_area}
    assert LayerSpec(0, 0) in changed_layers
    assert LayerSpec(21, 0) in changed_layers
