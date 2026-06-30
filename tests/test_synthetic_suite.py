from __future__ import annotations

import json
from pathlib import Path

import gdstk
import pytest

from scgds.gdsdiff import CompareConfig, compare_gds_files
from scgds.gdsdiff.synthetic_suite import (
    ADVERSARIAL_SYNTHETIC_CASES,
    BASE_SYNTHETIC_CASES,
    SYNTHETIC_CASES,
    build_synthetic_suite,
    validate_synthetic_case_with_cpu_oracle,
)


def test_synthetic_smoke_suite_generates_deterministic_manifests_and_valid_gds(tmp_path):
    output_dir = tmp_path / "smoke"
    records = build_synthetic_suite(output_dir, scale="smoke")
    first_manifest = json.loads((output_dir / "suite_manifest.json").read_text(encoding="utf-8"))

    rebuilt_records = build_synthetic_suite(output_dir, scale="smoke")
    second_manifest = json.loads((output_dir / "suite_manifest.json").read_text(encoding="utf-8"))

    assert first_manifest == second_manifest
    assert len(BASE_SYNTHETIC_CASES) == 16
    assert len(ADVERSARIAL_SYNTHETIC_CASES) == 14
    assert len(records) == len(SYNTHETIC_CASES) == 30
    assert [record.name for record in rebuilt_records] == [record.name for record in records]
    assert {record.name for record in records}.issuperset(
        {
            "adversarial_nested_rings_islands",
            "adversarial_containment_no_edge_crossing",
            "adversarial_one_dbu_sliver_grid",
            "adversarial_hierarchy_transforms",
        }
    )

    for record in records:
        assert record.before_path.exists()
        assert record.after_path.exists()
        assert record.manifest_path.exists()
        assert record.changes_path.exists()
        assert gdstk.read_gds(record.before_path).top_level()
        assert gdstk.read_gds(record.after_path).top_level()
        manifest = json.loads(record.manifest_path.read_text(encoding="utf-8"))
        assert manifest["name"] == record.name
        assert manifest["scale"] == "smoke"
        assert manifest["edits"]
        assert manifest["counts"]["before_polygons"] == record.before_polygon_count
        assert manifest["counts"]["after_polygons"] == record.after_polygon_count


def test_synthetic_smoke_suite_cpu_oracle_matches_declared_edits(tmp_path):
    records = build_synthetic_suite(tmp_path / "smoke", scale="smoke")

    for record in records:
        validation = validate_synthetic_case_with_cpu_oracle(record)
        assert validation["equivalent"] is validation["expected_equivalent"], record.name
        assert validation["missing_layers"] == [], record.name
        assert validation["missing_windows"] == [], record.name


def test_synthetic_benchmark_scale_includes_large_sparse_and_hierarchy_cases(tmp_path):
    records = build_synthetic_suite(
        tmp_path / "benchmark",
        scale="benchmark",
        case_names=("sparse_large_rect_grid", "block_like_hierarchy_stress"),
    )
    by_name = {record.name: record for record in records}

    assert by_name["sparse_large_rect_grid"].before_polygon_count >= 10_000
    assert by_name["block_like_hierarchy_stress"].before_polygon_count >= 15_000
    assert by_name["block_like_hierarchy_stress"].before_polygon_count <= 25_000


def test_synthetic_representative_cases_match_metal_with_cpu_audit_when_available(tmp_path):
    if not _metal_available():
        pytest.skip("Metal ctypes backend is not available")
    records = build_synthetic_suite(
        tmp_path / "metal",
        scale="smoke",
        case_names=(
            "rect_add_remove_shift",
            "circle_breakpoints",
            "hollow_rect_circle",
            "layer_datatype_moves",
            "cell_references_arrays",
        ),
    )

    for record in records:
        result = compare_gds_files(
            CompareConfig(record.before_path, record.after_path, backend="metal", evidence_engine="gpu-polygon"),
            exact_diff_markdown_path=record.before_path.parent / "metal_exact_diff.md",
            audit_cpu_oracle=True,
        )
        payload = result.report.to_json_dict()
        assert payload["backend"]["evidence_engine_selected"] == "gpu-polygon"
        assert payload["exact_fallback"]["audit_cpu_oracle"] is True
        assert payload["exact_fallback"]["cpu_oracle_audit"]["match"] is True


def test_synthetic_adversarial_cases_match_metal_with_cpu_audit_when_available(tmp_path):
    if not _metal_available():
        pytest.skip("Metal ctypes backend is not available")
    records = build_synthetic_suite(
        tmp_path / "metal_adversarial",
        scale="smoke",
        case_names=tuple(definition.name for definition in ADVERSARIAL_SYNTHETIC_CASES),
    )

    for record in records:
        result = compare_gds_files(
            CompareConfig(record.before_path, record.after_path, backend="metal", evidence_engine="gpu-polygon"),
            exact_diff_markdown_path=record.before_path.parent / "metal_exact_diff.md",
            audit_cpu_oracle=True,
        )
        payload = result.report.to_json_dict()
        assert payload["backend"]["evidence_engine_selected"] == "gpu-polygon"
        assert payload["exact_fallback"]["audit_cpu_oracle"] is True
        assert payload["exact_fallback"]["cpu_oracle_audit"]["match"] is True, record.name


def _metal_available() -> bool:
    try:
        from scgds.gdsdiff.backends.metal_ctypes import metal_ctypes_available

        return metal_ctypes_available()
    except Exception:
        return False
