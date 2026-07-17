from __future__ import annotations

import json
from pathlib import Path

import gdstk
import numpy as np
import gdsdiff.api as api_module
import pytest

from gdsdiff import (
    AllowedRegion,
    CompareConfig,
    CompareStatus,
    DifferenceDirection,
    FailurePolicy,
    compare_gds_files,
)
from gdsdiff.backends.cuda_ctypes import cuda_ctypes_available
from gdsdiff.config import LayerSpec
from gdsdiff.exact_oracle import DiffFragment, ExactOracleResult, LayerExactDiff


def _write_pair(tmp_path, *, shifted=True):
    old_path = tmp_path / "old.gds"
    new_path = tmp_path / "new.gds"
    old = gdstk.Library(unit=1e-6, precision=1e-6)
    new = gdstk.Library(unit=1e-6, precision=1e-6)
    old_top = old.new_cell("top")
    new_top = new.new_cell("top")
    old_top.add(gdstk.rectangle((0, 0), (10, 10), layer=1, datatype=0))
    if shifted:
        new_top.add(gdstk.rectangle((2, 0), (12, 10), layer=1, datatype=0))
    else:
        new_top.add(gdstk.rectangle((0, 0), (10, 10), layer=1, datatype=0))
    old.write_gds(old_path)
    new.write_gds(new_path)
    return old_path, new_path


def _write_count_changed_pair(tmp_path):
    old_path = tmp_path / "old_count.gds"
    new_path = tmp_path / "new_count.gds"
    old = gdstk.Library(unit=1e-6, precision=1e-6)
    new = gdstk.Library(unit=1e-6, precision=1e-6)
    old_top = old.new_cell("top")
    new_top = new.new_cell("top")
    old_top.add(gdstk.rectangle((0, 0), (10, 10), layer=1, datatype=0))
    new_top.add(gdstk.rectangle((0, 0), (10, 10), layer=1, datatype=0))
    new_top.add(gdstk.rectangle((20, 0), (30, 10), layer=1, datatype=0))
    old.write_gds(old_path)
    new.write_gds(new_path)
    return old_path, new_path


def _write_unsupported_polygon_pair(tmp_path):
    old_path = tmp_path / "old_unsupported.gds"
    new_path = tmp_path / "new_unsupported.gds"
    old = gdstk.Library(unit=1e-6, precision=1e-6)
    new = gdstk.Library(unit=1e-6, precision=1e-6)
    old_top = old.new_cell("top")
    new_top = new.new_cell("top")
    old_top.add(gdstk.rectangle((0, 0), (10, 10), layer=1, datatype=0))
    new_top.add(gdstk.Polygon([(5, 0), (10, 5), (5, 10), (0, 5)], layer=1, datatype=0))
    old.write_gds(old_path)
    new.write_gds(new_path)
    return old_path, new_path


def _write_polygon_case(tmp_path, name: str, old_polygons, new_polygons):
    old_path = tmp_path / f"{name}_old.gds"
    new_path = tmp_path / f"{name}_new.gds"
    old = gdstk.Library(unit=1e-6, precision=1e-6)
    new = gdstk.Library(unit=1e-6, precision=1e-6)
    old_top = old.new_cell("top")
    new_top = new.new_cell("top")
    for points, layer, datatype in old_polygons:
        old_top.add(gdstk.Polygon(points, layer=layer, datatype=datatype))
    for points, layer, datatype in new_polygons:
        new_top.add(gdstk.Polygon(points, layer=layer, datatype=datatype))
    old.write_gds(old_path)
    new.write_gds(new_path)
    return old_path, new_path


def _run_gpu_polygon_case(old_path, new_path, markdown_path):
    if not cuda_ctypes_available():
        pytest.skip("CUDA ctypes backend is not available")
    result = compare_gds_files(
        CompareConfig(old_path, new_path, backend="cuda", evidence_engine="gpu-polygon"),
        exact_diff_markdown_path=markdown_path,
    )
    markdown = markdown_path.read_text(encoding="utf-8")
    assert len(markdown.splitlines()) <= 200
    assert "## Agent Minimal Read" in markdown
    assert "## Polygon Components" in markdown
    assert "## Input Polygon Inventory Delta" in markdown
    assert "points_sha256" in markdown
    payload = result.report.to_json_dict()
    assert payload["artifacts"]["vertex_sidecar"]
    sidecar_path = Path(payload["artifacts"]["vertex_sidecar"])
    assert sidecar_path.exists()
    sidecar_records = [json.loads(line) for line in sidecar_path.read_text(encoding="utf-8").splitlines()]
    assert all(record["vertices_dbu"] for record in sidecar_records)
    return result, result.report.to_json_dict(), markdown


def test_public_api_equivalent_pair_returns_zero_exit(tmp_path):
    old_path, new_path = _write_pair(tmp_path, shifted=False)

    result = compare_gds_files(CompareConfig(old_path, new_path, tile_size=10))

    assert result.status == CompareStatus.EQUIVALENT
    assert result.exit_code == 0
    assert result.report.to_json_dict()["status"] == "equivalent"


def test_cpu_audit_summary_normalizes_collinear_loop_vertices():
    layer = LayerSpec(1, 0)
    gpu_fragment = DiffFragment(
        points=((4, 4), (0, 4), (0, 2), (1, 2), (1, 3), (3, 3), (3, 2), (0, 2), (0, 0), (4, 0)),
        bbox=(0, 0, 4, 4),
        twice_area=20,
    )
    cpu_fragment = DiffFragment(
        points=((4, 0), (4, 4), (0, 4), (0, 2), (1, 2), (1, 3), (3, 3), (3, 2), (1, 2), (0, 2), (0, 0)),
        bbox=(0, 0, 4, 4),
        twice_area=20,
    )
    gpu_result = ExactOracleResult((LayerExactDiff(layer=layer, old_minus_new=(gpu_fragment,), xor=(gpu_fragment,)),))
    cpu_result = ExactOracleResult((LayerExactDiff(layer=layer, old_minus_new=(cpu_fragment,), xor=(cpu_fragment,)),))

    summary = api_module._exact_result_audit_summary(gpu_result, cpu_result)

    assert summary["match"] is True
    assert summary["mismatch_count"] == 0


def test_gpu_polygon_metal_source_edge_report_contract(tmp_path, monkeypatch):
    old_path, new_path = _write_pair(tmp_path)
    exact_diff_path = tmp_path / "exact_diff.md"
    import gdsdiff.backends.metal_ctypes as metal_ctypes

    monkeypatch.setattr(
        api_module,
        "_select_backend",
        lambda config, exact=False: {
            "requested": "metal",
            "name": "metal",
            "cuda_available": False,
            "cuda_ctypes_available": False,
            "cuda_extension_available": False,
            "metal_available": True,
            "metal_ctypes_available": True,
            "core_extension_available": False,
            "nvcc_available": False,
            "fallback_reason": None,
        },
    )
    monkeypatch.setattr(api_module, "_native_buffers_rectangle_coverage_eligible", lambda *buffers: False)
    monkeypatch.setattr(metal_ctypes, "release_metal_device_memory", lambda: None)

    def fake_record_delta(old_buffer, new_buffer):
        old_layer = old_buffer.layer_table.layers[int(old_buffer.layer_ids[0])]
        new_layer = new_buffer.layer_table.layers[int(new_buffer.layer_ids[0])]
        return metal_ctypes.MetalPolygonRecordDelta(
            layers=np.asarray([old_layer.layer, new_layer.layer], dtype=np.int32),
            datatypes=np.asarray([old_layer.datatype, new_layer.datatype], dtype=np.int32),
            directions=np.asarray([1, 2], dtype=np.uint8),
            counts=np.asarray([1, 1], dtype=np.uint64),
            bboxes=np.asarray([old_buffer.bboxes[0], new_buffer.bboxes[0]], dtype=np.int64),
            vertex_counts=np.asarray(
                [
                    old_buffer.polygon_offsets[1] - old_buffer.polygon_offsets[0],
                    new_buffer.polygon_offsets[1] - new_buffer.polygon_offsets[0],
                ],
                dtype=np.uint64,
            ),
            flags=np.asarray([old_buffer.flags[0], new_buffer.flags[0]], dtype=np.uint32),
            hashes=np.zeros((2, 2), dtype=np.uint64),
            representative_indices=np.asarray([0, 0], dtype=np.uint64),
            old_input_count=old_buffer.polygon_count,
            new_input_count=new_buffer.polygon_count,
            matched_record_count=0,
            reduced_key_count=2,
            metal_kernel_count=2,
            metal_event_elapsed_s=0.001,
            transferred_bytes=256,
        )

    monkeypatch.setattr(metal_ctypes, "run_metal_polygon_record_delta_from_hashes", fake_record_delta)

    result = compare_gds_files(
        CompareConfig(old_path, new_path, backend="metal", evidence_engine="gpu-polygon"),
        exact_diff_markdown_path=exact_diff_path,
    )
    payload = result.report.to_json_dict()
    component_area = payload["exact_fallback"]["gpu_polygon_component_area"]

    assert result.status == CompareStatus.DIFFERENCES_FOUND
    assert payload["backend"]["name"] == "metal"
    assert payload["backend"]["gpu_polygon_available"] is True
    assert payload["backend"]["gpu_record_delta_available"] is True
    assert payload["backend"]["metal_kernel_count"] >= 3
    assert payload["exact_fallback"]["mode"] == "gpu-polygon-metal-source-edge-records"
    assert payload["exact_fallback"]["engine_status"] == "source-edge-boundary-records-complete"
    assert payload["exact_fallback"]["gpu_record_delta_accelerator"] == "metal"
    assert component_area["gpu_trapezoid_fragments_available"] is False
    assert component_area["gpu_source_edge_records_available"] is True
    assert component_area["merged_boundary_loop_records_available"] is True
    assert payload["exact_verification_complete"] is True
    assert "gpu-polygon METAL source-edge diff" in exact_diff_path.read_text(encoding="utf-8")


def test_public_api_unexpected_difference_returns_failure_exit(tmp_path):
    old_path, new_path = _write_pair(tmp_path)

    result = compare_gds_files(CompareConfig(old_path, new_path, backend="cpu", tile_size=10))

    assert result.status == CompareStatus.UNEXPECTED_DIFFERENCES
    assert result.exit_code == 1
    assert result.policy_result.to_json()["unexpected_area_dbu2"] == "40"


def test_public_api_allowed_difference_respects_failure_policy(tmp_path):
    old_path, new_path = _write_pair(tmp_path)
    config = CompareConfig(
        old_path,
        new_path,
        tile_size=10,
        allowed_regions=(AllowedRegion(layers=((1, 0),), direction=DifferenceDirection.BOTH, bbox=(0, 0, 12, 10)),),
    )

    result = compare_gds_files(config)
    strict_result = compare_gds_files(
        CompareConfig(
            old_path,
            new_path,
            tile_size=10,
            allowed_regions=(AllowedRegion(layers=((1, 0),), direction=DifferenceDirection.BOTH, bbox=(0, 0, 12, 10)),),
            failure_policy=FailurePolicy.FAIL_ON_ANY_DIFF,
        )
    )

    assert result.status == CompareStatus.EXPECTED_DIFFERENCES_ONLY
    assert result.exit_code == 0
    assert strict_result.exit_code == 1


def test_public_api_exact_mode_matches_cpu_prefilter_status_and_areas(tmp_path):
    old_path, new_path = _write_pair(tmp_path)
    config = CompareConfig(old_path, new_path, backend="cpu", tile_size=10)

    accelerated = compare_gds_files(config)
    exact = compare_gds_files(config, exact=True)

    assert accelerated.status == exact.status
    assert accelerated.policy_result.to_json()["unexpected_area_dbu2"] == exact.policy_result.to_json()["unexpected_area_dbu2"]
    assert exact.report.backend["exact_mode"] is True


def test_windowed_report_does_not_claim_full_exact_verification(tmp_path, monkeypatch):
    old_path, new_path = _write_pair(tmp_path)
    monkeypatch.setattr(api_module, "_should_use_layer_exact", lambda *args, **kwargs: False)

    result = compare_gds_files(CompareConfig(old_path, new_path, backend="cpu", tile_size=10))
    payload = result.report.to_json_dict()

    assert payload["exact_fallback"]["mode"] == "windowed"
    assert payload["exact_verification_complete"] is False
    assert payload["warnings"]
    assert {layer["exact_status"] for layer in payload["layers"]} == {"windowed"}


def test_exact_report_claims_full_exact_verification(tmp_path):
    old_path, new_path = _write_pair(tmp_path)

    result = compare_gds_files(CompareConfig(old_path, new_path, tile_size=10), exact=True)
    payload = result.report.to_json_dict()

    assert payload["exact_fallback"]["mode"] == "full"
    assert payload["exact_verification_complete"] is True
    assert {layer["exact_status"] for layer in payload["layers"]} == {"complete"}


def test_exact_component_rect_fallback_reports_backend_stats(tmp_path):
    old_path, new_path = _write_pair(tmp_path)

    result = compare_gds_files(
        CompareConfig(old_path, new_path, tile_size=10),
        exact=True,
        exact_component_backend="rect-fallback",
    )
    payload = result.report.to_json_dict()
    backend = payload["exact_fallback"]["exact_component_backend"]

    assert result.status == CompareStatus.UNEXPECTED_DIFFERENCES
    assert payload["exact_verification_complete"] is True
    assert backend["name"] == "manhattan-rect-exact-component"
    assert backend["component_partition"] is True
    assert backend["stats"]["accelerated_tasks"] == 1
    assert backend["stats"]["fallback_tasks"] == 0
    assert backend["stats"]["unsupported_tasks"] == 0
    assert result.policy_result.to_json()["unexpected_area_dbu2"] == "40"


def test_exact_component_rect_backend_fails_closed_on_unsupported_geometry(tmp_path):
    old_path, new_path = _write_unsupported_polygon_pair(tmp_path)

    try:
        compare_gds_files(
            CompareConfig(old_path, new_path, tile_size=10),
            exact=True,
            exact_component_backend="rect",
        )
    except Exception as exc:
        assert "rectangle" in str(exc) or "component" in str(exc)
    else:
        raise AssertionError("strict rectangle exact backend should reject unsupported geometry")


def test_exact_component_cuda_rect_reports_backend_stats(tmp_path):
    if not cuda_ctypes_available():
        pytest.skip("CUDA ctypes backend is not available")
    old_path, new_path = _write_pair(tmp_path)

    result = compare_gds_files(
        CompareConfig(old_path, new_path, tile_size=10),
        exact=True,
        exact_component_backend="cuda-rect",
    )
    payload = result.report.to_json_dict()
    backend = payload["exact_fallback"]["exact_component_backend"]

    assert result.status == CompareStatus.UNEXPECTED_DIFFERENCES
    assert backend["name"] == "cuda-manhattan-rect-exact-component"
    assert backend["requires_gpu"] is True
    assert backend["stats"]["accelerated_tasks"] == 0
    assert backend["stats"]["cuda_rect_set_tasks"] == 1
    assert backend["stats"]["fallback_tasks"] == 0
    assert backend["stats"]["unsupported_tasks"] == 0
    assert result.policy_result.to_json()["unexpected_area_dbu2"] == "40"


def test_accelerated_default_uses_native_geometry_guard_with_exact_verification(tmp_path, monkeypatch):
    old_path, new_path = _write_pair(tmp_path)
    exact_diff_path = tmp_path / "exact_diff.md"
    monkeypatch.setattr(
        api_module,
        "_select_backend",
        lambda config, exact=False: {
            "requested": "cuda",
            "name": "cuda",
            "cuda_available": True,
            "cuda_ctypes_available": True,
            "cuda_extension_available": False,
            "core_extension_available": False,
            "nvcc_available": False,
            "fallback_reason": None,
        },
    )

    result = compare_gds_files(
        CompareConfig(old_path, new_path, backend="cuda", tile_size=10),
        exact_diff_markdown_path=exact_diff_path,
        verify_exact_after_fast=True,
    )
    payload = result.report.to_json_dict()
    markdown = exact_diff_path.read_text(encoding="utf-8")

    assert result.status == CompareStatus.UNEXPECTED_DIFFERENCES
    assert result.policy_result.to_json()["unexpected_area_dbu2"] == "40"
    assert payload["exact_verification_complete"] is True
    assert payload["exact_fallback"]["mode"] == "native-geometry-fast-plus-full-exact"
    assert payload["exact_fallback"]["fast_status"] == "unexpected-differences"
    assert payload["backend"]["native_geometry_guard"] is True
    assert payload["backend"]["exact_verification_after_fast"] is True
    assert payload["timings"]["fast_status_s"] >= 0
    assert payload["timings"]["exact_verification_s"] >= 0
    assert {layer["exact_status"] for layer in payload["layers"]} == {"complete"}
    assert "- exact_verification_complete: `true`" in markdown
    assert "## Agent Minimal Read" in markdown
    assert "| 1 | 0 |" in markdown
    assert not any(line.startswith("| 0 | 0 |") for line in markdown.splitlines())
    assert "points_sha256" in markdown
    assert "vertices_preview_dbu" in markdown


def test_cuda_can_force_geometry_prefilter_path(tmp_path, monkeypatch):
    old_path, new_path = _write_pair(tmp_path)
    monkeypatch.setattr(
        api_module,
        "_select_backend",
        lambda config, exact=False: {
            "requested": "cuda",
            "name": "cuda",
            "cuda_available": True,
            "cuda_ctypes_available": True,
            "cuda_extension_available": False,
            "core_extension_available": False,
            "nvcc_available": False,
            "fallback_reason": None,
        },
    )
    monkeypatch.setattr(api_module, "_try_accelerated_byte_identity_fast_path", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("byte identity shortcut used")))
    monkeypatch.setattr(api_module, "_try_native_geometry_guard_path", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("native geometry shortcut used")))
    monkeypatch.setattr(api_module, "_try_cuda_hierarchy_fast_path", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("hierarchy shortcut used")))
    monkeypatch.setattr(
        api_module,
        "_run_tile_prefilter",
        lambda backend_info, old_buffer, old_ids, new_buffer, new_ids, tile_config: api_module.run_cpu_tile_prefilter(
            old_buffer,
            old_ids,
            new_buffer,
            new_ids,
            tile_config,
        ),
    )

    result = compare_gds_files(
        CompareConfig(old_path, new_path, backend="cuda", tile_size=10),
        use_fast_shortcuts=False,
    )
    payload = result.report.to_json_dict()

    assert result.status == CompareStatus.UNEXPECTED_DIFFERENCES
    assert payload["backend"]["name"] == "cuda"
    assert payload["backend"]["fast_shortcuts_enabled"] is False
    assert payload["exact_fallback"]["mode"] == "windowed"
    assert payload["exact_verification_complete"] is False


def test_cuda_forced_geometry_can_attach_full_exact_verification(tmp_path, monkeypatch):
    old_path, new_path = _write_pair(tmp_path)
    monkeypatch.setattr(
        api_module,
        "_select_backend",
        lambda config, exact=False: {
            "requested": "cuda",
            "name": "cuda",
            "cuda_available": True,
            "cuda_ctypes_available": True,
            "cuda_extension_available": False,
            "core_extension_available": False,
            "nvcc_available": False,
            "fallback_reason": None,
        },
    )
    monkeypatch.setattr(
        api_module,
        "_run_tile_prefilter",
        lambda backend_info, old_buffer, old_ids, new_buffer, new_ids, tile_config: api_module.run_cpu_tile_prefilter(
            old_buffer,
            old_ids,
            new_buffer,
            new_ids,
            tile_config,
        ),
    )

    result = compare_gds_files(
        CompareConfig(old_path, new_path, backend="cuda", tile_size=10),
        use_fast_shortcuts=False,
        verify_exact_after_fast=True,
    )
    payload = result.report.to_json_dict()

    assert result.status == CompareStatus.UNEXPECTED_DIFFERENCES
    assert payload["exact_verification_complete"] is True
    assert payload["exact_fallback"]["mode"] == "windowed-plus-full-exact"
    assert payload["exact_fallback"]["fast_status"] == "unexpected-differences"
    assert payload["timings"]["geometry_status_s"] >= 0
    assert payload["timings"]["full_exact_oracle_s"] >= 0


def test_forced_geometry_can_reuse_cached_preprocessing(tmp_path, monkeypatch):
    old_path, new_path = _write_pair(tmp_path)
    cache_dir = tmp_path / "geometry_cache"
    monkeypatch.setattr(
        api_module,
        "_select_backend",
        lambda config, exact=False: {
            "requested": "cuda",
            "name": "cuda",
            "cuda_available": True,
            "cuda_ctypes_available": True,
            "cuda_extension_available": False,
            "core_extension_available": False,
            "nvcc_available": False,
            "fallback_reason": None,
        },
    )
    monkeypatch.setattr(
        api_module,
        "_run_tile_prefilter",
        lambda backend_info, old_buffer, old_ids, new_buffer, new_ids, tile_config: api_module.run_cpu_tile_prefilter(
            old_buffer,
            old_ids,
            new_buffer,
            new_ids,
            tile_config,
        ),
    )

    config = CompareConfig(old_path, new_path, backend="cuda", tile_size=10, geometry_cache_dir=cache_dir)
    first = compare_gds_files(config, use_fast_shortcuts=False)
    second = compare_gds_files(config, use_fast_shortcuts=False)
    first_payload = first.report.to_json_dict()
    second_payload = second.report.to_json_dict()

    assert first.status == second.status == CompareStatus.UNEXPECTED_DIFFERENCES
    assert first_payload["backend"]["geometry_cache_enabled"] is True
    assert second_payload["timings"]["old_geometry_cache_hit"] is True
    assert second_payload["timings"]["new_geometry_cache_hit"] is True
    assert second_payload["timings"]["extract_s"] == 0.0
    assert second_payload["exact_fallback"]["mode"] == "windowed"


def test_gpu_polygon_can_reuse_native_polygon_cache(tmp_path):
    if not cuda_ctypes_available():
        pytest.skip("CUDA ctypes backend is not available")
    old_path, new_path = _write_pair(tmp_path)
    cache_dir = tmp_path / "native_polygon_cache"
    config = CompareConfig(
        old_path,
        new_path,
        backend="cuda",
        evidence_engine="gpu-polygon",
        geometry_cache_dir=cache_dir,
        tile_size=10,
    )

    first = compare_gds_files(config, use_fast_shortcuts=False)
    second = compare_gds_files(config, use_fast_shortcuts=False)
    first_payload = first.report.to_json_dict()
    second_payload = second.report.to_json_dict()

    assert first.status == second.status == CompareStatus.DIFFERENCES_FOUND
    assert first_payload["timings"]["old_native_polygon_cache_hit"] is False
    assert first_payload["timings"]["new_native_polygon_cache_hit"] is False
    assert second_payload["timings"]["old_native_polygon_cache_hit"] is True
    assert second_payload["timings"]["new_native_polygon_cache_hit"] is True
    assert second_payload["backend"]["evidence_engine_selected"] == "gpu-polygon"
    assert second_payload["exact_fallback"]["mode"] == "gpu-polygon-cuda-rect-coverage"


def test_gpu_polygon_reports_complete_layer_removal_in_compact_diff(tmp_path):
    old_path, new_path = _write_polygon_case(
        tmp_path,
        "layer_removed",
        old_polygons=(
            (((0, 0), (3, 0), (3, 2), (0, 2)), 9, 0),
            (((10, 0), (12, 0), (12, 2), (10, 2)), 1, 0),
        ),
        new_polygons=(
            (((10, 0), (12, 0), (12, 2), (10, 2)), 1, 0),
        ),
    )

    result, payload, markdown = _run_gpu_polygon_case(old_path, new_path, tmp_path / "layer_removed_diff.md")

    assert result.status == CompareStatus.DIFFERENCES_FOUND
    assert payload["status"] == "differences-found"
    assert payload["evidence_status"] == "differences-found"
    assert payload["exact_verification_complete"] is True
    assert payload["unexpected_summary"]["unexpected_area_dbu2"] == "6"
    assert payload["inventory_delta_count"] == 1
    assert any(layer["layer"] == 9 and layer["datatype"] == 0 for layer in payload["layers"])
    assert "`coverage_before_only` | 9 | 0" in markdown
    assert "`inventory_before_only` | 9 | 0" in markdown
    assert "[0,0,3,2]" in markdown


def test_gpu_polygon_reports_missing_vertex_polygon_change_in_compact_diff(tmp_path):
    old_path, new_path = _write_polygon_case(
        tmp_path,
        "missing_vertex",
        old_polygons=(
            (((0, 0), (4, 0), (4, 4), (2, 5), (0, 4)), 7, 0),
        ),
        new_polygons=(
            (((0, 0), (4, 0), (4, 4), (0, 4)), 7, 0),
        ),
    )

    result, payload, markdown = _run_gpu_polygon_case(old_path, new_path, tmp_path / "missing_vertex_diff.md")

    assert result.status == CompareStatus.DIFFERENCES_FOUND
    assert payload["exact_verification_complete"] is True
    assert payload["exact_fallback"]["mode"] == "gpu-polygon-cuda-source-edge-records"
    assert payload["unexpected_summary"]["unexpected_area_dbu2"] == "2"
    assert payload["inventory_delta_count"] == 2
    assert "`coverage_before_only` | 7 | 0" in markdown
    assert "`inventory_before_only` | 7 | 0" in markdown
    assert "`inventory_after_only` | 7 | 0" in markdown
    assert "[0,4,4,5]" in markdown
    assert "[2,5]" in markdown


def test_gpu_polygon_reports_one_dbu_area_change_in_compact_diff(tmp_path):
    old_path, new_path = _write_polygon_case(
        tmp_path,
        "one_dbu_area",
        old_polygons=(
            (((0, 0), (2, 0), (2, 2), (0, 2)), 5, 0),
        ),
        new_polygons=(
            (((0, 0), (2, 0), (2, 2), (0, 1)), 5, 0),
        ),
    )

    result, payload, markdown = _run_gpu_polygon_case(old_path, new_path, tmp_path / "one_dbu_area_diff.md")

    assert result.status == CompareStatus.DIFFERENCES_FOUND
    assert payload["exact_verification_complete"] is True
    assert payload["exact_fallback"]["mode"] == "gpu-polygon-cuda-source-edge-records"
    assert payload["unexpected_summary"]["unexpected_area_dbu2"] == "1"
    assert payload["inventory_delta_count"] == 2
    assert "coverage_xor_area_dbu2: `1`" in markdown
    assert "`coverage_before_only` | 5 | 0" in markdown
    assert "[0,1,2,2]" in markdown


def test_gpu_polygon_reports_same_coverage_split_as_inventory_difference(tmp_path):
    old_path, new_path = _write_polygon_case(
        tmp_path,
        "same_coverage_split",
        old_polygons=(
            (((0, 0), (10, 0), (10, 10), (0, 10)), 8, 0),
        ),
        new_polygons=(
            (((0, 0), (5, 0), (5, 10), (0, 10)), 8, 0),
            (((5, 0), (10, 0), (10, 10), (5, 10)), 8, 0),
        ),
    )

    result, payload, markdown = _run_gpu_polygon_case(old_path, new_path, tmp_path / "same_coverage_split_diff.md")

    assert result.status == CompareStatus.DIFFERENCES_FOUND
    assert payload["evidence_status"] == "differences-found"
    assert payload["coverage_changed"] is False
    assert payload["inventory_changed"] is True
    assert payload["coverage_equal_but_inventory_changed"] is True
    assert payload["coverage_component_count"] == 0
    assert payload["inventory_delta_count"] == 3
    assert "No polygon-level differences were found." in markdown
    assert "- coverage_equal_but_inventory_changed: `true`" in markdown
    assert "`inventory_before_only` | 8 | 0" in markdown
    assert "`inventory_after_only` | 8 | 0" in markdown


def test_structural_guard_fast_path_attaches_exact_verification(tmp_path, monkeypatch):
    old_path, new_path = _write_count_changed_pair(tmp_path)
    monkeypatch.setattr(
        api_module,
        "_select_backend",
        lambda config, exact=False: {
            "requested": "cuda",
            "name": "cuda",
            "cuda_available": True,
            "cuda_ctypes_available": True,
            "cuda_extension_available": False,
            "core_extension_available": False,
            "nvcc_available": False,
            "fallback_reason": None,
        },
    )

    result = compare_gds_files(
        CompareConfig(old_path, new_path, backend="cuda", tile_size=10),
        use_structural_guard=True,
        verify_exact_after_fast=True,
    )
    payload = result.report.to_json_dict()

    assert result.status == CompareStatus.UNEXPECTED_DIFFERENCES
    assert payload["backend"]["structural_guard"] is True
    assert payload["exact_fallback"]["mode"] == "structural-count-fast-plus-full-exact"
    assert payload["exact_fallback"]["fast_status"] == "unexpected-differences"
    assert payload["exact_fallback"]["structural_mismatch_kind"] == "count"
    assert payload["exact_verification_complete"] is True
    assert payload["timings"]["structural_guard_s"] >= 0


def test_structural_guard_catches_equal_count_structural_change(tmp_path, monkeypatch):
    old_path, new_path = _write_pair(tmp_path, shifted=True)
    monkeypatch.setattr(
        api_module,
        "_select_backend",
        lambda config, exact=False: {
            "requested": "cuda",
            "name": "cuda",
            "cuda_available": True,
            "cuda_ctypes_available": True,
            "cuda_extension_available": False,
            "core_extension_available": False,
            "nvcc_available": False,
            "fallback_reason": None,
        },
    )

    result = compare_gds_files(
        CompareConfig(old_path, new_path, backend="cuda", tile_size=10),
        use_structural_guard=True,
        verify_exact_after_fast=True,
    )
    payload = result.report.to_json_dict()

    assert result.status == CompareStatus.UNEXPECTED_DIFFERENCES
    assert payload["backend"]["structural_guard"] is True
    assert payload["exact_fallback"]["mode"] == "structural-count-fast-plus-full-exact"
    assert payload["exact_fallback"]["structural_mismatch_kind"] == "hash"
    assert payload["exact_fallback"]["old_flattened_shape_count"] == payload["exact_fallback"]["new_flattened_shape_count"]
    assert payload["exact_fallback"]["old_structural_hash"] != payload["exact_fallback"]["new_structural_hash"]


def test_structural_guard_falls_back_when_structure_matches(tmp_path, monkeypatch):
    old_path, new_path = _write_pair(tmp_path, shifted=False)
    monkeypatch.setattr(
        api_module,
        "_select_backend",
        lambda config, exact=False: {
            "requested": "cuda",
            "name": "cuda",
            "cuda_available": True,
            "cuda_ctypes_available": True,
            "cuda_extension_available": False,
            "core_extension_available": False,
            "nvcc_available": False,
            "fallback_reason": None,
        },
    )

    result = compare_gds_files(
        CompareConfig(old_path, new_path, backend="cuda", tile_size=10),
        use_structural_guard=True,
        use_fast_shortcuts=False,
    )
    payload = result.report.to_json_dict()

    assert result.status == CompareStatus.EQUIVALENT
    assert "structural_guard" not in payload["backend"]


def test_native_geometry_guard_fast_path_attaches_exact_verification(tmp_path, monkeypatch):
    old_path, new_path = _write_count_changed_pair(tmp_path)
    monkeypatch.setattr(
        api_module,
        "_select_backend",
        lambda config, exact=False: {
            "requested": "cuda",
            "name": "cuda",
            "cuda_available": True,
            "cuda_ctypes_available": True,
            "cuda_extension_available": False,
            "core_extension_available": False,
            "nvcc_available": False,
            "fallback_reason": None,
        },
    )

    result = compare_gds_files(
        CompareConfig(old_path, new_path, backend="cuda", tile_size=10),
        use_native_geometry_guard=True,
        verify_exact_after_fast=True,
    )
    payload = result.report.to_json_dict()

    assert result.status == CompareStatus.UNEXPECTED_DIFFERENCES
    assert payload["backend"]["native_geometry_guard"] is True
    assert payload["exact_fallback"]["mode"] == "native-geometry-fast-plus-full-exact"
    assert payload["exact_fallback"]["fast_status"] == "unexpected-differences"
    assert payload["exact_fallback"]["geometry_mismatch_kind"] == "count"
    assert payload["exact_verification_complete"] is True
    assert payload["timings"]["native_geometry_guard_s"] >= 0


def test_native_geometry_guard_catches_equal_count_geometry_change(tmp_path, monkeypatch):
    old_path, new_path = _write_pair(tmp_path, shifted=True)
    monkeypatch.setattr(
        api_module,
        "_select_backend",
        lambda config, exact=False: {
            "requested": "cuda",
            "name": "cuda",
            "cuda_available": True,
            "cuda_ctypes_available": True,
            "cuda_extension_available": False,
            "core_extension_available": False,
            "nvcc_available": False,
            "fallback_reason": None,
        },
    )

    result = compare_gds_files(
        CompareConfig(old_path, new_path, backend="cuda", tile_size=10),
        use_native_geometry_guard=True,
        verify_exact_after_fast=True,
    )
    payload = result.report.to_json_dict()

    assert result.status == CompareStatus.UNEXPECTED_DIFFERENCES
    assert payload["backend"]["native_geometry_guard"] is True
    assert payload["exact_fallback"]["mode"] == "native-geometry-fast-plus-full-exact"
    assert payload["exact_fallback"]["old_flattened_shape_count"] == payload["exact_fallback"]["new_flattened_shape_count"]
    assert payload["exact_fallback"]["old_geometry_hash"] != payload["exact_fallback"]["new_geometry_hash"]
    assert payload["exact_fallback"]["geometry_mismatch_kind"] in {"layer-summary", "hash"}


def test_native_geometry_guard_falls_back_when_geometry_matches(tmp_path, monkeypatch):
    old_path, new_path = _write_pair(tmp_path, shifted=False)
    monkeypatch.setattr(
        api_module,
        "_select_backend",
        lambda config, exact=False: {
            "requested": "cuda",
            "name": "cuda",
            "cuda_available": True,
            "cuda_ctypes_available": True,
            "cuda_extension_available": False,
            "core_extension_available": False,
            "nvcc_available": False,
            "fallback_reason": None,
        },
    )

    result = compare_gds_files(
        CompareConfig(old_path, new_path, backend="cuda", tile_size=10),
        use_native_geometry_guard=True,
        use_fast_shortcuts=False,
    )
    payload = result.report.to_json_dict()

    assert result.status == CompareStatus.EQUIVALENT
    assert "native_geometry_guard" not in payload["backend"]
