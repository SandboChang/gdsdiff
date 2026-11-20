from __future__ import annotations

import json
import math

import gdstk
import pytest

from gdsdiff import CompareConfig, CompareStatus, compare_gds_files, extract_gds_polygons
from gdsdiff import api as api_module
from gdsdiff import source_edge_diff
from gdsdiff.backends import cuda_ctypes
from gdsdiff.backends.polygon_extract_ctypes import NativePolygonExtraction, PolygonExtractCtypesUnavailable
from gdsdiff.cli import main
from gdsdiff.native import NativeCapabilities, discover_native_capabilities
from gdsdiff.rect_coverage import RectCoverageUnsupported


def _capabilities(*, cuda=False):
    return NativeCapabilities("test", False, False, cuda, False, False, False, ())


def _pair(tmp_path, *, new_precision=1e-9):
    old, new = tmp_path / "old.gds", tmp_path / "new.gds"
    for path, x, precision in ((old, 0, 1e-9), (new, 2, new_precision)):
        lib = gdstk.Library(unit=1e-6, precision=precision)
        lib.new_cell("top").add(gdstk.rectangle((x, 0), (x + 10, 10), layer=1))
        lib.write_gds(path)
    return old, new


@pytest.mark.parametrize("required", [False, True])
def test_gpu_runtime_failure_is_reported_as_cpu_evidence(tmp_path, monkeypatch, capsys, required):
    monkeypatch.setattr(api_module, "discover_native_capabilities", _capabilities)
    old, new = _pair(tmp_path)
    report_path = tmp_path / "report.json"
    command = [str(old), str(new), "--backend", "metal", "--quiet", "--json-report", str(report_path)]
    if required:
        command.append("--require-gpu")

    assert main(command) == (2 if required else 1)
    assert "gpu-runtime: Metal backend runtime is not available" in capsys.readouterr().err
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == ("incomplete-evidence" if required else "differences-found")
    assert report["exact_verification_complete"] is True
    assert report["exact_fallback"]["cpu_evidence_status"] == "differences-found"
    assert report["backend"]["name"] == "cpu"
    assert report["backend"]["gpu_work_performed"] is False


def test_gpu_required_parse_failure_runs_cpu_once_not_as_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(api_module, "discover_native_capabilities", lambda: _capabilities(cuda=True))
    old, new = _pair(tmp_path)
    markdown = tmp_path / "exact.md"
    verify = api_module._run_full_exact_verification
    calls = []

    def fail_native(*args, **kwargs):
        raise PolygonExtractCtypesUnavailable("nonzero reference STRANS flags")

    def track_verify(*args, **kwargs):
        calls.append("cpu")
        return verify(*args, **kwargs)

    monkeypatch.setattr(api_module, "_gpu_polygon_native_inputs", fail_native)
    monkeypatch.setattr(api_module, "_run_full_exact_verification", track_verify)
    result = compare_gds_files(CompareConfig(old, new, require_gpu=True), audit_cpu_oracle=True, exact_diff_markdown_path=markdown)

    report = result.report.to_json_dict()
    assert calls == ["cpu"]
    assert result.exit_code == report["exit_code_category"] == 2
    assert report["exact_verification_complete"] is True
    assert report["exact_fallback"]["audit_cpu_oracle"] is False
    assert report["exact_fallback"]["failures"] == [{"stage": "native-parse", "reason": "nonzero reference STRANS flags"}]
    assert report["old_input"]["native_polygon_parse_complete"] is False
    assert report["backend"]["gpu_requirement_met"] is False
    assert report["vertex_sidecar_path"]
    assert "Exact evidence below is CPU-only." in markdown.read_text(encoding="utf-8")


def test_gpu_required_kernel_failure_cleans_up_before_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(api_module, "discover_native_capabilities", lambda: _capabilities(cuda=True))
    old, new = _pair(tmp_path)
    verify = api_module._run_full_exact_verification
    calls = []

    def native_inputs(*args, **kwargs):
        return (
            NativePolygonExtraction(old, "top", extract_gds_polygons(old).buffer),
            NativePolygonExtraction(new, "top", extract_gds_polygons(new).buffer),
            {},
        )

    def fail_coverage(*args):
        raise RectCoverageUnsupported("rectangle kernel failed")

    def fail_delta(*args):
        raise cuda_ctypes.CudaCtypesUnavailable("record delta kernel failed")

    def track_verify(*args, **kwargs):
        calls.append("cpu")
        return verify(*args, **kwargs)

    monkeypatch.setattr(api_module, "_gpu_polygon_native_inputs", native_inputs)
    monkeypatch.setattr(api_module, "run_cuda_rectangle_coverage_diff", fail_coverage)
    monkeypatch.setattr(cuda_ctypes, "run_cuda_polygon_record_delta", fail_delta)
    monkeypatch.setattr(cuda_ctypes, "release_cuda_device_memory", lambda: calls.append("release"))
    monkeypatch.setattr(api_module, "_run_full_exact_verification", track_verify)
    result = compare_gds_files(CompareConfig(old, new, require_gpu=True))

    assert result.exit_code == 2
    assert calls == ["release", "cpu"]
    assert result.report.old_input["native_polygon_parse_complete"] is True
    assert result.report.timings["cuda_release_s"] >= 0
    assert [item["stage"] for item in result.report.exact_fallback["failures"]] == ["gpu-coverage-xor", "gpu-record-delta"]


def test_gpu_grid_mismatch_falls_back_to_shared_cpu_grid(tmp_path, monkeypatch):
    monkeypatch.setattr(api_module, "discover_native_capabilities", lambda: _capabilities(cuda=True))
    old, new = _pair(tmp_path, new_precision=2e-9)
    result = compare_gds_files(CompareConfig(old, new, require_gpu=True))

    assert result.exit_code == 2
    assert result.report.exact_fallback["failures"][0]["stage"] == "native-grid"
    layer = result.exact_result.layers[0]
    assert layer.old_minus_new_twice_area == layer.new_minus_old_twice_area == 40_000_000


def test_gpu_required_rotation_only_change_matches_cpu_audit(tmp_path):
    if not discover_native_capabilities().cuda_ctypes_available:
        pytest.skip("CUDA polygon backend unavailable")
    old, new = tmp_path / "old.gds", tmp_path / "new.gds"
    for path, angle in ((old, math.pi / 2), (new, 3 * math.pi / 2)):
        lib = gdstk.Library(unit=1e-6, precision=1e-9)
        child = lib.new_cell("child")
        child.add(gdstk.Polygon([(1, 1), (5, 1), (3, 3)], layer=1))
        lib.new_cell("top").add(gdstk.Reference(child, (7, -12), rotation=angle))
        lib.write_gds(path)

    result = compare_gds_files(CompareConfig(old, new, backend="cuda", require_gpu=True), audit_cpu_oracle=True)
    assert result.status == CompareStatus.DIFFERENCES_FOUND
    assert result.report.backend["gpu_requirement_met"] is True
    assert result.report.backend["cuda_event_elapsed_s"] > 0
    timings = result.report.timings
    assert not timings["old_native_polygon_cpu_hashes"]
    assert not timings["new_native_polygon_cpu_hashes"]
    for phase in ("read", "scan", "normalize", "layer_table", "placement", "pack"):
        assert 0 <= timings[f"old_native_polygon_{phase}_s"] <= timings["native_parse_s"]
    assert 0 < timings["h2d_upload_s"] <= timings["gpu_source_record_delta_s"]
    assert result.report.exact_fallback["cpu_oracle_audit"]["match"] is True
    assert not result.report.exact_fallback["source_edge_cpu_oracle_fallback"]


def test_gpu_required_identical_input_does_not_use_cpu_identity_shortcut(tmp_path):
    if not discover_native_capabilities().cuda_ctypes_available:
        pytest.skip("CUDA polygon backend unavailable")
    old, _ = _pair(tmp_path)

    result = compare_gds_files(CompareConfig(old, old, backend="cuda", require_gpu=True))

    assert result.exit_code == 0
    assert result.report.backend["gpu_requirement_met"] is True
    assert result.report.backend["cuda_kernel_count"] > 0


def test_gpu_hole_correction_reuses_native_buffers_and_matches_independent_oracle(tmp_path, monkeypatch):
    if not discover_native_capabilities().cuda_ctypes_available:
        pytest.skip("CUDA polygon backend unavailable")
    old, new = tmp_path / "outer.gds", tmp_path / "inner.gds"
    for path, box in ((old, ((0, 0), (10, 10))), (new, ((3, 2), (8, 7)))):
        lib = gdstk.Library(unit=1e-6, precision=1e-6)
        lib.new_cell("top").add(gdstk.rectangle(*box, layer=1))
        lib.write_gds(path)

    monkeypatch.setattr(api_module, "_native_buffers_rectangle_coverage_eligible", lambda *args: False)
    sparse_oracle = api_module._run_aligned_sparse_exact_oracle

    def skip_early_sparse(old, new, *, reject_flagged_context=False):
        return None if reject_flagged_context else sparse_oracle(old, new)

    monkeypatch.setattr(api_module, "_run_aligned_sparse_exact_oracle", skip_early_sparse)
    read_inputs = api_module._extract_pair
    calls = []

    def track_read(*args, **kwargs):
        calls.append("gdstk")
        return read_inputs(*args, **kwargs)

    monkeypatch.setattr(api_module, "_extract_pair", track_read)
    config = CompareConfig(old, new, backend="cuda", require_gpu=True)
    fast = compare_gds_files(config, exact_diff_markdown_path=tmp_path / "fast.md")
    assert not calls
    audited = compare_gds_files(config, exact_diff_markdown_path=tmp_path / "audited.md", audit_cpu_oracle=True)

    assert calls == ["gdstk"]
    assert fast.exit_code == audited.exit_code == 1
    assert fast.exact_result == audited.exact_result
    assert fast.exact_result.layers[0].old_minus_new_twice_area == 150
    assert fast.report.exact_fallback["source_edge_cpu_oracle_fallback"]["input"] == "native-polygon-buffers"
    correction = audited.report.exact_fallback["source_edge_cpu_oracle_fallback"]
    assert correction["input"] == "independent-gdstk-extraction"
    assert correction["native_buffers_vs_independent_oracle"]["match"] is True
    assert (tmp_path / "fast_diff_vertices.jsonl").read_bytes() == (tmp_path / "audited_diff_vertices.jsonl").read_bytes()

    monkeypatch.setattr(
        api_module, "_run_aligned_sparse_exact_oracle",
        lambda *args, reject_flagged_context=False: None if reject_flagged_context else api_module.ExactOracleResult(()),
    )
    mismatch = compare_gds_files(config, audit_cpu_oracle=True)
    assert mismatch.status == CompareStatus.INTERNAL_MISMATCH
    assert mismatch.exit_code == 2
    assert mismatch.exact_result == audited.exact_result
    assert mismatch.report.inventory_delta_count is None
    assert "native inventory is not reported" in " ".join(mismatch.report.errors)


def _nested_sparse_pair(tmp_path, *, inventory_only=False):
    old, new = tmp_path / "before.gds", tmp_path / "after.gds"
    outer = [(0, 0), (20, 0), (0, 20)]
    inner = [(2, 2), (6, 2), (2, 6)]
    for path, points in ((old, outer), (new, inner)):
        lib = gdstk.Library(unit=1e-6, precision=1e-6)
        top = lib.new_cell("top")
        top.add(gdstk.rectangle((4, 3), (8, 4), layer=1))
        top.add(gdstk.rectangle((50, 50), (55, 55), layer=1))
        if inventory_only:
            top.add(gdstk.Polygon(outer, layer=1))
            points = inner if path == old else [(2, 2), (7, 2), (2, 6)]
        top.add(gdstk.Polygon(points, layer=1))
        lib.write_gds(path)
    return old, new


def _diff_gds_coverage(path):
    by_category = {}
    for cell in gdstk.read_gds(path).cells:
        for polygon in cell.polygons:
            by_category.setdefault((cell.name, polygon.layer, polygon.datatype), []).append(polygon)
    return by_category


def test_cuda_sparse_coverage_matches_exact_gdstk_and_audits_after_release(tmp_path, monkeypatch):
    if not discover_native_capabilities().cuda_ctypes_available:
        pytest.skip("CUDA polygon backend unavailable")
    old, new = _nested_sparse_pair(tmp_path)
    reference = compare_gds_files(
        CompareConfig(old, new, backend="cpu", evidence_engine="gdstk-oracle"),
        exact_diff_markdown_path=tmp_path / "reference.md",
        diff_gds_path=tmp_path / "reference.gds",
    )
    events = []
    release = cuda_ctypes.release_cuda_device_memory
    extract = api_module._extract_pair

    def track_release():
        events.append("release")
        return release()

    def track_extract(*args, **kwargs):
        assert events[-1] == "release"
        events.append("extract")
        return extract(*args, **kwargs)

    def forbid_remaining_coverage(*args, **kwargs):
        pytest.fail("the proven hole correction must skip the remaining coverage work")

    monkeypatch.setattr(cuda_ctypes, "release_cuda_device_memory", track_release)
    monkeypatch.setattr(api_module, "_extract_pair", track_extract)
    monkeypatch.setattr(api_module, "run_record_delta_surplus_coverage_probe", forbid_remaining_coverage)
    monkeypatch.setattr(source_edge_diff, "_merge_layer_side_fragments_with_stats", forbid_remaining_coverage)
    config = CompareConfig(old, new, backend="cuda", require_gpu=True)
    fast = compare_gds_files(config, exact_diff_markdown_path=tmp_path / "fast.md", diff_gds_path=tmp_path / "fast.gds")
    assert events == ["release"]
    assert fast.exit_code == 1
    assert fast.exact_result == reference.exact_result
    assert fast.report.backend["gpu_requirement_met"] is True
    assert fast.report.exact_fallback["mode"] == "gpu-polygon-cuda-record-delta-sparse-gdstk"
    assert fast.report.timings["native_sparse_coverage_s"] > 0
    assert fast.report.timings["record_delta_surplus_coverage_s"] == 0
    assert fast.report.exact_fallback["sparse_coverage"]["trigger"] == "nested-direction-fragments"
    assert (tmp_path / "fast_diff_vertices.jsonl").read_bytes() == (tmp_path / "reference_diff_vertices.jsonl").read_bytes()
    actual_gds = _diff_gds_coverage(tmp_path / "fast.gds")
    expected_gds = _diff_gds_coverage(tmp_path / "reference.gds")
    assert actual_gds and actual_gds.keys() == expected_gds.keys()
    for key in actual_gds:
        assert not gdstk.boolean(actual_gds[key], expected_gds[key], "xor", precision=0.001)

    events.clear()
    audited = compare_gds_files(config, audit_cpu_oracle=True)
    assert events == ["release", "extract"]
    assert audited.exact_result == reference.exact_result
    assert audited.report.exact_fallback["cpu_oracle_audit"]["match"] is True

    events.clear()
    monkeypatch.setattr(api_module, "_run_aligned_sparse_exact_oracle", lambda *args, **kwargs: api_module.ExactOracleResult(()))
    mismatch = compare_gds_files(config, audit_cpu_oracle=True, exact_diff_markdown_path=tmp_path / "mismatch.md")
    assert events == ["release", "extract"]
    assert mismatch.status == CompareStatus.INTERNAL_MISMATCH
    assert mismatch.exit_code == 2
    assert mismatch.exact_result == reference.exact_result
    assert not mismatch.report.exact_fallback["cpu_oracle_audit"]["match"]
    assert mismatch.report.exact_fallback["sparse_coverage"]["final_input"] == "independent-gdstk-extraction"
    assert mismatch.report.inventory_delta_count is None
    assert "polygon_inventory_delta_source" not in mismatch.report.timings
    sidecar = (tmp_path / "mismatch_diff_vertices.jsonl").read_text(encoding="utf-8")
    assert {json.loads(line)["kind"] for line in sidecar.splitlines()} == {"coverage_component"}


def test_cuda_keeps_aligned_inventory_only_edits_without_hole_shortcut(tmp_path):
    if not discover_native_capabilities().cuda_ctypes_available:
        pytest.skip("CUDA polygon backend unavailable")
    old, new = _nested_sparse_pair(tmp_path, inventory_only=True)
    result = compare_gds_files(CompareConfig(old, new, backend="cuda", require_gpu=True))
    assert "sparse_coverage" not in result.report.exact_fallback
    assert result.exit_code == 1
    assert result.report.coverage_changed is False
    assert result.report.coverage_equal_but_inventory_changed is True
    assert result.report.inventory_delta_count == 2


def test_cuda_sparse_error_continues_to_existing_coverage(tmp_path, monkeypatch):
    if not discover_native_capabilities().cuda_ctypes_available:
        pytest.skip("CUDA polygon backend unavailable")
    old, new = _nested_sparse_pair(tmp_path)
    oracle = api_module._run_aligned_sparse_exact_oracle

    def fail_early(old_buffer, new_buffer, *, reject_flagged_context=False):
        if reject_flagged_context:
            raise RuntimeError("sparse coverage unavailable")
        return oracle(old_buffer, new_buffer)

    monkeypatch.setattr(api_module, "_run_aligned_sparse_exact_oracle", fail_early)
    result = compare_gds_files(CompareConfig(old, new, backend="cuda", require_gpu=True))
    assert result.exit_code == 1
    assert result.report.backend["gpu_requirement_met"] is True
    assert result.report.exact_verification_complete
    assert "sparse_coverage" not in result.report.exact_fallback
    assert {
        "stage": "native-sparse-coverage", "reason": "sparse coverage unavailable",
    } in result.report.exact_fallback["failures"]


def test_gpu_coverage_failure_after_record_delta_retains_gpu_evidence(tmp_path, monkeypatch):
    if not discover_native_capabilities().cuda_ctypes_available:
        pytest.skip("CUDA polygon backend unavailable")
    old, new = _pair(tmp_path)

    def fail_source_edges(*args, **kwargs):
        raise RuntimeError("source-edge assembly failed")

    monkeypatch.setattr(api_module, "_native_buffers_rectangle_coverage_eligible", lambda *args: False)
    monkeypatch.setattr(api_module, "_run_aligned_sparse_exact_oracle", lambda *args, **kwargs: None)
    monkeypatch.setattr(api_module, "exact_result_from_source_edges", fail_source_edges)
    result = compare_gds_files(CompareConfig(old, new, backend="cuda", require_gpu=True))

    assert result.exit_code == 1
    assert result.report.exact_verification_complete
    assert result.report.backend["gpu_requirement_met"] is True
    assert result.report.backend["cuda_kernel_count"] > 0
    assert result.report.exact_fallback["mode"] == "gpu-record-delta-plus-cpu-oracle-fallback"
    assert result.report.exact_fallback["failures"] == [{"stage": "gpu-source-edge", "reason": "source-edge assembly failed"}]


def test_gpu_required_rejects_cpu_only_options(tmp_path):
    old, new = _pair(tmp_path)
    with pytest.raises(ValueError, match="require_gpu.*CPU-only"):
        compare_gds_files(CompareConfig(old, new, require_gpu=True), exact=True)
    with pytest.raises(ValueError, match="require_gpu.*CPU-only"):
        compare_gds_files(CompareConfig(old, new, require_gpu=True, evidence_engine="gdstk-oracle"))
