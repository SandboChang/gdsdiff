from __future__ import annotations

import json
import math

import gdstk
import pytest

from gdsdiff import CompareConfig, CompareStatus, compare_gds_files, extract_gds_polygons
from gdsdiff import api as api_module
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

    monkeypatch.setattr(api_module, "_run_aligned_sparse_exact_oracle", lambda *args: api_module.ExactOracleResult(()))
    mismatch = compare_gds_files(config, audit_cpu_oracle=True)
    assert mismatch.status == CompareStatus.INTERNAL_MISMATCH
    assert mismatch.exit_code == 2
    assert mismatch.exact_result == audited.exact_result
    assert mismatch.report.inventory_delta_count is None
    assert "native inventory is not reported" in " ".join(mismatch.report.errors)


def test_gpu_coverage_failure_after_record_delta_retains_gpu_evidence(tmp_path, monkeypatch):
    if not discover_native_capabilities().cuda_ctypes_available:
        pytest.skip("CUDA polygon backend unavailable")
    old, new = _pair(tmp_path)

    def fail_source_edges(*args):
        raise RuntimeError("source-edge assembly failed")

    monkeypatch.setattr(api_module, "_native_buffers_rectangle_coverage_eligible", lambda *args: False)
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
