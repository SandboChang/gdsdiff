from __future__ import annotations

import json
import subprocess
import sys

import gdstk
import pytest

from scgds.gdsdiff.native import discover_native_capabilities


def _write_pair(tmp_path):
    old_path = tmp_path / "old.gds"
    new_path = tmp_path / "new.gds"
    old = gdstk.Library(unit=1e-6, precision=1e-6)
    new = gdstk.Library(unit=1e-6, precision=1e-6)
    old_top = old.new_cell("top")
    new_top = new.new_cell("top")
    old_top.add(gdstk.rectangle((0, 0), (10, 10), layer=1, datatype=0))
    new_top.add(gdstk.rectangle((2, 0), (12, 10), layer=1, datatype=0))
    old.write_gds(old_path)
    new.write_gds(new_path)
    return old_path, new_path


def test_cli_writes_requested_artifacts_for_difference(tmp_path):
    old_path, new_path = _write_pair(tmp_path)
    report_path = tmp_path / "report.json"
    diff_path = tmp_path / "diff.gds"
    markdown_path = tmp_path / "history.md"
    exact_diff_markdown_path = tmp_path / "exact_diff.md"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scgds.gdsdiff",
            str(old_path),
            str(new_path),
            "--tile-size",
            "10",
            "--allow-region",
            "1:0,both,0,0,12,10",
            "--json-report",
            str(report_path),
            "--diff-gds",
            str(diff_path),
            "--markdown",
            str(markdown_path),
            "--exact-diff-markdown",
            str(exact_diff_markdown_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 1
    assert "status=differences-found" in completed.stdout
    assert "evidence_status=differences-found" in completed.stdout
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["status"] == "differences-found"
    assert data["evidence_status"] == "differences-found"
    assert data["artifacts"]["exact_diff_markdown"] == str(exact_diff_markdown_path)
    assert "GDS comparator" in markdown_path.read_text(encoding="utf-8")
    exact_diff_markdown = exact_diff_markdown_path.read_text(encoding="utf-8")
    assert "## Polygon Components" in exact_diff_markdown
    assert "vertices_preview_dbu" in exact_diff_markdown
    assert "points_sha256" in exact_diff_markdown


def test_gdsdiff_output_dir_writes_neutral_artifact_set_with_cpu_backend(tmp_path):
    old_path, new_path = _write_pair(tmp_path)
    output_dir = tmp_path / "gdsdiff_run"

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from scgds.gdsdiff.cli import main; raise SystemExit(main())",
            str(old_path),
            str(new_path),
            "--backend",
            "cpu",
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 1
    assert "status=differences-found" in completed.stdout
    assert "evidence_status=differences-found" in completed.stdout
    assert "differences_found=true" in completed.stdout
    report_path = output_dir / "report.json"
    exact_diff_markdown_path = output_dir / "exact_diff.md"
    sidecar_path = output_dir / "diff_vertices.jsonl"
    assert report_path.exists()
    assert exact_diff_markdown_path.exists()
    assert sidecar_path.exists()
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["status"] == "differences-found"
    assert data["evidence_status"] == "differences-found"
    assert data["backend"]["name"] == "cpu"
    assert data["backend"]["evidence_engine_selected"] == "gdstk-oracle"
    assert data["artifacts"]["exact_diff_markdown"] == str(exact_diff_markdown_path)
    assert data["artifacts"]["vertex_sidecar"] == str(sidecar_path)
    markdown = exact_diff_markdown_path.read_text(encoding="utf-8")
    assert "## Agent Minimal Read" in markdown
    assert "## Agent Review Checklist" in markdown
    assert "### Largest Directional Components" in markdown
    assert "before_only_bbox_union_dbu" in markdown
    assert "after_only_bbox_union_dbu" in markdown
    assert "coverage_before_only" in markdown
    assert sidecar_path.read_text(encoding="utf-8").strip()


def test_cli_returns_one_for_unexpected_difference(tmp_path):
    old_path, new_path = _write_pair(tmp_path)

    completed = subprocess.run(
        [sys.executable, "-m", "scgds.gdsdiff", str(old_path), str(new_path), "--tile-size", "10"],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 1
    assert "status=differences-found" in completed.stdout
    assert "evidence_status=differences-found" in completed.stdout


def test_cli_exact_component_backend_reports_stats(tmp_path):
    old_path, new_path = _write_pair(tmp_path)
    report_path = tmp_path / "report.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scgds.gdsdiff",
            str(old_path),
            str(new_path),
            "--exact",
            "--exact-component-backend",
            "rect-fallback",
            "--json-report",
            str(report_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 1
    data = json.loads(report_path.read_text(encoding="utf-8"))
    backend = data["exact_fallback"]["exact_component_backend"]
    assert data["exact_verification_complete"] is True
    assert backend["name"] == "manhattan-rect-exact-component"
    assert backend["stats"]["accelerated_tasks"] == 1
    assert backend["stats"]["fallback_tasks"] == 0
    assert backend["stats"]["unsupported_tasks"] == 0


def test_cli_cuda_exact_component_backend_reports_stats_when_available(tmp_path):
    if not discover_native_capabilities().cuda_ctypes_available:
        pytest.skip("CUDA ctypes backend is not available")
    old_path, new_path = _write_pair(tmp_path)
    report_path = tmp_path / "report.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scgds.gdsdiff",
            str(old_path),
            str(new_path),
            "--exact",
            "--exact-component-backend",
            "cuda-rect",
            "--json-report",
            str(report_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 1
    data = json.loads(report_path.read_text(encoding="utf-8"))
    backend = data["exact_fallback"]["exact_component_backend"]
    assert backend["name"] == "cuda-manhattan-rect-exact-component"
    assert backend["requires_gpu"] is True
    assert backend["stats"]["accelerated_tasks"] == 0
    assert backend["stats"]["cuda_rect_set_tasks"] == 1
    assert backend["stats"]["fallback_tasks"] == 0
    assert backend["stats"]["unsupported_tasks"] == 0


def test_cli_gpu_polygon_engine_uses_exact_cuda_rect_coverage_for_rectangle_input(tmp_path):
    old_path, new_path = _write_pair(tmp_path)
    report_path = tmp_path / "report.json"
    exact_diff_markdown_path = tmp_path / "exact_diff.md"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scgds.gdsdiff",
            str(old_path),
            str(new_path),
            "--backend",
            "cuda",
            "--evidence-engine",
            "gpu-polygon",
            "--json-report",
            str(report_path),
            "--exact-diff-markdown",
            str(exact_diff_markdown_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 1
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["status"] == "differences-found"
    assert data["evidence_status"] == "differences-found"
    assert data["differences_found"] is True
    assert data["inventory_changed"] is True
    assert data["backend"]["evidence_engine_selected"] == "gpu-polygon"
    assert data["backend"]["gpu_polygon_available"] is True
    assert data["backend"]["gpu_coverage_engine"] == "cuda-rect-coverage"
    assert data["backend"]["cuda_kernel_count"] > 0
    assert data["backend"]["transferred_bytes"] > 0
    assert data["exact_fallback"]["mode"] == "gpu-polygon-cuda-rect-coverage"
    assert data["exact_fallback"]["gpu_coverage"]["cuda_task_count"] > 0
    assert data["exact_verification_complete"] is True
    assert data["artifacts"]["vertex_sidecar"]
    markdown = exact_diff_markdown_path.read_text(encoding="utf-8")
    assert "- exact_verification_complete: `true`" in markdown
    assert "- evidence_status: `differences-found`" in markdown


def test_cli_gpu_polygon_engine_uses_source_edge_records_for_non_rectangle_without_audit(tmp_path):
    old_path = tmp_path / "old_tri.gds"
    new_path = tmp_path / "new_tri.gds"
    old_lib = gdstk.Library(unit=1e-6, precision=1e-9)
    old_top = old_lib.new_cell("top")
    old_top.add(gdstk.Polygon([(0, 0), (1, 0), (0, 1)], layer=1, datatype=0))
    old_lib.write_gds(old_path)
    new_lib = gdstk.Library(unit=1e-6, precision=1e-9)
    new_top = new_lib.new_cell("top")
    new_top.add(gdstk.Polygon([(0, 0), (2, 0), (0, 1)], layer=1, datatype=0))
    new_lib.write_gds(new_path)
    report_path = tmp_path / "report.json"
    exact_diff_markdown_path = tmp_path / "exact_diff.md"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scgds.gdsdiff",
            str(old_path),
            str(new_path),
            "--backend",
            "cuda",
            "--evidence-engine",
            "gpu-polygon",
            "--json-report",
            str(report_path),
            "--exact-diff-markdown",
            str(exact_diff_markdown_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 1
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["status"] == "differences-found"
    assert data["evidence_status"] == "differences-found"
    assert data["inventory_changed"] is True
    assert data["backend"]["gpu_polygon_available"] is True
    assert data["backend"]["gpu_record_delta_available"] is True
    assert data["backend"]["canonical_surplus_coverage_available"] is True
    assert data["backend"]["canonical_surplus_coverage_exact"] is False
    assert data["exact_fallback"]["mode"] == "gpu-polygon-cuda-source-edge-records"
    assert data["exact_fallback"]["engine_status"] == "source-edge-boundary-records-complete"
    assert data["exact_fallback"]["gpu_polygon_component_area"]["gpu_trapezoid_fragments_available"] is False
    assert data["exact_fallback"]["gpu_polygon_component_area"]["gpu_source_edge_records_available"] is True
    assert data["exact_verification_complete"] is True
    assert data["artifacts"]["vertex_sidecar"]
    markdown = exact_diff_markdown_path.read_text(encoding="utf-8")
    assert "GPU Polygon Evidence" in markdown
    assert "- exact_boundary_loops_available: `true`" in markdown


def test_cli_gpu_polygon_engine_cpu_audit_is_explicit(tmp_path):
    old_path, new_path = _write_pair(tmp_path)
    report_path = tmp_path / "report.json"
    exact_diff_markdown_path = tmp_path / "exact_diff.md"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scgds.gdsdiff",
            str(old_path),
            str(new_path),
            "--backend",
            "cuda",
            "--evidence-engine",
            "gpu-polygon",
            "--audit-cpu-oracle",
            "--json-report",
            str(report_path),
            "--exact-diff-markdown",
            str(exact_diff_markdown_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 1
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["status"] == "differences-found"
    assert data["evidence_status"] == "differences-found"
    assert data["backend"]["evidence_engine_selected"] == "gpu-polygon"
    assert data["backend"]["gpu_polygon_available"] is True
    assert data["backend"]["audit_cpu_oracle"] is True
    assert data["backend"]["cuda_kernel_count"] > 0
    assert data["exact_fallback"]["mode"] == "gpu-polygon-cuda-rect-coverage"
    assert data["exact_fallback"]["audit_cpu_oracle"] is True
    assert data["exact_fallback"]["cpu_oracle_audit"]["match"] is True
    assert data["exact_verification_complete"] is True
    assert data["artifacts"]["vertex_sidecar"]
    assert data["timings"]["cpu_oracle_audit_s"] >= 0
    markdown = exact_diff_markdown_path.read_text(encoding="utf-8")
    assert "- exact_verification_complete: `true`" in markdown
    assert "vertices_preview_dbu" in markdown


def test_cli_returns_two_for_invalid_input(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-m", "scgds.gdsdiff", str(tmp_path / "missing.gds"), str(tmp_path / "also_missing.gds")],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2
    assert "status=unsupported-input" in completed.stdout
    assert "evidence_status=unsupported-input" in completed.stdout


def test_cli_verify_exact_after_fast_attaches_exact_evidence_when_cuda_available(tmp_path):
    if not discover_native_capabilities().cuda_available:
        pytest.skip("CUDA fast backend is not available")
    old_path, new_path = _write_pair(tmp_path)
    report_path = tmp_path / "report.json"
    exact_diff_markdown_path = tmp_path / "exact_diff.md"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scgds.gdsdiff",
            str(old_path),
            str(new_path),
            "--backend",
            "cuda",
            "--tile-size",
            "10",
            "--verify-exact-after-fast",
            "--json-report",
            str(report_path),
            "--exact-diff-markdown",
            str(exact_diff_markdown_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 1
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["exact_verification_complete"] is True
    assert data["exact_fallback"]["mode"] == "gpu-polygon-cuda-rect-coverage"
    assert data["backend"]["evidence_engine_selected"] == "gpu-polygon"
    assert data["backend"]["gpu_polygon_available"] is True
    markdown = exact_diff_markdown_path.read_text(encoding="utf-8")
    assert "- exact_verification_complete: `true`" in markdown
    assert "vertices_preview_dbu" in markdown
    assert "points_sha256" in markdown
    assert "| `coverage_before_only` | 1 | 0 |" in markdown or "| `coverage_after_only` | 1 | 0 |" in markdown
