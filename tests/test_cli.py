from __future__ import annotations

import json
import subprocess
import sys

import gdstk
import pytest

from gds_tools_tk.gpu_compare.native import discover_native_capabilities


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


def test_cli_writes_requested_artifacts_for_allowed_difference(tmp_path):
    old_path, new_path = _write_pair(tmp_path)
    report_path = tmp_path / "report.json"
    diff_path = tmp_path / "diff.gds"
    markdown_path = tmp_path / "history.md"
    exact_diff_markdown_path = tmp_path / "exact_diff.md"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "gds_tools_tk.gpu_compare",
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

    assert completed.returncode == 0
    assert "status=expected-differences-only" in completed.stdout
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["status"] == "expected-differences-only"
    assert data["artifacts"]["diff_gds"] == str(diff_path)
    assert data["artifacts"]["exact_diff_markdown"] == str(exact_diff_markdown_path)
    assert diff_path.exists()
    assert "GDS comparator result" in markdown_path.read_text(encoding="utf-8")
    exact_diff_markdown = exact_diff_markdown_path.read_text(encoding="utf-8")
    assert "## Polygon Components" in exact_diff_markdown
    assert "vertices_preview_dbu" in exact_diff_markdown
    assert "points_sha256" in exact_diff_markdown


def test_cli_returns_one_for_unexpected_difference(tmp_path):
    old_path, new_path = _write_pair(tmp_path)

    completed = subprocess.run(
        [sys.executable, "-m", "gds_tools_tk.gpu_compare", str(old_path), str(new_path), "--tile-size", "10"],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 1
    assert "status=unexpected-differences" in completed.stdout


def test_cli_exact_component_backend_reports_stats(tmp_path):
    old_path, new_path = _write_pair(tmp_path)
    report_path = tmp_path / "report.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "gds_tools_tk.gpu_compare",
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
            "gds_tools_tk.gpu_compare",
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


def test_cli_returns_two_for_invalid_input(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-m", "gds_tools_tk.gpu_compare", str(tmp_path / "missing.gds"), str(tmp_path / "also_missing.gds")],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2
    assert "gds-gpu-compare:" in completed.stderr


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
            "gds_tools_tk.gpu_compare",
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
    assert data["exact_fallback"]["mode"] == "native-geometry-fast-plus-full-exact"
    assert data["backend"]["native_geometry_guard"] is True
    assert data["backend"]["exact_verification_after_fast"] is True
    markdown = exact_diff_markdown_path.read_text(encoding="utf-8")
    assert "- exact_verification_complete: `true`" in markdown
    assert "vertices_preview_dbu" in markdown
    assert "points_sha256" in markdown
    assert "| `unexpected` | `removal` | 1 | 0 |" in markdown or "| `unexpected` | `addition` | 1 | 0 |" in markdown
