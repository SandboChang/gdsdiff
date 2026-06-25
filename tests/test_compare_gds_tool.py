from __future__ import annotations

import json
import subprocess
import sys

import gdstk

from gds_tools_tk.tools.compare_gds import summarize


def _write_pair(tmp_path, *, shifted=True):
    old_path = tmp_path / "old.gds"
    new_path = tmp_path / "new.gds"
    old = gdstk.Library(unit=1e-6, precision=1e-6)
    new = gdstk.Library(unit=1e-6, precision=1e-6)
    old_top = old.new_cell("top")
    new_top = new.new_cell("top")
    old_top.add(gdstk.rectangle((0, 0), (10, 10), layer=1, datatype=0))
    new_top.add(gdstk.rectangle((2 if shifted else 0, 0), (12 if shifted else 10, 10), layer=1, datatype=0))
    old.write_gds(old_path)
    new.write_gds(new_path)
    return old_path, new_path


def test_summarize_legacy_stats_api(tmp_path):
    old_path, _ = _write_pair(tmp_path)

    summary = summarize(old_path)

    assert summary["cells"] == 1
    assert summary["top"] == "top"
    assert summary["polygon_count"] == 1
    assert summary["by_layer"][(1, 0)][0] == 1


def test_compare_gds_default_stats_stdout_unchanged_shape(tmp_path):
    old_path, new_path = _write_pair(tmp_path, shifted=False)

    completed = subprocess.run(
        [sys.executable, "-m", "gds_tools_tk.tools.compare_gds", str(old_path), str(new_path)],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0
    assert f"old: {old_path}" in completed.stdout
    assert "  polygons: 1" in completed.stdout
    assert "status=" not in completed.stdout


def test_compare_gds_exact_mode_returns_unexpected_exit(tmp_path):
    old_path, new_path = _write_pair(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "gds_tools_tk.tools.compare_gds",
            "--mode",
            "exact",
            str(old_path),
            str(new_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 1
    assert "status=unexpected-differences" in completed.stdout
    assert "backend=cpu" in completed.stdout


def test_compare_gds_accelerated_mode_writes_report_artifacts(tmp_path):
    old_path, new_path = _write_pair(tmp_path)
    report_path = tmp_path / "report.json"
    diff_path = tmp_path / "diff.gds"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "gds_tools_tk.tools.compare_gds",
            "--mode",
            "accelerated",
            "--tile-size",
            "10",
            "--allow-region",
            "1:0,both,0,0,12,10",
            "--json-report",
            str(report_path),
            "--diff-gds",
            str(diff_path),
            str(old_path),
            str(new_path),
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
    assert diff_path.exists()
