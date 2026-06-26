from __future__ import annotations

import json
import os

import gdstk

from gds_tools_tk.gdsdiff.acceptance_evidence import (
    AcceptanceCorpusPair,
    SCHEMA_VERSION,
    _busy_percent,
    default_acceptance_corpus,
    run_acceptance_evidence,
    validate_acceptance_evidence,
)


def _write(path, *, changed: bool = False) -> None:
    lib = gdstk.Library(unit=1e-6, precision=1e-6)
    cell = lib.new_cell("TOP")
    cell.add(gdstk.rectangle((0, 0), (10, 10), layer=1, datatype=0))
    if changed:
        cell.add(gdstk.rectangle((20, 0), (30, 10), layer=1, datatype=0))
    lib.write_gds(path)


def test_acceptance_evidence_writes_speed_and_exact_artifacts(tmp_path):
    before = tmp_path / "before.gds"
    after = tmp_path / "after.gds"
    output_dir = tmp_path / "evidence"
    _write(before)
    _write(after, changed=True)

    result = run_acceptance_evidence(
        (AcceptanceCorpusPair("tiny", before, after),),
        output_dir,
        min_speedup=0.0,
        tile_size=8,
        fast_backend="cpu",
    )

    validate_acceptance_evidence(result)
    output_path = output_dir / "acceptance_evidence.json"
    assert result["schema_version"] == SCHEMA_VERSION
    assert result["output_path"] == str(output_path)
    assert output_path.exists()
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["schema_version"] == SCHEMA_VERSION
    row = persisted["results"][0]
    assert row["pair"] == "tiny"
    assert row["fast_status_matches_exact"] is True
    assert row["exact_polygon_evidence_complete"] is True
    assert row["passes_speed_status_gate"] is True
    assert len(row["exact_policy_sha256"]) == 64
    assert len(row["exact_diff_markdown_sha256"]) == 64
    assert row["cpu_windowed_cpu_monitor"]["available"] is False
    assert row["cpu_windowed_all_cores_ge_50pct_seen"] is False
    assert persisted["all_monitored_cpu_runs_populated_all_cores_ge_50pct"] is None
    markdown = (output_dir / "tiny_exact_diff.md").read_text(encoding="utf-8")
    assert "- exact_verification_complete: `true`" in markdown
    assert "## Agent Minimal Read" in markdown
    assert "- coverage_changed: `true`" in markdown
    assert "points_sha256" in markdown
    assert "vertices_preview_dbu" in markdown


def test_acceptance_evidence_can_monitor_cpu(tmp_path):
    before = tmp_path / "before.gds"
    after = tmp_path / "after.gds"
    output_dir = tmp_path / "evidence"
    _write(before)
    _write(after, changed=True)

    result = run_acceptance_evidence(
        (AcceptanceCorpusPair("tiny", before, after),),
        output_dir,
        min_speedup=0.0,
        tile_size=8,
        fast_backend="cpu",
        monitor_cpu=True,
        monitor_interval_s=0.01,
    )

    monitor = result["results"][0]["cpu_windowed_cpu_monitor"]
    assert "available" in monitor
    assert isinstance(result["results"][0]["cpu_windowed_all_cores_ge_50pct_seen"], bool)
    if monitor["available"]:
        assert "logical_cpu_count" in monitor or monitor["sample_count"] == 0
    assert os.environ.get("GDS_GPU_COMPARE_PARALLEL_MIN_POLYGONS") is None
    assert os.environ.get("GDS_GPU_COMPARE_PARALLEL_EXTRACT_PAIR_MIN_BYTES") is None


def test_busy_percent_from_proc_stat_deltas():
    assert _busy_percent((100, 40), (200, 60)) == 80.0
    assert _busy_percent((100, 40), (100, 40)) == 0.0


def test_default_acceptance_corpus_discovers_recursive_before_after_pairs(tmp_path):
    root = tmp_path
    test_root = root / "references" / "testGDS"
    (test_root / "chip").mkdir(parents=True)
    (test_root / "medium").mkdir()
    (test_root / "equal_count").mkdir()
    (test_root / "chip" / "chip_before.gds").write_bytes(b"before")
    (test_root / "chip" / "chip_after.gds").write_bytes(b"after")
    (test_root / "chip" / "chip2_before.gds").write_bytes(b"before")
    (test_root / "chip" / "chip2_after.gds").write_bytes(b"after")
    (test_root / "medium" / "medium_before.gds").write_bytes(b"before")
    (test_root / "medium" / "medium_after.gds").write_bytes(b"after")
    (test_root / "equal_count" / "equal_count_before.gds").write_bytes(b"before")
    (test_root / "equal_count" / "equal_count_after.gds").write_bytes(b"after")
    (test_root / "orphan_before.gds").write_bytes(b"ignored")

    pairs = default_acceptance_corpus(root)
    by_name = {pair.name: pair for pair in pairs}

    assert tuple(sorted(by_name)) == ("chip", "chip2", "equal_count", "medium")
    assert by_name["medium"].top == "medium_compare_top"
    assert by_name["equal_count"].top == "equal_count_top"
    assert by_name["chip"].top is None
