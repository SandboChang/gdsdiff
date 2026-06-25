from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import gdstk


ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_PATH = ROOT / "benchmarks" / "gpu_compare" / "run_benchmarks.py"
SPEC = importlib.util.spec_from_file_location("run_gds_benchmarks", BENCHMARK_PATH)
benchmarks = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = benchmarks
SPEC.loader.exec_module(benchmarks)


def _write(path: Path, *, changed: bool = False) -> None:
    lib = gdstk.Library(unit=1e-6, precision=1e-6)
    cell = lib.new_cell("TOP")
    cell.add(gdstk.rectangle((0, 0), (10, 10), layer=1, datatype=0))
    if changed:
        cell.add(gdstk.rectangle((20, 0), (30, 10), layer=1, datatype=0))
    lib.write_gds(path)


def test_benchmark_result_schema_and_smoke_run(tmp_path):
    before = tmp_path / "before.gds"
    after = tmp_path / "after.gds"
    _write(before)
    _write(after, changed=True)
    pair = benchmarks.BenchmarkPair("tiny", before, after)

    result = benchmarks.run_benchmarks((pair,), modes=("windowed", "exact"), tile_sizes=(8,), repeats=1)

    benchmarks.validate_benchmark_result(result)
    assert result["schema_version"] == benchmarks.SCHEMA_VERSION
    assert [row["mode"] for row in result["results"]] == ["windowed", "exact"]
    assert all(row["pair"] == "tiny" for row in result["results"])
    assert all(row["mean_wall_s"] >= row["mean_total_s"] >= 0 for row in result["results"])


def test_benchmark_cli_writes_json(tmp_path):
    before = tmp_path / "before.gds"
    after = tmp_path / "after.gds"
    output = tmp_path / "benchmark.json"
    _write(before)
    _write(after, changed=True)

    exit_code = benchmarks.main(
        [
            "--pair",
            f"tiny:{before}:{after}",
            "--mode",
            "windowed",
            "--tile-size",
            "8",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    text = output.read_text(encoding="utf-8")
    assert "gds-gpu-compare-benchmark-v1" in text
