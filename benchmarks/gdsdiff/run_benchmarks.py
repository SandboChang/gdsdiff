#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

from gdsdiff import CompareConfig, compare_gds_files, discover_native_capabilities

SCHEMA_VERSION = "gdsdiff-benchmark-v1"


@dataclass(frozen=True)
class BenchmarkPair:
    name: str
    before: Path
    after: Path


def default_pairs(repo_root: Path) -> tuple[BenchmarkPair, ...]:
    candidates = (
        ("chip2", repo_root / "references/testGDS/chip/chip2_before.gds", repo_root / "references/testGDS/chip/chip2_after.gds"),
        ("medium", repo_root / "references/testGDS/medium/medium_before.gds", repo_root / "references/testGDS/medium/medium_after.gds"),
        ("block", repo_root / "references/testGDS/block/block_before.gds", repo_root / "references/testGDS/block/block_after.gds"),
    )
    return tuple(
        BenchmarkPair(name, before, after)
        for name, before, after in candidates
        if before.exists() and after.exists()
    )


def run_benchmarks(
    pairs: tuple[BenchmarkPair, ...],
    *,
    modes: tuple[str, ...] = ("windowed", "exact"),
    tile_sizes: tuple[int, ...] = (256, 1024),
    backend: str = "cpu",
    repeats: int = 1,
) -> dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()
    capabilities = discover_native_capabilities().to_json()
    rows = []
    for pair in pairs:
        for mode in modes:
            exact = mode == "exact"
            sizes = (tile_sizes[0],) if exact else tile_sizes
            for tile_size in sizes:
                run_timings = []
                statuses = []
                for _ in range(repeats):
                    start = perf_counter()
                    result = compare_gds_files(
                        CompareConfig(pair.before, pair.after, backend=backend, tile_size=tile_size),
                        exact=exact,
                    )
                    wall_s = perf_counter() - start
                    statuses.append(result.status.value)
                    run_timings.append({"wall_s": wall_s, **result.timings})
                rows.append(
                    {
                        "pair": pair.name,
                        "before": str(pair.before),
                        "after": str(pair.after),
                        "mode": mode,
                        "backend": backend,
                        "tile_size": tile_size,
                        "repeats": repeats,
                        "status": statuses[-1],
                        "statuses": statuses,
                        "mean_wall_s": mean(item["wall_s"] for item in run_timings),
                        "mean_extract_s": mean(item.get("extract_s", 0.0) for item in run_timings),
                        "mean_exact_s": mean(item.get("exact_s", 0.0) for item in run_timings),
                        "mean_total_s": mean(item.get("total_s", 0.0) for item in run_timings),
                    }
                )
    result = {
        "schema_version": SCHEMA_VERSION,
        "started": started,
        "backend": backend,
        "modes": list(modes),
        "tile_sizes": list(tile_sizes),
        "repeats": repeats,
        "capabilities": capabilities,
        "results": rows,
    }
    validate_benchmark_result(result)
    return result


def validate_benchmark_result(result: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "started",
        "backend",
        "modes",
        "tile_sizes",
        "repeats",
        "capabilities",
        "results",
    }
    missing = sorted(required - set(result))
    if missing:
        raise ValueError(f"benchmark result missing fields: {', '.join(missing)}")
    if result["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported benchmark schema")
    if not isinstance(result["results"], list):
        raise ValueError("benchmark results must be a list")
    for row in result["results"]:
        for field in ("pair", "mode", "backend", "tile_size", "status", "mean_wall_s", "mean_total_s"):
            if field not in row:
                raise ValueError(f"benchmark row missing field: {field}")
        if row["mean_wall_s"] < 0 or row["mean_total_s"] < 0:
            raise ValueError("benchmark timings must be nonnegative")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run reproducible GDS comparator benchmarks.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--pair", action="append", metavar="NAME:BEFORE:AFTER")
    parser.add_argument("--mode", action="append", choices=["windowed", "exact"], dest="modes")
    parser.add_argument("--tile-size", action="append", type=int, dest="tile_sizes")
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-pairs", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    pairs = _parse_pairs(args.pair) if args.pair else default_pairs(args.repo_root)
    if args.max_pairs is not None:
        pairs = pairs[: args.max_pairs]
    if not pairs:
        parser.error("no benchmark pairs found; pass --pair NAME:BEFORE:AFTER")
    result = run_benchmarks(
        pairs,
        modes=tuple(args.modes or ["windowed", "exact"]),
        tile_sizes=tuple(args.tile_sizes or [256, 1024]),
        backend=args.backend,
        repeats=args.repeats,
    )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


def _parse_pairs(values: list[str]) -> tuple[BenchmarkPair, ...]:
    pairs = []
    for value in values:
        parts = value.split(":")
        if len(parts) != 3:
            raise ValueError("--pair must have form NAME:BEFORE:AFTER")
        pairs.append(BenchmarkPair(parts[0], Path(parts[1]), Path(parts[2])))
    return tuple(pairs)


if __name__ == "__main__":
    raise SystemExit(main())
