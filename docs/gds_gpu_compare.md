# GDS GPU Comparator

The Plan 1 comparator checks whether two GDS files are geometrically equivalent
or whether their differences are confined to explicitly allowed regions.

The production correctness oracle is `gdstk.boolean`. CPU/CUDA broad-phase
code is used to find candidate windows and reduce work; it is not treated as
the final proof of geometry equality.

## Basic CLI

Run the package CLI from the repository root:

```bash
.venv/bin/python -m gds_tools_tk.gpu_compare references/testGDS/chip/chip2_before.gds references/testGDS/chip/chip2_after.gds --top chip --tile-size 256 --json-report outputs/chip2_compare.json --diff-gds outputs/chip2_diff.gds
```

Exit codes:

- `0`: equivalent, or only expected differences under the selected policy.
- `1`: unexpected geometry differences, or any difference when
  `--failure-policy fail-on-any-diff` is selected.
- `2`: invalid input, unresolved comparison, backend error, or artifact error.

Useful options:

- `--top NAME`: compare the same top cell name in both files.
- `--old-top OLD --new-top NEW`: compare different top cell names.
- `--layers all` or `--layers 1:0,2:0`: select layer/datatype pairs.
- `--tile-size N`: tune broad-phase tile granularity.
- `--backend cpu|auto|cuda`: select broad-phase backend. The current Python
  dispatch is CPU/fallback; CUDA kernels are qualified at the native test
  level but not used as the exact oracle.
- `--exact`: bypass the tiled broad phase and run the full exact oracle.
- `--allow-region L:D,DIRECTION,X0,Y0,X1,Y1[,HALO]`: mark a bounded expected
  difference.
- `--allow-whole-layer L:D,DIRECTION`: mark an entire layer/direction expected.
- `--json-report PATH`: write machine-readable report JSON.
- `--diff-gds PATH`: write a GDS showing classified diff fragments.
- `--markdown PATH`: write a design-history style Markdown summary.

Allowed directions are `addition`, `removal`, `both`, and `xor`.

## Workflow Wrapper

For agent/CI validation around generated project outputs, use:

```bash
.venv/bin/python tools/run_gds_geometry_check.py path/to/output.gds --artifacts-dir outputs/gds_check --rebuild-command ".venv/bin/python path/to/stage.py" --top TOP
```

The wrapper:

1. Copies the current output GDS as a temporary baseline.
2. Runs the optional rebuild command.
3. Compares the baseline against the rebuilt output.
4. Writes `result.json`, `compare_report.json`, `diff.gds`, and `summary.md`.
5. Optionally runs an assembly profile such as `--assembly-profile calibrator-model1`.
6. Optionally appends a design-history entry.
7. Moves the temporary baseline into `.dump/gds_geometry_check_<label>/`.

Use this wrapper for geometry-affecting recipe work because it preserves the
backup/readback policy and produces one machine-readable verdict.

## Benchmarking

Run the benchmark harness with:

```bash
.venv/bin/python benchmarks/gpu_compare/run_benchmarks.py --max-pairs 1 --mode windowed --tile-size 256 --output /tmp/gds_gpu_compare_benchmark_smoke.json
```

Benchmark rows include wall-clock time plus comparator extraction/exact/total
timings. They are end-to-end comparator timings, not kernel-only numbers.

## Chip And Wafer Guidance

Default to the smallest affected design scale. For chip edits, compare the chip
GDS exactly and inspect local diff components. For block or wafer assemblies,
prefer hierarchy/readback/profile checks unless the block or wafer geometry
itself changed. A wafer made from many repeated chip cells should not normally
be flattened into a full-wafer XOR just to validate a chip-level edit.
