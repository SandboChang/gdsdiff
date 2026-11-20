# gdsdiff Engine

`gdsdiff` compares two GDS files and emits neutral factual evidence about exact
geometry differences. It does not decide whether a difference is intended.

The default command uses `--backend auto`: a supported GPU evidence path when
available, with CPU `gdstk` exact fallback when no suitable GPU backend is
available.

## Basic CLI

Run the package CLI from the repository root:

```bash
.venv/bin/python -m gdsdiff before.gds after.gds --top TOP --output-dir /tmp/gdsdiff_chip2
```

`--output-dir` writes the standard artifact set: `report.json`,
`exact_diff.md`, and `diff_vertices.jsonl`. For agent validation, treat
`exact_diff.md` and its sidecar as temporary review artifacts unless the user
explicitly asks to retain them.

Exit codes:

- `0`: identical/equivalent. The policy-aware CPU oracle can also pass
  allowed-only differences.
- `1`: differences found. The neutral CLI also reports allowed changes as
  differences rather than judging intent.
- `2`: invalid input, incomplete evidence, internal mismatch, artifact error,
  or a required GPU comparison that did not run.

Useful options:

- `--top NAME`: compare the same top cell name in both files.
- `--old-top OLD --new-top NEW`: compare different top cell names.
- `--layers all` or `--layers 1:0,2:0`: select layer/datatype pairs.
- `--tile-size N`: tune broad-phase tile granularity.
- `--backend auto|cpu|cuda|metal`: select backend. `auto` prefers a supported
  GPU path and falls back to CPU exact evidence.
- `--evidence-engine auto|gdstk-oracle|gpu-polygon`: select the evidence
  generator. Use `--backend cpu --evidence-engine gdstk-oracle` for an
  independent CPU reference.
- `--require-gpu`: require real GPU comparison work, not just an available
  backend. An exact CPU-only fallback can still produce artifacts, but does
  not satisfy this option.
- `--exact`: bypass the tiled broad phase and run the full exact oracle.
- `--audit-cpu-oracle`: when using `--evidence-engine gpu-polygon`, run the
  independent CPU `gdstk` oracle on the original inputs after GPU cleanup.
- `--allow-region L:D,DIRECTION,X0,Y0,X1,Y1[,HALO]`: mark a bounded expected
  difference.
- `--allow-whole-layer L:D,DIRECTION`: mark an entire layer/direction expected.
- `--json-report PATH`: write machine-readable report JSON.
- `--diff-gds PATH`: write a GDS showing classified diff fragments.
- `--markdown PATH`: write a design-history style Markdown summary.
- `--exact-diff-markdown PATH`: write the compact factual exact-diff Markdown.

Allowed policy directions are `addition`, `removal`, `both`, and `xor`.

## Workflow Wrapper

The consuming `scgds` repository provides a project-output workflow wrapper:

```bash
.venv/bin/python -m scgds.tools.run_gds_geometry_check path/to/output.gds --artifacts-dir outputs/gds_check --rebuild-command ".venv/bin/python path/to/stage.py" --top TOP
```

The wrapper:

1. Copies the current output GDS as a temporary baseline.
2. Runs the optional rebuild command.
3. Compares the baseline against the rebuilt output.
4. Writes `result.json`, `compare_report.json`, `diff.gds`, and `summary.md`.
5. Optionally runs an assembly profile such as `--assembly-profile calibrator-model1`.
6. Optionally appends a design-history entry.
7. Moves the temporary baseline into `.dump/gds_geometry_check_<label>/`.

For work that needs the current compact output diff Markdown, call `gdsdiff`
with `--output-dir` or `--exact-diff-markdown` so the generated Markdown comes
from the standard exact-diff reporter. The wrapper is not required to use the
standalone comparator.

## Benchmarking

Run the benchmark harness with:

```bash
.venv/bin/python benchmarks/gdsdiff/run_benchmarks.py --pair smoke:before.gds:after.gds --mode windowed --tile-size 256 --output /tmp/gdsdiff_benchmark_smoke.json
```

Benchmark rows include API wall time plus comparator extraction, exact, and
total timings. To measure the default CUDA CLI with artifact output, follow
the [benchmark instructions](../benchmarks/gdsdiff/README.md) instead.

## GPU Polygon Evidence Engine

The `gpu-polygon` engine prepares integer DBU polygon buffers on the host.
Inventory and coverage are separate:

- Rectangle coverage takes priority when the native inputs qualify. It uses
  the accelerator's exact rectangle coverage implementation.
- For other CUDA inputs, device kernels compare source records for inventory
  and the coverage engine builds context and traces source edges.
- If tracing finds nested loops that already require exact hole correction,
  CUDA can skip the rest of boundary assembly for eligible aligned sparse
  input. `gdstk` computes directional coverage from the changed polygons and
  all same-layer context touching their bounds. The reduction checks integer
  vertices, not just device hashes.
- If the sparse reduction declines or the engine does not find that hole
  condition, the normal coverage route continues. Failing GPU stages are
  reported, and the comparator attempts exact CPU fallback where possible.
  Check `exact_verification_complete` before using any result.

`gpu-polygon-cuda-record-delta-sparse-gdstk` identifies the hybrid sparse route.
It does not imply the Boolean ran on CUDA. `--audit-cpu-oracle` uses
independently extracted inputs rather than trusting native buffer reuse.
A native-buffer audit mismatch returns exit 2 and withholds unverified
inventory. See the [report guide](gdsdiff_report.md) for the evidence fields.

Do not describe shortcut/native-guard timings as GPU polygon evidence timings.
Reports must distinguish shortcut verdicts, GPU kernel activity, and CPU oracle
audit time.

## Chip And Wafer Guidance

Default to the smallest affected design scale. For chip edits, compare the chip
GDS exactly and inspect local diff components. For block or wafer assemblies,
prefer hierarchy/readback/profile checks unless the block or wafer geometry
itself changed. A wafer made from many repeated chip cells should not normally
be flattened into a full-wafer XOR just to validate a chip-level edit.
