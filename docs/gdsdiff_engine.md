# gdsdiff Engine

`gdsdiff` compares two GDS files and emits neutral factual evidence about exact
geometry differences. It does not decide whether a difference is intended.

The default command uses `--backend auto`: a supported GPU evidence path when
available, with CPU `gdstk` exact fallback when no suitable GPU backend is
available.

## Basic CLI

Run the package CLI from the repository root:

```bash
.venv/bin/python -m gds_tools_tk.gdsdiff references/testGDS/chip/chip2_before.gds references/testGDS/chip/chip2_after.gds --top chip --output-dir results/0001_chip2_diff
```

`--output-dir` writes the standard artifact set: `report.json`,
`exact_diff.md`, and `diff_vertices.jsonl`. For agent validation, treat
`exact_diff.md` and its sidecar as temporary review artifacts unless the user
explicitly asks to retain them.

Exit codes:

- `0`: identical/equivalent, or only allowed differences under the selected
  policy.
- `1`: differences found under the active failure policy.
- `2`: invalid input, unresolved comparison, backend error, or artifact error.

Useful options:

- `--top NAME`: compare the same top cell name in both files.
- `--old-top OLD --new-top NEW`: compare different top cell names.
- `--layers all` or `--layers 1:0,2:0`: select layer/datatype pairs.
- `--tile-size N`: tune broad-phase tile granularity.
- `--backend auto|cpu|cuda|metal`: select backend. `auto` prefers a supported
  GPU path and falls back to CPU exact evidence.
- `--evidence-engine auto|gdstk-oracle|gpu-polygon`: select the evidence
  generator. `gpu-polygon` uses the native parsed polygon-array path. It can
  currently produce exact CUDA coverage evidence for rectangle-only inputs,
  GPU source-record deltas, and canonical source-surplus component probes for
  general inputs. General non-rectangle coverage XOR remains fail-closed when
  overlapping surplus polygons require the arbitrary-polygon Boolean kernel,
  unless `--audit-cpu-oracle` is explicitly requested.
- `--exact`: bypass the tiled broad phase and run the full exact oracle.
- `--audit-cpu-oracle`: when using `--evidence-engine gpu-polygon`, run the
  CPU `gdstk` oracle explicitly as audit evidence until GPU polygon records are
  implemented.
- `--allow-region L:D,DIRECTION,X0,Y0,X1,Y1[,HALO]`: mark a bounded expected
  difference.
- `--allow-whole-layer L:D,DIRECTION`: mark an entire layer/direction expected.
- `--json-report PATH`: write machine-readable report JSON.
- `--diff-gds PATH`: write a GDS showing classified diff fragments.
- `--markdown PATH`: write a design-history style Markdown summary.
- `--exact-diff-markdown PATH`: write the compact factual exact-diff Markdown.

Allowed policy directions are `addition`, `removal`, `both`, and `xor`.

## Workflow Wrapper

For agent/CI validation around generated project outputs, a wrapper exists:

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

For work that needs the current compact output diff Markdown, call `gdsdiff`
with `--output-dir` or `--exact-diff-markdown` so the generated Markdown comes
from the standard exact-diff reporter.

## Benchmarking

Run the benchmark harness with:

```bash
.venv/bin/python benchmarks/gdsdiff/run_benchmarks.py --max-pairs 1 --mode windowed --tile-size 256 --output /tmp/gdsdiff_benchmark_smoke.json
```

Benchmark rows include wall-clock time plus comparator extraction/exact/total
timings. They are end-to-end comparator timings, not kernel-only numbers.

## GPU Polygon Evidence Engine

The target evidence engine is `gpu-polygon`: GPU numeric diff records plus CPU
Markdown formatting. Current implementation state:

- Rectangle-only inputs: native GDS parsing plus CUDA rectangle coverage
  fragments produce exact directional coverage records without calling
  `gdstk.boolean` in the default evidence path.
- General non-rectangle inputs: CUDA source-record deltas and canonical
  source-surplus component probes are available. The source-surplus probe
  emits compact directional records for unmatched source polygons. In the CUDA
  path, this probe is derived from GPU record-delta rows with representative
  polygon indices, so it does not require a full CPU canonicalization pass.
  A CUDA component-area pass can compute context-expanded directional area
  totals for those dirty components, but this is still not final exact evidence
  until GPU boundary-loop/component records are emitted.
  It is marked exact only when changed polygons are conservatively proven
  independent. When surplus polygons overlap, the report remains fail-closed with
  `status=unresolved`, `canonical_surplus_coverage_available=true`,
  `canonical_surplus_coverage_exact=false`, and `gpu_polygon_available=false`.
- CPU audit: `--audit-cpu-oracle` runs the CPU `gdstk` oracle explicitly and
  labels the result as audit evidence, not GPU evidence.

Do not describe shortcut/native-guard timings as GPU polygon evidence timings.
Reports must distinguish shortcut verdicts, GPU kernel activity, and CPU oracle
audit time.

## Chip And Wafer Guidance

Default to the smallest affected design scale. For chip edits, compare the chip
GDS exactly and inspect local diff components. For block or wafer assemblies,
prefer hierarchy/readback/profile checks unless the block or wafer geometry
itself changed. A wafer made from many repeated chip cells should not normally
be flattened into a full-wafer XOR just to validate a chip-level edit.
