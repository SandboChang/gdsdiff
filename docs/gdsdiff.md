# gdsdiff

`gdsdiff` is the general-purpose GDS comparison command for this repository.
It reports exact factual differences between two GDS files and does not judge
whether those differences are intended.

## Basic Use

```bash
python -m gdsdiff inspect before.gds
python -m gdsdiff before.gds after.gds --top TOP --layers all --output-dir comparison
```

Use `--old-top OLD --new-top NEW` if the top cell names differ, or
`--layers 1:0,51:0` to select layer/datatype pairs. Omit `--top` only when the
intended selection is unambiguous. From a checkout use the repo-local
interpreter described in [AGENTS.md](../AGENTS.md).

The command writes the standard artifact set:

- `report.json`: machine-readable status, timings, backend details, and artifact paths.
- `exact_diff.md`: compact LLM-readable factual diff report.
- `diff_vertices.jsonl`: full vertices for every changed record referenced by Markdown.

## Backend Selection

By default, `gdsdiff` uses `--backend auto`.

- If a supported CUDA or Metal backend is available, it uses the neutral `gpu-polygon`
  evidence path.
- If no suitable GPU backend is available, it falls back to the CPU `gdstk`
  oracle while keeping the same neutral evidence contract.
- If a requested GPU comparison fails, the command reports the stage and tries
  the exact CPU oracle. With `--require-gpu`, a CPU-only result still returns
  exit 2. In JSON, `backend.gpu_work_performed` distinguishes actual GPU work
  from runtime availability, and `exact_fallback.cpu_evidence_status` records
  the separate CPU verdict. CPU audit and correction after real GPU work are
  reported separately from CPU-only fallback.

```bash
python -m gdsdiff before.gds after.gds --backend cuda --require-gpu --output-dir cuda_diff
python -m gdsdiff before.gds after.gds --backend cpu --evidence-engine gdstk-oracle --output-dir cpu_diff
```

The first command requires actual device work. The second is an independent
CPU-only reference that reads the original files, not a reuse of native GPU
inputs. Use the same top, layers, and allowed regions for both. Do not mix
`--require-gpu` with `--exact` or `--evidence-engine gdstk-oracle`.

The native GPU evidence path supports translation and unit-magnification
single references rotated by multiples of 90 degrees with no reflection or
absolute transform flags. Array references with nonidentity transforms and
other unsupported records have explicit diagnostics.

On the native GPU evidence path, `timings.native_parse_s` is the wall time to
prepare both inputs, before upload. Per-input `old_native_polygon_*` and
`new_native_polygon_*` fields break down reading, scanning, normalization, layer
mapping, placement planning, and packing. The inputs can be prepared in
parallel, so those times should not be added to get pair wall time. Uncached
CUDA preparation skips host polygon hashes; the CUDA record comparison hashes
the vertices on device. For that comparison, `h2d_upload_s` is the time spent in
host-to-device copy calls and is already included in `gpu_source_record_delta_s`.
The report also separates `layer_summary_s` (layer presence and polygon counts)
from `policy_classification_s` (applying allowed regions).

For sparse edits on large inputs, the CPU coverage audit can use an aligned
record reduction after independently reading both full files with `gdstk`. It
compares integer vertices and retains all common records whose bounds touch a
changed polygon. The exact backend reports when this reduction was used.

On CUDA, when the boundary engine detects nested loops that would require
exact hole correction, it can stop the remaining assembly early. This also
works when sparse edits change polygon counts or vertex offsets: the engine
verifies all omitted records, keeps touching same-layer context, and computes
exact inventory from the retained records. `gdstk` computes coverage on CPU.
The report uses `gpu-polygon-cuda-record-delta-sparse-gdstk` and records the
attempt in `native_sparse_coverage_s`. Other edits, or those that do not
qualify for the exact reduction, keep the normal coverage path.

An explicit `--audit-cpu-oracle` rereads both files independently after GPU
cleanup. For the different-record-layout reduction it checks both full
directional vertices and inventory, not just areas. A mismatch fails closed
and suppresses unverified native inventory.

`gdsdiff` discards a Boolean attempt's entire output if `gdstk` reports a
numerical failure. In the CLI, attempts replaced by verified recovery are
recorded in `exact_fallback.gdstk_boolean_diagnostics` and stay off stderr
when native capture is safe. Use `--output-dir` or `--json-report` to inspect
them. A failed recovery remains visible on stderr even if an independent CPU
fallback completes. If no verified result remains, the comparison exits 2
with `incomplete-evidence` and does not write a geometry sidecar or diff GDS.

`gdsdiff` is the active command name. The earlier GPU-comparator command name
is no longer an active entry point.

## Status Values

Default `gdsdiff` statuses are neutral:

- `identical`: no exact differences found.
- `differences-found`: one or more exact differences found.
- `unsupported-input`: unsupported GDS constructs prevented complete evidence.
- `incomplete-evidence`: evidence was generated but is not complete enough for
  acceptance review.
- `internal-mismatch`: GPU evidence disagreed with explicit CPU audit evidence.

Finding a difference is not a failure of the design. It is the comparator's
task. Intent review is left to the user or reviewing LLM.
Default neutral CLI evidence exits 0 when identical, 1 when differences are
found, and 2 when evidence is incomplete, invalid, or does not meet a requested
backend requirement. Running with allowances still reports neutral differences;
the policy-aware CPU oracle can return 0 for allowed-only changes.

## Review Protocol

Start by reading only these sections of `exact_diff.md`:

1. `Comparator Evidence`
2. `Agent Minimal Read`
3. `Layer Summary`

Expand `Polygon Components` for exact changed coverage component bboxes, areas,
hashes, and sidecar ids. Expand `Input Polygon Inventory Delta` when source
polygon formation matters, such as split/merge/refracture or vertex-level input
changes. Read `diff_vertices.jsonl` only for the specific sidecar ids that need
full vertex detail.
