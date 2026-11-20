# gdsdiff

`gdsdiff` is the general-purpose GDS comparison command for this repository.
It reports exact factual differences between two GDS files and does not judge
whether those differences are intended.

## Basic Use

```bash
gdsdiff before.gds after.gds --output-dir /tmp/gdsdiff_my_check
```

The command writes the standard artifact set:

- `report.json`: machine-readable status, timings, backend details, and artifact paths.
- `exact_diff.md`: compact LLM-readable factual diff report.
- `diff_vertices.jsonl`: full vertices for every changed record referenced by Markdown.

## Backend Selection

By default, `gdsdiff` uses `--backend auto`.

- If a supported CUDA backend is available, it uses the neutral `gpu-polygon`
  evidence path.
- If no suitable GPU backend is available, it falls back to the CPU `gdstk`
  oracle while keeping the same neutral evidence contract.
- If a requested GPU comparison fails, the command reports the stage and tries
  the exact CPU oracle. With `--require-gpu`, a CPU-only result still returns
  exit 2. In JSON, `backend.gpu_work_performed` distinguishes actual GPU work
  from runtime availability, and `exact_fallback.cpu_evidence_status` records
  the separate CPU verdict. CPU audit and correction after real GPU work are
  reported separately from CPU-only fallback.

The native GPU evidence path supports translation and unit-magnification
single references rotated by multiples of 90 degrees with no reflection or
absolute transform flags. Array references with nonidentity transforms and
other unsupported records have explicit diagnostics.

For sparse edits on large inputs, the CPU coverage audit can use an aligned
record reduction after independently reading both full files with `gdstk`. It
compares integer vertices and retains all common records whose bounds touch a
changed polygon. The exact backend reports when this reduction was used.

When nested boundaries need hole-aware correction, supported native inputs can
reuse their already-extracted polygon buffers. An explicit `--audit-cpu-oracle`
still rereads both files independently and checks the corrected vertices, not
just the areas. A mismatch in that check fails closed.

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
