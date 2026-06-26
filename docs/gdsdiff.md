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
- Use `--require-gpu` to fail instead of falling back to CPU.

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
