# GDS GPU Comparator Report

Reports use schema `gds-gpu-compare-report-v1`.

Important top-level fields:

- `status`: one of `equivalent`, `expected-differences-only`,
  `unexpected-differences`, `unresolved`, `invalid-input`, `backend-error`, or
  `artifact-error`.
- `exit_code_category`: the expected process exit code category, `0`, `1`, or
  `2`.
- `exact_verification_complete`: true only when exact verification reached a
  complete verdict.
- `old_input` and `new_input`: source GDS metadata and extraction stats.
- `semantics`: grid, layer, backend, path, and failure-policy settings.
- `selection`: selected top cells and layers.
- `backend`: requested backend, discovered capabilities, selected backend, and
  fallback reason.
- `prefilter`: broad-phase candidate-tile summary.
- `exact_fallback`: full or windowed exact verification details.
- `layers`: per-layer geometry evidence.
- `allowed_policy`: applied expected-difference policy.
- `unexpected_summary`: expected and unexpected total areas.
- `timings`: extraction, exact, and total timings.
- `artifacts`: emitted JSON, diff GDS, Markdown, or workflow artifact paths.

Layer entries include:

- `layer`, `datatype`
- `present_old`, `present_new`
- `polygon_count_old`, `polygon_count_new`
- `candidate_tile_count`, `exact_window_count`
- `xor_area_dbu2`
- `old_minus_new_area_dbu2`
- `new_minus_old_area_dbu2`
- `expected_area_dbu2`
- `unexpected_area_dbu2`
- `components`

Component entries include direction, bbox, exact area, and classification. Use
component bboxes and areas for local review; total layer area alone is only
triage.

Exact diff Markdown files are intended for staged LLM review, not linear
full-file reading. The default read protocol for agents is:

1. Read only `Comparator Evidence`, `Agent Minimal Read`, and `Layer Summary`.
   These sections are the minimum factual digest: verdict, exactness, total
   coverage area, inventory-change count, affected layers, union bboxes, and
   the largest changed coverage components.
2. Read `Polygon Components` only when the digest shows coverage changed and
   the agent needs every component bbox, area, direction, hash, and vertex
   preview.
3. Read `Input Polygon Inventory Delta` only when exact coverage area is zero
   but `polygon_inventory_delta_count` is nonzero, or when split, merge,
   fracture, or vertex-order changes are specifically under review.
4. Do not ingest full Markdown files blindly. If the digest is enough to judge
   intent, stop there and cite the sections used.

Workflow wrapper reports use a separate `result.json` around the comparator
report. It records one CI/agent verdict, artifact paths, rebuild command
result, archived backup path, optional assembly profile results, and any error.
