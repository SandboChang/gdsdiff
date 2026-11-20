# gdsdiff Report

Reports use schema `gdsdiff-report-v1`.

Important top-level fields:

- `status`: comparator/policy status. Default neutral `gdsdiff` statuses are
  `identical`, `differences-found`, `unsupported-input`,
  `incomplete-evidence`, and `internal-mismatch`; policy-aware flows can also
  report `equivalent`, `expected-differences-only`, `unexpected-differences`,
  `unresolved`, `invalid-input`, `backend-error`, or `artifact-error`.
- `exit_code_category`: the expected process exit code category, `0`, `1`, or
  `2`.
- `exact_verification_complete`: true only when exact verification reached a
  complete verdict.
- `old_input` and `new_input`: source GDS metadata and extraction stats.
- `semantics`: grid, layer, backend, path, and failure-policy settings.
- `selection`: selected top cells and layers.
- `backend`: requested and selected backends, discovered capabilities, and
  whether device work was performed.
- `prefilter`: broad-phase candidate-tile summary.
- `exact_fallback`: coverage method, failures, and CPU fallback or audit details.
- `layers`: per-layer geometry evidence.
- `allowed_policy`: applied expected-difference policy.
- `unexpected_summary`: expected and unexpected total areas.
- `timings`: extraction, exact, and total timings.
- `artifacts`: emitted JSON, diff GDS, Markdown, or workflow artifact paths.

## Check Exactness And GPU Use

Check `exact_verification_complete` together with `status` and
`exit_code_category`. A CPU-only fallback does not satisfy
`--require-gpu`; use `backend.gpu_work_performed` and
`backend.gpu_requirement_met`, not availability alone. The failure stage and
reason appear in `exact_fallback.failures`; `cpu_evidence_status` can record
the independent CPU verdict when the requested GPU run failed.

`exact_fallback.mode` tells you where coverage came from. In
`gpu-polygon-cuda-record-delta-sparse-gdstk`, CUDA compares source polygons
and `gdstk` computes exact coverage. This is not a CPU-only fallback.
`exact_fallback.sparse_coverage.coverage_scope` distinguishes the aligned
shortcut from `full-layout-verified-context-reduction`. The latter reports
the exterior record proof, retained polygon counts, and the counts of
partitioned directions and reflected components. A recovered Boolean
warning may still appear in the runtime log. Recovery requires exact loop
agreement; a declined reduction is not itself a failed GPU stage.

For the verified-context route, `cpu_oracle_audit.coverage_match` and
`cpu_oracle_audit.inventory_match` report the two independent checks.
`internal-mismatch` is an audit failure, even when verified CPU coverage is
available. Unverified native inventory is withheld.

## Read Timings

- `native_parse_s`: wall time to prepare both inputs. Per-input native stage
  timings can overlap; do not add them together.
- `gpu_source_record_delta_s`: wall time for record comparison, including
  host packing and transfers. `h2d_upload_s` is already inside it on this
  route; CUDA event times are not process wall time.
- `native_sparse_coverage_s`: time spent trying the sparse `gdstk` route,
  including declines. Use the mode to check whether its result was used.
- `native_sparse_proof_s` and `native_sparse_boolean_s`: proof (including
  retained inventory) and coverage on the verified-context route. Both are
  already included in `native_sparse_coverage_s`.
- `layer_summary_s`, `policy_classification_s`, `polygon_inventory_delta_s`,
  and `markdown_format_s`: host work after extraction, including the full
  vertex sidecar in the Markdown timing.
- `cpu_oracle_audit_s` is separate from GPU work; `cuda_release_s` or
  `metal_release_s` reports device cleanup. Measure CLI wall time externally
  to include startup, JSON report writing, and process teardown.

## Review Geometry

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

In `diff_vertices.jsonl`, check coverage separately from the source-polygon
inventory, including multiplicity. The sidecar is authoritative for vertices.
Different exact engines may start a loop at another vertex or reverse it;
that can change a vertex hash without changing any geometry. When comparing
engines, compare full vertex sequences modulo cyclic rotation or reversal,
grouped by direction, layer, and datatype. Do not treat matching areas alone
as proof of equal geometry or discard meaningful vertices.

For vertex parity, use the same allowed-region settings in both runs.
Classification can split coverage into new fragments. The unclassified
sidecar is the best reference for the underlying directional coverage.

Exact diff Markdown files are intended for staged LLM review, not linear
full-file reading. The default read protocol for agents is:

1. Read only `Comparator Evidence`, `Agent Minimal Read`, `Agent Review Checklist`,
   and `Layer Summary`.
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
