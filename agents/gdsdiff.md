# GDSDiff Instructions

## Durable Command And Implementation

- Use `gdsdiff` as the durable general-purpose command for arbitrary
  before/after GDS comparisons.
- Durable comparator APIs, backends, native sources, report builders, schemas,
  and reusable workflow helpers belong under `scgds/gdsdiff/` or another
  appropriate `scgds/` subpackage.
- Root-level scripts may call the library, but must not become the canonical
  implementation.
- The earlier GPU-comparator command name is not the active entry point.

## Invocation

- During agent workflows, do not use console-script shims such as `gdsdiff` or
  `.venv/bin/gdsdiff`; they can point at stale virtual environments.
- Use the repo-local Python module entrypoint:
  - Windows 11 through WSL: `.venv_wsl/bin/python -m scgds.gdsdiff`
  - macOS: `.venv_mac/bin/python -m scgds.gdsdiff`
  - Native Linux: `.venv/bin/python -m scgds.gdsdiff`
- If the matching repo-local interpreter is missing or cannot import
  `scgds.gdsdiff`, tell the user and skip the diff check.
- Do not install packages, repair shims, or create a new environment unless the
  user explicitly asks.

## Comparator Role And Report Semantics

- Treat the GDS comparator as a neutral evidence generator.
- It must report exact factual differences, not decide whether those
  differences are intended or unintended.
- Default user-facing evidence should use neutral statuses and directions such
  as `identical`, `differences-found`, `coverage_before_only`,
  `coverage_after_only`, `inventory_before_only`, and `inventory_after_only`.
- Avoid judgement labels such as expected/unexpected in default compact diff
  Markdown. Classification belongs to the agent's validation analysis, not to
  the comparator evidence itself.

## Exact-Diff Artifacts

- Default exact-diff artifacts are a compact Markdown report plus a full-vertex
  `*_diff_vertices.jsonl` sidecar.
- The Markdown should stay small enough for staged LLM review and include
  sidecar ids, hashes, bbox, area, and short previews.
- The sidecar is the lossless source for full changed vertices.
- Geometry validation that needs an output diff Markdown file should generate it
  through `gdsdiff`.
- Treat generated diff Markdown and JSONL sidecars as temporary review
  artifacts: read the staged sections needed for the decision, record only the
  relevant facts in durable notes/history, then discard the generated artifacts.
- If such artifacts were created inside the repository, discard them by moving
  them into a timestamped `.dump/` folder after recording the relevant facts.

## How To Read Exact-Diff Markdown

- Minimize LLM context by reading staged sections instead of blindly ingesting
  the whole file.
- First read:
  - `Comparator Evidence`
  - `Agent Minimal Read`
  - `Agent Review Checklist`
  - `Layer Summary`
- Expand into `Polygon Components` only when every coverage component is needed,
  when a changed bbox is outside the intended feature region, or when
  direction/area evidence is too coarse to classify the component.
- Expand into `Input Polygon Inventory Delta` only when coverage is unchanged
  but polygon inventory changed, or when split/merge/refracture evidence is
  under review.

## Acceleration Rules

- `gdsdiff` should default to automatic GPU acceleration when available and CPU
  exact fallback when no suitable GPU backend is present.
- Do not make the fast geometry design CUDA-specific.
- Keep durable accelerator contracts backend-neutral so CUDA, Apple
  Silicon/Metal, and portable CPU/native implementations can share the same
  parser/intermediate representation boundary.
- Treat `gdstk` as the exact CPU oracle and behavior reference.
- Do not rely on `gdstk` flattening as the required fast-path engine when a
  custom native/GPU pipeline is needed.
- For GPU operations, release device memory at the end of each compare
  operation so repeated runs do not accumulate VRAM.
- Reports should include a cleanup timing or explicit cleanup failure field,
  such as `cuda_release_s` for the CUDA backend.
- Optional CPU audit work must run after GPU cleanup, not while retaining the
  GPU context.
