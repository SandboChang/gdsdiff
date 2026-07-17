# Agent Notes

This repository is the standalone `gdsdiff` GDSII comparison library. Durable
implementation belongs under `src/gdsdiff/`; tests belong under `tests/`.
Read `agents/gdsdiff.md` before changing comparator semantics, reports,
artifacts, acceleration, or agent-facing diff workflows.

## Environment

- Identify the host OS and architecture before environment-sensitive work.
- Use `.venv_wsl/bin/python` for normal imports and runtime development on
  Windows 11 through WSL, and `.venv_wsl_test/bin/python` for every test run.
  Use `.venv_mac/bin/python` on macOS, and `.venv/bin/python` on native Linux.
- Do not use the base system Python for imports, builds, or tests.
- CPU behavior must remain usable without CUDA, Metal, CMake, or a compiler.
- Use the exact `uv` version required by `pyproject.toml`. On WSL under
  `/mnt/f`, keep `.venv_wsl` synchronized without test extras, target
  `.venv_wsl_test` with development/test extras, set `UV_LINK_MODE=copy`, and
  set `TMPDIR=/tmp` for pytest. Use `.venv_mac` on macOS and `.venv` on Linux.

## Development Rules

- Work on the current branch unless the user explicitly requests another one.
- Preserve neutral report wording and stable exit-code semantics.
- Treat report schemas, CLI behavior, and root exports as public contracts.
- Do not commit generated GDS, build products, native binaries, caches, or
  benchmark results.
- Use `.tmp/` for disposable local work and ignored `results/` for retained
  local qualification evidence.
- Run `python -m pytest` plus distribution build/inspection for packaging work.
- For performance-sensitive changes, compare exact semantics first, then use a
  warm-up and at least seven measured runs.
- CUDA and Metal are optional runtime capabilities. Never make their toolchains
  mandatory installation dependencies.
