# Contributing

Use Python 3.10 or newer and the pinned `uv` release. On WSL use:

```bash
UV_PROJECT_ENVIRONMENT=.venv_wsl UV_LINK_MODE=copy uv sync --locked --no-dev
UV_PROJECT_ENVIRONMENT=.venv_wsl_test UV_LINK_MODE=copy uv sync --locked --extra dev
TMPDIR=/tmp .venv_wsl_test/bin/python -m pytest
```

Keep WSL test dependencies in `.venv_wsl_test`; `.venv_wsl` remains the clean
runtime environment. Use `.venv_mac` on macOS and `.venv` on native Linux.
Ordinary `pip` installs remain a required distribution-compatibility gate.

Changes to comparison semantics, report fields, exit codes, root exports, or
native acceleration must include focused tests and compatibility notes. Keep
CPU-only operation fully supported.

For acceleration changes, qualify the full geometry and inventory evidence
against the independent CPU oracle before reporting timings. For release
compatibility, compare verified revisions in separate pinned environments.
Record deliberate changes separately from regressions. Use synthetic GDS
fixtures in durable tests; keep external layouts out of the repository. See the
[benchmark instructions](benchmarks/gdsdiff/README.md) and keep the
[user guide](docs/gdsdiff.md), [report guide](docs/gdsdiff_report.md), and
[agent workflow](agents/gdsdiff.md) aligned with the shipped behavior.
