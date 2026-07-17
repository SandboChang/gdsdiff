# Contributing

Use Python 3.10 or newer and the pinned `uv` release. On WSL use:

```bash
UV_PROJECT_ENVIRONMENT=.venv_wsl UV_LINK_MODE=copy uv sync --locked --extra dev
TMPDIR=/tmp .venv_wsl/bin/python -m pytest
```

Use `.venv_mac` on macOS and `.venv` on native Linux. Ordinary `pip` installs
remain a required distribution-compatibility gate.

Changes to comparison semantics, report fields, exit codes, root exports, or
native acceleration must include focused tests and compatibility notes. Keep
CPU-only operation fully supported.
