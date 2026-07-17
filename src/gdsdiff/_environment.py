from __future__ import annotations

import os


def get_environment(primary: str, legacy: str, default: str | None = None) -> str | None:
    """Read a branded setting while honoring the 0.1.x legacy fallback."""

    value = os.environ.get(primary)
    if value is not None:
        return value
    return os.environ.get(legacy, default)
