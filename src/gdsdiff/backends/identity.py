from __future__ import annotations

from pathlib import Path

from .base import FileIdentityResult


class AcceleratedIdentityUnavailable(RuntimeError):
    pass


def compare_file_identity_accelerated(old_path: str | Path, new_path: str | Path) -> FileIdentityResult:
    """Compare equal-size file bytes through the best available accelerator.

    CUDA is the current implementation. This wrapper is intentionally backend
    neutral so a Metal implementation for Apple Silicon can be added without
    changing the comparator's API-level fast path.
    """
    errors: list[str] = []
    try:
        from .cuda_ctypes import compare_files_cuda

        result = compare_files_cuda(old_path, new_path)
        return FileIdentityResult(
            old_size=result.old_size,
            new_size=result.new_size,
            same=result.same,
            method=result.method,
        )
    except Exception as exc:
        errors.append(f"cuda: {exc}")
    raise AcceleratedIdentityUnavailable("; ".join(errors) or "no accelerated identity backend available")
