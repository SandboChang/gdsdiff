from __future__ import annotations

from pathlib import Path

from .base import FileIdentityResult


class AcceleratedIdentityUnavailable(RuntimeError):
    pass


def compare_file_identity_accelerated(
    old_path: str | Path,
    new_path: str | Path,
    *,
    preferred_backend: str | None = None,
) -> FileIdentityResult:
    """Compare equal-size file bytes through the best available accelerator.

    The wrapper is backend neutral so CUDA and Metal can share the comparator's
    API-level fast path.
    """
    errors: list[str] = []
    backend_order = _backend_order(preferred_backend)
    for backend in backend_order:
        try:
            if backend == "cuda":
                from .cuda_ctypes import compare_files_cuda

                result = compare_files_cuda(old_path, new_path)
            elif backend == "metal":
                from .metal_ctypes import compare_files_metal

                result = compare_files_metal(old_path, new_path)
            else:
                continue
            return FileIdentityResult(
                old_size=result.old_size,
                new_size=result.new_size,
                same=result.same,
                method=result.method,
            )
        except Exception as exc:
            errors.append(f"{backend}: {exc}")
    raise AcceleratedIdentityUnavailable("; ".join(errors) or "no accelerated identity backend available")


def _backend_order(preferred_backend: str | None) -> tuple[str, ...]:
    if preferred_backend == "metal":
        return ("metal", "cuda")
    if preferred_backend == "cuda":
        return ("cuda", "metal")
    return ("cuda", "metal")
