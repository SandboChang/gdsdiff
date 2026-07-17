from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from .native import nvcc_architecture_flags, resolve_nvcc_path
from .native_cache import native_cache_dir


class CudaPreparationError(RuntimeError):
    """Raised when the optional CUDA backend cannot be prepared and verified."""


@dataclass(frozen=True)
class CudaToolchainReport:
    available: bool
    platform: str
    machine: str
    nvcc_path: str | None
    nvcc_version: str | None
    host_compiler: str | None
    driver_version: str | None
    devices: tuple[str, ...]
    architectures: tuple[str, ...]
    compile_flags: tuple[str, ...]
    cache_path: str
    reason: str | None = None

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CudaPreparationReport:
    ready: bool
    built: bool
    reused: bool
    verified: bool
    library_path: str
    toolchain: CudaToolchainReport

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def inspect_cuda_toolchain() -> CudaToolchainReport:
    """Inspect CUDA without importing, compiling, or loading the backend."""

    nvcc = resolve_nvcc_path()
    version = _command_output((nvcc, "--version")) if nvcc else None
    smi = _command_output(
        (
            "nvidia-smi",
            "--query-gpu=name,compute_cap,driver_version",
            "--format=csv,noheader,nounits",
        )
    )
    devices = tuple(line.strip() for line in (smi or "").splitlines() if line.strip())
    driver = None
    if devices:
        parts = [part.strip() for part in devices[0].split(",")]
        driver = parts[-1] if len(parts) >= 3 else None
    architectures = tuple(
        part.strip() for part in os.environ.get("GDSDIFF_CUDA_ARCHITECTURES", "").replace(";", ",").split(",") if part.strip()
    )
    available = nvcc is not None and bool(devices)
    reason = None
    if nvcc is None:
        reason = "nvcc was not found in PATH, CUDA_HOME, /usr/local/cuda, or /opt/cuda"
    elif not devices:
        reason = "no NVIDIA GPU/driver was visible through nvidia-smi"
    return CudaToolchainReport(
        available=available,
        platform=platform.system(),
        machine=platform.machine(),
        nvcc_path=nvcc,
        nvcc_version=version,
        host_compiler=_which("g++"),
        driver_version=driver,
        devices=devices,
        architectures=architectures,
        compile_flags=nvcc_architecture_flags(),
        cache_path=str(native_cache_dir()),
        reason=reason,
    )


def prepare_cuda() -> CudaPreparationReport:
    """Build the CUDA backend in the user cache and run a device smoke test."""

    try:
        toolchain = inspect_cuda_toolchain()
    except (OSError, ValueError) as exc:
        raise CudaPreparationError(str(exc)) from exc
    if not toolchain.available:
        raise CudaPreparationError(toolchain.reason or "CUDA toolchain is unavailable")
    try:
        from .backends.cuda_ctypes import compare_files_cuda, prepared_cuda_library_path

        library = prepared_cuda_library_path()
        existed = library.is_file()
        with tempfile.TemporaryDirectory(prefix="gdsdiff-cuda-smoke-") as directory:
            left = Path(directory) / "left.bin"
            right = Path(directory) / "right.bin"
            left.write_bytes(b"gdsdiff-cuda-smoke")
            right.write_bytes(b"gdsdiff-cuda-smoke")
            result = compare_files_cuda(left, right)
            if not result.same:
                raise CudaPreparationError("CUDA smoke comparison returned a mismatch")
    except CudaPreparationError:
        raise
    except Exception as exc:
        raise CudaPreparationError(str(exc)) from exc
    if not library.is_file():
        raise CudaPreparationError("CUDA preparation did not publish the expected native library")
    built = not existed
    return CudaPreparationReport(
        ready=True,
        built=built,
        reused=not built,
        verified=True,
        library_path=str(library),
        toolchain=toolchain,
    )


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="gdsdiff-prepare-cuda", description="Build and verify the optional gdsdiff CUDA backend.")
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable report.")
    args = parser.parse_args(argv)
    try:
        report = prepare_cuda()
    except CudaPreparationError as exc:
        if args.json:
            print(json.dumps({"ready": False, "error": str(exc)}, indent=2, sort_keys=True))
        else:
            print(f"gdsdiff-prepare-cuda: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report.to_json(), indent=2, sort_keys=True))
    else:
        action = "built" if report.built else "reused"
        print(f"CUDA backend ready ({action}): {report.library_path}")
    return 0


def _command_output(command: tuple[str | None, ...]) -> str | None:
    if not command or command[0] is None:
        return None
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    output = (completed.stdout or completed.stderr).strip()
    return output if completed.returncode == 0 and output else None


def _which(command: str) -> str | None:
    import shutil

    return shutil.which(command)


if __name__ == "__main__":
    raise SystemExit(main())
