from __future__ import annotations

import importlib.util
import shutil
from dataclasses import dataclass

from ._environment import get_environment
from .report import TOOL_VERSION

CUDA_ARCH_ENV = "GDSDIFF_CUDA_ARCHITECTURES"
LEGACY_CUDA_ARCH_ENV = "GDS_GPU_COMPARE_CUDA_ARCHITECTURES"


@dataclass(frozen=True)
class NativeCapabilities:
    tool_version: str
    core_extension_available: bool
    cuda_extension_available: bool
    cuda_ctypes_available: bool
    metal_ctypes_available: bool
    cmake_available: bool
    nvcc_available: bool
    cuda_architectures: tuple[str, ...]

    @property
    def cpu_available(self) -> bool:
        return True

    @property
    def cuda_available(self) -> bool:
        return self.cuda_extension_available or self.cuda_ctypes_available

    @property
    def metal_available(self) -> bool:
        return self.metal_ctypes_available

    def to_json(self) -> dict[str, object]:
        return {
            "tool_version": self.tool_version,
            "cpu_available": self.cpu_available,
            "core_extension_available": self.core_extension_available,
            "cuda_extension_available": self.cuda_extension_available,
            "cuda_ctypes_available": self.cuda_ctypes_available,
            "cuda_available": self.cuda_available,
            "metal_ctypes_available": self.metal_ctypes_available,
            "metal_available": self.metal_available,
            "cmake_available": self.cmake_available,
            "nvcc_available": self.nvcc_available,
            "cuda_architectures": list(self.cuda_architectures),
        }


def discover_native_capabilities() -> NativeCapabilities:
    return NativeCapabilities(
        tool_version=TOOL_VERSION,
        core_extension_available=_module_available("gdsdiff._gds_compare_core"),
        cuda_extension_available=_module_available("gdsdiff._gds_compare_cuda"),
        cuda_ctypes_available=_cuda_ctypes_available(),
        metal_ctypes_available=_metal_ctypes_available(),
        cmake_available=shutil.which("cmake") is not None,
        nvcc_available=shutil.which("nvcc") is not None,
        cuda_architectures=_cuda_architectures_from_env(),
    )


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _cuda_ctypes_available() -> bool:
    try:
        from .backends.cuda_ctypes import cuda_ctypes_available

        return cuda_ctypes_available()
    except Exception:
        return False


def _metal_ctypes_available() -> bool:
    try:
        from .backends.metal_ctypes import metal_ctypes_available

        return metal_ctypes_available()
    except Exception:
        return False


def _cuda_architectures_from_env() -> tuple[str, ...]:
    raw = get_environment(CUDA_ARCH_ENV, LEGACY_CUDA_ARCH_ENV, "") or ""
    return tuple(part.strip() for part in raw.replace(";", ",").split(",") if part.strip())
