"""Backend implementations for GPU compare accelerators."""

from .base import ExactComponentBackend, PrefilterBackend
from .cpu import CpuTileBackend
from .cuda import CudaBackendUnavailable, CudaTileBackend
from .exact import (
    CudaManhattanRectExactComponentBackend,
    GdstkExactComponentBackend,
    ManhattanRectExactComponentBackend,
    UnsupportedExactComponent,
)
from .metal import MetalExactUnavailable, MetalManhattanRectExactComponentBackend

__all__ = [
    "CpuTileBackend",
    "CudaBackendUnavailable",
    "CudaTileBackend",
    "CudaManhattanRectExactComponentBackend",
    "ExactComponentBackend",
    "GdstkExactComponentBackend",
    "ManhattanRectExactComponentBackend",
    "MetalExactUnavailable",
    "MetalManhattanRectExactComponentBackend",
    "PrefilterBackend",
    "UnsupportedExactComponent",
]
