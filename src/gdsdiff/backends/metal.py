from __future__ import annotations

import platform

from ..exact_oracle import ExactLayerTask, LayerExactDiff
from .base import BackendCapabilities
from .exact import UnsupportedExactComponent


class MetalExactUnavailable(RuntimeError):
    pass


class MetalManhattanRectExactComponentBackend:
    """Metal exact-component backend placeholder.

    The comparator's exact-component contract is intentionally shared by CUDA
    and Metal. This class provides the dispatch surface for Apple Silicon; the
    current implementation fails closed until a Metal shader implementation is
    added on macOS arm64.
    """

    def __init__(self, *, fallback_to_gdstk: bool = False) -> None:
        self.fallback_to_gdstk = fallback_to_gdstk

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(name="metal-manhattan-rect-exact-component", deterministic=True, requires_gpu=True)

    def compare_components(self, tasks: tuple[ExactLayerTask, ...]) -> tuple[LayerExactDiff, ...]:
        if platform.system() != "Darwin" or platform.machine() not in {"arm64", "aarch64"}:
            raise MetalExactUnavailable("Metal exact backend requires macOS on Apple Silicon")
        raise UnsupportedExactComponent("Metal exact component shader is not implemented yet")
