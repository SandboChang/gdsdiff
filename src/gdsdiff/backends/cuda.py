from __future__ import annotations

import importlib

from ..geometry import PolygonBuffer
from ..tiling import CpuTilePrefilterResult, TileConfig
from .base import BackendCapabilities


class CudaBackendUnavailable(RuntimeError):
    pass


class CudaTileBackend:
    def __init__(self) -> None:
        try:
            self._extension = importlib.import_module("gdsdiff._gds_compare_cuda")
        except ImportError as exc:
            raise CudaBackendUnavailable("CUDA backend extension is not available") from exc

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(name="cuda", deterministic=True, requires_gpu=True)

    def compare_tile_contributors(
        self,
        old_buffer: PolygonBuffer,
        old_canonical_ids,
        new_buffer: PolygonBuffer,
        new_canonical_ids,
        config: TileConfig,
    ) -> CpuTilePrefilterResult:
        if not hasattr(self._extension, "compare_tile_contributors"):
            raise CudaBackendUnavailable("CUDA backend extension lacks compare_tile_contributors")
        return self._extension.compare_tile_contributors(
            old_buffer,
            old_canonical_ids,
            new_buffer,
            new_canonical_ids,
            config,
        )
