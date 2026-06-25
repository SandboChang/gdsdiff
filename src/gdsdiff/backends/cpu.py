from __future__ import annotations

from ..geometry import PolygonBuffer
from ..tiling import CpuTilePrefilterResult, TileConfig, run_cpu_tile_prefilter
from .base import BackendCapabilities


class CpuTileBackend:
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(name="cpu", deterministic=True, requires_gpu=False)

    def compare_tile_contributors(
        self,
        old_buffer: PolygonBuffer,
        old_canonical_ids,
        new_buffer: PolygonBuffer,
        new_canonical_ids,
        config: TileConfig,
    ) -> CpuTilePrefilterResult:
        return run_cpu_tile_prefilter(old_buffer, old_canonical_ids, new_buffer, new_canonical_ids, config)
