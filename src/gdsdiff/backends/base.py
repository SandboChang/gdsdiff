from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from ..geometry import PolygonBuffer
from ..tiling import CpuTilePrefilterResult, TileConfig

if TYPE_CHECKING:
    from ..exact_oracle import ExactLayerTask, LayerExactDiff


@dataclass(frozen=True)
class BackendCapabilities:
    name: str
    deterministic: bool = True
    requires_gpu: bool = False


class PrefilterBackend(Protocol):
    def capabilities(self) -> BackendCapabilities:
        ...

    def compare_tile_contributors(
        self,
        old_buffer: PolygonBuffer,
        old_canonical_ids,
        new_buffer: PolygonBuffer,
        new_canonical_ids,
        config: TileConfig,
    ) -> CpuTilePrefilterResult:
        ...


@dataclass(frozen=True)
class FileIdentityResult:
    old_size: int
    new_size: int
    same: bool
    method: str


class FileIdentityBackend(Protocol):
    def capabilities(self) -> BackendCapabilities:
        ...

    def compare_files(self, old_path: str | Path, new_path: str | Path) -> FileIdentityResult:
        ...


class NativeGeometryBackend(Protocol):
    """Backend boundary for the planned non-gdstk fast geometry path.

    Implementations should consume a native GDS-derived intermediate
    representation rather than Python `gdstk.Polygon` objects. CUDA, Metal, and
    portable CPU backends should share this contract.
    """

    def capabilities(self) -> BackendCapabilities:
        ...


class ExactComponentBackend(Protocol):
    """Backend boundary for polygon-level exact evidence tasks.

    Implementations consume independent exact component tasks and return the
    same `LayerExactDiff` fragments as the `gdstk` oracle. CUDA and Metal
    backends should target this contract while the CPU/gdstk backend remains
    the reference oracle.
    """

    def capabilities(self) -> BackendCapabilities:
        ...

    def compare_components(self, tasks: tuple["ExactLayerTask", ...]) -> tuple["LayerExactDiff", ...]:
        ...
