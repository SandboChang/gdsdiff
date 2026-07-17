from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Literal

from .semantics import Backend, DifferenceDirection, EvidenceEngine, FailurePolicy, PathPolicy

LayerSelector = Literal["all"] | tuple["LayerSpec", ...]


def _decimal_or_none(value) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        result = value
    else:
        try:
            result = Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError(f"invalid decimal value: {value!r}") from exc
    if not result.is_finite() or result <= 0:
        raise ValueError(f"decimal value must be finite and positive: {value!r}")
    return result


@dataclass(frozen=True, order=True)
class LayerSpec:
    layer: int
    datatype: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.layer, int) or not isinstance(self.datatype, int):
            raise TypeError("layer and datatype must be integers")
        if self.layer < 0 or self.datatype < 0:
            raise ValueError("layer and datatype must be nonnegative")

    @classmethod
    def parse(cls, value: str | tuple[int, int] | "LayerSpec") -> "LayerSpec":
        if isinstance(value, cls):
            return value
        if isinstance(value, tuple):
            if len(value) != 2:
                raise ValueError("layer tuple must be (layer, datatype)")
            return cls(int(value[0]), int(value[1]))
        parts = str(value).strip().split(":")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(f"layer spec must have form L:D, got {value!r}")
        return cls(int(parts[0]), int(parts[1]))

    def to_json(self) -> dict[str, int]:
        return {"layer": self.layer, "datatype": self.datatype}


def parse_layer_selector(value: str | LayerSpec | Iterable[str | tuple[int, int] | LayerSpec]) -> LayerSelector:
    if value == "all":
        return "all"
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if not parts:
            raise ValueError("layer selector cannot be empty")
        layers = tuple(sorted({LayerSpec.parse(part) for part in parts}))
    elif isinstance(value, LayerSpec):
        layers = (value,)
    else:
        layers = tuple(sorted({LayerSpec.parse(part) for part in value}))
    if not layers:
        raise ValueError("layer selector cannot be empty")
    return layers


@dataclass(frozen=True)
class GridSpec:
    meters_per_dbu: Decimal | str | float | None = None
    source_old_precision_m: Decimal | str | float | None = None
    source_new_precision_m: Decimal | str | float | None = None
    selection_mode: str = "auto"
    rounding_mode: str = "nearest-ties-away-from-zero"
    max_coordinate: int | None = None
    max_snap_error_dbu: Decimal | str | float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "meters_per_dbu", _decimal_or_none(self.meters_per_dbu))
        object.__setattr__(self, "source_old_precision_m", _decimal_or_none(self.source_old_precision_m))
        object.__setattr__(self, "source_new_precision_m", _decimal_or_none(self.source_new_precision_m))
        object.__setattr__(self, "max_snap_error_dbu", _decimal_or_none(self.max_snap_error_dbu))
        if self.rounding_mode != "nearest-ties-away-from-zero":
            raise ValueError("unsupported rounding mode")
        if self.max_coordinate is not None and self.max_coordinate <= 0:
            raise ValueError("max_coordinate must be positive when set")

    def to_json(self) -> dict[str, object]:
        return {
            "meters_per_dbu": str(self.meters_per_dbu) if self.meters_per_dbu is not None else None,
            "source_old_precision_m": str(self.source_old_precision_m)
            if self.source_old_precision_m is not None
            else None,
            "source_new_precision_m": str(self.source_new_precision_m)
            if self.source_new_precision_m is not None
            else None,
            "selection_mode": self.selection_mode,
            "rounding_mode": self.rounding_mode,
            "max_coordinate": self.max_coordinate,
            "max_snap_error_dbu": str(self.max_snap_error_dbu) if self.max_snap_error_dbu is not None else None,
        }


@dataclass(frozen=True)
class AllowedRegion:
    layers: tuple[LayerSpec, ...]
    direction: DifferenceDirection | str = DifferenceDirection.BOTH
    bbox: tuple[int, int, int, int] | None = None
    whole_layer: bool = False
    halo_dbu: int = 0
    label: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        layers = tuple(sorted({LayerSpec.parse(layer) for layer in self.layers}))
        if not layers:
            raise ValueError("allowed region must list at least one layer")
        object.__setattr__(self, "layers", layers)
        object.__setattr__(self, "direction", DifferenceDirection.coerce(self.direction))
        if self.halo_dbu < 0:
            raise ValueError("allowed region halo_dbu must be nonnegative")
        if self.whole_layer:
            if self.bbox is not None:
                raise ValueError("whole-layer allowed region cannot also set bbox")
        elif self.bbox is None:
            raise ValueError("allowed region requires bbox unless whole_layer is true")
        else:
            x0, y0, x1, y1 = self.bbox
            if x0 >= x1 or y0 >= y1:
                raise ValueError("allowed region bbox must satisfy x0 < x1 and y0 < y1")

    def to_json(self) -> dict[str, object]:
        return {
            "layers": [layer.to_json() for layer in self.layers],
            "direction": self.direction.value,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "whole_layer": self.whole_layer,
            "halo_dbu": self.halo_dbu,
            "label": self.label,
            "reason": self.reason,
        }


def parse_allowed_region(value: str) -> AllowedRegion:
    """Parse ``L:D,DIRECTION,X0,Y0,X1,Y1[,HALO]`` into an allowance."""

    parts = [part.strip() for part in value.split(",")]
    if len(parts) not in {6, 7}:
        raise ValueError("allowed region must have form L:D,DIRECTION,X0,Y0,X1,Y1[,HALO]")
    layer = LayerSpec.parse(parts[0])
    direction = DifferenceDirection.coerce(parts[1])
    x0, y0, x1, y1 = (int(part) for part in parts[2:6])
    halo = int(parts[6]) if len(parts) == 7 else 0
    return AllowedRegion(
        layers=(layer,),
        direction=direction,
        bbox=(x0, y0, x1, y1),
        halo_dbu=halo,
    )


def parse_allowed_whole_layer(value: str) -> AllowedRegion:
    """Parse ``L:D,DIRECTION`` into a whole-layer allowance."""

    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError("whole-layer allowance must have form L:D,DIRECTION")
    return AllowedRegion(
        layers=(LayerSpec.parse(parts[0]),),
        direction=DifferenceDirection.coerce(parts[1]),
        whole_layer=True,
    )


@dataclass(frozen=True)
class CompareConfig:
    old_path: Path | str
    new_path: Path | str
    top: str | None = None
    old_top: str | None = None
    new_top: str | None = None
    require_top_name_match: bool = False
    layers: str | LayerSpec | Iterable[str | tuple[int, int] | LayerSpec] = "all"
    dbu: Decimal | str | float | None = None
    precision: Decimal | str | float | None = None
    path_policy: PathPolicy | str = PathPolicy.GDSTK
    backend: Backend | str = Backend.AUTO
    evidence_engine: EvidenceEngine | str = EvidenceEngine.AUTO
    tile_size: int = 1024
    allowed_regions: tuple[AllowedRegion, ...] = field(default_factory=tuple)
    allow_empty_selected_layer: bool = False
    strict: bool = True
    require_gpu: bool = False
    failure_policy: FailurePolicy | str = FailurePolicy.FAIL_ON_UNEXPECTED
    memory_budget_bytes: int | None = None
    geometry_cache_dir: Path | str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "old_path", Path(self.old_path))
        object.__setattr__(self, "new_path", Path(self.new_path))
        if self.geometry_cache_dir is not None:
            object.__setattr__(self, "geometry_cache_dir", Path(self.geometry_cache_dir))
        if self.top and (self.old_top or self.new_top):
            raise ValueError("top cannot be combined with old_top or new_top")
        if bool(self.old_top) != bool(self.new_top):
            raise ValueError("old_top and new_top must be supplied together")
        layers = parse_layer_selector(self.layers)
        object.__setattr__(self, "layers", layers)
        dbu = _decimal_or_none(self.dbu)
        precision = _decimal_or_none(self.precision)
        if dbu is not None and precision is not None and dbu != precision:
            raise ValueError("explicit dbu and precision must match in MVP semantics")
        object.__setattr__(self, "dbu", dbu)
        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "path_policy", PathPolicy.coerce(self.path_policy))
        backend = Backend.coerce(self.backend)
        if self.require_gpu and backend == Backend.CPU:
            raise ValueError("require_gpu cannot be used with backend='cpu'")
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "evidence_engine", EvidenceEngine.coerce(self.evidence_engine))
        object.__setattr__(self, "failure_policy", FailurePolicy.coerce(self.failure_policy))
        if self.tile_size <= 0:
            raise ValueError("tile_size must be positive")
        if self.memory_budget_bytes is not None and self.memory_budget_bytes <= 0:
            raise ValueError("memory_budget_bytes must be positive when set")
        object.__setattr__(self, "allowed_regions", tuple(self.allowed_regions))

    def to_json_dict(self) -> dict[str, object]:
        data = {
            "old_path": str(self.old_path),
            "new_path": str(self.new_path),
            "top": self.top,
            "old_top": self.old_top,
            "new_top": self.new_top,
            "require_top_name_match": self.require_top_name_match,
            "layers": "all" if self.layers == "all" else [layer.to_json() for layer in self.layers],
            "dbu": str(self.dbu) if self.dbu is not None else None,
            "precision": str(self.precision) if self.precision is not None else None,
            "path_policy": self.path_policy.value,
            "backend": self.backend.value,
            "evidence_engine": self.evidence_engine.value,
            "tile_size": self.tile_size,
            "allowed_regions": [region.to_json() for region in self.allowed_regions],
            "allow_empty_selected_layer": self.allow_empty_selected_layer,
            "strict": self.strict,
            "require_gpu": self.require_gpu,
            "failure_policy": self.failure_policy.value,
            "memory_budget_bytes": self.memory_budget_bytes,
        }
        if self.geometry_cache_dir is not None:
            data["geometry_cache_dir"] = str(self.geometry_cache_dir)
        return data
