from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path


def decimal_from_number(value: int | float | str | Decimal) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


@dataclass(frozen=True)
class GdsMetadata:
    path: Path
    unit_m: Decimal
    precision_m: Decimal
    cell_count: int
    top_names: tuple[str, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "unit_m": str(self.unit_m),
            "precision_m": str(self.precision_m),
            "cell_count": self.cell_count,
            "top_names": list(self.top_names),
        }


def read_gds_metadata(path: str | Path) -> GdsMetadata:
    import gdstk

    path = Path(path)
    lib = gdstk.read_gds(str(path))
    return GdsMetadata(
        path=path,
        unit_m=decimal_from_number(lib.unit),
        precision_m=decimal_from_number(lib.precision),
        cell_count=len(lib.cells),
        top_names=tuple(cell.name for cell in lib.top_level()),
    )
