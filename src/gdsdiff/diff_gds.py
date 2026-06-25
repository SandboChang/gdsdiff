from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gdstk

from .policy import ClassifiedComponent, PolicyClassificationResult


@dataclass(frozen=True)
class DiffGdsWriteResult:
    path: Path
    component_count: int
    category_cells: tuple[str, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "component_count": self.component_count,
            "category_cells": list(self.category_cells),
        }


def write_diff_gds(
    policy_result: PolicyClassificationResult,
    path: str | Path,
    *,
    unit: float = 1e-6,
    precision: float = 1e-9,
    top_cell_name: str = "GDS_GPU_COMPARE_DIFF",
) -> DiffGdsWriteResult:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lib = gdstk.Library(unit=unit, precision=precision)
    top = lib.new_cell(top_cell_name)
    cells: dict[str, gdstk.Cell] = {}
    component_count = 0
    for component in _iter_components(policy_result):
        cell_name = _category_cell_name(component)
        cell = cells.get(cell_name)
        if cell is None:
            cell = lib.new_cell(cell_name)
            cells[cell_name] = cell
            top.add(gdstk.Reference(cell))
        cell.add(
            gdstk.Polygon(
                component.points,
                layer=component.layer.layer,
                datatype=component.layer.datatype,
            )
        )
        component_count += 1
    lib.write_gds(path)
    return DiffGdsWriteResult(path=path, component_count=component_count, category_cells=tuple(sorted(cells)))


def _iter_components(policy_result: PolicyClassificationResult) -> tuple[ClassifiedComponent, ...]:
    components = []
    for layer in policy_result.layers:
        components.extend(layer.components)
    return tuple(sorted(components, key=lambda component: (component.classification, component.direction.value, component.layer, component.bbox)))


def _category_cell_name(component: ClassifiedComponent) -> str:
    return f"{component.classification}_{component.direction.value}".upper().replace("-", "_")
