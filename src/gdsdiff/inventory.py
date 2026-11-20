from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import gdstk

from .config import LayerSpec, parse_layer_selector
from .gds_metadata import GdsMetadata, decimal_from_number


@dataclass(frozen=True)
class RecordInventory:
    polygons: int = 0
    paths: int = 0
    labels: int = 0
    references: int = 0
    cells: int = 0

    def to_json(self) -> dict[str, int]:
        return {
            "polygons": self.polygons,
            "paths": self.paths,
            "labels": self.labels,
            "references": self.references,
            "cells": self.cells,
        }


@dataclass(frozen=True)
class GdsInventory:
    metadata: GdsMetadata
    selected_top: str
    reachable_cells: tuple[str, ...]
    layers: tuple[LayerSpec, ...]
    records: RecordInventory
    ignored_record_counts: dict[str, int]

    def to_json(self) -> dict[str, object]:
        return {
            "metadata": self.metadata.to_json(),
            "selected_top": self.selected_top,
            "reachable_cells": list(self.reachable_cells),
            "layers": [layer.to_json() for layer in self.layers],
            "records": self.records.to_json(),
            "ignored_record_counts": dict(self.ignored_record_counts),
        }


@dataclass(frozen=True)
class SelectedLayerPresence:
    layer: LayerSpec
    present_old: bool
    present_new: bool

    def to_json(self) -> dict[str, object]:
        return {
            "layer": self.layer.layer,
            "datatype": self.layer.datatype,
            "present_old": self.present_old,
            "present_new": self.present_new,
        }


def inventory_gds(path: str | Path, *, top_name: str | None = None) -> GdsInventory:
    """Inventory stored records and geometry layers reachable from a cell."""
    path = Path(path)
    lib = gdstk.read_gds(str(path))
    top = select_top_cell(lib, top_name=top_name)
    reachable = _reachable_cells(top)
    final_layers = _inventory_layers(reachable)
    records = _record_inventory(reachable)
    metadata = GdsMetadata(
        path=path,
        unit_m=decimal_from_number(lib.unit),
        precision_m=decimal_from_number(lib.precision),
        cell_count=len(lib.cells),
        top_names=tuple(cell.name for cell in lib.top_level()),
    )
    return GdsInventory(
        metadata=metadata,
        selected_top=top.name,
        reachable_cells=tuple(sorted(cell.name for cell in reachable)),
        layers=final_layers,
        records=records,
        ignored_record_counts={"labels": records.labels},
    )


def select_top_cell(lib: gdstk.Library, *, top_name: str | None = None) -> gdstk.Cell:
    if top_name is not None:
        for cell in lib.cells:
            if cell.name == top_name:
                return cell
        raise ValueError(f"top cell {top_name!r} not found")
    tops = lib.top_level()
    if len(tops) != 1:
        names = ", ".join(cell.name for cell in tops) or "<none>"
        raise ValueError(f"expected exactly one top-level cell, found {len(tops)}: {names}")
    return tops[0]


def resolve_selected_layers(
    old_inventory: GdsInventory,
    new_inventory: GdsInventory,
    selector: str | LayerSpec | Iterable[str | tuple[int, int] | LayerSpec] = "all",
    *,
    allow_empty_selected_layer: bool = False,
) -> tuple[SelectedLayerPresence, ...]:
    old_layers = set(old_inventory.layers)
    new_layers = set(new_inventory.layers)
    if selector == "all":
        selected = tuple(sorted(old_layers | new_layers))
    else:
        parsed = parse_layer_selector(selector)
        if parsed == "all":
            selected = tuple(sorted(old_layers | new_layers))
        else:
            selected = parsed
    result = tuple(
        SelectedLayerPresence(layer=layer, present_old=layer in old_layers, present_new=layer in new_layers)
        for layer in selected
    )
    absent_both = [presence.layer for presence in result if not presence.present_old and not presence.present_new]
    if absent_both and not allow_empty_selected_layer:
        missing = ", ".join(f"{layer.layer}:{layer.datatype}" for layer in absent_both)
        raise ValueError(f"selected layer(s) absent in both inputs: {missing}")
    return result


def _reachable_cells(top: gdstk.Cell) -> tuple[gdstk.Cell, ...]:
    seen = {top.name}
    pending = [top]
    out: list[gdstk.Cell] = []
    while pending:
        cell = pending.pop()
        out.append(cell)
        for ref in cell.references:
            child = ref.cell
            if isinstance(child, gdstk.Cell) and child.name not in seen:
                seen.add(child.name)
                pending.append(child)
    return tuple(out)


def _inventory_layers(cells: tuple[gdstk.Cell, ...]) -> tuple[LayerSpec, ...]:
    pairs: set[tuple[int, int]] = set()
    for cell in cells:
        # Placements cannot change layer tags. Polygonize paths only in their defining cell.
        polygons = cell.get_polygons(apply_repetitions=False, depth=0) if cell.paths else cell.polygons
        pairs.update((polygon.layer, polygon.datatype) for polygon in polygons)
    return tuple(LayerSpec(*pair) for pair in sorted(pairs))


def _record_inventory(cells: tuple[gdstk.Cell, ...]) -> RecordInventory:
    polygons = sum(len(cell.polygons) for cell in cells)
    paths = sum(len(cell.paths) for cell in cells)
    labels = sum(len(cell.labels) for cell in cells)
    references = sum(len(cell.references) for cell in cells)
    return RecordInventory(polygons=polygons, paths=paths, labels=labels, references=references, cells=len(cells))
