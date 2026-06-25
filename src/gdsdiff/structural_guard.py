from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path


class StructuralGuardUnavailable(ValueError):
    pass


@dataclass(frozen=True)
class CellStructure:
    name: str
    local_shape_count: int = 0
    references: tuple[tuple[str, int], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class StructuralSummary:
    path: Path
    selected_top: str
    cell_count: int
    flattened_shape_count: int
    reachable_cell_count: int

    def to_json(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "selected_top": self.selected_top,
            "cell_count": self.cell_count,
            "flattened_shape_count": self.flattened_shape_count,
            "reachable_cell_count": self.reachable_cell_count,
        }


def summarize_gds_structure(path: str | Path, *, top_name: str | None = None) -> StructuralSummary:
    path = Path(path)
    cells = _parse_cells(path)
    if not cells:
        raise StructuralGuardUnavailable("no cells found")
    top = _select_top(cells, top_name)
    visiting: set[str] = set()
    seen: set[str] = set()
    count = _flattened_shape_count(top, cells, visiting, seen)
    return StructuralSummary(
        path=path,
        selected_top=top,
        cell_count=len(cells),
        flattened_shape_count=count,
        reachable_cell_count=len(seen),
    )


def _parse_cells(path: Path) -> dict[str, CellStructure]:
    cells: dict[str, CellStructure] = {}
    current_name: str | None = None
    local_shape_count = 0
    references: list[tuple[str, int]] = []
    element_kind: int | None = None
    pending_sname: str | None = None
    pending_multiplier = 1

    data = path.read_bytes()
    offset = 0
    size = len(data)
    while offset + 4 <= size:
        record_len, record_type, _data_type = struct.unpack_from(">HBB", data, offset)
        if record_len < 4 or offset + record_len > size:
            raise StructuralGuardUnavailable("invalid GDS record length")
        payload = data[offset + 4 : offset + record_len]
        offset += record_len

        if record_type == 0x05:  # BGNSTR
            current_name = None
            local_shape_count = 0
            references = []
        elif record_type == 0x06:  # STRNAME
            current_name = _decode_gds_string(payload)
        elif record_type in (0x08, 0x09):  # BOUNDARY, PATH
            element_kind = record_type
        elif record_type in (0x0A, 0x0B):  # SREF, AREF
            element_kind = record_type
            pending_sname = None
            pending_multiplier = 1
        elif record_type == 0x12 and element_kind in (0x0A, 0x0B):  # SNAME
            pending_sname = _decode_gds_string(payload)
        elif record_type == 0x13 and element_kind == 0x0B:  # COLROW
            if len(payload) < 4:
                raise StructuralGuardUnavailable("invalid AREF COLROW record")
            columns, rows = struct.unpack_from(">hh", payload, 0)
            if columns <= 0 or rows <= 0:
                raise StructuralGuardUnavailable("invalid AREF repetition")
            pending_multiplier = int(columns) * int(rows)
        elif record_type == 0x11:  # ENDEL
            if element_kind in (0x08, 0x09):
                local_shape_count += 1
            elif element_kind in (0x0A, 0x0B):
                if pending_sname is None:
                    raise StructuralGuardUnavailable("reference missing SNAME")
                references.append((pending_sname, pending_multiplier))
            element_kind = None
            pending_sname = None
            pending_multiplier = 1
        elif record_type == 0x07:  # ENDSTR
            if current_name is None:
                raise StructuralGuardUnavailable("cell missing STRNAME")
            cells[current_name] = CellStructure(
                name=current_name,
                local_shape_count=local_shape_count,
                references=tuple(references),
            )
            current_name = None
            local_shape_count = 0
            references = []

    if offset != size:
        raise StructuralGuardUnavailable("trailing partial GDS record")
    return cells


def _select_top(cells: dict[str, CellStructure], top_name: str | None) -> str:
    if top_name is not None:
        if top_name not in cells:
            raise StructuralGuardUnavailable(f"top cell {top_name!r} not found")
        return top_name
    referenced = {child for cell in cells.values() for child, _multiplier in cell.references}
    tops = sorted(set(cells) - referenced)
    if len(tops) != 1:
        raise StructuralGuardUnavailable(f"expected exactly one top cell, found {len(tops)}")
    return tops[0]


def _flattened_shape_count(
    name: str,
    cells: dict[str, CellStructure],
    visiting: set[str],
    seen: set[str],
    memo: dict[str, int] | None = None,
) -> int:
    memo = memo if memo is not None else {}
    if name in memo:
        seen.add(name)
        return memo[name]
    if name in visiting:
        raise StructuralGuardUnavailable("cell reference cycle detected")
    cell = cells.get(name)
    if cell is None:
        raise StructuralGuardUnavailable(f"unresolved reference to {name!r}")
    visiting.add(name)
    seen.add(name)
    total = cell.local_shape_count
    for child_name, multiplier in cell.references:
        total += multiplier * _flattened_shape_count(child_name, cells, visiting, seen, memo)
    visiting.remove(name)
    memo[name] = total
    return total


def _decode_gds_string(payload: bytes) -> str:
    return payload.rstrip(b"\0").decode("ascii")
