from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gdstk


@dataclass(frozen=True)
class FixtureCase:
    name: str
    expected_equivalent: bool
    expected_layer_areas: dict[tuple[int, int], tuple[str, str, str]]


def write_fixture_pair(case: FixtureCase, root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    old_path = root / f"{case.name}_old.gds"
    new_path = root / f"{case.name}_new.gds"
    builder = _BUILDERS[case.name]
    old_lib, new_lib = builder()
    old_lib.write_gds(old_path)
    new_lib.write_gds(new_path)
    return old_path, new_path


def fixture_cases() -> tuple[FixtureCase, ...]:
    return (
        FixtureCase("identical", True, {}),
        FixtureCase("shifted", False, {(1, 0): ("20", "20", "40")}),
        FixtureCase("addition", False, {(1, 0): ("0", "100", "100")}),
        FixtureCase("removal", False, {(1, 0): ("100", "0", "100")}),
        FixtureCase("fractured_equivalent", True, {}),
        FixtureCase("duplicate_equivalent", True, {}),
        FixtureCase("layer_change", False, {(1, 0): ("100", "0", "100"), (2, 0): ("0", "100", "100")}),
        FixtureCase("reference_repetition_shift", False, {(1, 0): ("20", "20", "40")}),
    )


def _library() -> tuple[gdstk.Library, gdstk.Cell]:
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    return lib, lib.new_cell("top")


def _rect(x0, y0, x1, y1, *, layer=1, datatype=0):
    return gdstk.rectangle((x0, y0), (x1, y1), layer=layer, datatype=datatype)


def _identical():
    old, old_top = _library()
    new, new_top = _library()
    old_top.add(_rect(0, 0, 10, 10))
    new_top.add(_rect(0, 0, 10, 10))
    return old, new


def _shifted():
    old, old_top = _library()
    new, new_top = _library()
    old_top.add(_rect(0, 0, 10, 10))
    new_top.add(_rect(2, 0, 12, 10))
    return old, new


def _addition():
    old, old_top = _library()
    new, new_top = _library()
    old_top.add(_rect(0, 0, 10, 10))
    new_top.add(_rect(0, 0, 10, 10))
    new_top.add(_rect(20, 0, 30, 10))
    return old, new


def _removal():
    old, old_top = _library()
    new, new_top = _library()
    old_top.add(_rect(0, 0, 10, 10))
    old_top.add(_rect(20, 0, 30, 10))
    new_top.add(_rect(0, 0, 10, 10))
    return old, new


def _fractured_equivalent():
    old, old_top = _library()
    new, new_top = _library()
    old_top.add(_rect(0, 0, 10, 10))
    new_top.add(_rect(0, 0, 5, 10))
    new_top.add(_rect(5, 0, 10, 10))
    return old, new


def _duplicate_equivalent():
    old, old_top = _library()
    new, new_top = _library()
    old_top.add(_rect(0, 0, 10, 10))
    old_top.add(_rect(0, 0, 10, 10))
    new_top.add(_rect(0, 0, 10, 10))
    return old, new


def _layer_change():
    old, old_top = _library()
    new, new_top = _library()
    old_top.add(_rect(0, 0, 10, 10, layer=1))
    new_top.add(_rect(0, 0, 10, 10, layer=2))
    return old, new


def _reference_repetition_shift():
    old = gdstk.Library(unit=1e-6, precision=1e-9)
    old_child = old.new_cell("child")
    old_child.add(_rect(0, 0, 10, 10))
    old_top = old.new_cell("top")
    old_top.add(gdstk.Reference(old_child, (0, 0)))
    old_top.add(gdstk.Reference(old_child, (20, 0)))

    new = gdstk.Library(unit=1e-6, precision=1e-9)
    new_child = new.new_cell("child")
    new_child.add(_rect(0, 0, 10, 10))
    new_shifted = new.new_cell("child_shifted")
    new_shifted.add(_rect(2, 0, 12, 10))
    new_top = new.new_cell("top")
    new_top.add(gdstk.Reference(new_child, (0, 0)))
    new_top.add(gdstk.Reference(new_shifted, (20, 0)))
    return old, new


_BUILDERS = {
    "identical": _identical,
    "shifted": _shifted,
    "addition": _addition,
    "removal": _removal,
    "fractured_equivalent": _fractured_equivalent,
    "duplicate_equivalent": _duplicate_equivalent,
    "layer_change": _layer_change,
    "reference_repetition_shift": _reference_repetition_shift,
}
