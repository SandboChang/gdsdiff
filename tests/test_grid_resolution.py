from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import gdstk
import pytest

from scgds.gdsdiff import GdsMetadata, read_gds_metadata, resolve_comparison_grid, snap_to_grid
from scgds.gdsdiff.grid import validate_int64_coordinate


def _metadata(precision: str, unit: str = "1e-6") -> GdsMetadata:
    return GdsMetadata(
        path=Path("fixture.gds"),
        unit_m=Decimal(unit),
        precision_m=Decimal(precision),
        cell_count=1,
        top_names=("top",),
    )


def test_read_gds_metadata_from_fixture(tmp_path):
    path = tmp_path / "metadata.gds"
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell("top")
    cell.add(gdstk.rectangle((0, 0), (1, 1), layer=1, datatype=2))
    lib.write_gds(path)

    metadata = read_gds_metadata(path)

    assert metadata.unit_m == Decimal("0.000001")
    assert metadata.precision_m == Decimal("1E-9")
    assert metadata.cell_count == 1
    assert metadata.top_names == ("top",)


def test_resolve_equal_precision_grid():
    result = resolve_comparison_grid(_metadata("1e-9"), _metadata("1e-9"))

    assert result.grid.meters_per_dbu == Decimal("1E-9")
    assert result.method == "equal-source-precision"
    assert result.grid.selection_mode == "auto"


def test_resolve_integer_related_precision_grid_uses_finer_precision():
    result = resolve_comparison_grid(_metadata("5e-9"), _metadata("1e-9"))

    assert result.grid.meters_per_dbu == Decimal("1E-9")
    assert result.method == "integer-related-source-precision"


def test_resolve_rational_common_subgrid():
    result = resolve_comparison_grid(_metadata("1.5e-9"), _metadata("1e-9"))

    assert result.grid.meters_per_dbu == Decimal("5E-10")
    assert result.method == "rational-common-subgrid"


def test_resolve_explicit_grid_and_conflict():
    result = resolve_comparison_grid(_metadata("5e-9"), _metadata("1e-9"), dbu="2e-9")

    assert result.grid.meters_per_dbu == Decimal("2E-9")
    assert result.method == "explicit"
    assert result.grid.selection_mode == "explicit"

    with pytest.raises(ValueError, match="dbu and precision"):
        resolve_comparison_grid(_metadata("5e-9"), _metadata("1e-9"), dbu="2e-9", precision="1e-9")


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        ("0.49e-9", 0),
        ("0.5e-9", 1),
        ("1.5e-9", 2),
        ("-0.49e-9", 0),
        ("-0.5e-9", -1),
        ("-1.5e-9", -2),
    ),
)
def test_snap_to_grid_nearest_ties_away_from_zero(value, expected):
    assert snap_to_grid(value, "1e-9") == expected


def test_validate_coordinate_safety_limit():
    validate_int64_coordinate(2**20)

    with pytest.raises(OverflowError, match="safety"):
        validate_int64_coordinate(2**62 + 1)
