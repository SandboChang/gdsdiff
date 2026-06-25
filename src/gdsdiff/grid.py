from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from math import gcd, lcm

from .config import GridSpec
from .gds_metadata import GdsMetadata, decimal_from_number

INT64_LIMIT = 2**63 - 1
DEFAULT_COORDINATE_SAFETY_LIMIT = 2**62
MAX_AUTO_GRID_DENOMINATOR = 10**18


@dataclass(frozen=True)
class GridResolution:
    grid: GridSpec
    ratio_residual: Decimal
    method: str

    def to_json(self) -> dict[str, object]:
        data = self.grid.to_json()
        data["ratio_residual"] = str(self.ratio_residual)
        data["method"] = self.method
        return data


def _fraction_from_decimal(value: Decimal) -> Fraction:
    return Fraction(value)


def _decimal_from_fraction(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


def _rational_gcd(left: Fraction, right: Fraction) -> Fraction:
    numerator = gcd(abs(left.numerator), abs(right.numerator))
    denominator = lcm(left.denominator, right.denominator)
    return Fraction(numerator, denominator)


def _integer_related(left: Decimal, right: Decimal) -> bool:
    larger = max(left, right)
    smaller = min(left, right)
    ratio = larger / smaller
    return ratio == ratio.to_integral_value()


def resolve_comparison_grid(
    old: GdsMetadata,
    new: GdsMetadata,
    *,
    dbu: Decimal | str | float | None = None,
    precision: Decimal | str | float | None = None,
    max_denominator: int = MAX_AUTO_GRID_DENOMINATOR,
) -> GridResolution:
    explicit_dbu = decimal_from_number(dbu) if dbu is not None else None
    explicit_precision = decimal_from_number(precision) if precision is not None else None
    if explicit_dbu is not None and explicit_precision is not None and explicit_dbu != explicit_precision:
        raise ValueError("explicit dbu and precision must match")
    explicit = explicit_dbu if explicit_dbu is not None else explicit_precision
    if explicit is not None:
        _validate_positive_decimal(explicit, "comparison grid")
        return GridResolution(
            grid=GridSpec(
                meters_per_dbu=explicit,
                source_old_precision_m=old.precision_m,
                source_new_precision_m=new.precision_m,
                selection_mode="explicit",
            ),
            ratio_residual=Decimal(0),
            method="explicit",
        )

    old_precision = old.precision_m
    new_precision = new.precision_m
    _validate_positive_decimal(old_precision, "old precision")
    _validate_positive_decimal(new_precision, "new precision")

    if old_precision == new_precision:
        selected = old_precision
        method = "equal-source-precision"
    elif _integer_related(old_precision, new_precision):
        selected = min(old_precision, new_precision)
        method = "integer-related-source-precision"
    else:
        old_fraction = _fraction_from_decimal(old_precision)
        new_fraction = _fraction_from_decimal(new_precision)
        selected_fraction = _rational_gcd(old_fraction, new_fraction)
        if selected_fraction.denominator > max_denominator:
            raise ValueError("automatic grid denominator exceeds limit; pass explicit dbu")
        selected = _decimal_from_fraction(selected_fraction)
        method = "rational-common-subgrid"

    return GridResolution(
        grid=GridSpec(
            meters_per_dbu=selected,
            source_old_precision_m=old_precision,
            source_new_precision_m=new_precision,
            selection_mode="auto",
        ),
        ratio_residual=Decimal(0),
        method=method,
    )


def snap_to_grid(value_m: Decimal | str | float | int, meters_per_dbu: Decimal | str | float) -> int:
    value = decimal_from_number(value_m)
    q = decimal_from_number(meters_per_dbu)
    _validate_positive_decimal(q, "meters_per_dbu")
    scaled = value / q
    sign = -1 if scaled < 0 else 1
    magnitude = abs(scaled)
    whole = int(magnitude)
    fraction = magnitude - whole
    if fraction >= Decimal("0.5"):
        whole += 1
    snapped = sign * whole
    validate_int64_coordinate(snapped)
    return snapped


def max_snap_error_dbu(values_m: list[Decimal], meters_per_dbu: Decimal) -> Decimal:
    if not values_m:
        return Decimal(0)
    errors = []
    for value in values_m:
        snapped = Decimal(snap_to_grid(value, meters_per_dbu))
        errors.append(abs(value / meters_per_dbu - snapped))
    return max(errors)


def validate_int64_coordinate(value: int, *, safety_limit: int = DEFAULT_COORDINATE_SAFETY_LIMIT) -> None:
    if abs(value) > INT64_LIMIT:
        raise OverflowError("coordinate exceeds signed int64 range")
    if abs(value) > safety_limit:
        raise OverflowError("coordinate exceeds configured safety limit")


def _validate_positive_decimal(value: Decimal, name: str) -> None:
    if not value.is_finite() or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
