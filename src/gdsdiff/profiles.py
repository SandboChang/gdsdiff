from __future__ import annotations

import csv
import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import gdstk

from .api import CompareRunResult


@dataclass(frozen=True)
class AssemblyProfile:
    name: str
    gds_path: Path
    expected_top: str | None = None
    expected_reference_count: int | None = None
    expected_layers: tuple[tuple[int, int], ...] = ()
    expected_files: tuple[Path, ...] = ()
    manifest_path: Path | None = None
    manifest_required_columns: tuple[str, ...] = ()
    manifest_expected_rows: int | None = None


@dataclass(frozen=True)
class ProfileCheck:
    name: str
    passed: bool
    observed: object
    expected: object = None
    message: str = ""

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "passed": self.passed,
            "observed": self.observed,
            "expected": self.expected,
            "message": self.message,
        }


@dataclass(frozen=True)
class ProfileResult:
    profile_name: str
    checks: tuple[ProfileCheck, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_json(self) -> dict[str, object]:
        return {
            "profile_name": self.profile_name,
            "passed": self.passed,
            "checks": [check.to_json() for check in self.checks],
        }


@dataclass(frozen=True)
class CombinedProfileSummary:
    geometry: CompareRunResult | None
    assembly: ProfileResult | None

    @property
    def passed(self) -> bool:
        geometry_ok = True if self.geometry is None else self.geometry.exit_code == 0
        assembly_ok = True if self.assembly is None else self.assembly.passed
        return geometry_ok and assembly_ok

    def render_markdown(self) -> str:
        lines = ["### GDS Comparator Profile Summary", ""]
        if self.geometry is not None:
            lines.extend(
                [
                    f"- Geometry status: {self.geometry.status.value}.",
                    f"- Geometry exit code: {self.geometry.exit_code}.",
                ]
            )
        if self.assembly is not None:
            lines.append(f"- Assembly profile: {self.assembly.profile_name}.")
            lines.append(f"- Assembly checks passed: {self.assembly.passed}.")
            for check in self.assembly.checks:
                status = "PASS" if check.passed else "FAIL"
                lines.append(f"  - {status} {check.name}: observed={check.observed!r} expected={check.expected!r}")
        return "\n".join(lines) + "\n"


def check_assembly_profile(profile: AssemblyProfile) -> ProfileResult:
    checks: list[ProfileCheck] = []
    for path in profile.expected_files:
        checks.append(ProfileCheck(name=f"file:{path.name}", passed=path.exists(), observed=str(path), expected="exists"))
    if profile.gds_path.exists():
        lib = gdstk.read_gds(str(profile.gds_path))
        tops = lib.top_level()
        top = _resolve_top(tops, profile.expected_top)
        checks.append(ProfileCheck(name="top", passed=top is not None, observed=None if top is None else top.name, expected=profile.expected_top))
        if top is not None:
            if profile.expected_reference_count is not None:
                observed = len(top.references)
                checks.append(
                    ProfileCheck(
                        name="top_reference_count",
                        passed=observed == profile.expected_reference_count,
                        observed=observed,
                        expected=profile.expected_reference_count,
                    )
                )
            if profile.expected_layers:
                observed_layers = {(polygon.layer, polygon.datatype) for polygon in top.get_polygons(apply_repetitions=True)}
                expected_layers = set(profile.expected_layers)
                checks.append(
                    ProfileCheck(
                        name="expected_layers",
                        passed=expected_layers <= observed_layers,
                        observed=sorted(observed_layers),
                        expected=sorted(expected_layers),
                    )
                )
    else:
        checks.append(ProfileCheck(name="gds_path", passed=False, observed=str(profile.gds_path), expected="exists"))
    if profile.manifest_path is not None:
        checks.extend(_check_manifest(profile.manifest_path, profile.manifest_required_columns, profile.manifest_expected_rows))
    return ProfileResult(profile.name, tuple(checks))


def combined_profile_summary(geometry: CompareRunResult | None = None, assembly: ProfileResult | None = None) -> CombinedProfileSummary:
    return CombinedProfileSummary(geometry=geometry, assembly=assembly)


def calibrator_model1_profile(project_dir: str | Path | None = None) -> AssemblyProfile:
    config = importlib.import_module("projects.Calibrator.designs.model1.config")
    wafer = importlib.import_module("projects.Calibrator.designs.model1.wafer")
    project_path = Path(project_dir) if project_dir is not None else Path(config.project_dir)
    placements = wafer.auto_chip_placements()
    output_dir = Path(config.resolved_output_dir)
    if project_dir is not None:
        output_dir = project_path / "outputs"
    wafer_path = config.output_path("wafer", output_dir)
    manifest_path = config.output_path("manifest", output_dir)
    expected_files = (wafer_path, manifest_path, output_dir / "chips")
    expected_layers = (
        (config.ld_laserGuide_general["layer"], config.ld_laserGuide_general["datatype"]),
        (config.ld_wafer["layer"], config.ld_wafer["datatype"]),
    )
    return AssemblyProfile(
        name="Calibrator model1 wafer",
        gds_path=wafer_path,
        expected_top=config.wafer_cell_name,
        expected_reference_count=len(placements),
        expected_layers=expected_layers,
        expected_files=expected_files,
        manifest_path=manifest_path,
        manifest_required_columns=("chip_ix", "chip_iy", "chip_label", "wafer_x_um", "wafer_y_um", "area_scale", "linear_scale"),
        manifest_expected_rows=len(placements),
    )


def _resolve_top(tops: list[gdstk.Cell], expected_top: str | None) -> gdstk.Cell | None:
    if expected_top is None:
        return tops[0] if len(tops) == 1 else None
    for top in tops:
        if top.name == expected_top:
            return top
    return None


def _check_manifest(path: Path, required_columns: Iterable[str], expected_rows: int | None) -> tuple[ProfileCheck, ...]:
    if not path.exists():
        return (ProfileCheck(name="manifest_path", passed=False, observed=str(path), expected="exists"),)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    checks = []
    columns = set(rows[0]) if rows else set()
    required = set(required_columns)
    checks.append(ProfileCheck(name="manifest_columns", passed=required <= columns, observed=sorted(columns), expected=sorted(required)))
    if expected_rows is not None:
        checks.append(ProfileCheck(name="manifest_row_count", passed=len(rows) == expected_rows, observed=len(rows), expected=expected_rows))
    return tuple(checks)
