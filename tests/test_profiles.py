from __future__ import annotations

import csv

import gdstk

from scgds.gdsdiff import AssemblyProfile, CompareConfig, combined_profile_summary, compare_gds_files
from scgds.gdsdiff.profiles import calibrator_model1_profile, check_assembly_profile
from scgds.tests.golden_gds import ROOT


def _write_wafer(path, *, reference_count=2, include_layer=True):
    lib = gdstk.Library(unit=1e-6, precision=1e-6)
    chip = lib.new_cell("chip")
    chip.add(gdstk.rectangle((0, 0), (10, 10), layer=1, datatype=0))
    wafer = lib.new_cell("wafer")
    if include_layer:
        wafer.add(gdstk.rectangle((-100, -100), (100, 100), layer=61, datatype=0))
    for index in range(reference_count):
        wafer.add(gdstk.Reference(chip, (index * 20, 0)))
    lib.write_gds(path)


def _write_manifest(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["chip_ix", "chip_iy", "chip_label", "wafer_x_um", "wafer_y_um", "area_scale", "linear_scale"])
        writer.writeheader()
        writer.writerows(rows)


def test_assembly_profile_passes_for_expected_wafer_and_manifest(tmp_path):
    wafer_path = tmp_path / "wafer.gds"
    manifest_path = tmp_path / "manifest.csv"
    _write_wafer(wafer_path, reference_count=2)
    _write_manifest(
        manifest_path,
        [
            {"chip_ix": 0, "chip_iy": 0, "chip_label": "a", "wafer_x_um": 0, "wafer_y_um": 0, "area_scale": 1, "linear_scale": 1},
            {"chip_ix": 1, "chip_iy": 0, "chip_label": "b", "wafer_x_um": 20, "wafer_y_um": 0, "area_scale": 1, "linear_scale": 1},
        ],
    )
    profile = AssemblyProfile(
        name="synthetic wafer",
        gds_path=wafer_path,
        expected_top="wafer",
        expected_reference_count=2,
        expected_layers=((61, 0),),
        expected_files=(wafer_path, manifest_path),
        manifest_path=manifest_path,
        manifest_required_columns=("chip_ix", "chip_iy", "chip_label", "wafer_x_um", "wafer_y_um", "area_scale", "linear_scale"),
        manifest_expected_rows=2,
    )

    result = check_assembly_profile(profile)

    assert result.passed
    assert {check.name for check in result.checks} >= {"top_reference_count", "manifest_row_count", "expected_layers"}


def test_profile_distinguishes_manifest_failure_from_geometry(tmp_path):
    wafer_path = tmp_path / "wafer.gds"
    manifest_path = tmp_path / "manifest.csv"
    _write_wafer(wafer_path, reference_count=2)
    _write_manifest(manifest_path, [])
    profile = AssemblyProfile(
        name="manifest mismatch",
        gds_path=wafer_path,
        expected_top="wafer",
        expected_reference_count=2,
        manifest_path=manifest_path,
        manifest_required_columns=("chip_ix",),
        manifest_expected_rows=2,
    )

    result = check_assembly_profile(profile)

    assert not result.passed
    assert next(check for check in result.checks if check.name == "top_reference_count").passed
    assert not next(check for check in result.checks if check.name == "manifest_row_count").passed


def test_profile_distinguishes_placement_failure_from_geometry(tmp_path):
    wafer_path = tmp_path / "wafer.gds"
    _write_wafer(wafer_path, reference_count=1)
    profile = AssemblyProfile(name="placement mismatch", gds_path=wafer_path, expected_top="wafer", expected_reference_count=2)

    result = check_assembly_profile(profile)

    assert not result.passed
    ref_check = next(check for check in result.checks if check.name == "top_reference_count")
    assert ref_check.observed == 1
    assert ref_check.expected == 2


def test_combined_summary_keeps_geometry_and_assembly_status_separate(tmp_path):
    old_path = tmp_path / "old.gds"
    new_path = tmp_path / "new.gds"
    _write_wafer(old_path, reference_count=1)
    _write_wafer(new_path, reference_count=1)
    geometry = compare_gds_files(CompareConfig(old_path, new_path, tile_size=10))
    profile = check_assembly_profile(AssemblyProfile(name="bad assembly", gds_path=new_path, expected_top="wafer", expected_reference_count=2))

    summary = combined_profile_summary(geometry=geometry, assembly=profile)
    markdown = summary.render_markdown()

    assert geometry.exit_code == 0
    assert not profile.passed
    assert not summary.passed
    assert "Geometry status: equivalent" in markdown
    assert "FAIL top_reference_count" in markdown


def test_calibrator_profile_derives_counts_and_paths_from_project_modules():
    profile = calibrator_model1_profile()
    config = __import__("projects.Calibrator.designs.model1.config", fromlist=["dummy"])
    wafer = __import__("projects.Calibrator.designs.model1.wafer", fromlist=["dummy"])

    assert profile.gds_path == config.output_path("wafer")
    assert profile.manifest_path == config.output_path("manifest")
    assert profile.expected_reference_count == len(wafer.auto_chip_placements())
    assert profile.expected_reference_count != 0
    assert str(ROOT) in str(profile.gds_path)
