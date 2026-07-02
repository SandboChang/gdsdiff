from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import gdstk

from scgds.gdsdiff import extract_gds_polygons, run_full_exact_oracle
from scgds.gdsdiff.templates import extract_template, find_template


def _rect(x0, y0, x1, y1, *, layer=1, datatype=0):
    return gdstk.rectangle((x0, y0), (x1, y1), layer=layer, datatype=datatype)


def _write_lib(path: Path, cells) -> None:
    lib = gdstk.Library(unit=1e-6, precision=1e-6)
    for cell in cells:
        lib.add(cell)
    lib.write_gds(path)


def _new_cell(name: str) -> gdstk.Cell:
    return gdstk.Cell(name)


def _assert_gds_equivalent(left: Path, right: Path, *, top_name: str = "TEMPLATE", layers: str = "all") -> None:
    left_buffer = extract_gds_polygons(left, top_name=top_name, layers=layers).buffer
    right_buffer = extract_gds_polygons(right, top_name=top_name, layers=layers).buffer
    assert run_full_exact_oracle(left_buffer, right_buffer).equivalent


def _write_template(path: Path) -> None:
    top = _new_cell("TEMPLATE")
    top.add(_rect(0, 0, 10, 10))
    top.add(_rect(14, 1, 16, 3))
    _write_lib(path, (top,))


def _add_template_copy(cell: gdstk.Cell, x: float, y: float, *, marker: bool = True, layer: int = 1) -> None:
    cell.add(_rect(x, y, x + 10, y + 10, layer=layer))
    if marker:
        cell.add(_rect(x + 14, y + 1, x + 16, y + 3, layer=layer))


def _transform_points(points, matrix):
    a, b, c, d = matrix
    transformed = [(a * x + b * y, c * x + d * y) for x, y in points]
    x0 = min(x for x, _y in transformed)
    y0 = min(y for _x, y in transformed)
    return [(x - x0, y - y0) for x, y in transformed]


def test_extract_template_normalizes_intersecting_references_arrays_and_filters_layers(tmp_path):
    source_path = tmp_path / "source.gds"
    template_path = tmp_path / "template.gds"
    metadata_path = tmp_path / "template.json"
    expected_path = tmp_path / "expected.gds"

    lib = gdstk.Library(unit=1e-6, precision=1e-6)
    child = lib.new_cell("DEVICE")
    child.add(_rect(0, 0, 10, 10, layer=1))
    child.add(_rect(0, 0, 8, 8, layer=2))
    top = lib.new_cell("TOP")
    top.add(_rect(300, 300, 310, 310, layer=1))
    top.add(gdstk.Reference(child, (100, 50)))
    top.add(gdstk.Reference(child, (120, 60), columns=2, rows=1, spacing=(15, 0)))
    lib.write_gds(source_path)

    result = extract_template(
        source_path,
        top_name="TOP",
        layers="1:0",
        bbox_user=(95, 45, 136, 75),
        output_gds=template_path,
        metadata_path=metadata_path,
    )

    expected = gdstk.Library(unit=1e-6, precision=1e-6)
    expected_top = expected.new_cell("TEMPLATE")
    expected_top.add(_rect(5, 5, 15, 15, layer=1))
    expected_top.add(_rect(25, 15, 35, 25, layer=1))
    expected_top.add(_rect(40, 15, 50, 25, layer=1))
    expected.write_gds(expected_path)

    _assert_gds_equivalent(template_path, expected_path, layers="1:0")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert result.metadata == metadata
    assert metadata["source"]["top"] == "TOP"
    assert metadata["source"]["bbox_user"] == [95, 45, 136, 75]
    assert metadata["source"]["selection_mode"] == "intersects"
    assert metadata["source"]["layers"] == [{"datatype": 0, "layer": 1}]
    assert metadata["template"]["top"] == "TEMPLATE"
    assert metadata["template"]["polygon_count"] == 3
    assert metadata["template"]["bbox_dbu"] == [5, 5, 50, 25]
    assert metadata["template"]["layer_summary"][0]["polygon_count"] == 3


def test_find_template_reports_translated_matches_and_failed_candidates(tmp_path):
    template_path = tmp_path / "template.gds"
    target_path = tmp_path / "target.gds"
    output_dir = tmp_path / "find"
    _write_template(template_path)

    target = gdstk.Library(unit=1e-6, precision=1e-6)
    top = target.new_cell("TARGET")
    _add_template_copy(top, 100, 50)
    _add_template_copy(top, 200, 70)
    _add_template_copy(top, 300, 90, marker=False)
    _add_template_copy(top, 400, 90, layer=2)
    target.write_gds(target_path)

    result = find_template(
        template_path,
        target_path,
        template_top="TEMPLATE",
        target_top="TARGET",
        layers="1:0",
        output_dir=output_dir,
        transforms=("translate",),
    )

    payload = json.loads((output_dir / "matches.json").read_text(encoding="utf-8"))
    assert result.result == payload
    markdown = (output_dir / "matches.md").read_text(encoding="utf-8")
    assert "## Per-Layer Candidate Details" in markdown
    assert "template_area_dbu2" in markdown
    assert payload["summary"] == {"candidate_count": 3, "difference_count": 1, "match_count": 2}
    assert Counter(candidate["status"] for candidate in payload["candidates"]) == Counter({"identical": 2, "differences-found": 1})
    assert sorted(candidate["bbox_dbu"] for candidate in payload["candidates"] if candidate["status"] == "identical") == [
        [100, 50, 116, 60],
        [200, 70, 216, 80],
    ]
    failed = next(candidate for candidate in payload["candidates"] if candidate["status"] == "differences-found")
    assert failed["target_polygon_count"] == 1
    assert failed["layers"][0]["target_only_area_dbu2"] == "0"
    assert failed["layers"][0]["template_only_area_dbu2"] == "4"


def test_find_template_requires_enabled_rot90_and_mirror_transforms(tmp_path):
    template_path = tmp_path / "template_l.gds"
    target_path = tmp_path / "target_l.gds"
    l_points = [(0, 0), (12, 0), (12, 3), (5, 3), (5, 5), (9, 5), (9, 8), (5, 8), (5, 12), (0, 12)]

    template = gdstk.Library(unit=1e-6, precision=1e-6)
    template_top = template.new_cell("TEMPLATE")
    template_top.add(gdstk.Polygon(l_points, layer=1, datatype=0))
    template.write_gds(template_path)

    target = gdstk.Library(unit=1e-6, precision=1e-6)
    target_top = target.new_cell("TARGET")
    rot90_points = [(x + 100, y + 20) for x, y in _transform_points(l_points, (0, -1, 1, 0))]
    mirror_points = [(x + 200, y + 20) for x, y in _transform_points(l_points, (1, 0, 0, -1))]
    target_top.add(gdstk.Polygon(rot90_points, layer=1, datatype=0))
    target_top.add(gdstk.Polygon(mirror_points, layer=1, datatype=0))
    target.write_gds(target_path)

    translate_only = find_template(
        template_path,
        target_path,
        template_top="TEMPLATE",
        target_top="TARGET",
        layers="1:0",
        output_dir=tmp_path / "translate_only",
        transforms=("translate",),
    )
    assert translate_only.result["summary"] == {"candidate_count": 0, "difference_count": 0, "match_count": 0}

    rotated = find_template(
        template_path,
        target_path,
        template_top="TEMPLATE",
        target_top="TARGET",
        layers="1:0",
        output_dir=tmp_path / "rotated",
        transforms=("translate", "rot90"),
    )
    assert rotated.result["summary"]["match_count"] == 1
    assert rotated.result["candidates"][0]["transform"] == "rot90"
    assert rotated.result["candidates"][0]["bbox_dbu"] == [100, 20, 112, 32]

    mirrored = find_template(
        template_path,
        target_path,
        template_top="TEMPLATE",
        target_top="TARGET",
        layers="1:0",
        output_dir=tmp_path / "mirrored",
        transforms=("translate", "rot90"),
        allow_mirror=True,
    )
    assert mirrored.result["summary"]["match_count"] == 2
    assert sorted(candidate["transform"] for candidate in mirrored.result["candidates"]) == ["rot90", "translate+mirror-x"]
