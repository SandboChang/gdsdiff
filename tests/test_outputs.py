from __future__ import annotations

import json
from datetime import date, datetime, timezone

import gdstk
import numpy as np
import pytest

from gds_tools_tk.gpu_compare import (
    AllowedRegion,
    CompareReport,
    CompareStatus,
    DifferenceDirection,
    DiffFragment,
    ExactOracleResult,
    LayerExactDiff,
    LayerSpec,
    build_polygon_inventory_delta,
    classify_exact_differences,
    render_design_history_entry,
    render_exact_diff_markdown,
    run_full_exact_oracle,
    validate_report_dict,
    write_diff_gds,
    write_exact_diff_markdown,
)
from gds_tools_tk.gpu_compare.geometry import LayerTable, PolygonBuffer, polygon_bbox, polygon_twice_area
from gds_tools_tk.gpu_compare.report import build_layer_reports


def _buffer(polygons, layer=LayerSpec(1, 0)):
    vertices = []
    offsets = [0]
    bboxes = []
    for polygon in polygons:
        vertices.extend(polygon)
        offsets.append(len(vertices))
        bboxes.append([min(x for x, _ in polygon), min(y for _, y in polygon), max(x for x, _ in polygon), max(y for _, y in polygon)])
    return PolygonBuffer(
        vertices_xy=np.asarray(vertices, dtype=np.int64).reshape((-1, 2)),
        polygon_offsets=np.asarray(offsets, dtype=np.uint64),
        layer_ids=np.zeros(len(polygons), dtype=np.uint32),
        bboxes=np.asarray(bboxes, dtype=np.int64).reshape((len(polygons), 4)),
        flags=np.zeros(len(polygons), dtype=np.uint32),
        source_ids=np.arange(len(polygons), dtype=np.uint64),
        layer_table=LayerTable((layer,)),
    )


def _rect(x0, y0, x1, y1):
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def _shift_exact(layer=LayerSpec(1, 0)):
    old = _buffer([_rect(0, 0, 10, 10)], layer)
    new = _buffer([_rect(2, 0, 12, 10)], layer)
    return run_full_exact_oracle(old, new)


def _classified_shift():
    layer = LayerSpec(1, 0)
    exact = _shift_exact(layer)
    policy = (
        AllowedRegion(layers=(layer,), direction=DifferenceDirection.REMOVAL, bbox=(0, 0, 2, 10)),
    )
    return exact, classify_exact_differences(exact, policy)


def test_build_layer_reports_validates_json_schema():
    exact, policy = _classified_shift()
    layer_reports = build_layer_reports(
        exact,
        policy_result=policy,
        presence_by_layer={LayerSpec(1, 0): (True, True)},
        polygon_counts_by_layer={LayerSpec(1, 0): (1, 1)},
        candidate_tile_counts_by_layer={LayerSpec(1, 0): 2},
        exact_window_counts_by_layer={LayerSpec(1, 0): 2},
    )
    report = CompareReport(
        status=policy.status,
        exact_verification_complete=True,
        old_input={"path": "old.gds"},
        new_input={"path": "new.gds"},
        semantics={"dbu": "1"},
        selection={"layers": [{"layer": 1, "datatype": 0}]},
        backend={"name": "cpu"},
        prefilter={"candidate_tile_count": 2},
        exact_fallback={"window_count": 2},
        layers=layer_reports,
        allowed_policy=policy.effective_policy,
        unexpected_summary={"unexpected_area_dbu2": "20"},
    )

    data = report.to_json_dict()

    validate_report_dict(data)
    assert data["layers"][0]["old_minus_new_area_dbu2"] == "20"
    assert data["layers"][0]["expected_area_dbu2"] == "20"
    assert data["layers"][0]["unexpected_area_dbu2"] == "20"
    assert {component["classification"] for component in data["layers"][0]["components"]} == {"expected", "unexpected"}


def test_diff_gds_empty_result_writes_top_cell(tmp_path):
    exact = ExactOracleResult(())
    policy = classify_exact_differences(exact, ())

    result = write_diff_gds(policy, tmp_path / "empty_diff.gds")
    lib = gdstk.read_gds(result.path)

    assert result.component_count == 0
    assert result.category_cells == ()
    assert {cell.name for cell in lib.cells} == {"GDS_GPU_COMPARE_DIFF"}


def test_diff_gds_preserves_layer_datatype_and_category_coverage(tmp_path):
    exact, policy = _classified_shift()

    result = write_diff_gds(policy, tmp_path / "diff.gds")
    lib = gdstk.read_gds(result.path)
    cell_names = {cell.name for cell in lib.cells}
    polygons = []
    for top in lib.top_level():
        polygons.extend(top.get_polygons(apply_repetitions=True))

    assert {"EXPECTED_REMOVAL", "UNEXPECTED_ADDITION"} <= cell_names
    assert result.component_count == 2
    assert {(polygon.layer, polygon.datatype) for polygon in polygons} == {(1, 0)}
    assert round(sum(polygon.area() for polygon in polygons)) == 40


def test_output_helpers_handle_multiple_layers():
    layer1 = LayerSpec(1, 0)
    layer2 = LayerSpec(2, 0)
    points1 = _rect(0, 0, 10, 10)
    points2 = _rect(20, 0, 30, 10)
    exact = ExactOracleResult(
        (
            LayerExactDiff(
                layer=layer1,
                new_minus_old=((_fragment(points1)),),
                xor=((_fragment(points1)),),
            ),
            LayerExactDiff(
                layer=layer2,
                old_minus_new=((_fragment(points2)),),
                xor=((_fragment(points2)),),
            ),
        )
    )
    policy = classify_exact_differences(
        exact,
        (
            AllowedRegion(layers=(layer1,), direction=DifferenceDirection.ADDITION, whole_layer=True),
            AllowedRegion(layers=(layer2,), direction=DifferenceDirection.REMOVAL, whole_layer=True),
        ),
    )

    reports = build_layer_reports(exact, policy_result=policy)

    assert [report.layer for report in reports] == [layer1, layer2]
    assert [report.expected_area_dbu2 for report in reports] == ["100", "100"]
    assert policy.status == CompareStatus.EXPECTED_DIFFERENCES_ONLY


def test_design_history_markdown_is_stable():
    _, policy = _classified_shift()

    markdown = render_design_history_entry(
        "GPU comparator check",
        policy,
        diff_gds_path="outputs/diff.gds",
        tests_run=("pytest gpu_compare",),
        notes=("classified crossing geometry exactly",),
        entry_date=date(2026, 6, 24),
    )

    assert markdown == (
        "### 2026-06-24 - GPU comparator check\n"
        "\n"
        "- Status: unexpected-differences.\n"
        "- Expected diff area dbu^2: 20.\n"
        "- Unexpected diff area dbu^2: 20.\n"
        "- Diff GDS: `outputs/diff.gds`.\n"
        "- Tests run:\n"
        "  - `pytest gpu_compare`\n"
        "- Notes:\n"
        "  - classified crossing geometry exactly\n"
        "- Layer summary:\n"
        "  - 1:0 expected=20 unexpected=20 components=2\n"
    )


def test_exact_diff_markdown_includes_polygon_vertices(tmp_path):
    exact, policy = _classified_shift()
    old = _buffer([_rect(0, 0, 10, 10)])
    new = _buffer([_rect(2, 0, 12, 10)])
    delta = build_polygon_inventory_delta(old, new)

    markdown = render_exact_diff_markdown(
        policy,
        title="Exact evidence",
        source_mode="full",
        exact_verification_complete=True,
        polygon_inventory_delta=delta,
        inventory_limit=None,
        component_detail_mode="full",
        generated_at=datetime(2026, 6, 25, tzinfo=timezone.utc),
    )

    assert "# Exact evidence" in markdown
    assert "- source_mode: `full`" in markdown
    assert "- exact_verification_complete: `true`" in markdown
    assert "- component_count: `2`" in markdown
    assert "## Polygon Components" in markdown
    assert '"classification": "expected"' in markdown
    assert '"classification": "unexpected"' in markdown
    assert '"vertices_dbu": [' in markdown
    assert '"bbox_dbu": [' in markdown
    assert "- points_sha256: `" in markdown
    assert "## Input Polygon Inventory Delta" in markdown
    assert "- removed_polygon_records: `1`" in markdown
    assert "- added_polygon_records: `1`" in markdown
    assert "- exact_coverage_xor_area_dbu2: `40`" in markdown

    target = tmp_path / "exact_diff.md"
    written = write_exact_diff_markdown(
        policy,
        target,
        title="Exact evidence",
        source_mode="full",
        exact_verification_complete=True,
        polygon_inventory_delta=delta,
        inventory_limit=None,
        component_detail_mode="full",
        generated_at=datetime(2026, 6, 25, tzinfo=timezone.utc),
    )

    assert written.path == target
    assert written.component_count == 2
    assert written.vertex_sidecar_path is not None
    assert written.vertex_sidecar_path.exists()
    written_markdown = target.read_text(encoding="utf-8")
    assert "# Exact evidence" in written_markdown
    assert "- vertex_sidecar_path: `" in written_markdown
    sidecar_records = [json.loads(line) for line in written.vertex_sidecar_path.read_text(encoding="utf-8").splitlines()]
    assert {record["kind"] for record in sidecar_records} == {"coverage_component", "source_polygon_inventory"}
    assert len(sidecar_records) == 4
    assert all(record["vertices_dbu"] for record in sidecar_records)


def test_exact_diff_markdown_reports_polygon_split_with_zero_coverage_change():
    layer = LayerSpec(1, 0)
    old = _buffer([_rect(0, 0, 10, 10)], layer)
    new = _buffer([_rect(0, 0, 5, 10), _rect(5, 0, 10, 10)], layer)
    exact = run_full_exact_oracle(old, new)
    policy = classify_exact_differences(exact)
    delta = build_polygon_inventory_delta(old, new)

    markdown = render_exact_diff_markdown(
        policy,
        title="Split evidence",
        source_mode="full",
        exact_verification_complete=True,
        polygon_inventory_delta=delta,
        inventory_limit=None,
        component_detail_mode="full",
        generated_at=datetime(2026, 6, 25, tzinfo=timezone.utc),
    )

    assert policy.status == CompareStatus.EQUIVALENT
    assert "- component_count: `0`" in markdown
    assert "No polygon-level differences were found." in markdown
    assert "## Input Polygon Inventory Delta" in markdown
    assert "- removed_polygon_records: `1`" in markdown
    assert "- added_polygon_records: `2`" in markdown
    assert "- exact_coverage_xor_area_dbu2: `0`" in markdown
    assert "- coverage_equal_but_inventory_changed: `true`" in markdown
    assert '"direction": "inventory_before_only"' in markdown
    assert '"direction": "inventory_after_only"' in markdown
    assert '"vertices_dbu": [' in markdown


def test_exact_diff_markdown_truncates_inventory_records_explicitly():
    layer = LayerSpec(1, 0)
    old = _buffer([_rect(index * 20, 0, index * 20 + 10, 10) for index in range(3)], layer)
    new = _buffer([_rect(index * 20 + 1, 0, index * 20 + 11, 10) for index in range(3)], layer)
    exact = run_full_exact_oracle(old, new)
    policy = classify_exact_differences(exact)
    delta = build_polygon_inventory_delta(old, new)

    markdown = render_exact_diff_markdown(
        policy,
        polygon_inventory_delta=delta,
        inventory_limit=2,
        generated_at=datetime(2026, 6, 25, tzinfo=timezone.utc),
    )

    assert "- changed_polygon_records: `6`" in markdown
    assert "### Inventory Truncation" in markdown
    assert "- omitted_polygon_record_count: `4`" in markdown


def test_exact_diff_markdown_warns_when_verification_is_incomplete():
    _, policy = _classified_shift()

    markdown = render_exact_diff_markdown(
        policy,
        source_mode="windowed",
        exact_verification_complete=False,
        component_limit=1,
        vertex_limit=2,
        component_detail_mode="full",
        generated_at=datetime(2026, 6, 25, tzinfo=timezone.utc),
    )

    assert "## Evidence Completeness Warning" in markdown
    assert "- omitted_component_count: `1`" in markdown
    assert "- omitted_vertex_count: `2`" in markdown


def test_json_write_failure_does_not_replace_existing_target(tmp_path, monkeypatch):
    exact, policy = _classified_shift()
    report = CompareReport(
        status=policy.status,
        exact_verification_complete=True,
        old_input={},
        new_input={},
        semantics={},
        selection={},
        backend={},
        prefilter={},
        exact_fallback={},
        layers=build_layer_reports(exact, policy_result=policy),
    )
    target = tmp_path / "report.json"
    target.write_text("old report\n", encoding="utf-8")

    def fail_replace(source, destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr("gds_tools_tk.gpu_compare.report.os.replace", fail_replace)

    with pytest.raises(OSError, match="simulated"):
        report.write_json(target)

    assert target.read_text(encoding="utf-8") == "old report\n"


def _fragment(points):
    return DiffFragment(
        points=points,
        bbox=polygon_bbox(points),
        twice_area=polygon_twice_area(points),
    )
