from __future__ import annotations

import json

import pytest

from gdsdiff import (
    CompareReport,
    CompareStatus,
    ComponentSummary,
    FailurePolicy,
    LayerReport,
    LayerSpec,
    validate_report_dict,
)


def _base_report(**kwargs) -> CompareReport:
    data = {
        "status": CompareStatus.EQUIVALENT,
        "exact_verification_complete": True,
        "old_input": {"path": "old.gds"},
        "new_input": {"path": "new.gds"},
        "semantics": {"grid": "1E-9"},
        "selection": {"layers": "all"},
        "backend": {"requested": "cpu", "actual": "cpu"},
        "prefilter": {"candidate_tile_count": 0},
        "exact_fallback": {"window_count": 0},
        "layers": (),
        "allowed_policy": {"mode": "deny"},
        "unexpected_summary": {"area_dbu2": "0"},
        "timings": {},
        "memory": {},
        "warnings": (),
        "errors": (),
        "artifacts": {},
    }
    data.update(kwargs)
    return CompareReport(**data)


def test_equivalent_report_validates_and_serializes():
    report = _base_report()

    payload = report.to_json_dict()

    assert payload["schema_version"] == "gdsdiff-report-v1"
    assert payload["status"] == "equivalent"
    assert payload["exit_code_category"] == 0
    assert payload["exact_verification_complete"] is True
    validate_report_dict(payload)


def test_expected_only_exit_code_depends_on_failure_policy():
    expected = _base_report(status=CompareStatus.EXPECTED_DIFFERENCES_ONLY)
    strict = _base_report(
        status=CompareStatus.EXPECTED_DIFFERENCES_ONLY,
        failure_policy=FailurePolicy.FAIL_ON_ANY_DIFF,
    )

    assert expected.to_json_dict()["exit_code_category"] == 0
    assert strict.to_json_dict()["exit_code_category"] == 1


@pytest.mark.parametrize(
    ("status", "exit_code"),
    (
        (CompareStatus.UNEXPECTED_DIFFERENCES, 1),
        (CompareStatus.UNRESOLVED, 2),
        (CompareStatus.INVALID_INPUT, 2),
        (CompareStatus.BACKEND_ERROR, 2),
        (CompareStatus.ARTIFACT_ERROR, 2),
    ),
)
def test_non_success_report_exit_codes(status, exit_code):
    assert _base_report(status=status, exact_verification_complete=False).to_json_dict()["exit_code_category"] == exit_code


def test_layer_report_preserves_exact_area_strings_and_truncates_components():
    layer = LayerReport(
        layer=LayerSpec(1, 2),
        present_old=True,
        present_new=True,
        xor_area_dbu2="123.5",
        old_minus_new_area_dbu2=100,
        new_minus_old_area_dbu2="23.5",
        unexpected_area_dbu2="23.5",
        exact_status="verified",
        components=(
            ComponentSummary("removal", (0, 0, 1, 1), 100, "expected"),
            ComponentSummary("addition", (2, 2, 3, 3), "23.5", "unexpected"),
        ),
        component_limit=1,
    )

    payload = layer.to_json()

    assert payload["xor_area_dbu2"] == "123.5"
    assert payload["old_minus_new_area_dbu2"] == "100"
    assert payload["new_minus_old_area_dbu2"] == "23.5"
    assert len(payload["components"]) == 1
    assert payload["truncated_component_count"] == 1
    validate_report_dict(_base_report(layers=(layer,)).to_json_dict())


def test_invalid_area_string_is_rejected():
    layer = LayerReport(
        layer=LayerSpec(1, 2),
        present_old=True,
        present_new=True,
        xor_area_dbu2="1.25",
    )

    with pytest.raises(ValueError, match="invalid exact area"):
        layer.to_json()


def test_missing_exact_verification_flag_is_rejected():
    payload = _base_report().to_json_dict()
    del payload["exact_verification_complete"]

    with pytest.raises(ValueError, match="missing required"):
        validate_report_dict(payload)


def test_write_json_is_atomic_and_readable(tmp_path):
    path = tmp_path / "report.json"
    written = _base_report(command=("gdsdiff", "--old", "old.gds")).write_json(path)

    assert written == path
    payload = json.loads(path.read_text())
    assert payload["command"] == ["gdsdiff", "--old", "old.gds"]
    validate_report_dict(payload)
