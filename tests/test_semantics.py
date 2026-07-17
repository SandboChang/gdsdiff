from __future__ import annotations

import pytest

from gdsdiff import Backend, CompareStatus, FailurePolicy, PathPolicy, exit_code_for_status


def test_string_enums_coerce_valid_values():
    assert Backend.coerce("cuda") is Backend.CUDA
    assert Backend.coerce(Backend.CPU) is Backend.CPU
    assert PathPolicy.coerce("report-only") is PathPolicy.REPORT_ONLY


def test_string_enums_reject_unknown_values():
    with pytest.raises(ValueError, match="invalid Backend"):
        Backend.coerce("opencl")

    with pytest.raises(ValueError, match="invalid PathPolicy"):
        PathPolicy.coerce("ignore")


@pytest.mark.parametrize(
    ("status", "policy", "expected"),
    (
        (CompareStatus.EQUIVALENT, FailurePolicy.FAIL_ON_UNEXPECTED, 0),
        (CompareStatus.EXPECTED_DIFFERENCES_ONLY, FailurePolicy.FAIL_ON_UNEXPECTED, 0),
        (CompareStatus.EXPECTED_DIFFERENCES_ONLY, FailurePolicy.FAIL_ON_ANY_DIFF, 1),
        (CompareStatus.UNEXPECTED_DIFFERENCES, FailurePolicy.FAIL_ON_UNEXPECTED, 1),
        (CompareStatus.UNRESOLVED, FailurePolicy.FAIL_ON_UNEXPECTED, 2),
        (CompareStatus.INVALID_INPUT, FailurePolicy.FAIL_ON_UNEXPECTED, 2),
        (CompareStatus.BACKEND_ERROR, FailurePolicy.FAIL_ON_UNEXPECTED, 2),
        (CompareStatus.ARTIFACT_ERROR, FailurePolicy.FAIL_ON_UNEXPECTED, 2),
    ),
)
def test_exit_code_mapping(status, policy, expected):
    assert exit_code_for_status(status, policy) == expected
