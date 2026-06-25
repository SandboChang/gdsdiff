from __future__ import annotations

from enum import Enum


class _StringEnum(str, Enum):
    @classmethod
    def coerce(cls, value):
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value))
        except ValueError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise ValueError(f"invalid {cls.__name__}: {value!r}; expected one of {allowed}") from exc


class Backend(_StringEnum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    METAL = "metal"


class PathPolicy(_StringEnum):
    GDSTK = "gdstk"
    REJECT = "reject"
    REPORT_ONLY = "report-only"


class DifferenceDirection(_StringEnum):
    BOTH = "both"
    ADDITION = "addition"
    REMOVAL = "removal"
    XOR = "xor"


class FailurePolicy(_StringEnum):
    FAIL_ON_UNEXPECTED = "fail-on-unexpected"
    FAIL_ON_ANY_DIFF = "fail-on-any-diff"


class CompareStatus(_StringEnum):
    EQUIVALENT = "equivalent"
    EXPECTED_DIFFERENCES_ONLY = "expected-differences-only"
    UNEXPECTED_DIFFERENCES = "unexpected-differences"
    UNRESOLVED = "unresolved"
    INVALID_INPUT = "invalid-input"
    BACKEND_ERROR = "backend-error"
    ARTIFACT_ERROR = "artifact-error"


def exit_code_for_status(status: CompareStatus | str, failure_policy: FailurePolicy | str = FailurePolicy.FAIL_ON_UNEXPECTED) -> int:
    status = CompareStatus.coerce(status)
    failure_policy = FailurePolicy.coerce(failure_policy)
    if status == CompareStatus.EQUIVALENT:
        return 0
    if status == CompareStatus.EXPECTED_DIFFERENCES_ONLY:
        return 1 if failure_policy == FailurePolicy.FAIL_ON_ANY_DIFF else 0
    if status == CompareStatus.UNEXPECTED_DIFFERENCES:
        return 1
    return 2
