from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import LayerSpec
from .exact_oracle import ExactOracleResult, format_twice_area
from .policy import PolicyClassificationResult
from .semantics import CompareStatus, FailurePolicy, exit_code_for_status

REPORT_SCHEMA_VERSION = "gdsdiff-report-v1"
TOOL_VERSION = "0.1.0-dev"


def _area_string(value: int | str) -> str:
    if isinstance(value, int):
        return str(value)
    text = str(value)
    if not text or text == "-":
        raise ValueError("area string cannot be empty")
    if text.startswith("-"):
        body = text[1:]
    else:
        body = text
    if body.endswith(".5"):
        body = body[:-2]
    if not body.isdigit():
        raise ValueError(f"invalid exact area string: {value!r}")
    return text


@dataclass(frozen=True)
class ComponentSummary:
    direction: str
    bbox: tuple[int, int, int, int]
    area_dbu2: int | str
    classification: str = "unclassified"

    def to_json(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "bbox": list(self.bbox),
            "area_dbu2": _area_string(self.area_dbu2),
            "classification": self.classification,
        }


@dataclass(frozen=True)
class LayerReport:
    layer: LayerSpec
    present_old: bool
    present_new: bool
    xor_area_dbu2: int | str = 0
    old_minus_new_area_dbu2: int | str = 0
    new_minus_old_area_dbu2: int | str = 0
    expected_area_dbu2: int | str = 0
    unexpected_area_dbu2: int | str = 0
    polygon_count_old: int = 0
    polygon_count_new: int = 0
    candidate_tile_count: int = 0
    exact_window_count: int = 0
    exact_status: str = "not-run"
    components: tuple[ComponentSummary, ...] = field(default_factory=tuple)
    component_limit: int = 100

    def to_json(self) -> dict[str, Any]:
        shown = self.components[: self.component_limit]
        return {
            "layer": self.layer.layer,
            "datatype": self.layer.datatype,
            "present_old": self.present_old,
            "present_new": self.present_new,
            "polygon_count_old": self.polygon_count_old,
            "polygon_count_new": self.polygon_count_new,
            "candidate_tile_count": self.candidate_tile_count,
            "exact_window_count": self.exact_window_count,
            "xor_area_dbu2": _area_string(self.xor_area_dbu2),
            "old_minus_new_area_dbu2": _area_string(self.old_minus_new_area_dbu2),
            "new_minus_old_area_dbu2": _area_string(self.new_minus_old_area_dbu2),
            "expected_area_dbu2": _area_string(self.expected_area_dbu2),
            "unexpected_area_dbu2": _area_string(self.unexpected_area_dbu2),
            "components": [component.to_json() for component in shown],
            "truncated_component_count": max(0, len(self.components) - len(shown)),
            "exact_status": self.exact_status,
        }


@dataclass(frozen=True)
class CompareReport:
    status: CompareStatus | str
    exact_verification_complete: bool
    old_input: dict[str, Any]
    new_input: dict[str, Any]
    semantics: dict[str, Any]
    selection: dict[str, Any]
    backend: dict[str, Any]
    prefilter: dict[str, Any]
    exact_fallback: dict[str, Any]
    layers: tuple[LayerReport, ...] = field(default_factory=tuple)
    allowed_policy: dict[str, Any] = field(default_factory=dict)
    unexpected_summary: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, Any] = field(default_factory=dict)
    memory: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    errors: tuple[str, ...] = field(default_factory=tuple)
    artifacts: dict[str, Any] = field(default_factory=dict)
    evidence_status: str | None = None
    differences_found: bool | None = None
    coverage_changed: bool | None = None
    inventory_changed: bool | None = None
    coverage_equal_but_inventory_changed: bool | None = None
    coverage_component_count: int | None = None
    inventory_delta_count: int | None = None
    vertex_sidecar_path: str | None = None
    unsupported_feature_count: int = 0
    command: tuple[str, ...] = field(default_factory=tuple)
    failure_policy: FailurePolicy | str = FailurePolicy.FAIL_ON_UNEXPECTED
    tool_version: str = TOOL_VERSION

    def to_json_dict(self) -> dict[str, Any]:
        status = CompareStatus.coerce(self.status)
        failure_policy = FailurePolicy.coerce(self.failure_policy)
        coverage_changed = self.coverage_changed
        if coverage_changed is None:
            coverage_changed = any(
                _area_string(layer.xor_area_dbu2) != "0"
                for layer in self.layers
            )
        differences_found = self.differences_found
        if differences_found is None:
            differences_found = status in {
                CompareStatus.DIFFERENCES_FOUND,
                CompareStatus.EXPECTED_DIFFERENCES_ONLY,
                CompareStatus.UNEXPECTED_DIFFERENCES,
            }
        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "tool_version": self.tool_version,
            "command": list(self.command),
            "status": status.value,
            "exit_code_category": exit_code_for_status(status, failure_policy),
            "evidence_status": self.evidence_status or _default_evidence_status(status),
            "differences_found": differences_found,
            "coverage_changed": coverage_changed,
            "inventory_changed": self.inventory_changed,
            "coverage_equal_but_inventory_changed": self.coverage_equal_but_inventory_changed,
            "coverage_component_count": self.coverage_component_count,
            "inventory_delta_count": self.inventory_delta_count,
            "vertex_sidecar_path": self.vertex_sidecar_path,
            "unsupported_feature_count": self.unsupported_feature_count,
            "exact_verification_complete": self.exact_verification_complete,
            "old_input": self.old_input,
            "new_input": self.new_input,
            "semantics": self.semantics,
            "selection": self.selection,
            "backend": self.backend,
            "prefilter": self.prefilter,
            "exact_fallback": self.exact_fallback,
            "layers": [layer.to_json() for layer in self.layers],
            "allowed_policy": self.allowed_policy,
            "unexpected_summary": self.unexpected_summary,
            "timings": self.timings,
            "memory": self.memory,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "artifacts": self.artifacts,
        }
        validate_report_dict(report)
        return report

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_json_dict(), indent=indent, sort_keys=True)

    def write_json(self, path: str | Path, *, indent: int | None = 2) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_json(indent=indent)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(data)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
        except Exception:
            raise
        return path


def build_layer_reports(
    exact_result: ExactOracleResult,
    *,
    policy_result: PolicyClassificationResult | None = None,
    presence_by_layer: dict[LayerSpec, tuple[bool, bool]] | None = None,
    polygon_counts_by_layer: dict[LayerSpec, tuple[int, int]] | None = None,
    candidate_tile_counts_by_layer: dict[LayerSpec, int] | None = None,
    exact_window_counts_by_layer: dict[LayerSpec, int] | None = None,
    exact_status: str = "complete",
    component_limit: int = 100,
) -> tuple[LayerReport, ...]:
    policy_by_layer = {layer.layer: layer for layer in policy_result.layers} if policy_result else {}
    reports = []
    for layer_diff in exact_result.layers:
        layer = layer_diff.layer
        present_old, present_new = (presence_by_layer or {}).get(layer, (False, False))
        polygon_count_old, polygon_count_new = (polygon_counts_by_layer or {}).get(layer, (0, 0))
        policy_layer = policy_by_layer.get(layer)
        components = ()
        expected_area = "0"
        unexpected_area = "0"
        if policy_layer is not None:
            expected_area = format_twice_area(policy_layer.expected_twice_area)
            unexpected_area = format_twice_area(policy_layer.unexpected_twice_area)
            components = tuple(
                ComponentSummary(
                    direction=component.direction.value,
                    bbox=component.bbox,
                    area_dbu2=component.area_dbu2_string,
                    classification=component.classification,
                )
                for component in policy_layer.components
            )
        reports.append(
            LayerReport(
                layer=layer,
                present_old=present_old,
                present_new=present_new,
                xor_area_dbu2=format_twice_area(layer_diff.xor_twice_area),
                old_minus_new_area_dbu2=format_twice_area(layer_diff.old_minus_new_twice_area),
                new_minus_old_area_dbu2=format_twice_area(layer_diff.new_minus_old_twice_area),
                expected_area_dbu2=expected_area,
                unexpected_area_dbu2=unexpected_area,
                polygon_count_old=polygon_count_old,
                polygon_count_new=polygon_count_new,
                candidate_tile_count=(candidate_tile_counts_by_layer or {}).get(layer, 0),
                exact_window_count=(exact_window_counts_by_layer or {}).get(layer, 0),
                exact_status=exact_status,
                components=components,
                component_limit=component_limit,
            )
        )
    return tuple(sorted(reports, key=lambda report: report.layer))


def validate_report_dict(report: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "tool_version",
        "status",
        "exit_code_category",
        "evidence_status",
        "differences_found",
        "coverage_changed",
        "inventory_changed",
        "coverage_equal_but_inventory_changed",
        "coverage_component_count",
        "inventory_delta_count",
        "vertex_sidecar_path",
        "unsupported_feature_count",
        "exact_verification_complete",
        "old_input",
        "new_input",
        "semantics",
        "selection",
        "backend",
        "prefilter",
        "exact_fallback",
        "layers",
        "allowed_policy",
        "unexpected_summary",
        "timings",
        "memory",
        "warnings",
        "errors",
        "artifacts",
    }
    missing = sorted(required - set(report))
    if missing:
        raise ValueError(f"report missing required fields: {', '.join(missing)}")
    if report["schema_version"] != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported report schema_version")
    CompareStatus.coerce(report["status"])
    if report["exit_code_category"] not in {0, 1, 2}:
        raise ValueError("exit_code_category must be 0, 1, or 2")
    if report["evidence_status"] not in {
        "identical",
        "differences-found",
        "unsupported-input",
        "incomplete-evidence",
        "internal-mismatch",
    }:
        raise ValueError("invalid evidence_status")
    for key in ("differences_found", "coverage_changed"):
        if not isinstance(report[key], bool):
            raise ValueError(f"{key} must be a boolean")
    if not isinstance(report["exact_verification_complete"], bool):
        raise ValueError("exact_verification_complete must be a boolean")
    if not isinstance(report["layers"], list):
        raise ValueError("layers must be a list")
    for layer in report["layers"]:
        _validate_layer_report(layer)


def _validate_layer_report(layer: dict[str, Any]) -> None:
    required = {
        "layer",
        "datatype",
        "present_old",
        "present_new",
        "xor_area_dbu2",
        "old_minus_new_area_dbu2",
        "new_minus_old_area_dbu2",
        "expected_area_dbu2",
        "unexpected_area_dbu2",
        "exact_status",
    }
    missing = sorted(required - set(layer))
    if missing:
        raise ValueError(f"layer report missing required fields: {', '.join(missing)}")
    LayerSpec(layer["layer"], layer["datatype"])
    if not isinstance(layer["present_old"], bool) or not isinstance(layer["present_new"], bool):
        raise ValueError("layer presence fields must be boolean")
    for key in (
        "xor_area_dbu2",
        "old_minus_new_area_dbu2",
        "new_minus_old_area_dbu2",
        "expected_area_dbu2",
        "unexpected_area_dbu2",
    ):
        _area_string(layer[key])


def _default_evidence_status(status: CompareStatus) -> str:
    if status in {CompareStatus.IDENTICAL, CompareStatus.EQUIVALENT}:
        return "identical"
    if status in {
        CompareStatus.DIFFERENCES_FOUND,
        CompareStatus.EXPECTED_DIFFERENCES_ONLY,
        CompareStatus.UNEXPECTED_DIFFERENCES,
    }:
        return "differences-found"
    if status in {CompareStatus.INVALID_INPUT, CompareStatus.UNSUPPORTED_INPUT}:
        return "unsupported-input"
    if status in {CompareStatus.UNRESOLVED, CompareStatus.INCOMPLETE_EVIDENCE}:
        return "incomplete-evidence"
    if status == CompareStatus.INTERNAL_MISMATCH:
        return "internal-mismatch"
    return "incomplete-evidence"
