from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from .exact_oracle import format_twice_area
from .geometry import PolygonBuffer, polygon_bbox, polygon_twice_area
from .policy import ClassifiedComponent, PolicyClassificationResult
from .semantics import DifferenceDirection

DEFAULT_VERTEX_INLINE_LIMIT = 8
DEFAULT_INVENTORY_RECORD_LIMIT = None
DEFAULT_COMPONENT_RECORD_LIMIT = 100
DEFAULT_COMPONENT_DETAIL_MODE = "compact"
DEFAULT_DIGEST_COMPONENT_LIMIT = 8


@dataclass(frozen=True)
class ExactDiffMarkdownWriteResult:
    path: Path
    component_count: int
    vertex_sidecar_path: Path | None = None


@dataclass(frozen=True)
class PolygonInventoryRecord:
    layer: int
    datatype: int
    direction: DifferenceDirection
    points: tuple[tuple[int, int], ...]
    bbox: tuple[int, int, int, int]
    twice_area: int

    @property
    def area_dbu2_string(self) -> str:
        return format_twice_area(self.twice_area)


@dataclass(frozen=True)
class PolygonInventoryDelta:
    removed: tuple[PolygonInventoryRecord, ...]
    added: tuple[PolygonInventoryRecord, ...]

    @property
    def changed_count(self) -> int:
        return len(self.removed) + len(self.added)

    @property
    def layers(self) -> tuple[tuple[int, int], ...]:
        return tuple(sorted({(record.layer, record.datatype) for record in self.removed + self.added}))


def build_polygon_inventory_delta(old_buffer: PolygonBuffer, new_buffer: PolygonBuffer) -> PolygonInventoryDelta:
    return _inventory_delta_for_indices(
        old_buffer, range(old_buffer.polygon_count), new_buffer, range(new_buffer.polygon_count),
    )


def _inventory_delta_for_indices(
    old_buffer: PolygonBuffer,
    old_indices: Iterable[int],
    new_buffer: PolygonBuffer,
    new_indices: Iterable[int],
) -> PolygonInventoryDelta:
    old_counter, old_indices_by_key = _polygon_inventory_index(old_buffer, old_indices)
    new_counter, new_indices_by_key = _polygon_inventory_index(new_buffer, new_indices)
    removed: list[PolygonInventoryRecord] = []
    added: list[PolygonInventoryRecord] = []
    for key in sorted(set(old_counter) | set(new_counter)):
        removed_count = max(0, old_counter[key] - new_counter[key])
        added_count = max(0, new_counter[key] - old_counter[key])
        removed.extend(
            _polygon_inventory_record(old_buffer, index, DifferenceDirection.REMOVAL)
            for index in old_indices_by_key[key][:removed_count]
        )
        added.extend(
            _polygon_inventory_record(new_buffer, index, DifferenceDirection.ADDITION)
            for index in new_indices_by_key[key][:added_count]
        )
    return PolygonInventoryDelta(
        removed=tuple(sorted(removed, key=_inventory_sort_key)),
        added=tuple(sorted(added, key=_inventory_sort_key)),
    )


def build_polygon_inventory_delta_from_record_delta(
    old_buffer: PolygonBuffer,
    new_buffer: PolygonBuffer,
    record_delta,
) -> PolygonInventoryDelta:
    removed: list[PolygonInventoryRecord] = []
    added: list[PolygonInventoryRecord] = []
    for row in range(record_delta.record_count):
        count = int(record_delta.counts[row])
        representative_index = int(record_delta.representative_indices[row])
        if int(record_delta.directions[row]) == 1:
            removed.extend(
                _polygon_inventory_record(old_buffer, representative_index, DifferenceDirection.REMOVAL)
                for _ in range(count)
            )
        else:
            added.extend(
                _polygon_inventory_record(new_buffer, representative_index, DifferenceDirection.ADDITION)
                for _ in range(count)
            )
    return _polygon_inventory_delta_from_records(removed, added)


def _polygon_inventory_delta_from_records(
    removed_records: list[PolygonInventoryRecord],
    added_records: list[PolygonInventoryRecord],
) -> PolygonInventoryDelta:
    old_counter = Counter(_inventory_key(record) for record in removed_records)
    new_counter = Counter(_inventory_key(record) for record in added_records)
    old_by_key = defaultdict(list)
    new_by_key = defaultdict(list)
    for record in removed_records:
        old_by_key[_inventory_key(record)].append(record)
    for record in added_records:
        new_by_key[_inventory_key(record)].append(record)
    removed: list[PolygonInventoryRecord] = []
    added: list[PolygonInventoryRecord] = []
    for key in sorted(set(old_counter) | set(new_counter)):
        removed_count = max(0, old_counter[key] - new_counter[key])
        added_count = max(0, new_counter[key] - old_counter[key])
        removed.extend(old_by_key[key][:removed_count])
        added.extend(new_by_key[key][:added_count])
    return PolygonInventoryDelta(
        removed=tuple(sorted(removed, key=_inventory_sort_key)),
        added=tuple(sorted(added, key=_inventory_sort_key)),
    )


def render_exact_diff_markdown(
    policy_result: PolicyClassificationResult,
    *,
    title: str = "GDS exact diff",
    source_mode: str = "exact",
    exact_verification_complete: bool = True,
    polygon_inventory_delta: PolygonInventoryDelta | None = None,
    component_limit: int | None = DEFAULT_COMPONENT_RECORD_LIMIT,
    vertex_limit: int | None = DEFAULT_VERTEX_INLINE_LIMIT,
    inventory_limit: int | None = DEFAULT_INVENTORY_RECORD_LIMIT,
    component_detail_mode: str = DEFAULT_COMPONENT_DETAIL_MODE,
    vertex_sidecar_path: str | Path | None = None,
    sidecar_refs: Mapping[object, str] | None = None,
    generated_at: datetime | None = None,
) -> str:
    """Render a deterministic, LLM-friendly Markdown diff from classified exact components."""
    generated_at = generated_at or datetime.now(timezone.utc)
    components = _components(policy_result)
    shown_components = components if component_limit is None else components[: max(component_limit, 0)]
    coverage_twice_area = _coverage_twice_area(components)
    inventory_changed = polygon_inventory_delta is not None and polygon_inventory_delta.changed_count > 0
    evidence_status = _neutral_evidence_status(
        coverage_twice_area=coverage_twice_area,
        inventory_changed=inventory_changed,
        exact_verification_complete=exact_verification_complete,
    )
    sidecar_path_text = str(vertex_sidecar_path) if vertex_sidecar_path is not None else "not-written"
    lines = [
        f"# {title}",
        "",
        "## Comparator Evidence",
        "",
        f"- evidence_status: `{evidence_status}`",
        f"- differences_found: `{str(evidence_status == 'differences-found').lower()}`",
        f"- source_mode: `{source_mode}`",
        f"- exact_verification_complete: `{str(exact_verification_complete).lower()}`",
        f"- generated_at_utc: `{generated_at.astimezone(timezone.utc).isoformat()}`",
        f"- coverage_changed: `{str(coverage_twice_area != 0).lower()}`",
        f"- inventory_changed: `{str(inventory_changed).lower()}`",
        f"- coverage_equal_but_inventory_changed: `{str(coverage_twice_area == 0 and inventory_changed).lower()}`",
        f"- coverage_before_only_area_dbu2: `{format_twice_area(_coverage_direction_twice_area(components, DifferenceDirection.REMOVAL))}`",
        f"- coverage_after_only_area_dbu2: `{format_twice_area(_coverage_direction_twice_area(components, DifferenceDirection.ADDITION))}`",
        f"- coverage_xor_area_dbu2: `{format_twice_area(coverage_twice_area)}`",
        f"- layer_count: `{len(policy_result.layers)}`",
        f"- coverage_component_count: `{len(components)}`",
        f"- component_count: `{len(components)}`",
        f"- polygon_inventory_delta_count: `{polygon_inventory_delta.changed_count if polygon_inventory_delta is not None else 'not-recorded'}`",
        f"- vertex_sidecar_path: `{sidecar_path_text}`",
        "",
    ]
    if not exact_verification_complete:
        lines.extend(
            [
                "## Evidence Completeness Warning",
                "",
                "This file does not prove complete full-file polygon evidence because the comparator run did not complete exact verification. Use an exact run, or a GPU run explicitly verified by the exact oracle, for acceptance review.",
                "",
            ]
        )
    lines.extend(
        _render_agent_minimal_read(
            policy_result,
            components,
            polygon_inventory_delta=polygon_inventory_delta,
            digest_component_limit=DEFAULT_DIGEST_COMPONENT_LIMIT,
            exact_verification_complete=exact_verification_complete,
        )
    )
    lines.extend(_render_agent_review_checklist())
    lines.extend(_render_layer_summary(policy_result))
    lines.extend(
        _render_components(
            shown_components,
            vertex_limit=vertex_limit,
            detail_mode=component_detail_mode,
            sidecar_refs=sidecar_refs,
        )
    )
    if polygon_inventory_delta is not None:
        lines.extend(
            _render_polygon_inventory_delta(
                polygon_inventory_delta,
                policy_result=policy_result,
                inventory_limit=inventory_limit,
                vertex_limit=vertex_limit,
                detail_mode=component_detail_mode,
                sidecar_refs=sidecar_refs,
            )
        )
    if component_limit is not None and len(components) > len(shown_components):
        lines.extend(
            [
                "",
                "## Truncation",
                "",
                f"- omitted_component_count: `{len(components) - len(shown_components)}`",
                "- note: rerun with no component limit to obtain every polygon-level component.",
            ]
        )
    return "\n".join(lines) + "\n"


def _render_agent_minimal_read(
    policy_result: PolicyClassificationResult,
    components: tuple[ClassifiedComponent, ...],
    *,
    polygon_inventory_delta: PolygonInventoryDelta | None,
    digest_component_limit: int,
    exact_verification_complete: bool,
) -> list[str]:
    coverage_twice_area = policy_result.expected_twice_area + policy_result.unexpected_twice_area
    inventory_changed = polygon_inventory_delta is not None and polygon_inventory_delta.changed_count > 0
    evidence_status = _neutral_evidence_status(
        coverage_twice_area=coverage_twice_area,
        inventory_changed=inventory_changed,
        exact_verification_complete=exact_verification_complete,
    )
    count_by_direction = Counter(_coverage_direction_label(component.direction) for component in components)
    largest = tuple(sorted(components, key=lambda component: (-abs(component.twice_area), _component_sort_key(component)))[:digest_component_limit])
    lines = [
        "## Agent Minimal Read",
        "",
        "- purpose: `default first section for LLM review; read this before expanding detail tables`",
        "- recommended_minimum_sections: `Comparator Evidence`, `Agent Minimal Read`, `Agent Review Checklist`, `Layer Summary`",
        "- expand_polygon_components_when: `need exact bbox/area/hash/sidecar id for every changed coverage component`",
        "- expand_inventory_delta_when: `coverage_changed is false but inventory_changed is true, or polygon formation changed`",
        f"- evidence_status: `{evidence_status}`",
        f"- coverage_changed: `{str(coverage_twice_area != 0).lower()}`",
        f"- coverage_xor_area_dbu2: `{format_twice_area(coverage_twice_area)}`",
        f"- inventory_changed: `{str(inventory_changed).lower()}`",
        f"- coverage_equal_but_inventory_changed: `{str(coverage_twice_area == 0 and inventory_changed).lower()}`",
        f"- inventory_delta_count: `{polygon_inventory_delta.changed_count if polygon_inventory_delta is not None else 'not-recorded'}`",
        f"- coverage_component_count: `{len(components)}`",
    ]
    lines.append(f"- coverage_component_counts_by_direction: `{_coverage_direction_count_text(count_by_direction)}`")
    lines.append("")
    lines.extend(_render_agent_layer_digest(policy_result))
    lines.extend(_render_agent_largest_component_digest(largest, total_count=len(components)))
    lines.extend(_render_agent_largest_directional_component_digest(components))
    return lines


def _render_agent_layer_digest(policy_result: PolicyClassificationResult) -> list[str]:
    lines = [
        "### Affected Layer Digest",
        "",
        "| layer | datatype | coverage_xor_area_dbu2 | coverage_before_only_area_dbu2 | coverage_after_only_area_dbu2 | before_only_components | after_only_components | before_only_bbox_union_dbu | after_only_bbox_union_dbu | coverage_xor_bbox_union_dbu | components |",
        "|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|",
    ]
    for layer in policy_result.layers:
        exact_twice_area = _coverage_twice_area(layer.components)
        before_components = _coverage_direction_components(layer.components, DifferenceDirection.REMOVAL)
        after_components = _coverage_direction_components(layer.components, DifferenceDirection.ADDITION)
        bboxes = [component.bbox for component in layer.components]
        bbox_union = _bbox_union(bboxes)
        lines.append(
            "| {layer} | {datatype} | {area} | {before_area} | {after_area} | {before_count} | {after_count} | `{before_bbox}` | `{after_bbox}` | `{bbox}` | {count} |".format(
                layer=layer.layer.layer,
                datatype=layer.layer.datatype,
                area=format_twice_area(exact_twice_area),
                before_area=format_twice_area(_coverage_twice_area(before_components)),
                after_area=format_twice_area(_coverage_twice_area(after_components)),
                before_count=len(before_components),
                after_count=len(after_components),
                before_bbox=_bbox_json(_bbox_union(tuple(component.bbox for component in before_components))),
                after_bbox=_bbox_json(_bbox_union(tuple(component.bbox for component in after_components))),
                count=len(layer.components),
                bbox=_bbox_json(bbox_union),
            )
        )
    if not policy_result.layers:
        lines.append("| - | - | 0 | 0 | 0 | 0 | 0 | `none` | `none` | `none` | 0 |")
    lines.append("")
    return lines


def _render_agent_largest_component_digest(
    components: tuple[ClassifiedComponent, ...],
    *,
    total_count: int,
) -> list[str]:
    lines = [
        "### Largest Component Digest",
        "",
        "| rank | direction | layer | datatype | bbox_dbu | area_dbu2 | vertex_count | points_sha256 |",
        "|---:|---|---:|---:|---|---:|---:|---|",
    ]
    for rank, component in enumerate(components, start=1):
        lines.append(
            "| {rank} | `{direction}` | {layer} | {datatype} | `{bbox}` | `{area}` | {vertex_count} | `{sha}` |".format(
                rank=rank,
                direction=_coverage_direction_label(component.direction),
                layer=component.layer.layer,
                datatype=component.layer.datatype,
                bbox=json.dumps(list(component.bbox), separators=(",", ":")),
                area=component.area_dbu2_string,
                vertex_count=len(component.points),
                sha=_points_hash(component),
            )
        )
    if not components:
        lines.append("| - | - | - | - | `none` | 0 | 0 | `none` |")
    if total_count > len(components):
        lines.append(
            "| ... | `see Polygon Components` | - | - | `-` | `-` | - | `-` |"
        )
    lines.append("")
    return lines


def _render_agent_largest_directional_component_digest(
    components: tuple[ClassifiedComponent, ...],
) -> list[str]:
    lines = [
        "### Largest Directional Components",
        "",
        "| direction | layer | datatype | bbox_dbu | area_dbu2 | vertex_count | points_sha256 |",
        "|---|---:|---:|---|---:|---:|---|",
    ]
    rows = []
    for direction in (DifferenceDirection.REMOVAL, DifferenceDirection.ADDITION):
        direction_components = _coverage_direction_components(components, direction)
        if direction_components:
            rows.append(
                sorted(
                    direction_components,
                    key=lambda component: (-abs(component.twice_area), _component_sort_key(component)),
                )[0]
            )
    for component in rows:
        lines.append(
            "| `{direction}` | {layer} | {datatype} | `{bbox}` | `{area}` | {vertex_count} | `{sha}` |".format(
                direction=_coverage_direction_label(component.direction),
                layer=component.layer.layer,
                datatype=component.layer.datatype,
                bbox=json.dumps(list(component.bbox), separators=(",", ":")),
                area=component.area_dbu2_string,
                vertex_count=len(component.points),
                sha=_points_hash(component),
            )
        )
    if not rows:
        lines.append("| - | - | - | `none` | 0 | 0 | `none` |")
    lines.append("")
    return lines


def _render_agent_review_checklist() -> list[str]:
    return [
        "## Agent Review Checklist",
        "",
        "- intent_contract_required: `record intended feature location or bbox, affected layers, intended direction such as add/remove/grow/shrink/move, and forbidden layers or regions before accepting the diff`",
        "- locality_check: `compare affected layer and directional bbox evidence against the intended feature region; expand Polygon Components when a bbox is outside that region or too coarse to judge`",
        "- directionality_check: `coverage_before_only is old-only material and coverage_after_only is new-only material; grow, shrink, and move edits should have directional bboxes and areas consistent with that intent`",
        "- outside_region_check: `look for changed layers or component bboxes outside the intended feature area; if a precise window is known, optionally rerun with --allow-region as a stricter second pass`",
        "- forbidden_layer_check: `nonzero coverage on layers outside the intended layer set requires explicit design justification before acceptance`",
        "- readback_invariant_check: `verify decisive dimensions, counts, overlaps, and containment from GDS coordinates; total layer area and polygon count are triage only`",
        "- reviewer_classification_required: `classify each nontrivial component as expected, unexpected, or unresolved in the closing report or design history, tied to layer, direction, bbox, and area`",
        "",
    ]


def write_exact_diff_markdown(
    policy_result: PolicyClassificationResult,
    path: str | Path,
    *,
    title: str = "GDS exact diff",
    source_mode: str = "exact",
    exact_verification_complete: bool = True,
    polygon_inventory_delta: PolygonInventoryDelta | None = None,
    component_limit: int | None = DEFAULT_COMPONENT_RECORD_LIMIT,
    vertex_limit: int | None = DEFAULT_VERTEX_INLINE_LIMIT,
    inventory_limit: int | None = DEFAULT_INVENTORY_RECORD_LIMIT,
    component_detail_mode: str = DEFAULT_COMPONENT_DETAIL_MODE,
    vertex_sidecar_path: str | Path | None = None,
    generated_at: datetime | None = None,
) -> ExactDiffMarkdownWriteResult:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path = Path(vertex_sidecar_path) if vertex_sidecar_path is not None else _default_vertex_sidecar_path(path)
    sidecar_refs = _write_vertex_sidecar(
        policy_result,
        polygon_inventory_delta=polygon_inventory_delta,
        path=sidecar_path,
    )
    path.write_text(
        render_exact_diff_markdown(
            policy_result,
            title=title,
            source_mode=source_mode,
            exact_verification_complete=exact_verification_complete,
            polygon_inventory_delta=polygon_inventory_delta,
            component_limit=component_limit,
            vertex_limit=vertex_limit,
            inventory_limit=inventory_limit,
            component_detail_mode=component_detail_mode,
            vertex_sidecar_path=sidecar_path,
            sidecar_refs=sidecar_refs,
            generated_at=generated_at,
        ),
        encoding="utf-8",
    )
    return ExactDiffMarkdownWriteResult(path=path, component_count=len(_components(policy_result)), vertex_sidecar_path=sidecar_path)


def _render_layer_summary(policy_result: PolicyClassificationResult) -> list[str]:
    lines = ["## Layer Summary", ""]
    if not policy_result.layers:
        lines.append("- no differences")
        lines.append("")
        return lines
    lines.extend(
        [
            "| layer | datatype | coverage_before_only_area_dbu2 | coverage_after_only_area_dbu2 | before_only_components | after_only_components | before_only_bbox_union_dbu | after_only_bbox_union_dbu | components |",
            "|---:|---:|---:|---:|---:|---:|---|---|---:|",
        ]
    )
    for layer in policy_result.layers:
        before_components = _coverage_direction_components(layer.components, DifferenceDirection.REMOVAL)
        after_components = _coverage_direction_components(layer.components, DifferenceDirection.ADDITION)
        lines.append(
            "| {layer} | {datatype} | {before} | {after} | {before_count} | {after_count} | `{before_bbox}` | `{after_bbox}` | {count} |".format(
                layer=layer.layer.layer,
                datatype=layer.layer.datatype,
                before=format_twice_area(_coverage_twice_area(before_components)),
                after=format_twice_area(_coverage_twice_area(after_components)),
                before_count=len(before_components),
                after_count=len(after_components),
                before_bbox=_bbox_json(_bbox_union(tuple(component.bbox for component in before_components))),
                after_bbox=_bbox_json(_bbox_union(tuple(component.bbox for component in after_components))),
                count=len(layer.components),
            )
        )
    lines.append("")
    return lines


def _render_polygon_inventory_delta(
    delta: PolygonInventoryDelta,
    *,
    policy_result: PolicyClassificationResult,
    inventory_limit: int | None,
    vertex_limit: int | None,
    detail_mode: str,
    sidecar_refs: Mapping[object, str] | None,
) -> list[str]:
    records = tuple(sorted(delta.removed + delta.added, key=_inventory_sort_key))
    shown_records = records if inventory_limit is None else records[: max(inventory_limit, 0)]
    coverage_area = policy_result.expected_twice_area + policy_result.unexpected_twice_area
    lines = [
        "## Input Polygon Inventory Delta",
        "",
        "This section compares source polygon records before boolean union/XOR. It is intended to expose split, merge, fracture, or vertex-order-level input changes that may have zero physical coverage difference.",
        "",
        f"- removed_polygon_records: `{len(delta.removed)}`",
        f"- added_polygon_records: `{len(delta.added)}`",
        f"- changed_polygon_records: `{delta.changed_count}`",
        f"- exact_coverage_xor_area_dbu2: `{format_twice_area(coverage_area)}`",
    ]
    lines.append(f"- coverage_equal_but_inventory_changed: `{str(delta.changed_count > 0 and coverage_area == 0).lower()}`")
    lines.append("")
    lines.extend(_render_inventory_layer_summary(delta))
    if not records:
        lines.extend(["No input polygon inventory changes were found.", ""])
        return lines
    lines.extend(["### Polygon Record Changes", ""])
    if detail_mode == "compact":
        lines.extend(
            [
                "| index | direction | layer | datatype | bbox_dbu | area_dbu2 | vertex_count | sidecar_id | points_sha256 | omitted_vertices | vertices_preview_dbu |",
                "|---:|---|---:|---:|---|---:|---:|---|---|---:|---|",
            ]
        )
        for index, record in enumerate(shown_records, start=1):
            points = record.points if vertex_limit is None else record.points[: max(vertex_limit, 0)]
            omitted = len(record.points) - len(points)
            lines.append(
                "| {index} | `{direction}` | {layer} | {datatype} | `{bbox}` | `{area}` | {vertex_count} | `{sidecar}` | `{sha}` | {omitted} | `{vertices}` |".format(
                    index=index,
                    direction=_inventory_direction_label(record.direction),
                    layer=record.layer,
                    datatype=record.datatype,
                    bbox=json.dumps(list(record.bbox), separators=(",", ":")),
                    area=record.area_dbu2_string,
                    vertex_count=len(record.points),
                    sidecar=_sidecar_ref(sidecar_refs, record),
                    sha=_inventory_points_hash(record),
                    omitted=omitted,
                    vertices=json.dumps([list(point) for point in points], separators=(",", ":")),
                )
            )
        lines.append("")
    elif detail_mode == "full":
        for index, record in enumerate(shown_records, start=1):
            points = record.points if vertex_limit is None else record.points[: max(vertex_limit, 0)]
            truncated_vertices = len(record.points) - len(points)
            lines.extend(
                [
                    f"#### Polygon Record {index}",
                    "",
                    f"- direction: `{_inventory_direction_label(record.direction)}`",
                    f"- layer: `{record.layer}`",
                    f"- datatype: `{record.datatype}`",
                    f"- bbox_dbu: `{list(record.bbox)}`",
                    f"- area_dbu2: `{record.area_dbu2_string}`",
                    f"- vertex_count: `{len(record.points)}`",
                    f"- sidecar_id: `{_sidecar_ref(sidecar_refs, record)}`",
                    f"- points_sha256: `{_inventory_points_hash(record)}`",
                ]
            )
            if truncated_vertices:
                lines.append(f"- omitted_vertex_count: `{truncated_vertices}`")
            lines.extend(
                [
                    "",
                    "```json",
                    json.dumps(
                        {
                            "direction": _inventory_direction_label(record.direction),
                            "layer": record.layer,
                            "datatype": record.datatype,
                            "bbox_dbu": list(record.bbox),
                            "area_dbu2": record.area_dbu2_string,
                            "sidecar_id": _sidecar_ref(sidecar_refs, record),
                            "vertices_dbu": [list(point) for point in points],
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    "```",
                    "",
                ]
            )
    else:
        raise ValueError(f"unsupported component detail mode: {detail_mode!r}")
    if inventory_limit is not None and len(records) > len(shown_records):
        lines.extend(
            [
                "### Inventory Truncation",
                "",
                f"- omitted_polygon_record_count: `{len(records) - len(shown_records)}`",
                "- note: rerun with no inventory limit to obtain every source polygon record delta.",
                "",
            ]
        )
    return lines


def _render_inventory_layer_summary(delta: PolygonInventoryDelta) -> list[str]:
    lines = [
        "### Inventory Layer Summary",
        "",
        "| layer | datatype | removed_records | added_records | removed_area_dbu2 | added_area_dbu2 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for layer, datatype in delta.layers:
        removed = tuple(record for record in delta.removed if record.layer == layer and record.datatype == datatype)
        added = tuple(record for record in delta.added if record.layer == layer and record.datatype == datatype)
        lines.append(
            "| {layer} | {datatype} | {removed_count} | {added_count} | {removed_area} | {added_area} |".format(
                layer=layer,
                datatype=datatype,
                removed_count=len(removed),
                added_count=len(added),
                removed_area=format_twice_area(sum(record.twice_area for record in removed)),
                added_area=format_twice_area(sum(record.twice_area for record in added)),
            )
        )
    if not delta.layers:
        lines.append("| - | - | 0 | 0 | 0 | 0 |")
    lines.append("")
    return lines


def _render_components(
    components: tuple[ClassifiedComponent, ...],
    *,
    vertex_limit: int | None,
    detail_mode: str,
    sidecar_refs: Mapping[object, str] | None = None,
) -> list[str]:
    lines = ["## Polygon Components", ""]
    if not components:
        lines.extend(["No polygon-level differences were found.", ""])
        return lines
    if detail_mode == "compact":
        lines.extend(
            [
                "| index | direction | layer | datatype | bbox_dbu | area_dbu2 | vertex_count | sidecar_id | omitted_vertices | points_sha256 | vertices_preview_dbu |",
                "|---:|---|---:|---:|---|---:|---:|---|---:|---|---|",
            ]
        )
        for index, component in enumerate(components, start=1):
            points = component.points if vertex_limit is None else component.points[: max(vertex_limit, 0)]
            omitted = len(component.points) - len(points)
            lines.append(
                "| {index} | `{direction}` | {layer} | {datatype} | `{bbox}` | `{area}` | {vertex_count} | `{sidecar}` | {omitted} | `{sha}` | `{vertices}` |".format(
                    index=index,
                    direction=_coverage_direction_label(component.direction),
                    layer=component.layer.layer,
                    datatype=component.layer.datatype,
                    bbox=json.dumps(list(component.bbox), separators=(",", ":")),
                    area=component.area_dbu2_string,
                    vertex_count=len(component.points),
                    sidecar=_sidecar_ref(sidecar_refs, component),
                    omitted=omitted,
                    sha=_points_hash(component),
                    vertices=json.dumps([list(point) for point in points], separators=(",", ":")),
                )
            )
        lines.append("")
        return lines
    if detail_mode != "full":
        raise ValueError(f"unsupported component detail mode: {detail_mode!r}")
    for index, component in enumerate(components, start=1):
        points = component.points if vertex_limit is None else component.points[: max(vertex_limit, 0)]
        truncated_vertices = len(component.points) - len(points)
        lines.extend(
            [
                f"### Component {index}",
                "",
                f"- classification: `{component.classification}`",
                f"- direction: `{_coverage_direction_label(component.direction)}`",
                f"- layer: `{component.layer.layer}`",
                f"- datatype: `{component.layer.datatype}`",
                f"- bbox_dbu: `{list(component.bbox)}`",
                f"- area_dbu2: `{component.area_dbu2_string}`",
                f"- vertex_count: `{len(component.points)}`",
                f"- sidecar_id: `{_sidecar_ref(sidecar_refs, component)}`",
                f"- points_sha256: `{_points_hash(component)}`",
            ]
        )
        if truncated_vertices:
            lines.append(f"- omitted_vertex_count: `{truncated_vertices}`")
        lines.extend(
            [
                "",
                "```json",
                json.dumps(
                    {
                        "classification": component.classification,
                        "direction": _coverage_direction_label(component.direction),
                        "layer": component.layer.layer,
                        "datatype": component.layer.datatype,
                        "bbox_dbu": list(component.bbox),
                        "area_dbu2": component.area_dbu2_string,
                        "sidecar_id": _sidecar_ref(sidecar_refs, component),
                        "vertices_dbu": [list(point) for point in points],
                    },
                    indent=2,
                    sort_keys=True,
                ),
                "```",
                "",
            ]
        )
    return lines


def _components(policy_result: PolicyClassificationResult) -> tuple[ClassifiedComponent, ...]:
    return tuple(
        sorted(
            (component for layer in policy_result.layers for component in layer.components),
            key=_component_sort_key,
        )
    )


def _component_sort_key(component: ClassifiedComponent) -> tuple[object, ...]:
    return (
        component.layer.layer,
        component.layer.datatype,
        component.classification,
        component.direction.value,
        component.bbox,
        component.points,
    )


def _bbox_union(bboxes: list[tuple[int, int, int, int]] | tuple[tuple[int, int, int, int], ...]) -> tuple[int, int, int, int] | None:
    if not bboxes:
        return None
    return (
        min(bbox[0] for bbox in bboxes),
        min(bbox[1] for bbox in bboxes),
        max(bbox[2] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes),
    )


def _bbox_json(bbox: tuple[int, int, int, int] | None) -> str:
    if bbox is None:
        return "none"
    return json.dumps(list(bbox), separators=(",", ":"))


def _coverage_direction_components(
    components: tuple[ClassifiedComponent, ...],
    direction: DifferenceDirection,
) -> tuple[ClassifiedComponent, ...]:
    return tuple(component for component in components if component.direction == direction)


def _coverage_direction_count_text(counts: Counter[str]) -> str:
    if not counts:
        return "none"
    preferred_order = (
        _coverage_direction_label(DifferenceDirection.REMOVAL),
        _coverage_direction_label(DifferenceDirection.ADDITION),
    )
    parts = [f"{label}={counts[label]}" for label in preferred_order if counts[label]]
    parts.extend(f"{label}={counts[label]}" for label in sorted(set(counts) - set(preferred_order)))
    return ", ".join(parts)


def _points_hash(component: ClassifiedComponent) -> str:
    payload = json.dumps([list(point) for point in component.points], separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _polygon_inventory_index(
    buffer: PolygonBuffer, indices: Iterable[int],
) -> tuple[Counter[tuple[object, ...]], dict[tuple[object, ...], list[int]]]:
    counter: Counter[tuple[object, ...]] = Counter()
    indices_by_key: dict[tuple[object, ...], list[int]] = defaultdict(list)
    for index in indices:
        layer = buffer.layer_table.layers[int(buffer.layer_ids[index])]
        vertices = buffer.polygon_vertices(index)
        if len(vertices) == 0:
            continue
        key = _buffer_polygon_inventory_key(buffer, index, layer.layer, layer.datatype, vertices)
        counter[key] += 1
        indices_by_key[key].append(index)
    return counter, indices_by_key


def _buffer_polygon_inventory_key(
    buffer: PolygonBuffer,
    index: int,
    layer: int,
    datatype: int,
    vertices,
) -> tuple[object, ...]:
    if len(vertices) == 4 and _is_axis_aligned_rectangle_vertices(vertices):
        x0, y0, x1, y1 = (int(value) for value in buffer.bboxes[index])
        return layer, datatype, "rect", x0, y0, x1, y1
    points = tuple((int(x), int(y)) for x, y in vertices)
    return layer, datatype, "poly", _canonical_inventory_points(points)


def _polygon_inventory_record(
    buffer: PolygonBuffer,
    index: int,
    direction: DifferenceDirection,
) -> PolygonInventoryRecord:
    layer = buffer.layer_table.layers[int(buffer.layer_ids[index])]
    points = tuple((int(x), int(y)) for x, y in buffer.polygon_vertices(index))
    return PolygonInventoryRecord(
        layer=layer.layer,
        datatype=layer.datatype,
        direction=direction,
        points=points,
        bbox=polygon_bbox(points),
        twice_area=polygon_twice_area(points),
    )


def _polygon_inventory_records(buffer: PolygonBuffer, direction: DifferenceDirection) -> tuple[PolygonInventoryRecord, ...]:
    return tuple(
        _polygon_inventory_record(buffer, index, direction)
        for index in range(buffer.polygon_count)
        if len(buffer.polygon_vertices(index)) > 0
    )


def _inventory_key(record: PolygonInventoryRecord) -> tuple[object, ...]:
    if len(record.points) == 4 and _is_axis_aligned_rectangle_points(record.points):
        return record.layer, record.datatype, "rect", *record.bbox
    return record.layer, record.datatype, "poly", _canonical_inventory_points(record.points)


def _inventory_sort_key(record: PolygonInventoryRecord) -> tuple[object, ...]:
    return record.layer, record.datatype, record.direction.value, record.bbox, _canonical_inventory_points(record.points)


def _inventory_points_hash(record: PolygonInventoryRecord) -> str:
    payload = json.dumps([list(point) for point in record.points], separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_inventory_points(points: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    if not points:
        return points
    forward = _min_cyclic_rotation(points)
    reverse = _min_cyclic_rotation(tuple(reversed(points)))
    return min(forward, reverse)


def _min_cyclic_rotation(points: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    index = _least_cyclic_rotation_index(points)
    return points[index:] + points[:index]


def _least_cyclic_rotation_index(points: tuple[tuple[int, int], ...]) -> int:
    count = len(points)
    if count <= 1:
        return 0
    doubled = points + points
    first = 0
    second = 1
    offset = 0
    while first < count and second < count and offset < count:
        a = doubled[first + offset]
        b = doubled[second + offset]
        if a == b:
            offset += 1
            continue
        if a > b:
            first = first + offset + 1
            if first <= second:
                first = second + 1
        else:
            second = second + offset + 1
            if second <= first:
                second = first + 1
        offset = 0
    return min(first, second)


def _is_axis_aligned_rectangle_vertices(vertices) -> bool:
    return len({int(value) for value in vertices[:, 0]}) == 2 and len({int(value) for value in vertices[:, 1]}) == 2


def _is_axis_aligned_rectangle_points(points: tuple[tuple[int, int], ...]) -> bool:
    return len({x for x, _ in points}) == 2 and len({y for _, y in points}) == 2


def _coverage_twice_area(components: tuple[ClassifiedComponent, ...]) -> int:
    return sum(component.twice_area for component in components)


def _coverage_direction_twice_area(
    components: tuple[ClassifiedComponent, ...],
    direction: DifferenceDirection,
) -> int:
    return sum(component.twice_area for component in components if component.direction == direction)


def _neutral_evidence_status(
    *,
    coverage_twice_area: int,
    inventory_changed: bool,
    exact_verification_complete: bool,
) -> str:
    if not exact_verification_complete:
        return "incomplete-evidence"
    if coverage_twice_area or inventory_changed:
        return "differences-found"
    return "identical"


def _coverage_direction_label(direction: DifferenceDirection) -> str:
    if direction == DifferenceDirection.REMOVAL:
        return "coverage_before_only"
    if direction == DifferenceDirection.ADDITION:
        return "coverage_after_only"
    return f"coverage_{direction.value}"


def _inventory_direction_label(direction: DifferenceDirection) -> str:
    if direction == DifferenceDirection.REMOVAL:
        return "inventory_before_only"
    if direction == DifferenceDirection.ADDITION:
        return "inventory_after_only"
    return f"inventory_{direction.value}"


def _sidecar_ref(sidecar_refs: Mapping[object, str] | None, record: object) -> str:
    if sidecar_refs is None:
        return "not-written"
    return sidecar_refs.get(record, "not-written")


def _default_vertex_sidecar_path(path: Path) -> Path:
    stem = path.stem
    if stem == "exact_diff":
        return path.with_name("diff_vertices.jsonl")
    if stem.endswith("_exact_diff"):
        stem = stem[: -len("_exact_diff")]
    return path.with_name(f"{stem}_diff_vertices.jsonl")


def _write_vertex_sidecar(
    policy_result: PolicyClassificationResult,
    *,
    polygon_inventory_delta: PolygonInventoryDelta | None,
    path: Path,
) -> dict[object, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    refs: dict[object, str] = {}
    components = _components(policy_result)
    inventory_records = (
        tuple(sorted(polygon_inventory_delta.removed + polygon_inventory_delta.added, key=_inventory_sort_key))
        if polygon_inventory_delta is not None
        else ()
    )
    with path.open("w", encoding="utf-8") as handle:
        for index, component in enumerate(components, start=1):
            record_id = f"coverage-{index:06d}"
            refs[component] = record_id
            handle.write(
                json.dumps(
                    {
                        "id": record_id,
                        "kind": "coverage_component",
                        "direction": _coverage_direction_label(component.direction),
                        "layer": component.layer.layer,
                        "datatype": component.layer.datatype,
                        "bbox_dbu": list(component.bbox),
                        "area_dbu2": component.area_dbu2_string,
                        "twice_area_dbu2": component.twice_area,
                        "vertex_count": len(component.points),
                        "points_sha256": _points_hash(component),
                        "vertices_dbu": [list(point) for point in component.points],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
        for index, record in enumerate(inventory_records, start=1):
            record_id = f"inventory-{index:06d}"
            refs[record] = record_id
            handle.write(
                json.dumps(
                    {
                        "id": record_id,
                        "kind": "source_polygon_inventory",
                        "direction": _inventory_direction_label(record.direction),
                        "layer": record.layer,
                        "datatype": record.datatype,
                        "bbox_dbu": list(record.bbox),
                        "area_dbu2": record.area_dbu2_string,
                        "twice_area_dbu2": record.twice_area,
                        "vertex_count": len(record.points),
                        "points_sha256": _inventory_points_hash(record),
                        "vertices_dbu": [list(point) for point in record.points],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    return refs
