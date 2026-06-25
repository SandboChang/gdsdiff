from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .exact_oracle import format_twice_area
from .geometry import PolygonBuffer, polygon_bbox, polygon_twice_area
from .policy import ClassifiedComponent, PolicyClassificationResult
from .semantics import DifferenceDirection

DEFAULT_VERTEX_INLINE_LIMIT = 8
DEFAULT_INVENTORY_RECORD_LIMIT = 0
DEFAULT_COMPONENT_RECORD_LIMIT = 100
DEFAULT_COMPONENT_DETAIL_MODE = "compact"
DEFAULT_DIGEST_COMPONENT_LIMIT = 8


@dataclass(frozen=True)
class ExactDiffMarkdownWriteResult:
    path: Path
    component_count: int


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
    old_records = _polygon_inventory_records(old_buffer, DifferenceDirection.REMOVAL)
    new_records = _polygon_inventory_records(new_buffer, DifferenceDirection.ADDITION)
    old_counter = Counter(_inventory_key(record) for record in old_records)
    new_counter = Counter(_inventory_key(record) for record in new_records)
    old_by_key = defaultdict(list)
    new_by_key = defaultdict(list)
    for record in old_records:
        old_by_key[_inventory_key(record)].append(record)
    for record in new_records:
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
    generated_at: datetime | None = None,
) -> str:
    """Render a deterministic, LLM-friendly Markdown diff from classified exact components."""
    generated_at = generated_at or datetime.now(timezone.utc)
    components = _components(policy_result)
    shown_components = components if component_limit is None else components[: max(component_limit, 0)]
    lines = [
        f"# {title}",
        "",
        "## Comparator Evidence",
        "",
        f"- status: `{policy_result.status.value}`",
        f"- source_mode: `{source_mode}`",
        f"- exact_verification_complete: `{str(exact_verification_complete).lower()}`",
        f"- generated_at_utc: `{generated_at.astimezone(timezone.utc).isoformat()}`",
        f"- expected_area_dbu2: `{format_twice_area(policy_result.expected_twice_area)}`",
        f"- unexpected_area_dbu2: `{format_twice_area(policy_result.unexpected_twice_area)}`",
        f"- layer_count: `{len(policy_result.layers)}`",
        f"- component_count: `{len(components)}`",
        f"- polygon_inventory_delta_count: `{polygon_inventory_delta.changed_count if polygon_inventory_delta is not None else 'not-recorded'}`",
        "",
    ]
    if not exact_verification_complete:
        lines.extend(
            [
                "## Exactness Warning",
                "",
                "This file does not prove full-file polygon equivalence because the comparator run did not complete full exact verification. Use an `exact=True` run, or a GPU run explicitly verified by the exact oracle, for acceptance evidence.",
                "",
            ]
        )
    lines.extend(
        _render_agent_minimal_read(
            policy_result,
            components,
            polygon_inventory_delta=polygon_inventory_delta,
            digest_component_limit=DEFAULT_DIGEST_COMPONENT_LIMIT,
        )
    )
    lines.extend(_render_layer_summary(policy_result))
    lines.extend(_render_components(shown_components, vertex_limit=vertex_limit, detail_mode=component_detail_mode))
    if polygon_inventory_delta is not None:
        lines.extend(
            _render_polygon_inventory_delta(
                polygon_inventory_delta,
                policy_result=policy_result,
                inventory_limit=inventory_limit,
                vertex_limit=vertex_limit,
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
) -> list[str]:
    coverage_twice_area = policy_result.expected_twice_area + policy_result.unexpected_twice_area
    inventory_changed = polygon_inventory_delta is not None and polygon_inventory_delta.changed_count > 0
    count_by_kind = Counter((component.classification, component.direction.value) for component in components)
    largest = tuple(sorted(components, key=lambda component: (-abs(component.twice_area), _component_sort_key(component)))[:digest_component_limit])
    lines = [
        "## Agent Minimal Read",
        "",
        "- purpose: `default first section for LLM review; read this before expanding detail tables`",
        "- recommended_minimum_sections: `Comparator Evidence`, `Agent Minimal Read`, `Layer Summary`",
        "- expand_polygon_components_when: `need exact bbox/area/hash for every changed coverage component`",
        "- expand_inventory_delta_when: `exact coverage area is zero but polygon_inventory_delta_count is nonzero, or split/merge/refracture is suspected`",
        f"- coverage_changed: `{str(coverage_twice_area != 0).lower()}`",
        f"- coverage_xor_area_dbu2: `{format_twice_area(coverage_twice_area)}`",
        f"- inventory_changed: `{str(inventory_changed).lower()}`",
        f"- inventory_delta_count: `{polygon_inventory_delta.changed_count if polygon_inventory_delta is not None else 'not-recorded'}`",
        f"- component_count: `{len(components)}`",
    ]
    if count_by_kind:
        counts = ", ".join(
            f"{classification}/{direction}={count}"
            for (classification, direction), count in sorted(count_by_kind.items())
        )
    else:
        counts = "none"
    lines.append(f"- component_counts_by_class_direction: `{counts}`")
    lines.append("")
    lines.extend(_render_agent_layer_digest(policy_result))
    lines.extend(_render_agent_largest_component_digest(largest, total_count=len(components)))
    return lines


def _render_agent_layer_digest(policy_result: PolicyClassificationResult) -> list[str]:
    lines = [
        "### Affected Layer Digest",
        "",
        "| layer | datatype | exact_area_dbu2 | components | bbox_union_dbu |",
        "|---:|---:|---:|---:|---|",
    ]
    for layer in policy_result.layers:
        exact_twice_area = layer.expected_twice_area + layer.unexpected_twice_area
        bboxes = [component.bbox for component in layer.components]
        bbox_union = _bbox_union(bboxes)
        lines.append(
            "| {layer} | {datatype} | {area} | {count} | `{bbox}` |".format(
                layer=layer.layer.layer,
                datatype=layer.layer.datatype,
                area=format_twice_area(exact_twice_area),
                count=len(layer.components),
                bbox=list(bbox_union) if bbox_union is not None else "none",
            )
        )
    if not policy_result.layers:
        lines.append("| - | - | 0 | 0 | `none` |")
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
        "| rank | classification | direction | layer | datatype | bbox_dbu | area_dbu2 | vertex_count | points_sha256 |",
        "|---:|---|---|---:|---:|---|---:|---:|---|",
    ]
    for rank, component in enumerate(components, start=1):
        lines.append(
            "| {rank} | `{classification}` | `{direction}` | {layer} | {datatype} | `{bbox}` | `{area}` | {vertex_count} | `{sha}` |".format(
                rank=rank,
                classification=component.classification,
                direction=component.direction.value,
                layer=component.layer.layer,
                datatype=component.layer.datatype,
                bbox=json.dumps(list(component.bbox), separators=(",", ":")),
                area=component.area_dbu2_string,
                vertex_count=len(component.points),
                sha=_points_hash(component),
            )
        )
    if not components:
        lines.append("| - | - | - | - | - | `none` | 0 | 0 | `none` |")
    if total_count > len(components):
        lines.append(
            "| ... | `omitted_from_digest` | `see Polygon Components` | - | - | `-` | `-` | - | `-` |"
        )
    lines.append("")
    return lines


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
    generated_at: datetime | None = None,
) -> ExactDiffMarkdownWriteResult:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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
            generated_at=generated_at,
        ),
        encoding="utf-8",
    )
    return ExactDiffMarkdownWriteResult(path=path, component_count=len(_components(policy_result)))


def _render_layer_summary(policy_result: PolicyClassificationResult) -> list[str]:
    lines = ["## Layer Summary", ""]
    if not policy_result.layers:
        lines.append("- no differences")
        lines.append("")
        return lines
    lines.extend(
        [
            "| layer | datatype | expected_area_dbu2 | unexpected_area_dbu2 | components |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for layer in policy_result.layers:
        lines.append(
            "| {layer} | {datatype} | {expected} | {unexpected} | {count} |".format(
                layer=layer.layer.layer,
                datatype=layer.layer.datatype,
                expected=format_twice_area(layer.expected_twice_area),
                unexpected=format_twice_area(layer.unexpected_twice_area),
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
    if delta.changed_count and coverage_area == 0:
        lines.append("- interpretation: `polygon inventory changed but exact coverage XOR is zero; likely split/merge/refracture/no-area-change unless vertices indicate otherwise`")
    elif delta.changed_count:
        lines.append("- interpretation: `polygon inventory changed and exact coverage also changed; inspect both this section and Polygon Components`")
    else:
        lines.append("- interpretation: `no source polygon inventory changes after normalization`")
    lines.append("")
    lines.extend(_render_inventory_layer_summary(delta))
    if not records:
        lines.extend(["No input polygon inventory changes were found.", ""])
        return lines
    lines.extend(["### Polygon Record Changes", ""])
    for index, record in enumerate(shown_records, start=1):
        points = record.points if vertex_limit is None else record.points[: max(vertex_limit, 0)]
        truncated_vertices = len(record.points) - len(points)
        lines.extend(
            [
                f"#### Polygon Record {index}",
                "",
                f"- direction: `{record.direction.value}`",
                f"- layer: `{record.layer}`",
                f"- datatype: `{record.datatype}`",
                f"- bbox_dbu: `{list(record.bbox)}`",
                f"- area_dbu2: `{record.area_dbu2_string}`",
                f"- vertex_count: `{len(record.points)}`",
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
                        "direction": record.direction.value,
                        "layer": record.layer,
                        "datatype": record.datatype,
                        "bbox_dbu": list(record.bbox),
                        "area_dbu2": record.area_dbu2_string,
                        "vertices_dbu": [list(point) for point in points],
                    },
                    indent=2,
                    sort_keys=True,
                ),
                "```",
                "",
            ]
        )
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
) -> list[str]:
    lines = ["## Polygon Components", ""]
    if not components:
        lines.extend(["No polygon-level differences were found.", ""])
        return lines
    if detail_mode == "compact":
        lines.extend(
            [
                "| index | classification | direction | layer | datatype | bbox_dbu | area_dbu2 | vertex_count | omitted_vertices | points_sha256 | vertices_preview_dbu |",
                "|---:|---|---|---:|---:|---|---:|---:|---:|---|---|",
            ]
        )
        for index, component in enumerate(components, start=1):
            points = component.points if vertex_limit is None else component.points[: max(vertex_limit, 0)]
            omitted = len(component.points) - len(points)
            lines.append(
                "| {index} | `{classification}` | `{direction}` | {layer} | {datatype} | `{bbox}` | `{area}` | {vertex_count} | {omitted} | `{sha}` | `{vertices}` |".format(
                    index=index,
                    classification=component.classification,
                    direction=component.direction.value,
                    layer=component.layer.layer,
                    datatype=component.layer.datatype,
                    bbox=json.dumps(list(component.bbox), separators=(",", ":")),
                    area=component.area_dbu2_string,
                    vertex_count=len(component.points),
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
                f"- direction: `{component.direction.value}`",
                f"- layer: `{component.layer.layer}`",
                f"- datatype: `{component.layer.datatype}`",
                f"- bbox_dbu: `{list(component.bbox)}`",
                f"- area_dbu2: `{component.area_dbu2_string}`",
                f"- vertex_count: `{len(component.points)}`",
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
                        "direction": component.direction.value,
                        "layer": component.layer.layer,
                        "datatype": component.layer.datatype,
                        "bbox_dbu": list(component.bbox),
                        "area_dbu2": component.area_dbu2_string,
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


def _points_hash(component: ClassifiedComponent) -> str:
    payload = json.dumps([list(point) for point in component.points], separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _polygon_inventory_records(buffer: PolygonBuffer, direction: DifferenceDirection) -> tuple[PolygonInventoryRecord, ...]:
    records: list[PolygonInventoryRecord] = []
    for index in range(buffer.polygon_count):
        layer = buffer.layer_table.layers[int(buffer.layer_ids[index])]
        points = tuple((int(x), int(y)) for x, y in buffer.polygon_vertices(index))
        if not points:
            continue
        records.append(
            PolygonInventoryRecord(
                layer=layer.layer,
                datatype=layer.datatype,
                direction=direction,
                points=points,
                bbox=polygon_bbox(points),
                twice_area=polygon_twice_area(points),
            )
        )
    return tuple(records)


def _inventory_key(record: PolygonInventoryRecord) -> tuple[object, ...]:
    return record.layer, record.datatype, record.points


def _inventory_sort_key(record: PolygonInventoryRecord) -> tuple[object, ...]:
    return record.layer, record.datatype, record.direction.value, record.bbox, record.points


def _inventory_points_hash(record: PolygonInventoryRecord) -> str:
    payload = json.dumps([list(point) for point in record.points], separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
