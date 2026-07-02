from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Iterable

import gdstk
import numpy as np

from .config import LayerSpec, parse_layer_selector
from .exact_oracle import ExactOracleResult, format_twice_area, run_full_exact_oracle
from .extract import extract_gds_polygons
from .gds_metadata import decimal_from_number
from .geometry import (
    LayerTable,
    PolygonBuffer,
    normalize_polygon_points,
    polygon_bbox,
    polygon_flags,
    polygon_twice_area,
)
from .grid import snap_to_grid
from .hierarchy_extract import extract_hierarchy_window
from .spatial import BBox, bboxes_intersect


USER_BBOX = tuple[float, float, float, float]
TRANSFORM_TRANSLATE = "translate"
TRANSFORM_ROT90 = "rot90"


@dataclass(frozen=True)
class TemplateExtractionResult:
    output_gds: Path
    metadata_path: Path | None
    metadata: dict[str, object]


@dataclass(frozen=True)
class TemplateFindResult:
    output_dir: Path
    matches_json: Path
    matches_markdown: Path
    result: dict[str, object]


@dataclass(frozen=True)
class _GdsScale:
    unit: Decimal
    precision: Decimal

    @property
    def dbu_to_user(self) -> Decimal:
        return self.precision / self.unit

    @property
    def user_to_dbu(self) -> Decimal:
        return self.unit / self.precision


@dataclass(frozen=True)
class _PolygonRecord:
    index: int
    layer: LayerSpec
    points: tuple[tuple[int, int], ...]
    bbox: BBox
    twice_area: int


@dataclass(frozen=True)
class _TransformVariant:
    name: str
    records: tuple[_PolygonRecord, ...]
    bbox: BBox
    anchor: _PolygonRecord


@dataclass(frozen=True)
class _MatchCandidate:
    status: str
    transform: str
    origin_dbu: tuple[int, int]
    bbox_dbu: BBox
    polygon_count: int
    target_polygon_count: int
    layers: tuple[dict[str, object], ...]
    exact: ExactOracleResult = field(repr=False)


def extract_template(
    source_gds: str | Path,
    *,
    output_gds: str | Path,
    metadata_path: str | Path | None = None,
    top_name: str | None = None,
    layers: str | LayerSpec | Iterable[str | tuple[int, int] | LayerSpec] = "all",
    bbox_user: USER_BBOX,
    template_top: str = "TEMPLATE",
) -> TemplateExtractionResult:
    """Extract a bbox-selected template from a GDS into a standalone GDS."""

    source_path = Path(source_gds)
    output_path = Path(output_gds)
    metadata = Path(metadata_path) if metadata_path is not None else None

    lib = gdstk.read_gds(str(source_path))
    scale = _scale_from_library(lib)
    bbox_dbu = _user_bbox_to_dbu(bbox_user, scale)
    extracted = extract_hierarchy_window(
        lib,
        window_bbox=bbox_dbu,
        top_name=top_name,
        layers=layers,
        meters_per_dbu=scale.precision,
    )
    normalized = _translate_buffer(extracted.buffer, -bbox_dbu[0], -bbox_dbu[1])
    _write_buffer_gds(normalized, output_path, cell_name=template_top, scale=scale)

    payload = _template_metadata(
        source_path=source_path,
        output_path=output_path,
        source_top=extracted.top_name,
        template_top=template_top,
        layers=layers,
        source_bbox_user=bbox_user,
        source_bbox_dbu=bbox_dbu,
        normalized=normalized,
        scale=scale,
        extraction_stats=extracted.stats.to_json(),
    )
    if metadata is not None:
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return TemplateExtractionResult(output_gds=output_path, metadata_path=metadata, metadata=payload)


def find_template(
    template_gds: str | Path,
    target_gds: str | Path,
    *,
    output_dir: str | Path,
    template_top: str = "TEMPLATE",
    target_top: str | None = None,
    layers: str | LayerSpec | Iterable[str | tuple[int, int] | LayerSpec] = "all",
    transforms: Iterable[str] = (TRANSFORM_TRANSLATE,),
    allow_mirror: bool = False,
) -> TemplateFindResult:
    """Find exact normalized copies of a template inside a target GDS."""

    template_path = Path(template_gds)
    target_path = Path(target_gds)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    template_extraction = extract_gds_polygons(template_path, top_name=template_top, layers=layers)
    target_extraction = extract_gds_polygons(target_path, top_name=target_top, layers=layers)
    template_records = _records_from_buffer(template_extraction.buffer)
    target_records = _records_from_buffer(target_extraction.buffer)
    if not template_records:
        raise ValueError("template contains no selected polygons")

    matrices = _transform_matrices(tuple(transforms), allow_mirror=allow_mirror)
    target_by_anchor = _target_anchor_index(target_records)
    candidates: list[_MatchCandidate] = []
    seen: set[tuple[BBox, str]] = set()

    for transform_name, matrix in matrices:
        variant = _transform_template_records(template_records, transform_name, matrix)
        anchor_key = _anchor_key(variant.anchor)
        for target_anchor in target_by_anchor.get(anchor_key, ()):
            origin = (
                target_anchor.bbox[0] - variant.anchor.bbox[0],
                target_anchor.bbox[1] - variant.anchor.bbox[1],
            )
            candidate_bbox = _shift_bbox(variant.bbox, origin)
            seen_key = (candidate_bbox, transform_name)
            if seen_key in seen:
                continue
            seen.add(seen_key)
            placed_template = _shift_records(variant.records, origin)
            target_subset = tuple(record for record in target_records if bboxes_intersect(record.bbox, candidate_bbox))
            template_buffer = _buffer_from_records(placed_template)
            target_buffer = _buffer_from_records(target_subset)
            exact = run_full_exact_oracle(template_buffer, target_buffer, workers=1)
            candidates.append(
                _MatchCandidate(
                    status="identical" if exact.equivalent else "differences-found",
                    transform=transform_name,
                    origin_dbu=origin,
                    bbox_dbu=candidate_bbox,
                    polygon_count=len(placed_template),
                    target_polygon_count=len(target_subset),
                    layers=_candidate_layer_summary(template_buffer, target_buffer, exact),
                    exact=exact,
                )
            )

    candidates.sort(key=lambda item: (item.status != "identical", item.bbox_dbu, item.transform))
    result = _find_result_payload(
        template_path=template_path,
        target_path=target_path,
        template_top=template_extraction.top_name,
        target_top=target_extraction.top_name,
        layers=layers,
        transforms=tuple(name for name, _matrix in matrices),
        allow_mirror=allow_mirror,
        template_buffer=template_extraction.buffer,
        target_buffer=target_extraction.buffer,
        candidates=tuple(candidates),
    )
    matches_json = output / "matches.json"
    matches_markdown = output / "matches.md"
    matches_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    matches_markdown.write_text(_render_matches_markdown(result), encoding="utf-8")
    return TemplateFindResult(
        output_dir=output,
        matches_json=matches_json,
        matches_markdown=matches_markdown,
        result=result,
    )


def _scale_from_library(lib: gdstk.Library) -> _GdsScale:
    return _GdsScale(unit=decimal_from_number(lib.unit), precision=decimal_from_number(lib.precision))


def _user_bbox_to_dbu(bbox: USER_BBOX, scale: _GdsScale) -> BBox:
    if len(bbox) != 4:
        raise ValueError("bbox must contain x0,y0,x1,y1")
    x0, y0, x1, y1 = (decimal_from_number(value) for value in bbox)
    if x0 >= x1 or y0 >= y1:
        raise ValueError("bbox must satisfy x0 < x1 and y0 < y1")
    return (
        snap_to_grid(x0 * scale.unit, scale.precision),
        snap_to_grid(y0 * scale.unit, scale.precision),
        snap_to_grid(x1 * scale.unit, scale.precision),
        snap_to_grid(y1 * scale.unit, scale.precision),
    )


def _bbox_dbu_to_user(bbox: BBox, scale: _GdsScale) -> list[float]:
    factor = scale.dbu_to_user
    return [float(Decimal(value) * factor) for value in bbox]


def _points_to_user(points: tuple[tuple[int, int], ...], scale: _GdsScale) -> list[tuple[float, float]]:
    factor = scale.dbu_to_user
    return [(float(Decimal(x) * factor), float(Decimal(y) * factor)) for x, y in points]


def _translate_buffer(buffer: PolygonBuffer, dx: int, dy: int) -> PolygonBuffer:
    vertices = buffer.vertices_xy.copy()
    if len(vertices):
        vertices[:, 0] += dx
        vertices[:, 1] += dy
    bboxes = buffer.bboxes.copy()
    if len(bboxes):
        bboxes[:, 0] += dx
        bboxes[:, 2] += dx
        bboxes[:, 1] += dy
        bboxes[:, 3] += dy
    return PolygonBuffer(
        vertices_xy=vertices,
        polygon_offsets=buffer.polygon_offsets.copy(),
        layer_ids=buffer.layer_ids.copy(),
        bboxes=bboxes,
        flags=buffer.flags.copy(),
        source_ids=buffer.source_ids.copy(),
        layer_table=buffer.layer_table,
        polygon_hashes=buffer.polygon_hashes.copy() if buffer.polygon_hashes is not None else None,
    )


def _write_buffer_gds(buffer: PolygonBuffer, path: Path, *, cell_name: str, scale: _GdsScale) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lib = gdstk.Library(unit=float(scale.unit), precision=float(scale.precision))
    cell = lib.new_cell(cell_name)
    for record in _records_from_buffer(buffer):
        cell.add(
            gdstk.Polygon(
                _points_to_user(record.points, scale),
                layer=record.layer.layer,
                datatype=record.layer.datatype,
            )
        )
    lib.write_gds(str(path))


def _records_from_buffer(buffer: PolygonBuffer) -> tuple[_PolygonRecord, ...]:
    records = []
    for index in range(buffer.polygon_count):
        layer = buffer.layer_table.layers[int(buffer.layer_ids[index])]
        points = tuple((int(x), int(y)) for x, y in buffer.polygon_vertices(index))
        records.append(
            _PolygonRecord(
                index=index,
                layer=layer,
                points=points,
                bbox=tuple(int(value) for value in buffer.bboxes[index]),
                twice_area=polygon_twice_area(points),
            )
        )
    return tuple(records)


def _buffer_from_records(records: Iterable[_PolygonRecord]) -> PolygonBuffer:
    record_tuple = tuple(records)
    layers = tuple(sorted({record.layer for record in record_tuple}))
    layer_table = LayerTable(layers)
    layer_index = layer_table.index_by_layer
    vertices: list[tuple[int, int]] = []
    offsets = [0]
    layer_ids = []
    bboxes = []
    flags = []
    source_ids = []
    for source_id, record in enumerate(record_tuple):
        vertices.extend(record.points)
        offsets.append(len(vertices))
        layer_ids.append(layer_index[record.layer])
        bboxes.append(record.bbox)
        flags.append(polygon_flags(record.points))
        source_ids.append(source_id)
    return PolygonBuffer(
        vertices_xy=np.asarray(vertices, dtype=np.int64).reshape((-1, 2)),
        polygon_offsets=np.asarray(offsets, dtype=np.uint64),
        layer_ids=np.asarray(layer_ids, dtype=np.uint32),
        bboxes=np.asarray(bboxes, dtype=np.int64).reshape((len(record_tuple), 4)),
        flags=np.asarray(flags, dtype=np.uint32),
        source_ids=np.asarray(source_ids, dtype=np.uint64),
        layer_table=layer_table,
    )


def _template_metadata(
    *,
    source_path: Path,
    output_path: Path,
    source_top: str,
    template_top: str,
    layers,
    source_bbox_user: USER_BBOX,
    source_bbox_dbu: BBox,
    normalized: PolygonBuffer,
    scale: _GdsScale,
    extraction_stats: dict[str, object],
) -> dict[str, object]:
    bbox = _buffer_bbox(normalized)
    return {
        "mode": "gdsdiff-template-extract-v1",
        "source": {
            "path": str(source_path),
            "top": source_top,
            "bbox_user": list(source_bbox_user),
            "bbox_dbu": list(source_bbox_dbu),
            "layers": _layers_json(layers),
            "selection_mode": "intersects",
        },
        "template": {
            "path": str(output_path),
            "top": template_top,
            "normalization_origin_dbu": [source_bbox_dbu[0], source_bbox_dbu[1]],
            "bbox_dbu": list(bbox) if bbox is not None else None,
            "bbox_user": _bbox_dbu_to_user(bbox, scale) if bbox is not None else None,
            "polygon_count": normalized.polygon_count,
            "vertex_count": normalized.vertex_count,
            "layer_summary": _buffer_layer_summary(normalized, scale=scale),
            "signature_sha256": _buffer_signature(normalized),
        },
        "gds": {"unit": str(scale.unit), "precision": str(scale.precision)},
        "extraction": extraction_stats,
    }


def _layers_json(layers) -> object:
    parsed = parse_layer_selector(layers)
    if parsed == "all":
        return "all"
    return [layer.to_json() for layer in parsed]


def _buffer_bbox(buffer: PolygonBuffer) -> BBox | None:
    if buffer.polygon_count == 0:
        return None
    return (
        int(buffer.bboxes[:, 0].min()),
        int(buffer.bboxes[:, 1].min()),
        int(buffer.bboxes[:, 2].max()),
        int(buffer.bboxes[:, 3].max()),
    )


def _buffer_layer_summary(buffer: PolygonBuffer, *, scale: _GdsScale | None = None) -> list[dict[str, object]]:
    summary: dict[LayerSpec, dict[str, object]] = {}
    for record in _records_from_buffer(buffer):
        item = summary.setdefault(
            record.layer,
            {
                "layer": record.layer.layer,
                "datatype": record.layer.datatype,
                "polygon_count": 0,
                "twice_area_dbu2": 0,
                "bbox_dbu": None,
            },
        )
        item["polygon_count"] = int(item["polygon_count"]) + 1
        item["twice_area_dbu2"] = int(item["twice_area_dbu2"]) + record.twice_area
        item["bbox_dbu"] = list(_bbox_union((tuple(item["bbox_dbu"]) if item["bbox_dbu"] is not None else None, record.bbox)))
    out = []
    for layer in sorted(summary):
        item = dict(summary[layer])
        item["area_dbu2"] = format_twice_area(int(item.pop("twice_area_dbu2")))
        if scale is not None:
            factor = scale.dbu_to_user * scale.dbu_to_user
            item["area_user2"] = float((Decimal(int(summary[layer]["twice_area_dbu2"])) / Decimal(2)) * factor)
        out.append(item)
    return out


def _bbox_union(bboxes: Iterable[BBox | None]) -> BBox:
    clean = tuple(bbox for bbox in bboxes if bbox is not None)
    if not clean:
        raise ValueError("bbox union requires at least one bbox")
    return (
        min(bbox[0] for bbox in clean),
        min(bbox[1] for bbox in clean),
        max(bbox[2] for bbox in clean),
        max(bbox[3] for bbox in clean),
    )


def _buffer_signature(buffer: PolygonBuffer) -> str:
    records = []
    for record in _records_from_buffer(buffer):
        records.append((record.layer.layer, record.layer.datatype, record.points))
    payload = repr(tuple(sorted(records))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _transform_matrices(transforms: tuple[str, ...], *, allow_mirror: bool) -> tuple[tuple[str, tuple[int, int, int, int]], ...]:
    normalized = {item.strip().lower() for item in transforms}
    if not normalized or TRANSFORM_TRANSLATE in normalized:
        matrices = [("translate", (1, 0, 0, 1))]
    else:
        matrices = []
    if TRANSFORM_ROT90 in normalized:
        matrices.extend(
            [
                ("rot90", (0, -1, 1, 0)),
                ("rot180", (-1, 0, 0, -1)),
                ("rot270", (0, 1, -1, 0)),
            ]
        )
    unknown = normalized - {TRANSFORM_TRANSLATE, TRANSFORM_ROT90}
    if unknown:
        raise ValueError(f"unknown template transform(s): {', '.join(sorted(unknown))}")
    if allow_mirror:
        base = tuple(matrices)
        matrices.extend((f"{name}+mirror-x", (a, -b, c, -d)) for name, (a, b, c, d) in base)
    dedup: dict[tuple[int, int, int, int], str] = {}
    for name, matrix in matrices:
        dedup.setdefault(matrix, name)
    return tuple((name, matrix) for matrix, name in dedup.items())


def _transform_template_records(
    records: tuple[_PolygonRecord, ...],
    name: str,
    matrix: tuple[int, int, int, int],
) -> _TransformVariant:
    transformed_records = []
    a, b, c, d = matrix
    for record in records:
        points = normalize_polygon_points([(a * x + b * y, c * x + d * y) for x, y in record.points])
        transformed_records.append(
            _PolygonRecord(
                index=record.index,
                layer=record.layer,
                points=points,
                bbox=polygon_bbox(points),
                twice_area=polygon_twice_area(points),
            )
        )
    bbox = _bbox_union(tuple(record.bbox for record in transformed_records))
    normalized = tuple(
        _PolygonRecord(
            index=record.index,
            layer=record.layer,
            points=tuple((x - bbox[0], y - bbox[1]) for x, y in record.points),
            bbox=(record.bbox[0] - bbox[0], record.bbox[1] - bbox[1], record.bbox[2] - bbox[0], record.bbox[3] - bbox[1]),
            twice_area=record.twice_area,
        )
        for record in transformed_records
    )
    variant_bbox = _bbox_union(tuple(record.bbox for record in normalized))
    anchor = _select_anchor(normalized)
    return _TransformVariant(name=name, records=normalized, bbox=variant_bbox, anchor=anchor)


def _select_anchor(records: tuple[_PolygonRecord, ...]) -> _PolygonRecord:
    return min(records, key=lambda record: (-record.twice_area, record.layer, record.bbox, record.points))


def _target_anchor_index(records: tuple[_PolygonRecord, ...]) -> dict[tuple[object, ...], list[_PolygonRecord]]:
    out: dict[tuple[object, ...], list[_PolygonRecord]] = {}
    for record in records:
        out.setdefault(_anchor_key(record), []).append(record)
    return out


def _anchor_key(record: _PolygonRecord) -> tuple[object, ...]:
    x0, y0, x1, y1 = record.bbox
    normalized = tuple((x - x0, y - y0) for x, y in record.points)
    return (
        record.layer,
        x1 - x0,
        y1 - y0,
        record.twice_area,
        len(record.points),
        _canonical_points(normalized),
    )


def _canonical_points(points: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    if not points:
        return points
    forward = _min_cyclic_rotation(points)
    reverse = _min_cyclic_rotation(tuple(reversed(points)))
    return min(forward, reverse)


def _min_cyclic_rotation(points: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return min(points[index:] + points[:index] for index in range(len(points)))


def _shift_records(records: tuple[_PolygonRecord, ...], origin: tuple[int, int]) -> tuple[_PolygonRecord, ...]:
    dx, dy = origin
    return tuple(
        _PolygonRecord(
            index=record.index,
            layer=record.layer,
            points=tuple((x + dx, y + dy) for x, y in record.points),
            bbox=_shift_bbox(record.bbox, origin),
            twice_area=record.twice_area,
        )
        for record in records
    )


def _shift_bbox(bbox: BBox, origin: tuple[int, int]) -> BBox:
    dx, dy = origin
    return bbox[0] + dx, bbox[1] + dy, bbox[2] + dx, bbox[3] + dy


def _candidate_layer_summary(
    template_buffer: PolygonBuffer,
    target_buffer: PolygonBuffer,
    exact: ExactOracleResult,
) -> tuple[dict[str, object], ...]:
    template = _raw_layer_metrics(template_buffer)
    target = _raw_layer_metrics(target_buffer)
    exact_by_layer = {layer.layer: layer for layer in exact.layers}
    rows = []
    for layer in sorted(set(template) | set(target) | set(exact_by_layer)):
        template_metrics = template.get(layer, {"polygon_count": 0, "twice_area": 0})
        target_metrics = target.get(layer, {"polygon_count": 0, "twice_area": 0})
        diff = exact_by_layer.get(layer)
        rows.append(
            {
                "layer": layer.layer,
                "datatype": layer.datatype,
                "template_polygon_count": template_metrics["polygon_count"],
                "target_polygon_count": target_metrics["polygon_count"],
                "template_area_dbu2": format_twice_area(int(template_metrics["twice_area"])),
                "target_area_dbu2": format_twice_area(int(target_metrics["twice_area"])),
                "template_only_area_dbu2": format_twice_area(diff.old_minus_new_twice_area if diff is not None else 0),
                "target_only_area_dbu2": format_twice_area(diff.new_minus_old_twice_area if diff is not None else 0),
                "xor_area_dbu2": format_twice_area(diff.xor_twice_area if diff is not None else 0),
                "template_only_count": len(diff.old_minus_new) if diff is not None else 0,
                "target_only_count": len(diff.new_minus_old) if diff is not None else 0,
            }
        )
    return tuple(rows)


def _raw_layer_metrics(buffer: PolygonBuffer) -> dict[LayerSpec, dict[str, int]]:
    metrics: dict[LayerSpec, dict[str, int]] = {}
    for record in _records_from_buffer(buffer):
        item = metrics.setdefault(record.layer, {"polygon_count": 0, "twice_area": 0})
        item["polygon_count"] += 1
        item["twice_area"] += record.twice_area
    return metrics


def _find_result_payload(
    *,
    template_path: Path,
    target_path: Path,
    template_top: str,
    target_top: str,
    layers,
    transforms: tuple[str, ...],
    allow_mirror: bool,
    template_buffer: PolygonBuffer,
    target_buffer: PolygonBuffer,
    candidates: tuple[_MatchCandidate, ...],
) -> dict[str, object]:
    return {
        "mode": "gdsdiff-template-find-v1",
        "template": {
            "path": str(template_path),
            "top": template_top,
            "polygon_count": template_buffer.polygon_count,
            "signature_sha256": _buffer_signature(template_buffer),
        },
        "target": {
            "path": str(target_path),
            "top": target_top,
            "polygon_count": target_buffer.polygon_count,
        },
        "selection": {
            "layers": _layers_json(layers),
            "transforms": list(transforms),
            "allow_mirror": allow_mirror,
        },
        "summary": {
            "candidate_count": len(candidates),
            "match_count": sum(1 for candidate in candidates if candidate.status == "identical"),
            "difference_count": sum(1 for candidate in candidates if candidate.status != "identical"),
        },
        "candidates": [
            {
                "status": candidate.status,
                "transform": candidate.transform,
                "origin_dbu": list(candidate.origin_dbu),
                "bbox_dbu": list(candidate.bbox_dbu),
                "template_polygon_count": candidate.polygon_count,
                "target_polygon_count": candidate.target_polygon_count,
                "layers": list(candidate.layers),
            }
            for candidate in candidates
        ],
    }


def _render_matches_markdown(result: dict[str, object]) -> str:
    summary = result["summary"]
    lines = [
        "# GDS Template Matches",
        "",
        f"- mode: `{result['mode']}`",
        f"- template: `{result['template']['path']}` top `{result['template']['top']}`",
        f"- target: `{result['target']['path']}` top `{result['target']['top']}`",
        f"- candidate_count: `{summary['candidate_count']}`",
        f"- match_count: `{summary['match_count']}`",
        f"- difference_count: `{summary['difference_count']}`",
        "",
        "| status | transform | bbox_dbu | template_polygons | target_polygons |",
        "|---|---|---|---:|---:|",
    ]
    for candidate in result["candidates"]:
        lines.append(
            "| {status} | {transform} | `{bbox}` | {template_count} | {target_count} |".format(
                status=candidate["status"],
                transform=candidate["transform"],
                bbox=candidate["bbox_dbu"],
                template_count=candidate["template_polygon_count"],
                target_count=candidate["target_polygon_count"],
            )
        )
    if result["candidates"]:
        lines.extend(
            [
                "",
                "## Per-Layer Candidate Details",
                "",
            ]
        )
    for index, candidate in enumerate(result["candidates"], start=1):
        lines.extend(
            [
                f"### Candidate {index}",
                "",
                f"- status: `{candidate['status']}`",
                f"- transform: `{candidate['transform']}`",
                f"- bbox_dbu: `{candidate['bbox_dbu']}`",
                "",
                "| layer | template_count | target_count | template_area_dbu2 | target_area_dbu2 | xor_area_dbu2 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for layer in candidate["layers"]:
            lines.append(
                "| {layer}:{datatype} | {template_count} | {target_count} | {template_area} | {target_area} | {xor_area} |".format(
                    layer=layer["layer"],
                    datatype=layer["datatype"],
                    template_count=layer["template_polygon_count"],
                    target_count=layer["target_polygon_count"],
                    template_area=layer["template_area_dbu2"],
                    target_area=layer["target_area_dbu2"],
                    xor_area=layer["xor_area_dbu2"],
                )
            )
        lines.append("")
    lines.append("")
    return "\n".join(lines)
