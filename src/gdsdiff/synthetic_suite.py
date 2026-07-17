from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import gdstk

from .api import compare_gds_files
from .config import CompareConfig, LayerSpec
from .exact_oracle import run_full_exact_oracle
from .extract import extract_gds_polygons
from .native import discover_native_capabilities

LayerTuple = tuple[int, int]
BBox = tuple[float, float, float, float]

_UNIT = 1e-6
_PRECISION = 1e-9
_DBU_PER_UM = int(round(_UNIT / _PRECISION))


@dataclass(frozen=True)
class ScaleProfile:
    name: str
    sparse_grid_n: int
    mixed_count: int
    hollow_grid_n: int
    hierarchy_cols: int
    hierarchy_rows: int
    reference_cols: int
    reference_rows: int


SCALE_PROFILES: dict[str, ScaleProfile] = {
    "smoke": ScaleProfile(
        name="smoke",
        sparse_grid_n=6,
        mixed_count=24,
        hollow_grid_n=4,
        hierarchy_cols=4,
        hierarchy_rows=3,
        reference_cols=5,
        reference_rows=4,
    ),
    "benchmark": ScaleProfile(
        name="benchmark",
        sparse_grid_n=110,
        mixed_count=900,
        hollow_grid_n=28,
        hierarchy_cols=60,
        hierarchy_rows=24,
        reference_cols=80,
        reference_rows=80,
    ),
    "stress": ScaleProfile(
        name="stress",
        sparse_grid_n=300,
        mixed_count=5000,
        hollow_grid_n=80,
        hierarchy_cols=180,
        hierarchy_rows=80,
        reference_cols=250,
        reference_rows=200,
    ),
}


@dataclass(frozen=True)
class SyntheticEdit:
    operation_id: str
    shape_id: str
    operation: str
    expected_direction: str
    before_layer: LayerTuple | None = None
    after_layer: LayerTuple | None = None
    before_bbox_um: BBox | None = None
    after_bbox_um: BBox | None = None
    notes: str = ""

    @property
    def coverage_changed(self) -> bool:
        return self.expected_direction != "inventory_only"

    def changed_layers(self) -> tuple[LayerTuple, ...]:
        if not self.coverage_changed:
            return ()
        layers: list[LayerTuple] = []
        if self.before_layer is not None:
            layers.append(self.before_layer)
        if self.after_layer is not None and self.after_layer not in layers:
            layers.append(self.after_layer)
        return tuple(layers)

    def to_json(self) -> dict[str, object]:
        return {
            "operation_id": self.operation_id,
            "shape_id": self.shape_id,
            "operation": self.operation,
            "expected_direction": self.expected_direction,
            "before_layer": list(self.before_layer) if self.before_layer is not None else None,
            "after_layer": list(self.after_layer) if self.after_layer is not None else None,
            "before_bbox_um": list(self.before_bbox_um) if self.before_bbox_um is not None else None,
            "after_bbox_um": list(self.after_bbox_um) if self.after_bbox_um is not None else None,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class SyntheticCaseBuild:
    before: gdstk.Library
    after: gdstk.Library
    edits: tuple[SyntheticEdit, ...]
    equivalent_expected: bool = False
    notes: str = ""


@dataclass(frozen=True)
class SyntheticCaseDefinition:
    name: str
    title: str
    builder: Callable[[ScaleProfile], SyntheticCaseBuild]


@dataclass(frozen=True)
class SyntheticCaseRecord:
    name: str
    title: str
    scale: str
    before_path: Path
    after_path: Path
    manifest_path: Path
    changes_path: Path
    before_polygon_count: int
    after_polygon_count: int
    before_vertex_count: int
    after_vertex_count: int
    expected_changed_layers: tuple[LayerTuple, ...]
    equivalent_expected: bool
    edits: tuple[SyntheticEdit, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "title": self.title,
            "scale": self.scale,
            "paths": {
                "before_gds": str(self.before_path),
                "after_gds": str(self.after_path),
                "manifest": str(self.manifest_path),
                "changes": str(self.changes_path),
            },
            "counts": {
                "before_polygons": self.before_polygon_count,
                "after_polygons": self.after_polygon_count,
                "before_vertices": self.before_vertex_count,
                "after_vertices": self.after_vertex_count,
            },
            "expected_changed_layers": [list(layer) for layer in self.expected_changed_layers],
            "equivalent_expected": self.equivalent_expected,
            "edits": [edit.to_json() for edit in self.edits],
        }


@dataclass(frozen=True)
class SyntheticBenchmarkRun:
    case_name: str
    scale: str
    mode: str
    status: str
    exit_code: int
    wall_s: float
    total_s: float | None
    changed_layers: tuple[LayerTuple, ...]
    audit_match: bool | None
    json_report_path: Path
    exact_diff_markdown_path: Path
    stdout_path: Path
    stderr_path: Path
    command_path: Path
    error: str | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "case_name": self.case_name,
            "scale": self.scale,
            "mode": self.mode,
            "status": self.status,
            "exit_code": self.exit_code,
            "wall_s": self.wall_s,
            "total_s": self.total_s,
            "changed_layers": [list(layer) for layer in self.changed_layers],
            "audit_match": self.audit_match,
            "artifacts": {
                "json_report": str(self.json_report_path),
                "exact_diff_markdown": str(self.exact_diff_markdown_path),
                "stdout": str(self.stdout_path),
                "stderr": str(self.stderr_path),
                "command": str(self.command_path),
            },
            "error": self.error,
        }


def build_synthetic_suite(
    output_dir: str | Path,
    *,
    scale: str = "benchmark",
    case_names: Iterable[str] | None = None,
) -> tuple[SyntheticCaseRecord, ...]:
    profile = _profile(scale)
    selected = set(case_names) if case_names is not None else None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[SyntheticCaseRecord] = []
    for definition in SYNTHETIC_CASES:
        if selected is not None and definition.name not in selected:
            continue
        records.append(_write_case(output_dir, definition, profile))
    suite_manifest = {
        "suite": "gdsdiff-synthetic-readiness-corpus",
        "scale": profile.name,
        "case_count": len(records),
        "cases": [record.to_json() for record in records],
    }
    (output_dir / "suite_manifest.json").write_text(_json_dumps(suite_manifest), encoding="utf-8")
    return tuple(records)


def benchmark_synthetic_records(
    records: Sequence[SyntheticCaseRecord],
    *,
    run_cpu: bool = True,
    run_metal: bool = True,
    run_metal_audit: bool = True,
    stress_cpu_case_limit: int = 2,
) -> tuple[SyntheticBenchmarkRun, ...]:
    rows: list[SyntheticBenchmarkRun] = []
    stress_cpu_count = 0
    for record in records:
        allow_cpu = run_cpu
        allow_metal_audit = run_metal_audit
        if record.scale == "stress":
            allow_cpu = run_cpu and stress_cpu_count < stress_cpu_case_limit
            allow_metal_audit = run_metal_audit and stress_cpu_count < stress_cpu_case_limit
            if allow_cpu:
                stress_cpu_count += 1
        if allow_cpu:
            rows.append(_benchmark_one(record, mode="cpu"))
        if run_metal:
            rows.append(_benchmark_one(record, mode="metal"))
        if run_metal and allow_metal_audit:
            rows.append(_benchmark_one(record, mode="metal_audit"))
    return tuple(rows)


def write_benchmark_summary(
    output_dir: str | Path,
    records: Sequence[SyntheticCaseRecord],
    runs: Sequence[SyntheticBenchmarkRun],
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_json_path = output_dir / "benchmark_runs.json"
    run_json_path.write_text(_json_dumps([run.to_json() for run in runs]), encoding="utf-8")
    summary_path = output_dir / "benchmark_summary.md"
    summary_path.write_text(_benchmark_markdown(records, runs, run_json_path), encoding="utf-8")
    return summary_path


def _write_case(output_dir: Path, definition: SyntheticCaseDefinition, profile: ScaleProfile) -> SyntheticCaseRecord:
    case_dir = output_dir / definition.name
    case_dir.mkdir(parents=True, exist_ok=True)
    built = definition.builder(profile)
    before_path = case_dir / "before.gds"
    after_path = case_dir / "after.gds"
    built.before.write_gds(before_path)
    built.after.write_gds(after_path)
    before_stats = _library_stats(built.before)
    after_stats = _library_stats(built.after)
    expected_changed_layers = _expected_changed_layers(built.edits)
    manifest_path = case_dir / "manifest.json"
    changes_path = case_dir / "changes.md"
    record = SyntheticCaseRecord(
        name=definition.name,
        title=definition.title,
        scale=profile.name,
        before_path=before_path,
        after_path=after_path,
        manifest_path=manifest_path,
        changes_path=changes_path,
        before_polygon_count=before_stats["polygon_count"],
        after_polygon_count=after_stats["polygon_count"],
        before_vertex_count=before_stats["vertex_count"],
        after_vertex_count=after_stats["vertex_count"],
        expected_changed_layers=expected_changed_layers,
        equivalent_expected=built.equivalent_expected,
        edits=built.edits,
    )
    manifest = record.to_json() | {
        "unit_m": _UNIT,
        "precision_m": _PRECISION,
        "dbu_per_um": _DBU_PER_UM,
        "notes": built.notes,
    }
    manifest_path.write_text(_json_dumps(manifest), encoding="utf-8")
    changes_path.write_text(_changes_markdown(record, built.notes), encoding="utf-8")
    return record


def _benchmark_one(record: SyntheticCaseRecord, *, mode: str) -> SyntheticBenchmarkRun:
    case_dir = record.before_path.parent
    artifact_dir = case_dir / "gdsdiff" / mode
    artifact_dir.mkdir(parents=True, exist_ok=True)
    json_report_path = artifact_dir / "report.json"
    exact_diff_markdown_path = artifact_dir / "exact_diff.md"
    stdout_path = artifact_dir / "stdout.txt"
    stderr_path = artifact_dir / "stderr.txt"
    command_path = artifact_dir / "command.json"
    if mode == "cpu":
        config = CompareConfig(
            record.before_path,
            record.after_path,
            backend="cpu",
            evidence_engine="gdstk-oracle",
        )
        audit = False
    elif mode == "metal":
        config = CompareConfig(
            record.before_path,
            record.after_path,
            backend="metal",
            evidence_engine="gpu-polygon",
        )
        audit = False
    elif mode == "metal_audit":
        config = CompareConfig(
            record.before_path,
            record.after_path,
            backend="metal",
            evidence_engine="gpu-polygon",
        )
        audit = True
    else:
        raise ValueError(f"unknown benchmark mode: {mode!r}")

    command_payload = {
        "mode": mode,
        "config": config.to_json_dict(),
        "audit_cpu_oracle": audit,
        "json_report_path": str(json_report_path),
        "exact_diff_markdown_path": str(exact_diff_markdown_path),
        "synthetic_case": record.name,
        "synthetic_scale": record.scale,
    }
    start = time.perf_counter()
    try:
        result = compare_gds_files(
            config,
            json_report_path=json_report_path,
            exact_diff_markdown_path=exact_diff_markdown_path,
            audit_cpu_oracle=audit,
            command=("gdsdiff.synthetic_suite", "benchmark", record.scale, record.name, mode),
        )
        wall_s = time.perf_counter() - start
        payload = result.report.to_json_dict()
        stdout_path.write_text(
            "\n".join(
                [
                    f"status={payload.get('status')}",
                    f"evidence_status={payload.get('evidence_status')}",
                    f"differences_found={payload.get('differences_found')}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        stderr_path.write_text("\n".join(str(item) for item in payload.get("warnings", ())) + "\n", encoding="utf-8")
        row = SyntheticBenchmarkRun(
            case_name=record.name,
            scale=record.scale,
            mode=mode,
            status=str(payload.get("status")),
            exit_code=result.exit_code,
            wall_s=wall_s,
            total_s=_float_or_none(payload.get("timings", {}).get("total_s") if isinstance(payload.get("timings"), dict) else None),
            changed_layers=_changed_layers_from_report(payload),
            audit_match=_audit_match_from_report(payload),
            json_report_path=json_report_path,
            exact_diff_markdown_path=exact_diff_markdown_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            command_path=command_path,
        )
    except Exception as exc:
        wall_s = time.perf_counter() - start
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        row = SyntheticBenchmarkRun(
            case_name=record.name,
            scale=record.scale,
            mode=mode,
            status="error",
            exit_code=2,
            wall_s=wall_s,
            total_s=None,
            changed_layers=(),
            audit_match=None,
            json_report_path=json_report_path,
            exact_diff_markdown_path=exact_diff_markdown_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            command_path=command_path,
            error=f"{type(exc).__name__}: {exc}",
        )
    command_path.write_text(_json_dumps(command_payload | {"result": row.to_json()}), encoding="utf-8")
    return row


def _benchmark_markdown(
    records: Sequence[SyntheticCaseRecord],
    runs: Sequence[SyntheticBenchmarkRun],
    run_json_path: Path,
) -> str:
    by_case_mode = {(run.scale, run.case_name, run.mode): run for run in runs}
    lines = [
        "# GDSDiff Synthetic Readiness Benchmark",
        "",
        f"- Host capabilities: `{json.dumps(discover_native_capabilities().to_json(), sort_keys=True)}`",
        f"- Run JSON: `{run_json_path}`",
        "",
        "| Scale | Case | Before polys | After polys | Changed layers | CPU wall s | Metal wall s | Speedup | Metal audit | Artifacts |",
        "|---|---:|---:|---:|---|---:|---:|---:|---|---|",
    ]
    for record in records:
        cpu = by_case_mode.get((record.scale, record.name, "cpu"))
        metal = by_case_mode.get((record.scale, record.name, "metal"))
        metal_audit = by_case_mode.get((record.scale, record.name, "metal_audit"))
        cpu_wall = cpu.wall_s if cpu is not None and cpu.error is None else None
        metal_wall = metal.wall_s if metal is not None and metal.error is None else None
        speedup = (cpu_wall / metal_wall) if cpu_wall and metal_wall and metal_wall > 0 else None
        layers = _format_layers(record.expected_changed_layers)
        audit = (
            "match"
            if metal_audit is not None and metal_audit.audit_match is True
            else "mismatch"
            if metal_audit is not None and metal_audit.audit_match is False
            else "not-run"
            if metal_audit is None
            else metal_audit.status
        )
        artifact_bits = []
        if cpu is not None:
            artifact_bits.append(f"cpu `{cpu.json_report_path}`")
        if metal is not None:
            artifact_bits.append(f"metal `{metal.json_report_path}`")
        if metal_audit is not None:
            artifact_bits.append(f"audit `{metal_audit.json_report_path}`")
        lines.append(
            "| {scale} | `{case}` | {before} | {after} | {layers} | {cpu_wall} | {metal_wall} | {speedup} | {audit} | {artifacts} |".format(
                scale=record.scale,
                case=record.name,
                before=record.before_polygon_count,
                after=record.after_polygon_count,
                layers=layers or "none",
                cpu_wall=_format_seconds(cpu_wall),
                metal_wall=_format_seconds(metal_wall),
                speedup=f"{speedup:.2f}x" if speedup is not None else "n/a",
                audit=audit,
                artifacts="<br>".join(artifact_bits) if artifact_bits else "none",
            )
        )
    lines.append("")
    return "\n".join(lines)


def _changes_markdown(record: SyntheticCaseRecord, notes: str) -> str:
    lines = [
        f"# {record.title}",
        "",
        f"- Case: `{record.name}`",
        f"- Scale: `{record.scale}`",
        f"- Before GDS: `{record.before_path}`",
        f"- After GDS: `{record.after_path}`",
        f"- Expected equivalent coverage: `{str(record.equivalent_expected).lower()}`",
        f"- Expected changed layers: `{_format_layers(record.expected_changed_layers) or 'none'}`",
        f"- Before polygons/vertices: `{record.before_polygon_count}` / `{record.before_vertex_count}`",
        f"- After polygons/vertices: `{record.after_polygon_count}` / `{record.after_vertex_count}`",
    ]
    if notes:
        lines.extend(["", "## Notes", "", notes])
    lines.extend(["", "## Intended Changes", ""])
    for edit in record.edits:
        lines.extend(
            [
                f"### {edit.operation_id}",
                "",
                f"- shape_id: `{edit.shape_id}`",
                f"- operation: `{edit.operation}`",
                f"- expected_direction: `{edit.expected_direction}`",
                f"- before_layer: `{_format_layer(edit.before_layer) if edit.before_layer else 'none'}`",
                f"- after_layer: `{_format_layer(edit.after_layer) if edit.after_layer else 'none'}`",
                f"- before_bbox_um: `{edit.before_bbox_um}`",
                f"- after_bbox_um: `{edit.after_bbox_um}`",
                f"- notes: {edit.notes}",
                "",
            ]
        )
    return "\n".join(lines)


def _case_rect_add_remove_shift(_profile: ScaleProfile) -> SyntheticCaseBuild:
    layer = (1, 0)
    old, old_top = _library()
    new, new_top = _library()
    stable = _rect(0, 0, 10, 10, layer)
    old_removed = _rect(20, 0, 30, 10, layer)
    old_shift = _rect(40, 0, 50, 10, layer)
    new_shift = _rect(43, 0, 53, 10, layer)
    added = _rect(60, 0, 70, 10, layer)
    _add(old_top, stable, old_removed, old_shift)
    _add(new_top, _rect(0, 0, 10, 10, layer), new_shift, added)
    edits = (
        SyntheticEdit("rect_remove_001", "removed_rect", "remove", "coverage_before_only", before_layer=layer, before_bbox_um=_bbox(old_removed)),
        SyntheticEdit("rect_shift_001", "shifted_rect", "shift", "coverage_both", before_layer=layer, after_layer=layer, before_bbox_um=_bbox(old_shift), after_bbox_um=_bbox(new_shift)),
        SyntheticEdit("rect_add_001", "added_rect", "add", "coverage_after_only", after_layer=layer, after_bbox_um=_bbox(added)),
    )
    return SyntheticCaseBuild(old, new, edits, notes="Simple axis-aligned add, remove, and shift operations.")


def _case_overlap_rect_unions(_profile: ScaleProfile) -> SyntheticCaseBuild:
    layer = (2, 0)
    old, old_top = _library()
    new, new_top = _library()
    old_a = _rect(0, 0, 20, 20, layer)
    old_b = _rect(10, 0, 30, 20, layer)
    new_a = _rect(0, 0, 20, 20, layer)
    new_b = _rect(15, 0, 35, 20, layer)
    new_c = _rect(5, 10, 25, 30, layer)
    _add(old_top, old_a, old_b)
    _add(new_top, new_a, new_b, new_c)
    edits = (
        SyntheticEdit("overlap_shift_001", "overlap_rect_b", "shift_overlapping_rect", "coverage_both", before_layer=layer, after_layer=layer, before_bbox_um=_bbox(old_b), after_bbox_um=_bbox(new_b), notes="The shifted rectangle overlaps a stable neighbor."),
        SyntheticEdit("overlap_add_001", "overlap_rect_c", "add_overlapping_rect", "coverage_after_only", after_layer=layer, after_bbox_um=_bbox(new_c)),
    )
    return SyntheticCaseBuild(old, new, edits)


def _case_circle_breakpoints(_profile: ScaleProfile) -> SyntheticCaseBuild:
    layer = (3, 0)
    old, old_top = _library()
    new, new_top = _library()
    old_circle = _ellipse_polygon(0, 0, 10, 10, 32, layer)
    new_circle = _ellipse_polygon(0, 0, 10, 10, 48, layer)
    old_ellipse = _ellipse_polygon(35, 0, 13, 8, 40, layer)
    new_ellipse = _ellipse_polygon(35, 0, 10.5, 8, 40, layer)
    _add(old_top, old_circle, old_ellipse)
    _add(new_top, new_circle, new_ellipse)
    edits = (
        SyntheticEdit("circle_breakpoint_001", "circle_same_radius", "change_curve_breakpoints", "coverage_both", before_layer=layer, after_layer=layer, before_bbox_um=_bbox(old_circle), after_bbox_um=_bbox(new_circle), notes="Same radius with different polygonal approximation."),
        SyntheticEdit("ellipse_thin_001", "ellipse_thinned", "thin_curve", "coverage_before_only", before_layer=layer, after_layer=layer, before_bbox_um=_bbox(old_ellipse), after_bbox_um=_bbox(new_ellipse)),
    )
    return SyntheticCaseBuild(old, new, edits)


def _case_star_concave(_profile: ScaleProfile) -> SyntheticCaseBuild:
    layer = (4, 0)
    old, old_top = _library()
    new, new_top = _library()
    old_star = _star_polygon(0, 0, 12, 5, 5, 0.0, layer)
    new_star = _star_polygon(0, 0, 12, 5, 5, 0.2, layer)
    old_removed = _star_polygon(35, 0, 10, 4, 7, 0.1, layer)
    new_added = _star_polygon(70, 0, 11, 4, 6, -0.3, layer)
    _add(old_top, old_star, old_removed)
    _add(new_top, new_star, new_added)
    edits = (
        SyntheticEdit("star_rotate_001", "five_point_star", "rotate_concave_polygon", "coverage_both", before_layer=layer, after_layer=layer, before_bbox_um=_bbox(old_star), after_bbox_um=_bbox(new_star)),
        SyntheticEdit("star_remove_001", "seven_point_star", "remove_concave_polygon", "coverage_before_only", before_layer=layer, before_bbox_um=_bbox(old_removed)),
        SyntheticEdit("star_add_001", "six_point_star", "add_concave_polygon", "coverage_after_only", after_layer=layer, after_bbox_um=_bbox(new_added)),
    )
    return SyntheticCaseBuild(old, new, edits)


def _case_hollow_rect_circle(_profile: ScaleProfile) -> SyntheticCaseBuild:
    layer = (5, 0)
    old, old_top = _library()
    new, new_top = _library()
    old_ring = _rect_ring(0, 0, 34, 34, 9, 9, 25, 25, layer)
    new_ring = _rect_ring(0, 0, 34, 34, 7, 7, 27, 27, layer)
    old_donut = _ellipse_ring(70, 17, 16, 16, 7, 7, 48, layer)
    new_donut = _ellipse_ring(70, 17, 16, 16, 9, 5, 64, layer)
    _add(old_top, *old_ring, *old_donut)
    _add(new_top, *new_ring, *new_donut)
    edits = (
        SyntheticEdit("hollow_rect_001", "rectangular_hole", "resize_inner_hole", "coverage_before_only", before_layer=layer, after_layer=layer, before_bbox_um=(7, 7, 27, 27), after_bbox_um=(7, 7, 27, 27)),
        SyntheticEdit("hollow_circle_001", "circular_hole", "resize_and_refracture_inner_hole", "coverage_both", before_layer=layer, after_layer=layer, before_bbox_um=(63, 10, 77, 24), after_bbox_um=(61, 12, 79, 22)),
    )
    return SyntheticCaseBuild(old, new, edits)


def _case_nested_holes_bridge(_profile: ScaleProfile) -> SyntheticCaseBuild:
    layer = (6, 0)
    old, old_top = _library()
    new, new_top = _library()
    outer = _rect(0, 0, 90, 42, layer)
    holes_old = [_rect(10, 10, 22, 30, layer), _rect(38, 10, 50, 30, layer), _rect(66, 10, 78, 30, layer)]
    holes_new = [_rect(10, 10, 22, 30, layer), _rect(66, 10, 78, 30, layer), _rect(42, 0, 46, 42, layer)]
    old_polys = _boolean(outer, holes_old, "not", layer)
    new_polys = _boolean(_rect(0, 0, 90, 42, layer), holes_new, "not", layer)
    _add(old_top, *old_polys)
    _add(new_top, *new_polys)
    edits = (
        SyntheticEdit("nested_fill_001", "middle_hole", "fill_inner_hole", "coverage_after_only", before_layer=layer, after_layer=layer, before_bbox_um=(38, 10, 50, 30), after_bbox_um=(38, 10, 50, 30)),
        SyntheticEdit("nested_bridge_cut_001", "vertical_slot", "add_bridge_cut", "coverage_before_only", before_layer=layer, after_layer=layer, before_bbox_um=(42, 0, 46, 42), after_bbox_um=(42, 0, 46, 42)),
    )
    return SyntheticCaseBuild(old, new, edits)


def _case_thin_slivers_near_collinear(_profile: ScaleProfile) -> SyntheticCaseBuild:
    layer = (7, 0)
    old, old_top = _library()
    new, new_top = _library()
    old_sliver = _rect(0, 0, 100, 0.2, layer)
    new_sliver = _rect(0, 0, 80, 0.2, layer)
    old_poly = gdstk.Polygon([(0, 10), (50, 10.001), (100, 10), (100, 12), (0, 12)], layer=layer[0], datatype=layer[1])
    new_poly = gdstk.Polygon([(0, 10), (40, 10.001), (100, 10.15), (100, 12), (0, 12)], layer=layer[0], datatype=layer[1])
    added = _rect(120, 0, 160, 0.15, layer)
    _add(old_top, old_sliver, old_poly)
    _add(new_top, new_sliver, new_poly, added)
    edits = (
        SyntheticEdit("sliver_shorten_001", "thin_rectangle", "shorten_thin_polygon", "coverage_before_only", before_layer=layer, after_layer=layer, before_bbox_um=_bbox(old_sliver), after_bbox_um=_bbox(new_sliver)),
        SyntheticEdit("sliver_collinear_001", "near_collinear_polygon", "perturb_near_collinear_vertex", "coverage_before_only", before_layer=layer, after_layer=layer, before_bbox_um=_bbox(old_poly), after_bbox_um=_bbox(new_poly)),
        SyntheticEdit("sliver_add_001", "added_sliver", "add_thin_polygon", "coverage_after_only", after_layer=layer, after_bbox_um=_bbox(added)),
    )
    return SyntheticCaseBuild(old, new, edits)


def _case_layer_datatype_moves(_profile: ScaleProfile) -> SyntheticCaseBuild:
    old_layer_a = (8, 0)
    old_layer_b = (8, 1)
    new_layer_a = (9, 0)
    new_layer_b = (8, 2)
    old, old_top = _library()
    new, new_top = _library()
    rect_a_old = _rect(0, 0, 12, 12, old_layer_a)
    rect_b_old = _rect(30, 0, 42, 12, old_layer_b)
    rect_a_new = _rect(0, 0, 12, 12, new_layer_a)
    rect_b_new = _rect(30, 0, 42, 12, new_layer_b)
    _add(old_top, rect_a_old, rect_b_old)
    _add(new_top, rect_a_new, rect_b_new)
    edits = (
        SyntheticEdit("layer_move_001", "layer_moved_rect", "move_layer", "coverage_both", before_layer=old_layer_a, after_layer=new_layer_a, before_bbox_um=_bbox(rect_a_old), after_bbox_um=_bbox(rect_a_new)),
        SyntheticEdit("datatype_move_001", "datatype_moved_rect", "move_datatype", "coverage_both", before_layer=old_layer_b, after_layer=new_layer_b, before_bbox_um=_bbox(rect_b_old), after_bbox_um=_bbox(rect_b_new)),
    )
    return SyntheticCaseBuild(old, new, edits)


def _case_same_coverage_refracturing(_profile: ScaleProfile) -> SyntheticCaseBuild:
    layer = (10, 0)
    old, old_top = _library()
    new, new_top = _library()
    old_rect = _rect(0, 0, 40, 20, layer)
    left = _rect(0, 0, 20, 20, layer)
    right = _rect(20, 0, 40, 20, layer)
    old_top.add(old_rect)
    _add(new_top, left, right)
    edits = (
        SyntheticEdit("refracture_001", "split_rectangle", "split_equivalent_coverage", "inventory_only", before_layer=layer, after_layer=layer, before_bbox_um=_bbox(old_rect), after_bbox_um=_bbox(old_rect), notes="Coverage is intentionally identical; polygon inventory changes only."),
    )
    return SyntheticCaseBuild(old, new, edits, equivalent_expected=True)


def _case_touching_edges_points(_profile: ScaleProfile) -> SyntheticCaseBuild:
    layer = (11, 0)
    old, old_top = _library()
    new, new_top = _library()
    old_left = _rect(0, 0, 10, 10, layer)
    old_right = _rect(10, 0, 20, 10, layer)
    old_corner = _rect(30, 0, 40, 10, layer)
    new_left = _rect(0, 0, 10, 10, layer)
    new_right = _rect(10.2, 0, 20.2, 10, layer)
    new_corner = _rect(39.8, 9.8, 49.8, 19.8, layer)
    _add(old_top, old_left, old_right, old_corner)
    _add(new_top, new_left, new_right, new_corner)
    edits = (
        SyntheticEdit("touch_gap_001", "edge_touching_pair", "open_small_gap", "coverage_both", before_layer=layer, after_layer=layer, before_bbox_um=(10, 0, 20, 10), after_bbox_um=(10.2, 0, 20.2, 10)),
        SyntheticEdit("touch_point_overlap_001", "corner_touching_square", "move_to_tiny_overlap", "coverage_both", before_layer=layer, after_layer=layer, before_bbox_um=_bbox(old_corner), after_bbox_um=_bbox(new_corner)),
    )
    return SyntheticCaseBuild(old, new, edits)


def _case_cell_references_arrays(profile: ScaleProfile) -> SyntheticCaseBuild:
    layer = (12, 0)
    old = gdstk.Library(unit=_UNIT, precision=_PRECISION)
    old_child = old.new_cell("leaf")
    old_child.add(_rect(0, 0, 8, 8, layer))
    old_top = old.new_cell("top")
    old_top.add(gdstk.Reference(old_child, (0, 0), columns=profile.hierarchy_cols, rows=profile.hierarchy_rows, spacing=(16, 16)))

    new = gdstk.Library(unit=_UNIT, precision=_PRECISION)
    new_child = new.new_cell("leaf")
    new_child.add(_rect(0, 0, 8, 8, layer))
    new_shifted = new.new_cell("leaf_shifted")
    new_shifted.add(_rect(1.5, 0, 9.5, 8, layer))
    new_top = new.new_cell("top")
    new_top.add(gdstk.Reference(new_child, (0, 0), columns=max(1, profile.hierarchy_cols - 1), rows=profile.hierarchy_rows, spacing=(16, 16)))
    new_top.add(gdstk.Reference(new_shifted, ((profile.hierarchy_cols - 1) * 16, 0), rows=profile.hierarchy_rows, spacing=(16, 16)))
    target_y1 = (profile.hierarchy_rows - 1) * 16 + 8
    old_bbox = ((profile.hierarchy_cols - 1) * 16, 0, (profile.hierarchy_cols - 1) * 16 + 8, target_y1)
    new_bbox = ((profile.hierarchy_cols - 1) * 16 + 1.5, 0, (profile.hierarchy_cols - 1) * 16 + 9.5, target_y1)
    edits = (
        SyntheticEdit("hierarchy_array_shift_001", "last_array_column", "shift_referenced_array_column", "coverage_both", before_layer=layer, after_layer=layer, before_bbox_um=old_bbox, after_bbox_um=new_bbox, notes="Only the final array column uses a shifted leaf cell."),
    )
    return SyntheticCaseBuild(old, new, edits)


def _case_sparse_large_rect_grid(profile: ScaleProfile) -> SyntheticCaseBuild:
    layer = (13, 0)
    old, old_top = _library()
    new, new_top = _library()
    n = profile.sparse_grid_n
    removed = {(1, 1), (n // 2, n // 2)}
    shifted = {(n - 2, n - 2)}
    added_positions = [(n + 2, 0), (n + 3, 1)]
    for ix in range(n):
        for iy in range(n):
            poly = _grid_rect(ix, iy, layer)
            old_top.add(poly)
            if (ix, iy) in removed:
                continue
            if (ix, iy) in shifted:
                new_top.add(_grid_rect(ix, iy, layer, dx=0.7))
            else:
                new_top.add(_grid_rect(ix, iy, layer))
    for pos in added_positions:
        new_top.add(_grid_rect(pos[0], pos[1], layer))
    edits = (
        SyntheticEdit("sparse_remove_001", "grid_rect_1_1", "remove_grid_rect", "coverage_before_only", before_layer=layer, before_bbox_um=_grid_bbox(1, 1)),
        SyntheticEdit("sparse_remove_002", "grid_rect_mid", "remove_grid_rect", "coverage_before_only", before_layer=layer, before_bbox_um=_grid_bbox(n // 2, n // 2)),
        SyntheticEdit("sparse_shift_001", "grid_rect_last", "shift_grid_rect", "coverage_both", before_layer=layer, after_layer=layer, before_bbox_um=_grid_bbox(n - 2, n - 2), after_bbox_um=_grid_bbox(n - 2, n - 2, dx=0.7)),
        SyntheticEdit("sparse_add_001", "grid_rect_extra_0", "add_grid_rect", "coverage_after_only", after_layer=layer, after_bbox_um=_grid_bbox(n + 2, 0)),
        SyntheticEdit("sparse_add_002", "grid_rect_extra_1", "add_grid_rect", "coverage_after_only", after_layer=layer, after_bbox_um=_grid_bbox(n + 3, 1)),
    )
    return SyntheticCaseBuild(old, new, edits)


def _case_mixed_complex_polygons(profile: ScaleProfile) -> SyntheticCaseBuild:
    layer = (14, 0)
    old, old_top = _library()
    new, new_top = _library()
    count = profile.mixed_count
    changed = {3, count // 3, (2 * count) // 3}
    removed_index = count // 2
    for index in range(count):
        x = (index % 40) * 18.0
        y = (index // 40) * 18.0
        kind = index % 3
        old_poly = _mixed_shape(kind, x, y, index, layer, changed=False)
        old_top.add(old_poly)
        if index == removed_index:
            continue
        new_top.add(_mixed_shape(kind, x, y, index, layer, changed=index in changed))
    added = _star_polygon(-35, -35, 8, 3, 6, 0.2, layer)
    new_top.add(added)
    edits = tuple(
        SyntheticEdit(
            f"mixed_change_{slot:03d}",
            f"mixed_shape_{slot}",
            "modify_complex_polygon",
            "coverage_both",
            before_layer=layer,
            after_layer=layer,
            before_bbox_um=_bbox(_mixed_shape(slot % 3, (slot % 40) * 18.0, (slot // 40) * 18.0, slot, layer, changed=False)),
            after_bbox_um=_bbox(_mixed_shape(slot % 3, (slot % 40) * 18.0, (slot // 40) * 18.0, slot, layer, changed=True)),
        )
        for slot in sorted(changed)
    ) + (
        SyntheticEdit("mixed_remove_001", f"mixed_shape_{removed_index}", "remove_complex_polygon", "coverage_before_only", before_layer=layer, before_bbox_um=_bbox(_mixed_shape(removed_index % 3, (removed_index % 40) * 18.0, (removed_index // 40) * 18.0, removed_index, layer, changed=False))),
        SyntheticEdit("mixed_add_001", "mixed_added_star", "add_complex_polygon", "coverage_after_only", after_layer=layer, after_bbox_um=_bbox(added)),
    )
    return SyntheticCaseBuild(old, new, edits)


def _case_hollow_grid_many(profile: ScaleProfile) -> SyntheticCaseBuild:
    layer = (15, 0)
    old, old_top = _library()
    new, new_top = _library()
    n = profile.hollow_grid_n
    changed = (n // 2, n // 2)
    removed = (max(0, n - 2), 1)
    for ix in range(n):
        for iy in range(n):
            x = ix * 18.0
            y = iy * 18.0
            old_ring = _rect_ring(x, y, x + 12, y + 12, x + 4, y + 4, x + 8, y + 8, layer)
            for poly in old_ring:
                old_top.add(poly)
            if (ix, iy) == removed:
                continue
            if (ix, iy) == changed:
                new_ring = _rect_ring(x, y, x + 12, y + 12, x + 3, y + 3, x + 9, y + 9, layer)
            else:
                new_ring = _rect_ring(x, y, x + 12, y + 12, x + 4, y + 4, x + 8, y + 8, layer)
            for poly in new_ring:
                new_top.add(poly)
    edits = (
        SyntheticEdit("hollow_grid_resize_001", "central_ring_hole", "resize_grid_ring_hole", "coverage_both", before_layer=layer, after_layer=layer, before_bbox_um=(changed[0] * 18 + 4, changed[1] * 18 + 4, changed[0] * 18 + 8, changed[1] * 18 + 8), after_bbox_um=(changed[0] * 18 + 3, changed[1] * 18 + 3, changed[0] * 18 + 9, changed[1] * 18 + 9)),
        SyntheticEdit("hollow_grid_remove_001", "removed_ring", "remove_hollow_ring", "coverage_before_only", before_layer=layer, before_bbox_um=(removed[0] * 18, removed[1] * 18, removed[0] * 18 + 12, removed[1] * 18 + 12)),
    )
    return SyntheticCaseBuild(old, new, edits)


def _case_multi_layer_mixed_edits(_profile: ScaleProfile) -> SyntheticCaseBuild:
    layers = ((16, 0), (16, 1), (17, 0), (18, 2))
    old, old_top = _library()
    new, new_top = _library()
    edits: list[SyntheticEdit] = []
    for idx, layer in enumerate(layers):
        for j in range(8):
            x = idx * 80 + j * 8
            rect = _rect(x, 0, x + 5, 5, layer)
            old_top.add(rect)
            if idx == 0 and j == 2:
                edits.append(SyntheticEdit("multi_remove_001", "layer16_removed", "remove_layer_shape", "coverage_before_only", before_layer=layer, before_bbox_um=_bbox(rect)))
                continue
            if idx == 1 and j == 3:
                moved_layer = (19, 0)
                new_rect = _rect(x, 0, x + 5, 5, moved_layer)
                new_top.add(new_rect)
                edits.append(SyntheticEdit("multi_layer_move_001", "datatype_shape_moved_layer", "move_to_new_layer", "coverage_both", before_layer=layer, after_layer=moved_layer, before_bbox_um=_bbox(rect), after_bbox_um=_bbox(new_rect)))
                continue
            if idx == 2 and j == 4:
                new_rect = _rect(x + 1, 0, x + 6, 5, layer)
                new_top.add(new_rect)
                edits.append(SyntheticEdit("multi_shift_001", "layer17_shifted", "shift_layer_shape", "coverage_both", before_layer=layer, after_layer=layer, before_bbox_um=_bbox(rect), after_bbox_um=_bbox(new_rect)))
                continue
            new_top.add(_rect(x, 0, x + 5, 5, layer))
    added = _ellipse_polygon(360, 0, 5, 5, 24, layers[-1])
    new_top.add(added)
    edits.append(SyntheticEdit("multi_add_001", "layer18_added_circle", "add_layer_shape", "coverage_after_only", after_layer=layers[-1], after_bbox_um=_bbox(added)))
    return SyntheticCaseBuild(old, new, tuple(edits))


def _case_block_like_hierarchy_stress(profile: ScaleProfile) -> SyntheticCaseBuild:
    layer = (20, 0)
    old = gdstk.Library(unit=_UNIT, precision=_PRECISION)
    old_block = old.new_cell("block")
    _add(old_block, _rect(0, 0, 8, 8, layer), _rect(10, 0, 18, 8, layer), _star_polygon(9, 16, 5, 2, 5, 0.0, layer))
    old_top = old.new_cell("top")

    new = gdstk.Library(unit=_UNIT, precision=_PRECISION)
    new_block = new.new_cell("block")
    _add(new_block, _rect(0, 0, 8, 8, layer), _rect(10, 0, 18, 8, layer), _star_polygon(9, 16, 5, 2, 5, 0.0, layer))
    new_block_modified = new.new_cell("block_modified")
    _add(new_block_modified, _rect(0, 0, 8, 8, layer), _rect(11.5, 0, 19.5, 8, layer), _star_polygon(9, 16, 5, 2, 5, 0.25, layer))
    new_top = new.new_cell("top")

    cols = profile.reference_cols
    rows = profile.reference_rows
    modified = {(cols // 2, rows // 2), (max(0, cols - 3), max(0, rows - 2))}
    for ix in range(cols):
        for iy in range(rows):
            origin = (ix * 28, iy * 28)
            old_top.add(gdstk.Reference(old_block, origin))
            new_top.add(gdstk.Reference(new_block_modified if (ix, iy) in modified else new_block, origin))
    edits = tuple(
        SyntheticEdit(
            f"block_modify_{idx:03d}",
            f"block_ref_{ix}_{iy}",
            "modify_referenced_block",
            "coverage_both",
            before_layer=layer,
            after_layer=layer,
            before_bbox_um=(ix * 28, iy * 28, ix * 28 + 20, iy * 28 + 22),
            after_bbox_um=(ix * 28, iy * 28, ix * 28 + 20, iy * 28 + 22),
            notes="One referenced block instance uses shifted metal and a rotated star.",
        )
        for idx, (ix, iy) in enumerate(sorted(modified), start=1)
    )
    return SyntheticCaseBuild(old, new, edits)


def _case_adversarial_nested_rings_islands(_profile: ScaleProfile) -> SyntheticCaseBuild:
    layer = (21, 0)
    old, old_top = _library()
    new, new_top = _library()
    old_outer = _rect(0, 0, 90, 70, layer)
    old_holes = (
        _rect(10, 10, 25, 25, layer),
        _rect(35, 10, 50, 25, layer),
        _rect(20, 40, 70, 58, layer),
    )
    new_holes = (
        _rect(10, 10, 25, 25, layer),
        _rect(32, 7, 54, 28, layer),
        _rect(20, 40, 70, 58, layer),
    )
    island = _rect(41, 15, 46, 20, layer)
    inner_ring_old = _rect_ring(30, 42, 60, 56, 38, 46, 52, 52, layer)
    inner_ring_new = _rect_ring(30, 42, 60, 56, 40, 48, 50, 51, layer)
    _add(old_top, *_boolean(old_outer, old_holes, "not", layer), *inner_ring_old)
    _add(new_top, *_boolean(_rect(0, 0, 90, 70, layer), new_holes, "not", layer), island, *inner_ring_new)
    edits = (
        SyntheticEdit(
            "adv_nested_hole_expand_001",
            "expanded_rect_hole",
            "expand_inner_hole_inside_outer_ring",
            "coverage_before_only",
            before_layer=layer,
            after_layer=layer,
            before_bbox_um=(32, 7, 54, 28),
            after_bbox_um=(32, 7, 54, 28),
            notes="The hole is enlarged, creating before-only ring fragments fully contained by the outer polygon.",
        ),
        SyntheticEdit(
            "adv_nested_island_add_001",
            "island_inside_hole",
            "add_island_inside_hole",
            "coverage_after_only",
            after_layer=layer,
            after_bbox_um=_bbox(island),
            notes="A disconnected island is added inside a void.",
        ),
        SyntheticEdit(
            "adv_nested_inner_ring_fill_001",
            "nested_ring_hole",
            "shrink_nested_ring_hole",
            "coverage_after_only",
            before_layer=layer,
            after_layer=layer,
            before_bbox_um=(38, 46, 52, 52),
            after_bbox_um=(38, 46, 52, 52),
            notes="The inner ring hole shrinks, producing after-only coverage inside nested topology.",
        ),
    )
    return SyntheticCaseBuild(old, new, edits, notes="Nested ring, multi-hole, and island-in-hole topology.")


def _case_adversarial_hole_touching_boundary(_profile: ScaleProfile) -> SyntheticCaseBuild:
    layer = (22, 0)
    old, old_top = _library()
    new, new_top = _library()
    old_outer = _rect(0, 0, 60, 42, layer)
    old_hole = _rect(12, 10, 28, 30, layer)
    new_edge_cut = _rect(0, 10, 28, 30, layer)
    old_diamond = _diamond_polygon(46, 21, 7, layer)
    new_diamond = _diamond_polygon(53, 21, 7, layer)
    _add(old_top, *_boolean(old_outer, (old_hole, old_diamond), "not", layer))
    _add(new_top, *_boolean(_rect(0, 0, 60, 42, layer), (new_edge_cut, new_diamond), "not", layer))
    edits = (
        SyntheticEdit(
            "adv_hole_edge_touch_001",
            "rect_hole_to_edge_notch",
            "grow_hole_until_it_touches_outer_edge",
            "coverage_before_only",
            before_layer=layer,
            after_layer=layer,
            before_bbox_um=(0, 10, 12, 30),
            after_bbox_um=(0, 10, 12, 30),
            notes="A true hole becomes a notch connected to the outer boundary.",
        ),
        SyntheticEdit(
            "adv_hole_point_touch_001",
            "diamond_hole_to_boundary_point",
            "move_hole_until_one_vertex_touches_boundary",
            "coverage_both",
            before_layer=layer,
            after_layer=layer,
            before_bbox_um=(39, 14, 53, 28),
            after_bbox_um=(46, 14, 60, 28),
            notes="The moved diamond cutout has a point on the outer boundary.",
        ),
    )
    return SyntheticCaseBuild(old, new, edits)


def _case_adversarial_containment_no_edge_crossing(_profile: ScaleProfile) -> SyntheticCaseBuild:
    layer = (23, 0)
    old, old_top = _library()
    new, new_top = _library()
    small_old = _rect(0, 0, 10, 10, layer)
    grown_new = _rect(-5, -5, 15, 15, layer)
    large_old = _rect(35, 0, 75, 35, layer)
    shrunk_new = _rect(48, 9, 62, 24, layer)
    _add(old_top, small_old, large_old)
    _add(new_top, grown_new, shrunk_new)
    edits = (
        SyntheticEdit(
            "adv_containment_grow_001",
            "small_rect_grown_around_old",
            "grow_polygon_fully_containing_before_polygon",
            "coverage_after_only",
            before_layer=layer,
            after_layer=layer,
            before_bbox_um=_bbox(small_old),
            after_bbox_um=_bbox(grown_new),
            notes="The before polygon is fully contained by the after polygon, with no boundary crossing.",
        ),
        SyntheticEdit(
            "adv_containment_shrink_001",
            "large_rect_shrunk_inside_old",
            "shrink_polygon_fully_inside_before_polygon",
            "coverage_before_only",
            before_layer=layer,
            after_layer=layer,
            before_bbox_um=_bbox(large_old),
            after_bbox_um=_bbox(shrunk_new),
            notes="The after polygon is fully contained by the before polygon.",
        ),
    )
    return SyntheticCaseBuild(old, new, edits)


def _case_adversarial_reversed_winding_equivalent(_profile: ScaleProfile) -> SyntheticCaseBuild:
    layer = (24, 0)
    old, old_top = _library()
    new, new_top = _library()
    points = ((0, 0), (34, 0), (34, 18), (18, 18), (18, 30), (0, 30))
    old_poly = gdstk.Polygon(points, layer=layer[0], datatype=layer[1])
    new_poly = gdstk.Polygon(tuple(reversed(points)), layer=layer[0], datatype=layer[1])
    old_top.add(old_poly)
    new_top.add(new_poly)
    edits = (
        SyntheticEdit(
            "adv_winding_reverse_001",
            "same_polygon_reversed_winding",
            "reverse_vertex_winding_without_changing_coverage",
            "inventory_only",
            before_layer=layer,
            after_layer=layer,
            before_bbox_um=_bbox(old_poly),
            after_bbox_um=_bbox(new_poly),
            notes="Coverage is identical; only source vertex order changes.",
        ),
    )
    return SyntheticCaseBuild(old, new, edits, equivalent_expected=True)


def _case_adversarial_duplicate_polygons_equivalent(_profile: ScaleProfile) -> SyntheticCaseBuild:
    layer = (25, 0)
    old, old_top = _library()
    new, new_top = _library()
    rect = _rect(0, 0, 30, 20, layer)
    old_top.add(rect)
    old_top.add(_rect(0, 0, 30, 20, layer))
    _add(new_top, _rect(0, 0, 15, 20, layer), _rect(15, 0, 30, 20, layer))
    edits = (
        SyntheticEdit(
            "adv_duplicate_refracture_001",
            "duplicate_rect_to_split_rect",
            "remove_duplicate_and_refracture_without_changing_coverage",
            "inventory_only",
            before_layer=layer,
            after_layer=layer,
            before_bbox_um=_bbox(rect),
            after_bbox_um=_bbox(rect),
            notes="Duplicate coincident source polygons and split target polygons describe identical coverage.",
        ),
    )
    return SyntheticCaseBuild(old, new, edits, equivalent_expected=True)


def _case_adversarial_partial_collinear_overlap(_profile: ScaleProfile) -> SyntheticCaseBuild:
    layer = (26, 0)
    old, old_top = _library()
    new, new_top = _library()
    base_old = _rect(0, 0, 60, 5, layer)
    tab_old = _rect(12, 5, 42, 18, layer)
    tab_new = _rect(12, 5.001, 42, 18.001, layer)
    _add(old_top, base_old, tab_old, _rect(72, 0, 92, 20, layer))
    _add(new_top, _rect(0, 0, 60, 5, layer), tab_new, _rect(72, 0, 92, 20, layer))
    edits = (
        SyntheticEdit(
            "adv_collinear_gap_001",
            "tab_with_partially_collinear_edge",
            "open_one_dbu_gap_along_partially_collinear_edge",
            "coverage_both",
            before_layer=layer,
            after_layer=layer,
            before_bbox_um=_bbox(tab_old),
            after_bbox_um=_bbox(tab_new),
            notes="A tab sharing part of a base rectangle edge moves by one DBU.",
        ),
    )
    return SyntheticCaseBuild(old, new, edits)


def _case_adversarial_one_dbu_touch_tjunction(_profile: ScaleProfile) -> SyntheticCaseBuild:
    layer = (27, 0)
    old, old_top = _library()
    new, new_top = _library()
    corner_a = _rect(0, 0, 10, 10, layer)
    corner_b_old = _rect(10, 10, 20, 20, layer)
    corner_b_new = _rect(10.001, 10.001, 20.001, 20.001, layer)
    tee_stem_old = _rect(42, 0, 46, 18, layer)
    tee_bar = _rect(32, 18, 56, 24, layer)
    tee_stem_new = _rect(42, 0, 46.001, 18, layer)
    _add(old_top, corner_a, corner_b_old, tee_stem_old, tee_bar)
    _add(new_top, _rect(0, 0, 10, 10, layer), corner_b_new, tee_stem_new, _rect(32, 18, 56, 24, layer))
    edits = (
        SyntheticEdit(
            "adv_point_touch_gap_001",
            "corner_touch_pair",
            "open_one_dbu_gap_from_point_touch",
            "coverage_both",
            before_layer=layer,
            after_layer=layer,
            before_bbox_um=_bbox(corner_b_old),
            after_bbox_um=_bbox(corner_b_new),
        ),
        SyntheticEdit(
            "adv_tjunction_overlap_001",
            "t_junction_stem",
            "extend_t_junction_stem_by_one_dbu",
            "coverage_after_only",
            before_layer=layer,
            after_layer=layer,
            before_bbox_um=_bbox(tee_stem_old),
            after_bbox_um=(46, 0, 46.001, 18),
        ),
    )
    return SyntheticCaseBuild(old, new, edits)


def _case_adversarial_one_dbu_sliver_grid(profile: ScaleProfile) -> SyntheticCaseBuild:
    layer = (28, 0)
    old, old_top = _library()
    new, new_top = _library()
    n = max(6, min(profile.sparse_grid_n, 180))
    removed = {1, n // 2}
    extended = n - 2
    for index in range(n):
        y = index * 0.01
        old_sliver = _rect(0, y, 20, y + 0.001, layer)
        old_top.add(old_sliver)
        if index in removed:
            continue
        if index == extended:
            new_top.add(_rect(0, y, 22, y + 0.001, layer))
        else:
            new_top.add(_rect(0, y, 20, y + 0.001, layer))
    edits = (
        SyntheticEdit("adv_sliver_remove_001", "one_dbu_sliver_1", "remove_one_dbu_sliver", "coverage_before_only", before_layer=layer, before_bbox_um=(0, 0.01, 20, 0.011)),
        SyntheticEdit("adv_sliver_remove_002", "one_dbu_sliver_mid", "remove_one_dbu_sliver", "coverage_before_only", before_layer=layer, before_bbox_um=(0, (n // 2) * 0.01, 20, (n // 2) * 0.01 + 0.001)),
        SyntheticEdit("adv_sliver_extend_001", "one_dbu_sliver_extended", "extend_one_dbu_sliver", "coverage_after_only", before_layer=layer, after_layer=layer, before_bbox_um=(20, extended * 0.01, 22, extended * 0.01 + 0.001), after_bbox_um=(20, extended * 0.01, 22, extended * 0.01 + 0.001)),
    )
    return SyntheticCaseBuild(old, new, edits, notes=f"Scale-aware one-DBU sliver set with {n} source slivers.")


def _case_adversarial_large_coordinates_tiny_delta(_profile: ScaleProfile) -> SyntheticCaseBuild:
    layer = (29, 0)
    old, old_top = _library()
    new, new_top = _library()
    base_x = 1_000_000.0
    base_y = -900_000.0
    old_rect = _rect(base_x, base_y, base_x + 25, base_y + 12, layer)
    new_rect = _rect(base_x + 0.001, base_y, base_x + 25.001, base_y + 12, layer)
    old_far = _rect(-1_000_000.0, 900_000.0, -999_980.0, 900_020.0, layer)
    new_far = _rect(-1_000_000.0, 900_000.001, -999_980.0, 900_020.001, layer)
    _add(old_top, old_rect, old_far)
    _add(new_top, new_rect, new_far)
    edits = (
        SyntheticEdit("adv_large_coord_shift_001", "large_positive_coordinate_rect", "shift_large_coordinate_rect_by_one_dbu", "coverage_both", before_layer=layer, after_layer=layer, before_bbox_um=_bbox(old_rect), after_bbox_um=_bbox(new_rect)),
        SyntheticEdit("adv_large_coord_shift_002", "large_negative_coordinate_rect", "shift_large_negative_coordinate_rect_by_one_dbu", "coverage_both", before_layer=layer, after_layer=layer, before_bbox_um=_bbox(old_far), after_bbox_um=_bbox(new_far)),
    )
    return SyntheticCaseBuild(old, new, edits)


def _case_adversarial_snap_rounding_equivalent(_profile: ScaleProfile) -> SyntheticCaseBuild:
    layer = (30, 0)
    old, old_top = _library()
    new, new_top = _library()
    old_rect = _rect(0, 0, 10, 10, layer)
    new_rect = _rect(0.0004, 0.0004, 10.0004, 10.0004, layer)
    old_top.add(old_rect)
    new_top.add(new_rect)
    edits = (
        SyntheticEdit(
            "adv_snap_rounding_001",
            "sub_dbu_shifted_rect",
            "shift_below_gds_grid_resolution",
            "inventory_only",
            before_layer=layer,
            after_layer=layer,
            before_bbox_um=_bbox(old_rect),
            after_bbox_um=_bbox(new_rect),
            notes="The geometric source coordinates differ, but GDS quantization snaps both to the same DBU grid.",
        ),
    )
    return SyntheticCaseBuild(old, new, edits, equivalent_expected=True)


def _case_adversarial_curve_single_vertex_bump(_profile: ScaleProfile) -> SyntheticCaseBuild:
    layer = (31, 0)
    old, old_top = _library()
    new, new_top = _library()
    old_curve = _ellipse_polygon(0, 0, 16, 12, 48, layer)
    points = [(float(x), float(y)) for x, y in old_curve.points]
    x, y = points[7]
    points[7] = (x + 1.25, y - 0.75)
    new_curve = gdstk.Polygon(points, layer=layer[0], datatype=layer[1])
    _add(old_top, old_curve, _ellipse_polygon(45, 0, 10, 10, 64, layer))
    _add(new_top, new_curve, _ellipse_polygon(45, 0, 10, 10, 64, layer))
    edits = (
        SyntheticEdit(
            "adv_curve_vertex_bump_001",
            "single_vertex_bumped_ellipse",
            "move_one_curve_breakpoint",
            "coverage_both",
            before_layer=layer,
            after_layer=layer,
            before_bbox_um=_bbox(old_curve),
            after_bbox_um=_bbox(new_curve),
        ),
    )
    return SyntheticCaseBuild(old, new, edits)


def _case_adversarial_dense_sparse_curve_overlap(_profile: ScaleProfile) -> SyntheticCaseBuild:
    layer = (32, 0)
    old, old_top = _library()
    new, new_top = _library()
    dense = _ellipse_polygon(0, 0, 20, 20, 128, layer)
    sparse = _ellipse_polygon(0, 0, 20, 20, 20, layer, angle=math.pi / 20)
    old_overlap = _ellipse_polygon(42, 0, 16, 10, 96, layer)
    new_overlap = _ellipse_polygon(42.001, 0, 16, 10, 24, layer)
    _add(old_top, dense, old_overlap)
    _add(new_top, sparse, new_overlap)
    edits = (
        SyntheticEdit("adv_curve_refracture_001", "dense_to_sparse_circle", "replace_dense_curve_with_sparse_curve", "coverage_both", before_layer=layer, after_layer=layer, before_bbox_um=_bbox(dense), after_bbox_um=_bbox(sparse)),
        SyntheticEdit("adv_curve_overlap_001", "overlapping_curve_changed_breakpoints", "shift_and_refracture_overlapping_curve", "coverage_both", before_layer=layer, after_layer=layer, before_bbox_um=_bbox(old_overlap), after_bbox_um=_bbox(new_overlap)),
    )
    return SyntheticCaseBuild(old, new, edits)


def _case_adversarial_hierarchy_transforms(profile: ScaleProfile) -> SyntheticCaseBuild:
    layer = (33, 0)
    old = gdstk.Library(unit=_UNIT, precision=_PRECISION)
    old_leaf = old.new_cell("leaf")
    _add(old_leaf, _rect(0, 0, 8, 4, layer), _star_polygon(4, 10, 4, 1.5, 5, 0.0, layer))
    old_top = old.new_cell("top")
    old_top.add(gdstk.Reference(old_leaf, (0, 0), columns=max(2, profile.hierarchy_cols // 2), rows=2, spacing=(20, 24)))
    old_top.add(gdstk.Reference(old_leaf, (120, 0), rotation=math.pi / 2, magnification=2.0))
    old_top.add(gdstk.Reference(old_leaf, (170, 40), rotation=math.pi / 4, magnification=1.5, x_reflection=True))

    new = gdstk.Library(unit=_UNIT, precision=_PRECISION)
    new_leaf = new.new_cell("leaf")
    _add(new_leaf, _rect(0, 0, 8, 4, layer), _star_polygon(4, 10, 4, 1.5, 5, 0.0, layer))
    new_leaf_shifted = new.new_cell("leaf_shifted")
    _add(new_leaf_shifted, _rect(1, 0, 9, 4, layer), _star_polygon(4, 10, 4, 1.5, 5, 0.2, layer))
    new_top = new.new_cell("top")
    new_top.add(gdstk.Reference(new_leaf, (0, 0), columns=max(2, profile.hierarchy_cols // 2), rows=2, spacing=(20, 24)))
    new_top.add(gdstk.Reference(new_leaf_shifted, (120, 0), rotation=math.pi / 2, magnification=2.0))
    new_top.add(gdstk.Reference(new_leaf, (170, 40), rotation=math.pi / 4, magnification=1.5, x_reflection=True))
    edits = (
        SyntheticEdit(
            "adv_hierarchy_transform_001",
            "rotated_magnified_reference",
            "modify_child_used_by_rotated_magnified_reference",
            "coverage_both",
            before_layer=layer,
            after_layer=layer,
            before_bbox_um=(108, -2, 124, 24),
            after_bbox_um=(108, -2, 124, 24),
            notes="Only a rotated and magnified reference changes; stable array and reflected reference remain unchanged.",
        ),
    )
    return SyntheticCaseBuild(old, new, edits)


def _case_adversarial_array_pitch_one_dbu(profile: ScaleProfile) -> SyntheticCaseBuild:
    layer = (34, 0)
    old = gdstk.Library(unit=_UNIT, precision=_PRECISION)
    old_leaf = old.new_cell("leaf")
    old_leaf.add(_rect(0, 0, 4, 4, layer))
    old_top = old.new_cell("top")
    rows = max(3, min(profile.hierarchy_rows, 20))
    cols = max(3, min(profile.hierarchy_cols, 20))
    old_top.add(gdstk.Reference(old_leaf, (0, 0), columns=cols, rows=rows, spacing=(8, 8)))

    new = gdstk.Library(unit=_UNIT, precision=_PRECISION)
    new_leaf = new.new_cell("leaf")
    new_leaf.add(_rect(0, 0, 4, 4, layer))
    new_top = new.new_cell("top")
    new_top.add(gdstk.Reference(new_leaf, (0, 0), columns=cols, rows=max(1, rows - 1), spacing=(8, 8)))
    new_top.add(gdstk.Reference(new_leaf, (0, (rows - 1) * 8 + 0.001), columns=cols, rows=1, spacing=(8, 8)))
    y0 = (rows - 1) * 8
    edits = (
        SyntheticEdit(
            "adv_array_pitch_001",
            "last_array_row",
            "shift_array_row_by_one_dbu",
            "coverage_both",
            before_layer=layer,
            after_layer=layer,
            before_bbox_um=(0, y0, (cols - 1) * 8 + 4, y0 + 4),
            after_bbox_um=(0, y0 + 0.001, (cols - 1) * 8 + 4, y0 + 4.001),
        ),
    )
    return SyntheticCaseBuild(old, new, edits)


BASE_SYNTHETIC_CASES: tuple[SyntheticCaseDefinition, ...] = (
    SyntheticCaseDefinition("rect_add_remove_shift", "Rectangle Add Remove Shift", _case_rect_add_remove_shift),
    SyntheticCaseDefinition("overlap_rect_unions", "Overlapping Rectangle Unions", _case_overlap_rect_unions),
    SyntheticCaseDefinition("circle_breakpoints", "Circle And Ellipse Breakpoints", _case_circle_breakpoints),
    SyntheticCaseDefinition("star_concave", "Concave Star Polygons", _case_star_concave),
    SyntheticCaseDefinition("hollow_rect_circle", "Hollow Rectangle And Donut", _case_hollow_rect_circle),
    SyntheticCaseDefinition("nested_holes_bridge", "Nested Holes And Bridge Cuts", _case_nested_holes_bridge),
    SyntheticCaseDefinition("thin_slivers_near_collinear", "Thin Slivers And Near-Collinear Vertices", _case_thin_slivers_near_collinear),
    SyntheticCaseDefinition("layer_datatype_moves", "Layer And Datatype Moves", _case_layer_datatype_moves),
    SyntheticCaseDefinition("same_coverage_refracturing", "Same-Coverage Refracturing", _case_same_coverage_refracturing),
    SyntheticCaseDefinition("touching_edges_points", "Touching Edges And Points", _case_touching_edges_points),
    SyntheticCaseDefinition("cell_references_arrays", "Cell References And Arrays", _case_cell_references_arrays),
    SyntheticCaseDefinition("sparse_large_rect_grid", "Sparse Large Rectangle Grid", _case_sparse_large_rect_grid),
    SyntheticCaseDefinition("mixed_complex_polygons", "Mixed Complex Polygon Set", _case_mixed_complex_polygons),
    SyntheticCaseDefinition("hollow_grid_many", "Many Hollow Grid Rings", _case_hollow_grid_many),
    SyntheticCaseDefinition("multi_layer_mixed_edits", "Multi-Layer Mixed Edits", _case_multi_layer_mixed_edits),
    SyntheticCaseDefinition("block_like_hierarchy_stress", "Block-Like Hierarchy Stress", _case_block_like_hierarchy_stress),
)


ADVERSARIAL_SYNTHETIC_CASES: tuple[SyntheticCaseDefinition, ...] = (
    SyntheticCaseDefinition("adversarial_nested_rings_islands", "Adversarial Nested Rings And Islands", _case_adversarial_nested_rings_islands),
    SyntheticCaseDefinition("adversarial_hole_touching_boundary", "Adversarial Hole Touching Boundary", _case_adversarial_hole_touching_boundary),
    SyntheticCaseDefinition("adversarial_containment_no_edge_crossing", "Adversarial Containment Without Edge Crossing", _case_adversarial_containment_no_edge_crossing),
    SyntheticCaseDefinition("adversarial_reversed_winding_equivalent", "Adversarial Reversed Winding Equivalent Coverage", _case_adversarial_reversed_winding_equivalent),
    SyntheticCaseDefinition("adversarial_duplicate_polygons_equivalent", "Adversarial Duplicate Polygons Equivalent Coverage", _case_adversarial_duplicate_polygons_equivalent),
    SyntheticCaseDefinition("adversarial_partial_collinear_overlap", "Adversarial Partial Collinear Overlap", _case_adversarial_partial_collinear_overlap),
    SyntheticCaseDefinition("adversarial_one_dbu_touch_tjunction", "Adversarial One-DBU Touch And T-Junction", _case_adversarial_one_dbu_touch_tjunction),
    SyntheticCaseDefinition("adversarial_one_dbu_sliver_grid", "Adversarial One-DBU Sliver Grid", _case_adversarial_one_dbu_sliver_grid),
    SyntheticCaseDefinition("adversarial_large_coordinates_tiny_delta", "Adversarial Large Coordinates Tiny Delta", _case_adversarial_large_coordinates_tiny_delta),
    SyntheticCaseDefinition("adversarial_snap_rounding_equivalent", "Adversarial Snap Rounding Equivalent Coverage", _case_adversarial_snap_rounding_equivalent),
    SyntheticCaseDefinition("adversarial_curve_single_vertex_bump", "Adversarial Curve Single Vertex Bump", _case_adversarial_curve_single_vertex_bump),
    SyntheticCaseDefinition("adversarial_dense_sparse_curve_overlap", "Adversarial Dense Sparse Curve Overlap", _case_adversarial_dense_sparse_curve_overlap),
    SyntheticCaseDefinition("adversarial_hierarchy_transforms", "Adversarial Hierarchy Transforms", _case_adversarial_hierarchy_transforms),
    SyntheticCaseDefinition("adversarial_array_pitch_one_dbu", "Adversarial Array Pitch One DBU", _case_adversarial_array_pitch_one_dbu),
)


SYNTHETIC_CASES: tuple[SyntheticCaseDefinition, ...] = BASE_SYNTHETIC_CASES + ADVERSARIAL_SYNTHETIC_CASES


def _library() -> tuple[gdstk.Library, gdstk.Cell]:
    lib = gdstk.Library(unit=_UNIT, precision=_PRECISION)
    return lib, lib.new_cell("top")


def _profile(scale: str) -> ScaleProfile:
    try:
        return SCALE_PROFILES[scale]
    except KeyError as exc:
        raise ValueError(f"unknown synthetic suite scale {scale!r}; expected one of {', '.join(SCALE_PROFILES)}") from exc


def _rect(x0: float, y0: float, x1: float, y1: float, layer: LayerTuple) -> gdstk.Polygon:
    return gdstk.rectangle((x0, y0), (x1, y1), layer=layer[0], datatype=layer[1])


def _grid_rect(ix: int, iy: int, layer: LayerTuple, *, dx: float = 0.0, dy: float = 0.0) -> gdstk.Polygon:
    x0, y0, x1, y1 = _grid_bbox(ix, iy, dx=dx, dy=dy)
    return _rect(x0, y0, x1, y1, layer)


def _grid_bbox(ix: int, iy: int, *, dx: float = 0.0, dy: float = 0.0) -> BBox:
    x = ix * 6.0 + dx
    y = iy * 6.0 + dy
    return (x, y, x + 3.0, y + 3.0)


def _ellipse_polygon(
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    point_count: int,
    layer: LayerTuple,
    *,
    angle: float = 0.0,
) -> gdstk.Polygon:
    points = []
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    for index in range(point_count):
        theta = 2 * math.pi * index / point_count
        x = rx * math.cos(theta)
        y = ry * math.sin(theta)
        points.append((cx + cos_a * x - sin_a * y, cy + sin_a * x + cos_a * y))
    return gdstk.Polygon(points, layer=layer[0], datatype=layer[1])


def _star_polygon(
    cx: float,
    cy: float,
    outer_radius: float,
    inner_radius: float,
    point_count: int,
    angle: float,
    layer: LayerTuple,
) -> gdstk.Polygon:
    points = []
    for index in range(point_count * 2):
        radius = outer_radius if index % 2 == 0 else inner_radius
        theta = angle + math.pi * index / point_count
        points.append((cx + radius * math.cos(theta), cy + radius * math.sin(theta)))
    return gdstk.Polygon(points, layer=layer[0], datatype=layer[1])


def _diamond_polygon(cx: float, cy: float, radius: float, layer: LayerTuple) -> gdstk.Polygon:
    return gdstk.Polygon(
        ((cx, cy + radius), (cx + radius, cy), (cx, cy - radius), (cx - radius, cy)),
        layer=layer[0],
        datatype=layer[1],
    )


def _rect_ring(
    ox0: float,
    oy0: float,
    ox1: float,
    oy1: float,
    ix0: float,
    iy0: float,
    ix1: float,
    iy1: float,
    layer: LayerTuple,
) -> tuple[gdstk.Polygon, ...]:
    return _boolean(_rect(ox0, oy0, ox1, oy1, layer), [_rect(ix0, iy0, ix1, iy1, layer)], "not", layer)


def _ellipse_ring(
    cx: float,
    cy: float,
    outer_rx: float,
    outer_ry: float,
    inner_rx: float,
    inner_ry: float,
    point_count: int,
    layer: LayerTuple,
) -> tuple[gdstk.Polygon, ...]:
    outer = _ellipse_polygon(cx, cy, outer_rx, outer_ry, point_count, layer)
    inner = _ellipse_polygon(cx, cy, inner_rx, inner_ry, point_count, layer)
    return _boolean(outer, [inner], "not", layer)


def _boolean(
    subject: gdstk.Polygon | Sequence[gdstk.Polygon],
    clip: Sequence[gdstk.Polygon],
    operation: str,
    layer: LayerTuple,
) -> tuple[gdstk.Polygon, ...]:
    result = gdstk.boolean(subject, clip, operation, precision=1e-3, layer=layer[0], datatype=layer[1])
    return tuple(result or ())


def _mixed_shape(kind: int, x: float, y: float, index: int, layer: LayerTuple, *, changed: bool) -> gdstk.Polygon:
    if kind == 0:
        delta = 0.7 if changed else 0.0
        return _rect(x + delta, y, x + 8 + delta, y + 5, layer)
    if kind == 1:
        radius = 5.5 if changed else 5.0
        return _ellipse_polygon(x + 5, y + 5, radius, 3.5, 20 + (index % 5), layer, angle=0.1 * (index % 7))
    angle = 0.4 if changed else 0.0
    return _star_polygon(x + 5, y + 5, 5.5, 2.0, 5 + (index % 3), angle, layer)


def _add(cell: gdstk.Cell, *items: gdstk.Polygon | gdstk.Reference) -> None:
    for item in items:
        cell.add(item)


def _bbox(poly: gdstk.Polygon) -> BBox:
    points = poly.points
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _bbox_dbu(bbox_um: BBox) -> tuple[int, int, int, int]:
    return tuple(int(round(value * _DBU_PER_UM)) for value in bbox_um)  # type: ignore[return-value]


def _bboxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _library_stats(lib: gdstk.Library) -> dict[str, int]:
    top = _top_cell(lib)
    polygons = top.get_polygons(apply_repetitions=True, include_paths=True)
    return {
        "polygon_count": len(polygons),
        "vertex_count": sum(len(polygon.points) for polygon in polygons),
    }


def _top_cell(lib: gdstk.Library) -> gdstk.Cell:
    for cell in lib.cells:
        if cell.name == "top":
            return cell
    tops = lib.top_level()
    if len(tops) != 1:
        raise ValueError(f"synthetic library expected one top cell, found {len(tops)}")
    return tops[0]


def _expected_changed_layers(edits: Sequence[SyntheticEdit]) -> tuple[LayerTuple, ...]:
    layers = {layer for edit in edits for layer in edit.changed_layers()}
    return tuple(sorted(layers))


def _changed_layers_from_report(payload: dict[str, object]) -> tuple[LayerTuple, ...]:
    layers = []
    for item in payload.get("layers", ()):
        if not isinstance(item, dict):
            continue
        if str(item.get("xor_area_dbu2", "0")) == "0" and str(item.get("old_minus_new_area_dbu2", "0")) == "0" and str(item.get("new_minus_old_area_dbu2", "0")) == "0":
            continue
        layers.append((int(item["layer"]), int(item["datatype"])))
    return tuple(sorted(layers))


def _audit_match_from_report(payload: dict[str, object]) -> bool | None:
    exact_fallback = payload.get("exact_fallback")
    if not isinstance(exact_fallback, dict):
        return None
    audit = exact_fallback.get("cpu_oracle_audit")
    if not isinstance(audit, dict):
        return None
    match = audit.get("match")
    return bool(match) if match is not None else None


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_layer(layer: LayerTuple) -> str:
    return f"{layer[0]}:{layer[1]}"


def _format_layers(layers: Iterable[LayerTuple]) -> str:
    return ",".join(_format_layer(layer) for layer in layers)


def _format_seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _json_dumps(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def validate_synthetic_case_with_cpu_oracle(record: SyntheticCaseRecord) -> dict[str, object]:
    old = extract_gds_polygons(record.before_path).buffer
    new = extract_gds_polygons(record.after_path).buffer
    result = run_full_exact_oracle(old, new)
    by_layer = {layer.layer: layer for layer in result.layers}
    changed_layers = tuple(sorted(layer for layer, diff in by_layer.items() if diff.xor_twice_area))
    expected_layers = tuple(LayerSpec(*layer) for layer in record.expected_changed_layers)
    missing_layers = tuple(layer for layer in expected_layers if layer not in changed_layers)
    missing_windows: list[dict[str, object]] = []
    for edit in record.edits:
        if not edit.coverage_changed:
            continue
        for layer_key, bbox_um, side in _validation_sides(edit):
            if layer_key is None or bbox_um is None:
                continue
            diff = by_layer.get(LayerSpec(*layer_key))
            if diff is None:
                missing_windows.append({"operation_id": edit.operation_id, "layer": layer_key, "side": side, "reason": "layer missing"})
                continue
            fragments = diff.old_minus_new if side == "before" else diff.new_minus_old
            bbox_dbu = _bbox_dbu(bbox_um)
            if fragments and not any(_bboxes_intersect(fragment.bbox, bbox_dbu) for fragment in fragments):
                missing_windows.append({"operation_id": edit.operation_id, "layer": layer_key, "side": side, "bbox_dbu": bbox_dbu})
    return {
        "equivalent": result.equivalent,
        "expected_equivalent": record.equivalent_expected,
        "changed_layers": [[layer.layer, layer.datatype] for layer in changed_layers],
        "expected_changed_layers": [list(layer) for layer in record.expected_changed_layers],
        "missing_layers": [[layer.layer, layer.datatype] for layer in missing_layers],
        "missing_windows": missing_windows,
    }


def _validation_sides(edit: SyntheticEdit) -> tuple[tuple[LayerTuple | None, BBox | None, str], ...]:
    if edit.expected_direction == "coverage_before_only":
        return ((edit.before_layer, edit.before_bbox_um, "before"),)
    if edit.expected_direction == "coverage_after_only":
        return ((edit.after_layer, edit.after_bbox_um, "after"),)
    if edit.expected_direction == "coverage_both":
        return (
            (edit.before_layer, edit.before_bbox_um, "before"),
            (edit.after_layer, edit.after_bbox_um, "after"),
        )
    return ()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate and optionally benchmark synthetic gdsdiff readiness GDS pairs.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--scale", action="append", choices=tuple(SCALE_PROFILES), help="Scale profile to generate. May be repeated.")
    parser.add_argument("--case", action="append", dest="cases", help="Generate only the named case. May be repeated.")
    parser.add_argument("--benchmark", action="store_true", help="Run CPU and Metal gdsdiff comparisons after generation.")
    parser.add_argument("--no-cpu", action="store_true", help="Skip CPU oracle benchmark runs.")
    parser.add_argument("--no-metal", action="store_true", help="Skip Metal benchmark runs.")
    parser.add_argument("--no-metal-audit", action="store_true", help="Skip Metal runs with CPU audit.")
    parser.add_argument("--stress-cpu-case-limit", type=int, default=2)
    args = parser.parse_args(argv)

    scales = tuple(args.scale or ("benchmark",))
    all_records: list[SyntheticCaseRecord] = []
    for scale in scales:
        scale_dir = args.output_dir / scale if len(scales) > 1 else args.output_dir
        all_records.extend(build_synthetic_suite(scale_dir, scale=scale, case_names=args.cases))

    if args.benchmark:
        runs = benchmark_synthetic_records(
            all_records,
            run_cpu=not args.no_cpu,
            run_metal=not args.no_metal,
            run_metal_audit=not args.no_metal_audit,
            stress_cpu_case_limit=args.stress_cpu_case_limit,
        )
        summary = write_benchmark_summary(args.output_dir, all_records, runs)
        print(f"benchmark_summary={summary}")
    for record in all_records:
        print(f"{record.scale}/{record.name}: {record.before_path} {record.after_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
