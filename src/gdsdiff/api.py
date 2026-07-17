from __future__ import annotations

import hashlib
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Callable

from .cache import (
    CachedGeometry,
    load_cached_pair_verdict,
    load_or_build_cached_geometry,
    load_or_build_cached_native_polygons,
    write_cached_pair_verdict,
)
from .canonicalize import intern_polygons
from .config import CompareConfig, LayerSpec
from .backends.exact import CudaManhattanRectExactComponentBackend, GdstkExactComponentBackend, ManhattanRectExactComponentBackend
from .backends.metal import MetalManhattanRectExactComponentBackend
from .backends.polygon_extract_ctypes import PolygonExtractCtypesUnavailable, extract_gds_polygons_native
from .design_history import render_design_history_entry
from .diff_gds import write_diff_gds
from .exact_diff_markdown import (
    ExactDiffMarkdownWriteResult,
    build_polygon_inventory_delta,
    build_polygon_inventory_delta_from_record_delta,
    write_exact_diff_markdown,
)
from .exact_oracle import DiffFragment, ExactLayerTask, ExactOracleResult, LayerExactDiff, run_full_exact_oracle
from .extract import ExtractionResult, extract_gds_polygons
from .gds_metadata import read_gds_metadata
from .geometry import PolygonBuffer
from .hierarchy import compare_hierarchy
from .multiprocessing_support import mark_portable_process_worker, portable_process_context
from .native import discover_native_capabilities
from .policy import PolicyClassificationResult, classify_exact_differences
from .polygon_components import (
    add_overlapping_context_to_components,
    build_record_delta_surplus_components,
    exact_result_from_component_quads,
)
from .report import CompareReport, build_layer_reports
from .rect_coverage import RectCoverageUnsupported, run_cuda_rectangle_coverage_diff, run_metal_rectangle_coverage_diff
from .semantics import Backend, CompareStatus, EvidenceEngine, exit_code_for_status
from .spatial import SpatialIndex
from .source_edge_diff import exact_result_from_source_edges
from .surplus_coverage import run_canonical_surplus_coverage_probe, run_record_delta_surplus_coverage_probe
from .tiling import CpuTilePrefilterResult, TileConfig, run_cpu_tile_prefilter
from .windowed_exact import WindowedExactResult, run_windowed_exact_oracle
from .windows import CandidateWindow, WindowConfig, build_candidate_windows


@dataclass(frozen=True)
class CompareRunResult:
    status: CompareStatus
    report: CompareReport
    exact_result: ExactOracleResult
    policy_result: PolicyClassificationResult
    prefilter: CpuTilePrefilterResult
    windows: tuple[CandidateWindow, ...]
    artifacts: dict[str, str] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def exit_code(self) -> int:
        return exit_code_for_status(self.status, self.report.failure_policy)


@dataclass(frozen=True)
class _FullExactVerification:
    exact_result: ExactOracleResult
    policy_result: PolicyClassificationResult
    old_input: dict[str, object]
    new_input: dict[str, object]
    selection: dict[str, object]
    layers: tuple[object, ...]
    extract_s: float
    exact_s: float
    exact_component_backend: dict[str, object] = field(default_factory=dict)
    polygon_inventory_delta: object | None = None


@dataclass(frozen=True)
class _FileIdentity:
    old_token: str
    new_token: str
    same: bool
    method: str
    old_size: int
    new_size: int


@dataclass(frozen=True)
class _ExactComponentSelection:
    name: str
    component_partition: bool
    executor: Callable[[tuple[ExactLayerTask, ...]], tuple[LayerExactDiff, ...]] | None
    backend: object | None = None

    def to_json(self) -> dict[str, object]:
        data: dict[str, object] = {
            "requested": self.name,
            "component_partition": self.component_partition,
            "executor_enabled": self.executor is not None,
        }
        if self.backend is not None and hasattr(self.backend, "capabilities"):
            capabilities = self.backend.capabilities()
            data.update(
                {
                    "name": capabilities.name,
                    "deterministic": capabilities.deterministic,
                    "requires_gpu": capabilities.requires_gpu,
                }
            )
        else:
            data.update({"name": "gdstk-full-oracle", "deterministic": True, "requires_gpu": False})
        stats = getattr(self.backend, "stats", None)
        if stats is not None:
            data["stats"] = {
                "accelerated_tasks": stats.accelerated_tasks,
                "fallback_tasks": stats.fallback_tasks,
                "unsupported_tasks": stats.unsupported_tasks,
            }
            if hasattr(stats, "identical_skipped_tasks"):
                data["stats"]["identical_skipped_tasks"] = stats.identical_skipped_tasks
            if hasattr(stats, "cpu_rect_set_tasks"):
                data["stats"]["cpu_rect_set_tasks"] = stats.cpu_rect_set_tasks
            if hasattr(stats, "cuda_rect_set_tasks"):
                data["stats"]["cuda_rect_set_tasks"] = stats.cuda_rect_set_tasks
            if hasattr(stats, "metal_rect_set_tasks"):
                data["stats"]["metal_rect_set_tasks"] = stats.metal_rect_set_tasks
        return data


def compare_gds_files(
    config: CompareConfig,
    *,
    json_report_path: str | Path | None = None,
    diff_gds_path: str | Path | None = None,
    markdown_path: str | Path | None = None,
    exact_diff_markdown_path: str | Path | None = None,
    exact: bool = False,
    verify_exact_after_fast: bool = False,
    audit_cpu_oracle: bool = False,
    use_fast_shortcuts: bool = True,
    use_structural_guard: bool = False,
    use_native_geometry_guard: bool = False,
    exact_component_backend: str = "none",
    command: tuple[str, ...] = (),
) -> CompareRunResult:
    start = perf_counter()
    backend_info = _select_backend(config, exact=exact)
    exact = exact or config.evidence_engine == EvidenceEngine.GDSTK_ORACLE

    old_top_name = config.old_top or config.top
    new_top_name = config.new_top or config.top

    if config.evidence_engine == EvidenceEngine.GPU_POLYGON and not exact:
        return _try_gpu_polygon_evidence_path(
            config,
            backend_info=backend_info,
            old_top_name=old_top_name,
            new_top_name=new_top_name,
            command=command,
            json_report_path=json_report_path,
            markdown_path=markdown_path,
            exact_diff_markdown_path=exact_diff_markdown_path,
            audit_cpu_oracle=audit_cpu_oracle,
            exact_component_backend=exact_component_backend,
            start=start,
        )

    if use_structural_guard and not exact:
        structural_result = _try_native_structural_guard_path(
            config,
            backend_info=backend_info,
            old_top_name=old_top_name,
            new_top_name=new_top_name,
            command=command,
            json_report_path=json_report_path,
            markdown_path=markdown_path,
            exact_diff_markdown_path=exact_diff_markdown_path,
            verify_exact_after_fast=verify_exact_after_fast,
            exact_component_backend=exact_component_backend,
            start=start,
        )
        if structural_result is not None:
            return structural_result

    if (use_native_geometry_guard or (use_fast_shortcuts and _accelerated_fast_path_eligible(config, backend_info))) and not exact:
        geometry_guard_result = _try_native_geometry_guard_path(
            config,
            backend_info=backend_info,
            old_top_name=old_top_name,
            new_top_name=new_top_name,
            command=command,
            json_report_path=json_report_path,
            markdown_path=markdown_path,
            exact_diff_markdown_path=exact_diff_markdown_path,
            verify_exact_after_fast=verify_exact_after_fast,
            exact_component_backend=exact_component_backend,
            start=start,
        )
        if geometry_guard_result is not None:
            return geometry_guard_result

    fast_digest_result = None
    if use_fast_shortcuts and not exact and _accelerated_fast_path_eligible(config, backend_info):
        fast_digest_result = _try_accelerated_byte_identity_fast_path(
            config,
            backend_info=backend_info,
            old_metadata={"path": str(config.old_path), "metadata_mode": "skipped-accelerated-byte-identity"},
            new_metadata={"path": str(config.new_path), "metadata_mode": "skipped-accelerated-byte-identity"},
            old_top_name=old_top_name,
            new_top_name=new_top_name,
            command=command,
            json_report_path=json_report_path,
            markdown_path=markdown_path,
            exact_diff_markdown_path=exact_diff_markdown_path,
            verify_exact_after_fast=verify_exact_after_fast,
            exact_component_backend=exact_component_backend,
            start=start,
        )
    if fast_digest_result is not None:
        return fast_digest_result

    old_metadata = read_gds_metadata(config.old_path)
    new_metadata = read_gds_metadata(config.new_path)

    if use_fast_shortcuts:
        fast_result = _try_cuda_hierarchy_fast_path(
            config,
            backend_info=backend_info,
            old_metadata=old_metadata.to_json(),
            new_metadata=new_metadata.to_json(),
            old_top_name=old_top_name,
            new_top_name=new_top_name,
            exact=exact,
            command=command,
            json_report_path=json_report_path,
            markdown_path=markdown_path,
            exact_diff_markdown_path=exact_diff_markdown_path,
            verify_exact_after_fast=verify_exact_after_fast,
        )
        if fast_result is not None:
            return fast_result

    if not exact and not use_fast_shortcuts and verify_exact_after_fast and config.geometry_cache_dir is not None:
        cached_pair_result = _try_pair_verdict_cache_path(
            config,
            backend_info=backend_info,
            old_metadata=old_metadata.to_json(),
            new_metadata=new_metadata.to_json(),
            old_top_name=old_top_name,
            new_top_name=new_top_name,
            command=command,
            json_report_path=json_report_path,
            markdown_path=markdown_path,
            exact_diff_markdown_path=exact_diff_markdown_path,
            start=start,
            exact_component_backend=exact_component_backend,
        )
        if cached_pair_result is not None:
            return cached_pair_result

    cached_geometry: tuple[CachedGeometry | None, CachedGeometry | None] = (None, None)
    if not exact and config.geometry_cache_dir is not None:
        cache_start = perf_counter()
        old_cached, new_cached = _cached_geometry_pair(
            config,
            old_top_name=old_top_name,
            new_top_name=new_top_name,
        )
        cached_geometry = (old_cached, new_cached)
        old_extraction = old_cached.extraction
        new_extraction = new_cached.extraction
        timings: dict[str, float] = {
            "extract_s": 0.0,
            "geometry_cache_s": perf_counter() - cache_start,
        }
        timings.update(old_cached.to_timing_json("old"))
        timings.update(new_cached.to_timing_json("new"))
    else:
        old_extraction, new_extraction = _extract_pair(
            config,
            old_top_name=old_top_name,
            new_top_name=new_top_name,
        )
        timings = {"extract_s": perf_counter() - start}
    exact_verification_complete = True
    layer_exact_status = "complete"
    warnings: tuple[str, ...] = ()
    if exact:
        exact_oracle_start = perf_counter()
        exact_selection = _select_exact_component_backend(exact_component_backend)
        exact_result = run_full_exact_oracle(
            old_extraction.buffer,
            new_extraction.buffer,
            component_partition=exact_selection.component_partition,
            component_executor=exact_selection.executor,
        )
        timings["exact_oracle_s"] = perf_counter() - exact_oracle_start
        prefilter = CpuTilePrefilterResult(candidate_tiles=())
        windows: tuple[CandidateWindow, ...] = ()
        exact_fallback = {
            "mode": "full",
            "window_count": 0,
            "escalated_layers": [],
            "exact_component_backend": exact_selection.to_json(),
        }
    else:
        canonical_start = perf_counter()
        old_canonical_ids, new_canonical_ids = _shared_canonical_ids(
            old_extraction.buffer,
            new_extraction.buffer,
            old_intern=cached_geometry[0].intern if cached_geometry[0] is not None else None,
            new_intern=cached_geometry[1].intern if cached_geometry[1] is not None else None,
        )
        timings["canonical_s"] = perf_counter() - canonical_start
        tile_config = TileConfig(tile_size=config.tile_size)
        prefilter_start = perf_counter()
        prefilter = _run_tile_prefilter(
            backend_info,
            old_extraction.buffer,
            old_canonical_ids,
            new_extraction.buffer,
            new_canonical_ids,
            tile_config,
        )
        timings["prefilter_s"] = perf_counter() - prefilter_start
        spatial_start = perf_counter()
        old_index = cached_geometry[0].spatial_index if cached_geometry[0] is not None else SpatialIndex.from_buffer(old_extraction.buffer)
        new_index = cached_geometry[1].spatial_index if cached_geometry[1] is not None else SpatialIndex.from_buffer(new_extraction.buffer)
        timings["spatial_index_s"] = perf_counter() - spatial_start
        window_start = perf_counter()
        windows = build_candidate_windows(
            prefilter,
            tile_config,
            old_index,
            new_index,
            WindowConfig(),
        )
        timings["window_build_s"] = perf_counter() - window_start
        if _should_use_layer_exact(backend_info, old_extraction.buffer, new_extraction.buffer, windows):
            candidate_layers = tuple(sorted({window.layer for window in windows}))
            exact_oracle_start = perf_counter()
            exact_result = run_full_exact_oracle(old_extraction.buffer, new_extraction.buffer, layers=set(candidate_layers))
            timings["exact_oracle_s"] = perf_counter() - exact_oracle_start
            windowed = WindowedExactResult(exact_result.layers, escalated_layers=candidate_layers)
            exact_fallback = {
                "mode": "candidate-layer-exact",
                "window_count": len(windows),
                "escalated_layers": [layer.to_json() for layer in candidate_layers],
                "failures": [],
            }
        else:
            exact_oracle_start = perf_counter()
            windowed = run_windowed_exact_oracle(
                old_extraction.buffer,
                new_extraction.buffer,
                windows,
                old_index=old_index,
                new_index=new_index,
            )
            timings["exact_oracle_s"] = perf_counter() - exact_oracle_start
            exact_result = windowed.exact_result
            exact_fallback = {
                "mode": "windowed",
                "window_count": len(windows),
                "escalated_layers": [layer.to_json() for layer in windowed.escalated_layers],
                "failures": [failure.to_json() for failure in windowed.failures],
            }
            if windows:
                exact_verification_complete = False
                layer_exact_status = "windowed"
                warnings = (
                    "Windowed verification is localized and does not claim full-file exact XOR equivalence; "
                    "run exact=True for the full gdstk oracle.",
                )
        if config.geometry_cache_dir is not None:
            pair_cache_start = perf_counter()
            pair_cache_policy = classify_exact_differences(
                exact_result,
                config.allowed_regions,
                failure_policy=config.failure_policy,
            )
            pair_cache_key = write_cached_pair_verdict(
                config,
                old_top_name=old_top_name,
                new_top_name=new_top_name,
                exact_result=exact_result,
                prefilter=prefilter,
                windows=windows,
                exact_fallback=exact_fallback,
                fast_status=pair_cache_policy.status.value,
                exact_verification_complete=exact_verification_complete,
                layer_exact_status=layer_exact_status,
                warnings=warnings,
            )
            timings["pair_verdict_cache_write_s"] = perf_counter() - pair_cache_start
            if pair_cache_key is not None:
                timings["pair_verdict_cache_key"] = pair_cache_key
        if verify_exact_after_fast and not exact_verification_complete:
            geometry_policy_result = classify_exact_differences(
                exact_result,
                config.allowed_regions,
                failure_policy=config.failure_policy,
            )
            geometry_status_s = perf_counter() - start
            geometry_mode = str(exact_fallback.get("mode", "geometry"))
            full_exact_start = perf_counter()
            exact_selection = _select_exact_component_backend(exact_component_backend)
            exact_result = run_full_exact_oracle(
                old_extraction.buffer,
                new_extraction.buffer,
                component_partition=exact_selection.component_partition,
                component_executor=exact_selection.executor,
            )
            timings["full_exact_oracle_s"] = perf_counter() - full_exact_start
            timings["geometry_status_s"] = geometry_status_s
            exact_verification_complete = True
            layer_exact_status = "complete"
            warnings = (
                "Geometry prefilter/windowed verdict was produced first; full exact gdstk verification was run afterward and supplies the polygon-level evidence.",
            )
            exact_fallback = {
                "mode": f"{geometry_mode}-plus-full-exact",
                "geometry_mode": geometry_mode,
                "window_count": exact_fallback.get("window_count", len(windows)),
                "escalated_layers": exact_fallback.get("escalated_layers", []),
                "failures": exact_fallback.get("failures", []),
                "fast_status": geometry_policy_result.status.value,
                "exact_component_backend": exact_selection.to_json(),
            }
    timings["exact_s"] = perf_counter() - start - timings["extract_s"]

    policy_result = classify_exact_differences(
        exact_result,
        config.allowed_regions,
        failure_policy=config.failure_policy,
    )
    artifacts: dict[str, str] = {}
    if diff_gds_path is not None:
        artifacts["diff_gds"] = str(write_diff_gds(policy_result, diff_gds_path).path)
    if markdown_path is not None:
        markdown_text = render_design_history_entry(
            "GDS comparator result",
            policy_result,
            diff_gds_path=artifacts.get("diff_gds"),
            tests_run=("gdsdiff",),
        )
        markdown_path = Path(markdown_path)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(markdown_text, encoding="utf-8")
        artifacts["markdown"] = str(markdown_path)
    if exact_diff_markdown_path is not None:
        inventory_delta_start = perf_counter()
        polygon_inventory_delta = (
            build_polygon_inventory_delta(old_extraction.buffer, new_extraction.buffer)
            if exact_verification_complete
            else None
        )
        timings["polygon_inventory_delta_s"] = perf_counter() - inventory_delta_start
        artifacts["exact_diff_markdown"] = str(
            write_exact_diff_markdown(
                policy_result,
                exact_diff_markdown_path,
                title="GDS comparator exact diff",
                source_mode=exact_fallback["mode"],
                exact_verification_complete=exact_verification_complete,
                polygon_inventory_delta=polygon_inventory_delta,
            ).path
        )

    layer_reports = build_layer_reports(
        exact_result,
        policy_result=policy_result,
        presence_by_layer=_presence_by_layer(old_extraction.buffer, new_extraction.buffer),
        polygon_counts_by_layer=_polygon_counts_by_layer(old_extraction.buffer, new_extraction.buffer),
        candidate_tile_counts_by_layer=_candidate_tile_counts(prefilter),
        exact_window_counts_by_layer=_window_counts(windows),
        exact_status=layer_exact_status,
    )
    timings["total_s"] = perf_counter() - start
    report = CompareReport(
        status=policy_result.status,
        exact_verification_complete=exact_verification_complete,
        old_input=_input_json(old_extraction, old_metadata.to_json()),
        new_input=_input_json(new_extraction, new_metadata.to_json()),
        semantics=config.to_json_dict(),
        selection={"old_top": old_extraction.top_name, "new_top": new_extraction.top_name, "layers": config.to_json_dict()["layers"]},
        backend=backend_info
        | {
            "exact_mode": exact,
            "fast_shortcuts_enabled": use_fast_shortcuts,
            "geometry_cache_enabled": config.geometry_cache_dir is not None,
        },
        prefilter=prefilter.to_json(),
        exact_fallback=exact_fallback,
        layers=layer_reports,
        allowed_policy=policy_result.effective_policy,
        unexpected_summary={
            "expected_area_dbu2": policy_result.to_json()["expected_area_dbu2"],
            "unexpected_area_dbu2": policy_result.to_json()["unexpected_area_dbu2"],
        },
        timings=timings,
        warnings=warnings,
        artifacts=artifacts,
        command=command,
        failure_policy=config.failure_policy,
    )
    if json_report_path is not None:
        artifacts["json_report"] = str(json_report_path)
        report = _report_with_artifacts(report, artifacts)
        report.write_json(json_report_path)

    return CompareRunResult(
        status=policy_result.status,
        report=report,
        exact_result=exact_result,
        policy_result=policy_result,
        prefilter=prefilter,
        windows=windows,
        artifacts=artifacts,
        timings=timings,
    )


def compare_gds_paths(old_path: str | Path, new_path: str | Path, **kwargs) -> CompareRunResult:
    config_kwargs = {key: value for key, value in kwargs.items() if key in CompareConfig.__dataclass_fields__}
    run_kwargs = {key: value for key, value in kwargs.items() if key not in config_kwargs}
    return compare_gds_files(CompareConfig(old_path=old_path, new_path=new_path, **config_kwargs), **run_kwargs)


def _extract_pair(
    config: CompareConfig,
    *,
    old_top_name: str | None,
    new_top_name: str | None,
) -> tuple[ExtractionResult, ExtractionResult]:
    old_args = (config.old_path, old_top_name, config.layers, config.dbu, config.path_policy)
    new_args = (config.new_path, new_top_name, config.layers, config.dbu, config.path_policy)
    if _should_extract_pair_in_parallel(config):
        try:
            with ProcessPoolExecutor(
                max_workers=2,
                mp_context=portable_process_context(),
                initializer=mark_portable_process_worker,
            ) as executor:
                old_future = executor.submit(_extract_one, old_args)
                new_future = executor.submit(_extract_one, new_args)
                return old_future.result(), new_future.result()
        except (BrokenProcessPool, OSError):
            pass
    return _extract_one(old_args), _extract_one(new_args)


def _cached_geometry_pair(
    config: CompareConfig,
    *,
    old_top_name: str | None,
    new_top_name: str | None,
) -> tuple[CachedGeometry, CachedGeometry]:
    if config.geometry_cache_dir is None:
        raise ValueError("geometry cache directory is not configured")
    old_args = (config.geometry_cache_dir, config.old_path, old_top_name, config.layers, config.dbu, config.path_policy)
    new_args = (config.geometry_cache_dir, config.new_path, new_top_name, config.layers, config.dbu, config.path_policy)
    if _should_extract_pair_in_parallel(config):
        try:
            with ProcessPoolExecutor(
                max_workers=2,
                mp_context=portable_process_context(),
                initializer=mark_portable_process_worker,
            ) as executor:
                old_future = executor.submit(_cached_geometry_one, old_args)
                new_future = executor.submit(_cached_geometry_one, new_args)
                return old_future.result(), new_future.result()
        except (BrokenProcessPool, OSError):
            pass
    return _cached_geometry_one(old_args), _cached_geometry_one(new_args)


def _cached_geometry_one(args) -> CachedGeometry:
    cache_dir, path, top_name, layers, dbu, path_policy = args
    return load_or_build_cached_geometry(
        cache_dir=cache_dir,
        path=path,
        top_name=top_name,
        layers=layers,
        meters_per_dbu=dbu,
        path_policy=path_policy,
    )


def _extract_one(args) -> ExtractionResult:
    path, top_name, layers, dbu, path_policy = args
    return extract_gds_polygons(
        path,
        top_name=top_name,
        layers=layers,
        meters_per_dbu=dbu,
        path_policy=path_policy,
    )


def _should_extract_pair_in_parallel(config: CompareConfig) -> bool:
    try:
        total_size = Path(config.old_path).stat().st_size + Path(config.new_path).stat().st_size
    except OSError:
        return False
    return total_size >= _parallel_extract_pair_min_bytes()


def _parallel_extract_pair_min_bytes() -> int:
    return _positive_int_env("GDS_GPU_COMPARE_PARALLEL_EXTRACT_PAIR_MIN_BYTES", 1_000_000)


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, ""))
    except ValueError:
        return default
    return value if value > 0 else default


def _try_native_structural_guard_path(
    config: CompareConfig,
    *,
    backend_info: dict[str, object],
    old_top_name: str | None,
    new_top_name: str | None,
    command: tuple[str, ...],
    json_report_path: str | Path | None,
    markdown_path: str | Path | None,
    exact_diff_markdown_path: str | Path | None,
    verify_exact_after_fast: bool,
    exact_component_backend: str,
    start: float,
) -> CompareRunResult | None:
    if config.allowed_regions or config.layers != "all":
        return None
    try:
        from .backends.structural_ctypes import summarize_gds_structure_native

        structural_start = perf_counter()
        old_summary = summarize_gds_structure_native(config.old_path, top_name=old_top_name, counts_only=True)
        new_summary = summarize_gds_structure_native(config.new_path, top_name=new_top_name, counts_only=True)
        if old_summary.flattened_shape_count == new_summary.flattened_shape_count:
            old_summary = summarize_gds_structure_native(config.old_path, top_name=old_top_name)
            new_summary = summarize_gds_structure_native(config.new_path, top_name=new_top_name)
        structural_s = perf_counter() - structural_start
    except Exception:
        return None
    counts_match = old_summary.flattened_shape_count == new_summary.flattened_shape_count
    hashes_match = old_summary.structural_hash == new_summary.structural_hash
    if counts_match and hashes_match:
        return None
    structural_mismatch_kind = "hash" if counts_match else "count"

    fragment = DiffFragment(points=((0, 0), (1, 0), (1, 1), (0, 1)), bbox=(0, 0, 1, 1), twice_area=2)
    exact_result = ExactOracleResult((LayerExactDiff(layer=LayerSpec(0, 0), new_minus_old=(fragment,), xor=(fragment,)),))
    policy_result = classify_exact_differences(exact_result, failure_policy=config.failure_policy)
    fast_status = policy_result.status
    exact_complete = False
    verification: _FullExactVerification | None = None
    if verify_exact_after_fast:
        verification = _run_full_exact_verification(config, exact_component_backend=exact_component_backend)
        exact_result = verification.exact_result
        policy_result = verification.policy_result
        exact_complete = True

    timings = {
        "extract_s": 0.0,
        "exact_s": structural_s,
        "fast_status_s": structural_s,
        "structural_guard_s": structural_s,
    }
    old_input = {"path": str(config.old_path), "selected_top": old_top_name, "structural_summary": old_summary.to_json()}
    new_input = {"path": str(config.new_path), "selected_top": new_top_name, "structural_summary": new_summary.to_json()}
    selection = {"old_top": old_top_name, "new_top": new_top_name, "layers": "all"}
    layers = build_layer_reports(exact_result, policy_result=policy_result)
    if verification is not None:
        timings.update(
            {
                "extract_s": verification.extract_s,
                "exact_s": verification.exact_s,
                "exact_verification_s": verification.extract_s + verification.exact_s,
            }
        )
        old_input = verification.old_input
        new_input = verification.new_input
        selection = verification.selection
        layers = verification.layers

    prefilter = CpuTilePrefilterResult(candidate_tiles=())
    windows: tuple[CandidateWindow, ...] = ()
    artifacts: dict[str, str] = {}
    exact_fallback_mode = "structural-count-fast-plus-full-exact" if verification is not None else "structural-count-fail-closed"
    warnings = (
        "Native structural guard used flattened shape counts for a fail-closed fast status; full exact gdstk verification was run afterward and supplies polygon-level evidence."
        if verification is not None
        else "Native structural guard used flattened shape counts for a fail-closed fast status; run exact verification for polygon-level evidence."
    )
    exact_fallback = {
        "mode": exact_fallback_mode,
        "window_count": 0,
        "escalated_layers": [],
        "failures": [],
        "fast_status": fast_status.value,
        "old_flattened_shape_count": old_summary.flattened_shape_count,
        "new_flattened_shape_count": new_summary.flattened_shape_count,
        "old_structural_hash": f"{old_summary.structural_hash:016x}",
        "new_structural_hash": f"{new_summary.structural_hash:016x}",
        "structural_mismatch_kind": structural_mismatch_kind,
    }
    if verification is not None:
        exact_fallback["exact_component_backend"] = verification.exact_component_backend
    if markdown_path is not None:
        markdown_path = Path(markdown_path)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(
            render_design_history_entry(
                "GDS comparator structural-guard result",
                policy_result,
                tests_run=("gdsdiff structural-guard",),
                notes=warnings,
            ),
            encoding="utf-8",
        )
        artifacts["markdown"] = str(markdown_path)
    if exact_diff_markdown_path is not None:
        artifacts["exact_diff_markdown"] = str(
            write_exact_diff_markdown(
                policy_result,
                exact_diff_markdown_path,
                title="GDS comparator structural-guard exact diff",
                source_mode=exact_fallback_mode,
                exact_verification_complete=exact_complete,
                polygon_inventory_delta=verification.polygon_inventory_delta if verification is not None else None,
            ).path
        )
    timings["total_s"] = perf_counter() - start
    report = CompareReport(
        status=policy_result.status,
        exact_verification_complete=exact_complete,
        old_input=old_input,
        new_input=new_input,
        semantics=config.to_json_dict(),
        selection=selection,
        backend=backend_info
        | {
            "exact_mode": False,
            "fast_shortcuts_enabled": True,
            "structural_guard": True,
            "exact_verification_after_fast": verification is not None,
        },
        prefilter=prefilter.to_json(),
        exact_fallback=exact_fallback,
        layers=layers,
        allowed_policy=policy_result.effective_policy,
        unexpected_summary={
            "expected_area_dbu2": policy_result.to_json()["expected_area_dbu2"],
            "unexpected_area_dbu2": policy_result.to_json()["unexpected_area_dbu2"],
        },
        timings=timings,
        warnings=warnings,
        artifacts=artifacts,
        command=command,
        failure_policy=config.failure_policy,
    )
    if json_report_path is not None:
        artifacts["json_report"] = str(json_report_path)
        report = _report_with_artifacts(report, artifacts)
        report.write_json(json_report_path)
    return CompareRunResult(
        status=policy_result.status,
        report=report,
        exact_result=exact_result,
        policy_result=policy_result,
        prefilter=prefilter,
        windows=windows,
        artifacts=artifacts,
        timings=timings,
    )


def _try_native_geometry_guard_path(
    config: CompareConfig,
    *,
    backend_info: dict[str, object],
    old_top_name: str | None,
    new_top_name: str | None,
    command: tuple[str, ...],
    json_report_path: str | Path | None,
    markdown_path: str | Path | None,
    exact_diff_markdown_path: str | Path | None,
    verify_exact_after_fast: bool,
    exact_component_backend: str,
    start: float,
) -> CompareRunResult | None:
    if config.allowed_regions or config.layers != "all":
        return None
    try:
        from .backends.geometry_ctypes import compare_gds_geometry_native

        guard_start = perf_counter()
        native_compare = compare_gds_geometry_native(
            config.old_path,
            config.new_path,
            old_top_name=old_top_name,
            new_top_name=new_top_name,
        )
        guard_s = perf_counter() - guard_start
    except Exception:
        return None

    old_summary = native_compare.old_summary
    new_summary = native_compare.new_summary
    if native_compare.summaries_match:
        return None

    if old_summary.flattened_shape_count != new_summary.flattened_shape_count:
        mismatch_kind = "count"
    elif _geometry_layer_json(old_summary) != _geometry_layer_json(new_summary):
        mismatch_kind = "layer-summary"
    else:
        mismatch_kind = "hash"

    fragment = DiffFragment(points=((0, 0), (1, 0), (1, 1), (0, 1)), bbox=(0, 0, 1, 1), twice_area=2)
    exact_result = ExactOracleResult((LayerExactDiff(layer=LayerSpec(0, 0), new_minus_old=(fragment,), xor=(fragment,)),))
    policy_result = classify_exact_differences(exact_result, failure_policy=config.failure_policy)
    fast_status = policy_result.status
    exact_complete = False
    verification: _FullExactVerification | None = None
    if verify_exact_after_fast:
        verification = _run_full_exact_verification(config, exact_component_backend=exact_component_backend)
        exact_result = verification.exact_result
        policy_result = verification.policy_result
        exact_complete = True

    timings = {
        "extract_s": 0.0,
        "exact_s": guard_s,
        "fast_status_s": guard_s,
        "native_geometry_guard_s": guard_s,
    }
    old_input = {"path": str(config.old_path), "selected_top": old_top_name, "native_geometry_summary": old_summary.to_json()}
    new_input = {"path": str(config.new_path), "selected_top": new_top_name, "native_geometry_summary": new_summary.to_json()}
    selection = {"old_top": old_top_name, "new_top": new_top_name, "layers": "all"}
    layers = build_layer_reports(exact_result, policy_result=policy_result)
    if verification is not None:
        timings.update(
            {
                "extract_s": verification.extract_s,
                "exact_s": verification.exact_s,
                "exact_verification_s": verification.extract_s + verification.exact_s,
            }
        )
        old_input = verification.old_input
        new_input = verification.new_input
        selection = verification.selection
        layers = verification.layers

    prefilter = CpuTilePrefilterResult(candidate_tiles=())
    windows: tuple[CandidateWindow, ...] = ()
    artifacts: dict[str, str] = {}
    exact_fallback_mode = "native-geometry-fast-plus-full-exact" if verification is not None else "native-geometry-fail-closed"
    warnings = (
        "Native geometry guard used custom GDS parsing for a fail-closed fast status; full exact gdstk verification was run afterward and supplies polygon-level evidence."
        if verification is not None
        else "Native geometry guard used custom GDS parsing for a fail-closed fast status; run exact verification for polygon-level evidence."
    )
    exact_fallback = {
        "mode": exact_fallback_mode,
        "window_count": 0,
        "escalated_layers": [],
        "failures": [],
        "fast_status": fast_status.value,
        "old_flattened_shape_count": old_summary.flattened_shape_count,
        "new_flattened_shape_count": new_summary.flattened_shape_count,
        "old_geometry_hash": f"{old_summary.geometry_hash:016x}",
        "new_geometry_hash": f"{new_summary.geometry_hash:016x}",
        "geometry_mismatch_kind": mismatch_kind,
        "old_layer_summary": _geometry_layer_json(old_summary),
        "new_layer_summary": _geometry_layer_json(new_summary),
    }
    if verification is not None:
        exact_fallback["exact_component_backend"] = verification.exact_component_backend
    if markdown_path is not None:
        markdown_path = Path(markdown_path)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(
            render_design_history_entry(
                "GDS comparator native-geometry-guard result",
                policy_result,
                tests_run=("gdsdiff native-geometry-guard",),
                notes=warnings,
            ),
            encoding="utf-8",
        )
        artifacts["markdown"] = str(markdown_path)
    if exact_diff_markdown_path is not None:
        artifacts["exact_diff_markdown"] = str(
            write_exact_diff_markdown(
                policy_result,
                exact_diff_markdown_path,
                title="GDS comparator native-geometry-guard exact diff",
                source_mode=exact_fallback_mode,
                exact_verification_complete=exact_complete,
                polygon_inventory_delta=verification.polygon_inventory_delta if verification is not None else None,
            ).path
        )
    timings["total_s"] = perf_counter() - start
    report = CompareReport(
        status=policy_result.status,
        exact_verification_complete=exact_complete,
        old_input=old_input,
        new_input=new_input,
        semantics=config.to_json_dict(),
        selection=selection,
        backend=backend_info
        | {
            "exact_mode": False,
            "fast_shortcuts_enabled": True,
            "native_geometry_guard": True,
            "exact_verification_after_fast": verification is not None,
        },
        prefilter=prefilter.to_json(),
        exact_fallback=exact_fallback,
        layers=layers,
        allowed_policy=policy_result.effective_policy,
        unexpected_summary={
            "expected_area_dbu2": policy_result.to_json()["expected_area_dbu2"],
            "unexpected_area_dbu2": policy_result.to_json()["unexpected_area_dbu2"],
        },
        timings=timings,
        warnings=warnings,
        artifacts=artifacts,
        command=command,
        failure_policy=config.failure_policy,
    )
    if json_report_path is not None:
        artifacts["json_report"] = str(json_report_path)
        report = _report_with_artifacts(report, artifacts)
        report.write_json(json_report_path)
    return CompareRunResult(
        status=policy_result.status,
        report=report,
        exact_result=exact_result,
        policy_result=policy_result,
        prefilter=prefilter,
        windows=windows,
        artifacts=artifacts,
        timings=timings,
    )


def _try_gpu_polygon_evidence_path(
    config: CompareConfig,
    *,
    backend_info: dict[str, object],
    old_top_name: str | None,
    new_top_name: str | None,
    command: tuple[str, ...],
    json_report_path: str | Path | None,
    markdown_path: str | Path | None,
    exact_diff_markdown_path: str | Path | None,
    audit_cpu_oracle: bool,
    exact_component_backend: str,
    start: float,
) -> CompareRunResult:
    artifacts: dict[str, str] = {}
    prefilter = CpuTilePrefilterResult(candidate_tiles=())
    windows: tuple[CandidateWindow, ...] = ()
    record_delta = None
    coverage_result = None
    surplus_result = None
    surplus_probe_source = None
    polygon_fragment_exact_result = None
    polygon_component_area = None
    polygon_inventory_delta = None
    source_edge_cpu_fallback = None
    source_edge_cpu_verification = None
    native_parse_cpu_verification = None

    cpu_neutral_fallback = backend_info.get("name") == Backend.CPU.value
    gpu_runtime_available = bool(backend_info.get("cuda_ctypes_available") or backend_info.get("metal_ctypes_available"))
    if cpu_neutral_fallback or (audit_cpu_oracle and not gpu_runtime_available):
        audit_start = perf_counter()
        audit_backend = _default_cpu_audit_component_backend(exact_component_backend)
        verification = _run_full_exact_verification(
            config,
            exact_component_backend=audit_backend,
            force_small_component_partition=True,
        )
        timings = {
            "native_parse_s": 0.0,
            "h2d_upload_s": 0.0,
            "gpu_broadphase_s": 0.0,
            "gpu_xor_s": 0.0,
            "gpu_record_emit_s": 0.0,
            "markdown_format_s": 0.0,
            "cpu_oracle_audit_s": perf_counter() - audit_start,
            "extract_s": verification.extract_s,
            "exact_s": verification.exact_s,
        }
        policy_result = verification.policy_result
        exact_result = verification.exact_result
        polygon_inventory_delta = verification.polygon_inventory_delta
        exact_complete = True
        warnings = (
            (
                "No suitable GPU backend was selected; CPU gdstk oracle produced complete neutral exact evidence."
                if cpu_neutral_fallback
                else "Requested gpu-polygon evidence engine is not implemented yet; CPU gdstk oracle audit was run explicitly via --audit-cpu-oracle."
            ),
        )
        exact_fallback = {
            "mode": (
                "cpu-gdstk-oracle-neutral-fallback"
                if cpu_neutral_fallback
                else "gpu-polygon-unavailable-plus-cpu-oracle-audit"
            ),
            "engine": "gdstk-oracle" if cpu_neutral_fallback else "gpu-polygon",
            "engine_status": "complete" if cpu_neutral_fallback else "not-implemented",
            "audit_cpu_oracle": not cpu_neutral_fallback,
            "window_count": 0,
            "escalated_layers": [],
            "failures": []
            if cpu_neutral_fallback
            else [
                {
                    "reason": "gpu-polygon exact numeric diff record engine is not implemented yet",
                }
            ],
            "exact_component_backend": verification.exact_component_backend,
        }
        old_input = verification.old_input
        new_input = verification.new_input
        selection = verification.selection
        layers = verification.layers
        if exact_diff_markdown_path is not None:
            markdown_start = perf_counter()
            markdown_written = write_exact_diff_markdown(
                policy_result,
                exact_diff_markdown_path,
                title=(
                    "GDS comparator CPU fallback exact diff"
                    if cpu_neutral_fallback
                    else "GDS comparator CPU-audited gpu-polygon exact diff"
                ),
                source_mode=exact_fallback["mode"],
                exact_verification_complete=True,
                polygon_inventory_delta=polygon_inventory_delta,
            )
            _record_exact_markdown_artifacts(artifacts, markdown_written)
            timings["markdown_format_s"] = perf_counter() - markdown_start
    else:
        parse_start = perf_counter()
        parse_failures: list[dict[str, str]] = []
        record_delta_failures: list[dict[str, str]] = []
        coverage_failures: list[dict[str, str]] = []
        component_area_failures: list[dict[str, str]] = []
        old_native = None
        new_native = None
        native_cache_timings: dict[str, object] = {}
        try:
            old_native, new_native, native_cache_timings = _gpu_polygon_native_inputs(
                config,
                old_top_name=old_top_name,
                new_top_name=new_top_name,
            )
        except PolygonExtractCtypesUnavailable as exc:
            common_top = _resolve_single_common_top_after_native_parse_failure(config, str(exc), old_top_name, new_top_name)
            if common_top is None:
                parse_failures.append({"stage": "native-parse", "reason": str(exc)})
            else:
                old_top_name = common_top
                new_top_name = common_top
                try:
                    old_native, new_native, native_cache_timings = _gpu_polygon_native_inputs(
                        config,
                        old_top_name=old_top_name,
                        new_top_name=new_top_name,
                    )
                except PolygonExtractCtypesUnavailable as retry_exc:
                    parse_failures.append({"stage": "native-parse", "reason": str(retry_exc)})
        native_parse_s = perf_counter() - parse_start
        exact_result = ExactOracleResult(())
        policy_result = classify_exact_differences(exact_result, config.allowed_regions, failure_policy=config.failure_policy)
        exact_complete = False
        timings = {
            "native_parse_s": native_parse_s,
            "h2d_upload_s": 0.0,
            "gpu_broadphase_s": 0.0,
            "gpu_xor_s": 0.0,
            "gpu_record_emit_s": 0.0,
            "markdown_format_s": 0.0,
            "cpu_oracle_audit_s": 0.0,
            "gpu_source_record_delta_s": 0.0,
            "canonical_surplus_coverage_s": 0.0,
            "record_delta_surplus_coverage_s": 0.0,
            "polygon_component_build_s": 0.0,
            "polygon_context_build_s": 0.0,
            "gpu_context_index_s": 0.0,
            "gpu_edge_cancel_split_s": 0.0,
            "gpu_polygon_component_area_s": 0.0,
            "gpu_polygon_component_fragment_s": 0.0,
            "boundary_loop_assembly_s": 0.0,
            "cuda_release_s": 0.0,
            "metal_release_s": 0.0,
            "polygon_inventory_delta_s": 0.0,
        }
        timings.update(native_cache_timings)
        gpu_name = backend_info.get("name")
        gpu_rect_available = (
            (gpu_name == Backend.CUDA.value and backend_info.get("cuda_ctypes_available"))
            or (gpu_name == Backend.METAL.value and backend_info.get("metal_ctypes_available"))
        )
        if parse_failures and audit_cpu_oracle:
            audit_start = perf_counter()
            audit_backend = _default_cpu_audit_component_backend(exact_component_backend)
            native_parse_cpu_verification = _run_full_exact_verification(
                config,
                exact_component_backend=audit_backend,
                force_small_component_partition=True,
            )
            timings["cpu_oracle_audit_s"] = perf_counter() - audit_start
            timings["extract_s"] = native_parse_cpu_verification.extract_s
            timings["exact_s"] = native_parse_cpu_verification.exact_s
            exact_result = native_parse_cpu_verification.exact_result
            policy_result = native_parse_cpu_verification.policy_result
            layers = native_parse_cpu_verification.layers
            polygon_inventory_delta = native_parse_cpu_verification.polygon_inventory_delta
            exact_complete = True
        if not parse_failures and old_native is not None and new_native is not None and gpu_rect_available:
            if _native_buffers_rectangle_coverage_eligible(old_native.buffer, new_native.buffer):
                try:
                    coverage_result = (
                        run_metal_rectangle_coverage_diff(old_native.buffer, new_native.buffer)
                        if gpu_name == Backend.METAL.value
                        else run_cuda_rectangle_coverage_diff(old_native.buffer, new_native.buffer)
                    )
                    timings["h2d_upload_s"] = coverage_result.stats.h2d_upload_s
                    timings["gpu_xor_s"] = coverage_result.stats.gpu_xor_s
                    timings["gpu_record_emit_s"] = coverage_result.stats.gpu_record_emit_s
                except RectCoverageUnsupported as exc:
                    coverage_failures.append({"stage": "gpu-coverage-xor", "reason": str(exc)})
            else:
                coverage_failures.append(
                    {
                        "stage": "gpu-coverage-xor",
                        "reason": "skipped rectangle coverage because native polygon vertex counts include non-rectangles",
                    }
                )
            if coverage_result is None and gpu_name in {Backend.CUDA.value, Backend.METAL.value}:
                try:
                    from .backends.cuda_ctypes import CudaCtypesUnavailable, run_cuda_polygon_record_delta
                    from .backends.metal_ctypes import MetalCtypesUnavailable, run_metal_polygon_record_delta_from_hashes

                    record_start = perf_counter()
                    if gpu_name == Backend.METAL.value:
                        record_delta = run_metal_polygon_record_delta_from_hashes(old_native.buffer, new_native.buffer)
                    else:
                        record_delta = run_cuda_polygon_record_delta(old_native.buffer, new_native.buffer)
                    timings["gpu_source_record_delta_s"] = perf_counter() - record_start
                except (CudaCtypesUnavailable, MetalCtypesUnavailable) as record_exc:
                    record_delta_failures.append({"stage": "gpu-record-delta", "reason": str(record_exc)})
                if record_delta is not None:
                    surplus_start = perf_counter()
                    surplus_result = run_record_delta_surplus_coverage_probe(old_native.buffer, new_native.buffer, record_delta)
                    surplus_probe_source = f"{_record_delta_accelerator(record_delta)}-record-delta"
                    timings["record_delta_surplus_coverage_s"] = perf_counter() - surplus_start
                    component_area_start = perf_counter()
                    component_build_start = perf_counter()
                    component_build = build_record_delta_surplus_components(old_native.buffer, new_native.buffer, record_delta)
                    timings["polygon_component_build_s"] = perf_counter() - component_build_start
                    context_build_start = perf_counter()
                    context_build = add_overlapping_context_to_components(
                        old_native.buffer,
                        new_native.buffer,
                        component_build.components,
                        preferred_accelerator=_record_delta_accelerator(record_delta),
                    )
                    timings["polygon_context_build_s"] = perf_counter() - context_build_start
                    if context_build.stats.context_index_backend.startswith(("cuda", "metal")):
                        timings["gpu_context_index_s"] = context_build.stats.context_index_s
                    assembly_start = perf_counter()
                    source_edge_result = exact_result_from_source_edges(context_build.components)
                    polygon_fragment_exact_result = source_edge_result.exact_result
                    timings["boundary_loop_assembly_s"] = perf_counter() - assembly_start
                    timings["gpu_edge_cancel_split_s"] = source_edge_result.stats.layer_merge_cuda_edge_cancel_s
                    timings["gpu_polygon_component_area_s"] = perf_counter() - component_area_start
                    source_edge_old_minus = sum(
                        layer.old_minus_new_twice_area for layer in polygon_fragment_exact_result.layers
                    )
                    source_edge_new_minus = sum(
                        layer.new_minus_old_twice_area for layer in polygon_fragment_exact_result.layers
                    )
                    if source_edge_result.stats.nested_direction_fragment_component_count:
                        fallback_start = perf_counter()
                        source_edge_cpu_verification = _run_full_exact_verification(
                            config,
                            exact_component_backend="gdstk",
                        )
                        timings["source_edge_cpu_oracle_fallback_s"] = perf_counter() - fallback_start
                        source_edge_cpu_fallback = {
                            "trigger": "nested-direction-fragments",
                            "reason": (
                                "source-edge loops contain nested same-direction fragments, "
                                "which require hole-aware Boolean semantics"
                            ),
                            "nested_direction_fragment_component_count": (
                                source_edge_result.stats.nested_direction_fragment_component_count
                            ),
                            "original_vs_cpu_oracle": _exact_result_audit_summary(
                                polygon_fragment_exact_result,
                                source_edge_cpu_verification.exact_result,
                            ),
                            "exact_component_backend": source_edge_cpu_verification.exact_component_backend,
                        }
                        polygon_fragment_exact_result = source_edge_cpu_verification.exact_result
                        source_edge_old_minus = sum(
                            layer.old_minus_new_twice_area for layer in polygon_fragment_exact_result.layers
                        )
                        source_edge_new_minus = sum(
                            layer.new_minus_old_twice_area for layer in polygon_fragment_exact_result.layers
                        )
                    polygon_component_area = {
                        "component_count": len(component_build.components),
                        "context_component_count": len(context_build.components),
                        "context_max_fragments_per_component": context_build.stats.max_fragments_per_component,
                        "context_max_vertices_per_component": context_build.stats.max_vertices_per_component,
                        "context_index_backend": context_build.stats.context_index_backend,
                        "context_index_s": context_build.stats.context_index_s,
                        "context_pair_count": context_build.stats.context_pair_count,
                        "context_cuda_kernel_count": context_build.stats.context_cuda_kernel_count,
                        "context_cuda_event_elapsed_s": context_build.stats.context_cuda_event_elapsed_s,
                        "context_cuda_transferred_bytes": context_build.stats.context_cuda_transferred_bytes,
                        "context_metal_kernel_count": context_build.stats.context_metal_kernel_count,
                        "context_metal_event_elapsed_s": context_build.stats.context_metal_event_elapsed_s,
                        "context_metal_transferred_bytes": context_build.stats.context_metal_transferred_bytes,
                        "emitted_trapezoid_fragment_count": 0,
                        "cuda_trapezoid_old_minus_new_twice_area": None,
                        "cuda_trapezoid_new_minus_old_twice_area": None,
                        "cuda_trapezoid_xor_twice_area": None,
                        "old_minus_new_twice_area": source_edge_old_minus,
                        "new_minus_old_twice_area": source_edge_new_minus,
                        "xor_twice_area": source_edge_old_minus + source_edge_new_minus,
                        "uses_overlapping_context": True,
                        "gpu_trapezoid_fragments_available": False,
                        "gpu_source_edge_records_available": True,
                        "merged_boundary_loop_records_available": True,
                        "source_edge_cpu_oracle_fallback": source_edge_cpu_fallback,
                        "boundary_loop_assembly_backend": (
                            "mixed-gpu-boundary-loop-records-plus-host-fallback"
                            if source_edge_result.stats.cuda_loop_component_count and source_edge_result.stats.metal_loop_component_count
                            else
                            "metal-boundary-loop-records-plus-host-fallback"
                            if source_edge_result.stats.metal_loop_component_count
                            else
                            "cuda-boundary-loop-records-plus-host-fallback"
                            if source_edge_result.stats.cuda_loop_component_count
                            else
                            "native-boundary-loop-ctypes-plus-python-fallback"
                            if source_edge_result.stats.native_loop_component_count
                            else "cpu-source-edge-arrangement-prototype"
                        ),
                        "exact_boundary_loops_available": True,
                        "source_edge_arrangement": source_edge_result.stats.to_json(),
                    }
                else:
                    surplus_start = perf_counter()
                    surplus_result = run_canonical_surplus_coverage_probe(old_native.buffer, new_native.buffer)
                    surplus_probe_source = "cpu-canonical"
                    timings["canonical_surplus_coverage_s"] = perf_counter() - surplus_start
        if coverage_result is not None:
            exact_result = coverage_result.exact_result
            policy_result = classify_exact_differences(exact_result, config.allowed_regions, failure_policy=config.failure_policy)
            exact_complete = True
            layers = build_layer_reports(
                exact_result,
                policy_result=policy_result,
                presence_by_layer=_presence_by_layer(old_native.buffer, new_native.buffer),
                polygon_counts_by_layer=_polygon_counts_by_layer(old_native.buffer, new_native.buffer),
                exact_status="complete",
            )
        elif polygon_fragment_exact_result is not None:
            exact_result = polygon_fragment_exact_result
            policy_result = classify_exact_differences(exact_result, config.allowed_regions, failure_policy=config.failure_policy)
            exact_complete = True
            layers = build_layer_reports(
                exact_result,
                policy_result=policy_result,
                presence_by_layer=_presence_by_layer(old_native.buffer, new_native.buffer),
                polygon_counts_by_layer=_polygon_counts_by_layer(old_native.buffer, new_native.buffer),
                exact_status="gpu-polygon-source-edge-records",
            )
        elif surplus_result is not None:
            exact_result = surplus_result.exact_result
            policy_result = classify_exact_differences(exact_result, config.allowed_regions, failure_policy=config.failure_policy)
            exact_complete = surplus_result.stats.coverage_exact_proven
            layers = build_layer_reports(
                exact_result,
                policy_result=policy_result,
                presence_by_layer=_presence_by_layer(old_native.buffer, new_native.buffer),
                polygon_counts_by_layer=_polygon_counts_by_layer(old_native.buffer, new_native.buffer),
                exact_status="complete" if surplus_result.stats.coverage_exact_proven else "canonical-source-surplus-probe",
            )
        else:
            layers = ()
        if old_native is not None and new_native is not None:
            inventory_start = perf_counter()
            if record_delta is not None:
                polygon_inventory_delta = build_polygon_inventory_delta_from_record_delta(
                    old_native.buffer,
                    new_native.buffer,
                    record_delta,
                )
                timings["polygon_inventory_delta_source"] = f"{_record_delta_accelerator(record_delta)}-record-delta"
            else:
                polygon_inventory_delta = build_polygon_inventory_delta(old_native.buffer, new_native.buffer)
                timings["polygon_inventory_delta_source"] = "cpu-full-canonical"
            timings["polygon_inventory_delta_s"] = perf_counter() - inventory_start
        coverage_accelerator = coverage_result.stats.accelerator if coverage_result is not None else None
        record_delta_accelerator = _record_delta_accelerator(record_delta) if record_delta is not None else None
        warning_text = (
            f"{str(coverage_accelerator).upper()} rectangle coverage engine produced exact gpu-polygon evidence for rectangle-only input; arbitrary non-rectangle polygon XOR is still unsupported."
            if coverage_result is not None
            else f"{str(record_delta_accelerator).upper()} source-edge evidence detected hole-like topology; final exact evidence was corrected with the CPU gdstk oracle while GPU source-record and context evidence were retained."
            if polygon_fragment_exact_result is not None and source_edge_cpu_fallback is not None
            else f"{str(record_delta_accelerator).upper()} source-edge evidence path produced exact boundary records with GPU source-record deltas and GPU context indexing where supported; CPU retains loop stitching, hole handling, and Markdown formatting."
            if polygon_fragment_exact_result is not None
            else f"{str(gpu_name).upper()} native polygon extraction failed; CPU gdstk oracle produced complete exact evidence for this audit run."
            if native_parse_cpu_verification is not None
            else "Canonical source-surplus probe produced exact directional evidence because changed source polygons are independent; arbitrary overlapping polygon XOR remains unsupported."
            if surplus_result is not None and surplus_result.stats.coverage_exact_proven
            else "Canonical source-surplus probe emitted directional component records, but the run remains fail-closed because overlapping surplus polygons require the general GPU polygon XOR engine."
            if surplus_result is not None
            else
            "Requested gpu-polygon evidence engine emitted GPU source-record deltas but is fail-closed because the general polygon coverage XOR engine is not implemented yet."
            if record_delta is not None
            else "Requested gpu-polygon evidence engine is fail-closed because the general polygon GPU XOR record engine is not implemented yet."
            if not parse_failures
            else "Requested gpu-polygon evidence engine is fail-closed because native polygon parsing failed before GPU XOR."
        )
        warnings = (warning_text,)
        exact_fallback = {
            "mode": f"gpu-polygon-{coverage_accelerator}-rect-coverage"
            if coverage_result is not None and coverage_accelerator is not None
            else f"gpu-polygon-{record_delta_accelerator}-source-edge-cpu-oracle-fallback"
            if polygon_fragment_exact_result is not None and source_edge_cpu_fallback is not None
            else f"gpu-polygon-{record_delta_accelerator}-source-edge-records"
            if polygon_fragment_exact_result is not None
            else f"gpu-polygon-{gpu_name}-native-parse-cpu-oracle-fallback"
            if native_parse_cpu_verification is not None
            else "canonical-source-surplus-exact"
            if surplus_result is not None and surplus_result.stats.coverage_exact_proven
            else "canonical-source-surplus-probe"
            if surplus_result is not None
            else f"gpu-polygon-{record_delta_accelerator}-record-delta-no-coverage-xor"
            if record_delta is not None
            else "gpu-polygon-unavailable",
            "engine": "gpu-polygon",
            "engine_status": "rect-coverage-complete"
            if coverage_result is not None
            else "source-edge-cpu-oracle-corrected"
            if polygon_fragment_exact_result is not None and source_edge_cpu_fallback is not None
            else "source-edge-boundary-records-complete"
            if polygon_fragment_exact_result is not None
            else "native-parse-cpu-oracle-corrected"
            if native_parse_cpu_verification is not None
            else "canonical-source-surplus-exact"
            if surplus_result is not None and surplus_result.stats.coverage_exact_proven
            else "canonical-source-surplus-probe-only"
            if surplus_result is not None
            else "record-delta-only"
            if record_delta is not None
            else "not-implemented"
            if not parse_failures
            else "native-parse-failed",
            "audit_cpu_oracle": False,
            "window_count": 0,
            "escalated_layers": [],
            "failures": parse_failures
            + record_delta_failures
            + component_area_failures
            + ([] if coverage_result is not None or polygon_fragment_exact_result is not None or surplus_result is not None else coverage_failures)
            + (
                []
                if coverage_result is not None or polygon_fragment_exact_result is not None or (surplus_result is not None and surplus_result.stats.coverage_exact_proven)
                else [
                {
                    "reason": "gpu-polygon coverage XOR engine is not implemented yet",
                }
                ]
            ),
            "old_polygon_count": old_native.buffer.polygon_count if old_native is not None else None,
            "new_polygon_count": new_native.buffer.polygon_count if new_native is not None else None,
            "old_vertex_count": old_native.buffer.vertex_count if old_native is not None else None,
            "new_vertex_count": new_native.buffer.vertex_count if new_native is not None else None,
            "gpu_record_delta_count": record_delta.record_count if record_delta is not None else 0,
            "gpu_record_delta_matched_count": record_delta.matched_record_count if record_delta is not None else 0,
            "gpu_record_delta_reduced_key_count": record_delta.reduced_key_count if record_delta is not None else 0,
            "gpu_record_delta_accelerator": record_delta_accelerator,
            "gpu_coverage": coverage_result.stats.to_json() if coverage_result is not None else None,
            "canonical_surplus_coverage": surplus_result.stats.to_json() if surplus_result is not None else None,
            "surplus_probe_source": surplus_probe_source,
            "gpu_polygon_component_area": polygon_component_area,
            "source_edge_cpu_oracle_fallback": source_edge_cpu_fallback,
        }
        if native_parse_cpu_verification is not None:
            exact_fallback["audit_cpu_oracle"] = True
            exact_fallback["cpu_oracle_audit"] = _exact_result_audit_summary(
                exact_result,
                native_parse_cpu_verification.exact_result,
            )
            exact_fallback["exact_component_backend"] = native_parse_cpu_verification.exact_component_backend
            exact_fallback["native_parse_cpu_oracle_fallback"] = {
                "trigger": "native-polygon-parse-failed",
                "reason": "GPU native polygon extraction failed before acceleration; CPU gdstk oracle supplied complete exact evidence.",
                "failures": parse_failures,
                "exact_component_backend": native_parse_cpu_verification.exact_component_backend,
            }
        if backend_info.get("cuda_ctypes_available"):
            warnings = _release_cuda_device_memory_after_gpu_operation(timings, warnings)
        if backend_info.get("metal_ctypes_available"):
            warnings = _release_metal_device_memory_after_gpu_operation(timings, warnings)
        if audit_cpu_oracle and exact_complete:
            audit_start = perf_counter()
            audit_backend = _default_cpu_audit_component_backend(exact_component_backend)
            verification = source_edge_cpu_verification or _run_full_exact_verification(
                config,
                exact_component_backend=audit_backend,
                force_small_component_partition=True,
            )
            timings["cpu_oracle_audit_s"] = perf_counter() - audit_start
            audit_summary = _exact_result_audit_summary(exact_result, verification.exact_result)
            exact_fallback["audit_cpu_oracle"] = True
            exact_fallback["exact_component_backend"] = verification.exact_component_backend
            if (
                not audit_summary["match"]
                and coverage_result is None
                and polygon_fragment_exact_result is not None
            ):
                source_edge_cpu_fallback = {
                    "trigger": "cpu-oracle-audit-mismatch",
                    "reason": (
                        "source-edge evidence did not match the CPU gdstk exact oracle; "
                        "final exact evidence was promoted to the CPU oracle result"
                    ),
                    "original_vs_cpu_oracle": audit_summary,
                    "exact_component_backend": verification.exact_component_backend,
                }
                exact_result = verification.exact_result
                polygon_fragment_exact_result = exact_result
                policy_result = classify_exact_differences(
                    exact_result,
                    config.allowed_regions,
                    failure_policy=config.failure_policy,
                )
                layers = build_layer_reports(
                    exact_result,
                    policy_result=policy_result,
                    presence_by_layer=_presence_by_layer(old_native.buffer, new_native.buffer),
                    polygon_counts_by_layer=_polygon_counts_by_layer(old_native.buffer, new_native.buffer),
                    exact_status="gpu-polygon-source-edge-cpu-oracle-fallback",
                )
                if polygon_component_area is not None:
                    polygon_component_area = dict(polygon_component_area)
                    polygon_component_area["source_edge_cpu_oracle_fallback"] = source_edge_cpu_fallback
                    exact_fallback["gpu_polygon_component_area"] = polygon_component_area
                exact_fallback["mode"] = f"gpu-polygon-{record_delta_accelerator}-source-edge-cpu-oracle-fallback"
                exact_fallback["engine_status"] = "source-edge-cpu-oracle-corrected"
                exact_fallback["source_edge_cpu_oracle_fallback"] = source_edge_cpu_fallback
                exact_fallback["cpu_oracle_audit_original_mismatch"] = audit_summary
                audit_summary = _exact_result_audit_summary(exact_result, verification.exact_result)
                warnings = warnings + (
                    "CPU gdstk audit corrected source-edge GPU evidence; original mismatch is recorded in exact_fallback.cpu_oracle_audit_original_mismatch.",
                )
            exact_fallback["cpu_oracle_audit"] = audit_summary
            if not audit_summary["match"]:
                warnings = warnings + ("CPU gdstk audit did not match GPU evidence; see exact_fallback.cpu_oracle_audit.",)
        old_input = _native_polygon_input_json(config.old_path, old_top_name, old_native)
        new_input = _native_polygon_input_json(config.new_path, new_top_name, new_native)
        selection = {"old_top": old_top_name, "new_top": new_top_name, "layers": config.to_json_dict()["layers"]}
        if exact_diff_markdown_path is not None:
            markdown_start = perf_counter()
            if coverage_result is not None:
                markdown_written = write_exact_diff_markdown(
                    policy_result,
                    exact_diff_markdown_path,
                    title=f"GDS comparator gpu-polygon {str(coverage_accelerator).upper()} rectangle coverage diff",
                    source_mode=exact_fallback["mode"],
                    exact_verification_complete=True,
                    polygon_inventory_delta=polygon_inventory_delta,
                )
                _record_exact_markdown_artifacts(artifacts, markdown_written)
            elif polygon_fragment_exact_result is not None:
                markdown_written = write_exact_diff_markdown(
                    policy_result,
                    exact_diff_markdown_path,
                    title=f"GDS comparator gpu-polygon {str(record_delta_accelerator).upper()} source-edge diff",
                    source_mode=exact_fallback["mode"],
                    exact_verification_complete=True,
                    polygon_inventory_delta=polygon_inventory_delta,
                )
                if polygon_component_area is not None:
                    _append_gpu_polygon_component_area_markdown(markdown_written.path, polygon_component_area)
                _record_exact_markdown_artifacts(artifacts, markdown_written)
            elif surplus_result is not None:
                markdown_written = write_exact_diff_markdown(
                    policy_result,
                    exact_diff_markdown_path,
                    title="GDS comparator canonical source-surplus diff",
                    source_mode=exact_fallback["mode"],
                    exact_verification_complete=surplus_result.stats.coverage_exact_proven,
                    polygon_inventory_delta=polygon_inventory_delta,
                )
                if polygon_component_area is not None:
                    _append_gpu_polygon_component_area_markdown(markdown_written.path, polygon_component_area)
                _record_exact_markdown_artifacts(artifacts, markdown_written)
            elif native_parse_cpu_verification is not None:
                markdown_written = write_exact_diff_markdown(
                    policy_result,
                    exact_diff_markdown_path,
                    title="GDS comparator GPU native-parse CPU oracle fallback diff",
                    source_mode=exact_fallback["mode"],
                    exact_verification_complete=True,
                    polygon_inventory_delta=polygon_inventory_delta,
                )
                _record_exact_markdown_artifacts(artifacts, markdown_written)
            elif record_delta is not None:
                _write_gpu_polygon_record_delta_markdown(exact_diff_markdown_path, warnings[0], record_delta)
                artifacts["exact_diff_markdown"] = str(exact_diff_markdown_path)
            else:
                _write_gpu_polygon_unavailable_markdown(exact_diff_markdown_path, warnings[0])
                artifacts["exact_diff_markdown"] = str(exact_diff_markdown_path)
            timings["markdown_format_s"] = perf_counter() - markdown_start

    if markdown_path is not None:
        markdown_path = Path(markdown_path)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(
            "\n".join(
                [
                    "### GDS comparator gpu-polygon evidence result",
                    "",
                    f"- Status: {'cpu-audited' if audit_cpu_oracle else 'unresolved'}.",
                    f"- Note: {warnings[0]}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        artifacts["markdown"] = str(markdown_path)

    timings["total_s"] = perf_counter() - start
    audit_summary = exact_fallback.get("cpu_oracle_audit") if isinstance(exact_fallback, dict) else None
    unsupported_feature_count = _gpu_polygon_unsupported_feature_count(exact_fallback, exact_complete=exact_complete)
    status = _neutral_compare_status(
        policy_result,
        polygon_inventory_delta=polygon_inventory_delta,
        exact_complete=exact_complete,
        unsupported_feature_count=unsupported_feature_count,
        audit_match=(audit_summary or {}).get("match") if isinstance(audit_summary, dict) else None,
    )
    neutral_fields = _neutral_evidence_report_fields(
        status,
        policy_result,
        polygon_inventory_delta=polygon_inventory_delta,
        exact_complete=exact_complete,
        unsupported_feature_count=unsupported_feature_count,
        vertex_sidecar_path=artifacts.get("vertex_sidecar"),
    )
    report = CompareReport(
        status=status,
        exact_verification_complete=exact_complete,
        old_input=old_input,
        new_input=new_input,
        semantics=config.to_json_dict(),
        selection=selection,
        backend=backend_info
        | {
            "exact_mode": False,
            "evidence_engine_requested": config.evidence_engine.value,
            "evidence_engine_selected": "gdstk-oracle" if cpu_neutral_fallback else "gpu-polygon",
            "gpu_polygon_available": (
                False if cpu_neutral_fallback else coverage_result is not None or polygon_fragment_exact_result is not None
            ),
            "cpu_neutral_fallback": cpu_neutral_fallback,
            "gpu_record_delta_available": record_delta is not None,
            "canonical_surplus_coverage_available": surplus_result is not None,
            "canonical_surplus_coverage_exact": (
                surplus_result.stats.coverage_exact_proven if surplus_result is not None else False
            ),
            "gpu_coverage_engine": (
                f"{coverage_result.stats.accelerator}-rect-coverage" if coverage_result is not None else None
            ),
            "metal_coverage_engine": "metal-rect-coverage"
            if coverage_result is not None and coverage_result.stats.accelerator == "metal"
            else None,
            "audit_cpu_oracle": audit_cpu_oracle,
            "cuda_kernel_count": (
                (
                    record_delta.cuda_kernel_count
                    if record_delta is not None and _record_delta_accelerator(record_delta) == "cuda"
                    else 0
                )
                + (
                    coverage_result.stats.cuda_kernel_count
                    if coverage_result is not None and coverage_result.stats.accelerator == "cuda"
                    else 0
                )
                + int(
                    polygon_component_area.get("context_cuda_kernel_count", 0)
                    if polygon_component_area is not None
                    else 0
                )
                + (
                    1
                    if polygon_fragment_exact_result is not None and record_delta is not None and _record_delta_accelerator(record_delta) == "cuda"
                    else 0
                )
            ),
            "metal_kernel_count": (
                (
                    record_delta.metal_kernel_count
                    if record_delta is not None and _record_delta_accelerator(record_delta) == "metal"
                    else 0
                )
                + (1 if coverage_result is not None and coverage_result.stats.accelerator == "metal" else 0)
                + int(
                    polygon_component_area.get("context_metal_kernel_count", 0)
                    if polygon_component_area is not None
                    else 0
                )
                + (
                    1
                    if polygon_fragment_exact_result is not None and record_delta is not None and _record_delta_accelerator(record_delta) == "metal"
                    else 0
                )
            ),
            "cuda_event_elapsed_s": (
                record_delta.cuda_event_elapsed_s
                if record_delta is not None and _record_delta_accelerator(record_delta) == "cuda"
                else 0.0
            ),
            "metal_event_elapsed_s": (
                record_delta.metal_event_elapsed_s
                if record_delta is not None and _record_delta_accelerator(record_delta) == "metal"
                else 0.0
            ),
            "transferred_bytes": (
                (record_delta.transferred_bytes if record_delta is not None else 0)
                + (coverage_result.stats.transferred_bytes if coverage_result is not None else 0)
                + int(polygon_component_area.get("transferred_bytes", 0) if polygon_component_area is not None else 0)
            ),
        },
        prefilter=prefilter.to_json(),
        exact_fallback=exact_fallback,
        layers=layers,
        allowed_policy=policy_result.effective_policy,
        unexpected_summary={
            "expected_area_dbu2": policy_result.to_json()["expected_area_dbu2"],
            "unexpected_area_dbu2": policy_result.to_json()["unexpected_area_dbu2"],
        },
        timings=timings,
        warnings=warnings,
        errors=() if audit_cpu_oracle else warnings,
        artifacts=artifacts,
        **neutral_fields,
        command=command,
        failure_policy=config.failure_policy,
    )
    if json_report_path is not None:
        artifacts["json_report"] = str(json_report_path)
        report = _report_with_artifacts(report, artifacts)
        report.write_json(json_report_path)
    return CompareRunResult(
        status=CompareStatus.coerce(status),
        report=report,
        exact_result=exact_result,
        policy_result=policy_result,
        prefilter=prefilter,
        windows=windows,
        artifacts=artifacts,
        timings=timings,
    )


def _gpu_polygon_native_inputs(
    config: CompareConfig,
    *,
    old_top_name: str | None,
    new_top_name: str | None,
):
    if config.geometry_cache_dir is None:
        old_extract_args = (config.old_path, old_top_name, config.layers)
        new_extract_args = (config.new_path, new_top_name, config.layers)
        if _should_extract_pair_in_parallel(config):
            with ThreadPoolExecutor(max_workers=2) as executor:
                old_future = executor.submit(_extract_native_one, old_extract_args)
                new_future = executor.submit(_extract_native_one, new_extract_args)
                return old_future.result(), new_future.result(), {}
        return (
            _extract_native_one(old_extract_args),
            _extract_native_one(new_extract_args),
            {},
        )
    old_cache_args = (config.geometry_cache_dir, config.old_path, old_top_name, config.layers)
    new_cache_args = (config.geometry_cache_dir, config.new_path, new_top_name, config.layers)
    if _should_extract_pair_in_parallel(config):
        with ThreadPoolExecutor(max_workers=2) as executor:
            old_future = executor.submit(_cached_native_polygon_one, old_cache_args)
            new_future = executor.submit(_cached_native_polygon_one, new_cache_args)
            old_cached = old_future.result()
            new_cached = new_future.result()
        timings: dict[str, object] = {}
        timings.update(old_cached.to_timing_json("old"))
        timings.update(new_cached.to_timing_json("new"))
        return old_cached.extraction, new_cached.extraction, timings
    old_cached = _cached_native_polygon_one(old_cache_args)
    new_cached = _cached_native_polygon_one(new_cache_args)
    timings: dict[str, object] = {}
    timings.update(old_cached.to_timing_json("old"))
    timings.update(new_cached.to_timing_json("new"))
    return old_cached.extraction, new_cached.extraction, timings


def _native_buffers_rectangle_coverage_eligible(*buffers: PolygonBuffer) -> bool:
    for buffer in buffers:
        if buffer.polygon_count == 0:
            continue
        active = buffer.flags == 0
        if not bool(active.any()):
            continue
        vertex_counts = buffer.polygon_offsets[1:] - buffer.polygon_offsets[:-1]
        if not bool((vertex_counts[active] == 4).all()):
            return False
    return True


def _extract_native_one(args):
    path, top_name, layers = args
    return extract_gds_polygons_native(path, top_name=top_name, layers=layers)


def _cached_native_polygon_one(args):
    cache_dir, path, top_name, layers = args
    return load_or_build_cached_native_polygons(
        cache_dir=cache_dir,
        path=path,
        top_name=top_name,
        layers=layers,
    )


def _default_cpu_audit_component_backend(exact_component_backend: str) -> str:
    return "gdstk" if exact_component_backend == "none" else exact_component_backend


def _record_delta_accelerator(record_delta) -> str:
    return str(getattr(record_delta, "accelerator", "cuda"))


def _release_cuda_device_memory_after_gpu_operation(timings: dict[str, float], warnings: tuple[str, ...]) -> tuple[str, ...]:
    release_start = perf_counter()
    try:
        from .backends.cuda_ctypes import CudaCtypesUnavailable, release_cuda_device_memory

        release_cuda_device_memory()
    except CudaCtypesUnavailable as exc:
        return warnings + (f"CUDA device memory release failed after gpu-polygon operation: {exc}",)
    finally:
        timings["cuda_release_s"] = perf_counter() - release_start
    return warnings


def _release_metal_device_memory_after_gpu_operation(timings: dict[str, float], warnings: tuple[str, ...]) -> tuple[str, ...]:
    release_start = perf_counter()
    try:
        from .backends.metal_ctypes import MetalCtypesUnavailable, release_metal_device_memory

        release_metal_device_memory()
    except MetalCtypesUnavailable as exc:
        return warnings + (f"Metal device memory release failed after gpu-polygon operation: {exc}",)
    finally:
        timings["metal_release_s"] = perf_counter() - release_start
    return warnings


def _exact_result_audit_summary(gpu_result: ExactOracleResult, cpu_result: ExactOracleResult) -> dict[str, object]:
    gpu_layers = _exact_result_layer_audit_map(gpu_result)
    cpu_layers = _exact_result_layer_audit_map(cpu_result)
    mismatches = []
    for layer in sorted(set(gpu_layers) | set(cpu_layers)):
        gpu_layer = gpu_layers.get(layer)
        cpu_layer = cpu_layers.get(layer)
        if gpu_layer != cpu_layer:
            mismatches.append(
                {
                    "layer": [layer[0], layer[1]],
                    "gpu": gpu_layer,
                    "cpu": cpu_layer,
                }
            )
    return {
        "oracle": "gdstk-full-exact",
        "match": not mismatches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
        "truncated_mismatch_count": max(0, len(mismatches) - 20),
    }


def _exact_result_layer_audit_map(result: ExactOracleResult) -> dict[tuple[int, int], dict[str, object]]:
    layers = {}
    for layer in result.layers:
        old_fragments = tuple(
            _audit_fragment_json(fragment)
            for fragment in sorted(layer.old_minus_new, key=_audit_fragment_sort_key)
        )
        new_fragments = tuple(
            _audit_fragment_json(fragment)
            for fragment in sorted(layer.new_minus_old, key=_audit_fragment_sort_key)
        )
        if not old_fragments and not new_fragments:
            continue
        layers[(layer.layer.layer, layer.layer.datatype)] = {
            "old_minus_new_twice_area": sum(fragment.twice_area for fragment in layer.old_minus_new),
            "new_minus_old_twice_area": sum(fragment.twice_area for fragment in layer.new_minus_old),
            "xor_twice_area": sum(fragment.twice_area for fragment in layer.xor),
            "old_minus_new_count": len(old_fragments),
            "new_minus_old_count": len(new_fragments),
            "xor_count": len(layer.xor),
            "old_minus_new_fragments": old_fragments,
            "new_minus_old_fragments": new_fragments,
        }
    return layers


def _audit_fragment_sort_key(fragment: DiffFragment) -> tuple[tuple[int, int, int, int], int, int]:
    return fragment.bbox, fragment.twice_area, len(fragment.points)


def _audit_fragment_json(fragment: DiffFragment) -> dict[str, object]:
    points = _audit_canonical_points(fragment.points)
    return {
        "bbox": list(fragment.bbox),
        "twice_area": fragment.twice_area,
        "vertex_count": len(points),
        "points_sha256": _audit_points_hash(points),
    }


def _audit_canonical_points(points: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    cleaned = _audit_clean_loop_points(points)
    if not cleaned:
        return ()
    candidates: list[tuple[tuple[int, int], ...]] = []
    for sequence in (cleaned, tuple(reversed(cleaned))):
        candidates.extend(sequence[index:] + sequence[:index] for index in range(len(sequence)))
    return min(candidates)


def _audit_clean_loop_points(points: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    cleaned: list[tuple[int, int]] = []
    for point in points:
        if not cleaned or cleaned[-1] != point:
            cleaned.append(point)
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    changed = True
    while changed and len(cleaned) >= 3:
        changed = False
        next_points: list[tuple[int, int]] = []
        for index, point in enumerate(cleaned):
            previous = cleaned[index - 1]
            following = cleaned[(index + 1) % len(cleaned)]
            if _audit_collinear(previous, point, following):
                changed = True
                continue
            next_points.append(point)
        cleaned = next_points
    return tuple(cleaned)


def _audit_collinear(
    left: tuple[int, int],
    middle: tuple[int, int],
    right: tuple[int, int],
) -> bool:
    return (middle[0] - left[0]) * (right[1] - middle[1]) == (middle[1] - left[1]) * (right[0] - middle[0])


def _audit_points_hash(points: tuple[tuple[int, int], ...]) -> str:
    payload = ";".join(f"{x},{y}" for x, y in points).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _resolve_single_common_top_after_native_parse_failure(
    config: CompareConfig,
    reason: str,
    old_top_name: str | None,
    new_top_name: str | None,
) -> str | None:
    if old_top_name is not None or new_top_name is not None or "expected exactly one top cell" not in reason:
        return None
    try:
        old_metadata = read_gds_metadata(config.old_path)
        new_metadata = read_gds_metadata(config.new_path)
    except Exception:
        return None
    common_tops = sorted(set(old_metadata.top_names) & set(new_metadata.top_names))
    if len(common_tops) != 1:
        return None
    return common_tops[0]


def _write_gpu_polygon_unavailable_markdown(path: str | Path, warning: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# GDS comparator gpu-polygon exact diff",
                "",
                "## Comparator Evidence",
                "",
                "- status: `unresolved`",
                "- source_mode: `gpu-polygon-unavailable`",
                "- exact_verification_complete: `false`",
                "- expected_area_dbu2: `0`",
                "- unexpected_area_dbu2: `0`",
                "- layer_count: `0`",
                "- component_count: `0`",
                "- polygon_inventory_delta_count: `not-recorded`",
                "",
                "## Exactness Warning",
                "",
                warning,
                "",
                "## Agent Minimal Read",
                "",
                "- coverage_changed: `unknown`",
                "- coverage_xor_area_dbu2: `unknown`",
                "- inventory_changed: `unknown`",
                "- component_count: `0`",
                "- note: `gpu-polygon numeric diff records are not available yet; rerun with --audit-cpu-oracle for CPU gdstk evidence`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_gpu_polygon_record_delta_markdown(path: str | Path, warning: str, record_delta, *, row_limit: int = 200) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    accelerator = _record_delta_accelerator(record_delta)
    shown = min(record_delta.record_count, row_limit)
    omitted = max(0, record_delta.record_count - shown)
    lines = [
        "# GDS comparator gpu-polygon exact diff",
        "",
        "## Comparator Evidence",
        "",
        "- status: `unresolved`",
        f"- source_mode: `gpu-polygon-{accelerator}-record-delta-no-coverage-xor`",
        "- exact_verification_complete: `false`",
        "- coverage_xor_available: `false`",
        f"- gpu_record_delta_accelerator: `{accelerator}`",
        "- gpu_record_delta_available: `true`",
        f"- gpu_record_delta_count: `{record_delta.record_count}`",
        f"- matched_source_record_count: `{record_delta.matched_record_count}`",
        f"- reduced_source_record_key_count: `{record_delta.reduced_key_count}`",
        f"- gpu_kernel_count: `{getattr(record_delta, accelerator + '_kernel_count', 0)}`",
        f"- gpu_event_elapsed_s: `{getattr(record_delta, accelerator + '_event_elapsed_s', 0.0):.9f}`",
        f"- transferred_bytes: `{record_delta.transferred_bytes}`",
        "",
        "## Exactness Warning",
        "",
        warning,
        "",
        "## Agent Minimal Read",
        "",
        "- coverage_changed: `unknown`",
        "- coverage_xor_area_dbu2: `unknown`",
        f"- source_polygon_records_changed: `{'true' if record_delta.record_count else 'false'}`",
        f"- source_record_delta_rows: `{record_delta.record_count}`",
        "- note: `these rows are GPU-emitted source polygon record deltas; they are not yet coverage XOR components`",
        "",
        "## GPU Source Record Delta",
        "",
        "| direction | count | layer | datatype | bbox_dbu | vertex_count | flags | hash0 | hash1 |",
        "|---|---:|---:|---:|---|---:|---:|---|---|",
    ]
    for index in range(shown):
        direction = "old_minus_new" if int(record_delta.directions[index]) == 1 else "new_minus_old"
        bbox = [int(value) for value in record_delta.bboxes[index]]
        hash0 = f"{int(record_delta.hashes[index][0]):016x}"
        hash1 = f"{int(record_delta.hashes[index][1]):016x}"
        lines.append(
            "| {direction} | {count} | {layer} | {datatype} | `{bbox}` | {vertex_count} | {flags} | `{hash0}` | `{hash1}` |".format(
                direction=direction,
                count=int(record_delta.counts[index]),
                layer=int(record_delta.layers[index]),
                datatype=int(record_delta.datatypes[index]),
                bbox=bbox,
                vertex_count=int(record_delta.vertex_counts[index]),
                flags=int(record_delta.flags[index]),
                hash0=hash0,
                hash1=hash1,
            )
        )
    if omitted:
        lines.extend(
            [
                "",
                "## Omitted Rows",
                "",
                f"- omitted_gpu_record_delta_rows: `{omitted}`",
                "- note: `rerun with a larger row limit after the coverage-XOR engine is available if full source-record inventory is needed`",
            ]
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _append_gpu_polygon_component_area_markdown(path: str | Path, component_area: dict[str, object]) -> None:
    path = Path(path)
    gpu_trapezoid = bool(component_area.get("gpu_trapezoid_fragments_available", False))
    gpu_source_edge = bool(component_area.get("gpu_source_edge_records_available", False))
    exact_boundary = bool(component_area.get("exact_boundary_loops_available", False))
    lines = [
        "",
        "## GPU Polygon Evidence",
        "",
        f"- gpu_trapezoid_fragments_available: `{str(gpu_trapezoid).lower()}`",
        f"- gpu_source_edge_records_available: `{str(gpu_source_edge).lower()}`",
        f"- merged_boundary_loop_records_available: `{str(bool(component_area.get('merged_boundary_loop_records_available', False))).lower()}`",
        f"- boundary_loop_assembly_backend: `{component_area.get('boundary_loop_assembly_backend', 'none')}`",
        f"- exact_boundary_loops_available: `{str(exact_boundary).lower()}`",
        "- note: `default CUDA evidence uses GPU context indexing and GPU edge cancel/split when supported, then CPU formats compact exact boundary records and Markdown`",
        f"- component_count: `{component_area.get('component_count', 0)}`",
        f"- context_component_count: `{component_area.get('context_component_count', 0)}`",
        f"- context_max_fragments_per_component: `{component_area.get('context_max_fragments_per_component', 0)}`",
        f"- context_max_vertices_per_component: `{component_area.get('context_max_vertices_per_component', 0)}`",
        f"- emitted_trapezoid_fragment_count: `{component_area.get('emitted_trapezoid_fragment_count', 0)}`",
        f"- old_minus_new_twice_area_dbu2: `{component_area.get('old_minus_new_twice_area', 0)}`",
        f"- new_minus_old_twice_area_dbu2: `{component_area.get('new_minus_old_twice_area', 0)}`",
        f"- xor_twice_area_dbu2: `{component_area.get('xor_twice_area', 0)}`",
        "",
    ]
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _native_polygon_input_json(path: str | Path, top_name: str | None, extraction) -> dict[str, object]:
    data: dict[str, object] = {
        "path": str(path),
        "selected_top": top_name,
        "input_mode": "native-polygon-array",
    }
    if extraction is None:
        data["native_polygon_parse_complete"] = False
        return data
    data.update(
        {
            "native_polygon_parse_complete": True,
            "polygon_count": extraction.buffer.polygon_count,
            "vertex_count": extraction.buffer.vertex_count,
            "layers": [layer.to_json() for layer in extraction.buffer.layer_table.layers],
        }
    )
    return data


def _geometry_layer_json(summary) -> tuple[dict[str, object], ...]:
    return tuple(layer.to_json() for layer in summary.layers)


def _try_pair_verdict_cache_path(
    config: CompareConfig,
    *,
    backend_info: dict[str, object],
    old_metadata: dict[str, object],
    new_metadata: dict[str, object],
    old_top_name: str | None,
    new_top_name: str | None,
    command: tuple[str, ...],
    json_report_path: str | Path | None,
    markdown_path: str | Path | None,
    exact_diff_markdown_path: str | Path | None,
    start: float,
    exact_component_backend: str,
) -> CompareRunResult | None:
    cached = load_cached_pair_verdict(config, old_top_name=old_top_name, new_top_name=new_top_name)
    if cached is None:
        return None
    cache_status_start = perf_counter()
    geometry_status_s = perf_counter() - cache_status_start + cached.load_s
    verification = _run_full_exact_verification(config, exact_component_backend=exact_component_backend)
    exact_result = verification.exact_result
    policy_result = verification.policy_result
    timings = {
        "extract_s": verification.extract_s,
        "exact_s": verification.exact_s,
        "fast_status_s": geometry_status_s,
        "geometry_status_s": geometry_status_s,
        "pair_verdict_cache_hit": True,
        "pair_verdict_cache_load_s": cached.load_s,
        "pair_verdict_cache_key": cached.cache_key,
        "exact_verification_s": verification.extract_s + verification.exact_s,
        "total_s": perf_counter() - start,
    }
    artifacts: dict[str, str] = {}
    warnings = (
        "Cached full-geometry prefilter/windowed verdict was reused by file-content cache key; full exact gdstk verification was run afterward and supplies the polygon-level evidence.",
    )
    exact_fallback = dict(cached.exact_fallback)
    exact_fallback["mode"] = f"{exact_fallback.get('mode', 'geometry')}-cache-plus-full-exact"
    exact_fallback["fast_status"] = cached.fast_status
    exact_fallback["pair_verdict_cache_hit"] = True
    exact_fallback["pair_verdict_cache_key"] = cached.cache_key
    exact_fallback["exact_component_backend"] = verification.exact_component_backend
    layers = verification.layers
    if markdown_path is not None:
        markdown_path = Path(markdown_path)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(
            render_design_history_entry(
                "GDS comparator cached geometry-prefilter result",
                policy_result,
                tests_run=("gdsdiff geometry-cache",),
                notes=warnings,
            ),
            encoding="utf-8",
        )
        artifacts["markdown"] = str(markdown_path)
    if exact_diff_markdown_path is not None:
        artifacts["exact_diff_markdown"] = str(
            write_exact_diff_markdown(
                policy_result,
                exact_diff_markdown_path,
                title="GDS comparator cached geometry-prefilter exact diff",
                source_mode=str(exact_fallback["mode"]),
                exact_verification_complete=True,
                polygon_inventory_delta=verification.polygon_inventory_delta,
            ).path
        )
    report = CompareReport(
        status=policy_result.status,
        exact_verification_complete=True,
        old_input=verification.old_input,
        new_input=verification.new_input,
        semantics=config.to_json_dict(),
        selection=verification.selection,
        backend=backend_info
        | {
            "exact_mode": False,
            "fast_shortcuts_enabled": False,
            "geometry_cache_enabled": True,
            "pair_verdict_cache_hit": True,
        },
        prefilter=CpuTilePrefilterResult(candidate_tiles=()).to_json(),
        exact_fallback=exact_fallback,
        layers=layers,
        allowed_policy=policy_result.effective_policy,
        unexpected_summary={
            "expected_area_dbu2": policy_result.to_json()["expected_area_dbu2"],
            "unexpected_area_dbu2": policy_result.to_json()["unexpected_area_dbu2"],
        },
        timings=timings,
        warnings=warnings,
        artifacts=artifacts,
        command=command,
        failure_policy=config.failure_policy,
    )
    if json_report_path is not None:
        artifacts["json_report"] = str(json_report_path)
        report = _report_with_artifacts(report, artifacts)
        report.write_json(json_report_path)
    return CompareRunResult(
        status=policy_result.status,
        report=report,
        exact_result=exact_result,
        policy_result=policy_result,
        prefilter=CpuTilePrefilterResult(candidate_tiles=()),
        windows=(),
        artifacts=artifacts,
        timings=timings,
    )


def _try_cuda_hierarchy_fast_path(
    config: CompareConfig,
    *,
    backend_info: dict[str, object],
    old_metadata: dict[str, object],
    new_metadata: dict[str, object],
    old_top_name: str | None,
    new_top_name: str | None,
    exact: bool,
    command: tuple[str, ...],
    json_report_path: str | Path | None,
    markdown_path: str | Path | None,
    exact_diff_markdown_path: str | Path | None,
    verify_exact_after_fast: bool,
) -> CompareRunResult | None:
    if exact or not _accelerated_fast_path_eligible(config, backend_info):
        return None
    start = perf_counter()
    digest_result = _try_accelerated_byte_identity_fast_path(
        config,
        backend_info=backend_info,
        old_metadata=old_metadata,
        new_metadata=new_metadata,
        old_top_name=old_top_name,
        new_top_name=new_top_name,
        command=command,
        json_report_path=json_report_path,
        markdown_path=markdown_path,
            exact_diff_markdown_path=exact_diff_markdown_path,
            verify_exact_after_fast=verify_exact_after_fast,
            exact_component_backend="none",
            start=start,
        )
    if digest_result is not None:
        return digest_result
    try:
        import gdstk

        old_lib = gdstk.read_gds(str(config.old_path))
        new_lib = gdstk.read_gds(str(config.new_path))
        hierarchy = compare_hierarchy(old_lib, new_lib, old_top=old_top_name, new_top=new_top_name)
    except Exception:
        return None
    top = config.top or old_top_name or new_top_name
    warnings = (
        "CUDA hierarchy fast path used; geometry differences are detected by canonical hierarchy digest and reported fail-closed.",
    )
    if hierarchy.identical:
        exact_result = ExactOracleResult(())
    else:
        fragment = DiffFragment(points=((0, 0), (1, 0), (1, 1), (0, 1)), bbox=(0, 0, 1, 1), twice_area=2)
        exact_result = ExactOracleResult((LayerExactDiff(layer=LayerSpec(0, 0), new_minus_old=(fragment,), xor=(fragment,)),))
    policy_result = classify_exact_differences(exact_result, failure_policy=config.failure_policy)
    prefilter = CpuTilePrefilterResult(candidate_tiles=())
    windows: tuple[CandidateWindow, ...] = ()
    timings = {"extract_s": 0.0, "exact_s": perf_counter() - start, "total_s": perf_counter() - start}
    artifacts: dict[str, str] = {}
    layer_reports = build_layer_reports(exact_result, policy_result=policy_result)
    report = CompareReport(
        status=policy_result.status,
        exact_verification_complete=hierarchy.identical,
        old_input=old_metadata | {"selected_top": hierarchy.old_top, "extraction": {"mode": "hierarchy-fast"}},
        new_input=new_metadata | {"selected_top": hierarchy.new_top, "extraction": {"mode": "hierarchy-fast"}},
        semantics=config.to_json_dict(),
        selection={"old_top": hierarchy.old_top, "new_top": hierarchy.new_top, "layers": "all"},
        backend=backend_info | {"exact_mode": False, "hierarchy_fast_path": True},
        prefilter=prefilter.to_json(),
        exact_fallback={
            "mode": "hierarchy-digest",
            "window_count": len(hierarchy.changed_references),
            "escalated_layers": [],
            "failures": [],
            "old_digest": hierarchy.old_digest,
            "new_digest": hierarchy.new_digest,
            "matched_subtree_count": hierarchy.matched_subtree_count,
            "changed_reference_count": len(hierarchy.changed_references),
        },
        layers=layer_reports,
        allowed_policy=policy_result.effective_policy,
        unexpected_summary={
            "expected_area_dbu2": policy_result.to_json()["expected_area_dbu2"],
            "unexpected_area_dbu2": policy_result.to_json()["unexpected_area_dbu2"],
        },
        timings=timings,
        warnings=warnings,
        artifacts=artifacts,
        command=command,
        failure_policy=config.failure_policy,
    )
    if markdown_path is not None:
        markdown_path = Path(markdown_path)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(
            render_design_history_entry(
                "GDS comparator hierarchy-fast result",
                policy_result,
                tests_run=("gdsdiff hierarchy-fast",),
                notes=warnings,
            ),
            encoding="utf-8",
        )
        artifacts["markdown"] = str(markdown_path)
    if exact_diff_markdown_path is not None:
        artifacts["exact_diff_markdown"] = str(
            write_exact_diff_markdown(
                policy_result,
                exact_diff_markdown_path,
                title="GDS comparator hierarchy-fast exact diff",
                source_mode="hierarchy-digest",
                exact_verification_complete=hierarchy.identical,
            ).path
    )
    if json_report_path is not None:
        artifacts["json_report"] = str(json_report_path)
        report = _report_with_artifacts(report, artifacts)
        report.write_json(json_report_path)
    return CompareRunResult(
        status=policy_result.status,
        report=report,
        exact_result=exact_result,
        policy_result=policy_result,
        prefilter=prefilter,
        windows=windows,
        artifacts=artifacts,
        timings=timings,
    )


def _try_accelerated_byte_identity_fast_path(
    config: CompareConfig,
    *,
    backend_info: dict[str, object],
    old_metadata: dict[str, object],
    new_metadata: dict[str, object],
    old_top_name: str | None,
    new_top_name: str | None,
    command: tuple[str, ...],
    json_report_path: str | Path | None,
    markdown_path: str | Path | None,
    exact_diff_markdown_path: str | Path | None,
    verify_exact_after_fast: bool,
    exact_component_backend: str,
    start: float,
) -> CompareRunResult | None:
    identity = _file_identity(config.old_path, config.new_path, preferred_backend=str(backend_info.get("name", "")))
    if identity.same:
        exact_result = ExactOracleResult(())
        exact_complete = True
    else:
        fragment = DiffFragment(points=((0, 0), (1, 0), (1, 1), (0, 1)), bbox=(0, 0, 1, 1), twice_area=2)
        exact_result = ExactOracleResult((LayerExactDiff(layer=LayerSpec(0, 0), new_minus_old=(fragment,), xor=(fragment,)),))
        exact_complete = False
    policy_result = classify_exact_differences(exact_result, failure_policy=config.failure_policy)
    fast_status = policy_result.status
    fast_status_s = perf_counter() - start
    prefilter = CpuTilePrefilterResult(candidate_tiles=())
    windows: tuple[CandidateWindow, ...] = ()
    verification: _FullExactVerification | None = None
    if verify_exact_after_fast:
        verification = _run_full_exact_verification(config, exact_component_backend=exact_component_backend)
        exact_result = verification.exact_result
        policy_result = verification.policy_result
        exact_complete = True
    timings = {"extract_s": 0.0, "exact_s": fast_status_s, "fast_status_s": fast_status_s}
    if verification is not None:
        timings.update(
            {
                "extract_s": verification.extract_s,
                "exact_s": verification.exact_s,
                "exact_verification_s": verification.extract_s + verification.exact_s,
            }
        )
    timings["total_s"] = perf_counter() - start
    artifacts: dict[str, str] = {}
    warnings = _fast_path_warnings(verified=verification is not None)
    exact_fallback_mode = "file-digest-fast-plus-full-exact" if verification is not None else "file-digest-fail-closed"
    old_input = old_metadata | {"selected_top": old_top_name, "extraction": {"mode": "accelerated-byte-identity"}}
    new_input = new_metadata | {"selected_top": new_top_name, "extraction": {"mode": "accelerated-byte-identity"}}
    selection = {"old_top": old_top_name, "new_top": new_top_name, "layers": "all"}
    layers = build_layer_reports(exact_result, policy_result=policy_result)
    if verification is not None:
        old_input = verification.old_input
        new_input = verification.new_input
        selection = verification.selection
        layers = verification.layers
    exact_fallback = {
        "mode": exact_fallback_mode,
        "window_count": 0,
        "escalated_layers": [],
        "failures": [],
        "old_identity": identity.old_token,
        "new_identity": identity.new_token,
        "identity_method": identity.method,
        "old_size": identity.old_size,
        "new_size": identity.new_size,
        "fast_status": fast_status.value,
    }
    if verification is not None:
        exact_fallback["exact_component_backend"] = verification.exact_component_backend
    report = CompareReport(
        status=policy_result.status,
        exact_verification_complete=exact_complete,
        old_input=old_input,
        new_input=new_input,
        semantics=config.to_json_dict(),
        selection=selection,
        backend=backend_info
        | {
            "exact_mode": False,
            "file_digest_fast_path": True,
            "accelerated_byte_identity_fast_path": True,
            "exact_verification_after_fast": verification is not None,
        },
        prefilter=prefilter.to_json(),
        exact_fallback=exact_fallback,
        layers=layers,
        allowed_policy=policy_result.effective_policy,
        unexpected_summary={
            "expected_area_dbu2": policy_result.to_json()["expected_area_dbu2"],
            "unexpected_area_dbu2": policy_result.to_json()["unexpected_area_dbu2"],
        },
        timings=timings,
        warnings=warnings,
        artifacts=artifacts,
        command=command,
        failure_policy=config.failure_policy,
    )
    if markdown_path is not None:
        markdown_path = Path(markdown_path)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(
            render_design_history_entry(
                "GDS comparator accelerated-byte-identity result",
                policy_result,
                tests_run=("gdsdiff accelerated-byte-identity",),
                notes=warnings,
            ),
            encoding="utf-8",
        )
        artifacts["markdown"] = str(markdown_path)
    if exact_diff_markdown_path is not None:
        artifacts["exact_diff_markdown"] = str(
            write_exact_diff_markdown(
                policy_result,
                exact_diff_markdown_path,
                title="GDS comparator accelerated-byte-identity exact diff",
                source_mode=exact_fallback_mode,
                exact_verification_complete=exact_complete,
                polygon_inventory_delta=verification.polygon_inventory_delta if verification is not None else None,
            ).path
    )
    if json_report_path is not None:
        artifacts["json_report"] = str(json_report_path)
        report = _report_with_artifacts(report, artifacts)
        report.write_json(json_report_path)
    return CompareRunResult(
        status=policy_result.status,
        report=report,
        exact_result=exact_result,
        policy_result=policy_result,
        prefilter=prefilter,
        windows=windows,
        artifacts=artifacts,
        timings=timings,
    )


def _select_exact_component_backend(name: str) -> _ExactComponentSelection:
    normalized = str(name or "none").strip().lower()
    if normalized in {"none", "gdstk", "gdstk-full"}:
        return _ExactComponentSelection(name=normalized, component_partition=False, executor=None)
    if normalized in {"gdstk-component", "component-gdstk"}:
        backend = GdstkExactComponentBackend()
        return _ExactComponentSelection(
            name=normalized,
            component_partition=True,
            executor=backend.compare_components,
            backend=backend,
        )
    if normalized in {"rect", "manhattan-rect"}:
        backend = ManhattanRectExactComponentBackend(fallback_to_gdstk=False)
        return _ExactComponentSelection(
            name=normalized,
            component_partition=True,
            executor=backend.compare_components,
            backend=backend,
        )
    if normalized in {"rect-fallback", "manhattan-rect-fallback"}:
        backend = ManhattanRectExactComponentBackend(fallback_to_gdstk=True)
        return _ExactComponentSelection(
            name=normalized,
            component_partition=True,
            executor=backend.compare_components,
            backend=backend,
        )
    if normalized in {"cuda-rect", "cuda-manhattan-rect"}:
        backend = CudaManhattanRectExactComponentBackend(fallback_to_gdstk=False)
        return _ExactComponentSelection(
            name=normalized,
            component_partition=True,
            executor=backend.compare_components,
            backend=backend,
        )
    if normalized in {"cuda-rect-fallback", "cuda-manhattan-rect-fallback"}:
        backend = CudaManhattanRectExactComponentBackend(fallback_to_gdstk=True)
        return _ExactComponentSelection(
            name=normalized,
            component_partition=True,
            executor=backend.compare_components,
            backend=backend,
        )
    if normalized in {"metal-rect", "metal-manhattan-rect"}:
        backend = MetalManhattanRectExactComponentBackend(fallback_to_gdstk=False)
        return _ExactComponentSelection(
            name=normalized,
            component_partition=True,
            executor=backend.compare_components,
            backend=backend,
        )
    if normalized in {"metal-rect-fallback", "metal-manhattan-rect-fallback"}:
        backend = MetalManhattanRectExactComponentBackend(fallback_to_gdstk=True)
        return _ExactComponentSelection(
            name=normalized,
            component_partition=True,
            executor=backend.compare_components,
            backend=backend,
        )
    raise ValueError(f"unknown exact component backend: {name!r}")


def _run_full_exact_verification(
    config: CompareConfig,
    *,
    exact_component_backend: str = "none",
    force_small_component_partition: bool = False,
) -> _FullExactVerification:
    start = perf_counter()
    old_metadata = read_gds_metadata(config.old_path)
    new_metadata = read_gds_metadata(config.new_path)
    old_top_name = config.old_top or config.top
    new_top_name = config.new_top or config.top
    if old_top_name is None and new_top_name is None:
        common_tops = sorted(set(old_metadata.top_names) & set(new_metadata.top_names))
        if len(common_tops) == 1:
            old_top_name = common_tops[0]
            new_top_name = common_tops[0]
    old_extraction, new_extraction = _extract_pair(
        config,
        old_top_name=old_top_name,
        new_top_name=new_top_name,
    )
    extract_s = perf_counter() - start
    exact_start = perf_counter()
    exact_selection = _select_exact_component_backend(exact_component_backend)
    exact_result = run_full_exact_oracle(
        old_extraction.buffer,
        new_extraction.buffer,
        component_partition=exact_selection.component_partition,
        component_executor=exact_selection.executor,
        force_small_component_partition=force_small_component_partition,
    )
    exact_s = perf_counter() - exact_start
    policy_result = classify_exact_differences(
        exact_result,
        config.allowed_regions,
        failure_policy=config.failure_policy,
    )
    layer_reports = build_layer_reports(
        exact_result,
        policy_result=policy_result,
        presence_by_layer=_presence_by_layer(old_extraction.buffer, new_extraction.buffer),
        polygon_counts_by_layer=_polygon_counts_by_layer(old_extraction.buffer, new_extraction.buffer),
        exact_status="complete",
    )
    return _FullExactVerification(
        exact_result=exact_result,
        policy_result=policy_result,
        old_input=_input_json(old_extraction, old_metadata.to_json()),
        new_input=_input_json(new_extraction, new_metadata.to_json()),
        selection={
            "old_top": old_extraction.top_name,
            "new_top": new_extraction.top_name,
            "layers": config.to_json_dict()["layers"],
        },
        layers=layer_reports,
        extract_s=extract_s,
        exact_s=exact_s,
        exact_component_backend=exact_selection.to_json(),
        polygon_inventory_delta=build_polygon_inventory_delta(old_extraction.buffer, new_extraction.buffer),
    )


def _fast_path_warnings(*, verified: bool) -> tuple[str, ...]:
    if verified:
        return (
            "Accelerated byte-identity fast path produced the status verdict; full exact gdstk verification was run after the fast verdict and supplies the polygon-level evidence.",
        )
    return (
        "Accelerated byte-identity fast path used; differing files are reported fail-closed and require exact mode or verify_exact_after_fast for geometry evidence.",
    )


def _accelerated_fast_path_eligible(config: CompareConfig, backend_info: dict[str, object]) -> bool:
    return (
        backend_info.get("name") in {Backend.CUDA.value, Backend.METAL.value}
        and not config.allowed_regions
        and config.layers == "all"
    )


def _cuda_fast_path_eligible(config: CompareConfig, backend_info: dict[str, object]) -> bool:
    return _accelerated_fast_path_eligible(config, backend_info)


def _try_cuda_file_digest_fast_path(*args, **kwargs) -> CompareRunResult | None:
    return _try_accelerated_byte_identity_fast_path(*args, **kwargs)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(old_path: Path, new_path: Path, *, preferred_backend: str | None = None) -> _FileIdentity:
    old_path = Path(old_path)
    new_path = Path(new_path)
    old_stat = old_path.stat()
    new_stat = new_path.stat()
    if old_stat.st_ino == new_stat.st_ino and old_stat.st_dev == new_stat.st_dev:
        token = f"inode:{old_stat.st_dev}:{old_stat.st_ino}:size:{old_stat.st_size}"
        return _FileIdentity(token, token, True, "inode", old_stat.st_size, new_stat.st_size)
    if old_stat.st_size != new_stat.st_size:
        return _FileIdentity(
            f"size:{old_stat.st_size}",
            f"size:{new_stat.st_size}",
            False,
            "size",
            old_stat.st_size,
            new_stat.st_size,
        )
    try:
        from .backends.identity import compare_file_identity_accelerated

        cuda_result = compare_file_identity_accelerated(old_path, new_path, preferred_backend=preferred_backend)
        return _FileIdentity(
            f"{cuda_result.method}-size:{cuda_result.old_size}",
            f"{cuda_result.method}-size:{cuda_result.new_size}",
            cuda_result.same,
            cuda_result.method,
            cuda_result.old_size,
            cuda_result.new_size,
        )
    except Exception:
        pass
    old_digest = _file_sha256(old_path)
    new_digest = _file_sha256(new_path)
    return _FileIdentity(old_digest, new_digest, old_digest == new_digest, "sha256", old_stat.st_size, new_stat.st_size)


def _select_backend(config: CompareConfig, *, exact: bool = False) -> dict[str, object]:
    requested = Backend.coerce(config.backend)
    capabilities = discover_native_capabilities()
    base = {
        "requested": requested.value,
        "cuda_available": capabilities.cuda_available,
        "cuda_extension_available": capabilities.cuda_extension_available,
        "cuda_ctypes_available": capabilities.cuda_ctypes_available,
        "metal_available": capabilities.metal_available,
        "metal_ctypes_available": capabilities.metal_ctypes_available,
        "core_extension_available": capabilities.core_extension_available,
        "nvcc_available": capabilities.nvcc_available,
    }
    if requested == Backend.CPU:
        if config.require_gpu:
            raise ValueError("require_gpu cannot be satisfied by backend='cpu'")
        return base | {"name": Backend.CPU.value, "fallback_reason": None}
    if exact:
        return base | {"name": Backend.CPU.value, "fallback_reason": "exact mode uses the gdstk CPU oracle"}
    if requested == Backend.CUDA:
        if not capabilities.cuda_available:
            raise ValueError("backend='cuda' requested, but CUDA backend extension is not available")
        return base | {"name": Backend.CUDA.value, "fallback_reason": None}
    if requested == Backend.METAL:
        if not capabilities.metal_available:
            raise ValueError("backend='metal' requested, but Metal backend runtime is not available")
        return base | {"name": Backend.METAL.value, "fallback_reason": None}
    if requested == Backend.AUTO:
        if config.require_gpu:
            if capabilities.cuda_available:
                return base | {"name": Backend.CUDA.value, "fallback_reason": None}
            if capabilities.metal_available:
                return base | {"name": Backend.METAL.value, "fallback_reason": None}
            raise ValueError("require_gpu=True but no CUDA or Metal backend runtime is available")
        if capabilities.cuda_available:
            return base | {"name": Backend.CUDA.value, "fallback_reason": None}
        if capabilities.metal_available:
            return base | {"name": Backend.METAL.value, "fallback_reason": None}
        return base | {"name": Backend.CPU.value, "fallback_reason": "no CUDA or Metal backend runtime is available"}
    raise ValueError(f"backend {requested.value!r} is not implemented")


def _run_tile_prefilter(
    backend_info: dict[str, object],
    old_buffer: PolygonBuffer,
    old_canonical_ids,
    new_buffer: PolygonBuffer,
    new_canonical_ids,
    tile_config: TileConfig,
) -> CpuTilePrefilterResult:
    if backend_info.get("name") == Backend.CUDA.value:
        from .backends.cuda_ctypes import CudaCtypesUnavailable, run_cuda_ctypes_tile_prefilter

        try:
            return run_cuda_ctypes_tile_prefilter(old_buffer, old_canonical_ids, new_buffer, new_canonical_ids, tile_config)
        except CudaCtypesUnavailable:
            pass
    if backend_info.get("name") == Backend.METAL.value:
        from .backends.metal_ctypes import MetalCtypesUnavailable, run_metal_ctypes_tile_prefilter

        try:
            return run_metal_ctypes_tile_prefilter(old_buffer, old_canonical_ids, new_buffer, new_canonical_ids, tile_config)
        except MetalCtypesUnavailable:
            pass
    try:
        from .backends.cpu_ctypes import run_cpu_ctypes_tile_prefilter

        return run_cpu_ctypes_tile_prefilter(old_buffer, old_canonical_ids, new_buffer, new_canonical_ids, tile_config)
    except Exception:
        pass
    return run_cpu_tile_prefilter(old_buffer, old_canonical_ids, new_buffer, new_canonical_ids, tile_config)


def _should_use_layer_exact(
    backend_info: dict[str, object],
    old_buffer: PolygonBuffer,
    new_buffer: PolygonBuffer,
    windows: tuple[CandidateWindow, ...],
) -> bool:
    if not windows:
        return False
    total_polygons = old_buffer.polygon_count + new_buffer.polygon_count
    if total_polygons == 0:
        return False
    estimated_window_polygons = sum(window.estimated_old_polygons + window.estimated_new_polygons for window in windows)
    if estimated_window_polygons > total_polygons * 2:
        return True
    if len(windows) > 64 and estimated_window_polygons > total_polygons:
        return True
    return False


def _shared_canonical_ids(
    old_buffer: PolygonBuffer,
    new_buffer: PolygonBuffer,
    *,
    old_intern=None,
    new_intern=None,
) -> tuple[np.ndarray, np.ndarray]:
    import numpy as np

    old_intern = old_intern if old_intern is not None else intern_polygons(old_buffer)
    new_intern = new_intern if new_intern is not None else intern_polygons(new_buffer)
    unique_blobs = tuple(sorted(set(old_intern.unique_blobs) | set(new_intern.unique_blobs)))
    id_by_blob = {blob: index for index, blob in enumerate(unique_blobs)}
    old_by_index = {record.polygon_index: id_by_blob[record.blob] for record in old_intern.records}
    new_by_index = {record.polygon_index: id_by_blob[record.blob] for record in new_intern.records}
    return (
        np.asarray([old_by_index[index] for index in range(old_buffer.polygon_count)], dtype=np.uint64),
        np.asarray([new_by_index[index] for index in range(new_buffer.polygon_count)], dtype=np.uint64),
    )


def _input_json(extraction: ExtractionResult, metadata: dict[str, object]) -> dict[str, object]:
    data = dict(metadata)
    data.update({"selected_top": extraction.top_name, "extraction": extraction.stats.to_json()})
    return data


def _presence_by_layer(old_buffer: PolygonBuffer, new_buffer: PolygonBuffer) -> dict[LayerSpec, tuple[bool, bool]]:
    old_layers = _layers_present(old_buffer)
    new_layers = _layers_present(new_buffer)
    return {layer: (layer in old_layers, layer in new_layers) for layer in sorted(old_layers | new_layers)}


def _polygon_counts_by_layer(old_buffer: PolygonBuffer, new_buffer: PolygonBuffer) -> dict[LayerSpec, tuple[int, int]]:
    old_counts = _layer_polygon_counts(old_buffer)
    new_counts = _layer_polygon_counts(new_buffer)
    return {layer: (old_counts.get(layer, 0), new_counts.get(layer, 0)) for layer in sorted(set(old_counts) | set(new_counts))}


def _layers_present(buffer: PolygonBuffer) -> set[LayerSpec]:
    return {buffer.layer_table.layers[int(layer_id)] for layer_id in buffer.layer_ids}


def _layer_polygon_counts(buffer: PolygonBuffer) -> dict[LayerSpec, int]:
    counts: dict[LayerSpec, int] = {}
    for layer_id in buffer.layer_ids:
        layer = buffer.layer_table.layers[int(layer_id)]
        counts[layer] = counts.get(layer, 0) + 1
    return counts


def _candidate_tile_counts(prefilter: CpuTilePrefilterResult) -> dict[LayerSpec, int]:
    counts: dict[LayerSpec, int] = {}
    for tile in prefilter.candidate_tiles:
        counts[tile.layer] = counts.get(tile.layer, 0) + 1
    for candidate in prefilter.oversized_candidates:
        counts[candidate.layer] = counts.get(candidate.layer, 0) + 1
    return counts


def _window_counts(windows: tuple[CandidateWindow, ...]) -> dict[LayerSpec, int]:
    counts: dict[LayerSpec, int] = {}
    for window in windows:
        counts[window.layer] = counts.get(window.layer, 0) + 1
    return counts


def _record_exact_markdown_artifacts(
    artifacts: dict[str, str],
    written: ExactDiffMarkdownWriteResult,
) -> None:
    artifacts["exact_diff_markdown"] = str(written.path)
    if written.vertex_sidecar_path is not None:
        artifacts["vertex_sidecar"] = str(written.vertex_sidecar_path)


def _coverage_component_count(policy_result: PolicyClassificationResult) -> int:
    return sum(len(layer.components) for layer in policy_result.layers)


def _coverage_twice_area(policy_result: PolicyClassificationResult) -> int:
    return sum(component.twice_area for layer in policy_result.layers for component in layer.components)


def _neutral_compare_status(
    policy_result: PolicyClassificationResult,
    *,
    polygon_inventory_delta: object | None,
    exact_complete: bool,
    unsupported_feature_count: int = 0,
    audit_match: bool | None = None,
) -> CompareStatus:
    if audit_match is False:
        return CompareStatus.INTERNAL_MISMATCH
    if unsupported_feature_count:
        return CompareStatus.UNSUPPORTED_INPUT
    if not exact_complete:
        return CompareStatus.INCOMPLETE_EVIDENCE
    if _coverage_twice_area(policy_result) or _inventory_delta_count(polygon_inventory_delta):
        return CompareStatus.DIFFERENCES_FOUND
    return CompareStatus.IDENTICAL


def _neutral_evidence_report_fields(
    status: CompareStatus,
    policy_result: PolicyClassificationResult,
    *,
    polygon_inventory_delta: object | None,
    exact_complete: bool,
    unsupported_feature_count: int,
    vertex_sidecar_path: str | None,
) -> dict[str, object]:
    coverage_area = _coverage_twice_area(policy_result)
    inventory_delta_count = _inventory_delta_count(polygon_inventory_delta)
    inventory_changed = inventory_delta_count > 0
    return {
        "evidence_status": status.value if status in {
            CompareStatus.IDENTICAL,
            CompareStatus.DIFFERENCES_FOUND,
            CompareStatus.UNSUPPORTED_INPUT,
            CompareStatus.INCOMPLETE_EVIDENCE,
            CompareStatus.INTERNAL_MISMATCH,
        } else (
            "identical"
            if status == CompareStatus.EQUIVALENT
            else "differences-found"
            if status in {CompareStatus.EXPECTED_DIFFERENCES_ONLY, CompareStatus.UNEXPECTED_DIFFERENCES}
            else "incomplete-evidence"
        ),
        "differences_found": bool(coverage_area or inventory_changed),
        "coverage_changed": bool(coverage_area),
        "inventory_changed": inventory_changed,
        "coverage_equal_but_inventory_changed": bool(exact_complete and coverage_area == 0 and inventory_changed),
        "coverage_component_count": _coverage_component_count(policy_result),
        "inventory_delta_count": inventory_delta_count,
        "vertex_sidecar_path": vertex_sidecar_path,
        "unsupported_feature_count": unsupported_feature_count,
    }


def _inventory_delta_count(polygon_inventory_delta: object | None) -> int:
    return int(getattr(polygon_inventory_delta, "changed_count", 0) or 0)


def _gpu_polygon_unsupported_feature_count(exact_fallback: dict[str, object], *, exact_complete: bool) -> int:
    if exact_complete:
        return 0
    failures = exact_fallback.get("failures", ())
    return len(failures) if isinstance(failures, list) else 0


def _report_with_artifacts(report: CompareReport, artifacts: dict[str, str]) -> CompareReport:
    return CompareReport(
        status=report.status,
        exact_verification_complete=report.exact_verification_complete,
        old_input=report.old_input,
        new_input=report.new_input,
        semantics=report.semantics,
        selection=report.selection,
        backend=report.backend,
        prefilter=report.prefilter,
        exact_fallback=report.exact_fallback,
        layers=report.layers,
        allowed_policy=report.allowed_policy,
        unexpected_summary=report.unexpected_summary,
        timings=report.timings,
        memory=report.memory,
        warnings=report.warnings,
        errors=report.errors,
        artifacts=artifacts,
        evidence_status=report.evidence_status,
        differences_found=report.differences_found,
        coverage_changed=report.coverage_changed,
        inventory_changed=report.inventory_changed,
        coverage_equal_but_inventory_changed=report.coverage_equal_but_inventory_changed,
        coverage_component_count=report.coverage_component_count,
        inventory_delta_count=report.inventory_delta_count,
        vertex_sidecar_path=report.vertex_sidecar_path,
        unsupported_feature_count=report.unsupported_feature_count,
        command=report.command,
        failure_policy=report.failure_policy,
        tool_version=report.tool_version,
    )
