from __future__ import annotations

import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Callable

import gdstk
import numpy as np

from .cache import CachedGeometry, load_cached_pair_verdict, load_or_build_cached_geometry, write_cached_pair_verdict
from .canonicalize import intern_polygons
from .config import CompareConfig, LayerSpec
from .backends.exact import CudaManhattanRectExactComponentBackend, GdstkExactComponentBackend, ManhattanRectExactComponentBackend
from .backends.metal import MetalManhattanRectExactComponentBackend
from .design_history import render_design_history_entry
from .diff_gds import write_diff_gds
from .exact_diff_markdown import build_polygon_inventory_delta, write_exact_diff_markdown
from .exact_oracle import DiffFragment, ExactLayerTask, ExactOracleResult, LayerExactDiff, run_full_exact_oracle
from .extract import ExtractionResult, extract_gds_polygons
from .gds_metadata import read_gds_metadata
from .geometry import PolygonBuffer
from .hierarchy import compare_hierarchy
from .native import discover_native_capabilities
from .policy import PolicyClassificationResult, classify_exact_differences
from .report import CompareReport, build_layer_reports
from .semantics import Backend, CompareStatus, exit_code_for_status
from .spatial import SpatialIndex
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
    use_fast_shortcuts: bool = True,
    use_structural_guard: bool = False,
    use_native_geometry_guard: bool = False,
    exact_component_backend: str = "none",
    command: tuple[str, ...] = (),
) -> CompareRunResult:
    start = perf_counter()
    backend_info = _select_backend(config, exact=exact)

    old_top_name = config.old_top or config.top
    new_top_name = config.new_top or config.top

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
            tests_run=("gds-gpu-compare",),
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
        artifacts["json_report"] = str(report.write_json(json_report_path))
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
        with ProcessPoolExecutor(max_workers=2) as executor:
            old_future = executor.submit(_extract_one, old_args)
            new_future = executor.submit(_extract_one, new_args)
            return old_future.result(), new_future.result()
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
        with ProcessPoolExecutor(max_workers=2) as executor:
            old_future = executor.submit(_cached_geometry_one, old_args)
            new_future = executor.submit(_cached_geometry_one, new_args)
            return old_future.result(), new_future.result()
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
                tests_run=("gds-gpu-compare structural-guard",),
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
        artifacts["json_report"] = str(report.write_json(json_report_path))
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
                tests_run=("gds-gpu-compare native-geometry-guard",),
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
        artifacts["json_report"] = str(report.write_json(json_report_path))
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
                tests_run=("gds-gpu-compare geometry-cache",),
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
        artifacts["json_report"] = str(report.write_json(json_report_path))
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
                tests_run=("gds-gpu-compare hierarchy-fast",),
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
        artifacts["json_report"] = str(report.write_json(json_report_path))
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
    identity = _file_identity(config.old_path, config.new_path)
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
                tests_run=("gds-gpu-compare accelerated-byte-identity",),
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
        artifacts["json_report"] = str(report.write_json(json_report_path))
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


def _run_full_exact_verification(config: CompareConfig, *, exact_component_backend: str = "none") -> _FullExactVerification:
    start = perf_counter()
    old_metadata = read_gds_metadata(config.old_path)
    new_metadata = read_gds_metadata(config.new_path)
    old_top_name = config.old_top or config.top
    new_top_name = config.new_top or config.top
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
        backend_info.get("requested") == Backend.CUDA.value
        and backend_info.get("name") == Backend.CUDA.value
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


def _file_identity(old_path: Path, new_path: Path) -> _FileIdentity:
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

        cuda_result = compare_file_identity_accelerated(old_path, new_path)
        return _FileIdentity(
            f"cuda-byte-size:{cuda_result.old_size}",
            f"cuda-byte-size:{cuda_result.new_size}",
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
    if requested == Backend.AUTO:
        if config.require_gpu:
            if not capabilities.cuda_available:
                raise ValueError("require_gpu=True but CUDA backend extension is not available")
            return base | {"name": Backend.CUDA.value, "fallback_reason": None}
        if capabilities.cuda_available:
            return base | {"name": Backend.CUDA.value, "fallback_reason": None}
        return base | {"name": Backend.CPU.value, "fallback_reason": "CUDA backend extension is not available"}
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
        from .backends.cuda_ctypes import run_cuda_ctypes_tile_prefilter

        return run_cuda_ctypes_tile_prefilter(old_buffer, old_canonical_ids, new_buffer, new_canonical_ids, tile_config)
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
        command=report.command,
        failure_policy=report.failure_policy,
        tool_version=report.tool_version,
    )
