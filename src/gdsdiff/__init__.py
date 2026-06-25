"""GPU-assisted GDS geometry comparison tools."""
from .api import CompareRunResult, compare_gds_files, compare_gds_paths
from .config import AllowedRegion, CompareConfig, GridSpec, LayerSpec, parse_layer_selector
from .canonicalize import CanonicalRecord, InternResult, canonicalize_polygon, intern_polygons
from .design_history import DesignHistoryEntry, render_design_history_entry
from .diff_gds import DiffGdsWriteResult, write_diff_gds
from .exact_diff_markdown import (
    ExactDiffMarkdownWriteResult,
    PolygonInventoryDelta,
    PolygonInventoryRecord,
    build_polygon_inventory_delta,
    render_exact_diff_markdown,
    write_exact_diff_markdown,
)
from .extract import ExtractionResult, ExtractionStats, extract_gds_polygons
from .exact_oracle import DiffFragment, ExactLayerTask, ExactOracleResult, LayerExactDiff, format_twice_area, run_full_exact_oracle
from .gds_metadata import GdsMetadata, read_gds_metadata
from .grid import GridResolution, resolve_comparison_grid, snap_to_grid, validate_int64_coordinate
from .hierarchy import CellIdentity, HierarchyCompareResult, ReferenceChange, canonical_cell_identity, compare_hierarchy
from .hierarchy_extract import (
    WindowedHierarchyExtractionResult,
    WindowedHierarchyStats,
    extract_hierarchy_window,
    extract_hierarchy_window_from_gds,
)
from .inventory import GdsInventory, RecordInventory, SelectedLayerPresence, inventory_gds, resolve_selected_layers
from .native import CUDA_ARCH_ENV, NativeCapabilities, discover_native_capabilities
from .policy import ClassifiedComponent, LayerPolicyResult, PolicyClassificationResult, classify_exact_differences
from .profiles import AssemblyProfile, CombinedProfileSummary, ProfileCheck, ProfileResult, calibrator_model1_profile, check_assembly_profile, combined_profile_summary
from .report import CompareReport, ComponentSummary, LayerReport, build_layer_reports, validate_report_dict
from .semantics import Backend, CompareStatus, DifferenceDirection, EvidenceEngine, FailurePolicy, PathPolicy, exit_code_for_status
from .spatial import SpatialIndex, bbox_union, bboxes_intersect, expand_bbox
from .tiling import CpuTilePrefilterResult, OversizedPolygon, TileConfig, TileKey, bbox_tile_range, floor_div, run_cpu_tile_prefilter
from .windowed_exact import WindowedExactFailure, WindowedExactResult, run_windowed_exact_oracle
from .windows import CandidateWindow, WindowConfig, build_candidate_windows

__all__ = [
    "AllowedRegion",
    "AssemblyProfile",
    "Backend",
    "CandidateWindow",
    "CellIdentity",
    "CanonicalRecord",
    "ClassifiedComponent",
    "CompareConfig",
    "CompareReport",
    "CompareRunResult",
    "CompareStatus",
    "CombinedProfileSummary",
    "ComponentSummary",
    "CpuTilePrefilterResult",
    "DesignHistoryEntry",
    "DifferenceDirection",
    "EvidenceEngine",
    "DiffGdsWriteResult",
    "DiffFragment",
    "ExactOracleResult",
    "ExactDiffMarkdownWriteResult",
    "ExactLayerTask",
    "PolygonInventoryDelta",
    "PolygonInventoryRecord",
    "ExtractionResult",
    "ExtractionStats",
    "FailurePolicy",
    "GdsMetadata",
    "GridSpec",
    "GridResolution",
    "GdsInventory",
    "HierarchyCompareResult",
    "InternResult",
    "LayerReport",
    "LayerExactDiff",
    "LayerPolicyResult",
    "LayerSpec",
    "NativeCapabilities",
    "OversizedPolygon",
    "PathPolicy",
    "PolicyClassificationResult",
    "ProfileCheck",
    "ProfileResult",
    "RecordInventory",
    "ReferenceChange",
    "SelectedLayerPresence",
    "SpatialIndex",
    "TileConfig",
    "TileKey",
    "WindowConfig",
    "WindowedHierarchyExtractionResult",
    "WindowedHierarchyStats",
    "WindowedExactFailure",
    "WindowedExactResult",
    "CUDA_ARCH_ENV",
    "bbox_union",
    "bbox_tile_range",
    "bboxes_intersect",
    "build_candidate_windows",
    "build_layer_reports",
    "build_polygon_inventory_delta",
    "calibrator_model1_profile",
    "canonical_cell_identity",
    "classify_exact_differences",
    "check_assembly_profile",
    "combined_profile_summary",
    "compare_gds_files",
    "compare_gds_paths",
    "compare_hierarchy",
    "discover_native_capabilities",
    "expand_bbox",
    "extract_hierarchy_window",
    "extract_hierarchy_window_from_gds",
    "exit_code_for_status",
    "floor_div",
    "extract_gds_polygons",
    "canonicalize_polygon",
    "intern_polygons",
    "format_twice_area",
    "inventory_gds",
    "parse_layer_selector",
    "read_gds_metadata",
    "render_design_history_entry",
    "render_exact_diff_markdown",
    "resolve_selected_layers",
    "resolve_comparison_grid",
    "run_full_exact_oracle",
    "run_windowed_exact_oracle",
    "run_cpu_tile_prefilter",
    "snap_to_grid",
    "validate_int64_coordinate",
    "validate_report_dict",
    "write_diff_gds",
    "write_exact_diff_markdown",
]
