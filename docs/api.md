# Public Python API

The root `gdsdiff` namespace is the supported import surface. Important
contracts include:

- `CompareConfig`, `compare_gds_files`, and `compare_gds_paths` for complete
  comparisons.
- `extract_gds_polygons`, `ExtractionResult`, `PolygonBuffer`, and `LayerTable`
  for shared, zero-copy downstream geometry processing.
- `LayerSpec`, `parse_layer_selector`, `AllowedRegion`,
  `parse_allowed_region`, and `parse_allowed_whole_layer` for stable selection
  and policy parsing.
- `read_gds_metadata` and `GdsMetadata` for GDS unit/precision metadata.
- `inventory_gds` and `GdsInventory` for the selected cell's reachable geometry
  layers and stored record counts without expanding placements. Labels are
  counted but do not add geometry layers.
- `discover_native_capabilities` and `backend_diagnostics` for CPU-safe runtime
  capability inspection.

Deep modules remain importable for advanced use, but downstream packages
should prefer these root exports. `gdsdiff.__version__` is derived from installed
distribution metadata; reports use the same version.

## Compare and verify

```python
from gdsdiff import CompareConfig, compare_gds_files

result = compare_gds_files(
    CompareConfig(
        "before.gds", "after.gds",
        top="TOP", layers="all",
        backend="cuda", evidence_engine="gpu-polygon", require_gpu=True,
    ),
    json_report_path="comparison/report.json",
    exact_diff_markdown_path="comparison/exact_diff.md",
)
print(result.status, result.exit_code, result.report.backend["gpu_work_performed"])
```

`exact_diff_markdown_path` also writes the full-vertex JSONL sidecar.
Use `backend="cpu", evidence_engine="gdstk-oracle"` without `require_gpu` for
an independent CPU reference. Pass `audit_cpu_oracle=True` to
`compare_gds_files` to request a separate read of both inputs after GPU
cleanup. That audit is different from the normal sparse route that runs exact
`gdstk` coverage on native buffers while CUDA compares source records.

Use `result.report.to_json_dict()` for schema-validated evidence. A GPU
requirement failure or audit mismatch can return exit 2 even if verified CPU
coverage is available. See the [report guide](gdsdiff_report.md) for provenance,
timings, and artifact comparison.
