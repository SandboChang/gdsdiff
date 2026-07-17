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
- `discover_native_capabilities` and `backend_diagnostics` for CPU-safe runtime
  capability inspection.

Deep modules remain importable for advanced use, but downstream packages
should prefer these root exports. `gdsdiff.__version__` is derived from installed
distribution metadata; reports use the same version.
