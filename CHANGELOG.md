# Changelog

## 0.1.0rc1 - 2026-07-17

- Extract `gdsdiff` from `scgds` as an independently installable package while
  preserving its comparator history and public behavior.
- Replaced executable pickle geometry caches with versioned JSON/NumPy caches.
- Added side-effect-free CUDA inspection, explicit preparation, stronger native
  cache identity/permissions, packaged-schema validation, and release workflows.
