# Changelog

## Unreleased

- Prepare native polygon buffers in parallel without intermediate flattened
  polygons or redundant host hashing on the uncached CUDA path. Report input
  preparation phases and CUDA record upload time separately.

## 0.2.0

- Support unit-magnification, orthogonally rotated GDS references in the
  shared CUDA and Metal native polygon reader.
- Report GPU failures and retain exact CPU fallback evidence. GPU-required
  comparisons return exit 2 if no useful GPU comparison was performed.
- Compact CUDA overlap searches by tile and reuse eligible native polygon
  buffers for exact hole correction, with independent vertex-level CPU audit.
- Add a single-file `inspect` command and inventory geometry layers without
  expanding repeated placements.

## 0.1.0 - 2026-07-18

- Promote the independently verified `0.1.0rc1` package without behavior or
  dependency changes.
- Record successful OIDC Trusted Publishing, signed attestations, exact
  GitHub/TestPyPI artifact parity, clean indexed installation, and the complete
  Python 3.10/3.14 Ubuntu, macOS, and Windows release-candidate matrix.

## 0.1.0rc1 - 2026-07-17

- Extract `gdsdiff` from `scgds` as an independently installable package while
  preserving its comparator history and public behavior.
- Replaced executable pickle geometry caches with versioned JSON/NumPy caches.
- Added side-effect-free CUDA inspection, explicit preparation, stronger native
  cache identity/permissions, packaged-schema validation, and release workflows.
