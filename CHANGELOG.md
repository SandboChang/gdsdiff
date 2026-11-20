# Changelog

## Unreleased

- Prepare native polygon buffers in parallel without intermediate flattened
  polygons or redundant host hashing on the uncached CUDA path. Report input
  preparation phases and CUDA record upload time separately.
- Summarize layer counts in one pass and reuse exact integer coordinates when
  applying allowed regions, without changing the emitted geometry.
- Stop assembling boundaries once eligible CUDA comparisons prove they need
  exact hole correction. Then use sparse gdstk coverage on the aligned native
  buffers and keep the CUDA inventory evidence.
- Correct sparse hole edits even when polygon counts or vertex offsets differ.
  Verify omitted records and retained inventory exactly, recover from local
  Boolean warnings only with agreeing full vertices, and report the proof and
  numerical recovery separately from GPU execution.
- Reject partial Boolean output on numerical warnings if exact recovery fails.
- Use `GDSDIFF_NATIVE_CACHE_DIR` for cache overrides; the former
  consumer-specific alias is no longer read.

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

- Extract `gdsdiff` as an independently installable package while
  preserving its comparator history and public behavior.
- Replaced executable pickle geometry caches with versioned JSON/NumPy caches.
- Added side-effect-free CUDA inspection, explicit preparation, stronger native
  cache identity/permissions, packaged-schema validation, and release workflows.
