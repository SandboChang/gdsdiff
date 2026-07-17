# Standalone Extraction Qualification

Qualified on 2026-07-17 under WSL2 Linux x86_64 with Python 3.14.4.

- History: 15 relevant source commits retained with original authors, dates,
  and messages, followed by two standalone migration commits.
- Independence: no shipping or test import references `scgds` or `projects`.
- API: all 112 generic baseline root exports and signatures were preserved
  after normalizing package and version names. The standalone root has 118
  exports after adding the documented extraction types/parsers, version, and
  diagnostics; the Calibrator-specific profile factory moved back to `scgds`.
- Correctness: normalized report JSON, semantic diff GDS, and CLI help exactly
  match the embedded baseline.
- Tests: the final local WSL suite produced 263 passes and 62
  capability/fixture skips with zero failures or warnings. Peak test-process
  RSS in the extraction parity run was 158,440 KiB.
- Hosted portability: GitHub Actions run `29576086947` passed the complete
  packaging job and the CPU test matrix on Ubuntu, macOS, and Windows with
  Python 3.10 and 3.14. POSIX on-demand native-build tests are capability-skipped
  on Windows; the portable Python/CPU implementation remains fully exercised.
- CPU exact performance: seven-run median 0.191918416 seconds versus baseline
  0.208896740 seconds (8.13% faster); median absolute deviation was
  0.001739505 seconds.
- Packaging: wheel and sdist built successfully, passed `twine check`, and the
  wheel contains the report schema, `py.typed`, CUDA/Metal/native sources, and
  no tests or `scgds` package. A clean wheel-only environment passed
  `pip check`, imports, module CLI, and diagnostics outside the source tree.
- Optional hardware: CUDA tests remain skipped because `nvcc` is not installed;
  real CUDA qualification is a later blocking campaign gate. Metal simulation
  coverage passes. Hosted macOS exercised Metal when the runner exposed a
  stable runtime and used a narrowly classified capability skip when that
  optional runtime disappeared between probes; real Apple Silicon evidence is
  non-blocking.

No TestPyPI or PyPI upload, Git tag, or release was created.
