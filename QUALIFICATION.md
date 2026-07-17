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
- Tests: the final local WSL CPU-only suite produced 266 passes and 62
  capability/fixture skips; the CUDA-enabled suite produced 311 passes and 17
  capability/fixture skips. Both completed with zero failures or warnings.
  Peak test-process RSS in the extraction parity run was 158,440 KiB.
- Hosted portability: GitHub Actions run `29580448672` passed the complete
  packaging job and the CPU test matrix on Ubuntu, macOS, and Windows with
  Python 3.10 and 3.14. The workflow uses the current Node 24 action
  generations, `actions/checkout@v7` and `actions/setup-python@v6`. POSIX
  on-demand native-build tests are capability-skipped on Windows; the portable
  Python/CPU implementation remains fully exercised.
- CPU exact performance: seven-run median 0.191918416 seconds versus baseline
  0.208896740 seconds (8.13% faster); median absolute deviation was
  0.001739505 seconds.
- CUDA correctness: the complete suite passed on an NVIDIA GeForce RTX 5090
  (compute capability 12.0) through WSL2 with Windows display driver 596.36,
  CUDA Toolkit 13.3.1, and nvcc 13.3.73. The package's default
  `-arch=native` build emitted native SASS and required no manual nvcc flags.
- CUDA warm-path performance: on an identical 500-polygon `gpu-polygon`
  workload after one warm-up, the seven-run standalone median was
  0.014218929 seconds versus 0.061264152 seconds embedded (76.79% faster).
- Peak RSS: across 21 isolated representative exact-compare processes, the
  standalone median was 42,540 KiB versus 42,436 KiB embedded (+0.245%). The
  installed version remains derived from wheel metadata without eagerly loading
  Python's unrelated metadata/email/ZIP modules into every comparison process.
- Packaging: wheel and sdist built successfully, passed `twine check`, and the
  wheel contains the report schema, `py.typed`, CUDA/Metal/native sources, and
  no tests or `scgds` package. A clean wheel-only environment passed
  `pip check`, imports, module CLI, and diagnostics outside the source tree.
- Optional hardware: CUDA is locally qualified. Metal simulation coverage
  passes. Hosted macOS exercised Metal when the runner exposed a stable runtime
  and used a narrowly classified capability skip when that optional runtime
  disappeared between probes; real Apple Silicon evidence is non-blocking.

No TestPyPI or PyPI upload, Git tag, or release was created.
