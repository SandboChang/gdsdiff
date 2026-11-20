# Standalone Extraction Qualification

Qualified on 2026-07-17 under WSL2 Linux x86_64 with Python 3.14.4.

- History: 15 relevant source commits retained with original authors, dates,
  and messages, followed by two standalone migration commits.
- Independence: shipping code and tests do not import the original consuming project.
- API: all 112 generic baseline root exports and signatures were preserved
  after normalizing package and version names. The standalone root has 118
  exports after adding the documented extraction types/parsers, version, and
  diagnostics; the project-specific profile factory stayed with the consumer.
- Correctness: normalized report JSON, semantic diff GDS, and CLI help exactly
  match the embedded baseline.
- Tests: the extraction baseline CPU-only suite produced 266 passes and 62
  capability/fixture skips; the RC1 CUDA-enabled suite produced 329 passes and 17
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
  no tests or consumer package. A clean wheel-only environment passed
  `pip check`, imports, module CLI, and diagnostics outside the source tree.
- RC1 publication rehearsal: clean `/tmp` snapshots produced wheel and sdist;
  both rebuilt and installed outside the source tree with `uv`, and the wheels
  also installed with ordinary `pip`. Packaged schema validation, CLI exit
  codes, dependency checks, explicit cold/warm CUDA preparation, CPU/CUDA exact
  parity, and invalid-architecture atomic failure all passed from installed
  artifacts.
- Cache hardening: geometry and pair caches use non-executable JSON/NumPy
  serialization and ignore legacy pickle entries. Native keys include resolved
  compiler/toolkit and native GPU identity, with private POSIX cache permissions.
- Optional hardware: CUDA is locally qualified. Metal simulation coverage
  passes. Hosted macOS exercised Metal when the runner exposed a stable runtime
  and used a narrowly classified capability skip when that optional runtime
  disappeared between probes; real Apple Silicon evidence is non-blocking.

No TestPyPI or PyPI upload, Git tag, or release was created.

## 0.2.x Compatibility

Compatibility checks used separate locked environments for the released
`v0.1.0` (`6bdc0c7`), the 0.2.0 source revision (`753a352`), and the
optimized follow-up (`441498f`). Check the imported source path and revision
as well as the distribution version when reproducing these comparisons.

On the synthetic matrix, previously supported cases kept the same selection,
directional vertex evidence, inventory multiplicity, and exit meaning.
Orthogonal references newly complete on the native route. When native input
is rejected, optional GPU now reports the failure and returns exact CPU
evidence when possible. Required GPU still exits 2 if no device work ran.
An unrecovered Boolean warning is not exact evidence. The old consumer-specific
native-cache alias has been retired; use `GDSDIFF_NATIVE_CACHE_DIR`.

The verified-context correction compares full vertices and inventory against
an independent `gdstk` read. The older, unaudited source-edge route is not a
substitute for that check on complex contours: in qualification, an annular
fixture needed `--audit-cpu-oracle` to obtain the CPU reference fragments.
