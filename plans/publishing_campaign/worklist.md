# Publishing Campaign Worklist

## Gate 1: Reproducible development

- Status: completed
- Pin and configure `uv` without replacing platform-specific virtual environments.
- Lock dependencies and prove the existing suite before release-facing changes.

## Gate 2: `gdsdiff` hardening

- Status: completed
- Repair and validate the packaged report schema.
- Make backend inspection side-effect-free and add explicit CUDA preparation.
- Harden native cache identity and permissions.
- Replace executable pickle caches with safe versioned serialization.
- Add installed-package, CLI, version, and packaging tests.

## Gate 3: `ebeamtime` hardening

- Status: completed
- Enforce backend/`require_gpu` contracts and avoid small-AUTO GPU probing.
- Add explicit CUDA preparation and packaged-schema validation.
- Lock the stable `gdsdiff` dependency boundary.

## Gate 4: Distribution qualification

- Status: completed locally and from the TestPyPI index
- Build wheel and sdist from clean native-filesystem snapshots.
- Test wheel/sdist installation with both `uv` and `pip` outside source trees.
- Run CPU-only and installed-wheel CUDA smoke tests.

## Gate 5: Hosted release

- Status: completed
- Add pinned CI and Trusted Publishing workflows.
- Make repositories public, configure environments/publishers, rehearse on TestPyPI,
  then promote identical artifacts to PyPI with protected approval.

Hosted evidence on 2026-07-18: public `SandboChang/gdsdiff` tag `v0.1.0rc1` at
commit `4bc971c` passed the compilerless CPU, package/install smoke, and full
Ubuntu/macOS/Windows Python 3.10/3.14 matrix in Actions run `29593937569`.
Trusted Publishing run `29593937282` built once, uploaded signed artifacts to
TestPyPI, and passed a clean indexed install. The indexed wheel SHA-256 is
`b509364a657359b087bdbff8bedb09d1c72db91ca414a7c30190e14e8c24d578`;
the sdist SHA-256 is
`22132bcec744af4940464b120138a026c04b28f6fc6eaf6b89853b75eac111e8`.
Both files were byte-identical to the immutable GitHub workflow artifact.
Stable run `29594770086` exposed a short TestPyPI visibility delay after a
successful signed upload. A clean indexed install passed as soon as the files
became visible; future release verification now retries this bounded,
eventually consistent index transition before failing.
The protected production job then published signed `0.1.0` artifacts to PyPI:
wheel SHA-256
`0a561cf759639c515306d24568ff64ca148722bc4b0fa918326b8138e7b16c43`
and sdist SHA-256
`1a5f16a241b0e3797d027cdef2b20dd3c61128e27c7dd43e913285957605023e`.
Both production files were byte-identical to the immutable workflow artifact,
and a production-only clean install passed the installed-package smoke test and
dependency check. The original GitHub Release job lacked repository context;
after exact-asset verification, the release was created at `v0.1.0`, and future
jobs now pass `--repo "$GITHUB_REPOSITORY"` explicitly while retaining
`--verify-tag`.
The bounded TestPyPI loop also refreshes the package explicitly so uv cannot
retain a negative index response across attempts.

WSL environment isolation was requalified on 2026-07-18 after separating the
clean runtime environment from test tooling. `.venv_wsl` was recreated from
the committed lock with `--no-dev` and contains no pytest installation;
`.venv_wsl_test` was recreated with the development extra and passes the locked
dependency check. With `/usr/local/cuda/bin` exposed for CUDA-sensitive tests,
the complete test suite passed from `.venv_wsl_test` with 329 passed and 17
expected skips for unavailable Apple Metal hardware and optional local GDS
fixtures. CUDA tests used the verified RTX 5090 / CUDA Toolkit 13.3.73 backend.
