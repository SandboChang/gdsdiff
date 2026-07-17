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

- Status: release candidate completed; production promotion in progress
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
