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

- Status: completed locally; hosted TestPyPI installation remains pending
- Build wheel and sdist from clean native-filesystem snapshots.
- Test wheel/sdist installation with both `uv` and `pip` outside source trees.
- Run CPU-only and installed-wheel CUDA smoke tests.

## Gate 5: Hosted release

- Status: public-repository cutover, protected GitHub environments, and hosted
  cross-platform CI completed; waiting for account-side Trusted Publisher setup
  and TestPyPI/PyPI promotion
- Add pinned CI and Trusted Publishing workflows.
- Make repositories public, configure environments/publishers, rehearse on TestPyPI,
  then promote identical artifacts to PyPI with protected approval.

Hosted evidence on 2026-07-18: public `SandboChang/gdsdiff` commit `6a13637`
passed compilerless CPU, package/install smoke, and Ubuntu/macOS/Windows tests on
Python 3.10 and 3.14 in Actions run `29591224278`. GitHub environments
`testpypi` and approval-protected `pypi` are configured. No release tag will be
pushed until both package-index accounts trust `release.yml` with the matching
environment.
