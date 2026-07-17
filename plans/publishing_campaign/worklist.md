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

- Status: waiting for public-repository cutover, Trusted Publisher account setup,
  and TestPyPI/PyPI promotion
- Add pinned CI and Trusted Publishing workflows.
- Make repositories public, configure environments/publishers, rehearse on TestPyPI,
  then promote identical artifacts to PyPI with protected approval.
