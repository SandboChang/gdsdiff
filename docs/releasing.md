# Releasing

Releases use GitHub OIDC Trusted Publishing. Never add a PyPI API token to the
repository or an Actions secret.

## One-time setup

1. Confirm `gdsdiff` remains unclaimed on both PyPI and TestPyPI immediately
   before setup. Pending publishers do not reserve a name.
2. Enable two-factor authentication on the separate PyPI and TestPyPI accounts
   and retain their recovery codes securely.
3. Create GitHub environments `testpypi` and `pypi`. Protect `pypi` with a
   required manual approval while allowing the repository owner to approve.
4. Configure pending Trusted Publishers with these exact values:

   - Owner: `SandboChang`
   - Repository: `gdsdiff`
   - Workflow: `release.yml`
   - Environment: `testpypi` on TestPyPI and `pypi` on PyPI

## Candidate and stable sequence

- Candidate tags such as `v0.1.0rc1` build once, publish only to TestPyPI, and
  verify a clean installed package without sourcing third-party dependencies
  from TestPyPI.
- Fixes require a new candidate version. Published files are never replaced.
- A stable tag such as `v0.1.0` builds once, verifies the identical files on
  TestPyPI, pauses at the protected `pypi` environment, publishes those same
  files to PyPI, and then creates the GitHub release.
- The tag must exactly match the version in `pyproject.toml`; the workflow
  rejects mismatches and prereleases never enter the production job.
