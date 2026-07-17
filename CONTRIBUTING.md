# Contributing

Use Python 3.10 or newer in a repository-local virtual environment, install the
checkout with `python -m pip install -e ".[test]"`, and run
`python -m pytest` before submitting changes.

Changes to comparison semantics, report fields, exit codes, root exports, or
native acceleration must include focused tests and compatibility notes. Keep
CPU-only operation fully supported.
