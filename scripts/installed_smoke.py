from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path

import gdstk
import jsonschema

import gdsdiff


def main() -> int:
    package_path = Path(gdsdiff.__file__).resolve()
    if "site-packages" not in package_path.parts:
        raise RuntimeError(f"gdsdiff did not resolve from site-packages: {package_path}")
    if gdsdiff.__version__ != version("gdsdiff"):
        raise RuntimeError("runtime and distribution versions differ")
    schema = json.loads(files("gdsdiff").joinpath("schemas/gdsdiff_report-v1.schema.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)

    with tempfile.TemporaryDirectory(prefix="gdsdiff-installed-smoke-") as directory:
        root = Path(directory)
        before = _write_gds(root / "before.gds", 10)
        identical = _write_gds(root / "identical.gds", 10)
        changed = _write_gds(root / "changed.gds", 12)
        report = root / "report.json"
        _run((str(before), str(identical), "--backend", "cpu"), 0)
        _run((str(before), str(changed), "--backend", "cpu", "--json-report", str(report)), 1)
        _run((str(root / "missing.gds"), str(changed), "--backend", "cpu"), 2)
        jsonschema.validate(json.loads(report.read_text(encoding="utf-8")), schema)
    return 0


def _write_gds(path: Path, width: float) -> Path:
    library = gdstk.Library(unit=1e-6, precision=1e-9)
    top = library.new_cell("TOP")
    top.add(gdstk.rectangle((0, 0), (width, 10), layer=1, datatype=0))
    library.write_gds(path)
    return path


def _run(arguments: tuple[str, ...], expected: int) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "gdsdiff", *arguments], check=False, text=True, capture_output=True
    )
    if completed.returncode != expected:
        raise RuntimeError(
            f"unexpected gdsdiff exit {completed.returncode}, expected {expected}: {completed.stderr or completed.stdout}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
