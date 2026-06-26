from __future__ import annotations

import json
import sys
from pathlib import Path

import gdstk

from scgds.tools import run_gds_geometry_check as workflow


def _write_gds(path: Path, *, changed: bool = False, invalid: bool = False) -> None:
    if invalid:
        path.write_text("not a gds file", encoding="utf-8")
        return
    lib = gdstk.Library(unit=1e-6, precision=1e-6)
    cell = lib.new_cell("TOP")
    cell.add(gdstk.rectangle((0, 0), (10, 10), layer=1, datatype=0))
    if changed:
        cell.add(gdstk.rectangle((20, 0), (30, 10), layer=2, datatype=0))
    lib.write_gds(path)


def _write_rebuild_script(path: Path, output_gds: Path, *, changed: bool = False, invalid: bool = False, fail: bool = False) -> None:
    body = [
        "from pathlib import Path",
        "import gdstk",
        f"output = Path({str(output_gds)!r})",
    ]
    if fail:
        body.extend(["import sys", "sys.exit(7)"])
    elif invalid:
        body.append("output.write_text('not a gds file', encoding='utf-8')")
    else:
        body.extend(
            [
                "lib = gdstk.Library(unit=1e-6, precision=1e-6)",
                "cell = lib.new_cell('TOP')",
                "cell.add(gdstk.rectangle((0, 0), (10, 10), layer=1, datatype=0))",
            ]
        )
        if changed:
            body.append("cell.add(gdstk.rectangle((20, 0), (30, 10), layer=2, datatype=0))")
        body.append("lib.write_gds(output)")
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def _run(tmp_path: Path, output_gds: Path, rebuild_script: Path | None = None, **kwargs):
    command = (sys.executable, str(rebuild_script)) if rebuild_script is not None else ()
    options = workflow.GeometryCheckOptions(
        output_gds=output_gds,
        artifacts_dir=tmp_path / "artifacts",
        dump_dir=tmp_path / "dump",
        label="case",
        rebuild_command=command,
        tile_size=10,
        **kwargs,
    )
    return workflow.run_geometry_check(options)


def test_geometry_check_passes_expected_edit(tmp_path):
    output_gds = tmp_path / "chip.gds"
    script = tmp_path / "rebuild.py"
    _write_gds(output_gds)
    _write_rebuild_script(script, output_gds, changed=True)

    outcome = _run(tmp_path, output_gds, script, allow_whole_layer=("2:0,addition",))

    assert outcome.exit_code == 0
    assert outcome.status == "expected-differences-only"
    assert Path(outcome.backup_path).exists()
    result = json.loads((tmp_path / "artifacts" / "result.json").read_text(encoding="utf-8"))
    assert result["exit_code"] == 0
    assert result["comparator"]["status"] == "expected-differences-only"


def test_geometry_check_fails_unexpected_edit(tmp_path):
    output_gds = tmp_path / "chip.gds"
    script = tmp_path / "rebuild.py"
    _write_gds(output_gds)
    _write_rebuild_script(script, output_gds, changed=True)

    outcome = _run(tmp_path, output_gds, script)

    assert outcome.exit_code == 1
    assert outcome.status == "unexpected-differences"
    assert Path(outcome.artifacts["diff_gds"]).exists()


def test_geometry_check_reports_comparator_error(tmp_path):
    output_gds = tmp_path / "chip.gds"
    script = tmp_path / "rebuild.py"
    _write_gds(output_gds)
    _write_rebuild_script(script, output_gds, invalid=True)

    outcome = _run(tmp_path, output_gds, script)

    assert outcome.exit_code == 2
    assert outcome.status == "invalid-input"
    assert "input" in outcome.error.lower()
    assert Path(outcome.backup_path).exists()


def test_geometry_check_reports_missing_backup(tmp_path):
    outcome = _run(tmp_path, tmp_path / "missing.gds")

    assert outcome.exit_code == 2
    assert outcome.status == "missing-backup"
    assert "does not exist" in outcome.error


def test_geometry_check_stops_after_rebuild_failure(tmp_path):
    output_gds = tmp_path / "chip.gds"
    script = tmp_path / "rebuild.py"
    _write_gds(output_gds)
    _write_rebuild_script(script, output_gds, fail=True)

    outcome = _run(tmp_path, output_gds, script)

    assert outcome.exit_code == 2
    assert outcome.status == "rebuild-failed"
    assert outcome.rebuild["exit_code"] == 7
    assert Path(outcome.backup_path).exists()
    assert not (tmp_path / "artifacts" / "compare_report.json").exists()


def test_geometry_check_reports_artifact_write_failure(tmp_path):
    output_gds = tmp_path / "chip.gds"
    artifacts_file = tmp_path / "artifacts"
    _write_gds(output_gds)
    artifacts_file.write_text("not a directory", encoding="utf-8")
    options = workflow.GeometryCheckOptions(
        output_gds=output_gds,
        artifacts_dir=artifacts_file,
        dump_dir=tmp_path / "dump",
        label="artifact-fail",
    )

    outcome = workflow.run_geometry_check(options)

    assert outcome.exit_code == 2
    assert outcome.status == "artifact-error"
    assert "File exists" in outcome.error
