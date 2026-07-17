from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_cpp_core_compiles_and_matches_reference_cases(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is not available")
    repo = Path(__file__).resolve().parents[1]
    source_dir = repo / "src" / "gdsdiff" / "native_src" / "gdsdiff"
    binary = tmp_path / "test_core"

    compile_result = subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-O2",
            str(source_dir / "core.cpp"),
            str(source_dir / "test_core.cpp"),
            "-I",
            str(source_dir),
            "-o",
            str(binary),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert compile_result.returncode == 0, compile_result.stderr
    run_result = subprocess.run([str(binary)], check=False, text=True, capture_output=True)
    assert run_result.returncode == 0, run_result.stderr
    assert run_result.stdout == "cpp core tests passed\n"
