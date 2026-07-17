from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_cuda_assignment_kernels_match_cpu_reference(tmp_path):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc is not available")
    repo = Path(__file__).resolve().parents[1]
    source_dir = repo / "src" / "gdsdiff" / "native_src" / "gdsdiff"
    binary = tmp_path / "test_cuda_assignment"

    compile_result = subprocess.run(
        [
            nvcc,
            "-std=c++20",
            "-O2",
            str(source_dir / "core.cpp"),
            str(source_dir / "cuda_assignment.cu"),
            str(source_dir / "test_cuda_assignment.cu"),
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
    assert run_result.stdout == "cuda assignment tests passed\n"
