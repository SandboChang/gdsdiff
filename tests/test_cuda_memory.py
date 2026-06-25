from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_cuda_chunked_candidate_compaction_matches_whole_batch(tmp_path):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc is not available")
    repo = Path(__file__).resolve().parents[3]
    source_dir = repo / "gds_tools_tk" / "gpu_compare" / "native_src" / "gds_gpu_compare"
    binary = tmp_path / "test_cuda_memory"

    compile_result = subprocess.run(
        [
            nvcc,
            "-std=c++20",
            "-O2",
            str(source_dir / "core.cpp"),
            str(source_dir / "cuda_assignment.cu"),
            str(source_dir / "cuda_candidates.cu"),
            str(source_dir / "cuda_memory.cu"),
            str(source_dir / "test_cuda_memory.cu"),
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
    assert run_result.stdout == "cuda memory tests passed\n"
