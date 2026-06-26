from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_cuda_byte_compare_detects_equal_size_content_change(tmp_path):
    if shutil.which("nvcc") is None:
        pytest.skip("nvcc is not available")
    from scgds.gdsdiff.backends.cuda_ctypes import compare_files_cuda

    old_path = tmp_path / "old.bin"
    new_path = tmp_path / "new.bin"
    same_path = tmp_path / "same.bin"
    old_path.write_bytes(b"abcdef0123456789" * 1024)
    new_path.write_bytes(b"abcdef0123456799" * 1024)
    same_path.write_bytes(old_path.read_bytes())

    changed = compare_files_cuda(old_path, new_path)
    same = compare_files_cuda(old_path, same_path)

    assert changed.method == "cuda-byte-compare"
    assert changed.old_size == changed.new_size
    assert changed.same is False
    assert same.same is True


def test_cuda_candidate_compaction_matches_cpu_prefilter(tmp_path):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc is not available")
    repo = Path(__file__).resolve().parents[3]
    source_dir = repo / "scgds" / "gdsdiff" / "native_src" / "gdsdiff"
    binary = tmp_path / "test_cuda_candidates"

    compile_result = subprocess.run(
        [
            nvcc,
            "-std=c++20",
            "-O2",
            str(source_dir / "core.cpp"),
            str(source_dir / "cuda_assignment.cu"),
            str(source_dir / "cuda_candidates.cu"),
            str(source_dir / "test_cuda_candidates.cu"),
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
    assert run_result.stdout == "cuda candidate tests passed\n"
