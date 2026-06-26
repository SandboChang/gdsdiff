from __future__ import annotations

import pytest
import gdstk

from gds_tools_tk.gdsdiff import Backend, CompareConfig, CompareStatus, compare_gds_files


def _write_pair(tmp_path, *, shifted=True):
    old_path = tmp_path / "old.gds"
    new_path = tmp_path / "new.gds"
    old = gdstk.Library(unit=1e-6, precision=1e-6)
    new = gdstk.Library(unit=1e-6, precision=1e-6)
    old_top = old.new_cell("top")
    new_top = new.new_cell("top")
    old_top.add(gdstk.rectangle((0, 0), (10, 10), layer=1, datatype=0))
    new_top.add(gdstk.rectangle((2 if shifted else 0, 0), (12 if shifted else 10, 10), layer=1, datatype=0))
    old.write_gds(old_path)
    new.write_gds(new_path)
    return old_path, new_path


def test_auto_backend_uses_cuda_when_available(tmp_path):
    old_path, new_path = _write_pair(tmp_path)

    result = compare_gds_files(CompareConfig(old_path, new_path, backend=Backend.AUTO, tile_size=10))

    assert result.status == CompareStatus.UNEXPECTED_DIFFERENCES
    assert result.report.backend["requested"] == "auto"
    assert result.report.backend["name"] in {"cuda", "cpu"}
    if result.report.backend["name"] == "cuda":
        assert result.report.backend["fallback_reason"] is None
    else:
        assert result.report.backend["fallback_reason"]


def test_cpu_backend_has_no_fallback_reason(tmp_path):
    old_path, new_path = _write_pair(tmp_path, shifted=False)

    result = compare_gds_files(CompareConfig(old_path, new_path, backend=Backend.CPU, tile_size=10))

    assert result.status == CompareStatus.EQUIVALENT
    assert result.report.backend["requested"] == "cpu"
    assert result.report.backend["name"] == "cpu"
    assert result.report.backend["fallback_reason"] is None


def test_explicit_cuda_backend_runs_when_available(tmp_path):
    old_path, new_path = _write_pair(tmp_path)

    result = compare_gds_files(CompareConfig(old_path, new_path, backend=Backend.CUDA, tile_size=10))

    assert result.status == CompareStatus.UNEXPECTED_DIFFERENCES
    assert result.report.backend["name"] == "cuda"


def test_require_gpu_uses_cuda_when_available(tmp_path):
    old_path, new_path = _write_pair(tmp_path)

    result = compare_gds_files(CompareConfig(old_path, new_path, backend=Backend.AUTO, require_gpu=True, tile_size=10))

    assert result.report.backend["name"] == "cuda"
