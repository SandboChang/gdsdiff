from __future__ import annotations

import pytest
import gdstk

from gdsdiff import api as api_module
from gdsdiff import Backend, CompareConfig, CompareStatus, compare_gds_files
from gdsdiff.backends.identity import compare_file_identity_accelerated
from gdsdiff.native import NativeCapabilities, discover_native_capabilities


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


def _capabilities(*, cuda: bool = False, metal: bool = False) -> NativeCapabilities:
    return NativeCapabilities(
        tool_version="test",
        core_extension_available=False,
        cuda_extension_available=False,
        cuda_ctypes_available=cuda,
        metal_ctypes_available=metal,
        cmake_available=False,
        nvcc_available=False,
        cuda_architectures=(),
    )


def test_auto_backend_uses_cuda_when_available(tmp_path):
    old_path, new_path = _write_pair(tmp_path)

    result = compare_gds_files(CompareConfig(old_path, new_path, backend=Backend.AUTO, tile_size=10))

    assert result.status == CompareStatus.UNEXPECTED_DIFFERENCES
    assert result.report.backend["requested"] == "auto"
    assert result.report.backend["name"] in {"cuda", "metal", "cpu"}
    if result.report.backend["name"] in {"cuda", "metal"}:
        assert result.report.backend["fallback_reason"] is None
    else:
        assert result.report.backend["fallback_reason"]


def test_auto_backend_selects_metal_when_cuda_is_absent(monkeypatch):
    monkeypatch.setattr(api_module, "discover_native_capabilities", lambda: _capabilities(cuda=False, metal=True))

    selected = api_module._select_backend(CompareConfig("old.gds", "new.gds", backend=Backend.AUTO))

    assert selected["requested"] == "auto"
    assert selected["name"] == "metal"
    assert selected["fallback_reason"] is None


def test_auto_backend_prefers_cuda_when_both_gpu_backends_are_available(monkeypatch):
    monkeypatch.setattr(api_module, "discover_native_capabilities", lambda: _capabilities(cuda=True, metal=True))

    selected = api_module._select_backend(CompareConfig("old.gds", "new.gds", backend=Backend.AUTO))

    assert selected["name"] == "cuda"


def test_auto_backend_reports_no_gpu_fallback_reason(monkeypatch):
    monkeypatch.setattr(api_module, "discover_native_capabilities", lambda: _capabilities(cuda=False, metal=False))

    selected = api_module._select_backend(CompareConfig("old.gds", "new.gds", backend=Backend.AUTO))

    assert selected["name"] == "cpu"
    assert selected["fallback_reason"] == "no CUDA or Metal backend runtime is available"


def test_auto_selected_cuda_is_fast_path_eligible(monkeypatch):
    monkeypatch.setattr(api_module, "discover_native_capabilities", lambda: _capabilities(cuda=True, metal=False))
    config = CompareConfig("old.gds", "new.gds", backend=Backend.AUTO)

    selected = api_module._select_backend(config)

    assert selected["name"] == "cuda"
    assert api_module._accelerated_fast_path_eligible(config, selected)


def test_auto_selected_metal_is_fast_path_eligible(monkeypatch):
    monkeypatch.setattr(api_module, "discover_native_capabilities", lambda: _capabilities(cuda=False, metal=True))
    config = CompareConfig("old.gds", "new.gds", backend=Backend.AUTO)

    selected = api_module._select_backend(config)

    assert selected["name"] == "metal"
    assert api_module._accelerated_fast_path_eligible(config, selected)


def test_accelerated_identity_prefers_selected_metal_backend_for_size_mismatch(tmp_path):
    old_path = tmp_path / "old.bin"
    new_path = tmp_path / "new.bin"
    old_path.write_bytes(b"abc")
    new_path.write_bytes(b"abcd")

    identity = compare_file_identity_accelerated(old_path, new_path, preferred_backend="metal")

    assert identity.method == "metal-byte-compare"
    assert identity.same is False


def test_accelerated_identity_short_circuits_cuda_size_mismatch_without_toolchain(tmp_path):
    old_path = tmp_path / "old.bin"
    new_path = tmp_path / "new.bin"
    old_path.write_bytes(b"abc")
    new_path.write_bytes(b"abcd")

    identity = compare_file_identity_accelerated(old_path, new_path, preferred_backend="cuda")

    assert identity.method == "cuda-byte-compare"
    assert identity.old_size == 3
    assert identity.new_size == 4
    assert identity.same is False


def test_cpu_backend_has_no_fallback_reason(tmp_path):
    old_path, new_path = _write_pair(tmp_path, shifted=False)

    result = compare_gds_files(CompareConfig(old_path, new_path, backend=Backend.CPU, tile_size=10))

    assert result.status == CompareStatus.EQUIVALENT
    assert result.report.backend["requested"] == "cpu"
    assert result.report.backend["name"] == "cpu"
    assert result.report.backend["fallback_reason"] is None


def test_explicit_cuda_backend_runs_when_available(tmp_path):
    if not discover_native_capabilities().cuda_available:
        pytest.skip("CUDA backend is not available")
    old_path, new_path = _write_pair(tmp_path)

    result = compare_gds_files(CompareConfig(old_path, new_path, backend=Backend.CUDA, tile_size=10))

    assert result.status == CompareStatus.UNEXPECTED_DIFFERENCES
    assert result.report.backend["name"] == "cuda"


def test_require_gpu_uses_cuda_when_available(tmp_path):
    capabilities = discover_native_capabilities()
    if not (capabilities.cuda_available or capabilities.metal_available):
        pytest.skip("no GPU backend is available")
    old_path, new_path = _write_pair(tmp_path)

    result = compare_gds_files(CompareConfig(old_path, new_path, backend=Backend.AUTO, require_gpu=True, tile_size=10))

    assert result.report.backend["name"] in {"cuda", "metal"}
    assert result.report.backend["gpu_requirement_met"] is True


def test_explicit_metal_backend_reports_cpu_fallback_when_runtime_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(api_module, "discover_native_capabilities", _capabilities)
    old_path, new_path = _write_pair(tmp_path)

    result = compare_gds_files(CompareConfig(old_path, new_path, backend=Backend.METAL, tile_size=10))

    assert result.exit_code == 1
    assert result.report.backend["name"] == "cpu"
    assert result.report.backend["requested"] == "metal"
    assert "Metal backend runtime is not available" in result.report.backend["fallback_reason"]
