from __future__ import annotations

import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

import gdsdiff
import gdsdiff.native as native_module
from gdsdiff.native import CUDA_ARCH_ENV, LEGACY_CUDA_ARCH_ENV, discover_native_capabilities, nvcc_architecture_flags
from gdsdiff.native_cache import (
    ENV_NATIVE_CACHE_DIR,
    LEGACY_ENV_NATIVE_CACHE_DIR,
    cached_native_library_path,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_package_import_does_not_import_optional_native_or_cuda_extensions():
    assert "gdsdiff._gds_compare_core" not in sys.modules
    assert "gdsdiff._gds_compare_cuda" not in sys.modules
    assert callable(gdsdiff.discover_native_capabilities)


def test_native_capability_discovery_is_cpu_safe(monkeypatch):
    monkeypatch.setenv(CUDA_ARCH_ENV, "75;86,89")

    caps = discover_native_capabilities()
    data = caps.to_json()

    assert data["cpu_available"] is True
    assert isinstance(data["core_extension_available"], bool)
    assert isinstance(data["cuda_extension_available"], bool)
    assert isinstance(data["cuda_ctypes_available"], bool)
    assert data["cuda_available"] == (data["cuda_extension_available"] or data["cuda_ctypes_available"])
    assert isinstance(data["metal_ctypes_available"], bool)
    assert data["metal_available"] == data["metal_ctypes_available"]
    assert data["cuda_architectures"] == ["75", "86", "89"]
    assert data["cuda_compile_flags"] == [
        "--generate-code=arch=compute_75,code=sm_75",
        "--generate-code=arch=compute_86,code=sm_86",
        "--generate-code=arch=compute_89,code=sm_89",
    ]
    assert nvcc_architecture_flags() == (
        "--generate-code=arch=compute_75,code=sm_75",
        "--generate-code=arch=compute_86,code=sm_86",
        "--generate-code=arch=compute_89,code=sm_89",
    )
    assert data["tool_version"] == "0.1.0.dev0"


def test_default_cuda_architecture_is_native(monkeypatch):
    monkeypatch.delenv(CUDA_ARCH_ENV, raising=False)
    monkeypatch.delenv(LEGACY_CUDA_ARCH_ENV, raising=False)

    assert nvcc_architecture_flags() == ("-arch=native",)


def test_tool_lookup_cache_tracks_path_changes(monkeypatch):
    calls = []

    def fake_which(name, *, path=None):
        calls.append((name, path))
        return f"{path}/{name}" if path == "with-tools" else None

    native_module._which_tool.cache_clear()
    monkeypatch.setattr(native_module.shutil, "which", fake_which)
    assert native_module._which_tool("nvcc", "without-tools") is None
    assert native_module._which_tool("nvcc", "with-tools") == "with-tools/nvcc"
    assert native_module._which_tool("nvcc", "with-tools") == "with-tools/nvcc"
    assert calls == [("nvcc", "without-tools"), ("nvcc", "with-tools")]
    native_module._which_tool.cache_clear()


def test_pyproject_defines_cpu_safe_console_entry_point():
    pyproject = tomllib.loads((_repo_root() / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]
    optional_dependencies = pyproject["project"]["optional-dependencies"]

    assert pyproject["build-system"]["build-backend"] == "setuptools.build_meta"
    assert pyproject["project"]["name"] == "gdsdiff"
    assert pyproject["project"]["version"] == "0.1.0.dev0"
    assert pyproject["project"]["license"] == "GPL-3.0-only"
    assert pyproject["project"]["readme"] == "README.md"
    assert pyproject["project"]["scripts"]["gdsdiff"] == "gdsdiff.cli:main"
    assert pyproject["project"]["scripts"]["gdsdiff-diagnostics"] == "gdsdiff.diagnostics:main"
    assert "gdstk>=1.0.0" in dependencies
    assert "numpy>=1.26" in dependencies
    assert "pytest>=8" in optional_dependencies["test"]
    assert "tomli; python_version < '3.11'" in optional_dependencies["test"]
    assert "cmake>=3.24" in optional_dependencies["native-dev"]
    assert "pybind11>=2.12" in optional_dependencies["native-dev"]
    assert pyproject["tool"]["setuptools"]["packages"]["find"]["where"] == ["src"]
    assert pyproject["tool"]["setuptools"]["packages"]["find"]["include"] == ["gdsdiff*"]
    assert "native_src/gdsdiff/*" in pyproject["tool"]["setuptools"]["package-data"]["gdsdiff"]
    assert (_repo_root() / "src" / "gdsdiff" / "native_src" / "gdsdiff" / "metal_ctypes.mm").exists()


def test_native_cache_path_uses_branded_external_override(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_NATIVE_CACHE_DIR, str(tmp_path))

    path = cached_native_library_path("_probe.so")

    assert path.parent.parent == tmp_path
    assert path.name == "_probe.so"


def test_native_cache_and_cuda_architecture_honor_legacy_fallbacks(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV_NATIVE_CACHE_DIR, raising=False)
    monkeypatch.setenv(LEGACY_ENV_NATIVE_CACHE_DIR, str(tmp_path))
    monkeypatch.delenv(CUDA_ARCH_ENV, raising=False)
    monkeypatch.setenv(LEGACY_CUDA_ARCH_ENV, "89")

    assert cached_native_library_path("_probe.so").parent.parent == tmp_path
    assert discover_native_capabilities().cuda_architectures == ("89",)


def test_cmake_options_keep_cuda_and_python_extension_optional():
    assert not (_repo_root() / "CMakeLists.txt").exists()
    native_cmake = (
        _repo_root() / "src" / "gdsdiff" / "native_src" / "gdsdiff" / "CMakeLists.txt"
    ).read_text(encoding="utf-8")

    assert "add_library(gds_compare_core STATIC core.cpp)" in native_cmake
    assert "find_package(pybind11 CONFIG REQUIRED)" in native_cmake
    assert "pybind11_add_module(_gds_compare_core python_bindings.cpp)" in native_cmake
    assert "pybind11_add_module(_gds_compare_cuda cuda_bindings.cu)" in native_cmake
    assert "option(GDSDIFF_ENABLE_CUDA" in native_cmake
    assert "GDS_GPU_COMPARE_ENABLE_CUDA" in native_cmake
    assert "set(CMAKE_CUDA_ARCHITECTURES native)" in native_cmake
