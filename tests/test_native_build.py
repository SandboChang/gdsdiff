from __future__ import annotations

import sys
import tomllib
from pathlib import Path

from gds_tools_tk import gdsdiff
from gds_tools_tk.gdsdiff.native import CUDA_ARCH_ENV, discover_native_capabilities


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_package_import_does_not_import_optional_native_or_cuda_extensions():
    assert "gds_tools_tk.gdsdiff._gds_compare_core" not in sys.modules
    assert "gds_tools_tk.gdsdiff._gds_compare_cuda" not in sys.modules
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
    assert data["cuda_architectures"] == ["75", "86", "89"]
    assert data["tool_version"] == "0.1.0-dev"


def test_pyproject_defines_cpu_safe_console_entry_point():
    pyproject = tomllib.loads((_repo_root() / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["build-system"]["build-backend"] == "setuptools.build_meta"
    assert pyproject["project"]["scripts"]["gdsdiff"] == "gds_tools_tk.gdsdiff.cli:main"
    assert "gdstk" in pyproject["project"]["dependencies"]
    assert "numpy" in pyproject["project"]["dependencies"]


def test_cmake_options_keep_cuda_and_python_extension_optional():
    assert not (_repo_root() / "CMakeLists.txt").exists()
    native_cmake = (
        _repo_root() / "gds_tools_tk" / "gdsdiff" / "native_src" / "gdsdiff" / "CMakeLists.txt"
    ).read_text(encoding="utf-8")

    assert "add_library(gds_compare_core STATIC core.cpp)" in native_cmake
    assert "find_package(pybind11 CONFIG REQUIRED)" in native_cmake
    assert "pybind11_add_module(_gds_compare_core python_bindings.cpp)" in native_cmake
    assert "pybind11_add_module(_gds_compare_cuda cuda_bindings.cu)" in native_cmake
