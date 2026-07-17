from __future__ import annotations

import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

from scgds import gdsdiff
from scgds.gdsdiff.native_cache import cached_native_library_path
from scgds.gdsdiff.native import CUDA_ARCH_ENV, discover_native_capabilities


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_package_import_does_not_import_optional_native_or_cuda_extensions():
    assert "scgds.gdsdiff._gds_compare_core" not in sys.modules
    assert "scgds.gdsdiff._gds_compare_cuda" not in sys.modules
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
    assert data["tool_version"] == "0.1.0-dev"


def test_pyproject_defines_cpu_safe_console_entry_point():
    pyproject = tomllib.loads((_repo_root() / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]
    optional_dependencies = pyproject["project"]["optional-dependencies"]

    assert pyproject["build-system"]["build-backend"] == "setuptools.build_meta"
    assert pyproject["project"]["name"] == "scgds"
    assert pyproject["project"]["readme"] == "README.md"
    assert pyproject["project"]["scripts"]["gdsdiff"] == "scgds.gdsdiff.cli:main"
    assert "gdstk>=1.0.0" in dependencies
    assert "numpy>=1.26" in dependencies
    assert optional_dependencies["test"] == ["pytest", "tomli; python_version < '3.11'"]
    assert optional_dependencies["native-dev"] == ["cmake", "pybind11"]
    assert pyproject["tool"]["setuptools"]["packages"]["find"]["include"] == ["scgds*"]
    assert "scgds.tests*" in pyproject["tool"]["setuptools"]["packages"]["find"]["exclude"]
    assert "native_src/gdsdiff/*" in pyproject["tool"]["setuptools"]["package-data"]["scgds.gdsdiff"]
    assert (_repo_root() / "scgds" / "gdsdiff" / "native_src" / "gdsdiff" / "metal_ctypes.mm").exists()


def test_native_cache_path_uses_external_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SCGDS_NATIVE_CACHE_DIR", str(tmp_path))

    path = cached_native_library_path("_probe.so")

    assert path.parent.parent == tmp_path
    assert path.name == "_probe.so"


def test_cmake_options_keep_cuda_and_python_extension_optional():
    assert not (_repo_root() / "CMakeLists.txt").exists()
    native_cmake = (
        _repo_root() / "scgds" / "gdsdiff" / "native_src" / "gdsdiff" / "CMakeLists.txt"
    ).read_text(encoding="utf-8")

    assert "add_library(gds_compare_core STATIC core.cpp)" in native_cmake
    assert "find_package(pybind11 CONFIG REQUIRED)" in native_cmake
    assert "pybind11_add_module(_gds_compare_core python_bindings.cpp)" in native_cmake
    assert "pybind11_add_module(_gds_compare_cuda cuda_bindings.cu)" in native_cmake
