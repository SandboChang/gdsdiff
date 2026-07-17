from __future__ import annotations

import ctypes
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..native_cache import ensure_cached_native_library


class StructuralCtypesUnavailable(RuntimeError):
    pass


class _CStructuralResult(ctypes.Structure):
    _fields_ = [
        ("flattened_shape_count", ctypes.c_uint64),
        ("cell_count", ctypes.c_uint64),
        ("reachable_cell_count", ctypes.c_uint64),
        ("structural_hash", ctypes.c_uint64),
    ]


@dataclass(frozen=True)
class NativeStructuralSummary:
    path: Path
    flattened_shape_count: int
    cell_count: int
    reachable_cell_count: int
    structural_hash: int

    def to_json(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "flattened_shape_count": self.flattened_shape_count,
            "cell_count": self.cell_count,
            "reachable_cell_count": self.reachable_cell_count,
            "structural_hash": f"{self.structural_hash:016x}",
        }


def summarize_gds_structure_native(
    path: str | Path,
    *,
    top_name: str | None = None,
    counts_only: bool = False,
) -> NativeStructuralSummary:
    library = _load_library()
    result = _CStructuralResult()
    top_bytes = top_name.encode("utf-8") if top_name else None
    function = library.gds_structural_summary_counts_only if counts_only else library.gds_structural_summary
    status = function(
        str(path).encode("utf-8"),
        top_bytes,
        ctypes.byref(result),
    )
    if status != 0:
        message = library.gds_structural_last_error()
        raise StructuralCtypesUnavailable(message.decode("utf-8") if message else "native structural scan failed")
    return NativeStructuralSummary(
        path=Path(path),
        flattened_shape_count=int(result.flattened_shape_count),
        cell_count=int(result.cell_count),
        reachable_cell_count=int(result.reachable_cell_count),
        structural_hash=int(result.structural_hash),
    )


def structural_ctypes_available() -> bool:
    try:
        _load_library()
    except StructuralCtypesUnavailable:
        return False
    return True


def _load_library() -> ctypes.CDLL:
    path = _library_path()
    library = ctypes.CDLL(str(path))
    library.gds_structural_summary.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(_CStructuralResult),
    ]
    library.gds_structural_summary.restype = ctypes.c_int
    library.gds_structural_summary_counts_only.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(_CStructuralResult),
    ]
    library.gds_structural_summary_counts_only.restype = ctypes.c_int
    library.gds_structural_last_error.argtypes = []
    library.gds_structural_last_error.restype = ctypes.c_char_p
    return library


def _build_library(path: Path) -> None:
    source = _source_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "g++",
        "-std=c++17",
        "-O3",
        "-shared",
        "-fPIC",
        str(source),
        "-o",
        str(path),
    ]
    try:
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
    except FileNotFoundError as exc:
        raise StructuralCtypesUnavailable("g++ is not available") from exc
    if completed.returncode != 0:
        raise StructuralCtypesUnavailable((completed.stderr or completed.stdout).strip() or "g++ failed")


def _library_path() -> Path:
    return ensure_cached_native_library(
        "_gds_structural_scan.so",
        sources=(_source_path(),),
        build_key=("g++", "-std=c++17", "-O3", "-shared", "-fPIC"),
        builder=_build_library,
    )


def _source_path() -> Path:
    return Path(__file__).resolve().parents[1] / "native_src" / "gds_structural_scan.cpp"
