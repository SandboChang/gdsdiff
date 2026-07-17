from __future__ import annotations

import ctypes
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ..config import LayerSpec
from ..native_cache import ensure_cached_native_library


class GeometryCtypesUnavailable(RuntimeError):
    pass


class _CGeometryLayerResult(ctypes.Structure):
    _fields_ = [
        ("layer", ctypes.c_int32),
        ("datatype", ctypes.c_int32),
        ("count", ctypes.c_uint64),
        ("x0", ctypes.c_int64),
        ("y0", ctypes.c_int64),
        ("x1", ctypes.c_int64),
        ("y1", ctypes.c_int64),
    ]


class _CGeometrySummaryResult(ctypes.Structure):
    _fields_ = [
        ("flattened_shape_count", ctypes.c_uint64),
        ("layer_count", ctypes.c_uint64),
        ("geometry_hash", ctypes.c_uint64),
        ("layers", ctypes.POINTER(_CGeometryLayerResult)),
    ]


class _CGeometryCompareResult(ctypes.Structure):
    _fields_ = [
        ("old_flattened_shape_count", ctypes.c_uint64),
        ("new_flattened_shape_count", ctypes.c_uint64),
        ("old_layer_count", ctypes.c_uint64),
        ("new_layer_count", ctypes.c_uint64),
        ("old_geometry_hash", ctypes.c_uint64),
        ("new_geometry_hash", ctypes.c_uint64),
        ("summaries_match", ctypes.c_uint8),
        ("detail_computed", ctypes.c_uint8),
        ("old_layers", ctypes.POINTER(_CGeometryLayerResult)),
        ("new_layers", ctypes.POINTER(_CGeometryLayerResult)),
    ]


@dataclass(frozen=True)
class NativeGeometryLayerSummary:
    layer: LayerSpec
    count: int
    bbox: tuple[int, int, int, int]

    def to_json(self) -> dict[str, object]:
        return {"layer": self.layer.to_json(), "count": self.count, "bbox": list(self.bbox)}


@dataclass(frozen=True)
class NativeGeometrySummary:
    path: Path
    flattened_shape_count: int
    geometry_hash: int
    layers: tuple[NativeGeometryLayerSummary, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "flattened_shape_count": self.flattened_shape_count,
            "geometry_hash": f"{self.geometry_hash:016x}",
            "layers": [layer.to_json() for layer in self.layers],
        }


@dataclass(frozen=True)
class NativeGeometryCompare:
    old_summary: NativeGeometrySummary
    new_summary: NativeGeometrySummary
    summaries_match: bool
    detail_computed: bool


def summarize_gds_geometry_native(path: str | Path, *, top_name: str | None = None, counts_only: bool = False) -> NativeGeometrySummary:
    library = _load_library()
    result = _CGeometrySummaryResult()
    top_bytes = top_name.encode("utf-8") if top_name else None
    function = library.gds_geometry_summary_counts_only if counts_only else library.gds_geometry_summary
    status = function(str(path).encode("utf-8"), top_bytes, ctypes.byref(result))
    if status != 0:
        message = library.gds_geometry_last_error()
        raise GeometryCtypesUnavailable(message.decode("utf-8") if message else "native geometry scan failed")
    try:
        layers = []
        if result.layer_count:
            array = ctypes.cast(
                result.layers,
                ctypes.POINTER(_CGeometryLayerResult * int(result.layer_count)),
            ).contents
            for item in array:
                layers.append(
                    NativeGeometryLayerSummary(
                        layer=LayerSpec(int(item.layer), int(item.datatype)),
                        count=int(item.count),
                        bbox=(int(item.x0), int(item.y0), int(item.x1), int(item.y1)),
                    )
                )
        return NativeGeometrySummary(
            path=Path(path),
            flattened_shape_count=int(result.flattened_shape_count),
            geometry_hash=int(result.geometry_hash),
            layers=tuple(sorted(layers, key=lambda item: item.layer)),
        )
    finally:
        library.gds_geometry_free_summary(ctypes.byref(result))


def compare_gds_geometry_native(
    old_path: str | Path,
    new_path: str | Path,
    *,
    old_top_name: str | None = None,
    new_top_name: str | None = None,
) -> NativeGeometryCompare:
    library = _load_library()
    result = _CGeometryCompareResult()
    old_top_bytes = old_top_name.encode("utf-8") if old_top_name else None
    new_top_bytes = new_top_name.encode("utf-8") if new_top_name else None
    status = library.gds_geometry_compare(
        str(old_path).encode("utf-8"),
        str(new_path).encode("utf-8"),
        old_top_bytes,
        new_top_bytes,
        ctypes.byref(result),
    )
    if status != 0:
        message = library.gds_geometry_last_error()
        raise GeometryCtypesUnavailable(message.decode("utf-8") if message else "native geometry compare failed")
    try:
        old_layers = _layers_from_pointer(result.old_layers, int(result.old_layer_count))
        new_layers = _layers_from_pointer(result.new_layers, int(result.new_layer_count))
        return NativeGeometryCompare(
            old_summary=NativeGeometrySummary(
                path=Path(old_path),
                flattened_shape_count=int(result.old_flattened_shape_count),
                geometry_hash=int(result.old_geometry_hash),
                layers=old_layers,
            ),
            new_summary=NativeGeometrySummary(
                path=Path(new_path),
                flattened_shape_count=int(result.new_flattened_shape_count),
                geometry_hash=int(result.new_geometry_hash),
                layers=new_layers,
            ),
            summaries_match=bool(result.summaries_match),
            detail_computed=bool(result.detail_computed),
        )
    finally:
        library.gds_geometry_free_compare(ctypes.byref(result))


def geometry_ctypes_available() -> bool:
    try:
        _load_library()
    except GeometryCtypesUnavailable:
        return False
    return True


@lru_cache(maxsize=1)
def _load_library() -> ctypes.CDLL:
    path = _library_path()
    library = ctypes.CDLL(str(path))
    library.gds_geometry_summary.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(_CGeometrySummaryResult),
    ]
    library.gds_geometry_summary.restype = ctypes.c_int
    library.gds_geometry_summary_counts_only.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(_CGeometrySummaryResult),
    ]
    library.gds_geometry_summary_counts_only.restype = ctypes.c_int
    library.gds_geometry_compare.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(_CGeometryCompareResult),
    ]
    library.gds_geometry_compare.restype = ctypes.c_int
    library.gds_geometry_free_summary.argtypes = [ctypes.POINTER(_CGeometrySummaryResult)]
    library.gds_geometry_free_summary.restype = None
    library.gds_geometry_free_compare.argtypes = [ctypes.POINTER(_CGeometryCompareResult)]
    library.gds_geometry_free_compare.restype = None
    library.gds_geometry_last_error.argtypes = []
    library.gds_geometry_last_error.restype = ctypes.c_char_p
    return library


def _layers_from_pointer(pointer, count: int) -> tuple[NativeGeometryLayerSummary, ...]:
    if count == 0:
        return ()
    array = ctypes.cast(pointer, ctypes.POINTER(_CGeometryLayerResult * count)).contents
    layers = [
        NativeGeometryLayerSummary(
            layer=LayerSpec(int(item.layer), int(item.datatype)),
            count=int(item.count),
            bbox=(int(item.x0), int(item.y0), int(item.x1), int(item.y1)),
        )
        for item in array
    ]
    return tuple(sorted(layers, key=lambda item: item.layer))


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
        raise GeometryCtypesUnavailable("g++ is not available") from exc
    if completed.returncode != 0:
        raise GeometryCtypesUnavailable((completed.stderr or completed.stdout).strip() or "g++ failed")


def _library_path() -> Path:
    return ensure_cached_native_library(
        "_gds_geometry_scan.so",
        sources=(_source_path(),),
        build_key=("g++", "-std=c++17", "-O3", "-shared", "-fPIC"),
        builder=_build_library,
    )


def _source_path() -> Path:
    return Path(__file__).resolve().parents[1] / "native_src" / "gds_geometry_scan.cpp"
