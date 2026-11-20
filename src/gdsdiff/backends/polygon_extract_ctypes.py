from __future__ import annotations

import ctypes
import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np

from ..config import LayerSpec, parse_layer_selector
from ..geometry import LayerTable, PolygonBuffer
from ..native_cache import ensure_cached_native_library


_MAX_NATIVE_LAYER_VALUE = (1 << 15) - 1


class PolygonExtractCtypesUnavailable(RuntimeError):
    pass


class _CPlanInfo(ctypes.Structure):
    _fields_ = [
        ("polygon_count", ctypes.c_uint64),
        ("vertex_count", ctypes.c_uint64),
        ("layer_count", ctypes.c_uint64),
        ("read_s", ctypes.c_double),
        ("scan_s", ctypes.c_double),
        ("normalize_s", ctypes.c_double),
        ("layer_table_s", ctypes.c_double),
        ("placement_s", ctypes.c_double),
        ("oracle_reuse_eligible", ctypes.c_uint32),
    ]


class _COutputArrays(ctypes.Structure):
    _fields_ = [
        ("vertices_xy", ctypes.POINTER(ctypes.c_int64)),
        ("polygon_offsets", ctypes.POINTER(ctypes.c_uint64)),
        ("layer_ids", ctypes.POINTER(ctypes.c_uint32)),
        ("bboxes", ctypes.POINTER(ctypes.c_int64)),
        ("flags", ctypes.POINTER(ctypes.c_uint32)),
        ("source_ids", ctypes.POINTER(ctypes.c_uint64)),
        ("hashes", ctypes.POINTER(ctypes.c_uint64)),
        ("layer_pairs", ctypes.POINTER(ctypes.c_int32)),
    ]


@dataclass(frozen=True)
class NativePolygonExtraction:
    path: Path
    top_name: str | None
    buffer: PolygonBuffer
    native_parse_s: float | None = None
    oracle_reuse_eligible: bool = False
    phase_timings: dict[str, float] | None = None

    def to_timing_json(self, prefix: str) -> dict[str, float | bool]:
        if self.phase_timings is None:
            return {}
        return {
            **{f"{prefix}_native_polygon_{key}": value for key, value in self.phase_timings.items()},
            f"{prefix}_native_polygon_cpu_hashes": self.buffer.polygon_hashes is not None,
        }


def extract_gds_polygons_native(
    path: str | Path,
    *,
    top_name: str | None = None,
    layers: str | LayerSpec | tuple[str | tuple[int, int] | LayerSpec, ...] = "all",
    compute_hashes: bool = True,
) -> NativePolygonExtraction:
    start = perf_counter()
    path = Path(path)
    library = _load_library()
    info = _CPlanInfo()
    handle = ctypes.c_void_p()
    selected = parse_layer_selector(layers)
    pairs = (
        None if selected == "all" else np.asarray(
            [(s.layer, s.datatype) for s in selected if max(s.layer, s.datatype) <= _MAX_NATIVE_LAYER_VALUE],
            dtype=np.int32,
        )
    )
    top_bytes = top_name.encode("utf-8") if top_name else None
    _check_status(library, library.gds_polygon_extract_plan(
        os.fsencode(path),
        top_bytes,
        None if pairs is None else pairs.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        0 if pairs is None else len(pairs),
        pairs is None,
        _workers_per_input(),
        ctypes.byref(info),
        ctypes.byref(handle),
    ))
    try:
        buffer, pack_s = _emit_buffer(library, handle, info, compute_hashes)
    finally:
        library.gds_polygon_extract_plan_free(handle)
    return NativePolygonExtraction(
        path=path,
        top_name=top_name,
        buffer=buffer,
        native_parse_s=perf_counter() - start,
        oracle_reuse_eligible=bool(info.oracle_reuse_eligible),
        phase_timings={
            "read_s": info.read_s,
            "scan_s": info.scan_s,
            "normalize_s": info.normalize_s,
            "layer_table_s": info.layer_table_s,
            "placement_s": info.placement_s,
            "pack_s": pack_s,
        },
    )


def polygon_extract_ctypes_available() -> bool:
    try:
        _load_library()
    except PolygonExtractCtypesUnavailable:
        return False
    return True


def _workers_per_input() -> int:
    try:
        available = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available = os.cpu_count() or 1
    return max(1, min(available // 2, 12))


def _check_status(library: ctypes.CDLL, status: int) -> None:
    if status != 0:
        message = library.gds_polygon_extract_last_error()
        raise PolygonExtractCtypesUnavailable(message.decode("utf-8") if message else "native polygon extraction failed")


def _emit_buffer(library, handle, info, compute_hashes) -> tuple[PolygonBuffer, float]:
    polygons, vertices = info.polygon_count, info.vertex_count
    vertices_xy = np.empty((vertices, 2), dtype=np.int64)
    offsets = np.empty(polygons + 1, dtype=np.uint64)
    layer_ids = np.empty(polygons, dtype=np.uint32)
    bboxes = np.empty((polygons, 4), dtype=np.int64)
    flags = np.empty(polygons, dtype=np.uint32)
    source_ids = np.empty(polygons, dtype=np.uint64)
    hashes = np.empty((polygons, 2), dtype=np.uint64) if compute_hashes else None
    layer_pairs = np.empty((info.layer_count, 2), dtype=np.int32)
    arrays = (vertices_xy, offsets, layer_ids, bboxes, flags, source_ids, hashes, layer_pairs)
    output = _COutputArrays(*(
        a.ctypes.data_as(pointer_type) if a is not None else None
        for a, (_, pointer_type) in zip(arrays, _COutputArrays._fields_, strict=True)
    ))
    pack_s = ctypes.c_double()
    _check_status(library, library.gds_polygon_extract_emit(handle, ctypes.byref(output), ctypes.byref(pack_s)))
    return PolygonBuffer(
        vertices_xy=vertices_xy,
        polygon_offsets=offsets,
        layer_ids=layer_ids,
        bboxes=bboxes,
        flags=flags,
        source_ids=source_ids,
        layer_table=LayerTable(tuple(LayerSpec(int(layer), int(datatype)) for layer, datatype in layer_pairs)),
        polygon_hashes=hashes,
    ), pack_s.value


@lru_cache(maxsize=1)
def _load_library() -> ctypes.CDLL:
    path = _library_path()
    library = ctypes.CDLL(str(path))
    library.gds_polygon_extract_plan.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_uint64,
        ctypes.c_bool,
        ctypes.c_uint,
        ctypes.POINTER(_CPlanInfo),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.gds_polygon_extract_plan.restype = ctypes.c_int
    library.gds_polygon_extract_emit.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_COutputArrays),
        ctypes.POINTER(ctypes.c_double),
    ]
    library.gds_polygon_extract_emit.restype = ctypes.c_int
    library.gds_polygon_extract_plan_free.argtypes = [ctypes.c_void_p]
    library.gds_polygon_extract_plan_free.restype = None
    library.gds_polygon_extract_last_error.argtypes = []
    library.gds_polygon_extract_last_error.restype = ctypes.c_char_p
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
        "-pthread",
        str(source),
        "-o",
        str(path),
    ]
    try:
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
    except FileNotFoundError as exc:
        raise PolygonExtractCtypesUnavailable("g++ is not available") from exc
    if completed.returncode != 0:
        raise PolygonExtractCtypesUnavailable((completed.stderr or completed.stdout).strip() or "g++ failed")


def _library_path() -> Path:
    return ensure_cached_native_library(
        "_gds_polygon_extract.so",
        sources=(_source_path(),),
        build_key=("g++", "-std=c++17", "-O3", "-shared", "-fPIC", "-pthread"),
        builder=_build_library,
    )


def _source_path() -> Path:
    return Path(__file__).resolve().parents[1] / "native_src" / "gds_polygon_extract.cpp"
