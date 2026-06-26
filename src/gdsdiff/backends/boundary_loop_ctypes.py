from __future__ import annotations

import ctypes
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..exact_oracle import DiffFragment
from ..native_cache import cached_native_library_path


class BoundaryLoopCtypesUnavailable(RuntimeError):
    pass


class _CBoundaryLoopResult(ctypes.Structure):
    _fields_ = [
        ("fragment_count", ctypes.c_uint64),
        ("vertex_count", ctypes.c_uint64),
        ("vertices_xy", ctypes.POINTER(ctypes.c_int64)),
        ("fragment_offsets", ctypes.POINTER(ctypes.c_uint64)),
        ("bboxes", ctypes.POINTER(ctypes.c_int64)),
        ("twice_areas", ctypes.POINTER(ctypes.c_int64)),
        ("dropped_zero_area_loop_count", ctypes.c_uint64),
    ]


class _CBoundaryEdgesResult(ctypes.Structure):
    _fields_ = [
        ("edge_count", ctypes.c_uint64),
        ("edges", ctypes.POINTER(ctypes.c_int64)),
    ]


@dataclass(frozen=True)
class NativeBoundaryLoopResult:
    fragments: tuple[DiffFragment, ...]
    dropped_zero_area_loop_count: int


@dataclass(frozen=True)
class NativeBoundaryEdgesResult:
    edges: np.ndarray


def native_boundary_edges_from_fragments(fragments: tuple[DiffFragment, ...]) -> NativeBoundaryEdgesResult:
    library = _load_library()
    vertices, offsets = _pack_fragments(fragments)
    result = _CBoundaryEdgesResult()
    status = library.gds_boundary_edges_from_polygons(
        vertices.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        ctypes.c_uint64(len(fragments)),
        ctypes.byref(result),
    )
    if status != 0:
        message = library.gds_boundary_loops_last_error()
        raise BoundaryLoopCtypesUnavailable(message.decode("utf-8") if message else "native boundary edge assembly failed")
    try:
        edge_count = int(result.edge_count)
        edges = _copy_array(result.edges, edge_count * 4, np.int64).reshape((edge_count, 4))
        return NativeBoundaryEdgesResult(edges=edges)
    finally:
        library.gds_boundary_edges_free(ctypes.byref(result))


def native_boundary_loops_from_edges(edges: np.ndarray) -> NativeBoundaryLoopResult:
    library = _load_library()
    edges = np.ascontiguousarray(edges, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 4:
        raise ValueError("edges must have shape (N, 4)")
    result = _CBoundaryLoopResult()
    status = library.gds_boundary_loops_from_edges(
        edges.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_uint64(len(edges)),
        ctypes.byref(result),
    )
    if status != 0:
        message = library.gds_boundary_loops_last_error()
        raise BoundaryLoopCtypesUnavailable(message.decode("utf-8") if message else "native boundary loop assembly failed")
    try:
        count = int(result.fragment_count)
        vertex_count = int(result.vertex_count)
        vertices = _copy_array(result.vertices_xy, vertex_count * 2, np.int64).reshape((vertex_count, 2))
        offsets = _copy_array(result.fragment_offsets, count + 1, np.uint64)
        bboxes = _copy_array(result.bboxes, count * 4, np.int64).reshape((count, 4))
        twice_areas = _copy_array(result.twice_areas, count, np.int64)
        fragments = []
        for index in range(count):
            start = int(offsets[index])
            stop = int(offsets[index + 1])
            points = tuple((int(x), int(y)) for x, y in vertices[start:stop])
            bbox = tuple(int(value) for value in bboxes[index])
            fragments.append(DiffFragment(points=points, bbox=bbox, twice_area=int(twice_areas[index])))
        return NativeBoundaryLoopResult(
            fragments=tuple(fragments),
            dropped_zero_area_loop_count=int(result.dropped_zero_area_loop_count),
        )
    finally:
        library.gds_boundary_loops_free(ctypes.byref(result))


def boundary_loop_ctypes_available() -> bool:
    try:
        _load_library()
    except BoundaryLoopCtypesUnavailable:
        return False
    return True


def _copy_array(pointer, count: int, dtype) -> np.ndarray:
    if count == 0:
        return np.empty((0,), dtype=dtype)
    if not pointer:
        raise BoundaryLoopCtypesUnavailable("native boundary loop result returned a null array")
    ctype = np.ctypeslib.as_ctypes_type(dtype)
    array = ctypes.cast(pointer, ctypes.POINTER(ctype * count)).contents
    return np.frombuffer(array, dtype=dtype).copy()


def _load_library() -> ctypes.CDLL:
    path = _library_path()
    if not path.exists() or _source_path().stat().st_mtime > path.stat().st_mtime:
        _build_library(path)
    library = ctypes.CDLL(str(path))
    library.gds_boundary_edges_from_polygons.argtypes = [
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_uint64,
        ctypes.POINTER(_CBoundaryEdgesResult),
    ]
    library.gds_boundary_edges_from_polygons.restype = ctypes.c_int
    library.gds_boundary_edges_free.argtypes = [ctypes.POINTER(_CBoundaryEdgesResult)]
    library.gds_boundary_edges_free.restype = None
    library.gds_boundary_loops_from_edges.argtypes = [
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_uint64,
        ctypes.POINTER(_CBoundaryLoopResult),
    ]
    library.gds_boundary_loops_from_edges.restype = ctypes.c_int
    library.gds_boundary_loops_free.argtypes = [ctypes.POINTER(_CBoundaryLoopResult)]
    library.gds_boundary_loops_free.restype = None
    library.gds_boundary_loops_last_error.argtypes = []
    library.gds_boundary_loops_last_error.restype = ctypes.c_char_p
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
        raise BoundaryLoopCtypesUnavailable("g++ is not available") from exc
    if completed.returncode != 0:
        raise BoundaryLoopCtypesUnavailable((completed.stderr or completed.stdout).strip() or "g++ failed")


def _library_path() -> Path:
    return cached_native_library_path("_gds_boundary_loops.so")


def _source_path() -> Path:
    return Path(__file__).resolve().parents[1] / "native_src" / "gds_boundary_loops.cpp"


def _pack_fragments(fragments: tuple[DiffFragment, ...]) -> tuple[np.ndarray, np.ndarray]:
    vertex_rows: list[tuple[int, int]] = []
    offsets = [0]
    for fragment in fragments:
        vertex_rows.extend(fragment.points)
        offsets.append(len(vertex_rows))
    vertices = np.asarray(vertex_rows, dtype=np.int64).reshape((-1, 2))
    return vertices, np.asarray(offsets, dtype=np.uint64)
