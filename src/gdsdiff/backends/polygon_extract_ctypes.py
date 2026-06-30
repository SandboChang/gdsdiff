from __future__ import annotations

import ctypes
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import LayerSpec, parse_layer_selector
from ..geometry import LayerTable, PolygonBuffer
from ..native_cache import cached_native_library_path


class PolygonExtractCtypesUnavailable(RuntimeError):
    pass


class _CPolygonExtractResult(ctypes.Structure):
    _fields_ = [
        ("polygon_count", ctypes.c_uint64),
        ("vertex_count", ctypes.c_uint64),
        ("vertices_xy", ctypes.POINTER(ctypes.c_int64)),
        ("polygon_offsets", ctypes.POINTER(ctypes.c_uint64)),
        ("layers", ctypes.POINTER(ctypes.c_int32)),
        ("datatypes", ctypes.POINTER(ctypes.c_int32)),
        ("bboxes", ctypes.POINTER(ctypes.c_int64)),
        ("flags", ctypes.POINTER(ctypes.c_uint32)),
        ("source_ids", ctypes.POINTER(ctypes.c_uint64)),
        ("hashes", ctypes.POINTER(ctypes.c_uint64)),
    ]


@dataclass(frozen=True)
class NativePolygonExtraction:
    path: Path
    top_name: str | None
    buffer: PolygonBuffer
    native_parse_s: float | None = None


def extract_gds_polygons_native(
    path: str | Path,
    *,
    top_name: str | None = None,
    layers: str | LayerSpec | tuple[str | tuple[int, int] | LayerSpec, ...] = "all",
) -> NativePolygonExtraction:
    path = Path(path)
    library = _load_library()
    result = _CPolygonExtractResult()
    top_bytes = top_name.encode("utf-8") if top_name else None
    status = library.gds_polygon_extract(str(path).encode("utf-8"), top_bytes, ctypes.byref(result))
    if status != 0:
        message = library.gds_polygon_extract_last_error()
        raise PolygonExtractCtypesUnavailable(message.decode("utf-8") if message else "native polygon extraction failed")
    try:
        buffer = _buffer_from_result(result, layers=layers)
        return NativePolygonExtraction(path=path, top_name=top_name, buffer=buffer)
    finally:
        library.gds_polygon_extract_free(ctypes.byref(result))


def polygon_extract_ctypes_available() -> bool:
    try:
        _load_library()
    except PolygonExtractCtypesUnavailable:
        return False
    return True


def _buffer_from_result(
    result: _CPolygonExtractResult,
    *,
    layers: str | LayerSpec | tuple[str | tuple[int, int] | LayerSpec, ...],
) -> PolygonBuffer:
    polygon_count = int(result.polygon_count)
    vertex_count = int(result.vertex_count)
    vertices = _copy_array(result.vertices_xy, vertex_count * 2, np.int64).reshape((vertex_count, 2))
    offsets = _copy_array(result.polygon_offsets, polygon_count + 1, np.uint64)
    raw_layers = _copy_array(result.layers, polygon_count, np.int32)
    raw_datatypes = _copy_array(result.datatypes, polygon_count, np.int32)
    bboxes = _copy_array(result.bboxes, polygon_count * 4, np.int64).reshape((polygon_count, 4))
    flags = _copy_array(result.flags, polygon_count, np.uint32)
    source_ids = _copy_array(result.source_ids, polygon_count, np.uint64)
    selected = _selected_layers(layers)
    hashes = _copy_array(result.hashes, polygon_count * 2, np.uint64).reshape((polygon_count, 2))
    if selected is not None:
        keep = _selected_mask(raw_layers, raw_datatypes, selected)
        if not np.all(keep):
            vertices, offsets, raw_layers, raw_datatypes, bboxes, flags, source_ids, hashes = _filter_polygons(
                vertices,
                offsets,
                raw_layers,
                raw_datatypes,
                bboxes,
                flags,
                source_ids,
                hashes,
                keep,
            )
    table, layer_ids = _build_layer_table(raw_layers, raw_datatypes)
    return PolygonBuffer(
        vertices_xy=vertices.astype(np.int64, copy=False),
        polygon_offsets=offsets.astype(np.uint64, copy=False),
        layer_ids=layer_ids,
        bboxes=bboxes.astype(np.int64, copy=False),
        flags=flags.astype(np.uint32, copy=False),
        source_ids=source_ids.astype(np.uint64, copy=False),
        layer_table=table,
        polygon_hashes=hashes.astype(np.uint64, copy=False),
    )


def _selected_layers(layers) -> tuple[LayerSpec, ...] | None:
    if layers == "all":
        return None
    parsed = parse_layer_selector(layers)
    if parsed == "all":
        return None
    return parsed


def _selected_mask(raw_layers: np.ndarray, raw_datatypes: np.ndarray, selected: tuple[LayerSpec, ...]) -> np.ndarray:
    keep = np.zeros(raw_layers.shape, dtype=bool)
    for layer in selected:
        keep |= (raw_layers == layer.layer) & (raw_datatypes == layer.datatype)
    return keep


def _build_layer_table(raw_layers: np.ndarray, raw_datatypes: np.ndarray) -> tuple[LayerTable, np.ndarray]:
    if len(raw_layers) == 0:
        return LayerTable(()), np.empty((0,), dtype=np.uint32)
    pairs = np.column_stack((raw_layers, raw_datatypes)).astype(np.int32, copy=False)
    unique_pairs = np.unique(pairs, axis=0)
    table = LayerTable(tuple(LayerSpec(int(layer), int(datatype)) for layer, datatype in unique_pairs))
    layer_ids = np.empty(len(raw_layers), dtype=np.uint32)
    for index, (layer, datatype) in enumerate(unique_pairs):
        layer_ids[(raw_layers == layer) & (raw_datatypes == datatype)] = index
    return table, layer_ids


def _filter_polygons(
    vertices: np.ndarray,
    offsets: np.ndarray,
    raw_layers: np.ndarray,
    raw_datatypes: np.ndarray,
    bboxes: np.ndarray,
    flags: np.ndarray,
    source_ids: np.ndarray,
    hashes: np.ndarray,
    keep: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    kept_indices = np.nonzero(keep)[0]
    new_vertices: list[np.ndarray] = []
    new_offsets = [0]
    for index in kept_indices:
        start = int(offsets[index])
        stop = int(offsets[index + 1])
        part = vertices[start:stop]
        new_vertices.append(part)
        new_offsets.append(new_offsets[-1] + len(part))
    return (
        np.vstack(new_vertices).astype(np.int64, copy=False) if new_vertices else np.empty((0, 2), dtype=np.int64),
        np.asarray(new_offsets, dtype=np.uint64),
        raw_layers[keep].astype(np.int32, copy=False),
        raw_datatypes[keep].astype(np.int32, copy=False),
        bboxes[keep].astype(np.int64, copy=False),
        flags[keep].astype(np.uint32, copy=False),
        source_ids[keep].astype(np.uint64, copy=False),
        hashes[keep].astype(np.uint64, copy=False),
    )


def _copy_array(pointer, count: int, dtype) -> np.ndarray:
    if count == 0:
        return np.empty((0,), dtype=dtype)
    if not pointer:
        raise PolygonExtractCtypesUnavailable("native polygon extraction returned a null array")
    ctype = np.ctypeslib.as_ctypes_type(dtype)
    array = ctypes.cast(pointer, ctypes.POINTER(ctype * count)).contents
    return np.frombuffer(array, dtype=dtype).copy()


def _load_library() -> ctypes.CDLL:
    path = _library_path()
    if not path.exists() or _source_path().stat().st_mtime > path.stat().st_mtime:
        _build_library(path)
    library = ctypes.CDLL(str(path))
    library.gds_polygon_extract.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(_CPolygonExtractResult),
    ]
    library.gds_polygon_extract.restype = ctypes.c_int
    library.gds_polygon_extract_free.argtypes = [ctypes.POINTER(_CPolygonExtractResult)]
    library.gds_polygon_extract_free.restype = None
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
    return cached_native_library_path("_gds_polygon_extract.so")


def _source_path() -> Path:
    return Path(__file__).resolve().parents[1] / "native_src" / "gds_polygon_extract.cpp"
