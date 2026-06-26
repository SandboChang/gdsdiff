from __future__ import annotations

import ctypes
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import LayerSpec
from ..geometry import PolygonBuffer
from ..native_cache import cached_native_library_path
from ..tiling import CpuTilePrefilterResult, OversizedPolygon, TileConfig, TileKey


class CpuCtypesUnavailable(RuntimeError):
    pass


class _CResult(ctypes.Structure):
    _fields_ = [
        ("candidate_tiles", ctypes.POINTER(ctypes.c_int64)),
        ("candidate_count", ctypes.c_uint64),
        ("oversized", ctypes.POINTER(ctypes.c_int64)),
        ("oversized_count", ctypes.c_uint64),
        ("assignment_count", ctypes.c_uint64),
        ("collapsed_pair_count", ctypes.c_uint64),
        ("oversized_matched_count", ctypes.c_uint64),
    ]


@dataclass(frozen=True)
class _PackedBuffer:
    bboxes: np.ndarray
    layers: np.ndarray
    datatypes: np.ndarray
    canonical_ids: np.ndarray


def run_cpu_ctypes_tile_prefilter(
    old_buffer: PolygonBuffer,
    old_canonical_ids,
    new_buffer: PolygonBuffer,
    new_canonical_ids,
    config: TileConfig,
) -> CpuTilePrefilterResult:
    library = _load_library()
    old_packed = _pack_buffer(old_buffer, old_canonical_ids)
    new_packed = _pack_buffer(new_buffer, new_canonical_ids)
    old_packed, new_packed, skipped_matched = _skip_exact_tile_bbox_matches(old_packed, new_packed, config)
    result = _CResult()
    status = library.gds_cpu_prefilter(
        _ptr_i64(old_packed.bboxes),
        _ptr_i32(old_packed.layers),
        _ptr_i32(old_packed.datatypes),
        _ptr_u64(old_packed.canonical_ids),
        ctypes.c_uint64(len(old_packed.canonical_ids)),
        _ptr_i64(new_packed.bboxes),
        _ptr_i32(new_packed.layers),
        _ptr_i32(new_packed.datatypes),
        _ptr_u64(new_packed.canonical_ids),
        ctypes.c_uint64(len(new_packed.canonical_ids)),
        ctypes.c_int64(config.tile_size),
        ctypes.c_uint64(config.max_tiles_per_polygon),
        ctypes.byref(result),
    )
    if status != 0:
        message = library.gds_cpu_last_error()
        raise CpuCtypesUnavailable(message.decode("utf-8") if message else "native CPU prefilter failed")
    try:
        return CpuTilePrefilterResult(
            candidate_tiles=_candidate_tiles_from_result(result),
            oversized_candidates=_oversized_from_result(result),
            assignment_count=int(result.assignment_count),
            collapsed_pair_count=int(result.collapsed_pair_count),
            oversized_matched_count=int(result.oversized_matched_count) + skipped_matched,
        )
    finally:
        library.gds_cpu_free_result(ctypes.byref(result))


def cpu_ctypes_available() -> bool:
    try:
        _load_library()
    except CpuCtypesUnavailable:
        return False
    return True


def _candidate_tiles_from_result(result: _CResult) -> tuple[TileKey, ...]:
    if result.candidate_count == 0:
        return ()
    array = np.ctypeslib.as_array(result.candidate_tiles, shape=(int(result.candidate_count) * 4,))
    return tuple(
        TileKey(LayerSpec(int(array[index]), int(array[index + 1])), int(array[index + 2]), int(array[index + 3]))
        for index in range(0, len(array), 4)
    )


def _oversized_from_result(result: _CResult) -> tuple[OversizedPolygon, ...]:
    if result.oversized_count == 0:
        return ()
    array = np.ctypeslib.as_array(result.oversized, shape=(int(result.oversized_count) * 8,))
    return tuple(
        OversizedPolygon(
            layer=LayerSpec(int(array[index]), int(array[index + 1])),
            canonical_id=int(array[index + 2]),
            side_mask=int(array[index + 3]),
            tile_bbox=(int(array[index + 4]), int(array[index + 5]), int(array[index + 6]), int(array[index + 7])),
        )
        for index in range(0, len(array), 8)
    )


def _pack_buffer(buffer: PolygonBuffer, canonical_ids) -> _PackedBuffer:
    ids = np.ascontiguousarray(canonical_ids, dtype=np.uint64)
    if len(ids) != buffer.polygon_count:
        raise ValueError("canonical id vector length must match polygon count")
    layers = np.empty(buffer.polygon_count, dtype=np.int32)
    datatypes = np.empty(buffer.polygon_count, dtype=np.int32)
    for index, layer_id in enumerate(buffer.layer_ids):
        layer = buffer.layer_table.layers[int(layer_id)]
        layers[index] = layer.layer
        datatypes[index] = layer.datatype
    return _PackedBuffer(
        bboxes=np.ascontiguousarray(buffer.bboxes, dtype=np.int64),
        layers=np.ascontiguousarray(layers, dtype=np.int32),
        datatypes=np.ascontiguousarray(datatypes, dtype=np.int32),
        canonical_ids=ids,
    )


def _skip_exact_tile_bbox_matches(old: _PackedBuffer, new: _PackedBuffer, config: TileConfig) -> tuple[_PackedBuffer, _PackedBuffer, int]:
    side_masks: dict[tuple[int, int, int, int, int, int, int], int] = {}
    old_first: dict[tuple[int, int, int, int, int, int, int], int] = {}
    new_first: dict[tuple[int, int, int, int, int, int, int], int] = {}
    for index, key in enumerate(_tile_bbox_keys(old, config.tile_size)):
        side_masks[key] = side_masks.get(key, 0) | 1
        old_first.setdefault(key, index)
    for index, key in enumerate(_tile_bbox_keys(new, config.tile_size)):
        side_masks[key] = side_masks.get(key, 0) | 2
        new_first.setdefault(key, index)
    old_keep = [old_first[key] for key, mask in side_masks.items() if mask == 1]
    new_keep = [new_first[key] for key, mask in side_masks.items() if mask == 2]
    skipped = sum(1 for mask in side_masks.values() if mask == 3)
    return _take_packed(old, old_keep), _take_packed(new, new_keep), skipped


def _tile_bbox_keys(packed: _PackedBuffer, tile_size: int):
    tx0 = np.floor_divide(packed.bboxes[:, 0], tile_size)
    ty0 = np.floor_divide(packed.bboxes[:, 1], tile_size)
    tx1 = np.floor_divide(packed.bboxes[:, 2], tile_size)
    ty1 = np.floor_divide(packed.bboxes[:, 3], tile_size)
    for index in range(len(packed.canonical_ids)):
        yield (
            int(packed.layers[index]),
            int(packed.datatypes[index]),
            int(packed.canonical_ids[index]),
            int(tx0[index]),
            int(ty0[index]),
            int(tx1[index]),
            int(ty1[index]),
        )


def _take_packed(packed: _PackedBuffer, indices: list[int]) -> _PackedBuffer:
    if not indices:
        return _PackedBuffer(
            bboxes=np.empty((0, 4), dtype=np.int64),
            layers=np.empty((0,), dtype=np.int32),
            datatypes=np.empty((0,), dtype=np.int32),
            canonical_ids=np.empty((0,), dtype=np.uint64),
        )
    index_array = np.asarray(indices, dtype=np.int64)
    return _PackedBuffer(
        bboxes=np.ascontiguousarray(packed.bboxes[index_array], dtype=np.int64),
        layers=np.ascontiguousarray(packed.layers[index_array], dtype=np.int32),
        datatypes=np.ascontiguousarray(packed.datatypes[index_array], dtype=np.int32),
        canonical_ids=np.ascontiguousarray(packed.canonical_ids[index_array], dtype=np.uint64),
    )


def _ptr_i64(array: np.ndarray):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))


def _ptr_i32(array: np.ndarray):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


def _ptr_u64(array: np.ndarray):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))


def _load_library() -> ctypes.CDLL:
    path = _library_path()
    if not path.exists() or _source_path().stat().st_mtime > path.stat().st_mtime:
        _build_library(path)
    library = ctypes.CDLL(str(path))
    library.gds_cpu_prefilter.argtypes = [
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_uint64,
        ctypes.c_int64,
        ctypes.c_uint64,
        ctypes.POINTER(_CResult),
    ]
    library.gds_cpu_prefilter.restype = ctypes.c_int
    library.gds_cpu_last_error.argtypes = []
    library.gds_cpu_last_error.restype = ctypes.c_char_p
    library.gds_cpu_free_result.argtypes = [ctypes.POINTER(_CResult)]
    library.gds_cpu_free_result.restype = None
    return library


def _build_library(path: Path) -> None:
    source = _source_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "g++",
        "-std=c++17",
        "-O3",
        "-fopenmp",
        "-shared",
        "-fPIC",
        str(source),
        "-o",
        str(path),
    ]
    try:
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
    except FileNotFoundError as exc:
        raise CpuCtypesUnavailable("g++ is not available") from exc
    if completed.returncode != 0:
        raise CpuCtypesUnavailable((completed.stderr or completed.stdout).strip() or "g++ failed")


def _library_path() -> Path:
    return cached_native_library_path("_gds_compare_cpu_ctypes.so")


def _source_path() -> Path:
    return Path(__file__).resolve().parents[1] / "native_src" / "cpu_prefilter.cpp"
