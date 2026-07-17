from __future__ import annotations

import ctypes
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import LayerSpec
from ..exact_oracle import DiffFragment
from ..geometry import PolygonBuffer
from ..native_cache import cached_native_library_path
from ..polygon_components import PolygonComponent
from ..tiling import CpuTilePrefilterResult, OversizedPolygon, TileConfig, TileKey


class CudaCtypesUnavailable(RuntimeError):
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


class _CByteCompareResult(ctypes.Structure):
    _fields_ = [
        ("old_size", ctypes.c_uint64),
        ("new_size", ctypes.c_uint64),
        ("same", ctypes.c_uint8),
    ]


class _CPolygonRecordDeltaResult(ctypes.Structure):
    _fields_ = [
        ("record_count", ctypes.c_uint64),
        ("layers", ctypes.POINTER(ctypes.c_int32)),
        ("datatypes", ctypes.POINTER(ctypes.c_int32)),
        ("directions", ctypes.POINTER(ctypes.c_uint8)),
        ("counts", ctypes.POINTER(ctypes.c_uint64)),
        ("bboxes", ctypes.POINTER(ctypes.c_int64)),
        ("vertex_counts", ctypes.POINTER(ctypes.c_uint64)),
        ("flags", ctypes.POINTER(ctypes.c_uint32)),
        ("hashes", ctypes.POINTER(ctypes.c_uint64)),
        ("representative_indices", ctypes.POINTER(ctypes.c_uint64)),
        ("old_input_count", ctypes.c_uint64),
        ("new_input_count", ctypes.c_uint64),
        ("matched_record_count", ctypes.c_uint64),
        ("reduced_key_count", ctypes.c_uint64),
        ("cuda_kernel_count", ctypes.c_uint64),
        ("cuda_event_elapsed_ms", ctypes.c_float),
    ]


class _CBoundaryLoopResult(ctypes.Structure):
    _fields_ = [
        ("fragment_count", ctypes.c_uint64),
        ("vertex_count", ctypes.c_uint64),
        ("vertices_xy", ctypes.POINTER(ctypes.c_int64)),
        ("fragment_offsets", ctypes.POINTER(ctypes.c_uint64)),
        ("bboxes", ctypes.POINTER(ctypes.c_int64)),
        ("twice_areas", ctypes.POINTER(ctypes.c_int64)),
        ("dropped_zero_area_loop_count", ctypes.c_uint64),
        ("cuda_kernel_count", ctypes.c_uint64),
        ("cuda_event_elapsed_ms", ctypes.c_float),
    ]


class _CBoundaryEdgesResult(ctypes.Structure):
    _fields_ = [
        ("edge_count", ctypes.c_uint64),
        ("edges", ctypes.POINTER(ctypes.c_int64)),
        ("cuda_kernel_count", ctypes.c_uint64),
        ("cuda_event_elapsed_ms", ctypes.c_float),
    ]


class _CContextOverlapPairsResult(ctypes.Structure):
    _fields_ = [
        ("pair_count", ctypes.c_uint64),
        ("component_indices", ctypes.POINTER(ctypes.c_uint64)),
        ("polygon_indices", ctypes.POINTER(ctypes.c_uint64)),
        ("cuda_kernel_count", ctypes.c_uint64),
        ("cuda_event_elapsed_ms", ctypes.c_float),
    ]


@dataclass(frozen=True)
class _PackedBuffer:
    bboxes: np.ndarray
    layers: np.ndarray
    datatypes: np.ndarray
    canonical_ids: np.ndarray


@dataclass(frozen=True)
class CudaByteCompare:
    old_size: int
    new_size: int
    same: bool
    method: str = "cuda-byte-compare"


@dataclass(frozen=True)
class CudaPolygonRecordDelta:
    layers: np.ndarray
    datatypes: np.ndarray
    directions: np.ndarray
    counts: np.ndarray
    bboxes: np.ndarray
    vertex_counts: np.ndarray
    flags: np.ndarray
    hashes: np.ndarray
    representative_indices: np.ndarray
    old_input_count: int
    new_input_count: int
    matched_record_count: int
    reduced_key_count: int
    cuda_kernel_count: int
    cuda_event_elapsed_s: float
    transferred_bytes: int

    @property
    def record_count(self) -> int:
        return int(len(self.directions))


@dataclass(frozen=True)
class CudaBoundaryLoopResult:
    fragments: tuple[DiffFragment, ...]
    dropped_zero_area_loop_count: int
    cuda_kernel_count: int
    cuda_event_elapsed_s: float
    transferred_bytes: int


@dataclass(frozen=True)
class CudaBoundaryEdgesResult:
    edges: np.ndarray
    cuda_kernel_count: int
    cuda_event_elapsed_s: float
    transferred_bytes: int


@dataclass(frozen=True)
class CudaContextOverlapPairs:
    component_indices: np.ndarray
    polygon_indices: np.ndarray
    cuda_kernel_count: int
    cuda_event_elapsed_s: float
    transferred_bytes: int

    @property
    def pair_count(self) -> int:
        return int(len(self.component_indices))


def run_cuda_ctypes_tile_prefilter(
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
    status = library.gds_cuda_prefilter(
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
        message = library.gds_cuda_last_error()
        raise CudaCtypesUnavailable(message.decode("utf-8") if message else "CUDA prefilter failed")
    try:
        candidate_tiles = _candidate_tiles_from_result(result)
        oversized = _oversized_from_result(result)
        return CpuTilePrefilterResult(
            candidate_tiles=candidate_tiles,
            oversized_candidates=oversized,
            assignment_count=int(result.assignment_count),
            collapsed_pair_count=int(result.collapsed_pair_count),
            oversized_matched_count=int(result.oversized_matched_count) + skipped_matched,
        )
    finally:
        library.gds_cuda_free_result(ctypes.byref(result))


def cuda_ctypes_available() -> bool:
    try:
        _load_library()
    except CudaCtypesUnavailable:
        return False
    return True


def release_cuda_device_memory() -> None:
    library = _load_library()
    status = library.gds_cuda_release_device_memory()
    if status != 0:
        message = library.gds_cuda_last_error()
        raise CudaCtypesUnavailable(message.decode("utf-8") if message else "CUDA device memory release failed")


def compare_files_cuda(old_path: str | Path, new_path: str | Path) -> CudaByteCompare:
    old_size = Path(old_path).stat().st_size
    new_size = Path(new_path).stat().st_size
    if old_size != new_size:
        return CudaByteCompare(old_size=old_size, new_size=new_size, same=False)

    library = _load_library()
    result = _CByteCompareResult()
    status = library.gds_cuda_compare_files(
        str(old_path).encode("utf-8"),
        str(new_path).encode("utf-8"),
        ctypes.byref(result),
    )
    if status != 0:
        message = library.gds_cuda_last_error()
        raise CudaCtypesUnavailable(message.decode("utf-8") if message else "CUDA byte compare failed")
    return CudaByteCompare(old_size=int(result.old_size), new_size=int(result.new_size), same=bool(result.same))


def run_cuda_polygon_record_delta(old_buffer: PolygonBuffer, new_buffer: PolygonBuffer) -> CudaPolygonRecordDelta:
    library = _load_library()
    old_packed = _pack_polygon_record_buffer(old_buffer)
    new_packed = _pack_polygon_record_buffer(new_buffer)
    result = _CPolygonRecordDeltaResult()
    status = library.gds_cuda_polygon_record_delta(
        _ptr_i64(old_packed["vertices"]),
        ctypes.c_uint64(old_buffer.vertex_count),
        _ptr_u64(old_packed["offsets"]),
        _ptr_i32(old_packed["layers"]),
        _ptr_i32(old_packed["datatypes"]),
        _ptr_i64(old_packed["bboxes"]),
        _ptr_u32(old_packed["flags"]),
        ctypes.c_uint64(old_buffer.polygon_count),
        _ptr_i64(new_packed["vertices"]),
        ctypes.c_uint64(new_buffer.vertex_count),
        _ptr_u64(new_packed["offsets"]),
        _ptr_i32(new_packed["layers"]),
        _ptr_i32(new_packed["datatypes"]),
        _ptr_i64(new_packed["bboxes"]),
        _ptr_u32(new_packed["flags"]),
        ctypes.c_uint64(new_buffer.polygon_count),
        ctypes.byref(result),
    )
    if status != 0:
        message = library.gds_cuda_last_error()
        raise CudaCtypesUnavailable(message.decode("utf-8") if message else "CUDA polygon record delta failed")
    try:
        count = int(result.record_count)
        return CudaPolygonRecordDelta(
            layers=_copy_result_array(result.layers, count, np.int32),
            datatypes=_copy_result_array(result.datatypes, count, np.int32),
            directions=_copy_result_array(result.directions, count, np.uint8),
            counts=_copy_result_array(result.counts, count, np.uint64),
            bboxes=_copy_result_array(result.bboxes, count * 4, np.int64).reshape((count, 4)),
            vertex_counts=_copy_result_array(result.vertex_counts, count, np.uint64),
            flags=_copy_result_array(result.flags, count, np.uint32),
            hashes=_copy_result_array(result.hashes, count * 2, np.uint64).reshape((count, 2)),
            representative_indices=_copy_result_array(result.representative_indices, count, np.uint64),
            old_input_count=int(result.old_input_count),
            new_input_count=int(result.new_input_count),
            matched_record_count=int(result.matched_record_count),
            reduced_key_count=int(result.reduced_key_count),
            cuda_kernel_count=int(result.cuda_kernel_count),
            cuda_event_elapsed_s=float(result.cuda_event_elapsed_ms) / 1000.0,
            transferred_bytes=_polygon_record_transfer_bytes(old_packed, new_packed),
        )
    finally:
        library.gds_cuda_free_polygon_record_delta_result(ctypes.byref(result))


def run_cuda_polygon_record_delta_from_hashes(old_buffer: PolygonBuffer, new_buffer: PolygonBuffer) -> CudaPolygonRecordDelta:
    if old_buffer.polygon_hashes is None or new_buffer.polygon_hashes is None:
        return run_cuda_polygon_record_delta(old_buffer, new_buffer)
    library = _load_library()
    old_packed = _pack_polygon_hash_record_buffer(old_buffer)
    new_packed = _pack_polygon_hash_record_buffer(new_buffer)
    result = _CPolygonRecordDeltaResult()
    status = library.gds_cuda_polygon_record_delta_from_hashes(
        _ptr_u64(old_packed["offsets"]),
        _ptr_i32(old_packed["layers"]),
        _ptr_i32(old_packed["datatypes"]),
        _ptr_i64(old_packed["bboxes"]),
        _ptr_u32(old_packed["flags"]),
        _ptr_u64(old_packed["hashes"]),
        ctypes.c_uint64(old_buffer.polygon_count),
        _ptr_u64(new_packed["offsets"]),
        _ptr_i32(new_packed["layers"]),
        _ptr_i32(new_packed["datatypes"]),
        _ptr_i64(new_packed["bboxes"]),
        _ptr_u32(new_packed["flags"]),
        _ptr_u64(new_packed["hashes"]),
        ctypes.c_uint64(new_buffer.polygon_count),
        ctypes.byref(result),
    )
    if status != 0:
        message = library.gds_cuda_last_error()
        raise CudaCtypesUnavailable(message.decode("utf-8") if message else "CUDA polygon hash record delta failed")
    try:
        count = int(result.record_count)
        return CudaPolygonRecordDelta(
            layers=_copy_result_array(result.layers, count, np.int32),
            datatypes=_copy_result_array(result.datatypes, count, np.int32),
            directions=_copy_result_array(result.directions, count, np.uint8),
            counts=_copy_result_array(result.counts, count, np.uint64),
            bboxes=_copy_result_array(result.bboxes, count * 4, np.int64).reshape((count, 4)),
            vertex_counts=_copy_result_array(result.vertex_counts, count, np.uint64),
            flags=_copy_result_array(result.flags, count, np.uint32),
            hashes=_copy_result_array(result.hashes, count * 2, np.uint64).reshape((count, 2)),
            representative_indices=_copy_result_array(result.representative_indices, count, np.uint64),
            old_input_count=int(result.old_input_count),
            new_input_count=int(result.new_input_count),
            matched_record_count=int(result.matched_record_count),
            reduced_key_count=int(result.reduced_key_count),
            cuda_kernel_count=int(result.cuda_kernel_count),
            cuda_event_elapsed_s=float(result.cuda_event_elapsed_ms) / 1000.0,
            transferred_bytes=_polygon_record_transfer_bytes(old_packed, new_packed),
        )
    finally:
        library.gds_cuda_free_polygon_record_delta_result(ctypes.byref(result))


def run_cuda_rect_exact_areas(
    old_rects: np.ndarray,
    new_rects: np.ndarray,
    old_present: np.ndarray,
    new_present: np.ndarray,
) -> np.ndarray:
    library = _load_library()
    old_rects = np.ascontiguousarray(old_rects, dtype=np.int64)
    new_rects = np.ascontiguousarray(new_rects, dtype=np.int64)
    old_present = np.ascontiguousarray(old_present, dtype=np.uint8)
    new_present = np.ascontiguousarray(new_present, dtype=np.uint8)
    if old_rects.ndim != 2 or old_rects.shape[1] != 4:
        raise ValueError("old_rects must have shape (N, 4)")
    if new_rects.shape != old_rects.shape:
        raise ValueError("new_rects must match old_rects shape")
    count = old_rects.shape[0]
    if old_present.shape != (count,) or new_present.shape != (count,):
        raise ValueError("presence arrays must have shape (N,)")
    out = np.empty((count, 3), dtype=np.int64)
    status = library.gds_cuda_rect_exact_areas(
        _ptr_i64(old_rects),
        _ptr_i64(new_rects),
        old_present.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        new_present.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_uint64(count),
        _ptr_i64(out),
    )
    if status != 0:
        message = library.gds_cuda_last_error()
        raise CudaCtypesUnavailable(message.decode("utf-8") if message else "CUDA rect exact failed")
    return out


def run_cuda_rect_exact_fragments(
    old_rects: np.ndarray,
    new_rects: np.ndarray,
    old_present: np.ndarray,
    new_present: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    library = _load_library()
    old_rects = np.ascontiguousarray(old_rects, dtype=np.int64)
    new_rects = np.ascontiguousarray(new_rects, dtype=np.int64)
    old_present = np.ascontiguousarray(old_present, dtype=np.uint8)
    new_present = np.ascontiguousarray(new_present, dtype=np.uint8)
    if old_rects.ndim != 2 or old_rects.shape[1] != 4:
        raise ValueError("old_rects must have shape (N, 4)")
    if new_rects.shape != old_rects.shape:
        raise ValueError("new_rects must match old_rects shape")
    count = old_rects.shape[0]
    if old_present.shape != (count,) or new_present.shape != (count,):
        raise ValueError("presence arrays must have shape (N,)")
    rects = np.empty((count, 8, 4), dtype=np.int64)
    counts = np.empty((count, 2), dtype=np.uint8)
    areas = np.empty((count, 3), dtype=np.int64)
    status = library.gds_cuda_rect_exact_fragments(
        _ptr_i64(old_rects),
        _ptr_i64(new_rects),
        old_present.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        new_present.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_uint64(count),
        _ptr_i64(rects),
        counts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        _ptr_i64(areas),
    )
    if status != 0:
        message = library.gds_cuda_last_error()
        raise CudaCtypesUnavailable(message.decode("utf-8") if message else "CUDA rect exact fragments failed")
    return rects, counts, areas


def run_cuda_rect_set_exact_fragments(
    old_rects: np.ndarray,
    new_rects: np.ndarray,
    old_counts: np.ndarray,
    new_counts: np.ndarray,
    *,
    max_fragments: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    library = _load_library()
    old_rects = np.ascontiguousarray(old_rects, dtype=np.int64)
    new_rects = np.ascontiguousarray(new_rects, dtype=np.int64)
    old_counts = np.ascontiguousarray(old_counts, dtype=np.uint8)
    new_counts = np.ascontiguousarray(new_counts, dtype=np.uint8)
    if old_rects.ndim != 3 or old_rects.shape[2] != 4:
        raise ValueError("old_rects must have shape (N, max_rects, 4)")
    if new_rects.shape != old_rects.shape:
        raise ValueError("new_rects must match old_rects shape")
    task_count, max_rects, _ = old_rects.shape
    if old_counts.shape != (task_count,) or new_counts.shape != (task_count,):
        raise ValueError("count arrays must have shape (N,)")
    if max_rects <= 0 or max_rects > 16:
        raise ValueError("max_rects must be between 1 and 16")
    if max_fragments <= 0 or max_fragments > 4096:
        raise ValueError("max_fragments must be between 1 and 4096")
    rects = np.empty((task_count, 2, max_fragments, 4), dtype=np.int64)
    counts = np.empty((task_count, 2), dtype=np.uint16)
    areas = np.empty((task_count, 3), dtype=np.int64)
    status = library.gds_cuda_rect_set_exact_fragments(
        _ptr_i64(old_rects),
        _ptr_i64(new_rects),
        old_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        new_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_uint64(task_count),
        ctypes.c_uint16(max_rects),
        ctypes.c_uint16(max_fragments),
        _ptr_i64(rects),
        counts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        _ptr_i64(areas),
    )
    if status != 0:
        message = library.gds_cuda_last_error()
        raise CudaCtypesUnavailable(message.decode("utf-8") if message else "CUDA rect set exact fragments failed")
    return rects, counts, areas


def run_cuda_polygon_component_areas(components: tuple[PolygonComponent, ...]) -> np.ndarray:
    library = _load_library()
    packed = _pack_polygon_components(components)
    component_count = len(components)
    areas = np.zeros((component_count, 3), dtype=np.int64)
    status = library.gds_cuda_polygon_component_areas(
        _ptr_i64(packed["vertices"]),
        ctypes.c_uint64(len(packed["vertices"]) // 2),
        _ptr_u64(packed["polygon_offsets"]),
        packed["polygon_sides"].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_uint64(len(packed["polygon_sides"])),
        _ptr_u64(packed["component_offsets"]),
        ctypes.c_uint64(component_count),
        _ptr_i64(areas),
    )
    if status != 0:
        message = library.gds_cuda_last_error()
        raise CudaCtypesUnavailable(message.decode("utf-8") if message else "CUDA polygon component areas failed")
    return areas


def run_cuda_polygon_component_fragments(
    components: tuple[PolygonComponent, ...],
    *,
    max_fragments: int = 4096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    library = _load_library()
    if max_fragments <= 0 or max_fragments > 65535:
        raise ValueError("max_fragments must be between 1 and 65535")
    packed = _pack_polygon_components(components)
    component_count = len(components)
    quads = np.zeros((component_count, 2, max_fragments, 4, 2), dtype=np.int64)
    fragment_areas = np.zeros((component_count, 2, max_fragments), dtype=np.int64)
    counts = np.zeros((component_count, 2), dtype=np.uint16)
    areas = np.zeros((component_count, 3), dtype=np.int64)
    status = library.gds_cuda_polygon_component_fragments(
        _ptr_i64(packed["vertices"]),
        ctypes.c_uint64(len(packed["vertices"]) // 2),
        _ptr_u64(packed["polygon_offsets"]),
        packed["polygon_sides"].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_uint64(len(packed["polygon_sides"])),
        _ptr_u64(packed["component_offsets"]),
        ctypes.c_uint64(component_count),
        ctypes.c_uint16(max_fragments),
        _ptr_i64(quads),
        _ptr_i64(fragment_areas),
        counts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        _ptr_i64(areas),
    )
    if status != 0:
        message = library.gds_cuda_last_error()
        raise CudaCtypesUnavailable(message.decode("utf-8") if message else "CUDA polygon component fragments failed")
    return quads, fragment_areas, counts, areas


def run_cuda_one_sided_boundary_edges(
    vertices_xy: np.ndarray,
    polygon_offsets: np.ndarray,
    bboxes: np.ndarray,
    signed_twice_areas: np.ndarray,
    segment_points: np.ndarray,
    segment_polygon_indices: np.ndarray,
) -> np.ndarray:
    library = _load_library()
    vertices_xy = np.ascontiguousarray(vertices_xy, dtype=np.int64)
    polygon_offsets = np.ascontiguousarray(polygon_offsets, dtype=np.uint64)
    bboxes = np.ascontiguousarray(bboxes, dtype=np.int64)
    signed_twice_areas = np.ascontiguousarray(signed_twice_areas, dtype=np.int64)
    segment_points = np.ascontiguousarray(segment_points, dtype=np.float64)
    segment_polygon_indices = np.ascontiguousarray(segment_polygon_indices, dtype=np.uint32)
    if vertices_xy.ndim != 2 or vertices_xy.shape[1] != 2:
        raise ValueError("vertices_xy must have shape (N, 2)")
    if polygon_offsets.ndim != 1 or len(polygon_offsets) < 1:
        raise ValueError("polygon_offsets must be a non-empty vector")
    polygon_count = len(polygon_offsets) - 1
    if bboxes.shape != (polygon_count, 4):
        raise ValueError("bboxes must have shape (polygon_count, 4)")
    if signed_twice_areas.shape != (polygon_count,):
        raise ValueError("signed_twice_areas must have shape (polygon_count,)")
    if segment_points.ndim != 2 or segment_points.shape[1] != 4:
        raise ValueError("segment_points must have shape (segment_count, 4)")
    segment_count = len(segment_points)
    if segment_polygon_indices.shape != (segment_count,):
        raise ValueError("segment_polygon_indices must have shape (segment_count,)")
    out_edges = np.zeros((segment_count, 4), dtype=np.float64)
    out_keep = np.zeros((segment_count,), dtype=np.uint8)
    status = library.gds_cuda_one_sided_boundary_edges(
        _ptr_i64(vertices_xy.reshape((-1,))),
        ctypes.c_uint64(len(vertices_xy)),
        _ptr_u64(polygon_offsets),
        _ptr_i64(bboxes.reshape((-1,))),
        _ptr_i64(signed_twice_areas),
        ctypes.c_uint64(polygon_count),
        _ptr_f64(segment_points.reshape((-1,))),
        segment_polygon_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_uint64(segment_count),
        _ptr_f64(out_edges.reshape((-1,))),
        out_keep.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
    )
    if status != 0:
        message = library.gds_cuda_last_error()
        raise CudaCtypesUnavailable(message.decode("utf-8") if message else "CUDA one-sided boundary edges failed")
    return out_edges[out_keep.astype(bool)].reshape((-1, 2, 2))


def run_cuda_mixed_boundary_edges(
    vertices_xy: np.ndarray,
    polygon_offsets: np.ndarray,
    bboxes: np.ndarray,
    polygon_sides: np.ndarray,
    segment_points: np.ndarray,
    *,
    direction: str,
) -> np.ndarray:
    library = _load_library()
    vertices_xy = np.ascontiguousarray(vertices_xy, dtype=np.int64)
    polygon_offsets = np.ascontiguousarray(polygon_offsets, dtype=np.uint64)
    bboxes = np.ascontiguousarray(bboxes, dtype=np.int64)
    polygon_sides = np.ascontiguousarray(polygon_sides, dtype=np.uint8)
    segment_points = np.ascontiguousarray(segment_points, dtype=np.float64)
    if direction not in {"old", "new"}:
        raise ValueError("direction must be 'old' or 'new'")
    if vertices_xy.ndim != 2 or vertices_xy.shape[1] != 2:
        raise ValueError("vertices_xy must have shape (N, 2)")
    if polygon_offsets.ndim != 1 or len(polygon_offsets) < 1:
        raise ValueError("polygon_offsets must be a non-empty vector")
    polygon_count = len(polygon_offsets) - 1
    if bboxes.shape != (polygon_count, 4):
        raise ValueError("bboxes must have shape (polygon_count, 4)")
    if polygon_sides.shape != (polygon_count,):
        raise ValueError("polygon_sides must have shape (polygon_count,)")
    if segment_points.ndim != 2 or segment_points.shape[1] != 4:
        raise ValueError("segment_points must have shape (segment_count, 4)")
    segment_count = len(segment_points)
    out_edges = np.zeros((segment_count, 4), dtype=np.float64)
    out_keep = np.zeros((segment_count,), dtype=np.uint8)
    status = library.gds_cuda_mixed_boundary_edges(
        _ptr_i64(vertices_xy.reshape((-1,))),
        ctypes.c_uint64(len(vertices_xy)),
        _ptr_u64(polygon_offsets),
        _ptr_i64(bboxes.reshape((-1,))),
        polygon_sides.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_uint64(polygon_count),
        _ptr_f64(segment_points.reshape((-1,))),
        ctypes.c_uint64(segment_count),
        ctypes.c_uint8(1 if direction == "old" else 2),
        _ptr_f64(out_edges.reshape((-1,))),
        out_keep.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
    )
    if status != 0:
        message = library.gds_cuda_last_error()
        raise CudaCtypesUnavailable(message.decode("utf-8") if message else "CUDA mixed boundary edges failed")
    return out_edges[out_keep.astype(bool)].reshape((-1, 2, 2))


def run_cuda_one_sided_boundary_edges_unsplit_if_safe(
    vertices_xy: np.ndarray,
    polygon_offsets: np.ndarray,
    bboxes: np.ndarray,
    signed_twice_areas: np.ndarray,
    segment_points: np.ndarray,
    segment_polygon_indices: np.ndarray,
) -> tuple[np.ndarray, bool]:
    library = _load_library()
    vertices_xy = np.ascontiguousarray(vertices_xy, dtype=np.int64)
    polygon_offsets = np.ascontiguousarray(polygon_offsets, dtype=np.uint64)
    bboxes = np.ascontiguousarray(bboxes, dtype=np.int64)
    signed_twice_areas = np.ascontiguousarray(signed_twice_areas, dtype=np.int64)
    segment_points = np.ascontiguousarray(segment_points, dtype=np.float64)
    segment_polygon_indices = np.ascontiguousarray(segment_polygon_indices, dtype=np.uint32)
    if vertices_xy.ndim != 2 or vertices_xy.shape[1] != 2:
        raise ValueError("vertices_xy must have shape (N, 2)")
    if polygon_offsets.ndim != 1 or len(polygon_offsets) < 1:
        raise ValueError("polygon_offsets must be a non-empty vector")
    polygon_count = len(polygon_offsets) - 1
    if bboxes.shape != (polygon_count, 4):
        raise ValueError("bboxes must have shape (polygon_count, 4)")
    if signed_twice_areas.shape != (polygon_count,):
        raise ValueError("signed_twice_areas must have shape (polygon_count,)")
    if segment_points.ndim != 2 or segment_points.shape[1] != 4:
        raise ValueError("segment_points must have shape (segment_count, 4)")
    segment_count = len(segment_points)
    if segment_polygon_indices.shape != (segment_count,):
        raise ValueError("segment_polygon_indices must have shape (segment_count,)")
    out_edges = np.zeros((segment_count, 4), dtype=np.float64)
    out_keep = np.zeros((segment_count,), dtype=np.uint8)
    out_safe = ctypes.c_uint8(0)
    status = library.gds_cuda_one_sided_boundary_edges_unsplit_if_safe(
        _ptr_i64(vertices_xy.reshape((-1,))),
        ctypes.c_uint64(len(vertices_xy)),
        _ptr_u64(polygon_offsets),
        _ptr_i64(bboxes.reshape((-1,))),
        _ptr_i64(signed_twice_areas),
        ctypes.c_uint64(polygon_count),
        _ptr_f64(segment_points.reshape((-1,))),
        segment_polygon_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_uint64(segment_count),
        _ptr_f64(out_edges.reshape((-1,))),
        out_keep.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.byref(out_safe),
    )
    if status != 0:
        message = library.gds_cuda_last_error()
        raise CudaCtypesUnavailable(message.decode("utf-8") if message else "CUDA one-sided unsplit boundary edges failed")
    return out_edges[out_keep.astype(bool)].reshape((-1, 2, 2)), bool(out_safe.value)


def run_cuda_boundary_loops_from_edges(edges: np.ndarray) -> CudaBoundaryLoopResult:
    library = _load_library()
    edges = np.ascontiguousarray(edges, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 4:
        raise ValueError("edges must have shape (N, 4)")
    result = _CBoundaryLoopResult()
    status = library.gds_cuda_boundary_loops_from_edges(
        _ptr_i64(edges.reshape((-1,))),
        ctypes.c_uint64(len(edges)),
        ctypes.byref(result),
    )
    if status != 0:
        message = library.gds_cuda_last_error()
        raise CudaCtypesUnavailable(message.decode("utf-8") if message else "CUDA boundary loop tracing failed")
    try:
        count = int(result.fragment_count)
        vertex_count = int(result.vertex_count)
        vertices = _copy_result_array(result.vertices_xy, vertex_count * 2, np.int64).reshape((vertex_count, 2))
        offsets = _copy_result_array(result.fragment_offsets, count + 1, np.uint64)
        bboxes = _copy_result_array(result.bboxes, count * 4, np.int64).reshape((count, 4))
        twice_areas = _copy_result_array(result.twice_areas, count, np.int64)
        fragments = []
        for index in range(count):
            start = int(offsets[index])
            stop = int(offsets[index + 1])
            points = tuple((int(x), int(y)) for x, y in vertices[start:stop])
            bbox = tuple(int(value) for value in bboxes[index])
            fragments.append(DiffFragment(points=points, bbox=bbox, twice_area=int(twice_areas[index])))
        return CudaBoundaryLoopResult(
            fragments=tuple(sorted(fragments, key=lambda fragment: (fragment.bbox, fragment.twice_area, fragment.points))),
            dropped_zero_area_loop_count=int(result.dropped_zero_area_loop_count),
            cuda_kernel_count=int(result.cuda_kernel_count),
            cuda_event_elapsed_s=float(result.cuda_event_elapsed_ms) / 1000.0,
            transferred_bytes=int(edges.nbytes),
        )
    finally:
        library.gds_cuda_free_boundary_loop_result(ctypes.byref(result))


def run_cuda_axis_aligned_boundary_edges_from_fragments(fragments: tuple[DiffFragment, ...]) -> CudaBoundaryEdgesResult:
    library = _load_library()
    vertices, offsets = _pack_diff_fragments(fragments)
    result = _CBoundaryEdgesResult()
    status = library.gds_cuda_axis_aligned_boundary_edges_from_polygons(
        _ptr_i64(vertices.reshape((-1,))),
        _ptr_u64(offsets),
        ctypes.c_uint64(len(fragments)),
        ctypes.byref(result),
    )
    if status != 0:
        message = library.gds_cuda_last_error()
        raise CudaCtypesUnavailable(message.decode("utf-8") if message else "CUDA axis boundary edge assembly failed")
    try:
        count = int(result.edge_count)
        return CudaBoundaryEdgesResult(
            edges=_copy_result_array(result.edges, count * 4, np.int64).reshape((count, 4)),
            cuda_kernel_count=int(result.cuda_kernel_count),
            cuda_event_elapsed_s=float(result.cuda_event_elapsed_ms) / 1000.0,
            transferred_bytes=int(vertices.nbytes + offsets.nbytes),
        )
    finally:
        library.gds_cuda_free_boundary_edges_result(ctypes.byref(result))


def run_cuda_split_free_boundary_edges_from_fragments(fragments: tuple[DiffFragment, ...]) -> CudaBoundaryEdgesResult:
    library = _load_library()
    vertices, offsets = _pack_diff_fragments(fragments)
    result = _CBoundaryEdgesResult()
    status = library.gds_cuda_split_free_boundary_edges_from_polygons(
        _ptr_i64(vertices.reshape((-1,))),
        _ptr_u64(offsets),
        ctypes.c_uint64(len(fragments)),
        ctypes.byref(result),
    )
    if status != 0:
        message = library.gds_cuda_last_error()
        raise CudaCtypesUnavailable(message.decode("utf-8") if message else "CUDA split-free boundary edge assembly failed")
    try:
        count = int(result.edge_count)
        return CudaBoundaryEdgesResult(
            edges=_copy_result_array(result.edges, count * 4, np.int64).reshape((count, 4)),
            cuda_kernel_count=int(result.cuda_kernel_count),
            cuda_event_elapsed_s=float(result.cuda_event_elapsed_ms) / 1000.0,
            transferred_bytes=int(vertices.nbytes + offsets.nbytes),
        )
    finally:
        library.gds_cuda_free_boundary_edges_result(ctypes.byref(result))


def run_cuda_context_overlap_pairs(
    component_bboxes: np.ndarray,
    component_layers: np.ndarray,
    component_datatypes: np.ndarray,
    polygon_bboxes: np.ndarray,
    polygon_layers: np.ndarray,
    polygon_datatypes: np.ndarray,
) -> CudaContextOverlapPairs:
    library = _load_library()
    component_bboxes = np.ascontiguousarray(component_bboxes, dtype=np.int64)
    component_layers = np.ascontiguousarray(component_layers, dtype=np.int32)
    component_datatypes = np.ascontiguousarray(component_datatypes, dtype=np.int32)
    polygon_bboxes = np.ascontiguousarray(polygon_bboxes, dtype=np.int64)
    polygon_layers = np.ascontiguousarray(polygon_layers, dtype=np.int32)
    polygon_datatypes = np.ascontiguousarray(polygon_datatypes, dtype=np.int32)
    if component_bboxes.ndim != 2 or component_bboxes.shape[1] != 4:
        raise ValueError("component_bboxes must have shape (N, 4)")
    if polygon_bboxes.ndim != 2 or polygon_bboxes.shape[1] != 4:
        raise ValueError("polygon_bboxes must have shape (M, 4)")
    component_count = component_bboxes.shape[0]
    polygon_count = polygon_bboxes.shape[0]
    if component_layers.shape != (component_count,) or component_datatypes.shape != (component_count,):
        raise ValueError("component layer/datatype arrays must have shape (N,)")
    if polygon_layers.shape != (polygon_count,) or polygon_datatypes.shape != (polygon_count,):
        raise ValueError("polygon layer/datatype arrays must have shape (M,)")
    result = _CContextOverlapPairsResult()
    status = library.gds_cuda_context_overlap_pairs(
        _ptr_i64(component_bboxes.reshape((-1,))),
        _ptr_i32(component_layers),
        _ptr_i32(component_datatypes),
        ctypes.c_uint64(component_count),
        _ptr_i64(polygon_bboxes.reshape((-1,))),
        _ptr_i32(polygon_layers),
        _ptr_i32(polygon_datatypes),
        ctypes.c_uint64(polygon_count),
        ctypes.byref(result),
    )
    if status != 0:
        message = library.gds_cuda_last_error()
        raise CudaCtypesUnavailable(message.decode("utf-8") if message else "CUDA context overlap pairs failed")
    try:
        count = int(result.pair_count)
        return CudaContextOverlapPairs(
            component_indices=_copy_result_array(result.component_indices, count, np.uint64),
            polygon_indices=_copy_result_array(result.polygon_indices, count, np.uint64),
            cuda_kernel_count=int(result.cuda_kernel_count),
            cuda_event_elapsed_s=float(result.cuda_event_elapsed_ms) / 1000.0,
            transferred_bytes=int(
                component_bboxes.nbytes
                + component_layers.nbytes
                + component_datatypes.nbytes
                + polygon_bboxes.nbytes
                + polygon_layers.nbytes
                + polygon_datatypes.nbytes
            ),
        )
    finally:
        library.gds_cuda_free_context_overlap_pairs_result(ctypes.byref(result))


def _candidate_tiles_from_result(result: _CResult) -> tuple[TileKey, ...]:
    if result.candidate_count == 0:
        return ()
    array = np.ctypeslib.as_array(result.candidate_tiles, shape=(int(result.candidate_count) * 4,))
    tiles = [
        TileKey(LayerSpec(int(array[index]), int(array[index + 1])), int(array[index + 2]), int(array[index + 3]))
        for index in range(0, len(array), 4)
    ]
    return tuple(tiles)


def _oversized_from_result(result: _CResult) -> tuple[OversizedPolygon, ...]:
    if result.oversized_count == 0:
        return ()
    array = np.ctypeslib.as_array(result.oversized, shape=(int(result.oversized_count) * 8,))
    items = [
        OversizedPolygon(
            layer=LayerSpec(int(array[index]), int(array[index + 1])),
            canonical_id=int(array[index + 2]),
            side_mask=int(array[index + 3]),
            tile_bbox=(int(array[index + 4]), int(array[index + 5]), int(array[index + 6]), int(array[index + 7])),
        )
        for index in range(0, len(array), 8)
    ]
    return tuple(items)


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


def _ptr_u32(array: np.ndarray):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))


def _ptr_f64(array: np.ndarray):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _copy_result_array(pointer, count: int, dtype) -> np.ndarray:
    if count == 0:
        return np.empty((0,), dtype=dtype)
    if not pointer:
        raise CudaCtypesUnavailable("CUDA result returned null output array")
    ctype = np.ctypeslib.as_ctypes_type(dtype)
    array = ctypes.cast(pointer, ctypes.POINTER(ctype * count)).contents
    return np.frombuffer(array, dtype=dtype).copy()


def _pack_polygon_record_buffer(buffer: PolygonBuffer) -> dict[str, np.ndarray]:
    layer_values = np.asarray([layer.layer for layer in buffer.layer_table.layers], dtype=np.int32)
    datatype_values = np.asarray([layer.datatype for layer in buffer.layer_table.layers], dtype=np.int32)
    layers = np.ascontiguousarray(layer_values[buffer.layer_ids], dtype=np.int32)
    datatypes = np.ascontiguousarray(datatype_values[buffer.layer_ids], dtype=np.int32)
    return {
        "vertices": np.ascontiguousarray(buffer.vertices_xy.reshape((-1,)), dtype=np.int64),
        "offsets": np.ascontiguousarray(buffer.polygon_offsets, dtype=np.uint64),
        "layers": layers,
        "datatypes": datatypes,
        "bboxes": np.ascontiguousarray(buffer.bboxes.reshape((-1,)), dtype=np.int64),
        "flags": np.ascontiguousarray(buffer.flags, dtype=np.uint32),
    }


def _pack_polygon_hash_record_buffer(buffer: PolygonBuffer) -> dict[str, np.ndarray]:
    if buffer.polygon_hashes is None:
        raise CudaCtypesUnavailable("native polygon hashes are unavailable")
    layer_values = np.asarray([layer.layer for layer in buffer.layer_table.layers], dtype=np.int32)
    datatype_values = np.asarray([layer.datatype for layer in buffer.layer_table.layers], dtype=np.int32)
    layers = np.ascontiguousarray(layer_values[buffer.layer_ids], dtype=np.int32)
    datatypes = np.ascontiguousarray(datatype_values[buffer.layer_ids], dtype=np.int32)
    return {
        "offsets": np.ascontiguousarray(buffer.polygon_offsets, dtype=np.uint64),
        "layers": layers,
        "datatypes": datatypes,
        "bboxes": np.ascontiguousarray(buffer.bboxes.reshape((-1,)), dtype=np.int64),
        "flags": np.ascontiguousarray(buffer.flags, dtype=np.uint32),
        "hashes": np.ascontiguousarray(buffer.polygon_hashes.reshape((-1,)), dtype=np.uint64),
    }


def _pack_polygon_components(components: tuple[PolygonComponent, ...]) -> dict[str, np.ndarray]:
    vertices: list[tuple[int, int]] = []
    polygon_offsets = [0]
    polygon_sides: list[int] = []
    component_offsets = [0]
    polygon_count = 0
    for component in components:
        for polygon in component.polygons:
            vertices.extend(polygon.fragment.points)
            polygon_offsets.append(len(vertices))
            polygon_sides.append(1 if polygon.side == "old" else 2)
            polygon_count += 1
        component_offsets.append(polygon_count)
    return {
        "vertices": np.ascontiguousarray(np.asarray(vertices, dtype=np.int64).reshape((-1,)), dtype=np.int64),
        "polygon_offsets": np.ascontiguousarray(np.asarray(polygon_offsets, dtype=np.uint64), dtype=np.uint64),
        "polygon_sides": np.ascontiguousarray(np.asarray(polygon_sides, dtype=np.uint8), dtype=np.uint8),
        "component_offsets": np.ascontiguousarray(np.asarray(component_offsets, dtype=np.uint64), dtype=np.uint64),
    }


def _pack_diff_fragments(fragments: tuple[DiffFragment, ...]) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[tuple[int, int]] = []
    offsets = [0]
    for fragment in fragments:
        vertices.extend(fragment.points)
        offsets.append(len(vertices))
    return (
        np.ascontiguousarray(np.asarray(vertices, dtype=np.int64).reshape((-1, 2)), dtype=np.int64),
        np.ascontiguousarray(np.asarray(offsets, dtype=np.uint64), dtype=np.uint64),
    )


def _polygon_record_transfer_bytes(*packed_buffers: dict[str, np.ndarray]) -> int:
    return sum(int(array.nbytes) for packed in packed_buffers for array in packed.values())


def _load_library() -> ctypes.CDLL:
    path = _library_path()
    if not path.exists() or _source_path().stat().st_mtime > path.stat().st_mtime:
        _build_library(path)
    library = ctypes.CDLL(str(path))
    library.gds_cuda_prefilter.argtypes = [
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
    library.gds_cuda_prefilter.restype = ctypes.c_int
    library.gds_cuda_last_error.argtypes = []
    library.gds_cuda_last_error.restype = ctypes.c_char_p
    library.gds_cuda_release_device_memory.argtypes = []
    library.gds_cuda_release_device_memory.restype = ctypes.c_int
    library.gds_cuda_free_result.argtypes = [ctypes.POINTER(_CResult)]
    library.gds_cuda_free_result.restype = None
    library.gds_cuda_compare_files.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(_CByteCompareResult),
    ]
    library.gds_cuda_compare_files.restype = ctypes.c_int
    library.gds_cuda_polygon_record_delta.argtypes = [
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint64,
        ctypes.POINTER(_CPolygonRecordDeltaResult),
    ]
    library.gds_cuda_polygon_record_delta.restype = ctypes.c_int
    library.gds_cuda_polygon_record_delta_from_hashes.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_uint64,
        ctypes.POINTER(_CPolygonRecordDeltaResult),
    ]
    library.gds_cuda_polygon_record_delta_from_hashes.restype = ctypes.c_int
    library.gds_cuda_free_polygon_record_delta_result.argtypes = [ctypes.POINTER(_CPolygonRecordDeltaResult)]
    library.gds_cuda_free_polygon_record_delta_result.restype = None
    library.gds_cuda_boundary_loops_from_edges.argtypes = [
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_uint64,
        ctypes.POINTER(_CBoundaryLoopResult),
    ]
    library.gds_cuda_boundary_loops_from_edges.restype = ctypes.c_int
    library.gds_cuda_free_boundary_loop_result.argtypes = [ctypes.POINTER(_CBoundaryLoopResult)]
    library.gds_cuda_free_boundary_loop_result.restype = None
    library.gds_cuda_axis_aligned_boundary_edges_from_polygons.argtypes = [
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_uint64,
        ctypes.POINTER(_CBoundaryEdgesResult),
    ]
    library.gds_cuda_axis_aligned_boundary_edges_from_polygons.restype = ctypes.c_int
    library.gds_cuda_split_free_boundary_edges_from_polygons.argtypes = [
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_uint64,
        ctypes.POINTER(_CBoundaryEdgesResult),
    ]
    library.gds_cuda_split_free_boundary_edges_from_polygons.restype = ctypes.c_int
    library.gds_cuda_free_boundary_edges_result.argtypes = [ctypes.POINTER(_CBoundaryEdgesResult)]
    library.gds_cuda_free_boundary_edges_result.restype = None
    library.gds_cuda_context_overlap_pairs.argtypes = [
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_uint64,
        ctypes.POINTER(_CContextOverlapPairsResult),
    ]
    library.gds_cuda_context_overlap_pairs.restype = ctypes.c_int
    library.gds_cuda_free_context_overlap_pairs_result.argtypes = [ctypes.POINTER(_CContextOverlapPairsResult)]
    library.gds_cuda_free_context_overlap_pairs_result.restype = None
    library.gds_cuda_rect_exact_areas.argtypes = [
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_int64),
    ]
    library.gds_cuda_rect_exact_areas.restype = ctypes.c_int
    library.gds_cuda_rect_exact_fragments.argtypes = [
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_int64),
    ]
    library.gds_cuda_rect_exact_fragments.restype = ctypes.c_int
    library.gds_cuda_polygon_component_areas.argtypes = [
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_int64),
    ]
    library.gds_cuda_polygon_component_areas.restype = ctypes.c_int
    library.gds_cuda_polygon_component_fragments.argtypes = [
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_uint64,
        ctypes.c_uint16,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_int64),
    ]
    library.gds_cuda_polygon_component_fragments.restype = ctypes.c_int
    library.gds_cuda_one_sided_boundary_edges.argtypes = [
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_uint8),
    ]
    library.gds_cuda_one_sided_boundary_edges.restype = ctypes.c_int
    library.gds_cuda_mixed_boundary_edges.argtypes = [
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_uint64,
        ctypes.c_uint8,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_uint8),
    ]
    library.gds_cuda_mixed_boundary_edges.restype = ctypes.c_int
    library.gds_cuda_one_sided_boundary_edges_unsplit_if_safe.argtypes = [
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
    ]
    library.gds_cuda_one_sided_boundary_edges_unsplit_if_safe.restype = ctypes.c_int
    library.gds_cuda_rect_set_exact_fragments.argtypes = [
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint64,
        ctypes.c_uint16,
        ctypes.c_uint16,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_int64),
    ]
    library.gds_cuda_rect_set_exact_fragments.restype = ctypes.c_int
    return library


def _build_library(path: Path) -> None:
    source = _source_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "nvcc",
        "-std=c++17",
        "-O3",
        "--shared",
        "-Xcompiler",
        "-fPIC",
        str(source),
        "-o",
        str(path),
    ]
    try:
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
    except FileNotFoundError as exc:
        raise CudaCtypesUnavailable("nvcc is not available") from exc
    if completed.returncode != 0:
        raise CudaCtypesUnavailable((completed.stderr or completed.stdout).strip() or "nvcc failed")


def _library_path() -> Path:
    return cached_native_library_path("_gds_compare_cuda_ctypes.so")


def _source_path() -> Path:
    return Path(__file__).resolve().parents[1] / "native_src" / "gdsdiff" / "cuda_ctypes.cu"
