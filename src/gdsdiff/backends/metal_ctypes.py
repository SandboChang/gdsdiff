from __future__ import annotations

import ctypes
import platform
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..config import LayerSpec
from ..exact_oracle import DiffFragment
from ..geometry import PolygonBuffer
from ..native_cache import ensure_cached_native_library
from ..tiling import CpuTilePrefilterResult, OversizedPolygon, TileConfig, TileKey


class MetalCtypesUnavailable(RuntimeError):
    pass


class _CPrefilterResult(ctypes.Structure):
    _fields_ = [
        ("candidate_tiles", ctypes.POINTER(ctypes.c_int64)),
        ("candidate_count", ctypes.c_uint64),
        ("oversized", ctypes.POINTER(ctypes.c_int64)),
        ("oversized_count", ctypes.c_uint64),
        ("assignment_count", ctypes.c_uint64),
        ("collapsed_pair_count", ctypes.c_uint64),
        ("oversized_matched_count", ctypes.c_uint64),
        ("metal_kernel_count", ctypes.c_uint64),
        ("metal_event_elapsed_ms", ctypes.c_float),
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
        ("metal_kernel_count", ctypes.c_uint64),
        ("metal_event_elapsed_ms", ctypes.c_float),
    ]


class _CByteCompareResult(ctypes.Structure):
    _fields_ = [
        ("old_size", ctypes.c_uint64),
        ("new_size", ctypes.c_uint64),
        ("same", ctypes.c_uint8),
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
        ("metal_kernel_count", ctypes.c_uint64),
        ("metal_event_elapsed_ms", ctypes.c_float),
    ]


class _CContextOverlapPairsResult(ctypes.Structure):
    _fields_ = [
        ("pair_count", ctypes.c_uint64),
        ("component_indices", ctypes.POINTER(ctypes.c_uint64)),
        ("polygon_indices", ctypes.POINTER(ctypes.c_uint64)),
        ("metal_kernel_count", ctypes.c_uint64),
        ("metal_event_elapsed_ms", ctypes.c_float),
    ]


@dataclass(frozen=True)
class _PackedBuffer:
    bboxes: np.ndarray
    layers: np.ndarray
    datatypes: np.ndarray
    canonical_ids: np.ndarray


@dataclass(frozen=True)
class MetalByteCompare:
    old_size: int
    new_size: int
    same: bool
    method: str = "metal-byte-compare"


@dataclass(frozen=True)
class MetalPolygonRecordDelta:
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
    metal_kernel_count: int
    metal_event_elapsed_s: float
    transferred_bytes: int
    accelerator: str = "metal"

    @property
    def record_count(self) -> int:
        return int(len(self.directions))

    @property
    def cuda_kernel_count(self) -> int:
        return 0

    @property
    def cuda_event_elapsed_s(self) -> float:
        return 0.0


@dataclass(frozen=True)
class MetalBoundaryLoopResult:
    fragments: tuple[DiffFragment, ...]
    dropped_zero_area_loop_count: int
    metal_kernel_count: int
    metal_event_elapsed_s: float
    transferred_bytes: int
    accelerator: str = "metal"

    @property
    def cuda_kernel_count(self) -> int:
        return 0

    @property
    def cuda_event_elapsed_s(self) -> float:
        return 0.0


@dataclass(frozen=True)
class MetalContextOverlapPairs:
    component_indices: np.ndarray
    polygon_indices: np.ndarray
    metal_kernel_count: int
    metal_event_elapsed_s: float
    transferred_bytes: int
    accelerator: str = "metal"

    @property
    def pair_count(self) -> int:
        return int(len(self.component_indices))

    @property
    def cuda_kernel_count(self) -> int:
        return 0

    @property
    def cuda_event_elapsed_s(self) -> float:
        return 0.0


def metal_ctypes_available() -> bool:
    try:
        library = _load_library()
        return bool(library.gds_metal_available())
    except MetalCtypesUnavailable:
        return False


def release_metal_device_memory() -> None:
    library = _load_library()
    status = library.gds_metal_release_device_memory()
    if status != 0:
        message = library.gds_metal_last_error()
        raise MetalCtypesUnavailable(message.decode("utf-8") if message else "Metal device memory release failed")


def compare_files_metal(old_path: str | Path, new_path: str | Path) -> MetalByteCompare:
    old_size = Path(old_path).stat().st_size
    new_size = Path(new_path).stat().st_size
    if old_size != new_size:
        return MetalByteCompare(old_size=old_size, new_size=new_size, same=False)

    library = _load_library()
    result = _CByteCompareResult()
    status = library.gds_metal_compare_files(
        str(old_path).encode("utf-8"),
        str(new_path).encode("utf-8"),
        ctypes.byref(result),
    )
    if status != 0:
        message = library.gds_metal_last_error()
        raise MetalCtypesUnavailable(message.decode("utf-8") if message else "Metal byte compare failed")
    return MetalByteCompare(old_size=int(result.old_size), new_size=int(result.new_size), same=bool(result.same))


def run_metal_ctypes_tile_prefilter(
    old_buffer: PolygonBuffer,
    old_canonical_ids,
    new_buffer: PolygonBuffer,
    new_canonical_ids,
    config: TileConfig,
) -> CpuTilePrefilterResult:
    old_packed = _pack_buffer(old_buffer, old_canonical_ids)
    new_packed = _pack_buffer(new_buffer, new_canonical_ids)
    library = _load_library()
    result = _CPrefilterResult()
    status = library.gds_metal_prefilter(
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
        message = library.gds_metal_last_error()
        raise MetalCtypesUnavailable(message.decode("utf-8") if message else "Metal tile prefilter failed")
    try:
        return CpuTilePrefilterResult(
            candidate_tiles=_candidate_tiles_from_result(result),
            oversized_candidates=_oversized_from_result(result),
            assignment_count=int(result.assignment_count),
            collapsed_pair_count=int(result.collapsed_pair_count),
            oversized_matched_count=int(result.oversized_matched_count),
        )
    finally:
        library.gds_metal_free_prefilter_result(ctypes.byref(result))


def run_metal_rect_set_exact_fragments(
    old_rects: np.ndarray,
    new_rects: np.ndarray,
    old_counts: np.ndarray,
    new_counts: np.ndarray,
    *,
    max_fragments: int = 4096,
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
    rects = np.zeros((task_count, 2, max_fragments, 4), dtype=np.int64)
    counts = np.zeros((task_count, 2), dtype=np.uint16)
    areas = np.zeros((task_count, 3), dtype=np.int64)
    status = library.gds_metal_rect_set_exact_fragments(
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
        message = library.gds_metal_last_error()
        raise MetalCtypesUnavailable(message.decode("utf-8") if message else "Metal rectangle set exact failed")
    return rects, counts, areas


def run_metal_polygon_record_delta_from_hashes(old_buffer: PolygonBuffer, new_buffer: PolygonBuffer) -> MetalPolygonRecordDelta:
    if old_buffer.polygon_hashes is None or new_buffer.polygon_hashes is None:
        raise MetalCtypesUnavailable("native polygon hashes are unavailable")
    total_count = old_buffer.polygon_count + new_buffer.polygon_count
    max_records = _record_delta_max_polygons()
    if total_count > max_records:
        raise MetalCtypesUnavailable(
            f"Metal polygon record delta is limited to {max_records} total polygons; got {total_count}"
        )
    library = _load_library()
    old_packed = _pack_polygon_hash_record_buffer(old_buffer)
    new_packed = _pack_polygon_hash_record_buffer(new_buffer)
    result = _CPolygonRecordDeltaResult()
    status = library.gds_metal_polygon_record_delta_from_hashes(
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
        message = library.gds_metal_last_error()
        raise MetalCtypesUnavailable(message.decode("utf-8") if message else "Metal polygon record delta failed")
    try:
        count = int(result.record_count)
        return MetalPolygonRecordDelta(
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
            metal_kernel_count=int(result.metal_kernel_count),
            metal_event_elapsed_s=float(result.metal_event_elapsed_ms) / 1000.0,
            transferred_bytes=_polygon_record_transfer_bytes(old_packed, new_packed),
        )
    finally:
        library.gds_metal_free_polygon_record_delta_result(ctypes.byref(result))


def run_metal_context_overlap_pairs(
    component_bboxes: np.ndarray,
    component_layers: np.ndarray,
    component_datatypes: np.ndarray,
    polygon_bboxes: np.ndarray,
    polygon_layers: np.ndarray,
    polygon_datatypes: np.ndarray,
) -> MetalContextOverlapPairs:
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
    total_pairs = component_count * polygon_count
    max_pairs = _context_overlap_max_pairs()
    if total_pairs > max_pairs:
        raise MetalCtypesUnavailable(
            f"Metal context overlap prototype is limited to {max_pairs} component/polygon pairs; got {total_pairs}"
        )
    library = _load_library()
    result = _CContextOverlapPairsResult()
    status = library.gds_metal_context_overlap_pairs(
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
        message = library.gds_metal_last_error()
        raise MetalCtypesUnavailable(message.decode("utf-8") if message else "Metal context overlap pairs failed")
    try:
        count = int(result.pair_count)
        return MetalContextOverlapPairs(
            component_indices=_copy_result_array(result.component_indices, count, np.uint64),
            polygon_indices=_copy_result_array(result.polygon_indices, count, np.uint64),
            metal_kernel_count=int(result.metal_kernel_count),
            metal_event_elapsed_s=float(result.metal_event_elapsed_ms) / 1000.0,
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
        library.gds_metal_free_context_overlap_pairs_result(ctypes.byref(result))


def run_metal_boundary_loops_from_edges(edges: np.ndarray) -> MetalBoundaryLoopResult:
    edges = np.ascontiguousarray(edges, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 4:
        raise ValueError("edges must have shape (N, 4)")
    library = _load_library()
    result = _CBoundaryLoopResult()
    status = library.gds_metal_boundary_loops_from_edges(
        _ptr_i64(edges.reshape((-1,))),
        ctypes.c_uint64(len(edges)),
        ctypes.byref(result),
    )
    if status != 0:
        message = library.gds_metal_last_error()
        raise MetalCtypesUnavailable(message.decode("utf-8") if message else "Metal boundary loop tracing failed")
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
        return MetalBoundaryLoopResult(
            fragments=tuple(sorted(fragments, key=lambda fragment: (fragment.bbox, fragment.twice_area, fragment.points))),
            dropped_zero_area_loop_count=int(result.dropped_zero_area_loop_count),
            metal_kernel_count=int(result.metal_kernel_count),
            metal_event_elapsed_s=float(result.metal_event_elapsed_ms) / 1000.0,
            transferred_bytes=int(edges.nbytes),
        )
    finally:
        library.gds_metal_free_boundary_loop_result(ctypes.byref(result))


def run_metal_one_sided_boundary_edges(
    vertices_xy: np.ndarray,
    polygon_offsets: np.ndarray,
    bboxes: np.ndarray,
    signed_twice_areas: np.ndarray,
    segment_points: np.ndarray,
    segment_polygon_indices: np.ndarray,
) -> np.ndarray:
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
    library = _load_library()
    status = library.gds_metal_one_sided_boundary_edges(
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
        message = library.gds_metal_last_error()
        raise MetalCtypesUnavailable(message.decode("utf-8") if message else "Metal one-sided boundary edges failed")
    return out_edges[out_keep.astype(bool)].reshape((-1, 2, 2))


def run_metal_mixed_boundary_edges(
    vertices_xy: np.ndarray,
    polygon_offsets: np.ndarray,
    bboxes: np.ndarray,
    polygon_sides: np.ndarray,
    segment_points: np.ndarray,
    *,
    direction: str,
) -> np.ndarray:
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
    library = _load_library()
    status = library.gds_metal_mixed_boundary_edges(
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
        message = library.gds_metal_last_error()
        raise MetalCtypesUnavailable(message.decode("utf-8") if message else "Metal mixed boundary edges failed")
    return out_edges[out_keep.astype(bool)].reshape((-1, 2, 2))


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
        raise MetalCtypesUnavailable("Metal result returned null output array")
    ctype = np.ctypeslib.as_ctypes_type(dtype)
    array = ctypes.cast(pointer, ctypes.POINTER(ctype * count)).contents
    return np.frombuffer(array, dtype=dtype).copy()


def _candidate_tiles_from_result(result: _CPrefilterResult) -> tuple[TileKey, ...]:
    if result.candidate_count == 0:
        return ()
    array = np.ctypeslib.as_array(result.candidate_tiles, shape=(int(result.candidate_count) * 4,))
    return tuple(
        TileKey(LayerSpec(int(array[index]), int(array[index + 1])), int(array[index + 2]), int(array[index + 3]))
        for index in range(0, len(array), 4)
    )


def _oversized_from_result(result: _CPrefilterResult) -> tuple[OversizedPolygon, ...]:
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
    layer_values = np.asarray([layer.layer for layer in buffer.layer_table.layers], dtype=np.int32)
    datatype_values = np.asarray([layer.datatype for layer in buffer.layer_table.layers], dtype=np.int32)
    return _PackedBuffer(
        bboxes=np.ascontiguousarray(buffer.bboxes, dtype=np.int64),
        layers=np.ascontiguousarray(layer_values[buffer.layer_ids], dtype=np.int32),
        datatypes=np.ascontiguousarray(datatype_values[buffer.layer_ids], dtype=np.int32),
        canonical_ids=ids,
    )


def _pack_polygon_hash_record_buffer(buffer: PolygonBuffer) -> dict[str, np.ndarray]:
    if buffer.polygon_hashes is None:
        raise MetalCtypesUnavailable("native polygon hashes are unavailable")
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


def _polygon_record_transfer_bytes(*packed_buffers: dict[str, np.ndarray]) -> int:
    return sum(int(array.nbytes) for packed in packed_buffers for array in packed.values())


def _record_delta_max_polygons() -> int:
    import os

    try:
        value = int(os.environ.get("GDS_METAL_RECORD_DELTA_MAX_POLYGONS", ""))
    except ValueError:
        return 10_000_000
    return value if value > 0 else 10_000_000


def _context_overlap_max_pairs() -> int:
    import os

    try:
        value = int(os.environ.get("GDS_METAL_CONTEXT_OVERLAP_MAX_PAIRS", ""))
    except ValueError:
        return 64 * 1024 * 1024
    return value if value > 0 else 64 * 1024 * 1024


@lru_cache(maxsize=1)
def _load_library() -> ctypes.CDLL:
    if platform.system() != "Darwin" or platform.machine() not in {"arm64", "aarch64"}:
        raise MetalCtypesUnavailable("Metal ctypes backend requires macOS on Apple Silicon")
    path = _library_path()
    library = ctypes.CDLL(str(path))
    library.gds_metal_last_error.argtypes = []
    library.gds_metal_last_error.restype = ctypes.c_char_p
    library.gds_metal_available.argtypes = []
    library.gds_metal_available.restype = ctypes.c_int
    library.gds_metal_release_device_memory.argtypes = []
    library.gds_metal_release_device_memory.restype = ctypes.c_int
    library.gds_metal_compare_files.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(_CByteCompareResult),
    ]
    library.gds_metal_compare_files.restype = ctypes.c_int
    library.gds_metal_prefilter.argtypes = [
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
        ctypes.POINTER(_CPrefilterResult),
    ]
    library.gds_metal_prefilter.restype = ctypes.c_int
    library.gds_metal_free_prefilter_result.argtypes = [ctypes.POINTER(_CPrefilterResult)]
    library.gds_metal_free_prefilter_result.restype = None
    library.gds_metal_rect_set_exact_fragments.argtypes = [
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
    library.gds_metal_rect_set_exact_fragments.restype = ctypes.c_int
    library.gds_metal_polygon_record_delta_from_hashes.argtypes = [
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
    library.gds_metal_polygon_record_delta_from_hashes.restype = ctypes.c_int
    library.gds_metal_free_polygon_record_delta_result.argtypes = [ctypes.POINTER(_CPolygonRecordDeltaResult)]
    library.gds_metal_free_polygon_record_delta_result.restype = None
    library.gds_metal_context_overlap_pairs.argtypes = [
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
    library.gds_metal_context_overlap_pairs.restype = ctypes.c_int
    library.gds_metal_free_context_overlap_pairs_result.argtypes = [ctypes.POINTER(_CContextOverlapPairsResult)]
    library.gds_metal_free_context_overlap_pairs_result.restype = None
    library.gds_metal_boundary_loops_from_edges.argtypes = [
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_uint64,
        ctypes.POINTER(_CBoundaryLoopResult),
    ]
    library.gds_metal_boundary_loops_from_edges.restype = ctypes.c_int
    library.gds_metal_free_boundary_loop_result.argtypes = [ctypes.POINTER(_CBoundaryLoopResult)]
    library.gds_metal_free_boundary_loop_result.restype = None
    library.gds_metal_one_sided_boundary_edges.argtypes = [
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
    library.gds_metal_one_sided_boundary_edges.restype = ctypes.c_int
    library.gds_metal_mixed_boundary_edges.argtypes = [
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
    library.gds_metal_mixed_boundary_edges.restype = ctypes.c_int
    return library


def _build_library(path: Path) -> None:
    source = _source_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    compiler = _clangxx_path()
    command = [
        compiler,
        "-std=c++17",
        "-O3",
        "-dynamiclib",
        "-fPIC",
        "-fobjc-arc",
        "-isysroot",
        _sdk_path(),
        str(source),
        "-framework",
        "Foundation",
        "-framework",
        "Metal",
        "-o",
        str(path),
    ]
    try:
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
    except FileNotFoundError as exc:
        raise MetalCtypesUnavailable("clang++ is not available") from exc
    if completed.returncode != 0:
        raise MetalCtypesUnavailable((completed.stderr or completed.stdout).strip() or "clang++ failed")


def _clangxx_path() -> str:
    try:
        completed = subprocess.run(["xcrun", "--find", "clang++"], check=False, text=True, capture_output=True)
    except FileNotFoundError:
        return "clang++"
    candidate = completed.stdout.strip()
    return candidate if completed.returncode == 0 and candidate else "clang++"


def _sdk_path() -> str:
    try:
        completed = subprocess.run(["xcrun", "--show-sdk-path"], check=False, text=True, capture_output=True)
    except FileNotFoundError:
        return "/"
    candidate = completed.stdout.strip()
    return candidate if completed.returncode == 0 and candidate else "/"


def _library_path() -> Path:
    compiler = _clangxx_path()
    sdk = _sdk_path()
    return ensure_cached_native_library(
        "_gds_compare_metal_ctypes.so",
        sources=(_source_path(),),
        build_key=(
            compiler,
            "-std=c++17",
            "-O3",
            "-dynamiclib",
            "-fPIC",
            "-fobjc-arc",
            "-isysroot",
            sdk,
            "-framework",
            "Foundation",
            "-framework",
            "Metal",
        ),
        builder=_build_library,
    )


def _source_path() -> Path:
    return Path(__file__).resolve().parents[1] / "native_src" / "gdsdiff" / "metal_ctypes.mm"
