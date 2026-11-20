from __future__ import annotations

import hashlib
import os
import struct
from dataclasses import dataclass

import numpy as np

from ._environment import get_environment
from .geometry import PolygonBuffer
from .multiprocessing_support import fork_context

ENCODING_VERSION = 1
VALIDITY_VALID = 0
VALIDITY_CONSERVATIVE_RAW = 1
RECTANGLE_ENCODING_VERSION = 2
_WORKER_BUFFER = None
_WORKER_FORCE_HASH_COLLISION = False


@dataclass(frozen=True)
class CanonicalRecord:
    polygon_index: int
    blob: bytes
    digest128: bytes
    validity_class: int


@dataclass(frozen=True)
class InternResult:
    canonical_ids: np.ndarray
    unique_blobs: tuple[bytes, ...]
    records: tuple[CanonicalRecord, ...]
    duplicate_occurrence_count: int
    hash_collision_bucket_count: int


def canonicalize_polygon(points: np.ndarray, *, valid: bool = True) -> tuple[bytes, int]:
    point_tuple = tuple((int(x), int(y)) for x, y in points)
    if valid:
        encoded_points = _canonical_valid_points(point_tuple)
        validity = VALIDITY_VALID
    else:
        encoded_points = point_tuple
        validity = VALIDITY_CONSERVATIVE_RAW
    return _encode_points(encoded_points, validity), validity


def intern_polygons(buffer: PolygonBuffer, *, force_hash_collision: bool = False) -> InternResult:
    records = _canonical_records(buffer, force_hash_collision=force_hash_collision)

    unique_blobs = tuple(sorted({record.blob for record in records}))
    id_by_blob = {blob: index for index, blob in enumerate(unique_blobs)}
    canonical_ids = np.asarray([id_by_blob[record.blob] for record in records], dtype=np.uint64)

    digest_to_blobs: dict[bytes, set[bytes]] = {}
    for record in records:
        digest_to_blobs.setdefault(record.digest128, set()).add(record.blob)
    collision_buckets = sum(1 for blobs in digest_to_blobs.values() if len(blobs) > 1)

    return InternResult(
        canonical_ids=canonical_ids,
        unique_blobs=unique_blobs,
        records=tuple(records),
        duplicate_occurrence_count=len(records) - len(unique_blobs),
        hash_collision_bucket_count=collision_buckets,
    )


def _canonical_records(buffer: PolygonBuffer, *, force_hash_collision: bool) -> list[CanonicalRecord]:
    worker_count = _resolve_worker_count(buffer.polygon_count)
    if worker_count <= 1:
        return _canonical_records_range(buffer, 0, buffer.polygon_count, force_hash_collision)
    chunk_count = max(worker_count, worker_count * 8)
    chunk_size = max(1, (buffer.polygon_count + chunk_count - 1) // chunk_count)
    chunks = tuple((start, min(start + chunk_size, buffer.polygon_count)) for start in range(0, buffer.polygon_count, chunk_size))
    context = fork_context()
    if context is None:
        return _canonical_records_range(buffer, 0, buffer.polygon_count, force_hash_collision)
    global _WORKER_BUFFER, _WORKER_FORCE_HASH_COLLISION
    _WORKER_BUFFER = buffer
    _WORKER_FORCE_HASH_COLLISION = force_hash_collision
    try:
        try:
            with context.Pool(processes=worker_count) as pool:
                chunk_records = pool.map(_canonical_records_worker, chunks, chunksize=1)
        except OSError:
            return _canonical_records_range(buffer, 0, buffer.polygon_count, force_hash_collision)
    finally:
        _WORKER_BUFFER = None
        _WORKER_FORCE_HASH_COLLISION = False
    records = [record for chunk in chunk_records for record in chunk]
    records.sort(key=lambda record: record.polygon_index)
    return records


def _resolve_worker_count(polygon_count: int) -> int:
    if polygon_count < _parallel_min_polygons():
        return 1
    return max(1, os.cpu_count() or 1)


def _parallel_min_polygons() -> int:
    try:
        value = int(
            get_environment(
                "GDSDIFF_PARALLEL_MIN_POLYGONS",
                "GDS_GPU_COMPARE_PARALLEL_MIN_POLYGONS",
                "",
            )
            or ""
        )
    except ValueError:
        return 30_000
    return value if value > 0 else 30_000


def _canonical_records_worker(chunk: tuple[int, int]) -> list[CanonicalRecord]:
    if _WORKER_BUFFER is None:
        raise RuntimeError("canonicalization worker globals are not initialized")
    start, stop = chunk
    return _canonical_records_range(_WORKER_BUFFER, start, stop, _WORKER_FORCE_HASH_COLLISION)


def _canonical_records_range(
    buffer: PolygonBuffer,
    start: int,
    stop: int,
    force_hash_collision: bool,
) -> list[CanonicalRecord]:
    records: list[CanonicalRecord] = []
    for index in range(start, stop):
        valid = int(buffer.flags[index]) == 0
        if valid and int(buffer.polygon_offsets[index + 1] - buffer.polygon_offsets[index]) == 4 and _is_axis_aligned_rectangle(
            buffer.polygon_vertices(index)
        ):
            blob = _encode_rectangle_bbox(buffer.bboxes[index])
            validity = VALIDITY_VALID
        else:
            blob, validity = canonicalize_polygon(buffer.polygon_vertices(index), valid=valid)
        digest = b"\0" * 16 if force_hash_collision else hashlib.blake2b(blob, digest_size=16).digest()
        records.append(CanonicalRecord(index, blob, digest, validity))
    return records


def _is_axis_aligned_rectangle(points: np.ndarray) -> bool:
    if points.shape != (4, 2):
        return False
    return len({int(value) for value in points[:, 0]}) == 2 and len({int(value) for value in points[:, 1]}) == 2


def _encode_rectangle_bbox(bbox: np.ndarray) -> bytes:
    x0, y0, x1, y1 = (int(value) for value in bbox)
    return struct.pack(">IIqqqq", RECTANGLE_ENCODING_VERSION, VALIDITY_VALID, x0, y0, x1, y1)


def _canonical_valid_points(points: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    if not points:
        return points
    forward = _min_cyclic_rotation(points)
    reverse = _min_cyclic_rotation(tuple(reversed(points)))
    return min(forward, reverse)


def _min_cyclic_rotation(points: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    first_point = min(points)
    return min(points[index:] + points[:index] for index, point in enumerate(points) if point == first_point)


def _encode_points(points: tuple[tuple[int, int], ...], validity: int) -> bytes:
    out = bytearray()
    out.extend(struct.pack(">III", ENCODING_VERSION, validity, len(points)))
    for x, y in points:
        out.extend(struct.pack(">qq", x, y))
    return bytes(out)
