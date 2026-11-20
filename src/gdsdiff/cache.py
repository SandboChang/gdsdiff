from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np

from .canonicalize import CanonicalRecord, InternResult, intern_polygons
from .config import CompareConfig, LayerSpec
from .backends.polygon_extract_ctypes import NativePolygonExtraction, extract_gds_polygons_native
from .exact_oracle import ExactOracleResult
from .extract import ExtractionResult, ExtractionStats, extract_gds_polygons
from .geometry import LayerTable, PolygonBuffer
from .native_cache import _exclusive_cache_lock
from .semantics import PathPolicy
from .spatial import SpatialIndex
from .tiling import CpuTilePrefilterResult
from .windows import CandidateWindow

CACHE_SCHEMA_VERSION = "gdsdiff-geometry-cache-v2"
NATIVE_POLYGON_CACHE_SCHEMA_VERSION = "gdsdiff-native-polygon-cache-v4"
PAIR_VERDICT_SCHEMA_VERSION = "gdsdiff-pair-verdict-cache-v3"


@dataclass(frozen=True)
class CachedGeometry:
    extraction: ExtractionResult
    intern: InternResult
    spatial_index: SpatialIndex
    cache_key: str
    cache_hit: bool
    load_s: float = 0.0
    build_s: float = 0.0
    write_s: float = 0.0

    def to_timing_json(self, prefix: str) -> dict[str, float | bool | str]:
        return {
            f"{prefix}_geometry_cache_hit": self.cache_hit,
            f"{prefix}_geometry_cache_load_s": self.load_s,
            f"{prefix}_geometry_cache_build_s": self.build_s,
            f"{prefix}_geometry_cache_write_s": self.write_s,
            f"{prefix}_geometry_cache_key": self.cache_key,
        }


@dataclass(frozen=True)
class CachedNativePolygonExtraction:
    extraction: NativePolygonExtraction
    cache_key: str
    cache_hit: bool
    load_s: float = 0.0
    build_s: float = 0.0
    write_s: float = 0.0

    def to_timing_json(self, prefix: str) -> dict[str, float | bool | str]:
        return {
            f"{prefix}_native_polygon_cache_hit": self.cache_hit,
            f"{prefix}_native_polygon_cache_load_s": self.load_s,
            f"{prefix}_native_polygon_cache_build_s": self.build_s,
            f"{prefix}_native_polygon_cache_write_s": self.write_s,
            f"{prefix}_native_polygon_cache_key": self.cache_key,
        }


def load_or_build_cached_geometry(
    *,
    cache_dir: str | Path,
    path: str | Path,
    top_name: str | None,
    layers: str | LayerSpec | Iterable[str | tuple[int, int] | LayerSpec],
    meters_per_dbu,
    path_policy: PathPolicy | str,
) -> CachedGeometry:
    cache_dir = Path(cache_dir)
    path = Path(path)
    key = geometry_cache_key(
        path=path,
        top_name=top_name,
        layers=layers,
        meters_per_dbu=meters_per_dbu,
        path_policy=path_policy,
    )
    manifest_path = cache_dir / f"{key}.json"
    arrays_path = cache_dir / f"{key}.npz"
    load_start = perf_counter()
    cached = _read_array_cache(manifest_path, arrays_path, CACHE_SCHEMA_VERSION, key)
    if cached is not None:
        manifest, arrays = cached
        try:
            buffer = _buffer_from_cache(manifest, arrays)
            extraction = ExtractionResult(
                path=path,
                top_name=str(manifest["top_name"]),
                buffer=buffer,
                stats=ExtractionStats(
                    raw_polygon_count=int(manifest["raw_polygon_count"]),
                    kept_polygon_count=int(manifest["kept_polygon_count"]),
                    vertex_count=int(manifest["vertex_count"]),
                    max_snap_error_dbu=Decimal(str(manifest["max_snap_error_dbu"])),
                ),
            )
            intern = _intern_from_cache(
                arrays,
                duplicate_occurrence_count=int(manifest["duplicate_occurrence_count"]),
                hash_collision_bucket_count=int(manifest["hash_collision_bucket_count"]),
            )
            spatial_index = _spatial_from_cache(buffer, arrays)
        except (KeyError, TypeError, ValueError, IndexError):
            cached = None
        else:
            load_s = perf_counter() - load_start
            return CachedGeometry(
                extraction=extraction,
                intern=intern,
                spatial_index=spatial_index,
                cache_key=key,
                cache_hit=True,
                load_s=load_s,
            )

    build_start = perf_counter()
    extraction = extract_gds_polygons(
        path,
        top_name=top_name,
        layers=layers,
        meters_per_dbu=meters_per_dbu,
        path_policy=path_policy,
    )
    intern = intern_polygons(extraction.buffer)
    spatial_index = SpatialIndex.from_buffer(extraction.buffer)
    build_s = perf_counter() - build_start

    write_start = perf_counter()
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_key": key,
        "top_name": extraction.top_name,
        "raw_polygon_count": extraction.stats.raw_polygon_count,
        "kept_polygon_count": extraction.stats.kept_polygon_count,
        "vertex_count": extraction.stats.vertex_count,
        "max_snap_error_dbu": str(extraction.stats.max_snap_error_dbu),
        "layers": extraction.buffer.layer_table.to_json(),
        "polygon_hashes_present": extraction.buffer.polygon_hashes is not None,
        "duplicate_occurrence_count": intern.duplicate_occurrence_count,
        "hash_collision_bucket_count": intern.hash_collision_bucket_count,
    }
    arrays = _geometry_cache_arrays(extraction.buffer, intern)
    _write_array_cache(manifest_path, arrays_path, manifest, arrays)
    write_s = perf_counter() - write_start
    return CachedGeometry(
        extraction=extraction,
        intern=intern,
        spatial_index=spatial_index,
        cache_key=key,
        cache_hit=False,
        build_s=build_s,
        write_s=write_s,
    )


def load_or_build_cached_native_polygons(
    *,
    cache_dir: str | Path,
    path: str | Path,
    top_name: str | None,
    layers: str | LayerSpec | Iterable[str | tuple[int, int] | LayerSpec],
) -> CachedNativePolygonExtraction:
    cache_dir = Path(cache_dir)
    path = Path(path)
    key = native_polygon_cache_key(path=path, top_name=top_name, layers=layers)
    manifest_path = cache_dir / "native_polygons" / f"{key}.json"
    arrays_path = cache_dir / "native_polygons" / f"{key}.npz"
    load_start = perf_counter()
    cached = _read_array_cache(manifest_path, arrays_path, NATIVE_POLYGON_CACHE_SCHEMA_VERSION, key)
    if cached is not None:
        manifest, arrays = cached
        try:
            extraction = NativePolygonExtraction(
                path=path,
                top_name=manifest.get("top_name"),
                buffer=_buffer_from_cache(manifest, arrays),
                native_parse_s=manifest.get("native_parse_s"),
                oracle_reuse_eligible=manifest.get("oracle_reuse_eligible") is True,
            )
        except (KeyError, TypeError, ValueError, IndexError):
            cached = None
        else:
            load_s = perf_counter() - load_start
            return CachedNativePolygonExtraction(
                extraction=extraction,
                cache_key=key,
                cache_hit=True,
                load_s=load_s,
            )

    build_start = perf_counter()
    extraction = extract_gds_polygons_native(path, top_name=top_name, layers=layers)
    build_s = perf_counter() - build_start

    write_start = perf_counter()
    manifest = {
        "schema_version": NATIVE_POLYGON_CACHE_SCHEMA_VERSION,
        "cache_key": key,
        "top_name": extraction.top_name,
        "native_parse_s": extraction.native_parse_s,
        "oracle_reuse_eligible": extraction.oracle_reuse_eligible,
        "layers": extraction.buffer.layer_table.to_json(),
        "polygon_hashes_present": extraction.buffer.polygon_hashes is not None,
    }
    _write_array_cache(manifest_path, arrays_path, manifest, _buffer_cache_arrays(extraction.buffer))
    write_s = perf_counter() - write_start
    return CachedNativePolygonExtraction(
        extraction=extraction,
        cache_key=key,
        cache_hit=False,
        build_s=build_s,
        write_s=write_s,
    )


def geometry_cache_key(
    *,
    path: Path,
    top_name: str | None,
    layers,
    meters_per_dbu,
    path_policy,
) -> str:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "path_sha256": digest.hexdigest(),
        "path_size": stat.st_size,
        "top_name": top_name,
        "layers": _jsonable_layers(layers),
        "meters_per_dbu": str(meters_per_dbu),
        "path_policy": str(PathPolicy.coerce(path_policy).value),
    }
    return hashlib.sha256(json.dumps(metadata, sort_keys=True).encode("utf-8")).hexdigest()


def native_polygon_cache_key(
    *,
    path: Path,
    top_name: str | None,
    layers,
) -> str:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    metadata = {
        "schema_version": NATIVE_POLYGON_CACHE_SCHEMA_VERSION,
        "path_sha256": digest.hexdigest(),
        "path_size": stat.st_size,
        "top_name": top_name,
        "layers": _jsonable_layers(layers),
    }
    return hashlib.sha256(json.dumps(metadata, sort_keys=True).encode("utf-8")).hexdigest()


def _jsonable_layers(layers):
    if layers == "all":
        return "all"
    return [LayerSpec.parse(layer).to_json() for layer in layers]


@dataclass(frozen=True)
class CachedPairVerdict:
    fast_status: str
    exact_fallback: dict[str, object]
    exact_verification_complete: bool
    layer_exact_status: str
    warnings: tuple[str, ...]
    cache_key: str
    load_s: float


def load_cached_pair_verdict(config: CompareConfig, *, old_top_name: str | None, new_top_name: str | None) -> CachedPairVerdict | None:
    if config.geometry_cache_dir is None:
        return None
    key = pair_verdict_cache_key(config, old_top_name=old_top_name, new_top_name=new_top_name)
    path = Path(config.geometry_cache_dir) / "pair_verdicts" / f"{key}.json"
    if not path.exists():
        return None
    start = perf_counter()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    load_s = perf_counter() - start
    if payload.get("schema_version") != PAIR_VERDICT_SCHEMA_VERSION or payload.get("cache_key") != key:
        return None
    return CachedPairVerdict(
        fast_status=str(payload["fast_status"]),
        exact_fallback=payload["exact_fallback"],
        exact_verification_complete=bool(payload["exact_verification_complete"]),
        layer_exact_status=str(payload["layer_exact_status"]),
        warnings=tuple(payload["warnings"]),
        cache_key=key,
        load_s=load_s,
    )


def write_cached_pair_verdict(
    config: CompareConfig,
    *,
    old_top_name: str | None,
    new_top_name: str | None,
    exact_result: ExactOracleResult,
    prefilter: CpuTilePrefilterResult,
    windows: tuple[CandidateWindow, ...],
    exact_fallback: dict[str, object],
    fast_status: str,
    exact_verification_complete: bool,
    layer_exact_status: str,
    warnings: tuple[str, ...],
) -> str | None:
    if config.geometry_cache_dir is None:
        return None
    key = pair_verdict_cache_key(config, old_top_name=old_top_name, new_top_name=new_top_name)
    cache_dir = Path(config.geometry_cache_dir) / "pair_verdicts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    _atomic_write_json(
        path,
        {
            "schema_version": PAIR_VERDICT_SCHEMA_VERSION,
            "cache_key": key,
            "fast_status": fast_status,
            "exact_fallback": exact_fallback,
            "exact_verification_complete": exact_verification_complete,
            "layer_exact_status": layer_exact_status,
            "warnings": list(warnings),
        },
    )
    return key


def pair_verdict_cache_key(config: CompareConfig, *, old_top_name: str | None, new_top_name: str | None) -> str:
    old_key = geometry_cache_key(
        path=Path(config.old_path),
        top_name=old_top_name,
        layers=config.layers,
        meters_per_dbu=config.dbu,
        path_policy=config.path_policy,
    )
    new_key = geometry_cache_key(
        path=Path(config.new_path),
        top_name=new_top_name,
        layers=config.layers,
        meters_per_dbu=config.dbu,
        path_policy=config.path_policy,
    )
    metadata = {
        "schema_version": PAIR_VERDICT_SCHEMA_VERSION,
        "old_geometry_key": old_key,
        "new_geometry_key": new_key,
        "tile_size": config.tile_size,
        "layers": _jsonable_layers(config.layers),
        "allowed_regions": [region.to_json() for region in config.allowed_regions],
        "failure_policy": config.failure_policy.value,
    }
    return hashlib.sha256(json.dumps(metadata, sort_keys=True).encode("utf-8")).hexdigest()


def _buffer_cache_arrays(buffer: PolygonBuffer) -> dict[str, np.ndarray]:
    return {
        "vertices_xy": buffer.vertices_xy,
        "polygon_offsets": buffer.polygon_offsets,
        "layer_ids": buffer.layer_ids,
        "bboxes": buffer.bboxes,
        "flags": buffer.flags,
        "source_ids": buffer.source_ids,
        "polygon_hashes": (
            buffer.polygon_hashes if buffer.polygon_hashes is not None else np.empty((0, 2), dtype=np.uint64)
        ),
    }


def _geometry_cache_arrays(buffer: PolygonBuffer, intern: InternResult) -> dict[str, np.ndarray]:
    arrays = _buffer_cache_arrays(buffer)
    blob_offsets = [0]
    blob_bytes = bytearray()
    for blob in intern.unique_blobs:
        blob_bytes.extend(blob)
        blob_offsets.append(len(blob_bytes))
    arrays.update(
        {
            "canonical_ids": intern.canonical_ids,
            "unique_blob_bytes": np.frombuffer(bytes(blob_bytes), dtype=np.uint8),
            "unique_blob_offsets": np.asarray(blob_offsets, dtype=np.uint64),
            "record_polygon_indices": np.asarray([record.polygon_index for record in intern.records], dtype=np.uint64),
            "record_digests": np.frombuffer(b"".join(record.digest128 for record in intern.records), dtype=np.uint8).reshape((-1, 16)),
            "record_validity": np.asarray([record.validity_class for record in intern.records], dtype=np.uint8),
            "spatial_order": np.lexsort(
                (
                    np.arange(buffer.polygon_count, dtype=np.int64),
                    buffer.bboxes[:, 3],
                    buffer.bboxes[:, 2],
                    buffer.bboxes[:, 1],
                    buffer.bboxes[:, 0],
                    buffer.layer_ids,
                )
            ).astype(np.int64, copy=False),
        }
    )
    return arrays


def _buffer_from_cache(manifest: dict[str, object], arrays: dict[str, np.ndarray]) -> PolygonBuffer:
    layer_table = LayerTable(
        tuple(LayerSpec(int(item["layer"]), int(item["datatype"])) for item in manifest["layers"])
    )
    polygon_hashes = arrays["polygon_hashes"] if manifest.get("polygon_hashes_present") else None
    return PolygonBuffer(
        vertices_xy=np.asarray(arrays["vertices_xy"], dtype=np.int64),
        polygon_offsets=np.asarray(arrays["polygon_offsets"], dtype=np.uint64),
        layer_ids=np.asarray(arrays["layer_ids"], dtype=np.uint32),
        bboxes=np.asarray(arrays["bboxes"], dtype=np.int64),
        flags=np.asarray(arrays["flags"], dtype=np.uint32),
        source_ids=np.asarray(arrays["source_ids"], dtype=np.uint64),
        layer_table=layer_table,
        polygon_hashes=None if polygon_hashes is None else np.asarray(polygon_hashes, dtype=np.uint64),
    )


def _intern_from_cache(
    arrays: dict[str, np.ndarray],
    *,
    duplicate_occurrence_count: int,
    hash_collision_bucket_count: int,
) -> InternResult:
    offsets = arrays["unique_blob_offsets"]
    blob_data = arrays["unique_blob_bytes"].tobytes()
    unique_blobs = tuple(blob_data[int(offsets[index]) : int(offsets[index + 1])] for index in range(len(offsets) - 1))
    canonical_ids = np.asarray(arrays["canonical_ids"], dtype=np.uint64)
    indices = np.asarray(arrays["record_polygon_indices"], dtype=np.uint64)
    digests = np.asarray(arrays["record_digests"], dtype=np.uint8)
    validity = np.asarray(arrays["record_validity"], dtype=np.uint8)
    records = tuple(
        CanonicalRecord(
            int(index),
            unique_blobs[int(canonical_ids[int(index)])],
            bytes(digests[position]),
            int(validity[position]),
        )
        for position, index in enumerate(indices)
    )
    return InternResult(
        canonical_ids=canonical_ids,
        unique_blobs=unique_blobs,
        records=records,
        duplicate_occurrence_count=duplicate_occurrence_count,
        hash_collision_bucket_count=hash_collision_bucket_count,
    )


def _spatial_from_cache(buffer: PolygonBuffer, arrays: dict[str, np.ndarray]) -> SpatialIndex:
    order = np.asarray(arrays["spatial_order"], dtype=np.int64)
    entries = {}
    bbox_arrays = {}
    index_arrays = {}
    for layer_id, layer in enumerate(buffer.layer_table.layers):
        indices = order[buffer.layer_ids[order] == layer_id]
        bboxes = np.asarray(buffer.bboxes[indices], dtype=np.int64).reshape((-1, 4))
        entries[layer] = tuple((tuple(int(value) for value in bbox), int(index)) for bbox, index in zip(bboxes, indices))
        bbox_arrays[layer] = bboxes
        index_arrays[layer] = indices
    return SpatialIndex(entries, bbox_arrays_by_layer=bbox_arrays, index_arrays_by_layer=index_arrays)


def _read_array_cache(
    manifest_path: Path,
    arrays_path: Path,
    schema_version: str,
    cache_key: str,
) -> tuple[dict[str, object], dict[str, np.ndarray]] | None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != schema_version or manifest.get("cache_key") != cache_key:
            return None
        if _sha256_path(arrays_path) != manifest.get("arrays_sha256"):
            return None
        with np.load(arrays_path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, KeyError):
        return None
    return manifest, arrays


def _write_array_cache(
    manifest_path: Path,
    arrays_path: Path,
    manifest: dict[str, object],
    arrays: dict[str, np.ndarray],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = manifest_path.with_name(f".{manifest_path.name}.lock")
    with _exclusive_cache_lock(lock_path):
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{arrays_path.name}.", suffix=".tmp", dir=arrays_path.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.savez_compressed(handle, **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            temporary_path = Path(temporary_name)
            committed_manifest = {**manifest, "arrays_sha256": _sha256_path(temporary_path)}
            os.replace(temporary_path, arrays_path)
            _atomic_write_json(manifest_path, committed_manifest)
        finally:
            Path(temporary_name).unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
