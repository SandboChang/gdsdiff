from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

from .canonicalize import InternResult, intern_polygons
from .config import CompareConfig, LayerSpec
from .backends.polygon_extract_ctypes import NativePolygonExtraction, extract_gds_polygons_native
from .exact_oracle import ExactOracleResult
from .extract import ExtractionResult, extract_gds_polygons
from .semantics import PathPolicy
from .spatial import SpatialIndex
from .tiling import CpuTilePrefilterResult
from .windows import CandidateWindow

CACHE_SCHEMA_VERSION = "gds-gpu-compare-geometry-cache-v1"
NATIVE_POLYGON_CACHE_SCHEMA_VERSION = "gds-gpu-compare-native-polygon-cache-v1"
PAIR_VERDICT_SCHEMA_VERSION = "gds-gpu-compare-pair-verdict-cache-v2"


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
    cache_path = cache_dir / f"{key}.pkl"
    load_start = perf_counter()
    if cache_path.exists():
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
        load_s = perf_counter() - load_start
        if payload.get("schema_version") == CACHE_SCHEMA_VERSION and payload.get("cache_key") == key:
            return CachedGeometry(
                extraction=payload["extraction"],
                intern=payload["intern"],
                spatial_index=payload["spatial_index"],
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
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "cache_key": key,
                "extraction": extraction,
                "intern": intern,
                "spatial_index": spatial_index,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    tmp_path.replace(cache_path)
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
    cache_path = cache_dir / "native_polygons" / f"{key}.pkl"
    load_start = perf_counter()
    if cache_path.exists():
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
        load_s = perf_counter() - load_start
        if payload.get("schema_version") == NATIVE_POLYGON_CACHE_SCHEMA_VERSION and payload.get("cache_key") == key:
            return CachedNativePolygonExtraction(
                extraction=payload["extraction"],
                cache_key=key,
                cache_hit=True,
                load_s=load_s,
            )

    build_start = perf_counter()
    extraction = extract_gds_polygons_native(path, top_name=top_name, layers=layers)
    build_s = perf_counter() - build_start

    write_start = perf_counter()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(
            {
                "schema_version": NATIVE_POLYGON_CACHE_SCHEMA_VERSION,
                "cache_key": key,
                "extraction": extraction,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    tmp_path.replace(cache_path)
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
    path = Path(config.geometry_cache_dir) / "pair_verdicts" / f"{key}.pkl"
    if not path.exists():
        return None
    start = perf_counter()
    with path.open("rb") as handle:
        payload = pickle.load(handle)
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
    path = cache_dir / f"{key}.pkl"
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(
            {
                "schema_version": PAIR_VERDICT_SCHEMA_VERSION,
                "cache_key": key,
                "fast_status": fast_status,
                "exact_fallback": exact_fallback,
                "exact_verification_complete": exact_verification_complete,
                "layer_exact_status": layer_exact_status,
                "warnings": warnings,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    tmp_path.replace(path)
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
