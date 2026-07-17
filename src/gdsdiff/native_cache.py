from __future__ import annotations

import hashlib
import os
import platform
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterable, Iterator


ENV_NATIVE_CACHE_DIR = "GDSDIFF_NATIVE_CACHE_DIR"
LEGACY_ENV_NATIVE_CACHE_DIR = "SCGDS_NATIVE_CACHE_DIR"


def cached_native_library_path(filename: str) -> Path:
    return native_cache_dir() / filename


def content_addressed_library_path(
    filename: str,
    *,
    sources: Iterable[Path],
    build_key: Iterable[str] = (),
) -> Path:
    """Return a cache path keyed by source content and the build contract."""

    source_paths = tuple(Path(source) for source in sources)
    digest = hashlib.sha256(b"gdsdiff-native-cache-v1\0")
    for item in build_key:
        digest.update(str(item).encode("utf-8"))
        digest.update(b"\0")
    for source in source_paths:
        digest.update(source.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    source_hash = digest.hexdigest()[:20]
    candidate = Path(filename)
    suffix = candidate.suffix
    stem = candidate.name[: -len(suffix)] if suffix else candidate.name
    return native_cache_dir() / f"{stem}-{source_hash}{suffix}"


def ensure_cached_native_library(
    filename: str,
    *,
    sources: Iterable[Path],
    build_key: Iterable[str],
    builder: Callable[[Path], None],
) -> Path:
    """Build and atomically publish a content-addressed native library."""

    source_paths = tuple(Path(source) for source in sources)
    path = content_addressed_library_path(filename, sources=source_paths, build_key=build_key)
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with _exclusive_cache_lock(lock_path):
        if path.is_file():
            return path
        temporary = path.with_name(f".{path.stem}.{os.getpid()}.{uuid.uuid4().hex}{path.suffix}")
        try:
            builder(temporary)
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError(f"native builder did not create a non-empty library: {temporary}")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return path


def native_cache_dir() -> Path:
    configured = os.environ.get(ENV_NATIVE_CACHE_DIR) or os.environ.get(LEGACY_ENV_NATIVE_CACHE_DIR)
    if configured:
        base = Path(configured).expanduser()
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches" / "gdsdiff" / "native"
    elif os.name == "nt":
        root = os.environ.get("LOCALAPPDATA")
        base = Path(root).expanduser() if root else Path.home() / "AppData" / "Local"
        base = base / "gdsdiff" / "native"
    else:
        root = os.environ.get("XDG_CACHE_HOME")
        base = Path(root).expanduser() if root else Path.home() / ".cache"
        base = base / "gdsdiff" / "native"
    return base / _platform_tag()


def _platform_tag() -> str:
    system = platform.system().lower() or "unknown"
    machine = platform.machine().lower() or "unknown"
    python_tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    return f"{system}-{machine}-{python_tag}"


@contextmanager
def _exclusive_cache_lock(
    path: Path,
    *,
    timeout_s: float = 300.0,
    stale_after_s: float = 900.0,
) -> Iterator[None]:
    started = time.monotonic()
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except (FileExistsError, PermissionError) as exc:
            # Windows can report an existing O_EXCL lock as access denied when
            # another process or thread still owns the directory entry.
            if isinstance(exc, PermissionError) and sys.platform != "win32":
                raise
            try:
                age_s = time.time() - path.stat().st_mtime
            except FileNotFoundError:
                continue
            except PermissionError:
                # A Windows sharing violation is itself evidence that the lock
                # is live; do not misclassify it as a cache permission failure.
                age_s = 0.0
            if age_s > stale_after_s:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                continue
            if time.monotonic() - started >= timeout_s:
                raise TimeoutError(f"timed out waiting for native cache lock: {path}")
            time.sleep(0.05)
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.close(descriptor)
        descriptor = None
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
        path.unlink(missing_ok=True)
