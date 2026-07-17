from __future__ import annotations

import hashlib
import ctypes
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from functools import lru_cache
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
    for item in native_build_fingerprint(tuple(build_key)):
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
    _ensure_private_cache_directory(path.parent)
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


def native_build_fingerprint(build_key: tuple[str, ...]) -> tuple[str, ...]:
    """Augment a native build contract with compiler and device identity."""

    if not build_key:
        return build_key
    compiler = build_key[0]
    fingerprint = (*build_key, *_executable_fingerprint(compiler, os.environ.get("PATH")))
    if Path(compiler).name.lower().startswith("nvcc") and "-arch=native" in build_key:
        fingerprint = (*fingerprint, *_cuda_device_fingerprint(os.environ.get("CUDA_VISIBLE_DEVICES")))
    return fingerprint


@lru_cache(maxsize=32)
def _executable_fingerprint(command: str, path: str | None) -> tuple[str, ...]:
    resolved = shutil.which(command, path=path)
    if resolved is None:
        return (f"compiler={command}", "compiler_resolved=unavailable")
    executable = Path(resolved).resolve()
    try:
        stat = executable.stat()
        identity = f"{stat.st_size}:{stat.st_mtime_ns}"
    except OSError:
        identity = "stat-unavailable"
    try:
        completed = subprocess.run(
            [str(executable), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        version = (completed.stdout or completed.stderr).strip()
    except (OSError, subprocess.SubprocessError):
        version = "version-unavailable"
    return (
        f"compiler={command}",
        f"compiler_resolved={executable}",
        f"compiler_identity={identity}",
        f"compiler_version={version}",
    )


@lru_cache(maxsize=8)
def _cuda_device_fingerprint(visible_devices: str | None) -> tuple[str, ...]:
    driver_devices = _cuda_driver_api_fingerprint()
    if driver_devices is not None:
        return (f"CUDA_VISIBLE_DEVICES={visible_devices or ''}", f"cuda_driver_devices={driver_devices}")
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,compute_cap,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        devices = completed.stdout.strip() if completed.returncode == 0 else "unavailable"
    except (OSError, subprocess.SubprocessError):
        devices = "unavailable"
    return (f"CUDA_VISIBLE_DEVICES={visible_devices or ''}", f"cuda_devices={devices}")


def _cuda_driver_api_fingerprint() -> str | None:
    """Read visible device identity without creating a CUDA context."""

    try:
        driver = ctypes.CDLL("libcuda.so.1")
        driver.cuInit.argtypes = [ctypes.c_uint]
        driver.cuInit.restype = ctypes.c_int
        driver.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        driver.cuDeviceGetCount.restype = ctypes.c_int
        driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        driver.cuDeviceGet.restype = ctypes.c_int
        driver.cuDeviceComputeCapability.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        driver.cuDeviceComputeCapability.restype = ctypes.c_int
        driver.cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        driver.cuDeviceGetName.restype = ctypes.c_int
        driver.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        driver.cuDriverGetVersion.restype = ctypes.c_int
        if driver.cuInit(0) != 0:
            return None
        count = ctypes.c_int()
        version = ctypes.c_int()
        if driver.cuDeviceGetCount(ctypes.byref(count)) != 0 or driver.cuDriverGetVersion(ctypes.byref(version)) != 0:
            return None
        devices = []
        for ordinal in range(count.value):
            device = ctypes.c_int()
            major = ctypes.c_int()
            minor = ctypes.c_int()
            name = ctypes.create_string_buffer(256)
            if driver.cuDeviceGet(ctypes.byref(device), ordinal) != 0:
                return None
            if driver.cuDeviceComputeCapability(ctypes.byref(major), ctypes.byref(minor), device.value) != 0:
                return None
            if driver.cuDeviceGetName(name, len(name), device.value) != 0:
                return None
            devices.append(f"{ordinal}:{name.value.decode('utf-8', errors='replace')}:sm_{major.value}{minor.value}")
        return f"driver_api={version.value};" + ";".join(devices)
    except (AttributeError, OSError, ValueError):
        return None


def _ensure_private_cache_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == "nt":
        return
    if path.is_symlink():
        raise PermissionError(f"native cache directory must not be a symbolic link: {path}")
    stat = path.stat()
    if hasattr(os, "getuid") and stat.st_uid != os.getuid():
        raise PermissionError(f"native cache directory is not owned by the current user: {path}")
    if stat.st_mode & 0o077:
        path.chmod(stat.st_mode & ~0o077)


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
