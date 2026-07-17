from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import gdsdiff.native_cache as native_cache
from gdsdiff.native_cache import (
    ENV_NATIVE_CACHE_DIR,
    content_addressed_library_path,
    ensure_cached_native_library,
)


def test_content_address_changes_library_path(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_NATIVE_CACHE_DIR, str(tmp_path / "cache"))
    source = tmp_path / "source.cpp"
    source.write_text("first", encoding="utf-8")
    first = content_addressed_library_path("probe.so", sources=(source,), build_key=("g++", "-O3"))

    source.write_text("second", encoding="utf-8")
    second = content_addressed_library_path("probe.so", sources=(source,), build_key=("g++", "-O3"))

    assert first != second
    assert first.suffix == second.suffix == ".so"


def test_compiler_identity_changes_library_path(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_NATIVE_CACHE_DIR, str(tmp_path / "cache"))
    source = tmp_path / "source.cpp"
    source.write_text("source", encoding="utf-8")
    monkeypatch.setattr(native_cache, "_executable_fingerprint", lambda command, path: ("compiler=one",))
    first = content_addressed_library_path("probe.so", sources=(source,), build_key=("g++", "-O3"))
    monkeypatch.setattr(native_cache, "_executable_fingerprint", lambda command, path: ("compiler=two",))
    second = content_addressed_library_path("probe.so", sources=(source,), build_key=("g++", "-O3"))
    assert first != second


def test_native_cuda_device_changes_library_path(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_NATIVE_CACHE_DIR, str(tmp_path / "cache"))
    source = tmp_path / "source.cu"
    source.write_text("source", encoding="utf-8")
    monkeypatch.setattr(native_cache, "_executable_fingerprint", lambda command, path: ("nvcc=13.3",))
    monkeypatch.setattr(native_cache, "_cuda_device_fingerprint", lambda visible: ("device=sm_120",))
    first = content_addressed_library_path("probe.so", sources=(source,), build_key=("nvcc", "-arch=native"))
    monkeypatch.setattr(native_cache, "_cuda_device_fingerprint", lambda visible: ("device=sm_100",))
    second = content_addressed_library_path("probe.so", sources=(source,), build_key=("nvcc", "-arch=native"))
    assert first != second


def test_concurrent_cache_build_publishes_one_complete_library(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_NATIVE_CACHE_DIR, str(tmp_path / "cache"))
    source = tmp_path / "source.cpp"
    source.write_text("source", encoding="utf-8")
    build_count = 0
    build_count_lock = threading.Lock()

    def builder(path):
        nonlocal build_count
        with build_count_lock:
            build_count += 1
        time.sleep(0.05)
        path.write_bytes(b"complete-library")

    def ensure():
        return ensure_cached_native_library(
            "probe.so",
            sources=(source,),
            build_key=("fake-compiler",),
            builder=builder,
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        paths = tuple(executor.map(lambda _index: ensure(), range(8)))

    assert len(set(paths)) == 1
    assert paths[0].read_bytes() == b"complete-library"
    assert build_count == 1
    assert not tuple(paths[0].parent.glob(".*.lock"))
    if native_cache.os.name != "nt":
        assert paths[0].parent.stat().st_mode & 0o077 == 0


def test_failed_build_does_not_publish_partial_library(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_NATIVE_CACHE_DIR, str(tmp_path / "cache"))
    source = tmp_path / "source.cpp"
    source.write_text("source", encoding="utf-8")

    def builder(path):
        path.write_bytes(b"partial")
        raise RuntimeError("compile failed")

    with pytest.raises(RuntimeError, match="compile failed"):
        ensure_cached_native_library(
            "probe.so",
            sources=(source,),
            build_key=("fake-compiler",),
            builder=builder,
        )

    assert not tuple((tmp_path / "cache").rglob("*.so"))
    assert not tuple((tmp_path / "cache").rglob(".*.lock"))


def test_windows_access_denied_for_existing_lock_is_retried(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_NATIVE_CACHE_DIR, str(tmp_path / "cache"))
    monkeypatch.setattr(native_cache.sys, "platform", "win32")
    source = tmp_path / "source.cpp"
    source.write_text("source", encoding="utf-8")
    real_open = native_cache.os.open
    attempts = 0

    def open_with_one_sharing_violation(path, flags, mode=0o777):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError(13, "simulated Windows sharing violation", path)
        return real_open(path, flags, mode)

    monkeypatch.setattr(native_cache.os, "open", open_with_one_sharing_violation)
    path = ensure_cached_native_library(
        "probe.so",
        sources=(source,),
        build_key=("fake-compiler",),
        builder=lambda target: target.write_bytes(b"complete-library"),
    )

    assert attempts >= 2
    assert path.read_bytes() == b"complete-library"
