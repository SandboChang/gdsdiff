from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

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
