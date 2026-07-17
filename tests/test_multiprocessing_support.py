from __future__ import annotations

import multiprocessing as mp

from gdsdiff import multiprocessing_support
from gdsdiff import extract as extract_module


def test_fork_context_is_absent_when_platform_has_no_fork(monkeypatch):
    monkeypatch.setattr(mp, "get_all_start_methods", lambda: ["spawn"])

    assert multiprocessing_support.fork_context() is None


def test_fork_context_is_absent_when_another_thread_is_active(monkeypatch):
    monkeypatch.setattr(mp, "get_all_start_methods", lambda: ["fork", "spawn"])
    monkeypatch.setattr(multiprocessing_support.threading, "active_count", lambda: 2)

    assert multiprocessing_support.fork_context() is None


def test_spawned_pair_worker_never_uses_nested_fork(monkeypatch):
    monkeypatch.setattr(mp, "get_all_start_methods", lambda: ["fork", "spawn"])
    monkeypatch.setattr(multiprocessing_support.threading, "active_count", lambda: 1)
    monkeypatch.setattr(multiprocessing_support, "_IS_PORTABLE_PROCESS_WORKER", False)

    multiprocessing_support.mark_portable_process_worker()

    assert multiprocessing_support.fork_context() is None


def test_python_314_never_uses_deprecated_fork(monkeypatch):
    monkeypatch.setattr(multiprocessing_support.sys, "version_info", (3, 14, 0))
    monkeypatch.setattr(mp, "get_all_start_methods", lambda: ["fork", "spawn"])
    monkeypatch.setattr(multiprocessing_support.threading, "active_count", lambda: 1)
    monkeypatch.setattr(multiprocessing_support, "_IS_PORTABLE_PROCESS_WORKER", False)

    assert multiprocessing_support.fork_context() is None


def test_large_extraction_uses_serial_path_when_platform_has_no_fork(monkeypatch):
    monkeypatch.setattr(extract_module, "fork_context", lambda: None)
    monkeypatch.setattr(extract_module, "_resolve_extract_worker_count", lambda _count: 2)
    raw_polygons = []
    selected_layers = set()

    records, max_error = extract_module._extract_polygon_records_fast(raw_polygons, selected_layers, 1)

    assert records == []
    assert str(max_error) == "0.0"
