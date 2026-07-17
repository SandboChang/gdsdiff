from __future__ import annotations

import multiprocessing as mp
import sys
import threading
from multiprocessing.context import BaseContext

_IS_PORTABLE_PROCESS_WORKER = False


def mark_portable_process_worker() -> None:
    """Mark a spawned pair-extraction worker so it never forks again."""

    global _IS_PORTABLE_PROCESS_WORKER
    _IS_PORTABLE_PROCESS_WORKER = True


def fork_context() -> BaseContext | None:
    """Return a fork context when the platform supports one.

    Several hot paths share large, read-only geometry buffers through forked
    worker globals. Windows does not provide fork, so callers must retain a
    serial path instead of failing while attempting to create a pool.
    """

    if (
        _IS_PORTABLE_PROCESS_WORKER
        or sys.version_info >= (3, 14)
        or "fork" not in mp.get_all_start_methods()
        or threading.active_count() != 1
    ):
        return None
    return mp.get_context("fork")


def portable_process_context() -> BaseContext:
    """Return the portable context for independent, pickle-safe tasks."""

    return mp.get_context("spawn")
