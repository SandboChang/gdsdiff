"""Reject partial GDSTK Booleans and track the disposition of their diagnostics."""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import warnings
from collections import Counter
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from .config import LayerSpec

_BOOLEAN_WARNING_LOCK = threading.RLock()
_HOLE_LINK = b"[GDSTK] Unable to link hole in boolean operation."
_HOLE_LINK_LINES = {_HOLE_LINK + b"\n", _HOLE_LINK + b"\r\n"}
_DIAGNOSTICS_KEY = "gdstk_boolean_diagnostics"


@dataclass(frozen=True, eq=False)
class _BooleanDiagnostic:
    stage: str
    layer: LayerSpec | None
    operation: str
    message: str
    count: int


class _BooleanEvidenceError(RuntimeError):
    def __init__(self, message: str, diagnostic: _BooleanDiagnostic | None = None):
        super().__init__(message)
        self.diagnostic = diagnostic

    @property
    def stage(self) -> str:
        return self.diagnostic.stage if self.diagnostic is not None else "gdstk-boolean"


@dataclass
class _BooleanDiagnostics:
    pid: int = field(default_factory=os.getpid)
    # Identity keys count each attempt once, even if it crosses several handlers.
    attempts: dict[_BooleanDiagnostic, str | None] = field(default_factory=dict)

    def attach(self, exact_fallback: dict) -> dict:
        if not self.attempts:
            return exact_fallback
        counts = Counter()
        for row in exact_fallback.get(_DIAGNOSTICS_KEY, ()):
            key = (row["stage"], row["layer"], row["datatype"], row["operation"], row["message"], row["recovery"])
            counts[key] += row["count"]
        for diagnostic, recovery in self.attempts.items():
            layer = diagnostic.layer
            counts[
                diagnostic.stage, layer.layer if layer is not None else None,
                layer.datatype if layer is not None else None, diagnostic.operation, diagnostic.message, recovery,
            ] += diagnostic.count

        def sort_key(item):
            stage, layer, datatype, operation, message, recovery = item[0]
            return stage, -1 if layer is None else layer, -1 if datatype is None else datatype, operation, message, recovery or ""

        rows = [
            {
                "stage": stage, "layer": layer, "datatype": datatype, "operation": operation,
                "message": message, "count": count,
                "outcome": "verified-recovery" if recovery is not None else "failed", "recovery": recovery,
            }
            for (stage, layer, datatype, operation, message, recovery), count in sorted(counts.items(), key=sort_key)
        ]
        return exact_fallback | {_DIAGNOSTICS_KEY: rows}


_CURRENT_DIAGNOSTICS: ContextVar[_BooleanDiagnostics | None] = ContextVar(
    "gdsdiff_boolean_diagnostics", default=None,
)


@contextmanager
def _collect_boolean_diagnostics():
    diagnostics = _BooleanDiagnostics()
    token = _CURRENT_DIAGNOSTICS.set(diagnostics)
    try:
        yield diagnostics
    finally:
        _CURRENT_DIAGNOSTICS.reset(token)


def _attach_boolean_diagnostics(exact_fallback: dict) -> dict:
    diagnostics = _CURRENT_DIAGNOSTICS.get()
    return diagnostics.attach(exact_fallback) if diagnostics is not None else exact_fallback


def _record_boolean_failure(error: _BooleanEvidenceError) -> None:
    diagnostics = _CURRENT_DIAGNOSTICS.get()
    if error.diagnostic is not None and diagnostics is not None and diagnostics.pid == os.getpid():
        diagnostics.attempts.setdefault(error.diagnostic, None)


@contextmanager
def _verified_boolean_recovery(method: str):
    diagnostics = _CURRENT_DIAGNOSTICS.get()
    start = len(diagnostics.attempts) if diagnostics is not None else 0
    yield
    if diagnostics is not None and len(diagnostics.attempts) > start:
        for diagnostic in tuple(diagnostics.attempts)[start:]:
            diagnostics.attempts[diagnostic] = method


@dataclass
class _NativeBooleanDiagnostics:
    hole_link_count: int = 0


def _can_capture_stderr() -> bool:
    gil_enabled = getattr(sys, "_is_gil_enabled", None)
    # Changing fd 2 is process-wide, not thread-local.
    return (
        _CURRENT_DIAGNOSTICS.get() is not None
        and threading.current_thread() is threading.main_thread()
        and threading.active_count() == 1
        and (gil_enabled is None or gil_enabled())
        # Replacing a Windows console fd can close the handle Python's stderr uses.
        and not (sys.platform == "win32" and os.isatty(2))
    )


@contextmanager
def _capture_boolean_stderr():
    native = _NativeBooleanDiagnostics()
    if not _can_capture_stderr():
        yield native
        return
    with ExitStack() as stack:
        try:
            inheritable = os.get_inheritable(2)
            output = stack.enter_context(tempfile.TemporaryFile(mode="w+b"))
            original = os.dup(2)
            stack.callback(os.close, original)
        except OSError:
            yield native
            return
        try:
            os.dup2(output.fileno(), 2, inheritable=False)
        except OSError:
            # A platform can duplicate the fd before failing to set its inheritance flag.
            os.dup2(original, 2, inheritable=inheritable)
            yield native
            return
        completed = False
        try:
            yield native
            completed = True
        finally:
            os.dup2(original, 2, inheritable=inheritable)
            output.seek(0)
            line_start = True
            for chunk in iter(lambda: output.readline(65536), b""):
                if completed and line_start and chunk in _HOLE_LINK_LINES:
                    native.hole_link_count += 1
                else:
                    pending = memoryview(chunk)
                    while pending:
                        written = os.write(2, pending)
                        if written == 0:
                            raise OSError("unable to replay native Boolean stderr")
                        pending = pending[written:]
                line_start = chunk.endswith(b"\n")


def _checked_gdstk_boolean(
    operand1, operand2, operation: str, *, precision: float,
    stage: str = "gdstk-boolean", layer: LayerSpec | None = None,
):
    import gdstk

    # GDSTK can return a partial list after a numerical error.
    warning = None
    with _BOOLEAN_WARNING_LOCK, warnings.catch_warnings(), _capture_boolean_stderr() as native:
        warnings.simplefilter("error", RuntimeWarning)
        try:
            polygons = gdstk.boolean(operand1, operand2, operation, precision=precision)
        except RuntimeWarning as exc:
            warning = exc
    if warning is None and not native.hole_link_count:
        return polygons

    message = _HOLE_LINK.decode("ascii") if native.hole_link_count else str(warning)
    layer_label = f" on {layer.layer}:{layer.datatype}" if layer is not None else ""
    reason = f"gdstk {operation}{layer_label}: {message}"
    if warning is not None and native.hole_link_count:
        reason += f" ({warning})"
    failure = _BooleanEvidenceError(
        reason, _BooleanDiagnostic(stage, layer, operation, message, native.hole_link_count or 1),
    )
    _record_boolean_failure(failure)
    raise failure from warning
