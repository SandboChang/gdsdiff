from __future__ import annotations

import json
import os
import sys
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import gdstk
import numpy as np
import pytest

from gdsdiff import CompareConfig, ExactOracleResult, LayerSpec, compare_gds_files, extract_gds_polygons
from gdsdiff import api, exact_oracle
from gdsdiff._boolean_evidence import _BooleanEvidenceError, _checked_gdstk_boolean, _collect_boolean_diagnostics
from gdsdiff.backends import cuda_ctypes
from gdsdiff.backends.cuda_ctypes import CudaPolygonRecordDelta
from gdsdiff.backends.polygon_extract_ctypes import NativePolygonExtraction
from gdsdiff.cache import load_cached_pair_verdict
from gdsdiff.cli import main
from gdsdiff.exact_diff_markdown import _canonical_inventory_points
from gdsdiff.multiprocessing_support import fork_context
from gdsdiff.native import NativeCapabilities
from gdsdiff.polygon_components import PolygonComponentBuildResult, PolygonComponentStats
from gdsdiff.source_edge_diff import SourceEdgeDiffResult, SourceEdgeDiffStats
from gdsdiff.windows import CandidateWindow


LAYER = LayerSpec(7, 3)
NATIVE_MESSAGE = b"[GDSTK] Unable to link hole in boolean operation."
RECOVERABLE = (
    (((0, 7), (3, 1), (6, 8)), ((0, 8), (8, 3), (6, 1))),
    (((3, 5), (0, 8), (4, 5)),),
)
UNRECOVERABLE = (
    (((0, 8), (7, 4), (6, 0)),),
    (((8, 3), (2, 1), (1, 5)), ((3, 5), (4, 6), (2, 5))),
)
SIMPLE = (
    (((0, 0), (20, 0), (0, 20)),),
    (((2, 2), (6, 2), (2, 6)), ((8, 2), (12, 2), (8, 6))),
)


def _write_pair(directory, polygons, *, sy=1, layers=(LAYER,)):
    directory.mkdir()
    paths = tuple(directory / name for name in ("before.gds", "after.gds"))
    for path, shapes in zip(paths, polygons, strict=True):
        lib = gdstk.Library(unit=1e-6, precision=1e-6)
        top = lib.new_cell("TOP")
        for layer in layers:
            top.add(*(gdstk.Polygon([(x, sy * y) for x, y in shape], layer=layer.layer, datatype=layer.datatype)
                      for shape in shapes))
        lib.write_gds(path)
    return paths


def _mock_nested_cuda(monkeypatch, paths):
    native = tuple(
        NativePolygonExtraction(path, "TOP", extract_gds_polygons(path).buffer, oracle_reuse_eligible=True)
        for path in paths
    )
    old, new = (item.buffer for item in native)
    count = old.polygon_count + new.polygon_count
    delta = CudaPolygonRecordDelta(
        layers=np.full(count, LAYER.layer), datatypes=np.full(count, LAYER.datatype),
        directions=np.repeat([1, 2], [old.polygon_count, new.polygon_count]),
        counts=np.ones(count, dtype=np.uint64), bboxes=np.concatenate((old.bboxes, new.bboxes)),
        vertex_counts=np.concatenate((np.diff(old.polygon_offsets), np.diff(new.polygon_offsets))),
        flags=np.concatenate((old.flags, new.flags)), hashes=np.zeros((count, 2), dtype=np.uint64),
        representative_indices=np.r_[np.arange(old.polygon_count), np.arange(new.polygon_count)],
        old_input_count=old.polygon_count, new_input_count=new.polygon_count,
        matched_record_count=0, reduced_key_count=count, cuda_kernel_count=1, cuda_event_elapsed_s=0.001,
        transferred_bytes=0,
    )
    monkeypatch.setattr(api, "discover_native_capabilities", lambda: NativeCapabilities(
        "test", False, False, True, False, False, False, (),
    ))
    monkeypatch.setattr(api, "_gpu_polygon_native_inputs", lambda *args, **kwargs: (*native, {}))
    monkeypatch.setattr(cuda_ctypes, "run_cuda_polygon_record_delta", lambda *args: delta)
    monkeypatch.setattr(cuda_ctypes, "release_cuda_device_memory", lambda: None)
    monkeypatch.setattr(api, "add_overlapping_context_to_components", lambda *args, **kwargs: PolygonComponentBuildResult(
        (), PolygonComponentStats(0, 0, 0, 0),
    ))
    monkeypatch.setattr(
        api, "exact_result_from_source_edges",
        lambda components, *, exact_on_nested: exact_on_nested() or SourceEdgeDiffResult(
            ExactOracleResult(()), SourceEdgeDiffStats(1, 0, 0, 0, nested_direction_fragment_component_count=1),
        ),
    )


def _sidecar(path, *, sy=1):
    return Counter(
        (row["kind"], row["layer"], row["datatype"], row["direction"],
         _canonical_inventory_points(tuple((x, sy * y) for x, y in row["vertices_dbu"])))
        for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    )


def _failure(*args, **kwargs):
    os.write(2, NATIVE_MESSAGE + b"\n")
    warnings.warn("Error in boolean operation.", RuntimeWarning)
    return []


def _read_report(directory):
    return json.loads((directory / "report.json").read_text(encoding="utf-8"))


def test_native_recovered_hole_link_keeps_exact_vertices_and_stderr_quiet(tmp_path, monkeypatch, capfd):
    inputs = _write_pair(tmp_path / "original", RECOVERABLE)
    reference = _write_pair(tmp_path / "reflected", RECOVERABLE, sy=-1)
    _mock_nested_cuda(monkeypatch, inputs)
    cpu, gpu = tmp_path / "cpu", tmp_path / "gpu"

    assert main([*map(str, reference), "--backend", "cpu", "--output-dir", str(cpu)]) == 1
    assert main([*map(str, inputs), "--backend", "cuda", "--require-gpu", "--output-dir", str(gpu)]) == 1
    assert capfd.readouterr().err == ""

    report = _read_report(gpu)
    assert report["exact_verification_complete"] and report["backend"]["gpu_requirement_met"]
    assert report["errors"] == report["exact_fallback"]["failures"] == []
    assert report["exact_fallback"]["gdstk_boolean_diagnostics"] == [{
        "stage": "sparse-coverage", "layer": LAYER.layer, "datatype": LAYER.datatype, "operation": "not",
        "message": NATIVE_MESSAGE.decode(), "count": 1, "outcome": "verified-recovery",
        "recovery": "verified-sparse-coverage",
    }]
    assert _sidecar(gpu / "diff_vertices.jsonl") == _sidecar(cpu / "diff_vertices.jsonl", sy=-1)


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_native_unrecoverable_hole_link_is_visible_and_has_no_partial_artifacts(tmp_path, monkeypatch, capfd, backend):
    inputs = _write_pair(tmp_path / "inputs", UNRECOVERABLE)
    _mock_nested_cuda(monkeypatch, inputs)
    output, diff_gds = tmp_path / "failed", tmp_path / "diff.gds"
    command = [*map(str, inputs), "--backend", backend, "--quiet", "--output-dir", str(output), "--diff-gds", str(diff_gds)]
    if backend == "cuda":
        command.append("--require-gpu")

    assert main(command) == 2
    log = capfd.readouterr()
    assert log.out == ""
    assert "Unable to link hole" in log.err
    report = _read_report(output)
    assert report["status"] == "incomplete-evidence"
    assert not report["exact_verification_complete"]
    assert report["errors"] and report["exact_fallback"]["failures"]
    assert {row["outcome"] for row in report["exact_fallback"]["gdstk_boolean_diagnostics"]} == {"failed"}
    assert not (output / "diff_vertices.jsonl").exists()
    assert not diff_gds.exists()


def test_disagreeing_retries_are_visible_even_when_independent_cpu_completes(tmp_path, monkeypatch, capfd):
    inputs = _write_pair(tmp_path / "inputs", SIMPLE)
    _mock_nested_cuda(monkeypatch, inputs)
    cpu, gpu = tmp_path / "cpu", tmp_path / "gpu"
    assert main([*map(str, inputs), "--backend", "cpu", "--output-dir", str(cpu)]) == 1

    boolean = gdstk.boolean
    pending = iter((1, 2, 3))

    def warning_then_disagree(a, b, operation, **kwargs):
        attempt = next(pending, None)
        if attempt == 1:
            return _failure()
        result = boolean(a, b, operation, **kwargs)
        if attempt == 3:
            points = tuple(map(tuple, result[0].points))
            for i, (p, q) in enumerate(zip(points, points[1:] + points[:1], strict=True)):
                midpoint = tuple((x + y) / 2 for x, y in zip(p, q, strict=True))
                if p != q and all(x.is_integer() for x in midpoint):
                    replacement = gdstk.Polygon(points[:i + 1] + (midpoint,) + points[i + 1:])
                    assert replacement.area() == result[0].area()
                    result[0] = replacement
                    break
            else:
                pytest.fail("test Boolean should yield an edge with an integer midpoint")
        return result

    monkeypatch.setattr(gdstk, "boolean", warning_then_disagree)
    assert main([*map(str, inputs), "--backend", "cuda", "--require-gpu", "--output-dir", str(gpu)]) == 1
    assert "reflected vertex loops disagree" in capfd.readouterr().err
    report = _read_report(gpu)
    assert report["exact_verification_complete"] and report["backend"]["gpu_requirement_met"]
    assert report["exact_fallback"]["source_edge_cpu_oracle_fallback"]["input"] == "independent-gdstk-extraction"
    assert report["exact_fallback"]["gdstk_boolean_diagnostics"][0]["outcome"] == "failed"
    assert _sidecar(gpu / "diff_vertices.jsonl") == _sidecar(cpu / "diff_vertices.jsonl")


@pytest.mark.parametrize("operation", ["and", "or"], ids=["clip", "merge"])
def test_window_warning_discards_entire_layer_and_checks_full_replacement(tmp_path, monkeypatch, capfd, operation):
    inputs = _write_pair(tmp_path / "inputs", SIMPLE)
    reference = compare_gds_files(CompareConfig(*inputs, backend="cpu"), exact=True)
    windows = (CandidateWindow(LAYER, (0, 0, 20, 20), (-1, -1, 21, 21), 1, 1, 2, "test"),)
    monkeypatch.setattr(api, "build_candidate_windows", lambda *args: windows)
    monkeypatch.setattr(api, "_run_tile_prefilter", lambda backend, *args: api.run_cpu_tile_prefilter(*args))
    boolean = gdstk.boolean
    monkeypatch.setattr(gdstk, "boolean", lambda a, b, op, **kwargs: _failure() if op == operation else boolean(a, b, op, **kwargs))

    result = compare_gds_files(CompareConfig(*inputs, backend="cpu"))
    assert result.exact_result == reference.exact_result
    assert not result.report.exact_verification_complete
    assert result.report.exact_fallback["escalated_layers"] == [LAYER.to_json()]
    diagnostic, = result.report.exact_fallback["gdstk_boolean_diagnostics"]
    assert diagnostic["operation"] == operation
    assert diagnostic["recovery"] == "full-layer-oracle"
    assert capfd.readouterr().err == ""


def test_window_and_full_layer_failure_leave_no_exact_result(tmp_path, monkeypatch):
    inputs = _write_pair(tmp_path / "inputs", SIMPLE)
    windows = (CandidateWindow(LAYER, (0, 0, 20, 20), (-1, -1, 21, 21), 1, 1, 2, "test"),)
    monkeypatch.setattr(api, "build_candidate_windows", lambda *args: windows)
    monkeypatch.setattr(api, "_run_tile_prefilter", lambda backend, *args: api.run_cpu_tile_prefilter(*args))
    monkeypatch.setattr(gdstk, "boolean", _failure)

    result = compare_gds_files(CompareConfig(*inputs, backend="cpu"), exact_diff_markdown_path=tmp_path / "failed.md")
    assert result.exit_code == 2
    assert not result.exact_result.layers
    assert not result.report.exact_verification_complete
    assert not (tmp_path / "failed_diff_vertices.jsonl").exists()
    assert [(row["stage"], row["outcome"]) for row in result.report.exact_fallback["gdstk_boolean_diagnostics"]] == [
        ("full-oracle", "failed"), ("windowed-oracle", "failed"),
    ]


def test_policy_warning_never_publishes_classified_evidence(tmp_path, monkeypatch, capfd):
    inputs = _write_pair(tmp_path / "inputs", SIMPLE)
    output = tmp_path / "failure"
    boolean = gdstk.boolean
    monkeypatch.setattr(gdstk, "boolean", lambda a, b, op, **kwargs: _failure() if op == "and" else boolean(a, b, op, **kwargs))
    assert main([
        *map(str, inputs), "--backend", "cpu", "--allow-region", "7:3,removal,0,0,10,10",
        "--output-dir", str(output),
    ]) == 2
    assert "policy-classification" in capfd.readouterr().err
    report = _read_report(output)
    assert not report["exact_verification_complete"]
    assert report["exact_fallback"]["gdstk_boolean_diagnostics"][0]["stage"] == "policy-classification"
    assert not (output / "diff_vertices.jsonl").exists()


def test_native_only_failure_and_unrelated_stderr_are_handled_separately(monkeypatch, capfdbinary):
    other = b"other native stderr\n[GDSTK] unfamiliar diagnostic \xff\n"

    def native_only(*args, **kwargs):
        os.write(2, other + NATIVE_MESSAGE + b"\r\n")
        return []

    monkeypatch.setattr(gdstk, "boolean", native_only)
    inheritable = os.get_inheritable(2)
    with _collect_boolean_diagnostics() as diagnostics:
        with pytest.raises(_BooleanEvidenceError, match="Unable to link hole"):
            _checked_gdstk_boolean([], [], "or", precision=1.0)
    assert diagnostics.attach({})["gdstk_boolean_diagnostics"][0]["outcome"] == "failed"
    assert os.get_inheritable(2) == inheritable
    os.write(2, b"after Boolean\n")
    assert capfdbinary.readouterr().err == other + b"after Boolean\n"


def test_capture_setup_failure_restores_stderr_and_still_rejects_warning(monkeypatch, capfdbinary):
    duplicate = os.dup2
    failure_pending = True

    def fail_after_dup(source, dest, *, inheritable=True):
        nonlocal failure_pending
        result = duplicate(source, dest, inheritable=inheritable)
        if dest == 2 and failure_pending:
            failure_pending = False
            raise OSError("cannot set fd inheritance")
        return result

    monkeypatch.setattr(os, "dup2", fail_after_dup)
    monkeypatch.setattr(gdstk, "boolean", _failure)
    inheritable = os.get_inheritable(2)
    with _collect_boolean_diagnostics() as diagnostics:
        with pytest.raises(_BooleanEvidenceError, match="Error in boolean operation"):
            _checked_gdstk_boolean([], [], "or", precision=1.0)
    assert diagnostics.attach({})["gdstk_boolean_diagnostics"][0]["outcome"] == "failed"
    assert os.get_inheritable(2) == inheritable
    assert capfdbinary.readouterr().err == NATIVE_MESSAGE + b"\n"


def test_threaded_comparison_leaves_native_stderr_alone(monkeypatch, capfd):
    monkeypatch.setattr(gdstk, "boolean", _failure)

    def run():
        with _collect_boolean_diagnostics() as diagnostics:
            with pytest.raises(_BooleanEvidenceError):
                _checked_gdstk_boolean([], [], "not", precision=1.0)
        return diagnostics.attach({})["gdstk_boolean_diagnostics"][0]

    with ThreadPoolExecutor(1) as executor:
        assert executor.submit(run).result()["message"] == "Error in boolean operation."
    assert NATIVE_MESSAGE.decode() in capfd.readouterr().err


def test_windows_console_leaves_native_stderr_alone(monkeypatch, capfd):
    monkeypatch.setattr(gdstk, "boolean", _failure)
    with monkeypatch.context() as patch:
        patch.setattr(sys, "platform", "win32")
        patch.setattr(os, "isatty", lambda fd: fd == 2)
        with _collect_boolean_diagnostics() as diagnostics:
            with pytest.raises(_BooleanEvidenceError):
                _checked_gdstk_boolean([], [], "not", precision=1.0)
    assert diagnostics.attach({})["gdstk_boolean_diagnostics"][0]["message"] == "Error in boolean operation."
    assert NATIVE_MESSAGE.decode() in capfd.readouterr().err


def test_exact_worker_failure_diagnostics_reach_parent(tmp_path, monkeypatch, capfd):
    if fork_context() is None:
        pytest.skip("forked geometry workers are unavailable")
    other = LayerSpec(8, 0)
    inputs = _write_pair(tmp_path / "inputs", UNRECOVERABLE, layers=(LAYER, other))
    monkeypatch.setattr(exact_oracle, "_resolve_worker_count", lambda *_: 2)

    result = compare_gds_files(CompareConfig(*inputs, backend="cpu"), exact=True)
    assert result.exit_code == 2
    diagnostics = result.report.exact_fallback["gdstk_boolean_diagnostics"]
    assert {(row["layer"], row["datatype"], row["outcome"]) for row in diagnostics} == {
        (LAYER.layer, LAYER.datatype, "failed"), (other.layer, other.datatype, "failed"),
    }
    assert capfd.readouterr().err == ""


def test_legacy_windowed_pair_verdict_is_not_reused(tmp_path, monkeypatch):
    inputs = _write_pair(tmp_path / "inputs", SIMPLE)
    config = CompareConfig(*inputs, backend="cpu", geometry_cache_dir=tmp_path / "cache")
    monkeypatch.setattr(api, "_should_use_layer_exact", lambda *args: False)
    compare_gds_files(config, use_fast_shortcuts=False)
    cache_args = {"old_top_name": None, "new_top_name": None}
    cached = load_cached_pair_verdict(config, **cache_args)
    assert cached.exact_fallback["mode"] == "windowed"
    path = config.geometry_cache_dir / "pair_verdicts" / f"{cached.cache_key}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["checked_windowed_booleans"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_cached_pair_verdict(config, **cache_args) is None
    payload["exact_fallback"]["mode"] = "candidate-layer-exact"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_cached_pair_verdict(config, **cache_args) is not None
