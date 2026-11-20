from __future__ import annotations

import warnings

import gdstk
import numpy as np
import pytest

from gdsdiff import LayerSpec, run_full_exact_oracle, sparse_coverage
from gdsdiff.cli import main
from gdsdiff.exact_diff_markdown import build_polygon_inventory_delta
from gdsdiff.exact_oracle import DiffFragment, _BooleanEvidenceError
from gdsdiff.geometry import LayerTable, PolygonBuffer, polygon_bbox, polygon_flags, polygon_twice_area
from gdsdiff.sparse_coverage import _SparseCoverageDeclined, _fragment_vertices, _run_verified_sparse_exact_oracle


SHAPES = LayerSpec(1, 0)
DENSE = LayerSpec(1, 1)
OTHER = LayerSpec(5, 3)


def _rect(x, y, u, v):
    return ((x, y), (u, y), (u, v), (x, v))


def _buffer(records):
    layers = LayerTable(tuple(dict.fromkeys(layer for layer, _ in records)))
    layer_ids = layers.index_by_layer
    return PolygonBuffer(
        vertices_xy=np.array([point for _, points in records for point in points], dtype=np.int64).reshape((-1, 2)),
        polygon_offsets=np.r_[0, np.cumsum([len(points) for _, points in records])].astype(np.uint64),
        layer_ids=np.array([layer_ids[layer] for layer, _ in records], dtype=np.uint32),
        bboxes=np.array([polygon_bbox(points) for _, points in records], dtype=np.int64).reshape((-1, 4)),
        flags=np.array([polygon_flags(points) for _, points in records], dtype=np.uint32),
        source_ids=np.arange(len(records), dtype=np.uint64),
        layer_table=layers,
    )


def _directions(result):
    return {
        d.layer: (_fragment_vertices(d.old_minus_new), _fragment_vertices(d.new_minus_old))
        for d in result.layers
    }


def test_unaligned_correction_preserves_nested_coverage_context_and_inventory():
    outer = _rect(0, 0, 30, 30)
    hole = ((8, 8), (15, 8), (8, 15))
    mask = _rect(20, -2, 32, 32)
    remote = _rect(42, -4, 51, -2)
    added = ((40, 0), (45, 0), (40, 5))
    a = ((300, 100), (310, 100), (300, 110))
    b = ((300, 100), (310, 100), (305, 110))
    dense = [(DENSE, _rect(i * 5, 100, i * 5 + 3, 104)) for i in range(2048)]
    before = [(SHAPES, outer), (SHAPES, mask), (SHAPES, a), (SHAPES, b)] + dense
    after = [(OTHER, remote), (SHAPES, b[1:] + b[:1]), (SHAPES, a[::-1]), (SHAPES, mask),
             (SHAPES, hole), (SHAPES, added)] + list(reversed(dense))
    old, new = _buffer(before), _buffer(after)
    expected = run_full_exact_oracle(old, new, workers=1)
    result = _run_verified_sparse_exact_oracle(
        old, new, candidate_bounds={SHAPES: {polygon_bbox(p) for p in (outer, hole, added)}, OTHER: {polygon_bbox(remote)}},
    )
    independent = _run_verified_sparse_exact_oracle(old, new)

    assert _directions(result.exact_result) == _directions(expected)
    assert result.exact_result == independent.exact_result
    assert result.inventory == independent.inventory == build_polygon_inventory_delta(old, new)
    assert (result.old_retained, result.new_retained) == (2, 4)
    assert result.exact_result.layers[0].old_minus_new_twice_area == 1_151


def test_verified_sparse_retains_inventory_multiplicity_with_equal_coverage():
    polygon = ((0, 0), (4, 0), (0, 5))
    old = _buffer([(SHAPES, polygon)])
    new = _buffer([(SHAPES, polygon), (SHAPES, polygon[::-1])])
    result = _run_verified_sparse_exact_oracle(old, new, candidate_bounds={SHAPES: {polygon_bbox(polygon)}})
    assert result.exact_result.equivalent
    assert len(result.inventory.added) == 1
    assert not result.inventory.removed


def test_missing_gpu_candidate_is_detected_by_vertices_outside_candidate_bounds():
    local = _rect(0, 0, 10, 10)
    a = ((100, 0), (110, 0), (100, 10))
    b = ((100, 0), (110, 0), (102, 10))
    old = _buffer([(SHAPES, a)])
    new = _buffer([(SHAPES, local), (SHAPES, b)])
    with pytest.raises(_SparseCoverageDeclined, match="outside GPU candidate"):
        _run_verified_sparse_exact_oracle(old, new, candidate_bounds={SHAPES: {polygon_bbox(local)}})


def test_sparse_proof_rejects_unseeded_polygon_that_extends_outside_retained_context():
    local = _rect(0, 0, 10, 10)
    extends = _rect(10, 0, 30, 4)
    old = _buffer([])
    new = _buffer([(SHAPES, local), (SHAPES, extends)])
    with pytest.raises(_SparseCoverageDeclined, match="unmatched polygon"):
        _run_verified_sparse_exact_oracle(old, new, candidate_bounds={SHAPES: {polygon_bbox(local)}})


def test_closed_contact_includes_touching_flagged_polygon():
    touching = ((10, 1), (10, 2), (10, 3))
    added = _rect(0, 0, 10, 10)
    old = _buffer([(SHAPES, touching)])
    new = _buffer([(SHAPES, touching), (SHAPES, added)])
    with pytest.raises(_SparseCoverageDeclined, match="flagged polygons"):
        _run_verified_sparse_exact_oracle(old, new, candidate_bounds={SHAPES: {polygon_bbox(added)}})


def test_gdstk_warning_uses_disjoint_groups_before_reflection(monkeypatch):
    boolean = gdstk.boolean
    polygons = _rect(0, 0, 5, 5), _rect(20, 0, 25, 5)

    def fail_whole_layer(a, b, op, **kwargs):
        if len(a) > 1:
            warnings.warn("Error in boolean operation.", RuntimeWarning)
        return boolean(a, b, op, **kwargs)

    monkeypatch.setattr(gdstk, "boolean", fail_whole_layer)
    fragments, partitioned, reflected = sparse_coverage._directional_difference(SHAPES, polygons, ())
    assert {fragment.bbox for fragment in fragments} == {polygon_bbox(poly) for poly in polygons}
    assert (partitioned, reflected) == (1, 0)


def test_gdstk_warning_retries_with_two_agreeing_integer_reflections(monkeypatch):
    boolean = gdstk.boolean
    calls = []

    def fail_original(a, b, op, **kwargs):
        calls.append(kwargs["precision"])
        if len(calls) == 1:
            warnings.warn("Error in boolean operation.", RuntimeWarning)
            return []
        return boolean(a, b, op, **kwargs)

    monkeypatch.setattr(gdstk, "boolean", fail_original)
    polygon = ((-4, -8), (6, -8), (-4, 2))
    fragments, partitioned, reflected = sparse_coverage._directional_difference(SHAPES, (polygon,), ())
    assert (partitioned, reflected) == (1, 1)
    assert calls == [1.0, 1.0, 1.0]
    assert _fragment_vertices(fragments) == _fragment_vertices((DiffFragment(polygon, polygon_bbox(polygon), 100),))


def test_reflection_retry_refuses_same_area_with_different_full_vertices(monkeypatch):
    polygon = _rect(-2, -4, 2, 4)

    def violate_vertex_parity(layer, subject, clip, *, reflect=(1, 1)):
        if reflect == (1, 1):
            raise _BooleanEvidenceError("gdstk not: Error in boolean operation.")
        points = polygon if reflect == (1, -1) else polygon[:1] + ((0, -4),) + polygon[1:]
        return (DiffFragment(points, polygon_bbox(points), polygon_twice_area(points)),)

    monkeypatch.setattr(sparse_coverage, "_difference", violate_vertex_parity)
    with pytest.raises(_BooleanEvidenceError, match="reflected vertex loops disagree"):
        sparse_coverage._directional_difference(SHAPES, (polygon,), ())


def test_cpu_cli_fails_closed_on_a_boolean_warning(tmp_path, monkeypatch, capsys):
    inputs = []
    for name, offset in (("old", 0), ("new", 1)):
        lib = gdstk.Library()
        lib.new_cell("TOP").add(gdstk.rectangle((offset, 0), (offset + 3, 3)))
        path = tmp_path / f"{name}.gds"
        lib.write_gds(path)
        inputs.append(str(path))

    def partial(*args, **kwargs):
        warnings.warn("Error in boolean operation.", RuntimeWarning)
        return []

    monkeypatch.setattr(gdstk, "boolean", partial)
    output = tmp_path / "diff"
    assert main([*inputs, "--backend", "cpu", "--evidence-engine", "gdstk-oracle", "--output-dir", str(output)]) == 2
    assert "Error in boolean operation" in capsys.readouterr().err
    assert not (output / "diff_vertices.jsonl").exists()
