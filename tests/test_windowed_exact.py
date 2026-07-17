from __future__ import annotations

import numpy as np

from gdsdiff import windowed_exact
from gdsdiff import (
    CandidateWindow,
    LayerSpec,
    SpatialIndex,
    format_twice_area,
    run_full_exact_oracle,
    run_windowed_exact_oracle,
)
from gdsdiff.geometry import LayerTable, PolygonBuffer


def _buffer(polygons, layer=LayerSpec(1, 0)):
    vertices = []
    offsets = [0]
    bboxes = []
    for polygon in polygons:
        vertices.extend(polygon)
        offsets.append(len(vertices))
        bboxes.append([min(x for x, _ in polygon), min(y for _, y in polygon), max(x for x, _ in polygon), max(y for _, y in polygon)])
    return PolygonBuffer(
        vertices_xy=np.asarray(vertices, dtype=np.int64).reshape((-1, 2)),
        polygon_offsets=np.asarray(offsets, dtype=np.uint64),
        layer_ids=np.zeros(len(polygons), dtype=np.uint32),
        bboxes=np.asarray(bboxes, dtype=np.int64).reshape((len(polygons), 4)),
        flags=np.zeros(len(polygons), dtype=np.uint32),
        source_ids=np.arange(len(polygons), dtype=np.uint64),
        layer_table=LayerTable((layer,)),
    )


def _rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _window(layer, core, verification=None, old_count=0, new_count=0, reason="test"):
    return CandidateWindow(
        layer=layer,
        core_bbox=core,
        verification_bbox=verification or core,
        source_candidate_count=1,
        estimated_old_polygons=old_count,
        estimated_new_polygons=new_count,
        reason=reason,
    )


def _areas(result):
    return {
        layer.layer: (
            format_twice_area(layer.old_minus_new_twice_area),
            format_twice_area(layer.new_minus_old_twice_area),
            format_twice_area(layer.xor_twice_area),
        )
        for layer in result.layers
    }


def test_windowed_result_matches_full_oracle_for_boundary_shift():
    layer = LayerSpec(1, 0)
    old = _buffer([_rect(0, 0, 10, 10)], layer)
    new = _buffer([_rect(2, 0, 12, 10)], layer)
    windows = (
        _window(layer, (0, 0, 1, 10), (-2, -2, 3, 12)),
        _window(layer, (10, 0, 12, 10), (9, -2, 13, 12)),
    )

    full = run_full_exact_oracle(old, new)
    windowed = run_windowed_exact_oracle(old, new, windows)

    assert _areas(windowed) == _areas(full)
    assert windowed.escalated_layers == ()
    assert windowed.layers[0].old_minus_new[0].bbox == (0, 0, 2, 10)
    assert windowed.layers[0].new_minus_old[0].bbox == (10, 0, 12, 10)


def test_verification_halo_includes_masking_geometry_from_outside_core():
    layer = LayerSpec(1, 0)
    old = _buffer([_rect(0, 0, 10, 10), _rect(20, 0, 30, 10)], layer)
    new = _buffer([_rect(0, 0, 10, 10), _rect(22, 0, 32, 10)], layer)
    windows = (
        _window(layer, (20, 0, 21, 10), (0, -1, 31, 11)),
        _window(layer, (30, 0, 32, 10), (19, -1, 32, 11)),
    )

    full = run_full_exact_oracle(old, new)
    windowed = run_windowed_exact_oracle(old, new, windows)

    assert _areas(windowed) == _areas(full)
    assert windowed.layers[0].old_minus_new[0].bbox == (20, 0, 22, 10)
    assert windowed.layers[0].new_minus_old[0].bbox == (30, 0, 32, 10)


def test_unchanged_polygon_masks_changed_polygon_inside_window():
    layer = LayerSpec(1, 0)
    old = _buffer([_rect(0, 0, 10, 10), _rect(0, 0, 10, 10)], layer)
    new = _buffer([_rect(0, 0, 10, 10), _rect(1, 0, 11, 10)], layer)
    windows = (_window(layer, (10, 0, 11, 10), (-1, -1, 11, 11)),)

    full = run_full_exact_oracle(old, new)
    windowed = run_windowed_exact_oracle(old, new, windows)

    assert _areas(windowed) == _areas(full)
    assert format_twice_area(windowed.layers[0].old_minus_new_twice_area) == "0"
    assert windowed.layers[0].new_minus_old[0].bbox == (10, 0, 11, 10)


def test_different_fracture_remains_equivalent():
    layer = LayerSpec(1, 0)
    old = _buffer([_rect(0, 0, 10, 10)], layer)
    new = _buffer([_rect(0, 0, 5, 10), _rect(5, 0, 10, 10)], layer)
    windows = (_window(layer, (0, 0, 10, 10), (-1, -1, 11, 11)),)

    windowed = run_windowed_exact_oracle(old, new, windows)

    assert windowed.equivalent
    assert _areas(windowed) == {layer: ("0", "0", "0")}


def test_many_windows_are_merged_without_double_counting():
    layer = LayerSpec(1, 0)
    old = _buffer([_rect(0, 0, 40, 10)], layer)
    new = _buffer([_rect(1, 0, 41, 10)], layer)
    windows = (
        _window(layer, (0, 0, 1, 10), (-1, -1, 2, 11)),
        _window(layer, (0, 0, 1, 10), (-1, -1, 2, 11), reason="duplicate"),
        _window(layer, (40, 0, 41, 10), (39, -1, 41, 11)),
    )

    full = run_full_exact_oracle(old, new)
    windowed = run_windowed_exact_oracle(old, new, windows)

    assert _areas(windowed) == _areas(full)
    assert format_twice_area(windowed.layers[0].xor_twice_area) == "20"


def test_forced_small_window_splitting_matches_full_oracle():
    layer = LayerSpec(1, 0)
    old = _buffer([_rect(0, 0, 100, 10)], layer)
    new = _buffer([_rect(4, 0, 104, 10)], layer)
    windows = (
        _window(layer, (0, 0, 1, 10), (-1, -1, 5, 11)),
        _window(layer, (2, 0, 4, 10), (-1, -1, 6, 11)),
        _window(layer, (100, 0, 101, 10), (98, -1, 104, 11)),
        _window(layer, (102, 0, 104, 10), (99, -1, 104, 11)),
    )

    full = run_full_exact_oracle(old, new)
    windowed = run_windowed_exact_oracle(old, new, windows)

    assert _areas(windowed) == _areas(full)


def test_threshold_escalates_to_full_layer_diff():
    layer = LayerSpec(1, 0)
    old = _buffer([_rect(0, 0, 10, 10), _rect(20, 0, 30, 10)], layer)
    new = _buffer([_rect(0, 0, 10, 10), _rect(22, 0, 32, 10)], layer)
    old_index = SpatialIndex.from_buffer(old)
    new_index = SpatialIndex.from_buffer(new)
    windows = (_window(layer, (20, 0, 32, 10), (-1, -1, 32, 11)),)

    full = run_full_exact_oracle(old, new)
    windowed = run_windowed_exact_oracle(
        old,
        new,
        windows,
        old_index=old_index,
        new_index=new_index,
        max_polygons_per_window=1,
    )

    assert windowed.escalated_layers == (layer,)
    assert len(windowed.failures) == 1
    assert _areas(windowed) == _areas(full)


def test_gdstk_failure_escalates_to_full_layer_diff(monkeypatch):
    layer = LayerSpec(1, 0)
    old = _buffer([_rect(0, 0, 10, 10)], layer)
    new = _buffer([_rect(1, 0, 11, 10)], layer)
    windows = (_window(layer, (0, 0, 11, 10), (-1, -1, 11, 11)),)

    def fail_window(*args, **kwargs):
        raise RuntimeError("simulated")

    monkeypatch.setattr(windowed_exact, "_compute_window_directional_fragments", fail_window)

    full = run_full_exact_oracle(old, new)
    windowed = run_windowed_exact_oracle(old, new, windows)

    assert windowed.escalated_layers == (layer,)
    assert len(windowed.failures) == 1
    assert "gdstk boolean failed" in windowed.failures[0].reason
    assert _areas(windowed) == _areas(full)
