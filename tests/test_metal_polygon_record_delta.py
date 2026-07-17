from __future__ import annotations

import os
import gdstk
import numpy as np
import pytest

from gdsdiff.backends.metal_ctypes import (
    MetalCtypesUnavailable,
    compare_files_metal,
    metal_ctypes_available,
    run_metal_context_overlap_pairs,
    run_metal_boundary_loops_from_edges,
    run_metal_ctypes_tile_prefilter,
    run_metal_mixed_boundary_edges,
    run_metal_one_sided_boundary_edges,
    run_metal_polygon_record_delta_from_hashes,
)
from gdsdiff.backends.polygon_extract_ctypes import extract_gds_polygons_native
from gdsdiff.config import LayerSpec
from gdsdiff.exact_oracle import DiffFragment
from gdsdiff.geometry import LayerTable, PolygonBuffer, polygon_bbox, polygon_twice_area
from gdsdiff.source_edge_diff import (
    _SegmentRecord,
    _boundary_edges_for_direction,
    _boundary_edges_for_one_sided_union,
    _build_segments,
    _fragments_from_boundary_edges,
    _maybe_cuda_boundary_edges_for_direction,
    _maybe_cuda_boundary_edges_for_one_sided_union,
    _point_at,
    _point_key,
    _segment_splits,
    _signed_twice_area,
)
from gdsdiff import source_edge_diff as source_edges
from gdsdiff.tiling import TileConfig, run_cpu_tile_prefilter

requires_posix_native = pytest.mark.skipif(os.name == "nt", reason="POSIX on-demand native helpers are unavailable on Windows")


def _rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _diff_fragment(points):
    return DiffFragment(points=points, bbox=polygon_bbox(points), twice_area=polygon_twice_area(points))


def _buffer(polygons, layer=LayerSpec(1, 0)):
    vertices = []
    offsets = [0]
    for polygon in polygons:
        vertices.extend(polygon)
        offsets.append(len(vertices))
    bboxes = [
        [min(x for x, _ in polygon), min(y for _, y in polygon), max(x for x, _ in polygon), max(y for _, y in polygon)]
        for polygon in polygons
    ]
    return PolygonBuffer(
        vertices_xy=np.asarray(vertices, dtype=np.int64).reshape((-1, 2)),
        polygon_offsets=np.asarray(offsets, dtype=np.uint64),
        layer_ids=np.zeros(len(polygons), dtype=np.uint32),
        bboxes=np.asarray(bboxes, dtype=np.int64).reshape((-1, 4)),
        flags=np.zeros(len(polygons), dtype=np.uint32),
        source_ids=np.arange(len(polygons), dtype=np.uint64),
        layer_table=LayerTable((layer,)),
    )


def _write_changed_rect_pair(tmp_path):
    old_path = tmp_path / "old.gds"
    new_path = tmp_path / "new.gds"
    old_lib = gdstk.Library(unit=1e-6, precision=1e-9)
    old_top = old_lib.new_cell("top")
    old_top.add(gdstk.rectangle((0, 0), (1, 1), layer=1, datatype=0))
    old_top.add(gdstk.rectangle((2, 0), (3, 1), layer=1, datatype=0))
    old_lib.write_gds(old_path)

    new_lib = gdstk.Library(unit=1e-6, precision=1e-9)
    new_top = new_lib.new_cell("top")
    new_top.add(gdstk.rectangle((0, 0), (1, 1), layer=1, datatype=0))
    new_top.add(gdstk.rectangle((4, 0), (5, 1), layer=1, datatype=0))
    new_lib.write_gds(new_path)
    return old_path, new_path


def test_metal_byte_compare_reports_size_mismatch_before_runtime_probe(tmp_path):
    old_path = tmp_path / "old.bin"
    new_path = tmp_path / "new.bin"
    old_path.write_bytes(b"abc")
    new_path.write_bytes(b"abcd")

    result = compare_files_metal(old_path, new_path)

    assert result.method == "metal-byte-compare"
    assert result.old_size == 3
    assert result.new_size == 4
    assert result.same is False


def test_metal_byte_compare_matches_equal_size_files_when_available(tmp_path):
    if not metal_ctypes_available():
        pytest.skip("Metal ctypes backend is not available")
    old_path = tmp_path / "old.bin"
    new_path = tmp_path / "new.bin"
    old_path.write_bytes(b"abcdef")
    new_path.write_bytes(b"abcxef")

    result = compare_files_metal(old_path, new_path)

    assert result.old_size == 6
    assert result.new_size == 6
    assert result.same is False


@requires_posix_native
def test_native_polygon_extraction_preserves_polygon_hashes(tmp_path):
    old_path, _new_path = _write_changed_rect_pair(tmp_path)

    extraction = extract_gds_polygons_native(old_path)

    assert extraction.buffer.polygon_hashes is not None
    assert extraction.buffer.polygon_hashes.shape == (2, 2)


@requires_posix_native
def test_metal_polygon_record_delta_size_gate_is_checked_before_runtime_probe(tmp_path, monkeypatch):
    old_path, new_path = _write_changed_rect_pair(tmp_path)
    old_extraction = extract_gds_polygons_native(old_path)
    new_extraction = extract_gds_polygons_native(new_path)
    monkeypatch.setenv("GDS_METAL_RECORD_DELTA_MAX_POLYGONS", "1")

    with pytest.raises(MetalCtypesUnavailable, match="limited to 1 total polygons"):
        run_metal_polygon_record_delta_from_hashes(old_extraction.buffer, new_extraction.buffer)


def test_metal_polygon_record_delta_matches_cuda_record_contract_when_available(tmp_path):
    if not metal_ctypes_available():
        pytest.skip("Metal ctypes backend is not available")
    old_path, new_path = _write_changed_rect_pair(tmp_path)
    old_extraction = extract_gds_polygons_native(old_path)
    new_extraction = extract_gds_polygons_native(new_path)

    delta = run_metal_polygon_record_delta_from_hashes(old_extraction.buffer, new_extraction.buffer)

    assert delta.metal_kernel_count == 2
    assert delta.matched_record_count == 1
    assert delta.record_count == 2
    rows = {
        (
            int(delta.directions[index]),
            int(delta.layers[index]),
            int(delta.datatypes[index]),
            tuple(int(value) for value in delta.bboxes[index]),
            int(delta.counts[index]),
        )
        for index in range(delta.record_count)
    }
    assert rows == {
        (1, 1, 0, (2000, 0, 3000, 1000), 1),
        (2, 1, 0, (4000, 0, 5000, 1000), 1),
    }


def test_metal_tile_prefilter_rejects_canonical_id_mismatch_before_runtime_probe():
    old = _buffer([_rect(0, 0, 9, 9)])
    new = _buffer([])

    with pytest.raises(ValueError, match="canonical id vector"):
        run_metal_ctypes_tile_prefilter(old, [], new, [], TileConfig(tile_size=10))


def test_metal_tile_prefilter_fails_closed_when_runtime_unavailable():
    if metal_ctypes_available():
        pytest.skip("Metal ctypes backend is available")
    old = _buffer([_rect(0, 0, 9, 9)])
    new = _buffer([])

    with pytest.raises(MetalCtypesUnavailable, match="Metal"):
        run_metal_ctypes_tile_prefilter(old, [1], new, [], TileConfig(tile_size=10))


def test_metal_tile_prefilter_matches_cpu_oracle_when_available():
    if not metal_ctypes_available():
        pytest.skip("Metal ctypes backend is not available")
    old = _buffer([_rect(0, 0, 9, 9), _rect(0, 0, 100, 100), _rect(-11, -11, -1, -1)])
    new = _buffer([_rect(20, 0, 29, 9), _rect(0, 0, 100, 100)])
    config = TileConfig(tile_size=10, max_tiles_per_polygon=4)

    try:
        metal = run_metal_ctypes_tile_prefilter(old, [1, 9, 3], new, [2, 9], config)
    except MetalCtypesUnavailable as exc:
        pytest.skip(f"Metal tile prefilter sub-kernel is unavailable: {exc}")
    cpu = run_cpu_tile_prefilter(old, [1, 9, 3], new, [2, 9], config)

    assert metal == cpu


def test_source_edge_boundary_routing_uses_cpu_reference_after_cuda_for_one_sided(monkeypatch):
    segments = tuple(_SegmentRecord("target", 0, (float(index), 0.0), (float(index + 1), 0.0)) for index in range(256))
    splits = tuple((0.0, 1.0) for _segment in segments)
    polygons = tuple(
        (
            (index * 10, 0),
            (index * 10 + 1, 0),
            (index * 10 + 1, 1),
            (index * 10, 1),
        )
        for index in range(4)
    )
    bboxes = tuple((index * 10, 0, index * 10 + 1, 1) for index in range(4))
    expected_edges = _boundary_edges_for_one_sided_union(segments, splits, polygons, bboxes)
    monkeypatch.setattr(source_edges, "_cuda_boundary_edges_for_one_sided_union", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no cuda")))
    monkeypatch.setattr(source_edges, "_metal_boundary_edges_for_one_sided_union", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("metal boundary sampler used")))

    edges, used_gpu = _maybe_cuda_boundary_edges_for_one_sided_union(segments, splits, polygons, bboxes)

    assert used_gpu is False
    assert edges == expected_edges


def test_source_edge_boundary_routing_uses_cpu_reference_after_cuda_for_mixed(monkeypatch):
    segments = tuple(_SegmentRecord("old", 0, (float(index), 0.0), (float(index + 1), 0.0)) for index in range(128))
    splits = tuple((0.0, 1.0) for _segment in segments)
    polygons_by_side = {
        "old": (((0, 0), (1, 0), (1, 1), (0, 1)),),
        "new": (((2, 0), (3, 0), (3, 1), (2, 1)),),
    }
    bboxes_by_side = {
        "old": ((0, 0, 1, 1),),
        "new": ((2, 0, 3, 1),),
    }
    expected_edges = _boundary_edges_for_direction(segments, splits, polygons_by_side, bboxes_by_side, direction="old")
    monkeypatch.setattr(source_edges, "_cuda_boundary_edges_for_direction", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no cuda")))
    monkeypatch.setattr(source_edges, "_metal_boundary_edges_for_direction", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("metal boundary sampler used")))

    edges, used_gpu = _maybe_cuda_boundary_edges_for_direction(
        segments,
        splits,
        polygons_by_side,
        bboxes_by_side,
        direction="old",
    )

    assert used_gpu is False
    assert edges == expected_edges


def test_metal_context_overlap_size_gate_is_checked_before_runtime_probe(monkeypatch):
    monkeypatch.setenv("GDS_METAL_CONTEXT_OVERLAP_MAX_PAIRS", "3")
    component_bboxes = np.asarray([[0, 0, 10, 10], [20, 0, 30, 10]], dtype=np.int64)
    component_layers = np.asarray([1, 1], dtype=np.int32)
    component_datatypes = np.asarray([0, 0], dtype=np.int32)
    polygon_bboxes = np.asarray([[5, 0, 15, 10], [25, 0, 35, 10]], dtype=np.int64)
    polygon_layers = np.asarray([1, 1], dtype=np.int32)
    polygon_datatypes = np.asarray([0, 0], dtype=np.int32)

    with pytest.raises(MetalCtypesUnavailable, match="limited to 3 component/polygon pairs"):
        run_metal_context_overlap_pairs(
            component_bboxes,
            component_layers,
            component_datatypes,
            polygon_bboxes,
            polygon_layers,
            polygon_datatypes,
        )


def test_metal_context_overlap_pairs_match_cuda_contract_when_available():
    if not metal_ctypes_available():
        pytest.skip("Metal ctypes backend is not available")
    component_bboxes = np.asarray([[0, 0, 10, 10], [20, 0, 30, 10]], dtype=np.int64)
    component_layers = np.asarray([1, 2], dtype=np.int32)
    component_datatypes = np.asarray([0, 0], dtype=np.int32)
    polygon_bboxes = np.asarray(
        [
            [5, 0, 15, 10],
            [0, 0, 8, 8],
            [25, 0, 35, 10],
            [20, 20, 30, 30],
        ],
        dtype=np.int64,
    )
    polygon_layers = np.asarray([1, 2, 2, 2], dtype=np.int32)
    polygon_datatypes = np.asarray([0, 0, 0, 0], dtype=np.int32)

    pairs = run_metal_context_overlap_pairs(
        component_bboxes,
        component_layers,
        component_datatypes,
        polygon_bboxes,
        polygon_layers,
        polygon_datatypes,
    )

    assert pairs.metal_kernel_count == 1
    assert pairs.pair_count == 2
    assert list(zip(pairs.component_indices.tolist(), pairs.polygon_indices.tolist())) == [(0, 0), (1, 2)]


def test_metal_boundary_loops_validate_shape_before_runtime_probe():
    with pytest.raises(ValueError, match="edges"):
        run_metal_boundary_loops_from_edges(np.asarray([0, 0, 1, 0], dtype=np.int64))


def test_metal_boundary_loops_fail_closed_when_runtime_unavailable():
    if metal_ctypes_available():
        pytest.skip("Metal ctypes backend is available")
    edges = np.asarray(
        [
            (0, 0, 4, 0),
            (4, 0, 4, 2),
            (4, 2, 0, 2),
            (0, 2, 0, 0),
        ],
        dtype=np.int64,
    )

    with pytest.raises(MetalCtypesUnavailable, match="Metal"):
        run_metal_boundary_loops_from_edges(edges)


def test_metal_boundary_loops_trace_integer_edges_when_available():
    if not metal_ctypes_available():
        pytest.skip("Metal ctypes backend is not available")
    edges = np.asarray(
        [
            (0, 0, 4, 0),
            (4, 0, 4, 2),
            (4, 2, 0, 2),
            (0, 2, 0, 0),
            (10, 0, 12, 0),
            (12, 0, 12, 2),
            (12, 2, 10, 2),
            (10, 2, 10, 0),
        ],
        dtype=np.int64,
    )

    result = run_metal_boundary_loops_from_edges(edges)

    assert result.dropped_zero_area_loop_count == 0
    assert result.metal_kernel_count == 2
    assert [fragment.bbox for fragment in result.fragments] == [(0, 0, 4, 2), (10, 0, 12, 2)]
    assert [fragment.twice_area for fragment in result.fragments] == [16, 8]


def test_source_edge_loop_routing_tries_metal_after_cuda(monkeypatch):
    expected = _diff_fragment(((0, 0), (1, 0), (1, 1), (0, 1)))

    class FakeMetalResult:
        fragments = (expected,)
        dropped_zero_area_loop_count = 0

    monkeypatch.setattr(source_edges, "_try_cuda_fragments_from_integer_boundary_edges", lambda _edges: None)
    monkeypatch.setattr(source_edges, "_try_metal_fragments_from_integer_boundary_edges", lambda _edges: FakeMetalResult())

    fragments, dropped, backend = _fragments_from_boundary_edges(
        (
            ((0.0, 0.0), (1.0, 0.0)),
            ((1.0, 0.0), (1.0, 1.0)),
            ((1.0, 1.0), (0.0, 1.0)),
            ((0.0, 1.0), (0.0, 0.0)),
        )
    )

    assert backend == "metal"
    assert dropped == 0
    assert fragments == [expected]


def test_metal_one_sided_boundary_edges_validate_shape_before_runtime_probe():
    with pytest.raises(ValueError, match="vertices_xy"):
        run_metal_one_sided_boundary_edges(
            np.asarray([0, 0], dtype=np.int64),
            np.asarray([0], dtype=np.uint64),
            np.empty((0, 4), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0, 4), dtype=np.float64),
            np.empty((0,), dtype=np.uint32),
        )


def test_metal_one_sided_boundary_edges_fail_closed_when_runtime_unavailable():
    if metal_ctypes_available():
        pytest.skip("Metal ctypes backend is available")
    polygons = (((0, 0), (1000, 0), (1000, 1000), (0, 1000)),)
    segments = _build_segments({"target": polygons})

    with pytest.raises(MetalCtypesUnavailable, match="Metal"):
        run_metal_one_sided_boundary_edges(
            np.asarray([point for polygon in polygons for point in polygon], dtype=np.int64),
            np.asarray([0, 4], dtype=np.uint64),
            np.asarray(((0, 0, 1000, 1000),), dtype=np.int64),
            np.asarray([_signed_twice_area(polygons[0])], dtype=np.int64),
            np.asarray([(segment.start[0], segment.start[1], segment.stop[0], segment.stop[1]) for segment in segments], dtype=np.float64),
            np.asarray([segment.polygon_index for segment in segments], dtype=np.uint32),
        )


def test_metal_one_sided_boundary_edges_match_python_reference_when_available():
    if not metal_ctypes_available():
        pytest.skip("Metal ctypes backend is not available")
    polygons = (
        ((0, 0), (4000, 0), (4000, 4000), (0, 4000)),
        ((2000, 0), (6000, 0), (6000, 4000), (2000, 4000)),
    )
    bboxes = ((0, 0, 4000, 4000), (2000, 0, 6000, 4000))
    segments = _build_segments({"target": polygons})
    splits = _segment_splits(segments)
    expected = _boundary_edges_for_one_sided_union(segments, splits, polygons, bboxes)
    piece_rows = []
    piece_polygon_indices = []
    for segment_index, segment in enumerate(segments):
        for left_t, right_t in zip(splits[segment_index], splits[segment_index][1:]):
            if right_t - left_t <= 1e-10:
                continue
            start = _point_at(segment, left_t)
            stop = _point_at(segment, right_t)
            piece_rows.append((start[0], start[1], stop[0], stop[1]))
            piece_polygon_indices.append(segment.polygon_index)

    metal_edges = run_metal_one_sided_boundary_edges(
        np.asarray([point for polygon in polygons for point in polygon], dtype=np.int64),
        np.asarray([0, 4, 8], dtype=np.uint64),
        np.asarray(bboxes, dtype=np.int64),
        np.asarray([_signed_twice_area(polygon) for polygon in polygons], dtype=np.int64),
        np.asarray(piece_rows, dtype=np.float64),
        np.asarray(piece_polygon_indices, dtype=np.uint32),
    )
    actual = tuple(sorted(((_point_key((edge[0][0], edge[0][1])), _point_key((edge[1][0], edge[1][1]))) for edge in metal_edges)))

    assert actual == expected


def test_metal_mixed_boundary_edges_match_python_reference_when_available():
    if not metal_ctypes_available():
        pytest.skip("Metal ctypes backend is not available")
    polygons_by_side = {
        "old": (((0, 0), (4000, 0), (4000, 4000), (0, 4000)),),
        "new": (((2000, 0), (6000, 0), (6000, 4000), (2000, 4000)),),
    }
    bboxes_by_side = {
        "old": ((0, 0, 4000, 4000),),
        "new": ((2000, 0, 6000, 4000),),
    }
    segments = _build_segments(polygons_by_side)
    splits = _segment_splits(segments)
    piece_rows = []
    for segment_index, segment in enumerate(segments):
        for left_t, right_t in zip(splits[segment_index], splits[segment_index][1:]):
            if right_t - left_t <= 1e-10:
                continue
            start = _point_at(segment, left_t)
            stop = _point_at(segment, right_t)
            piece_rows.append((start[0], start[1], stop[0], stop[1]))

    vertices = np.asarray([point for points in polygons_by_side["old"] + polygons_by_side["new"] for point in points], dtype=np.int64)
    offsets = np.asarray([0, 4, 8], dtype=np.uint64)
    bboxes = np.asarray(bboxes_by_side["old"] + bboxes_by_side["new"], dtype=np.int64)
    sides = np.asarray([1, 2], dtype=np.uint8)

    for direction in ("old", "new"):
        expected = _boundary_edges_for_direction(segments, splits, polygons_by_side, bboxes_by_side, direction=direction)
        metal_edges = run_metal_mixed_boundary_edges(
            vertices,
            offsets,
            bboxes,
            sides,
            np.asarray(piece_rows, dtype=np.float64),
            direction=direction,
        )
        actual = tuple(sorted(((_point_key((edge[0][0], edge[0][1])), _point_key((edge[1][0], edge[1][1]))) for edge in metal_edges)))

        assert actual == expected
