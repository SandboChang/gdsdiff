from __future__ import annotations

import gdstk
import numpy as np
import pytest

from gds_tools_tk.gpu_compare.backends.cuda_ctypes import (
    CudaCtypesUnavailable,
    cuda_ctypes_available,
    run_cuda_axis_aligned_boundary_edges_from_fragments,
    run_cuda_boundary_loops_from_edges,
    run_cuda_context_overlap_pairs,
    run_cuda_mixed_boundary_edges,
    run_cuda_one_sided_boundary_edges,
    run_cuda_one_sided_boundary_edges_unsplit_if_safe,
    run_cuda_polygon_component_areas,
    run_cuda_polygon_component_fragments,
    run_cuda_polygon_record_delta,
    run_cuda_split_free_boundary_edges_from_fragments,
)
from gds_tools_tk.gpu_compare.backends.boundary_loop_ctypes import (
    boundary_loop_ctypes_available,
    native_boundary_edges_from_fragments,
    native_boundary_loops_from_edges,
)
from gds_tools_tk.gpu_compare.backends.polygon_extract_ctypes import extract_gds_polygons_native
from gds_tools_tk.gpu_compare.config import LayerSpec
from gds_tools_tk.gpu_compare.exact_oracle import DiffFragment
from gds_tools_tk.gpu_compare.geometry import polygon_bbox, polygon_twice_area
from gds_tools_tk.gpu_compare.polygon_components import (
    add_overlapping_context_to_components,
    build_record_delta_surplus_components,
    exact_result_from_component_quads,
)
from gds_tools_tk.gpu_compare.source_edge_diff import (
    _boundary_edges_for_direction,
    _boundary_edges_for_one_sided_union,
    _build_segments,
    _fragments_from_boundary_edges,
    _point_at,
    _point_key,
    _segment_splits,
    _signed_twice_area,
)


def test_native_boundary_loop_ctypes_traces_integer_edges():
    if not boundary_loop_ctypes_available():
        pytest.skip("native boundary loop backend is not available")

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

    result = native_boundary_loops_from_edges(edges)

    assert result.dropped_zero_area_loop_count == 0
    assert [fragment.bbox for fragment in result.fragments] == [(0, 0, 4, 2), (10, 0, 12, 2)]
    assert [fragment.twice_area for fragment in result.fragments] == [16, 8]


def test_native_boundary_edges_from_fragments_cancel_shared_collinear_edges():
    if not boundary_loop_ctypes_available():
        pytest.skip("native boundary loop backend is not available")

    fragments = (
        _diff_fragment(((0, 0), (2, 0), (2, 2), (0, 2))),
        _diff_fragment(((2, 0), (4, 0), (4, 2), (2, 2))),
        _diff_fragment(((4, 0), (6, 0), (6, 2), (4, 2))),
    )

    result = native_boundary_edges_from_fragments(fragments)

    assert result.edges.tolist() == [
        [0, 0, 2, 0],
        [0, 2, 0, 0],
        [2, 0, 4, 0],
        [2, 2, 0, 2],
        [4, 0, 6, 0],
        [4, 2, 2, 2],
        [6, 0, 6, 2],
        [6, 2, 4, 2],
    ]


def test_cuda_boundary_loop_traces_integer_edges():
    if not cuda_ctypes_available():
        pytest.skip("CUDA ctypes backend is not available")

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

    result = run_cuda_boundary_loops_from_edges(edges)

    assert result.dropped_zero_area_loop_count == 0
    assert result.cuda_kernel_count == 1
    assert [fragment.bbox for fragment in result.fragments] == [(0, 0, 4, 2), (10, 0, 12, 2)]
    assert [fragment.twice_area for fragment in result.fragments] == [16, 8]


def test_source_edge_diff_uses_cuda_loop_tracing_for_integer_edges():
    if not cuda_ctypes_available():
        pytest.skip("CUDA ctypes backend is not available")

    boundary_edges = []
    for index in range(20):
        x0 = index * 10
        x1 = x0 + 4
        y0 = 0
        y1 = 2
        boundary_edges.extend(
            [
                ((float(x0), float(y0)), (float(x1), float(y0))),
                ((float(x1), float(y0)), (float(x1), float(y1))),
                ((float(x1), float(y1)), (float(x0), float(y1))),
                ((float(x0), float(y1)), (float(x0), float(y0))),
            ]
        )

    fragments, dropped, loop_backend = _fragments_from_boundary_edges(tuple(boundary_edges))

    assert loop_backend == "cuda"
    assert dropped == 0
    assert len(fragments) == 20
    assert fragments[0].bbox == (0, 0, 4, 2)
    assert fragments[-1].bbox == (190, 0, 194, 2)


def test_cuda_context_overlap_pairs_match_bbox_layer_predicate():
    if not cuda_ctypes_available():
        pytest.skip("CUDA ctypes backend is not available")

    component_bboxes = np.asarray(
        [
            (0, 0, 10, 10),
            (20, 0, 30, 10),
            (0, 0, 10, 10),
        ],
        dtype=np.int64,
    )
    component_layers = np.asarray([1, 1, 2], dtype=np.int32)
    component_datatypes = np.asarray([0, 0, 0], dtype=np.int32)
    polygon_bboxes = np.asarray(
        [
            (5, 0, 15, 10),
            (10, 0, 20, 10),
            (25, 1, 29, 9),
            (5, 0, 15, 10),
        ],
        dtype=np.int64,
    )
    polygon_layers = np.asarray([1, 1, 1, 2], dtype=np.int32)
    polygon_datatypes = np.asarray([0, 0, 0, 0], dtype=np.int32)

    result = run_cuda_context_overlap_pairs(
        component_bboxes,
        component_layers,
        component_datatypes,
        polygon_bboxes,
        polygon_layers,
        polygon_datatypes,
    )

    assert result.cuda_kernel_count >= 2
    assert list(zip(result.component_indices.tolist(), result.polygon_indices.tolist())) == [(0, 0), (1, 2), (2, 3)]


def test_cuda_axis_aligned_boundary_edges_cancel_shared_edges():
    if not cuda_ctypes_available():
        pytest.skip("CUDA ctypes backend is not available")

    fragments = (
        _diff_fragment(((0, 0), (2, 0), (2, 2), (0, 2))),
        _diff_fragment(((2, 0), (4, 0), (4, 2), (2, 2))),
        _diff_fragment(((4, 0), (6, 0), (6, 2), (4, 2))),
    )

    result = run_cuda_axis_aligned_boundary_edges_from_fragments(fragments)

    assert result.cuda_kernel_count >= 4
    assert result.edges.tolist() == [
        [0, 0, 2, 0],
        [0, 2, 0, 0],
        [2, 0, 4, 0],
        [2, 2, 0, 2],
        [4, 0, 6, 0],
        [4, 2, 2, 2],
        [6, 0, 6, 2],
        [6, 2, 4, 2],
    ]


def test_cuda_split_free_boundary_edges_match_native_for_shared_sloped_edge():
    if not cuda_ctypes_available():
        pytest.skip("CUDA ctypes backend is not available")
    if not boundary_loop_ctypes_available():
        pytest.skip("native boundary loop backend is not available")

    fragments = (
        _diff_fragment(((0, 0), (4, 0), (2, 2))),
        _diff_fragment(((4, 0), (4, 4), (2, 2))),
    )

    cuda = run_cuda_split_free_boundary_edges_from_fragments(fragments)
    native = native_boundary_edges_from_fragments(fragments)

    assert cuda.cuda_kernel_count >= 3
    assert cuda.edges.tolist() == native.edges.tolist()


def test_cuda_split_free_boundary_edges_reject_partial_collinear_overlap():
    if not cuda_ctypes_available():
        pytest.skip("CUDA ctypes backend is not available")

    fragments = (
        _diff_fragment(((0, 0), (4, 4), (0, 4))),
        _diff_fragment(((2, 2), (6, 6), (2, 6))),
    )

    with pytest.raises(CudaCtypesUnavailable):
        run_cuda_split_free_boundary_edges_from_fragments(fragments)


def test_record_delta_components_group_by_layer_and_bbox_overlap(tmp_path):
    if not cuda_ctypes_available():
        pytest.skip("CUDA ctypes backend is not available")

    old_path = tmp_path / "old.gds"
    new_path = tmp_path / "new.gds"

    old_lib = gdstk.Library(unit=1e-6, precision=1e-9)
    old_top = old_lib.new_cell("top")
    old_top.add(gdstk.rectangle((0, 0), (4, 4), layer=1, datatype=0))
    old_top.add(gdstk.rectangle((2, 0), (6, 4), layer=1, datatype=0))
    old_top.add(gdstk.rectangle((20, 0), (24, 4), layer=1, datatype=0))
    old_top.add(gdstk.rectangle((0, 0), (4, 4), layer=2, datatype=0))
    old_lib.write_gds(old_path)

    new_lib = gdstk.Library(unit=1e-6, precision=1e-9)
    new_lib.new_cell("top")
    new_lib.write_gds(new_path)

    old_native = extract_gds_polygons_native(old_path)
    new_native = extract_gds_polygons_native(new_path)
    record_delta = run_cuda_polygon_record_delta(old_native.buffer, new_native.buffer)
    result = build_record_delta_surplus_components(old_native.buffer, new_native.buffer, record_delta)

    assert result.stats.component_count == 3
    assert result.stats.overlap_component_count == 1
    assert result.stats.max_fragments_per_component == 2
    assert result.stats.max_vertices_per_component == 8

    component_keys = [
        (component.layer, component.old_count, component.new_count, component.bbox)
        for component in result.components
    ]
    assert component_keys == [
        (LayerSpec(1, 0), 2, 0, (0, 0, 6000, 4000)),
        (LayerSpec(1, 0), 1, 0, (20000, 0, 24000, 4000)),
        (LayerSpec(2, 0), 1, 0, (0, 0, 4000, 4000)),
    ]


def test_cuda_one_sided_boundary_edges_match_python_reference():
    if not cuda_ctypes_available():
        pytest.skip("CUDA ctypes backend is not available")

    polygons = (
        ((0, 0), (4000, 0), (4000, 4000), (0, 4000)),
        ((2000, 0), (6000, 0), (6000, 4000), (2000, 4000)),
    )
    bboxes = ((0, 0, 4000, 4000), (2000, 0, 6000, 4000))
    segments = _build_segments({"target": polygons})
    splits = _segment_splits(segments)
    expected = _boundary_edges_for_one_sided_union(segments, splits, polygons, bboxes)

    vertices = np.asarray([point for polygon in polygons for point in polygon], dtype=np.int64)
    offsets = np.asarray([0, 4, 8], dtype=np.uint64)
    signed_areas = np.asarray([_signed_twice_area(polygon) for polygon in polygons], dtype=np.int64)
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

    cuda_edges = run_cuda_one_sided_boundary_edges(
        vertices,
        offsets,
        np.asarray(bboxes, dtype=np.int64),
        signed_areas,
        np.asarray(piece_rows, dtype=np.float64),
        np.asarray(piece_polygon_indices, dtype=np.uint32),
    )
    actual = tuple(sorted(((_point_key((edge[0][0], edge[0][1])), _point_key((edge[1][0], edge[1][1]))) for edge in cuda_edges)))

    assert actual == expected


def test_cuda_one_sided_unsplit_boundary_edges_reports_safe_and_fallback():
    if not cuda_ctypes_available():
        pytest.skip("CUDA ctypes backend is not available")

    disjoint_polygons = (
        ((0, 0), (1000, 0), (1000, 1000), (0, 1000)),
        ((2000, 0), (3000, 0), (3000, 1000), (2000, 1000)),
    )
    disjoint_segments = _build_segments({"target": disjoint_polygons})
    disjoint_vertices = np.asarray([point for polygon in disjoint_polygons for point in polygon], dtype=np.int64)
    disjoint_offsets = np.asarray([0, 4, 8], dtype=np.uint64)
    disjoint_bboxes = np.asarray(((0, 0, 1000, 1000), (2000, 0, 3000, 1000)), dtype=np.int64)
    disjoint_signed = np.asarray([_signed_twice_area(polygon) for polygon in disjoint_polygons], dtype=np.int64)
    disjoint_rows = np.asarray([(s.start[0], s.start[1], s.stop[0], s.stop[1]) for s in disjoint_segments], dtype=np.float64)
    disjoint_indices = np.asarray([s.polygon_index for s in disjoint_segments], dtype=np.uint32)

    disjoint_edges, disjoint_safe = run_cuda_one_sided_boundary_edges_unsplit_if_safe(
        disjoint_vertices,
        disjoint_offsets,
        disjoint_bboxes,
        disjoint_signed,
        disjoint_rows,
        disjoint_indices,
    )

    assert disjoint_safe is True
    assert len(disjoint_edges) == len(disjoint_segments)

    overlapping_polygons = (
        ((0, 0), (4000, 0), (4000, 4000), (0, 4000)),
        ((2000, 0), (6000, 0), (6000, 4000), (2000, 4000)),
    )
    overlapping_segments = _build_segments({"target": overlapping_polygons})
    overlapping_edges, overlapping_safe = run_cuda_one_sided_boundary_edges_unsplit_if_safe(
        np.asarray([point for polygon in overlapping_polygons for point in polygon], dtype=np.int64),
        np.asarray([0, 4, 8], dtype=np.uint64),
        np.asarray(((0, 0, 4000, 4000), (2000, 0, 6000, 4000)), dtype=np.int64),
        np.asarray([_signed_twice_area(polygon) for polygon in overlapping_polygons], dtype=np.int64),
        np.asarray([(s.start[0], s.start[1], s.stop[0], s.stop[1]) for s in overlapping_segments], dtype=np.float64),
        np.asarray([s.polygon_index for s in overlapping_segments], dtype=np.uint32),
    )

    assert overlapping_safe is False
    assert len(overlapping_edges) == 0


def test_cuda_mixed_boundary_edges_match_python_reference():
    if not cuda_ctypes_available():
        pytest.skip("CUDA ctypes backend is not available")

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
        cuda_edges = run_cuda_mixed_boundary_edges(
            vertices,
            offsets,
            bboxes,
            sides,
            np.asarray(piece_rows, dtype=np.float64),
            direction=direction,
        )
        actual = tuple(sorted(((_point_key((edge[0][0], edge[0][1])), _point_key((edge[1][0], edge[1][1]))) for edge in cuda_edges)))

        assert actual == expected


def test_record_delta_components_preserve_new_side(tmp_path):
    if not cuda_ctypes_available():
        pytest.skip("CUDA ctypes backend is not available")

    old_path = tmp_path / "old.gds"
    new_path = tmp_path / "new.gds"

    old_lib = gdstk.Library(unit=1e-6, precision=1e-9)
    old_lib.new_cell("top")
    old_lib.write_gds(old_path)

    new_lib = gdstk.Library(unit=1e-6, precision=1e-9)
    new_top = new_lib.new_cell("top")
    new_top.add(gdstk.Polygon([(0, 0), (2, 0), (1, 1)], layer=7, datatype=1))
    new_lib.write_gds(new_path)

    old_native = extract_gds_polygons_native(old_path)
    new_native = extract_gds_polygons_native(new_path)
    record_delta = run_cuda_polygon_record_delta(old_native.buffer, new_native.buffer)
    result = build_record_delta_surplus_components(old_native.buffer, new_native.buffer, record_delta)

    assert result.stats.component_count == 1
    component = result.components[0]
    assert component.layer == LayerSpec(7, 1)
    assert component.old_count == 0
    assert component.new_count == 1
    assert component.polygons[0].side == "new"


def test_cuda_polygon_component_areas_handles_triangle_removal(tmp_path):
    if not cuda_ctypes_available():
        pytest.skip("CUDA ctypes backend is not available")

    old_path = tmp_path / "old.gds"
    new_path = tmp_path / "new.gds"
    old_lib = gdstk.Library(unit=1e-6, precision=1e-9)
    old_top = old_lib.new_cell("top")
    old_top.add(gdstk.Polygon([(0, 0), (3, 0), (0, 2)], layer=1, datatype=0))
    old_lib.write_gds(old_path)
    new_lib = gdstk.Library(unit=1e-6, precision=1e-9)
    new_lib.new_cell("top")
    new_lib.write_gds(new_path)

    old_native = extract_gds_polygons_native(old_path)
    new_native = extract_gds_polygons_native(new_path)
    record_delta = run_cuda_polygon_record_delta(old_native.buffer, new_native.buffer)
    components = build_record_delta_surplus_components(old_native.buffer, new_native.buffer, record_delta).components
    areas = run_cuda_polygon_component_areas(components)

    assert areas.tolist() == [[6_000_000, 0, 6_000_000]]


def test_cuda_polygon_component_fragments_handles_triangle_removal(tmp_path):
    if not cuda_ctypes_available():
        pytest.skip("CUDA ctypes backend is not available")

    old_path = tmp_path / "old.gds"
    new_path = tmp_path / "new.gds"
    old_lib = gdstk.Library(unit=1e-6, precision=1e-9)
    old_top = old_lib.new_cell("top")
    old_top.add(gdstk.Polygon([(0, 0), (3, 0), (0, 2)], layer=1, datatype=0))
    old_lib.write_gds(old_path)
    new_lib = gdstk.Library(unit=1e-6, precision=1e-9)
    new_lib.new_cell("top")
    new_lib.write_gds(new_path)

    old_native = extract_gds_polygons_native(old_path)
    new_native = extract_gds_polygons_native(new_path)
    record_delta = run_cuda_polygon_record_delta(old_native.buffer, new_native.buffer)
    components = build_record_delta_surplus_components(old_native.buffer, new_native.buffer, record_delta).components
    quads, fragment_areas, counts, areas = run_cuda_polygon_component_fragments(components)

    assert areas.tolist() == [[6_000_000, 0, 6_000_000]]
    assert fragment_areas[0, 0, : counts[0, 0]].sum() == 6_000_000
    assert counts.tolist() == [[1, 0]]
    assert quads[0, 0, 0].tolist() == [[0, 0], [3000, 0], [3000, 0], [0, 2000]]


def test_cuda_polygon_component_areas_unions_overlapping_same_side_rectangles(tmp_path):
    if not cuda_ctypes_available():
        pytest.skip("CUDA ctypes backend is not available")

    old_path = tmp_path / "old.gds"
    new_path = tmp_path / "new.gds"
    old_lib = gdstk.Library(unit=1e-6, precision=1e-9)
    old_top = old_lib.new_cell("top")
    old_top.add(gdstk.rectangle((0, 0), (4, 4), layer=1, datatype=0))
    old_top.add(gdstk.rectangle((2, 0), (6, 4), layer=1, datatype=0))
    old_lib.write_gds(old_path)
    new_lib = gdstk.Library(unit=1e-6, precision=1e-9)
    new_lib.new_cell("top")
    new_lib.write_gds(new_path)

    old_native = extract_gds_polygons_native(old_path)
    new_native = extract_gds_polygons_native(new_path)
    record_delta = run_cuda_polygon_record_delta(old_native.buffer, new_native.buffer)
    components = build_record_delta_surplus_components(old_native.buffer, new_native.buffer, record_delta).components
    areas = run_cuda_polygon_component_areas(components)

    assert areas.tolist() == [[48_000_000, 0, 48_000_000]]


def test_cuda_polygon_component_fragments_unions_overlapping_same_side_rectangles(tmp_path):
    if not cuda_ctypes_available():
        pytest.skip("CUDA ctypes backend is not available")

    old_path = tmp_path / "old.gds"
    new_path = tmp_path / "new.gds"
    old_lib = gdstk.Library(unit=1e-6, precision=1e-9)
    old_top = old_lib.new_cell("top")
    old_top.add(gdstk.rectangle((0, 0), (4, 4), layer=1, datatype=0))
    old_top.add(gdstk.rectangle((2, 0), (6, 4), layer=1, datatype=0))
    old_lib.write_gds(old_path)
    new_lib = gdstk.Library(unit=1e-6, precision=1e-9)
    new_lib.new_cell("top")
    new_lib.write_gds(new_path)

    old_native = extract_gds_polygons_native(old_path)
    new_native = extract_gds_polygons_native(new_path)
    record_delta = run_cuda_polygon_record_delta(old_native.buffer, new_native.buffer)
    components = build_record_delta_surplus_components(old_native.buffer, new_native.buffer, record_delta).components
    _quads, fragment_areas, counts, areas = run_cuda_polygon_component_fragments(components)

    assert areas.tolist() == [[48_000_000, 0, 48_000_000]]
    assert fragment_areas[0, 0, : counts[0, 0]].sum() == 48_000_000
    assert counts.tolist() == [[3, 0]]


def test_cuda_polygon_component_areas_reports_directional_old_new_difference(tmp_path):
    if not cuda_ctypes_available():
        pytest.skip("CUDA ctypes backend is not available")

    old_path = tmp_path / "old.gds"
    new_path = tmp_path / "new.gds"
    old_lib = gdstk.Library(unit=1e-6, precision=1e-9)
    old_top = old_lib.new_cell("top")
    old_top.add(gdstk.rectangle((0, 0), (4, 4), layer=1, datatype=0))
    old_lib.write_gds(old_path)
    new_lib = gdstk.Library(unit=1e-6, precision=1e-9)
    new_top = new_lib.new_cell("top")
    new_top.add(gdstk.rectangle((2, 0), (6, 4), layer=1, datatype=0))
    new_lib.write_gds(new_path)

    old_native = extract_gds_polygons_native(old_path)
    new_native = extract_gds_polygons_native(new_path)
    record_delta = run_cuda_polygon_record_delta(old_native.buffer, new_native.buffer)
    components = build_record_delta_surplus_components(old_native.buffer, new_native.buffer, record_delta).components
    areas = run_cuda_polygon_component_areas(components)

    assert areas.tolist() == [[16_000_000, 16_000_000, 32_000_000]]


def test_cuda_polygon_component_fragments_reports_directional_old_new_difference(tmp_path):
    if not cuda_ctypes_available():
        pytest.skip("CUDA ctypes backend is not available")

    old_path = tmp_path / "old.gds"
    new_path = tmp_path / "new.gds"
    old_lib = gdstk.Library(unit=1e-6, precision=1e-9)
    old_top = old_lib.new_cell("top")
    old_top.add(gdstk.rectangle((0, 0), (4, 4), layer=1, datatype=0))
    old_lib.write_gds(old_path)
    new_lib = gdstk.Library(unit=1e-6, precision=1e-9)
    new_top = new_lib.new_cell("top")
    new_top.add(gdstk.rectangle((2, 0), (6, 4), layer=1, datatype=0))
    new_lib.write_gds(new_path)

    old_native = extract_gds_polygons_native(old_path)
    new_native = extract_gds_polygons_native(new_path)
    record_delta = run_cuda_polygon_record_delta(old_native.buffer, new_native.buffer)
    components = build_record_delta_surplus_components(old_native.buffer, new_native.buffer, record_delta).components
    _quads, fragment_areas, counts, areas = run_cuda_polygon_component_fragments(components)

    assert areas.tolist() == [[16_000_000, 16_000_000, 32_000_000]]
    assert fragment_areas[0, 0, : counts[0, 0]].sum() == 16_000_000
    assert fragment_areas[0, 1, : counts[0, 1]].sum() == 16_000_000
    assert counts.tolist() == [[1, 1]]


def test_context_components_clip_new_surplus_against_existing_old_coverage(tmp_path):
    if not cuda_ctypes_available():
        pytest.skip("CUDA ctypes backend is not available")

    old_path = tmp_path / "old.gds"
    new_path = tmp_path / "new.gds"
    old_lib = gdstk.Library(unit=1e-6, precision=1e-9)
    old_top = old_lib.new_cell("top")
    old_top.add(gdstk.rectangle((0, 0), (4, 4), layer=1, datatype=0))
    old_lib.write_gds(old_path)

    new_lib = gdstk.Library(unit=1e-6, precision=1e-9)
    new_top = new_lib.new_cell("top")
    new_top.add(gdstk.rectangle((0, 0), (4, 4), layer=1, datatype=0))
    new_top.add(gdstk.rectangle((2, 0), (6, 4), layer=1, datatype=0))
    new_lib.write_gds(new_path)

    old_native = extract_gds_polygons_native(old_path)
    new_native = extract_gds_polygons_native(new_path)
    record_delta = run_cuda_polygon_record_delta(old_native.buffer, new_native.buffer)
    surplus_components = build_record_delta_surplus_components(old_native.buffer, new_native.buffer, record_delta).components
    context_result = add_overlapping_context_to_components(old_native.buffer, new_native.buffer, surplus_components)
    context_components = context_result.components
    areas = run_cuda_polygon_component_areas(context_components)

    assert len(surplus_components[0].polygons) == 1
    assert context_result.stats.context_index_backend == "cuda-context-overlap-pairs"
    assert context_result.stats.context_pair_count == 3
    assert len(context_components[0].polygons) == 3
    assert areas.tolist() == [[0, 16_000_000, 16_000_000]]


def test_cuda_polygon_component_fragments_clip_new_surplus_against_existing_old_coverage(tmp_path):
    if not cuda_ctypes_available():
        pytest.skip("CUDA ctypes backend is not available")

    old_path = tmp_path / "old.gds"
    new_path = tmp_path / "new.gds"
    old_lib = gdstk.Library(unit=1e-6, precision=1e-9)
    old_top = old_lib.new_cell("top")
    old_top.add(gdstk.rectangle((0, 0), (4, 4), layer=1, datatype=0))
    old_lib.write_gds(old_path)

    new_lib = gdstk.Library(unit=1e-6, precision=1e-9)
    new_top = new_lib.new_cell("top")
    new_top.add(gdstk.rectangle((0, 0), (4, 4), layer=1, datatype=0))
    new_top.add(gdstk.rectangle((2, 0), (6, 4), layer=1, datatype=0))
    new_lib.write_gds(new_path)

    old_native = extract_gds_polygons_native(old_path)
    new_native = extract_gds_polygons_native(new_path)
    record_delta = run_cuda_polygon_record_delta(old_native.buffer, new_native.buffer)
    surplus_components = build_record_delta_surplus_components(old_native.buffer, new_native.buffer, record_delta).components
    context_components = add_overlapping_context_to_components(old_native.buffer, new_native.buffer, surplus_components).components
    _quads, fragment_areas, counts, areas = run_cuda_polygon_component_fragments(context_components)

    assert fragment_areas[0, 1, : counts[0, 1]].sum() == 16_000_000
    assert counts.tolist() == [[0, 1]]
    assert areas.tolist() == [[0, 16_000_000, 16_000_000]]


def test_exact_result_from_cuda_polygon_component_fragments(tmp_path):
    if not cuda_ctypes_available():
        pytest.skip("CUDA ctypes backend is not available")

    old_path = tmp_path / "old.gds"
    new_path = tmp_path / "new.gds"
    old_lib = gdstk.Library(unit=1e-6, precision=1e-9)
    old_top = old_lib.new_cell("top")
    old_top.add(gdstk.rectangle((0, 0), (4, 4), layer=1, datatype=0))
    old_lib.write_gds(old_path)
    new_lib = gdstk.Library(unit=1e-6, precision=1e-9)
    new_top = new_lib.new_cell("top")
    new_top.add(gdstk.rectangle((2, 0), (6, 4), layer=1, datatype=0))
    new_lib.write_gds(new_path)

    old_native = extract_gds_polygons_native(old_path)
    new_native = extract_gds_polygons_native(new_path)
    record_delta = run_cuda_polygon_record_delta(old_native.buffer, new_native.buffer)
    surplus_components = build_record_delta_surplus_components(old_native.buffer, new_native.buffer, record_delta).components
    context_components = add_overlapping_context_to_components(old_native.buffer, new_native.buffer, surplus_components).components
    quads, fragment_areas, counts, areas = run_cuda_polygon_component_fragments(context_components)
    exact_result = exact_result_from_component_quads(context_components, quads, fragment_areas, counts, areas)

    assert len(exact_result.layers) == 1
    layer = exact_result.layers[0]
    assert layer.layer == old_native.buffer.layer_table.layers[0]
    assert layer.old_minus_new_twice_area == 16_000_000
    assert layer.new_minus_old_twice_area == 16_000_000
    assert layer.xor_twice_area == 32_000_000


def _diff_fragment(points: tuple[tuple[int, int], ...]) -> DiffFragment:
    return DiffFragment(points=points, bbox=polygon_bbox(points), twice_area=polygon_twice_area(points))
