from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scgds.gdsdiff import LayerSpec, format_twice_area, run_full_exact_oracle
from scgds.gdsdiff.backends.exact import (
    CudaManhattanRectExactComponentBackend,
    GdstkExactComponentBackend,
    ManhattanRectExactComponentBackend,
    UnsupportedExactComponent,
)
from scgds.gdsdiff.backends.cuda_ctypes import (
    cuda_ctypes_available,
    run_cuda_rect_exact_areas,
    run_cuda_rect_exact_fragments,
    run_cuda_rect_set_exact_fragments,
)
from scgds.gdsdiff.backends.metal import MetalExactUnavailable, MetalManhattanRectExactComponentBackend
from scgds.gdsdiff.backends.metal_ctypes import metal_ctypes_available, run_metal_rect_set_exact_fragments
from scgds.gdsdiff.exact_oracle import ExactLayerTask, _component_task_specs, _sparse_rectangle_residual_task_specs
from scgds.gdsdiff.extract import extract_gds_polygons
from scgds.gdsdiff.geometry import LayerTable, PolygonBuffer


def _buffer(polygons, layer=LayerSpec(1, 0)):
    vertices = []
    offsets = [0]
    for polygon in polygons:
        vertices.extend(polygon)
        offsets.append(len(vertices))
    bboxes = []
    for polygon in polygons:
        bboxes.append([min(x for x, _ in polygon), min(y for _, y in polygon), max(x for x, _ in polygon), max(y for _, y in polygon)])
    return PolygonBuffer(
        vertices_xy=np.asarray(vertices, dtype=np.int64).reshape((-1, 2)),
        polygon_offsets=np.asarray(offsets, dtype=np.uint64),
        layer_ids=np.zeros(len(polygons), dtype=np.uint32),
        bboxes=np.asarray(bboxes, dtype=np.int64).reshape((-1, 4)),
        flags=np.zeros(len(polygons), dtype=np.uint32),
        source_ids=np.arange(len(polygons), dtype=np.uint64),
        layer_table=LayerTable((layer,)),
    )


def _rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _diamond(cx, cy, r):
    return [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]


def test_format_twice_area():
    assert format_twice_area(0) == "0"
    assert format_twice_area(2) == "1"
    assert format_twice_area(3) == "1.5"


def test_identical_buffers_are_equivalent():
    old = _buffer([_rect(0, 0, 10, 10)])
    new = _buffer([_rect(0, 0, 10, 10)])

    result = run_full_exact_oracle(old, new)

    assert result.equivalent
    assert result.layers[0].xor_twice_area == 0


def test_shifted_rectangle_has_directional_differences():
    old = _buffer([_rect(0, 0, 10, 10)])
    new = _buffer([_rect(2, 0, 12, 10)])

    result = run_full_exact_oracle(old, new)
    layer = result.layers[0]

    assert not result.equivalent
    assert format_twice_area(layer.old_minus_new_twice_area) == "20"
    assert format_twice_area(layer.new_minus_old_twice_area) == "20"
    assert format_twice_area(layer.xor_twice_area) == "40"
    assert layer.old_minus_new[0].bbox == (0, 0, 2, 10)
    assert layer.new_minus_old[0].bbox == (10, 0, 12, 10)


def test_different_fracture_with_same_coverage_is_equivalent():
    old = _buffer([_rect(0, 0, 10, 10)])
    new = _buffer([_rect(0, 0, 5, 10), _rect(5, 0, 10, 10)])

    result = run_full_exact_oracle(old, new)

    assert result.equivalent


def test_duplicate_removal_preserves_union_coverage():
    old = _buffer([_rect(0, 0, 10, 10), _rect(0, 0, 10, 10)])
    new = _buffer([_rect(0, 0, 10, 10)])

    result = run_full_exact_oracle(old, new)

    assert result.equivalent


def test_large_coordinates_are_rebased_before_boolean():
    base = 10**12
    old = _buffer([_rect(base, base, base + 10, base + 10)])
    new = _buffer([_rect(base + 1, base, base + 11, base + 10)])

    result = run_full_exact_oracle(old, new)
    layer = result.layers[0]

    assert format_twice_area(layer.old_minus_new_twice_area) == "10"
    assert layer.old_minus_new[0].bbox == (base, base, base + 1, base + 10)


def test_component_task_specs_split_independent_bbox_components():
    layer = LayerSpec(1, 0)
    tasks = _component_task_specs(
        layer,
        (
            tuple(_rect(0, 0, 10, 10)),
            tuple(_rect(100, 0, 110, 10)),
        ),
        (
            tuple(_rect(1, 0, 11, 10)),
            tuple(_rect(100, 0, 110, 10)),
        ),
        1.0,
    )

    assert len(tasks) == 1
    assert {task.partition for task in tasks} == {"bbox-component"}
    assert {len(task.old_points) + len(task.new_points) for task in tasks} == {2}


def test_component_partition_exact_result_matches_single_worker_result():
    old_polygons = []
    new_polygons = []
    for index in range(1100):
        x0 = index * 20
        old_polygons.append(_rect(x0, 0, x0 + 10, 10))
        if index == 7:
            new_polygons.append(_rect(x0 + 2, 0, x0 + 12, 10))
        else:
            new_polygons.append(_rect(x0, 0, x0 + 10, 10))
    old = _buffer(old_polygons)
    new = _buffer(new_polygons)

    single = run_full_exact_oracle(old, new, workers=1)
    component = run_full_exact_oracle(old, new, workers=2, component_partition=True)

    assert component.equivalent is False
    assert component.layers[0].old_minus_new_twice_area == single.layers[0].old_minus_new_twice_area
    assert component.layers[0].new_minus_old_twice_area == single.layers[0].new_minus_old_twice_area
    assert component.layers[0].xor_twice_area == single.layers[0].xor_twice_area


def test_exact_component_backend_matches_normal_exact_oracle():
    old_polygons = []
    new_polygons = []
    for index in range(1100):
        x0 = index * 20
        old_polygons.append(_rect(x0, 0, x0 + 10, 10))
        if index in {7, 700}:
            new_polygons.append(_rect(x0 + 2, 0, x0 + 12, 10))
        else:
            new_polygons.append(_rect(x0, 0, x0 + 10, 10))
    old = _buffer(old_polygons)
    new = _buffer(new_polygons)
    backend = GdstkExactComponentBackend()

    normal = run_full_exact_oracle(old, new, workers=1)
    accelerated_contract = run_full_exact_oracle(
        old,
        new,
        workers=2,
        component_partition=True,
        component_executor=backend.compare_components,
    )

    assert backend.capabilities().name == "gdstk-exact-component"
    assert backend.capabilities().requires_gpu is False
    assert accelerated_contract.equivalent is False
    assert accelerated_contract.layers[0].old_minus_new_twice_area == normal.layers[0].old_minus_new_twice_area
    assert accelerated_contract.layers[0].new_minus_old_twice_area == normal.layers[0].new_minus_old_twice_area
    assert accelerated_contract.layers[0].xor_twice_area == normal.layers[0].xor_twice_area


def test_exact_component_executor_receives_component_tasks():
    old_polygons = [_rect(index * 20, 0, index * 20 + 10, 10) for index in range(1100)]
    new_polygons = list(old_polygons)
    new_polygons[7] = _rect(7 * 20 + 2, 0, 7 * 20 + 12, 10)
    old = _buffer(old_polygons)
    new = _buffer(new_polygons)
    seen_partitions = []

    def executor(tasks: tuple[ExactLayerTask, ...]):
        seen_partitions.extend(task.partition for task in tasks)
        return GdstkExactComponentBackend().compare_components(tasks)

    result = run_full_exact_oracle(old, new, workers=2, component_partition=True, component_executor=executor)

    assert result.equivalent is False
    assert seen_partitions
    assert set(seen_partitions) == {"rectangle-pair"}


def test_exact_component_executor_skips_identical_rectangle_pairs_before_backend():
    old_polygons = [_rect(index * 20, 0, index * 20 + 10, 10) for index in range(1100)]
    new_polygons = list(old_polygons)
    old = _buffer(old_polygons)
    new = _buffer(new_polygons)
    seen_tasks = []

    def executor(tasks: tuple[ExactLayerTask, ...]):
        seen_tasks.extend(tasks)
        return GdstkExactComponentBackend().compare_components(tasks)

    result = run_full_exact_oracle(old, new, workers=2, component_partition=True, component_executor=executor)

    assert result.equivalent
    assert result.layers[0].xor_twice_area == 0
    assert seen_tasks == []


def test_component_tasks_skip_identical_non_rect_source_sets():
    layer = LayerSpec(1, 0)
    polygon_a = ((0, 0), (10, 0), (7, 4), (10, 10), (0, 10))
    polygon_b = ((30, 0), (40, 5), (30, 10), (25, 5))
    polygon_a_reversed = tuple(reversed(polygon_a))
    polygon_b_rotated = polygon_b[2:] + polygon_b[:2]

    tasks = _component_task_specs(
        layer,
        (polygon_a, polygon_b),
        (polygon_b_rotated, polygon_a_reversed),
        1.0,
    )

    assert tasks == ()


def test_sparse_rectangle_residual_tasks_drop_common_rectangles_before_components():
    layer = LayerSpec(1, 0)
    common = tuple(_rect(0, 0, 10, 10))
    changed_old = tuple(_rect(20, 0, 30, 10))
    changed_new = tuple(_rect(22, 0, 32, 10))

    tasks = _sparse_rectangle_residual_task_specs(
        layer,
        (common, changed_old),
        (common, changed_new),
        1.0,
    )

    assert tasks is not None
    assert len(tasks) == 1
    assert tasks[0].old_points == (changed_old,)
    assert tasks[0].new_points == (changed_new,)


def test_sparse_rectangle_residual_tasks_preserve_unmatched_duplicate_rectangles():
    layer = LayerSpec(1, 0)
    duplicate = tuple(_rect(0, 0, 10, 10))

    tasks = _sparse_rectangle_residual_task_specs(
        layer,
        (duplicate, duplicate),
        (duplicate,),
        1.0,
    )

    assert tasks == ()


def test_sparse_rectangle_residual_tasks_keep_common_rectangles_that_clip_residuals():
    layer = LayerSpec(1, 0)
    common = tuple(_rect(0, 0, 10, 10))
    overlapping_new = tuple(_rect(5, 0, 15, 10))

    tasks = _sparse_rectangle_residual_task_specs(
        layer,
        (common,),
        (common, overlapping_new),
        1.0,
    )

    assert tasks is None


def test_exact_component_executor_prefers_rectangle_pair_tasks_for_rectangle_layers():
    old_polygons = [_rect(index * 20, 0, index * 20 + 10, 10) for index in range(1100)]
    new_polygons = list(old_polygons)
    new_polygons[7] = _rect(7 * 20 + 2, 0, 7 * 20 + 12, 10)
    old = _buffer(old_polygons)
    new = _buffer(new_polygons)
    seen_partitions = []

    def executor(tasks: tuple[ExactLayerTask, ...]):
        seen_partitions.extend(task.partition for task in tasks)
        return CudaManhattanRectExactComponentBackend(fallback_to_gdstk=False).compare_components(tasks)

    result = run_full_exact_oracle(old, new, workers=2, component_partition=True, component_executor=executor)

    assert result.equivalent is False
    assert seen_partitions
    assert set(seen_partitions) == {"rectangle-pair"}


def test_manhattan_rect_backend_matches_gdstk_for_single_rectangle_components():
    old_polygons = []
    new_polygons = []
    for index in range(1100):
        x0 = index * 20
        old_polygons.append(_rect(x0, 0, x0 + 10, 10))
        if index == 7:
            new_polygons.append(_rect(x0 + 2, 0, x0 + 12, 10))
        elif index == 99:
            new_polygons.append(_rect(x0, 0, x0 + 5, 10))
        else:
            new_polygons.append(_rect(x0, 0, x0 + 10, 10))
    old = _buffer(old_polygons)
    new = _buffer(new_polygons)
    backend = ManhattanRectExactComponentBackend()

    expected = run_full_exact_oracle(old, new, workers=1)
    actual = run_full_exact_oracle(
        old,
        new,
        workers=2,
        component_partition=True,
        component_executor=backend.compare_components,
    )

    assert backend.capabilities().name == "manhattan-rect-exact-component"
    assert backend.stats.fallback_tasks == 0
    assert backend.stats.unsupported_tasks == 0
    assert backend.stats.accelerated_tasks == 2
    assert actual.layers[0].old_minus_new_twice_area == expected.layers[0].old_minus_new_twice_area
    assert actual.layers[0].new_minus_old_twice_area == expected.layers[0].new_minus_old_twice_area
    assert actual.layers[0].xor_twice_area == expected.layers[0].xor_twice_area


def test_manhattan_rect_backend_matches_gdstk_for_rectangle_set_component():
    old_polygons = [
        _rect(0, 0, 10, 10),
        _rect(10, 0, 20, 10),
    ]
    new_polygons = [
        _rect(5, 0, 15, 10),
        _rect(20, 0, 25, 10),
    ]
    old = _buffer(old_polygons)
    new = _buffer(new_polygons)
    backend = ManhattanRectExactComponentBackend()

    expected = run_full_exact_oracle(old, new, workers=1)
    actual = run_full_exact_oracle(
        old,
        new,
        workers=2,
        component_partition=True,
        component_executor=backend.compare_components,
    )

    assert backend.stats.cpu_rect_set_tasks == 1
    assert backend.stats.fallback_tasks == 0
    assert backend.stats.unsupported_tasks == 0
    assert actual.layers[0].old_minus_new_twice_area == expected.layers[0].old_minus_new_twice_area
    assert actual.layers[0].new_minus_old_twice_area == expected.layers[0].new_minus_old_twice_area
    assert actual.layers[0].xor_twice_area == expected.layers[0].xor_twice_area


def test_cuda_rect_set_exact_fragments_kernel_matches_expected_area():
    if not cuda_ctypes_available():
        pytest.skip("CUDA ctypes backend is not available")
    old_rects = np.asarray([[[0, 0, 10, 10], [10, 0, 20, 10]]], dtype=np.int64)
    new_rects = np.asarray([[[5, 0, 15, 10], [20, 0, 25, 10]]], dtype=np.int64)
    old_counts = np.asarray([2], dtype=np.uint8)
    new_counts = np.asarray([2], dtype=np.uint8)

    rects, counts, areas = run_cuda_rect_set_exact_fragments(old_rects, new_rects, old_counts, new_counts)

    assert counts.shape == (1, 2)
    assert areas.tolist() == [[200, 100, 300]]
    assert counts[0, 0] >= 1
    assert counts[0, 1] >= 1
    emitted_old_area = sum((r[2] - r[0]) * (r[3] - r[1]) * 2 for r in rects[0, 0, : counts[0, 0]])
    emitted_new_area = sum((r[2] - r[0]) * (r[3] - r[1]) * 2 for r in rects[0, 1, : counts[0, 1]])
    assert emitted_old_area == 200
    assert emitted_new_area == 100


def test_cuda_rect_backend_uses_cuda_for_rectangle_set_component():
    if not cuda_ctypes_available():
        pytest.skip("CUDA ctypes backend is not available")
    old_polygons = [
        _rect(0, 0, 10, 10),
        _rect(10, 0, 20, 10),
    ]
    new_polygons = [
        _rect(5, 0, 15, 10),
        _rect(20, 0, 25, 10),
    ]
    old = _buffer(old_polygons)
    new = _buffer(new_polygons)
    backend = CudaManhattanRectExactComponentBackend()

    expected = run_full_exact_oracle(old, new, workers=1)
    actual = run_full_exact_oracle(
        old,
        new,
        workers=2,
        component_partition=True,
        component_executor=backend.compare_components,
    )

    assert backend.stats.cuda_rect_set_tasks == 1
    assert backend.stats.cpu_rect_set_tasks == 0
    assert backend.stats.fallback_tasks == 0
    assert backend.stats.unsupported_tasks == 0
    assert actual.layers[0].old_minus_new_twice_area == expected.layers[0].old_minus_new_twice_area
    assert actual.layers[0].new_minus_old_twice_area == expected.layers[0].new_minus_old_twice_area
    assert actual.layers[0].xor_twice_area == expected.layers[0].xor_twice_area


def test_manhattan_rect_backend_fails_closed_on_unsupported_component():
    old_polygons = [_rect(index * 20, 0, index * 20 + 10, 10) for index in range(1100)]
    new_polygons = list(old_polygons)
    new_polygons[7] = _diamond(7 * 20 + 5, 5, 5)
    old = _buffer(old_polygons)
    new = _buffer(new_polygons)
    backend = ManhattanRectExactComponentBackend()

    with pytest.raises(UnsupportedExactComponent):
        run_full_exact_oracle(
            old,
            new,
            workers=2,
            component_partition=True,
            component_executor=backend.compare_components,
        )

    assert backend.stats.unsupported_tasks == 1


def test_manhattan_rect_backend_can_fallback_to_gdstk_for_unsupported_component():
    old_polygons = [_rect(index * 20, 0, index * 20 + 10, 10) for index in range(1100)]
    new_polygons = list(old_polygons)
    new_polygons[7] = _diamond(7 * 20 + 5, 5, 5)
    old = _buffer(old_polygons)
    new = _buffer(new_polygons)
    backend = ManhattanRectExactComponentBackend(fallback_to_gdstk=True)

    expected = run_full_exact_oracle(old, new, workers=1)
    actual = run_full_exact_oracle(
        old,
        new,
        workers=2,
        component_partition=True,
        component_executor=backend.compare_components,
    )

    assert backend.stats.unsupported_tasks == 1
    assert backend.stats.fallback_tasks == 1
    assert actual.layers[0].old_minus_new_twice_area == expected.layers[0].old_minus_new_twice_area
    assert actual.layers[0].new_minus_old_twice_area == expected.layers[0].new_minus_old_twice_area
    assert actual.layers[0].xor_twice_area == expected.layers[0].xor_twice_area


def test_manhattan_rect_backend_matches_equal_count_fixture():
    before = "references/testGDS/equal_count/equal_count_before.gds"
    after = "references/testGDS/equal_count/equal_count_after.gds"
    if not Path(before).exists() or not Path(after).exists():
        pytest.skip("local reference GDS fixtures are not present")

    old = extract_gds_polygons(before, top_name="equal_count_top").buffer
    new = extract_gds_polygons(after, top_name="equal_count_top").buffer
    backend = ManhattanRectExactComponentBackend()

    expected = run_full_exact_oracle(old, new, workers=1)
    actual = run_full_exact_oracle(
        old,
        new,
        workers=2,
        component_partition=True,
        component_executor=backend.compare_components,
    )

    assert backend.stats.fallback_tasks == 0
    assert backend.stats.unsupported_tasks == 0
    assert backend.stats.accelerated_tasks == 3
    assert actual.layers[0].old_minus_new_twice_area == expected.layers[0].old_minus_new_twice_area
    assert actual.layers[0].new_minus_old_twice_area == expected.layers[0].new_minus_old_twice_area
    assert actual.layers[0].xor_twice_area == expected.layers[0].xor_twice_area


def test_cuda_rect_exact_area_kernel_matches_known_rectangle_overlap():
    if not cuda_ctypes_available():
        pytest.skip("CUDA ctypes backend is not available")
    old_rects = np.asarray([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.int64)
    new_rects = np.asarray([[2, 0, 12, 10], [20, 0, 30, 10]], dtype=np.int64)
    present = np.asarray([1, 1], dtype=np.uint8)

    areas = run_cuda_rect_exact_areas(old_rects, new_rects, present, present)

    assert areas.tolist() == [[40, 40, 80], [200, 200, 400]]


def test_cuda_rect_exact_fragment_kernel_emits_difference_pieces():
    if not cuda_ctypes_available():
        pytest.skip("CUDA ctypes backend is not available")
    old_rects = np.asarray([[0, 0, 10, 10]], dtype=np.int64)
    new_rects = np.asarray([[2, 0, 12, 10]], dtype=np.int64)
    present = np.asarray([1], dtype=np.uint8)

    rects, counts, areas = run_cuda_rect_exact_fragments(old_rects, new_rects, present, present)

    assert counts.tolist() == [[1, 1]]
    assert rects[0, 0].tolist() == [0, 0, 2, 10]
    assert rects[0, 4].tolist() == [10, 0, 12, 10]
    assert areas.tolist() == [[40, 40, 80]]


def test_cuda_manhattan_rect_backend_matches_equal_count_fixture():
    if not cuda_ctypes_available():
        pytest.skip("CUDA ctypes backend is not available")
    before = "references/testGDS/equal_count/equal_count_before.gds"
    after = "references/testGDS/equal_count/equal_count_after.gds"
    if not Path(before).exists() or not Path(after).exists():
        pytest.skip("local reference GDS fixtures are not present")

    old = extract_gds_polygons(before, top_name="equal_count_top").buffer
    new = extract_gds_polygons(after, top_name="equal_count_top").buffer
    backend = CudaManhattanRectExactComponentBackend()

    expected = run_full_exact_oracle(old, new, workers=1)
    actual = run_full_exact_oracle(
        old,
        new,
        workers=2,
        component_partition=True,
        component_executor=backend.compare_components,
    )

    assert backend.capabilities().requires_gpu is True
    assert backend.stats.accelerated_tasks == 0
    assert backend.stats.cuda_rect_set_tasks == 1
    assert backend.stats.identical_skipped_tasks == 2
    assert backend.stats.fallback_tasks == 0
    assert backend.stats.unsupported_tasks == 0
    assert actual.layers[0].old_minus_new_twice_area == expected.layers[0].old_minus_new_twice_area
    assert actual.layers[0].new_minus_old_twice_area == expected.layers[0].new_minus_old_twice_area
    assert actual.layers[0].xor_twice_area == expected.layers[0].xor_twice_area


def test_metal_rect_set_exact_fragments_kernel_matches_expected_area():
    if not metal_ctypes_available():
        pytest.skip("Metal ctypes backend is not available")
    old_rects = np.asarray([[[0, 0, 10, 10], [10, 0, 20, 10]]], dtype=np.int64)
    new_rects = np.asarray([[[5, 0, 15, 10], [20, 0, 25, 10]]], dtype=np.int64)
    old_counts = np.asarray([2], dtype=np.uint8)
    new_counts = np.asarray([2], dtype=np.uint8)

    rects, counts, areas = run_metal_rect_set_exact_fragments(old_rects, new_rects, old_counts, new_counts)

    assert counts.shape == (1, 2)
    assert areas.tolist() == [[200, 100, 300]]
    assert counts[0, 0] >= 1
    assert counts[0, 1] >= 1
    emitted_old_area = sum((r[2] - r[0]) * (r[3] - r[1]) * 2 for r in rects[0, 0, : counts[0, 0]])
    emitted_new_area = sum((r[2] - r[0]) * (r[3] - r[1]) * 2 for r in rects[0, 1, : counts[0, 1]])
    assert emitted_old_area == 200
    assert emitted_new_area == 100


def test_metal_rect_backend_uses_metal_or_fails_closed():
    backend = MetalManhattanRectExactComponentBackend()

    assert backend.capabilities().name == "metal-manhattan-rect-exact-component"
    assert backend.capabilities().requires_gpu is True
    task = ExactLayerTask(
        LayerSpec(1, 0),
        1.0,
        (tuple(_rect(0, 0, 10, 10)), tuple(_rect(10, 0, 20, 10))),
        (tuple(_rect(5, 0, 15, 10)), tuple(_rect(20, 0, 25, 10))),
        partition="bbox-component",
    )
    if not metal_ctypes_available():
        with pytest.raises(MetalExactUnavailable):
            backend.compare_components((task,))
        return

    diff = backend.compare_components((task,))[0]

    assert backend.stats.metal_rect_set_tasks == 1
    assert backend.stats.cpu_rect_set_tasks == 0
    assert backend.stats.fallback_tasks == 0
    assert backend.stats.unsupported_tasks == 0
    assert diff.old_minus_new_twice_area == 200
    assert diff.new_minus_old_twice_area == 100
    assert diff.xor_twice_area == 300
