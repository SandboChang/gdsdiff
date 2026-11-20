from __future__ import annotations

import gdstk
import numpy as np
import pytest

from gdsdiff import (
    AllowedRegion,
    CompareStatus,
    DifferenceDirection,
    LayerSpec,
    classify_exact_differences,
    format_twice_area,
    run_full_exact_oracle,
)
from gdsdiff.geometry import LayerTable, PolygonBuffer
from gdsdiff.grid import snap_to_grid
from gdsdiff.policy import _global_points_from_gdstk


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


def _shift_result(layer=LayerSpec(1, 0)):
    old = _buffer([_rect(0, 0, 10, 10)], layer)
    new = _buffer([_rect(2, 0, 12, 10)], layer)
    return run_full_exact_oracle(old, new)


@pytest.mark.parametrize(
    "vertices",
    [
        [(-3, -2), (7, 1), (0, 5)],
        [(-3.5, -2.5), (7.5, 1.5), (0.5, 5.5)],
        [(-(1 << 54), -2), ((1 << 54) + 4, 1), (0, 5)],
    ],
)
def test_policy_grid_conversion_preserves_vertex_order_and_decimal_rounding(vertices):
    polygon = gdstk.Polygon(vertices)
    origin = (500, -1500)
    expected = tuple((snap_to_grid(x, 1) + origin[0], snap_to_grid(y, 1) + origin[1]) for x, y in polygon.points)
    assert _global_points_from_gdstk([polygon], origin) == (expected,)


def test_policy_grid_conversion_keeps_coordinate_bounds():
    polygon = gdstk.Polygon([(0, 0), (1 << 63, 1), (0, 5)])
    with pytest.raises(OverflowError, match="exceeds signed int64 range"):
        _global_points_from_gdstk([polygon], (0, 0))


def test_empty_policy_marks_all_differences_unexpected():
    result = classify_exact_differences(_shift_result())

    assert result.status == CompareStatus.UNEXPECTED_DIFFERENCES
    assert format_twice_area(result.expected_twice_area) == "0"
    assert format_twice_area(result.unexpected_twice_area) == "40"
    assert {component.classification for component in result.layers[0].components} == {"unexpected"}


def test_bbox_allowed_region_marks_inside_difference_expected():
    layer = LayerSpec(1, 0)
    policy = (
        AllowedRegion(
            layers=(layer,),
            direction=DifferenceDirection.REMOVAL,
            bbox=(0, 0, 2, 10),
            label="left edge",
        ),
    )

    result = classify_exact_differences(_shift_result(layer), policy)

    assert result.status == CompareStatus.UNEXPECTED_DIFFERENCES
    assert format_twice_area(result.expected_twice_area) == "20"
    assert format_twice_area(result.unexpected_twice_area) == "20"
    assert result.layers[0].expected_components[0].bbox == (0, 0, 2, 10)


def test_crossing_allowed_region_splits_component_exactly():
    layer = LayerSpec(1, 0)
    policy = (
        AllowedRegion(
            layers=(layer,),
            direction=DifferenceDirection.REMOVAL,
            bbox=(0, 0, 1, 10),
        ),
    )

    result = classify_exact_differences(_shift_result(layer), policy)

    assert format_twice_area(result.expected_twice_area) == "10"
    assert format_twice_area(result.unexpected_twice_area) == "30"
    expected = result.layers[0].expected_components[0]
    unexpected_removal = [component for component in result.layers[0].unexpected_components if component.direction == DifferenceDirection.REMOVAL][0]
    assert expected.bbox == (0, 0, 1, 10)
    assert unexpected_removal.bbox == (1, 0, 2, 10)


def test_wrong_layer_policy_does_not_allow_difference():
    layer = LayerSpec(1, 0)
    result = classify_exact_differences(
        _shift_result(layer),
        (AllowedRegion(layers=(LayerSpec(2, 0),), direction=DifferenceDirection.BOTH, bbox=(0, 0, 12, 10)),),
    )

    assert format_twice_area(result.expected_twice_area) == "0"
    assert format_twice_area(result.unexpected_twice_area) == "40"


def test_directional_addition_policy_does_not_allow_removal():
    layer = LayerSpec(1, 0)
    result = classify_exact_differences(
        _shift_result(layer),
        (AllowedRegion(layers=(layer,), direction=DifferenceDirection.ADDITION, bbox=(0, 0, 12, 10)),),
    )

    assert format_twice_area(result.expected_twice_area) == "20"
    assert format_twice_area(result.unexpected_twice_area) == "20"
    assert {component.direction for component in result.layers[0].unexpected_components} == {DifferenceDirection.REMOVAL}


def test_whole_layer_permission_allows_matching_direction_only():
    layer = LayerSpec(1, 0)
    result = classify_exact_differences(
        _shift_result(layer),
        (AllowedRegion(layers=(layer,), direction=DifferenceDirection.ADDITION, whole_layer=True),),
    )

    assert format_twice_area(result.expected_twice_area) == "20"
    assert format_twice_area(result.unexpected_twice_area) == "20"
    assert result.effective_policy["allowed_region_count"] == 1


def test_both_direction_policy_allows_addition_and_removal():
    layer = LayerSpec(1, 0)
    result = classify_exact_differences(
        _shift_result(layer),
        (AllowedRegion(layers=(layer,), direction=DifferenceDirection.BOTH, bbox=(0, 0, 12, 10)),),
    )

    assert result.status == CompareStatus.EXPECTED_DIFFERENCES_ONLY
    assert format_twice_area(result.expected_twice_area) == "40"
    assert format_twice_area(result.unexpected_twice_area) == "0"


def test_allowed_region_halo_expands_bbox():
    layer = LayerSpec(1, 0)
    result = classify_exact_differences(
        _shift_result(layer),
        (AllowedRegion(layers=(layer,), direction=DifferenceDirection.REMOVAL, bbox=(0, 0, 1, 10), halo_dbu=1),),
    )

    assert format_twice_area(result.expected_twice_area) == "20"
    assert format_twice_area(result.unexpected_twice_area) == "20"
