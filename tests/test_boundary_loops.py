from __future__ import annotations

from gdsdiff.boundary_loops import assemble_boundary_loops
from gdsdiff.exact_oracle import DiffFragment
from gdsdiff.geometry import polygon_bbox, polygon_twice_area


def test_assemble_boundary_loops_merges_adjacent_rectangles():
    fragments = (
        _fragment(((0, 0), (2, 0), (2, 2), (0, 2))),
        _fragment(((2, 0), (4, 0), (4, 2), (2, 2))),
    )

    result = assemble_boundary_loops(fragments)

    assert len(result.fragments) == 1
    assert result.fragments[0].bbox == (0, 0, 4, 2)
    assert result.fragments[0].twice_area == 16
    assert len(result.fragments[0].points) == 4
    assert result.stats.input_fragment_count == 2
    assert result.stats.loop_count == 1


def test_assemble_boundary_loops_preserves_l_shape():
    fragments = (
        _fragment(((0, 0), (2, 0), (2, 1), (0, 1))),
        _fragment(((0, 1), (1, 1), (1, 3), (0, 3))),
    )

    result = assemble_boundary_loops(fragments)

    assert len(result.fragments) == 1
    assert result.fragments[0].bbox == (0, 0, 2, 3)
    assert result.fragments[0].twice_area == 8
    assert len(result.fragments[0].points) == 6


def test_assemble_boundary_loops_merges_trapezoid_slabs():
    fragments = (
        _fragment(((0, 0), (2, 0), (2, 2), (0, 4))),
        _fragment(((2, 0), (4, 0), (4, 0), (2, 2))),
    )

    result = assemble_boundary_loops(fragments)

    assert len(result.fragments) == 1
    assert result.fragments[0].bbox == (0, 0, 4, 4)
    assert result.fragments[0].twice_area == 16
    assert len(result.fragments[0].points) == 3


def test_assemble_boundary_loops_keeps_disconnected_shapes_separate():
    fragments = (
        _fragment(((0, 0), (1, 0), (1, 1), (0, 1))),
        _fragment(((3, 0), (4, 0), (4, 1), (3, 1))),
    )

    result = assemble_boundary_loops(fragments)

    assert len(result.fragments) == 2
    assert [fragment.bbox for fragment in result.fragments] == [(0, 0, 1, 1), (3, 0, 4, 1)]
    assert [fragment.twice_area for fragment in result.fragments] == [2, 2]


def test_assemble_boundary_loops_stitches_nested_hole():
    fragments = (
        _fragment(((0, 0), (12, 0), (12, 2), (0, 2))),
        _fragment(((0, 2), (2, 2), (2, 8), (0, 8))),
        _fragment(((10, 2), (12, 2), (12, 8), (10, 8))),
        _fragment(((0, 8), (12, 8), (12, 10), (0, 10))),
    )

    result = assemble_boundary_loops(fragments)

    assert len(result.fragments) == 1
    assert result.fragments[0].bbox == (0, 0, 12, 10)
    assert result.fragments[0].twice_area == 144
    assert len(result.fragments[0].points) == 11


def _fragment(points: tuple[tuple[int, int], ...]) -> DiffFragment:
    return DiffFragment(points=points, bbox=polygon_bbox(points), twice_area=polygon_twice_area(points))
