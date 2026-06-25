from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import gcd
from time import perf_counter
from typing import Any

from .exact_oracle import DiffFragment
from .geometry import polygon_bbox, polygon_twice_area

Point = tuple[int, int]
Segment = tuple[Point, Point]


@dataclass(frozen=True)
class BoundaryLoopAssemblyStats:
    input_fragment_count: int
    boundary_edge_count: int
    loop_count: int
    dropped_zero_area_loop_count: int
    cuda_edge_cancel_s: float = 0.0
    cuda_edge_cancel_kernel_count: int = 0
    cuda_edge_cancel_event_elapsed_s: float = 0.0
    cuda_edge_cancel_transferred_bytes: int = 0
    native_edge_cancel_s: float = 0.0
    cuda_loop_trace_s: float = 0.0
    native_loop_trace_s: float = 0.0
    python_loop_trace_s: float = 0.0
    hole_stitch_s: float = 0.0

    def to_json(self) -> dict[str, int | float]:
        return {
            "input_fragment_count": self.input_fragment_count,
            "boundary_edge_count": self.boundary_edge_count,
            "loop_count": self.loop_count,
            "dropped_zero_area_loop_count": self.dropped_zero_area_loop_count,
            "cuda_edge_cancel_s": self.cuda_edge_cancel_s,
            "cuda_edge_cancel_kernel_count": self.cuda_edge_cancel_kernel_count,
            "cuda_edge_cancel_event_elapsed_s": self.cuda_edge_cancel_event_elapsed_s,
            "cuda_edge_cancel_transferred_bytes": self.cuda_edge_cancel_transferred_bytes,
            "native_edge_cancel_s": self.native_edge_cancel_s,
            "cuda_loop_trace_s": self.cuda_loop_trace_s,
            "native_loop_trace_s": self.native_loop_trace_s,
            "python_loop_trace_s": self.python_loop_trace_s,
            "hole_stitch_s": self.hole_stitch_s,
        }


@dataclass(frozen=True)
class BoundaryLoopAssemblyResult:
    fragments: tuple[DiffFragment, ...]
    stats: BoundaryLoopAssemblyStats


def assemble_boundary_loops(fragments: tuple[DiffFragment, ...]) -> BoundaryLoopAssemblyResult:
    """Cancel shared directed fragment edges and trace remaining boundary loops.

    This is a deterministic integer post-process for GPU-emitted trapezoid
    fragments. It deliberately avoids polygon Boolean operations; the input is
    assumed to be already decomposed into non-overlapping coverage cells.
    """
    cuda_edge_cancel_s = 0.0
    cuda_edge_cancel_kernel_count = 0
    cuda_edge_cancel_event_elapsed_s = 0.0
    cuda_edge_cancel_transferred_bytes = 0
    native_edge_cancel_s = 0.0
    cuda_loop_trace_s = 0.0
    native_loop_trace_s = 0.0
    python_loop_trace_s = 0.0
    edge_backend_result = _try_cuda_boundary_edges_from_fragments(fragments)
    if edge_backend_result is None:
        edge_backend_result = _try_native_boundary_edges_from_fragments(fragments)
    edge_array = edge_backend_result[0] if edge_backend_result is not None else None
    if edge_backend_result is not None:
        if edge_backend_result[2] == "cuda":
            cuda_edge_cancel_s = edge_backend_result[1]
            cuda_edge_cancel_kernel_count = edge_backend_result[3]
            cuda_edge_cancel_event_elapsed_s = edge_backend_result[4]
            cuda_edge_cancel_transferred_bytes = edge_backend_result[5]
        else:
            native_edge_cancel_s = edge_backend_result[1]
    if edge_array is not None:
        boundary_edge_count = int(len(edge_array))
        cuda_start = perf_counter()
        cuda_result = _try_cuda_boundary_loop_fragments_array(edge_array)
        cuda_loop_trace_s = perf_counter() - cuda_start
        native_loop_result = None
        if cuda_result is None:
            native_start = perf_counter()
            native_loop_result = _try_native_boundary_loop_fragments_array(edge_array)
            native_loop_trace_s = perf_counter() - native_start
        if cuda_result is not None:
            result = list(cuda_result.fragments)
            dropped = cuda_result.dropped_zero_area_loop_count
        elif native_loop_result is not None:
            result = list(native_loop_result.fragments)
            dropped = native_loop_result.dropped_zero_area_loop_count
        else:
            boundary_edges = _segments_from_edges_array(edge_array)
            python_start = perf_counter()
            result, dropped = _fragments_from_python_boundary_edges(boundary_edges)
            python_loop_trace_s = perf_counter() - python_start
    else:
        boundary_edges = _boundary_edges(fragments)
        boundary_edge_count = len(boundary_edges)
        cuda_start = perf_counter()
        cuda_result = _try_cuda_boundary_loop_fragments(boundary_edges)
        cuda_loop_trace_s = perf_counter() - cuda_start
        if cuda_result is not None:
            result = list(cuda_result.fragments)
            dropped = cuda_result.dropped_zero_area_loop_count
        else:
            python_start = perf_counter()
            result, dropped = _fragments_from_python_boundary_edges(boundary_edges)
            python_loop_trace_s = perf_counter() - python_start
    stitch_start = perf_counter()
    result_tuple = tuple(sorted(_stitch_nested_holes(tuple(result)), key=_fragment_sort_key))
    hole_stitch_s = perf_counter() - stitch_start
    return BoundaryLoopAssemblyResult(
        fragments=result_tuple,
        stats=BoundaryLoopAssemblyStats(
            input_fragment_count=len(fragments),
            boundary_edge_count=boundary_edge_count,
            loop_count=len(result_tuple),
            dropped_zero_area_loop_count=dropped,
            cuda_edge_cancel_s=cuda_edge_cancel_s,
            cuda_edge_cancel_kernel_count=cuda_edge_cancel_kernel_count,
            cuda_edge_cancel_event_elapsed_s=cuda_edge_cancel_event_elapsed_s,
            cuda_edge_cancel_transferred_bytes=cuda_edge_cancel_transferred_bytes,
            native_edge_cancel_s=native_edge_cancel_s,
            cuda_loop_trace_s=cuda_loop_trace_s,
            native_loop_trace_s=native_loop_trace_s,
            python_loop_trace_s=python_loop_trace_s,
            hole_stitch_s=hole_stitch_s,
        ),
    )


def _fragments_from_python_boundary_edges(boundary_edges: tuple[Segment, ...]) -> tuple[list[DiffFragment], int]:
    loops = _trace_loops(boundary_edges)
    result = []
    dropped = 0
    for loop in loops:
        points = _clean_loop(loop)
        if len(points) < 3:
            dropped += 1
            continue
        twice_area = polygon_twice_area(points)
        if twice_area == 0:
            dropped += 1
            continue
        result.append(DiffFragment(points=points, bbox=polygon_bbox(points), twice_area=twice_area))
    return result, dropped


def _try_cuda_boundary_edges_from_fragments(fragments: tuple[DiffFragment, ...]) -> tuple[Any, float, str, int, float, int] | None:
    if len(fragments) < 2 or sum(len(fragment.points) for fragment in fragments) < 512:
        return None
    try:
        from .backends.cuda_ctypes import (
            run_cuda_axis_aligned_boundary_edges_from_fragments,
            run_cuda_split_free_boundary_edges_from_fragments,
        )

        start = perf_counter()
        if _all_edges_axis_aligned(fragments):
            result = run_cuda_axis_aligned_boundary_edges_from_fragments(fragments)
        else:
            result = run_cuda_split_free_boundary_edges_from_fragments(fragments)
        return (
            result.edges,
            perf_counter() - start,
            "cuda",
            result.cuda_kernel_count,
            result.cuda_event_elapsed_s,
            result.transferred_bytes,
        )
    except Exception:
        return None


def _try_native_boundary_edges_from_fragments(fragments: tuple[DiffFragment, ...]) -> tuple[Any, float, str, int, float, int] | None:
    if len(fragments) < 2 or sum(len(fragment.points) for fragment in fragments) < 64:
        return None
    try:
        from .backends.boundary_loop_ctypes import native_boundary_edges_from_fragments

        start = perf_counter()
        edges = native_boundary_edges_from_fragments(fragments).edges
        return edges, perf_counter() - start, "native", 0, 0.0, 0
    except Exception:
        return None


def _all_edges_axis_aligned(fragments: tuple[DiffFragment, ...]) -> bool:
    for fragment in fragments:
        points = _clean_polygon(fragment.points)
        if len(points) < 3:
            continue
        for start, stop in _edges(points):
            if start[0] != stop[0] and start[1] != stop[1]:
                return False
    return True


def _try_cuda_boundary_loop_fragments_array(edges: Any):
    if len(edges) < 512:
        return None
    try:
        from .backends.cuda_ctypes import run_cuda_boundary_loops_from_edges

        return run_cuda_boundary_loops_from_edges(edges)
    except Exception:
        return None


def _try_native_boundary_loop_fragments_array(edges: Any):
    if len(edges) < 64:
        return None
    try:
        from .backends.boundary_loop_ctypes import native_boundary_loops_from_edges

        return native_boundary_loops_from_edges(edges)
    except Exception:
        return None


def _segments_from_edges_array(edges: Any) -> tuple[Segment, ...]:
    return tuple(
        ((int(edge[0]), int(edge[1])), (int(edge[2]), int(edge[3])))
        for edge in edges
    )


def _try_cuda_boundary_loop_fragments(edges: tuple[Segment, ...]):
    if len(edges) < 512:
        return None
    try:
        import numpy as np

        from .backends.cuda_ctypes import run_cuda_boundary_loops_from_edges

        return run_cuda_boundary_loops_from_edges(np.asarray([(a[0], a[1], b[0], b[1]) for a, b in edges], dtype=np.int64))
    except Exception:
        return None


def _boundary_edges(fragments: tuple[DiffFragment, ...]) -> tuple[Segment, ...]:
    counts: dict[Segment, int] = defaultdict(int)
    raw_edges: list[Segment] = []
    for fragment in fragments:
        points = _clean_polygon(fragment.points)
        if len(points) < 3:
            continue
        for edge in _edges(points):
            if edge[0] == edge[1]:
                continue
            raw_edges.append(edge)
    for edge in _split_collinear_edges(tuple(raw_edges)):
        reverse = (edge[1], edge[0])
        if counts.get(reverse, 0):
            counts[reverse] -= 1
            if counts[reverse] == 0:
                del counts[reverse]
        else:
            counts[edge] += 1
    boundary: list[Segment] = []
    for edge, count in counts.items():
        boundary.extend(edge for _ in range(count))
    return tuple(sorted(boundary))


def _stitch_nested_holes(fragments: tuple[DiffFragment, ...]) -> tuple[DiffFragment, ...]:
    if len(fragments) < 2:
        return fragments
    parents: dict[int, int | None] = {}
    for index, fragment in enumerate(fragments):
        parent_candidates = [
            other_index
            for other_index, other in enumerate(fragments)
            if other_index != index and _strictly_contains_fragment(other, fragment)
        ]
        parents[index] = (
            min(parent_candidates, key=lambda candidate: fragments[candidate].twice_area)
            if parent_candidates
            else None
        )
    depths = {index: _containment_depth(index, parents) for index in range(len(fragments))}
    holes_by_outer: dict[int, list[DiffFragment]] = defaultdict(list)
    output_indices: list[int] = []
    for index, depth in depths.items():
        if depth % 2 == 0:
            output_indices.append(index)
        else:
            parent = parents[index]
            if parent is not None:
                holes_by_outer[parent].append(fragments[index])

    stitched: list[DiffFragment] = []
    for index in output_indices:
        fragment = fragments[index]
        points = fragment.points
        unstitched_holes: list[DiffFragment] = []
        for hole in sorted(holes_by_outer.get(index, ()), key=_fragment_sort_key):
            try:
                points = _stitch_hole(points, hole.points)
            except ValueError:
                unstitched_holes.append(hole)
        points = _clean_polygon(points)
        stitched.append(DiffFragment(points=points, bbox=polygon_bbox(points), twice_area=polygon_twice_area(points)))
        stitched.extend(unstitched_holes)
    return tuple(stitched)


def _strictly_contains_fragment(outer: DiffFragment, inner: DiffFragment) -> bool:
    if not (
        outer.bbox[0] <= inner.bbox[0]
        and outer.bbox[1] <= inner.bbox[1]
        and outer.bbox[2] >= inner.bbox[2]
        and outer.bbox[3] >= inner.bbox[3]
    ):
        return False
    if outer.bbox == inner.bbox:
        return False
    return _point_in_polygon(inner.points[0], outer.points)


def _containment_depth(index: int, parents: dict[int, int | None]) -> int:
    depth = 0
    current = parents[index]
    while current is not None:
        depth += 1
        current = parents[current]
    return depth


def _stitch_hole(outer: tuple[Point, ...], hole: tuple[Point, ...]) -> tuple[Point, ...]:
    target_area = abs(polygon_twice_area(outer) - polygon_twice_area(hole))
    candidates = []
    for anchor in sorted(hole, key=lambda point: (point[0], point[1])):
        for bridge in _horizontal_bridge_points(outer, anchor):
            outer_cycle = _cycle_outer_with_bridge(outer, bridge)
            for sequence in (hole, tuple(reversed(hole))):
                sequence_anchor_index = sequence.index(anchor)
                hole_cycle = sequence[sequence_anchor_index:] + sequence[:sequence_anchor_index]
                points = _clean_polygon(outer_cycle + (bridge, anchor) + hole_cycle + (anchor,))
                candidates.append(points)
    if not candidates:
        raise ValueError("could not find a bridge point for nested hole")
    return min(candidates, key=lambda points: (abs(polygon_twice_area(points) - target_area), len(points), points))


def _horizontal_bridge_points(outer: tuple[Point, ...], anchor: Point) -> tuple[Point, ...]:
    anchor_x, anchor_y = anchor
    left_candidates: list[tuple[float, Point]] = []
    right_candidates: list[tuple[float, Point]] = []
    for start, stop in _edges(outer):
        y0, y1 = start[1], stop[1]
        x0, x1 = start[0], stop[0]
        if y0 == y1:
            if y0 == anchor_y:
                for x in (x0, x1):
                    if x < anchor_x:
                        left_candidates.append((float(x), (x, anchor_y)))
                    elif x > anchor_x:
                        right_candidates.append((float(x), (x, anchor_y)))
            continue
        if min(y0, y1) <= anchor_y <= max(y0, y1):
            x = x0 + (anchor_y - y0) * (x1 - x0) / (y1 - y0)
            if x < anchor_x:
                left_candidates.append((x, (round(x), anchor_y)))
            elif x > anchor_x:
                right_candidates.append((x, (round(x), anchor_y)))
    points = []
    if left_candidates:
        points.append(max(left_candidates, key=lambda item: item[0])[1])
    if right_candidates:
        points.append(min(right_candidates, key=lambda item: item[0])[1])
    return tuple(dict.fromkeys(points))


def _cycle_outer_with_bridge(outer: tuple[Point, ...], bridge: Point) -> tuple[Point, ...]:
    for index, point in enumerate(outer):
        if point == bridge:
            return outer[index:] + outer[:index]
    for index, (start, stop) in enumerate(_edges(outer)):
        if _point_on_segment(bridge, start, stop):
            return (bridge,) + outer[(index + 1) :] + outer[: index + 1]
    raise ValueError("bridge point is not on the outer loop")


def _point_on_segment(point: Point, start: Point, stop: Point) -> bool:
    if not (
        min(start[0], stop[0]) <= point[0] <= max(start[0], stop[0])
        and min(start[1], stop[1]) <= point[1] <= max(start[1], stop[1])
    ):
        return False
    return _collinear(start, point, stop)


def _point_in_polygon(point: Point, polygon: tuple[Point, ...]) -> bool:
    x, y = point
    inside = False
    for start, stop in _edges(polygon):
        if (start[1] > y) != (stop[1] > y):
            x_intersection = start[0] + (y - start[1]) * (stop[0] - start[0]) / (stop[1] - start[1])
            if x_intersection > x:
                inside = not inside
    return inside


def _split_collinear_edges(edges: tuple[Segment, ...]) -> tuple[Segment, ...]:
    by_line: dict[tuple[int, int, int], list[Segment]] = defaultdict(list)
    points_by_line: dict[tuple[int, int, int], dict[int, Point]] = defaultdict(dict)
    for edge in edges:
        key = _line_key(edge)
        by_line[key].append(edge)
        points_by_line[key][_projection(edge[0], key)] = edge[0]
        points_by_line[key][_projection(edge[1], key)] = edge[1]

    split_edges: list[Segment] = []
    for key, line_edges in by_line.items():
        line_points = sorted(points_by_line[key])
        for start, end in line_edges:
            start_t = _projection(start, key)
            end_t = _projection(end, key)
            low = min(start_t, end_t)
            high = max(start_t, end_t)
            breakpoints = [value for value in line_points if low <= value <= high]
            if start_t > end_t:
                breakpoints.reverse()
            for left_t, right_t in zip(breakpoints, breakpoints[1:]):
                if left_t == right_t:
                    continue
                split_edges.append((points_by_line[key][left_t], points_by_line[key][right_t]))
    return tuple(split_edges)


def _line_key(edge: Segment) -> tuple[int, int, int]:
    (x0, y0), (x1, y1) = edge
    dx = x1 - x0
    dy = y1 - y0
    scale = gcd(abs(dx), abs(dy))
    if scale == 0:
        raise ValueError("zero-length segment has no line key")
    ux = dx // scale
    uy = dy // scale
    if ux < 0 or (ux == 0 and uy < 0):
        ux = -ux
        uy = -uy
    c = -uy * x0 + ux * y0
    return ux, uy, c


def _projection(point: Point, key: tuple[int, int, int]) -> int:
    ux, uy, _ = key
    return point[0] * ux + point[1] * uy


def _trace_loops(edges: tuple[Segment, ...]) -> tuple[tuple[Point, ...], ...]:
    outgoing: dict[Point, list[Point]] = defaultdict(list)
    for start, end in edges:
        outgoing[start].append(end)
    for ends in outgoing.values():
        ends.sort()

    remaining: set[Segment] = set(edges)
    loops: list[tuple[Point, ...]] = []
    while remaining:
        start, end = min(remaining)
        remaining.remove((start, end))
        loop = [start, end]
        previous = start
        current = end
        while current != start:
            next_point = _select_next(previous, current, outgoing.get(current, ()), remaining)
            if next_point is None:
                break
            remaining.remove((current, next_point))
            loop.append(next_point)
            previous, current = current, next_point
        if loop[-1] == start:
            loop.pop()
        loops.append(tuple(loop))
    return tuple(sorted(loops))


def _select_next(
    previous: Point,
    current: Point,
    candidates: list[Point] | tuple[Point, ...],
    remaining: set[Segment],
) -> Point | None:
    available = [candidate for candidate in candidates if (current, candidate) in remaining]
    if not available:
        return None
    if len(available) == 1:
        return available[0]
    in_vec = (current[0] - previous[0], current[1] - previous[1])
    return min(available, key=lambda candidate: (_turn_key(in_vec, (candidate[0] - current[0], candidate[1] - current[1])), candidate))


def _turn_key(in_vec: Point, out_vec: Point) -> tuple[int, int, int]:
    # Prefer continuing the same contour with the smallest right-hand turn.
    cross = in_vec[0] * out_vec[1] - in_vec[1] * out_vec[0]
    dot = in_vec[0] * out_vec[0] + in_vec[1] * out_vec[1]
    if cross < 0:
        turn_class = 0
    elif cross == 0 and dot > 0:
        turn_class = 1
    elif cross > 0:
        turn_class = 2
    else:
        turn_class = 3
    return turn_class, -abs(cross), -dot


def _edges(points: tuple[Point, ...]) -> tuple[Segment, ...]:
    return tuple((point, points[(index + 1) % len(points)]) for index, point in enumerate(points))


def _clean_polygon(points: tuple[Point, ...]) -> tuple[Point, ...]:
    cleaned: list[Point] = []
    for point in points:
        if not cleaned or cleaned[-1] != point:
            cleaned.append(point)
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    return tuple(cleaned)


def _clean_loop(points: tuple[Point, ...]) -> tuple[Point, ...]:
    cleaned = list(_clean_polygon(points))
    changed = True
    while changed and len(cleaned) >= 3:
        changed = False
        next_points: list[Point] = []
        for index, point in enumerate(cleaned):
            previous = cleaned[index - 1]
            following = cleaned[(index + 1) % len(cleaned)]
            if _collinear(previous, point, following):
                changed = True
                continue
            next_points.append(point)
        cleaned = next_points
    return tuple(cleaned)


def _collinear(left: Point, middle: Point, right: Point) -> bool:
    return (middle[0] - left[0]) * (right[1] - middle[1]) == (middle[1] - left[1]) * (right[0] - middle[0])


def _fragment_sort_key(fragment: DiffFragment) -> tuple[tuple[int, int, int, int], int, tuple[Point, ...]]:
    return fragment.bbox, fragment.twice_area, fragment.points
