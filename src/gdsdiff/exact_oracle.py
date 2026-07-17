from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

from .config import LayerSpec
from .geometry import PolygonBuffer, polygon_bbox, polygon_twice_area
from .grid import snap_to_grid
from .multiprocessing_support import fork_context

_WORKER_OLD_BY_LAYER = None
_WORKER_NEW_BY_LAYER = None
_WORKER_PRECISION = 1.0
_WORKER_BUFFER = None


@dataclass(frozen=True)
class DiffFragment:
    points: tuple[tuple[int, int], ...]
    bbox: tuple[int, int, int, int]
    twice_area: int

    @property
    def area_dbu2_string(self) -> str:
        return format_twice_area(self.twice_area)


@dataclass(frozen=True)
class LayerExactDiff:
    layer: LayerSpec
    old_minus_new: tuple[DiffFragment, ...] = field(default_factory=tuple)
    new_minus_old: tuple[DiffFragment, ...] = field(default_factory=tuple)
    xor: tuple[DiffFragment, ...] = field(default_factory=tuple)

    @property
    def old_minus_new_twice_area(self) -> int:
        return sum(fragment.twice_area for fragment in self.old_minus_new)

    @property
    def new_minus_old_twice_area(self) -> int:
        return sum(fragment.twice_area for fragment in self.new_minus_old)

    @property
    def xor_twice_area(self) -> int:
        return sum(fragment.twice_area for fragment in self.xor)


@dataclass(frozen=True)
class ExactOracleResult:
    layers: tuple[LayerExactDiff, ...]

    @property
    def equivalent(self) -> bool:
        return all(layer.xor_twice_area == 0 for layer in self.layers)


@dataclass(frozen=True)
class ExactLayerTask:
    layer: LayerSpec
    precision: float
    old_points: tuple[tuple[tuple[int, int], ...], ...] | None = None
    new_points: tuple[tuple[tuple[int, int], ...], ...] | None = None
    clip_bbox: tuple[int, int, int, int] | None = None
    partition: str = "layer"


def run_full_exact_oracle(
    old_buffer: PolygonBuffer,
    new_buffer: PolygonBuffer,
    *,
    precision: float = 1.0,
    layers: set[LayerSpec] | None = None,
    workers: int | None = None,
    component_partition: bool = False,
    component_executor: Callable[[tuple[ExactLayerTask, ...]], tuple[LayerExactDiff, ...]] | None = None,
    force_small_component_partition: bool = False,
) -> ExactOracleResult:
    old_by_layer = _buffer_points_by_layer(old_buffer)
    new_by_layer = _buffer_points_by_layer(new_buffer)
    selected_layers = tuple(sorted((set(old_by_layer) | set(new_by_layer)) if layers is None else set(layers)))
    worker_count = _resolve_worker_count(workers, len(selected_layers))
    tasks_by_layer = {
        layer: _layer_task_specs(
            layer,
            old_by_layer.get(layer, ()),
            new_by_layer.get(layer, ()),
            precision,
            worker_count,
            component_partition=component_partition,
            prefer_rectangle_pairs=component_executor is not None,
            force_small_component_partition=force_small_component_partition,
        )
        for layer in selected_layers
    }
    flat_tasks = tuple(task for layer_tasks in tasks_by_layer.values() for task in layer_tasks)
    if component_executor is not None:
        stripe_diffs = component_executor(flat_tasks) if flat_tasks else ()
        by_layer: dict[LayerSpec, list[LayerExactDiff]] = {layer: [] for layer in selected_layers}
        for diff in stripe_diffs:
            by_layer[diff.layer].append(diff)
        diffs = tuple(
            _merge_layer_stripes(by_layer[layer]) if by_layer[layer] else LayerExactDiff(layer=layer)
            for layer in selected_layers
        )
    elif worker_count <= 1:
        diffs = tuple(
            _merge_layer_stripes(_exact_layer_diff_task(task) for task in tasks_by_layer[layer])
            if tasks_by_layer[layer]
            else LayerExactDiff(layer=layer)
            for layer in selected_layers
        )
    else:
        context = fork_context()
        global _WORKER_OLD_BY_LAYER, _WORKER_NEW_BY_LAYER, _WORKER_PRECISION
        _WORKER_OLD_BY_LAYER = old_by_layer
        _WORKER_NEW_BY_LAYER = new_by_layer
        _WORKER_PRECISION = precision
        try:
            if context is None:
                stripe_diffs = tuple(_exact_layer_diff_task(task) for task in flat_tasks)
            else:
                try:
                    with context.Pool(processes=worker_count) as pool:
                        stripe_diffs = tuple(pool.map(_exact_layer_diff_task, flat_tasks, chunksize=1))
                except OSError:
                    stripe_diffs = tuple(_exact_layer_diff_task(task) for task in flat_tasks)
        finally:
            _WORKER_OLD_BY_LAYER = None
            _WORKER_NEW_BY_LAYER = None
        by_layer: dict[LayerSpec, list[LayerExactDiff]] = {layer: [] for layer in selected_layers}
        for diff in stripe_diffs:
            by_layer[diff.layer].append(diff)
        diffs = tuple(
            _merge_layer_stripes(by_layer[layer]) if by_layer[layer] else LayerExactDiff(layer=layer)
            for layer in selected_layers
        )
    return ExactOracleResult(tuple(sorted(diffs, key=lambda diff: diff.layer)))


def _resolve_worker_count(workers: int | None, layer_count: int) -> int:
    if layer_count <= 0:
        return 1
    if workers is None:
        workers = os.cpu_count() or 1
    return max(1, int(workers))


def _layer_task_specs(
    layer: LayerSpec,
    old_points: list[tuple[tuple[int, int], ...]] | tuple[tuple[tuple[int, int], ...], ...],
    new_points: list[tuple[tuple[int, int], ...]] | tuple[tuple[tuple[int, int], ...], ...],
    precision: float,
    worker_count: int,
    component_partition: bool = False,
    prefer_rectangle_pairs: bool = False,
    force_small_component_partition: bool = False,
) -> tuple[ExactLayerTask, ...]:
    old_tuple = tuple(old_points)
    new_tuple = tuple(new_points)
    total = len(old_tuple) + len(new_tuple)
    if not force_small_component_partition and (worker_count <= 1 or total < 2048):
        return (ExactLayerTask(layer, precision, old_tuple, new_tuple),)
    if component_partition:
        if prefer_rectangle_pairs:
            rectangle_pair_tasks = _rectangle_pair_task_specs(layer, old_tuple, new_tuple, precision)
            if rectangle_pair_tasks is not None:
                return rectangle_pair_tasks
            sparse_rectangle_tasks = _sparse_rectangle_residual_task_specs(layer, old_tuple, new_tuple, precision)
            if sparse_rectangle_tasks is not None:
                return sparse_rectangle_tasks
        component_tasks, component_applicable = _component_task_specs_result(layer, old_tuple, new_tuple, precision)
        if component_applicable and (prefer_rectangle_pairs or len(component_tasks) > 1 or not component_tasks):
            return component_tasks
    if worker_count <= 1 or total < 2048:
        return (ExactLayerTask(layer, precision, old_tuple, new_tuple),)
    bbox = _points_bbox(old_tuple + new_tuple)
    if bbox is None:
        return (ExactLayerTask(layer, precision, old_tuple, new_tuple),)
    stripes = _stripe_tasks(layer, old_tuple, new_tuple, precision, bbox, worker_count)
    return stripes or (ExactLayerTask(layer, precision, old_tuple, new_tuple),)


def _component_task_specs(
    layer: LayerSpec,
    old_tuple: tuple[tuple[tuple[int, int], ...], ...],
    new_tuple: tuple[tuple[tuple[int, int], ...], ...],
    precision: float,
) -> tuple[ExactLayerTask, ...]:
    return _component_task_specs_result(layer, old_tuple, new_tuple, precision)[0]


def _component_task_specs_result(
    layer: LayerSpec,
    old_tuple: tuple[tuple[tuple[int, int], ...], ...],
    new_tuple: tuple[tuple[tuple[int, int], ...], ...],
    precision: float,
) -> tuple[tuple[ExactLayerTask, ...], bool]:
    components = _independent_bbox_components(old_tuple, new_tuple)
    if len(components) <= 1:
        return (), False
    tasks = []
    for old_component, new_component in components:
        if _is_identical_component_source_set(old_component, new_component):
            continue
        tasks.append(ExactLayerTask(layer, precision, old_component, new_component, partition="bbox-component"))
    return tuple(tasks), True


def _rectangle_pair_task_specs(
    layer: LayerSpec,
    old_tuple: tuple[tuple[tuple[int, int], ...], ...],
    new_tuple: tuple[tuple[tuple[int, int], ...], ...],
    precision: float,
) -> tuple[ExactLayerTask, ...] | None:
    if len(old_tuple) != len(new_tuple) or not old_tuple:
        return None
    tasks = []
    for old_points, new_points in zip(old_tuple, new_tuple):
        old_rect = _axis_aligned_rectangle_bbox(old_points)
        new_rect = _axis_aligned_rectangle_bbox(new_points)
        if old_rect is None or new_rect is None:
            return None
        if old_rect == new_rect:
            continue
        tasks.append(ExactLayerTask(layer, precision, (old_points,), (new_points,), partition="rectangle-pair"))
    return tuple(tasks)


def _sparse_rectangle_residual_task_specs(
    layer: LayerSpec,
    old_tuple: tuple[tuple[tuple[int, int], ...], ...],
    new_tuple: tuple[tuple[tuple[int, int], ...], ...],
    precision: float,
) -> tuple[ExactLayerTask, ...] | None:
    old_rects, old_non_rects = _split_rectangles(old_tuple)
    new_rects, new_non_rects = _split_rectangles(new_tuple)
    if not old_rects and not new_rects:
        return None
    common_rects = set(old_rects) & set(new_rects)
    if not common_rects:
        return None
    old_residual: list[tuple[tuple[int, int], ...]] = list(old_non_rects)
    new_residual: list[tuple[tuple[int, int], ...]] = list(new_non_rects)
    residual_bboxes = [polygon_bbox(points) for points in old_non_rects + new_non_rects]
    for rect, old_polygons in old_rects.items():
        if rect not in new_rects:
            residual_bboxes.append(rect)
            old_residual.append(old_polygons[0])
    for rect, new_polygons in new_rects.items():
        if rect not in old_rects:
            residual_bboxes.append(rect)
            new_residual.append(new_polygons[0])
    for rect, old_polygons in old_rects.items():
        if rect in new_rects and _bbox_overlaps_any(rect, residual_bboxes):
            old_residual.append(old_polygons[0])
    for rect, new_polygons in new_rects.items():
        if rect in old_rects and _bbox_overlaps_any(rect, residual_bboxes):
            new_residual.append(new_polygons[0])
    if len(old_residual) == len(old_tuple) and len(new_residual) == len(new_tuple):
        return None
    if not old_residual and not new_residual:
        return ()
    component_tasks, component_applicable = _component_task_specs_result(layer, tuple(old_residual), tuple(new_residual), precision)
    if component_applicable:
        return component_tasks
    return (ExactLayerTask(layer, precision, tuple(old_residual), tuple(new_residual), partition="rectangle-residual"),)


def _split_rectangles(
    polygons: tuple[tuple[tuple[int, int], ...], ...],
) -> tuple[dict[tuple[int, int, int, int], list[tuple[tuple[int, int], ...]]], list[tuple[tuple[int, int], ...]]]:
    rects: dict[tuple[int, int, int, int], list[tuple[tuple[int, int], ...]]] = {}
    non_rects: list[tuple[tuple[int, int], ...]] = []
    for points in polygons:
        rect = _axis_aligned_rectangle_bbox(points)
        if rect is None:
            non_rects.append(points)
        else:
            rects.setdefault(rect, []).append(points)
    return rects, non_rects


def _bbox_overlaps_any(
    bbox: tuple[int, int, int, int],
    others: list[tuple[int, int, int, int]],
) -> bool:
    return any(_bbox_overlaps_closed(bbox, other) for other in others)


def _is_axis_aligned_rectangle_points(points: tuple[tuple[int, int], ...]) -> bool:
    return _axis_aligned_rectangle_bbox(points) is not None


def _axis_aligned_rectangle_bbox(points: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int] | None:
    if len(points) != 4:
        return None
    xs = {point[0] for point in points}
    ys = {point[1] for point in points}
    if len(xs) != 2 or len(ys) != 2:
        return None
    x0, x1 = sorted(xs)
    y0, y1 = sorted(ys)
    if set(points) != {(x0, y0), (x0, y1), (x1, y0), (x1, y1)}:
        return None
    return x0, y0, x1, y1


def _is_identical_single_rectangle_component(
    old_component: tuple[tuple[tuple[int, int], ...], ...],
    new_component: tuple[tuple[tuple[int, int], ...], ...],
) -> bool:
    if len(old_component) != 1 or len(new_component) != 1:
        return False
    old_rect = _axis_aligned_rectangle_bbox(old_component[0])
    new_rect = _axis_aligned_rectangle_bbox(new_component[0])
    return old_rect is not None and old_rect == new_rect


def _is_identical_component_source_set(
    old_component: tuple[tuple[tuple[int, int], ...], ...],
    new_component: tuple[tuple[tuple[int, int], ...], ...],
) -> bool:
    if not old_component and not new_component:
        return True
    if _is_identical_single_rectangle_component(old_component, new_component):
        return True
    return {_canonical_polygon_key(points) for points in old_component} == {
        _canonical_polygon_key(points) for points in new_component
    }


def _canonical_polygon_key(points: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    cleaned = _clean_polygon_points(points)
    if len(cleaned) < 2:
        return cleaned
    candidates = []
    for sequence in (cleaned, tuple(reversed(cleaned))):
        for index in range(len(sequence)):
            candidates.append(sequence[index:] + sequence[:index])
    return min(candidates)


def _clean_polygon_points(points: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    cleaned: list[tuple[int, int]] = []
    for point in points:
        if not cleaned or cleaned[-1] != point:
            cleaned.append(point)
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    return tuple(cleaned)


def _independent_bbox_components(
    old_points: tuple[tuple[tuple[int, int], ...], ...],
    new_points: tuple[tuple[tuple[int, int], ...], ...],
) -> tuple[tuple[tuple[tuple[tuple[int, int], ...], ...], tuple[tuple[tuple[int, int], ...], ...]], ...]:
    records: list[tuple[int, int, tuple[int, int, int, int], tuple[tuple[int, int], ...]]] = []
    for side, polygons in ((0, old_points), (1, new_points)):
        for index, points in enumerate(polygons):
            records.append((side, index, polygon_bbox(points), points))
    if not records:
        return ()
    order = sorted(range(len(records)), key=lambda idx: records[idx][2][0])
    parent = list(range(len(records)))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    active: list[int] = []
    for idx in order:
        bbox = records[idx][2]
        active = [other for other in active if records[other][2][2] >= bbox[0]]
        for other in active:
            if _bbox_overlaps_closed(bbox, records[other][2]):
                union(idx, other)
        active.append(idx)
    grouped: dict[int, tuple[list[tuple[tuple[int, int], ...]], list[tuple[tuple[int, int], ...]]]] = {}
    for idx, record in enumerate(records):
        root = find(idx)
        old_group, new_group = grouped.setdefault(root, ([], []))
        if record[0] == 0:
            old_group.append(record[3])
        else:
            new_group.append(record[3])
    return tuple((tuple(old_group), tuple(new_group)) for old_group, new_group in grouped.values())


def _points_bbox(polygons: tuple[tuple[tuple[int, int], ...], ...]) -> tuple[int, int, int, int] | None:
    if not polygons:
        return None
    bboxes = [polygon_bbox(points) for points in polygons if points]
    if not bboxes:
        return None
    return (
        min(bbox[0] for bbox in bboxes),
        min(bbox[1] for bbox in bboxes),
        max(bbox[2] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes),
    )


def _stripe_tasks(
    layer: LayerSpec,
    old_tuple: tuple[tuple[tuple[int, int], ...], ...],
    new_tuple: tuple[tuple[tuple[int, int], ...], ...],
    precision: float,
    bbox: tuple[int, int, int, int],
    worker_count: int,
) -> tuple[ExactLayerTask, ...]:
    x0, y0, x1, y1 = bbox
    if x0 == x1:
        return ()
    stripe_count = worker_count
    width = x1 - x0 + 1
    tasks = []
    for index in range(stripe_count):
        sx0 = x0 + (width * index) // stripe_count
        sx1 = x0 + (width * (index + 1)) // stripe_count - 1
        if index == stripe_count - 1:
            sx1 = x1
        core = (sx0, y0, sx1, y1)
        tasks.append(ExactLayerTask(layer, precision, clip_bbox=core, partition="stripe"))
    return tuple(tasks)


def _bbox_overlaps_closed(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> bool:
    return not (left[2] < right[0] or right[2] < left[0] or left[3] < right[1] or right[3] < left[1])


def _exact_layer_diff_task(
    task: ExactLayerTask,
) -> LayerExactDiff:
    import gdstk

    layer = task.layer
    if task.old_points is None or task.new_points is None:
        if _WORKER_OLD_BY_LAYER is None or _WORKER_NEW_BY_LAYER is None:
            raise RuntimeError("exact oracle worker globals are not initialized")
        old_points = tuple(_WORKER_OLD_BY_LAYER.get(layer, ()))
        new_points = tuple(_WORKER_NEW_BY_LAYER.get(layer, ()))
        precision = _WORKER_PRECISION
    else:
        old_points = task.old_points
        new_points = task.new_points
        precision = task.precision
    clip_bbox = task.clip_bbox
    if clip_bbox is not None:
        old_points = tuple(points for points in old_points if _bbox_overlaps_closed(polygon_bbox(points), clip_bbox))
        new_points = tuple(points for points in new_points if _bbox_overlaps_closed(polygon_bbox(points), clip_bbox))
        if not old_points and not new_points:
            return LayerExactDiff(layer=layer)
    origin = _layer_origin(old_points, new_points)
    old_polys = _to_gdstk_polygons(old_points, layer, origin)
    new_polys = _to_gdstk_polygons(new_points, layer, origin)
    old_minus = gdstk.boolean(old_polys, new_polys, "not", precision=precision) or []
    new_minus = gdstk.boolean(new_polys, old_polys, "not", precision=precision) or []
    if clip_bbox is not None:
        clip_rect = [_bbox_rectangle(clip_bbox, layer, origin)]
        old_minus = gdstk.boolean(old_minus, clip_rect, "and", precision=precision) or []
        new_minus = gdstk.boolean(new_minus, clip_rect, "and", precision=precision) or []
    xor = gdstk.boolean(old_minus, new_minus, "or", precision=precision) or []
    old_fragments, new_fragments = _cancel_symmetric_direction_fragments(
        _fragments_from_gdstk(old_minus, origin),
        _fragments_from_gdstk(new_minus, origin),
    )
    return LayerExactDiff(
        layer=layer,
        old_minus_new=old_fragments,
        new_minus_old=new_fragments,
        xor=tuple(sorted(old_fragments + new_fragments, key=_fragment_sort_key)),
    )


def _merge_layer_stripes(stripes) -> LayerExactDiff:
    stripes = tuple(stripes)
    if not stripes:
        raise ValueError("cannot merge empty layer stripe list")
    if len(stripes) == 1:
        return stripes[0]
    layer = stripes[0].layer
    old_minus_points = tuple(fragment.points for stripe in stripes for fragment in stripe.old_minus_new)
    new_minus_points = tuple(fragment.points for stripe in stripes for fragment in stripe.new_minus_old)
    old_minus = _union_fragments(old_minus_points, layer)
    new_minus = _union_fragments(new_minus_points, layer)
    old_minus, new_minus = _cancel_symmetric_direction_fragments(old_minus, new_minus)
    xor = _union_fragments(tuple(fragment.points for fragment in old_minus + new_minus), layer)
    return LayerExactDiff(layer=layer, old_minus_new=old_minus, new_minus_old=new_minus, xor=xor)


def _union_fragments(polygons: tuple[tuple[tuple[int, int], ...], ...], layer: LayerSpec) -> tuple[DiffFragment, ...]:
    if not polygons:
        return ()
    import gdstk

    origin = _layer_origin(polygons)
    merged = gdstk.boolean(_to_gdstk_polygons(polygons, layer, origin), [], "or", precision=1.0) or []
    return _fragments_from_gdstk(merged, origin)


def _cancel_symmetric_direction_fragments(
    old_fragments: tuple[DiffFragment, ...],
    new_fragments: tuple[DiffFragment, ...],
) -> tuple[tuple[DiffFragment, ...], tuple[DiffFragment, ...]]:
    if not old_fragments or not new_fragments:
        return old_fragments, new_fragments
    old_counts: dict[tuple[tuple[int, int, int, int], int, tuple[tuple[int, int], ...]], int] = {}
    new_counts: dict[tuple[tuple[int, int, int, int], int, tuple[tuple[int, int], ...]], int] = {}
    for fragment in old_fragments:
        old_counts[_direction_fragment_key(fragment)] = old_counts.get(_direction_fragment_key(fragment), 0) + 1
    for fragment in new_fragments:
        new_counts[_direction_fragment_key(fragment)] = new_counts.get(_direction_fragment_key(fragment), 0) + 1
    old_out = []
    for fragment in old_fragments:
        key = _direction_fragment_key(fragment)
        if new_counts.get(key, 0):
            new_counts[key] -= 1
        else:
            old_out.append(fragment)
    new_out = []
    for fragment in new_fragments:
        key = _direction_fragment_key(fragment)
        if old_counts.get(key, 0):
            old_counts[key] -= 1
        else:
            new_out.append(fragment)
    return tuple(old_out), tuple(new_out)


def _direction_fragment_key(fragment: DiffFragment) -> tuple[tuple[int, int, int, int], int, tuple[tuple[int, int], ...]]:
    return fragment.bbox, fragment.twice_area, _canonical_polygon_key(fragment.points)


def _fragment_sort_key(fragment: DiffFragment) -> tuple[tuple[int, int, int, int], int, tuple[tuple[int, int], ...]]:
    return fragment.bbox, fragment.twice_area, fragment.points


def _bbox_rectangle(bbox: tuple[int, int, int, int], layer: LayerSpec, origin: tuple[int, int]) -> gdstk.Polygon:
    import gdstk

    ox, oy = origin
    x0, y0, x1, y1 = bbox
    return gdstk.rectangle((x0 - ox, y0 - oy), (x1 + 1 - ox, y1 + 1 - oy), layer=layer.layer, datatype=layer.datatype)




def format_twice_area(twice_area: int) -> str:
    whole, remainder = divmod(abs(twice_area), 2)
    sign = "-" if twice_area < 0 else ""
    if remainder:
        return f"{sign}{whole}.5"
    return f"{sign}{whole}"


def _buffer_points_by_layer(buffer: PolygonBuffer) -> dict[LayerSpec, list[tuple[tuple[int, int], ...]]]:
    worker_count = _resolve_worker_count(None, buffer.polygon_count)
    if worker_count <= 1 or buffer.polygon_count < 4096:
        return _buffer_points_by_layer_range(buffer, 0, buffer.polygon_count)

    chunk_count = max(worker_count, worker_count * 8)
    chunk_size = max(1, (buffer.polygon_count + chunk_count - 1) // chunk_count)
    chunks = tuple((start, min(start + chunk_size, buffer.polygon_count)) for start in range(0, buffer.polygon_count, chunk_size))
    context = fork_context()
    if context is None:
        return _buffer_points_by_layer_range(buffer, 0, buffer.polygon_count)

    global _WORKER_BUFFER
    _WORKER_BUFFER = buffer
    try:
        try:
            with context.Pool(processes=worker_count) as pool:
                chunk_results = pool.map(_buffer_points_by_layer_worker, chunks, chunksize=1)
        except OSError:
            return _buffer_points_by_layer_range(buffer, 0, buffer.polygon_count)
    finally:
        _WORKER_BUFFER = None

    out: dict[LayerSpec, list[tuple[tuple[int, int], ...]]] = {}
    for chunk in chunk_results:
        for layer, polygons in chunk.items():
            out.setdefault(layer, []).extend(polygons)
    return out


def _buffer_points_by_layer_worker(chunk: tuple[int, int]) -> dict[LayerSpec, list[tuple[tuple[int, int], ...]]]:
    if _WORKER_BUFFER is None:
        raise RuntimeError("exact oracle buffer worker globals are not initialized")
    start, stop = chunk
    return _buffer_points_by_layer_range(_WORKER_BUFFER, start, stop)


def _buffer_points_by_layer_range(
    buffer: PolygonBuffer,
    start: int,
    stop: int,
) -> dict[LayerSpec, list[tuple[tuple[int, int], ...]]]:
    out: dict[LayerSpec, list[tuple[tuple[int, int], ...]]] = {}
    for index in range(start, stop):
        layer = buffer.layer_table.layers[int(buffer.layer_ids[index])]
        vertices = tuple((int(x), int(y)) for x, y in buffer.polygon_vertices(index))
        out.setdefault(layer, []).append(vertices)
    return out


def _layer_origin(*collections: list[tuple[tuple[int, int], ...]] | tuple[tuple[tuple[int, int], ...], ...]) -> tuple[int, int]:
    xs = []
    ys = []
    for polygons in collections:
        for points in polygons:
            xs.extend(point[0] for point in points)
            ys.extend(point[1] for point in points)
    if not xs:
        return 0, 0
    return min(xs), min(ys)


def _to_gdstk_polygons(
    polygons: list[tuple[tuple[int, int], ...]] | tuple[tuple[tuple[int, int], ...], ...],
    layer: LayerSpec,
    origin: tuple[int, int],
) -> list[gdstk.Polygon]:
    import gdstk

    ox, oy = origin
    return [
        gdstk.Polygon([(x - ox, y - oy) for x, y in points], layer=layer.layer, datatype=layer.datatype)
        for points in polygons
    ]


def _fragments_from_gdstk(polygons: list[gdstk.Polygon], origin: tuple[int, int]) -> tuple[DiffFragment, ...]:
    ox, oy = origin
    fragments = []
    for polygon in polygons:
        points = tuple((snap_to_grid(point[0], 1) + ox, snap_to_grid(point[1], 1) + oy) for point in polygon.points)
        if not points:
            continue
        fragments.append(DiffFragment(points=points, bbox=polygon_bbox(points), twice_area=polygon_twice_area(points)))
    return tuple(fragments)
