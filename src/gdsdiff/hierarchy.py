from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable

from .config import LayerSpec
from .geometry import normalize_polygon_points
from .spatial import BBox, bbox_union


@dataclass(frozen=True)
class CellIdentity:
    digest: str
    local_polygon_count: int
    reference_count: int


@dataclass(frozen=True)
class ReferenceChange:
    kind: str
    bbox: BBox
    old_child_digest: str | None = None
    new_child_digest: str | None = None


@dataclass(frozen=True)
class HierarchyCompareResult:
    old_top: str
    new_top: str
    old_digest: str
    new_digest: str
    identical: bool
    matched_subtree_count: int
    changed_references: tuple[ReferenceChange, ...] = field(default_factory=tuple)


def compare_hierarchy(
    old_lib: gdstk.Library,
    new_lib: gdstk.Library,
    *,
    old_top: str | None = None,
    new_top: str | None = None,
) -> HierarchyCompareResult:
    old_cell = _select_top(old_lib, old_top)
    new_cell = _select_top(new_lib, new_top)
    old_cache: dict[int, CellIdentity] = {}
    new_cache: dict[int, CellIdentity] = {}
    old_identity = canonical_cell_identity(old_cell, old_cache)
    new_identity = canonical_cell_identity(new_cell, new_cache)
    matched: set[tuple[str, str]] = set()
    changes: list[ReferenceChange] = []
    _compare_cells(old_cell, new_cell, old_cache, new_cache, matched, changes)
    return HierarchyCompareResult(
        old_top=old_cell.name,
        new_top=new_cell.name,
        old_digest=old_identity.digest,
        new_digest=new_identity.digest,
        identical=old_identity.digest == new_identity.digest,
        matched_subtree_count=len(matched),
        changed_references=tuple(sorted(changes, key=lambda item: (item.kind, item.bbox, item.old_child_digest or "", item.new_child_digest or ""))),
    )


def canonical_cell_identity(cell: gdstk.Cell, cache: dict[int, CellIdentity] | None = None) -> CellIdentity:
    cache = cache if cache is not None else {}
    return _canonical_cell_identity(cell, cache, set())


def _canonical_cell_identity(cell: gdstk.Cell, cache: dict[int, CellIdentity], visiting: set[int]) -> CellIdentity:
    key = id(cell)
    if key in cache:
        return cache[key]
    if key in visiting:
        raise ValueError(f"hierarchy cycle detected at cell {cell.name!r}")
    visiting.add(key)
    polygon_records = sorted(_polygon_record(polygon) for polygon in cell.polygons)
    reference_records = []
    for reference in cell.references:
        child = _reference_cell(reference)
        child_identity = _canonical_cell_identity(child, cache, visiting)
        reference_records.append((_reference_transform_record(reference), child_identity.digest))
    reference_records.sort()
    digest = _digest_records(("cell", tuple(polygon_records), tuple(reference_records)))
    visiting.remove(key)
    identity = CellIdentity(
        digest=digest,
        local_polygon_count=len(polygon_records),
        reference_count=len(reference_records),
    )
    cache[key] = identity
    return identity


def _compare_cells(
    old_cell: gdstk.Cell,
    new_cell: gdstk.Cell,
    old_cache: dict[int, CellIdentity],
    new_cache: dict[int, CellIdentity],
    matched: set[tuple[str, str]],
    changes: list[ReferenceChange],
) -> None:
    old_identity = old_cache[id(old_cell)]
    new_identity = new_cache[id(new_cell)]
    if old_identity.digest == new_identity.digest:
        matched.add((old_identity.digest, new_identity.digest))
        return
    old_refs = _references_by_transform(old_cell, old_cache)
    new_refs = _references_by_transform(new_cell, new_cache)
    for transform in sorted(set(old_refs) | set(new_refs)):
        old_group = old_refs.get(transform, [])
        new_group = new_refs.get(transform, [])
        paired_count = min(len(old_group), len(new_group))
        for index in range(paired_count):
            old_ref, old_digest = old_group[index]
            new_ref, new_digest = new_group[index]
            if old_digest == new_digest:
                matched.add((old_digest, new_digest))
                continue
            changes.append(
                ReferenceChange(
                    kind="changed",
                    bbox=_reference_bbox_union((old_ref, new_ref)),
                    old_child_digest=old_digest,
                    new_child_digest=new_digest,
                )
            )
            _compare_cells(_reference_cell(old_ref), _reference_cell(new_ref), old_cache, new_cache, matched, changes)
        for old_ref, old_digest in old_group[paired_count:]:
            changes.append(ReferenceChange(kind="removed", bbox=_reference_bbox(old_ref), old_child_digest=old_digest))
        for new_ref, new_digest in new_group[paired_count:]:
            changes.append(ReferenceChange(kind="added", bbox=_reference_bbox(new_ref), new_child_digest=new_digest))


def _references_by_transform(cell: gdstk.Cell, cache: dict[int, CellIdentity]) -> dict[tuple[object, ...], list[tuple[gdstk.Reference, str]]]:
    grouped: dict[tuple[object, ...], list[tuple[gdstk.Reference, str]]] = {}
    for reference in cell.references:
        child = _reference_cell(reference)
        digest = cache[id(child)].digest
        grouped.setdefault(_reference_transform_record(reference), []).append((reference, digest))
    for group in grouped.values():
        group.sort(key=lambda item: item[1])
    return grouped


def _polygon_record(polygon: gdstk.Polygon) -> tuple[object, ...]:
    points = normalize_polygon_points([(_float_key(point[0]), _float_key(point[1])) for point in polygon.points])
    if points:
        rotations = [points[index:] + points[:index] for index in range(len(points))]
        reverse = tuple(reversed(points))
        rotations.extend(reverse[index:] + reverse[:index] for index in range(len(reverse)))
        points = min(rotations)
    return (LayerSpec(polygon.layer, polygon.datatype), points)


def _reference_transform_record(reference: gdstk.Reference) -> tuple[object, ...]:
    return (
        _float_pair(reference.origin),
        _float_key(reference.rotation or 0),
        _float_key(reference.magnification if reference.magnification is not None else 1),
        bool(reference.x_reflection),
        _repetition_record(reference.repetition),
    )


def _repetition_record(repetition) -> tuple[object, ...] | None:
    if repetition is None or repetition.size == 1:
        return None
    return tuple(sorted(_float_pair(offset) for offset in repetition.get_offsets()))


def _reference_cell(reference: gdstk.Reference) -> gdstk.Cell:
    if reference.cell is None:
        raise ValueError(f"unresolved reference to {reference.cell_name!r}")
    return reference.cell


def _reference_bbox(reference: gdstk.Reference) -> BBox:
    bbox = reference.bounding_box()
    if bbox is None:
        return (0, 0, 0, 0)
    return (
        int(_float_key(bbox[0][0])),
        int(_float_key(bbox[0][1])),
        int(_float_key(bbox[1][0])),
        int(_float_key(bbox[1][1])),
    )


def _reference_bbox_union(references: Iterable[gdstk.Reference]) -> BBox:
    return bbox_union(tuple(_reference_bbox(reference) for reference in references))


def _select_top(lib: gdstk.Library, name: str | None) -> gdstk.Cell:
    if name is not None:
        for cell in lib.cells:
            if cell.name == name:
                return cell
        raise ValueError(f"top cell {name!r} not found")
    tops = lib.top_level()
    if len(tops) != 1:
        raise ValueError("library must have exactly one top cell when top is not supplied")
    return tops[0]


def _float_key(value: float) -> int:
    return int(round(float(value) * 1_000_000_000))


def _float_pair(values) -> tuple[int, int]:
    return (_float_key(values[0]), _float_key(values[1]))


def _digest_records(records: object) -> str:
    return hashlib.sha256(repr(records).encode("utf-8")).hexdigest()
