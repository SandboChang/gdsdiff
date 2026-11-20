from __future__ import annotations

from dataclasses import dataclass, field

import gdstk

from ._boolean_evidence import _checked_gdstk_boolean
from .config import AllowedRegion, LayerSpec
from .exact_oracle import (
    DiffFragment, ExactOracleResult, LayerExactDiff, _gdstk_points_to_global_dbu, format_twice_area,
)
from .geometry import polygon_bbox, polygon_twice_area
from .semantics import CompareStatus, DifferenceDirection, FailurePolicy


@dataclass(frozen=True)
class ClassifiedComponent:
    layer: LayerSpec
    direction: DifferenceDirection
    classification: str
    points: tuple[tuple[int, int], ...]
    bbox: tuple[int, int, int, int]
    twice_area: int

    @property
    def area_dbu2_string(self) -> str:
        return format_twice_area(self.twice_area)

    def to_json(self) -> dict[str, object]:
        return {
            "layer": self.layer.to_json(),
            "direction": self.direction.value,
            "classification": self.classification,
            "bbox": list(self.bbox),
            "area_dbu2": self.area_dbu2_string,
        }


@dataclass(frozen=True)
class LayerPolicyResult:
    layer: LayerSpec
    expected_components: tuple[ClassifiedComponent, ...] = field(default_factory=tuple)
    unexpected_components: tuple[ClassifiedComponent, ...] = field(default_factory=tuple)

    @property
    def expected_twice_area(self) -> int:
        return sum(component.twice_area for component in self.expected_components)

    @property
    def unexpected_twice_area(self) -> int:
        return sum(component.twice_area for component in self.unexpected_components)

    @property
    def components(self) -> tuple[ClassifiedComponent, ...]:
        return tuple(sorted(self.expected_components + self.unexpected_components, key=_component_sort_key))


@dataclass(frozen=True)
class PolicyClassificationResult:
    layers: tuple[LayerPolicyResult, ...]
    effective_policy: dict[str, object]
    failure_policy: FailurePolicy = FailurePolicy.FAIL_ON_UNEXPECTED

    @property
    def expected_twice_area(self) -> int:
        return sum(layer.expected_twice_area for layer in self.layers)

    @property
    def unexpected_twice_area(self) -> int:
        return sum(layer.unexpected_twice_area for layer in self.layers)

    @property
    def status(self) -> CompareStatus:
        if self.unexpected_twice_area:
            return CompareStatus.UNEXPECTED_DIFFERENCES
        if self.expected_twice_area:
            return CompareStatus.EXPECTED_DIFFERENCES_ONLY
        return CompareStatus.EQUIVALENT

    def to_json(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "failure_policy": self.failure_policy.value,
            "expected_area_dbu2": format_twice_area(self.expected_twice_area),
            "unexpected_area_dbu2": format_twice_area(self.unexpected_twice_area),
            "effective_policy": self.effective_policy,
            "layers": [
                {
                    "layer": layer.layer.to_json(),
                    "expected_area_dbu2": format_twice_area(layer.expected_twice_area),
                    "unexpected_area_dbu2": format_twice_area(layer.unexpected_twice_area),
                    "components": [component.to_json() for component in layer.components],
                }
                for layer in self.layers
            ],
        }


def classify_exact_differences(
    exact_result: ExactOracleResult,
    allowed_regions: tuple[AllowedRegion, ...] | list[AllowedRegion] = (),
    *,
    failure_policy: FailurePolicy | str = FailurePolicy.FAIL_ON_UNEXPECTED,
    precision: float = 1.0,
) -> PolicyClassificationResult:
    failure_policy = FailurePolicy.coerce(failure_policy)
    policy = _compile_policy(tuple(allowed_regions))
    layer_results = []
    for layer_diff in exact_result.layers:
        expected: list[ClassifiedComponent] = []
        unexpected: list[ClassifiedComponent] = []
        _classify_fragments(
            layer_diff.layer,
            DifferenceDirection.REMOVAL,
            layer_diff.old_minus_new,
            policy,
            expected,
            unexpected,
            precision=precision,
        )
        _classify_fragments(
            layer_diff.layer,
            DifferenceDirection.ADDITION,
            layer_diff.new_minus_old,
            policy,
            expected,
            unexpected,
            precision=precision,
        )
        layer_results.append(
            LayerPolicyResult(
                layer=layer_diff.layer,
                expected_components=tuple(sorted(expected, key=_component_sort_key)),
                unexpected_components=tuple(sorted(unexpected, key=_component_sort_key)),
            )
        )
    return PolicyClassificationResult(
        layers=tuple(sorted(layer_results, key=lambda result: result.layer)),
        effective_policy=_effective_policy_json(tuple(allowed_regions)),
        failure_policy=failure_policy,
    )


@dataclass(frozen=True)
class _CompiledPolicy:
    bbox_regions: dict[tuple[LayerSpec, DifferenceDirection], tuple[AllowedRegion, ...]]
    whole_layer: frozenset[tuple[LayerSpec, DifferenceDirection]]


def _compile_policy(allowed_regions: tuple[AllowedRegion, ...]) -> _CompiledPolicy:
    bbox_regions: dict[tuple[LayerSpec, DifferenceDirection], list[AllowedRegion]] = {}
    whole_layer: set[tuple[LayerSpec, DifferenceDirection]] = set()
    for region in allowed_regions:
        for layer in region.layers:
            for direction in _region_directions(region.direction):
                key = (layer, direction)
                if region.whole_layer:
                    whole_layer.add(key)
                else:
                    bbox_regions.setdefault(key, []).append(region)
    return _CompiledPolicy(
        bbox_regions={key: tuple(value) for key, value in bbox_regions.items()},
        whole_layer=frozenset(whole_layer),
    )


def _region_directions(direction: DifferenceDirection) -> tuple[DifferenceDirection, ...]:
    if direction in {DifferenceDirection.BOTH, DifferenceDirection.XOR}:
        return DifferenceDirection.ADDITION, DifferenceDirection.REMOVAL
    if direction == DifferenceDirection.ADDITION:
        return (DifferenceDirection.ADDITION,)
    if direction == DifferenceDirection.REMOVAL:
        return (DifferenceDirection.REMOVAL,)
    raise ValueError(f"unsupported direction for allowed region: {direction}")


def _classify_fragments(
    layer: LayerSpec,
    direction: DifferenceDirection,
    fragments: tuple[DiffFragment, ...],
    policy: _CompiledPolicy,
    expected: list[ClassifiedComponent],
    unexpected: list[ClassifiedComponent],
    *,
    precision: float,
) -> None:
    if not fragments:
        return
    key = (layer, direction)
    if key in policy.whole_layer:
        expected.extend(_components_from_fragments(layer, direction, "expected", fragments))
        return
    regions = policy.bbox_regions.get(key, ())
    for fragment in fragments:
        if not regions:
            unexpected.append(_component_from_fragment(layer, direction, "unexpected", fragment))
            continue
        fragment_expected, fragment_unexpected = _split_fragment_by_regions(
            fragment,
            regions,
            layer,
            precision=precision,
        )
        expected.extend(
            _component_from_points(layer, direction, "expected", points)
            for points in fragment_expected
        )
        unexpected.extend(
            _component_from_points(layer, direction, "unexpected", points)
            for points in fragment_unexpected
        )


def _split_fragment_by_regions(
    fragment: DiffFragment,
    regions: tuple[AllowedRegion, ...],
    layer: LayerSpec,
    *,
    precision: float,
) -> tuple[tuple[tuple[tuple[int, int], ...], ...], tuple[tuple[tuple[int, int], ...], ...]]:
    origin = fragment.bbox[0], fragment.bbox[1]
    fragment_poly = _polygon_from_points(fragment.points, layer, origin)
    masks = [_region_polygon(region, layer, origin) for region in regions]
    options = {"precision": precision, "stage": "policy-classification", "layer": layer}
    allowed = _checked_gdstk_boolean([fragment_poly], masks, "and", **options)
    outside = _checked_gdstk_boolean([fragment_poly], masks, "not", **options)
    return _global_points_from_gdstk(allowed, origin), _global_points_from_gdstk(outside, origin)


def _polygon_from_points(points: tuple[tuple[int, int], ...], layer: LayerSpec, origin: tuple[int, int]) -> gdstk.Polygon:
    ox, oy = origin
    return gdstk.Polygon([(x - ox, y - oy) for x, y in points], layer=layer.layer, datatype=layer.datatype)


def _region_polygon(region: AllowedRegion, layer: LayerSpec, origin: tuple[int, int]) -> gdstk.Polygon:
    if region.bbox is None:
        raise ValueError("whole-layer regions should not be converted to bbox masks")
    ox, oy = origin
    x0, y0, x1, y1 = region.bbox
    halo = region.halo_dbu
    return gdstk.rectangle(
        (x0 - halo - ox, y0 - halo - oy),
        (x1 + halo - ox, y1 + halo - oy),
        layer=layer.layer,
        datatype=layer.datatype,
    )


def _global_points_from_gdstk(polygons: list[gdstk.Polygon], origin: tuple[int, int]) -> tuple[tuple[tuple[int, int], ...], ...]:
    result = []
    for polygon in polygons:
        points = _gdstk_points_to_global_dbu(polygon.points, origin)
        if len(points) >= 3 and polygon_twice_area(points):
            result.append(points)
    return tuple(result)


def _components_from_fragments(
    layer: LayerSpec,
    direction: DifferenceDirection,
    classification: str,
    fragments: tuple[DiffFragment, ...],
) -> tuple[ClassifiedComponent, ...]:
    return tuple(_component_from_fragment(layer, direction, classification, fragment) for fragment in fragments if fragment.twice_area)


def _component_from_fragment(
    layer: LayerSpec,
    direction: DifferenceDirection,
    classification: str,
    fragment: DiffFragment,
) -> ClassifiedComponent:
    return ClassifiedComponent(
        layer=layer,
        direction=direction,
        classification=classification,
        points=fragment.points,
        bbox=fragment.bbox,
        twice_area=fragment.twice_area,
    )


def _component_from_points(
    layer: LayerSpec,
    direction: DifferenceDirection,
    classification: str,
    points: tuple[tuple[int, int], ...],
) -> ClassifiedComponent:
    return ClassifiedComponent(
        layer=layer,
        direction=direction,
        classification=classification,
        points=points,
        bbox=polygon_bbox(points),
        twice_area=polygon_twice_area(points),
    )


def _component_sort_key(component: ClassifiedComponent) -> tuple[object, ...]:
    return (component.layer, component.direction.value, component.classification, component.bbox, component.points)


def _effective_policy_json(allowed_regions: tuple[AllowedRegion, ...]) -> dict[str, object]:
    return {
        "allowed_region_count": len(allowed_regions),
        "allowed_regions": [region.to_json() for region in allowed_regions],
    }
