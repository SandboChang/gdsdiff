from __future__ import annotations

from decimal import Decimal

import pytest

from gds_tools_tk.gpu_compare import (
    AllowedRegion,
    Backend,
    CompareConfig,
    DifferenceDirection,
    EvidenceEngine,
    FailurePolicy,
    GridSpec,
    LayerSpec,
    PathPolicy,
    parse_layer_selector,
)


def test_layer_spec_parse_and_sort():
    assert LayerSpec.parse("2:3") == LayerSpec(2, 3)
    assert LayerSpec.parse((5, 0)) == LayerSpec(5, 0)
    assert parse_layer_selector("3:0,1:2,3:0") == (LayerSpec(1, 2), LayerSpec(3, 0))


@pytest.mark.parametrize("value", ["", "1", "1:", "-1:0", "1:-2"])
def test_layer_spec_rejects_invalid(value):
    with pytest.raises((ValueError, TypeError)):
        parse_layer_selector(value)


def test_compare_config_accepts_default_all_layers():
    config = CompareConfig("old.gds", "new.gds")

    assert config.old_path.name == "old.gds"
    assert config.new_path.name == "new.gds"
    assert config.layers == "all"
    assert config.backend is Backend.AUTO
    assert config.evidence_engine is EvidenceEngine.AUTO
    assert config.path_policy is PathPolicy.GDSTK
    assert config.failure_policy is FailurePolicy.FAIL_ON_UNEXPECTED


def test_compare_config_serialization_is_deterministic():
    config = CompareConfig(
        "old.gds",
        "new.gds",
        layers="7:1,2:0",
        dbu="1e-9",
        precision=Decimal("1e-9"),
        backend="cpu",
        evidence_engine="gdstk-oracle",
        path_policy="reject",
        tile_size=256,
    )

    assert config.to_json_dict() == {
        "old_path": "old.gds",
        "new_path": "new.gds",
        "top": None,
        "old_top": None,
        "new_top": None,
        "require_top_name_match": False,
        "layers": [{"layer": 2, "datatype": 0}, {"layer": 7, "datatype": 1}],
        "dbu": "1E-9",
        "precision": "1E-9",
        "path_policy": "reject",
        "backend": "cpu",
        "evidence_engine": "gdstk-oracle",
        "tile_size": 256,
        "allowed_regions": [],
        "allow_empty_selected_layer": False,
        "strict": True,
        "require_gpu": False,
        "failure_policy": "fail-on-unexpected",
        "memory_budget_bytes": None,
    }


def test_compare_config_rejects_conflicting_top_selectors():
    with pytest.raises(ValueError, match="top cannot be combined"):
        CompareConfig("old.gds", "new.gds", top="top", old_top="old", new_top="new")

    with pytest.raises(ValueError, match="old_top and new_top"):
        CompareConfig("old.gds", "new.gds", old_top="old")


def test_compare_config_rejects_conflicting_dbu_and_precision():
    with pytest.raises(ValueError, match="dbu and precision"):
        CompareConfig("old.gds", "new.gds", dbu="1e-9", precision="2e-9")


def test_compare_config_rejects_invalid_backend_and_tile_size():
    with pytest.raises(ValueError, match="invalid Backend"):
        CompareConfig("old.gds", "new.gds", backend="gpu")

    with pytest.raises(ValueError, match="invalid EvidenceEngine"):
        CompareConfig("old.gds", "new.gds", evidence_engine="not-real")

    with pytest.raises(ValueError, match="tile_size"):
        CompareConfig("old.gds", "new.gds", tile_size=0)

    with pytest.raises(ValueError, match="require_gpu"):
        CompareConfig("old.gds", "new.gds", backend="cpu", require_gpu=True)


def test_allowed_region_validation_and_serialization():
    region = AllowedRegion(
        layers=(LayerSpec(1, 2), LayerSpec(0, 0)),
        direction="addition",
        bbox=(0, 10, 20, 30),
        halo_dbu=2,
        label="expected contact",
    )

    assert region.direction is DifferenceDirection.ADDITION
    assert region.to_json()["layers"] == [{"layer": 0, "datatype": 0}, {"layer": 1, "datatype": 2}]
    assert region.to_json()["bbox"] == [0, 10, 20, 30]


def test_allowed_region_requires_bbox_or_explicit_whole_layer():
    with pytest.raises(ValueError, match="requires bbox"):
        AllowedRegion(layers=(LayerSpec(1, 0),))

    whole = AllowedRegion(layers=(LayerSpec(1, 0),), whole_layer=True)
    assert whole.to_json()["whole_layer"] is True

    with pytest.raises(ValueError, match="cannot also set bbox"):
        AllowedRegion(layers=(LayerSpec(1, 0),), whole_layer=True, bbox=(0, 0, 1, 1))

    with pytest.raises(ValueError, match="x0 < x1"):
        AllowedRegion(layers=(LayerSpec(1, 0),), bbox=(0, 0, 0, 1))


def test_grid_spec_uses_decimal_values_and_rounding_policy():
    grid = GridSpec(meters_per_dbu="1e-9", source_old_precision_m="1e-9", max_snap_error_dbu="0.5")

    assert grid.meters_per_dbu == Decimal("1e-9")
    assert grid.to_json()["meters_per_dbu"] == "1E-9"

    with pytest.raises(ValueError, match="unsupported rounding"):
        GridSpec(rounding_mode="bankers")
