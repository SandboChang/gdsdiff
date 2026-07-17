from __future__ import annotations

import random
from pathlib import Path

import gdstk

from gdsdiff import CompareConfig, compare_gds_files


def _rect(rng: random.Random) -> tuple[int, int, int, int]:
    x0 = rng.randint(-80, 120)
    y0 = rng.randint(-80, 120)
    width = rng.randint(2, 35)
    height = rng.randint(2, 35)
    return x0, y0, x0 + width, y0 + height


def _records(seed: int, *, count: int = 18) -> list[tuple[tuple[int, int], tuple[int, int, int, int]]]:
    rng = random.Random(seed)
    layers = ((1, 0), (2, 0), (7, 3))
    records = []
    for index in range(count):
        records.append((layers[index % len(layers)], _rect(rng)))
    records.extend(records[:3])
    records.append(((2, 0), (-5, -5, 25, 25)))
    records.append(((2, 0), (0, 0, 35, 20)))
    return records


def _mutate(records: list[tuple[tuple[int, int], tuple[int, int, int, int]]], seed: int):
    rng = random.Random(seed + 10_000)
    mutated = list(records)
    if mutated:
        mutated.pop(rng.randrange(len(mutated)))
    mutated.append(((1, 0), _rect(rng)))
    mutated.append(((9, 1), (seed * 3, -20, seed * 3 + 13, -3)))
    return mutated


def _write_layout(path: Path, records, *, references: bool = True) -> None:
    lib = gdstk.Library(unit=1e-6, precision=1e-6)
    top = lib.new_cell("TOP")
    for layer, bbox in records:
        top.add(gdstk.rectangle(bbox[:2], bbox[2:], layer=layer[0], datatype=layer[1]))
    if references:
        leaf = lib.new_cell("LEAF")
        leaf.add(gdstk.rectangle((-6, -6), (6, 6), layer=12, datatype=0))
        leaf.add(gdstk.rectangle((-3, -14), (3, 14), layer=12, datatype=0))
        top.add(gdstk.Reference(leaf, (-30, 40)))
        top.add(gdstk.Reference(leaf, (70, -50)))
    lib.write_gds(path)


def _diff_signature(result):
    signature = []
    for layer in result.exact_result.layers:
        if layer.old_minus_new_twice_area == 0 and layer.new_minus_old_twice_area == 0 and layer.xor_twice_area == 0:
            continue
        signature.append(
            (
                layer.layer.layer,
                layer.layer.datatype,
                layer.old_minus_new_twice_area,
                layer.new_minus_old_twice_area,
                layer.xor_twice_area,
                tuple(sorted(fragment.bbox for fragment in layer.xor)),
            )
        )
    return tuple(signature)


def test_seeded_windowed_results_match_full_exact_oracle(tmp_path):
    for seed in range(8):
        before = tmp_path / f"seed_{seed}_before.gds"
        after = tmp_path / f"seed_{seed}_after.gds"
        records = _records(seed)
        _write_layout(before, records)
        _write_layout(after, _mutate(records, seed))
        exact = compare_gds_files(CompareConfig(before, after, backend="cpu", tile_size=64), exact=True)

        for tile_size in (7, 23, 64):
            windowed = compare_gds_files(CompareConfig(before, after, backend="cpu", tile_size=tile_size))
            assert windowed.status == exact.status
            assert _diff_signature(windowed) == _diff_signature(exact)


def test_equivalent_duplicate_overlap_reference_layouts_stay_equivalent(tmp_path):
    before = tmp_path / "same_before.gds"
    after = tmp_path / "same_after.gds"
    records = _records(41, count=24)
    _write_layout(before, records)
    _write_layout(after, list(records))

    for tile_size in (5, 19, 101):
        result = compare_gds_files(CompareConfig(before, after, backend="cpu", tile_size=tile_size))
        assert result.exit_code == 0
        assert result.exact_result.equivalent


def test_tile_boundary_mutations_are_reported_for_all_tile_sizes(tmp_path):
    before = tmp_path / "boundary_before.gds"
    after = tmp_path / "boundary_after.gds"
    base = [((3, 0), (-25, -25, 25, 25)), ((3, 0), (32, -8, 60, 8))]
    changed = list(base) + [((3, 0), (15, 15, 35, 35)), ((4, 0), (-18, 40, 18, 52))]
    _write_layout(before, base, references=False)
    _write_layout(after, changed, references=False)
    exact = compare_gds_files(CompareConfig(before, after, backend="cpu", tile_size=128), exact=True)

    signatures = []
    for tile_size in (8, 16, 31):
        result = compare_gds_files(CompareConfig(before, after, backend="cpu", tile_size=tile_size))
        assert result.exit_code == 1
        signatures.append(_diff_signature(result))
    assert signatures == [_diff_signature(exact)] * len(signatures)
