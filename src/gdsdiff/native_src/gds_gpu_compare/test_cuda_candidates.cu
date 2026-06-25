#include "cuda_candidates.cuh"

#include <algorithm>
#include <cassert>
#include <iostream>

using namespace gds_gpu_compare;

namespace {

PolygonRecord rect(std::int64_t x0, std::int64_t y0, std::int64_t x1, std::int64_t y1, LayerSpec layer = {1, 0}) {
    return PolygonRecord{
        layer,
        BBox{x0, y0, x1, y1},
        std::vector<Point>{{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}},
        true,
    };
}

void assert_same_result(
    const std::vector<PolygonRecord>& old_polygons,
    const std::vector<std::uint64_t>& old_ids,
    const std::vector<PolygonRecord>& new_polygons,
    const std::vector<std::uint64_t>& new_ids,
    const TileConfig& config) {
    auto cpu = run_tile_prefilter(old_polygons, old_ids, new_polygons, new_ids, config);
    auto cuda = run_cuda_candidate_compaction(old_polygons, old_ids, new_polygons, new_ids, config);
    std::sort(cpu.candidate_tiles.begin(), cpu.candidate_tiles.end());
    std::sort(cuda.candidate_tiles.begin(), cuda.candidate_tiles.end());
    std::sort(cpu.oversized_candidates.begin(), cpu.oversized_candidates.end());
    std::sort(cuda.oversized_candidates.begin(), cuda.oversized_candidates.end());
    assert(cuda.candidate_tiles == cpu.candidate_tiles);
    assert(cuda.oversized_candidates == cpu.oversized_candidates);
    assert(cuda.assignment_count == cpu.assignment_count);
    assert(cuda.collapsed_pair_count == cpu.collapsed_pair_count);
    assert(cuda.oversized_matched_count == cpu.oversized_matched_count);
}

}  // namespace

int main() {
    assert_same_result({}, {}, {}, {}, TileConfig{10, 4096});
    assert_same_result(
        {rect(0, 0, 9, 9)},
        {1},
        {rect(0, 0, 9, 9)},
        {1},
        TileConfig{10, 4096});
    assert_same_result(
        {rect(0, 0, 9, 9)},
        {1},
        {rect(20, 0, 29, 9)},
        {2},
        TileConfig{10, 4096});
    assert_same_result(
        {rect(0, 0, 9, 9), rect(0, 0, 9, 9)},
        {5, 5},
        {rect(0, 0, 9, 9)},
        {5},
        TileConfig{10, 4096});
    assert_same_result(
        {rect(-11, -11, -1, -1)},
        {1},
        {},
        {},
        TileConfig{10, 4096});
    assert_same_result(
        {rect(0, 0, 100, 100)},
        {9},
        {},
        {},
        TileConfig{10, 4});
    assert_same_result(
        {rect(0, 0, 100, 100)},
        {9},
        {rect(0, 0, 100, 100)},
        {9},
        TileConfig{10, 4});

    std::vector<PolygonRecord> old_polygons;
    std::vector<PolygonRecord> new_polygons;
    std::vector<std::uint64_t> old_ids;
    std::vector<std::uint64_t> new_ids;
    for (int i = 0; i < 513; ++i) {
        old_polygons.push_back(rect(i * 7 - 90, i % 17 - 40, i * 7 - 82, i % 17 - 31, LayerSpec{i % 5, i % 3}));
        old_ids.push_back(static_cast<std::uint64_t>(i % 23));
        new_polygons.push_back(rect(i * 7 - 88, i % 17 - 40, i * 7 - 80, i % 17 - 31, LayerSpec{i % 5, i % 3}));
        new_ids.push_back(static_cast<std::uint64_t>((i + 1) % 23));
    }
    assert_same_result(old_polygons, old_ids, new_polygons, new_ids, TileConfig{16, 4096});

    std::cout << "cuda candidate tests passed\n";
    return 0;
}
