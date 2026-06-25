#include "cuda_memory.cuh"
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

void sort_result(TilePrefilterResult& result) {
    std::sort(result.candidate_tiles.begin(), result.candidate_tiles.end());
    std::sort(result.oversized_candidates.begin(), result.oversized_candidates.end());
}

void assert_chunked_matches(
    const std::vector<PolygonRecord>& old_polygons,
    const std::vector<std::uint64_t>& old_ids,
    const std::vector<PolygonRecord>& new_polygons,
    const std::vector<std::uint64_t>& new_ids,
    const TileConfig& config,
    const CudaMemoryBudget& budget,
    bool expect_chunking) {
    auto cpu = run_tile_prefilter(old_polygons, old_ids, new_polygons, new_ids, config);
    auto whole_cuda = run_cuda_candidate_compaction(old_polygons, old_ids, new_polygons, new_ids, config);
    auto chunked = run_cuda_candidate_compaction_chunked(old_polygons, old_ids, new_polygons, new_ids, config, budget);
    sort_result(cpu);
    sort_result(whole_cuda);
    sort_result(chunked.prefilter);
    assert(chunked.prefilter.candidate_tiles == cpu.candidate_tiles);
    assert(chunked.prefilter.candidate_tiles == whole_cuda.candidate_tiles);
    assert(chunked.prefilter.oversized_candidates == cpu.oversized_candidates);
    assert(chunked.prefilter.oversized_candidates == whole_cuda.oversized_candidates);
    assert(chunked.stats.chunk_count >= 1);
    assert(chunked.stats.used_chunking == expect_chunking);
}

}  // namespace

int main() {
    assert_chunked_matches({}, {}, {}, {}, TileConfig{10, 4096}, CudaMemoryBudget{64, 64}, false);
    assert_chunked_matches(
        {rect(0, 0, 9, 45), rect(20, 0, 29, 9, LayerSpec{2, 0})},
        {1, 2},
        {rect(0, 10, 9, 55), rect(25, 0, 34, 9, LayerSpec{2, 0})},
        {3, 4},
        TileConfig{10, 4096},
        CudaMemoryBudget{128, 64},
        true);
    assert_chunked_matches(
        {rect(0, 0, 100, 100)},
        {9},
        {},
        {},
        TileConfig{10, 4},
        CudaMemoryBudget{64, 64},
        false);

    std::vector<PolygonRecord> old_polygons;
    std::vector<PolygonRecord> new_polygons;
    std::vector<std::uint64_t> old_ids;
    std::vector<std::uint64_t> new_ids;
    for (int i = 0; i < 160; ++i) {
        old_polygons.push_back(rect(i - 80, i * 5 - 300, i - 71, i * 5 - 276, LayerSpec{i % 3, i % 2}));
        old_ids.push_back(static_cast<std::uint64_t>(i % 11));
        new_polygons.push_back(rect(i - 78, i * 5 - 298, i - 69, i * 5 - 274, LayerSpec{i % 3, i % 2}));
        new_ids.push_back(static_cast<std::uint64_t>((i + 3) % 11));
    }
    assert_chunked_matches(old_polygons, old_ids, new_polygons, new_ids, TileConfig{8, 4096}, CudaMemoryBudget{512, 64}, true);

    bool config_error = false;
    try {
        (void)run_cuda_candidate_compaction_chunked(
            {rect(0, 0, 1, 1)},
            {1},
            {},
            {},
            TileConfig{10, 4096},
            CudaMemoryBudget{0, 64});
    } catch (const std::invalid_argument&) {
        config_error = true;
    }
    assert(config_error);

    std::cout << "cuda memory tests passed\n";
    return 0;
}
