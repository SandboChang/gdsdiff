#include "core.hpp"

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

}  // namespace

int main() {
    assert(floor_div(-1, 10) == -1);
    assert(floor_div(-10, 10) == -1);
    assert(floor_div(-11, 10) == -2);
    assert((bbox_tile_range(BBox{-11, -1, 10, 20}, 10) == BBox{-2, -1, 1, 2}));

    const auto a = rect(0, 0, 10, 10);
    const PolygonRecord a_rotated{
        a.layer,
        a.bbox,
        std::vector<Point>{{10, 10}, {0, 10}, {0, 0}, {10, 0}},
        true,
    };
    assert(canonical_blob(a.points, true) == canonical_blob(a_rotated.points, true));
    assert(canonical_blob(a.points, false) != canonical_blob(a_rotated.points, false));

    const std::vector<std::string> blobs{
        canonical_blob(a.points, true),
        canonical_blob(a_rotated.points, true),
        canonical_blob(rect(20, 0, 30, 10).points, true),
    };
    const auto ids = intern_canonical_blobs(blobs);
    assert(ids.size() == 3);
    assert(ids[0] == ids[1]);
    assert(ids[0] != ids[2]);

    const auto identical = run_tile_prefilter(
        std::vector<PolygonRecord>{rect(0, 0, 9, 9)},
        std::vector<std::uint64_t>{7},
        std::vector<PolygonRecord>{rect(0, 0, 9, 9)},
        std::vector<std::uint64_t>{7},
        TileConfig{10, 4096});
    assert(identical.candidate_tiles.empty());
    assert(identical.assignment_count == 2);
    assert(identical.collapsed_pair_count == 1);

    const auto changed = run_tile_prefilter(
        std::vector<PolygonRecord>{rect(0, 0, 9, 9)},
        std::vector<std::uint64_t>{1},
        std::vector<PolygonRecord>{rect(20, 0, 29, 9)},
        std::vector<std::uint64_t>{2},
        TileConfig{10, 4096});
    assert(changed.candidate_tiles.size() == 2);
    assert((changed.candidate_tiles[0] == TileKey{{1, 0}, 0, 0}));
    assert((changed.candidate_tiles[1] == TileKey{{1, 0}, 2, 0}));

    const auto oversized = run_tile_prefilter(
        std::vector<PolygonRecord>{rect(0, 0, 100, 100)},
        std::vector<std::uint64_t>{9},
        std::vector<PolygonRecord>{},
        std::vector<std::uint64_t>{},
        TileConfig{10, 4});
    assert(oversized.candidate_tiles.empty());
    assert(oversized.oversized_candidates.size() == 1);
    assert(oversized.oversized_candidates[0].side_mask == kOldSide);
    assert((oversized.oversized_candidates[0].tile_bbox == BBox{0, 0, 10, 10}));

    bool length_error = false;
    try {
        (void)run_tile_prefilter(
            std::vector<PolygonRecord>{rect(0, 0, 1, 1)},
            std::vector<std::uint64_t>{},
            std::vector<PolygonRecord>{},
            std::vector<std::uint64_t>{},
            TileConfig{10, 4096});
    } catch (const std::invalid_argument&) {
        length_error = true;
    }
    assert(length_error);

    std::cout << "cpp core tests passed\n";
    return 0;
}
