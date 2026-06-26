#include "cuda_assignment.cuh"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <stdexcept>

using namespace gdsdiff;

namespace {

PolygonRecord rect(std::int64_t x0, std::int64_t y0, std::int64_t x1, std::int64_t y1, LayerSpec layer = {1, 0}) {
    return PolygonRecord{
        layer,
        BBox{x0, y0, x1, y1},
        std::vector<Point>{{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}},
        true,
    };
}

std::vector<TileAssignment> cpu_assignments(
    const std::vector<PolygonRecord>& polygons,
    const std::vector<std::uint64_t>& ids,
    std::uint8_t side,
    const TileConfig& config,
    std::vector<OversizedPolygon>* oversized) {
    std::vector<TileAssignment> assignments;
    for (std::size_t i = 0; i < polygons.size(); ++i) {
        const auto range = bbox_tile_range(polygons[i].bbox, config.tile_size);
        const std::uint64_t span_x = static_cast<std::uint64_t>(range.x1 - range.x0 + 1);
        const std::uint64_t span_y = static_cast<std::uint64_t>(range.y1 - range.y0 + 1);
        const std::uint64_t count = span_x * span_y;
        if (count > config.max_tiles_per_polygon) {
            oversized->push_back(OversizedPolygon{polygons[i].layer, ids[i], side, range});
            continue;
        }
        for (std::int64_t tx = range.x0; tx <= range.x1; ++tx) {
            for (std::int64_t ty = range.y0; ty <= range.y1; ++ty) {
                assignments.push_back(TileAssignment{polygons[i].layer, tx, ty, ids[i], side});
            }
        }
    }
    return assignments;
}

void assert_matches_cpu(
    const std::vector<PolygonRecord>& polygons,
    const std::vector<std::uint64_t>& ids,
    std::uint8_t side,
    const TileConfig& config) {
    std::vector<OversizedPolygon> expected_oversized;
    auto expected = cpu_assignments(polygons, ids, side, config, &expected_oversized);
    auto actual = run_cuda_assignment(polygons, ids, side, config);
    std::sort(expected.begin(), expected.end());
    std::sort(actual.assignments.begin(), actual.assignments.end());
    std::sort(expected_oversized.begin(), expected_oversized.end());
    std::sort(actual.oversized.begin(), actual.oversized.end());
    assert(actual.assignments == expected);
    assert(actual.oversized == expected_oversized);
    assert(actual.total_assignment_count == expected.size());
}

}  // namespace

int main() {
    assert_matches_cpu({}, {}, kOldSide, TileConfig{10, 4096});
    assert_matches_cpu(
        std::vector<PolygonRecord>{rect(-11, -11, -1, -1)},
        std::vector<std::uint64_t>{1},
        kOldSide,
        TileConfig{10, 4096});
    assert_matches_cpu(
        std::vector<PolygonRecord>{rect(0, 0, 10, 10)},
        std::vector<std::uint64_t>{2},
        kNewSide,
        TileConfig{10, 4096});
    assert_matches_cpu(
        std::vector<PolygonRecord>{rect(0, 0, 100, 100)},
        std::vector<std::uint64_t>{3},
        kOldSide,
        TileConfig{10, 4});

    std::vector<PolygonRecord> many;
    std::vector<std::uint64_t> ids;
    for (int i = 0; i < 777; ++i) {
        many.push_back(rect(i * 3, -i, i * 3 + 11, -i + 7, LayerSpec{i % 4, i % 2}));
        ids.push_back(static_cast<std::uint64_t>(i % 9));
    }
    assert_matches_cpu(many, ids, kNewSide, TileConfig{10, 4096});

    bool length_error = false;
    try {
        (void)run_cuda_assignment(std::vector<PolygonRecord>{rect(0, 0, 1, 1)}, {}, kOldSide, TileConfig{10, 4096});
    } catch (const std::invalid_argument&) {
        length_error = true;
    }
    assert(length_error);

    bool config_error = false;
    try {
        (void)run_cuda_assignment(std::vector<PolygonRecord>{rect(0, 0, 1, 1)}, {1}, kOldSide, TileConfig{0, 4096});
    } catch (const std::invalid_argument&) {
        config_error = true;
    }
    assert(config_error);

    std::cout << "cuda assignment tests passed\n";
    return 0;
}
