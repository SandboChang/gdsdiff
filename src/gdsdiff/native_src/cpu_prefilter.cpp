#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr std::uint8_t kOldSide = 1;
constexpr std::uint8_t kNewSide = 2;
constexpr std::uint8_t kBothSides = kOldSide | kNewSide;

thread_local std::string g_last_error;

struct Polygon {
    std::int64_t x0;
    std::int64_t y0;
    std::int64_t x1;
    std::int64_t y1;
    std::int32_t layer;
    std::int32_t datatype;
    std::uint64_t canonical_id;
};

struct AssignmentKey {
    std::int32_t layer;
    std::int32_t datatype;
    std::int64_t tx;
    std::int64_t ty;
    std::uint64_t canonical_id;

    bool operator<(const AssignmentKey& other) const {
        return std::tie(layer, datatype, tx, ty, canonical_id)
            < std::tie(other.layer, other.datatype, other.tx, other.ty, other.canonical_id);
    }

    bool operator==(const AssignmentKey& other) const {
        return layer == other.layer && datatype == other.datatype && tx == other.tx && ty == other.ty
            && canonical_id == other.canonical_id;
    }
};

struct Assignment {
    AssignmentKey key;
    std::uint8_t side;
};

struct CandidateTile {
    std::int32_t layer;
    std::int32_t datatype;
    std::int64_t tx;
    std::int64_t ty;

    bool operator<(const CandidateTile& other) const {
        return std::tie(layer, datatype, tx, ty) < std::tie(other.layer, other.datatype, other.tx, other.ty);
    }

    bool operator==(const CandidateTile& other) const {
        return layer == other.layer && datatype == other.datatype && tx == other.tx && ty == other.ty;
    }
};

struct OversizedKey {
    std::int32_t layer;
    std::int32_t datatype;
    std::uint64_t canonical_id;
    std::int64_t tx0;
    std::int64_t ty0;
    std::int64_t tx1;
    std::int64_t ty1;

    bool operator<(const OversizedKey& other) const {
        return std::tie(layer, datatype, canonical_id, tx0, ty0, tx1, ty1)
            < std::tie(other.layer, other.datatype, other.canonical_id, other.tx0, other.ty0, other.tx1, other.ty1);
    }
};

struct OversizedItem {
    OversizedKey key;
    std::uint8_t side;
};

std::int64_t floor_div(std::int64_t value, std::int64_t divisor) {
    std::int64_t quotient = value / divisor;
    const std::int64_t remainder = value % divisor;
    if (remainder != 0 && ((remainder < 0) != (divisor < 0))) {
        --quotient;
    }
    return quotient;
}

std::vector<Polygon> pack_polygons(
    const std::int64_t* bboxes,
    const std::int32_t* layers,
    const std::int32_t* datatypes,
    const std::uint64_t* canonical_ids,
    std::uint64_t count) {
    std::vector<Polygon> out;
    out.reserve(count);
    for (std::uint64_t i = 0; i < count; ++i) {
        out.push_back(Polygon{
            bboxes[i * 4 + 0],
            bboxes[i * 4 + 1],
            bboxes[i * 4 + 2],
            bboxes[i * 4 + 3],
            layers[i],
            datatypes[i],
            canonical_ids[i],
        });
    }
    return out;
}

void append_side(
    std::vector<Assignment>& assignments,
    std::vector<OversizedItem>& oversized_items,
    const std::vector<Polygon>& polygons,
    std::int64_t tile_size,
    std::uint64_t max_tiles_per_polygon,
    std::uint8_t side) {
    if (polygons.empty()) {
        return;
    }
    const int thread_count =
#ifdef _OPENMP
        std::max(1, omp_get_max_threads());
#else
        1;
#endif
    std::vector<std::vector<Assignment>> local_assignments(thread_count);
    std::vector<std::vector<OversizedItem>> local_oversized(thread_count);

#pragma omp parallel
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        auto& local_a = local_assignments[tid];
        auto& local_o = local_oversized[tid];
#pragma omp for schedule(dynamic, 64)
        for (std::int64_t index = 0; index < static_cast<std::int64_t>(polygons.size()); ++index) {
            const Polygon& polygon = polygons[static_cast<std::size_t>(index)];
            const std::int64_t tx0 = floor_div(polygon.x0, tile_size);
            const std::int64_t ty0 = floor_div(polygon.y0, tile_size);
            const std::int64_t tx1 = floor_div(polygon.x1, tile_size);
            const std::int64_t ty1 = floor_div(polygon.y1, tile_size);
            const std::int64_t span_x = tx1 - tx0 + 1;
            const std::int64_t span_y = ty1 - ty0 + 1;
            if (span_x <= 0 || span_y <= 0) {
                continue;
            }
            const auto span_x_u = static_cast<std::uint64_t>(span_x);
            const auto span_y_u = static_cast<std::uint64_t>(span_y);
            if (span_y_u != 0 && span_x_u > UINT64_MAX / span_y_u) {
                continue;
            }
            const std::uint64_t count = span_x_u * span_y_u;
            if (count > max_tiles_per_polygon) {
                local_o.push_back(OversizedItem{
                    OversizedKey{polygon.layer, polygon.datatype, polygon.canonical_id, tx0, ty0, tx1, ty1},
                    side,
                });
                continue;
            }
            local_a.reserve(local_a.size() + static_cast<std::size_t>(count));
            for (std::int64_t tx = tx0; tx <= tx1; ++tx) {
                for (std::int64_t ty = ty0; ty <= ty1; ++ty) {
                    local_a.push_back(Assignment{
                        AssignmentKey{polygon.layer, polygon.datatype, tx, ty, polygon.canonical_id},
                        side,
                    });
                }
            }
        }
    }

    std::size_t assignment_total = assignments.size();
    std::size_t oversized_total = oversized_items.size();
    for (const auto& chunk : local_assignments) {
        assignment_total += chunk.size();
    }
    for (const auto& chunk : local_oversized) {
        oversized_total += chunk.size();
    }
    assignments.reserve(assignment_total);
    oversized_items.reserve(oversized_total);
    for (auto& chunk : local_assignments) {
        assignments.insert(assignments.end(), chunk.begin(), chunk.end());
    }
    for (auto& chunk : local_oversized) {
        oversized_items.insert(oversized_items.end(), chunk.begin(), chunk.end());
    }
}

}  // namespace

extern "C" {

struct GdsCpuPrefilterResult {
    std::int64_t* candidate_tiles;
    std::uint64_t candidate_count;
    std::int64_t* oversized;
    std::uint64_t oversized_count;
    std::uint64_t assignment_count;
    std::uint64_t collapsed_pair_count;
    std::uint64_t oversized_matched_count;
};

const char* gds_cpu_last_error() {
    return g_last_error.c_str();
}

void gds_cpu_free_result(GdsCpuPrefilterResult* result) {
    if (!result) {
        return;
    }
    std::free(result->candidate_tiles);
    std::free(result->oversized);
    result->candidate_tiles = nullptr;
    result->oversized = nullptr;
    result->candidate_count = 0;
    result->oversized_count = 0;
}

int gds_cpu_prefilter(
    const std::int64_t* old_bboxes,
    const std::int32_t* old_layers,
    const std::int32_t* old_datatypes,
    const std::uint64_t* old_ids,
    std::uint64_t old_count,
    const std::int64_t* new_bboxes,
    const std::int32_t* new_layers,
    const std::int32_t* new_datatypes,
    const std::uint64_t* new_ids,
    std::uint64_t new_count,
    std::int64_t tile_size,
    std::uint64_t max_tiles_per_polygon,
    GdsCpuPrefilterResult* result) {
    try {
        if (!result) {
            throw std::invalid_argument("result pointer is null");
        }
        *result = GdsCpuPrefilterResult{};
        if (tile_size <= 0 || max_tiles_per_polygon == 0) {
            throw std::invalid_argument("invalid tile config");
        }
        std::vector<Assignment> assignments;
        std::vector<OversizedItem> oversized_items;
        const auto old_polygons = pack_polygons(old_bboxes, old_layers, old_datatypes, old_ids, old_count);
        const auto new_polygons = pack_polygons(new_bboxes, new_layers, new_datatypes, new_ids, new_count);
        append_side(assignments, oversized_items, old_polygons, tile_size, max_tiles_per_polygon, kOldSide);
        append_side(assignments, oversized_items, new_polygons, tile_size, max_tiles_per_polygon, kNewSide);
        result->assignment_count = assignments.size();

        std::sort(assignments.begin(), assignments.end(), [](const Assignment& left, const Assignment& right) {
            return left.key < right.key;
        });
        std::vector<CandidateTile> candidate_tiles;
        for (std::size_t i = 0; i < assignments.size();) {
            const AssignmentKey key = assignments[i].key;
            std::uint8_t side = 0;
            do {
                side = static_cast<std::uint8_t>(side | assignments[i].side);
                ++i;
            } while (i < assignments.size() && assignments[i].key == key);
            ++result->collapsed_pair_count;
            if (side != kBothSides) {
                candidate_tiles.push_back(CandidateTile{key.layer, key.datatype, key.tx, key.ty});
            }
        }
        std::sort(candidate_tiles.begin(), candidate_tiles.end());
        candidate_tiles.erase(std::unique(candidate_tiles.begin(), candidate_tiles.end()), candidate_tiles.end());
        result->candidate_count = candidate_tiles.size();
        if (!candidate_tiles.empty()) {
            result->candidate_tiles = static_cast<std::int64_t*>(std::malloc(sizeof(std::int64_t) * candidate_tiles.size() * 4));
            if (!result->candidate_tiles) {
                throw std::bad_alloc();
            }
            for (std::size_t i = 0; i < candidate_tiles.size(); ++i) {
                result->candidate_tiles[i * 4 + 0] = candidate_tiles[i].layer;
                result->candidate_tiles[i * 4 + 1] = candidate_tiles[i].datatype;
                result->candidate_tiles[i * 4 + 2] = candidate_tiles[i].tx;
                result->candidate_tiles[i * 4 + 3] = candidate_tiles[i].ty;
            }
        }

        std::sort(oversized_items.begin(), oversized_items.end(), [](const OversizedItem& left, const OversizedItem& right) {
            return left.key < right.key;
        });
        std::vector<OversizedItem> oversized_out;
        for (std::size_t i = 0; i < oversized_items.size();) {
            const OversizedKey key = oversized_items[i].key;
            std::uint8_t side = 0;
            do {
                side = static_cast<std::uint8_t>(side | oversized_items[i].side);
                ++i;
            } while (i < oversized_items.size() && !(key < oversized_items[i].key) && !(oversized_items[i].key < key));
            if (side == kBothSides) {
                ++result->oversized_matched_count;
            } else {
                oversized_out.push_back(OversizedItem{key, side});
            }
        }
        result->oversized_count = oversized_out.size();
        if (!oversized_out.empty()) {
            result->oversized = static_cast<std::int64_t*>(std::malloc(sizeof(std::int64_t) * oversized_out.size() * 8));
            if (!result->oversized) {
                throw std::bad_alloc();
            }
            for (std::size_t i = 0; i < oversized_out.size(); ++i) {
                const OversizedItem& item = oversized_out[i];
                result->oversized[i * 8 + 0] = item.key.layer;
                result->oversized[i * 8 + 1] = item.key.datatype;
                result->oversized[i * 8 + 2] = static_cast<std::int64_t>(item.key.canonical_id);
                result->oversized[i * 8 + 3] = item.side;
                result->oversized[i * 8 + 4] = item.key.tx0;
                result->oversized[i * 8 + 5] = item.key.ty0;
                result->oversized[i * 8 + 6] = item.key.tx1;
                result->oversized[i * 8 + 7] = item.key.ty1;
            }
        }
        g_last_error.clear();
        return 0;
    } catch (const std::exception& exc) {
        g_last_error = exc.what();
        if (result) {
            gds_cpu_free_result(result);
        }
        return 1;
    } catch (...) {
        g_last_error = "unknown CPU prefilter error";
        if (result) {
            gds_cpu_free_result(result);
        }
        return 1;
    }
}

}  // extern "C"
