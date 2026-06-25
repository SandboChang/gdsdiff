#include "core.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <utility>

namespace gds_gpu_compare {
namespace {

void append_u32_be(std::string& out, std::uint32_t value) {
    for (int shift = 24; shift >= 0; shift -= 8) {
        out.push_back(static_cast<char>((value >> shift) & 0xff));
    }
}

void append_i64_be(std::string& out, std::int64_t value) {
    const auto encoded = static_cast<std::uint64_t>(value);
    for (int shift = 56; shift >= 0; shift -= 8) {
        out.push_back(static_cast<char>((encoded >> shift) & 0xff));
    }
}

std::vector<Point> min_cyclic_rotation(const std::vector<Point>& points) {
    if (points.empty()) {
        return points;
    }
    std::vector<Point> best;
    bool initialized = false;
    for (std::size_t i = 0; i < points.size(); ++i) {
        std::vector<Point> rotated;
        rotated.reserve(points.size());
        rotated.insert(rotated.end(), points.begin() + static_cast<std::ptrdiff_t>(i), points.end());
        rotated.insert(rotated.end(), points.begin(), points.begin() + static_cast<std::ptrdiff_t>(i));
        if (!initialized || rotated < best) {
            best = std::move(rotated);
            initialized = true;
        }
    }
    return best;
}

std::vector<Point> canonical_valid_points(const std::vector<Point>& points) {
    auto forward = min_cyclic_rotation(points);
    auto reversed = points;
    std::reverse(reversed.begin(), reversed.end());
    auto reverse = min_cyclic_rotation(reversed);
    return std::min(forward, reverse);
}

using PairKey = std::pair<TileKey, std::uint64_t>;

void add_side(
    std::map<PairKey, std::uint8_t>& pair_side_masks,
    std::map<std::tuple<LayerSpec, std::uint64_t, BBox>, std::uint8_t>& oversized_side_masks,
    const PolygonRecord& polygon,
    std::uint64_t canonical_id,
    std::uint8_t side,
    const TileConfig& config,
    std::uint64_t& assignment_count) {
    const BBox tile_bbox = bbox_tile_range(polygon.bbox, config.tile_size);
    const std::int64_t span_x = tile_bbox.x1 - tile_bbox.x0 + 1;
    const std::int64_t span_y = tile_bbox.y1 - tile_bbox.y0 + 1;
    if (span_x <= 0 || span_y <= 0) {
        throw std::overflow_error("invalid tile span");
    }
    const auto span_x_u = static_cast<std::uint64_t>(span_x);
    const auto span_y_u = static_cast<std::uint64_t>(span_y);
    if (span_x_u > std::numeric_limits<std::uint64_t>::max() / span_y_u) {
        throw std::overflow_error("tile span overflow");
    }
    const std::uint64_t count = span_x_u * span_y_u;
    if (count > config.max_tiles_per_polygon) {
        auto key = std::make_tuple(polygon.layer, canonical_id, tile_bbox);
        oversized_side_masks[key] = static_cast<std::uint8_t>(oversized_side_masks[key] | side);
        return;
    }
    for (std::int64_t tx = tile_bbox.x0; tx <= tile_bbox.x1; ++tx) {
        for (std::int64_t ty = tile_bbox.y0; ty <= tile_bbox.y1; ++ty) {
            PairKey key{TileKey{polygon.layer, tx, ty}, canonical_id};
            pair_side_masks[key] = static_cast<std::uint8_t>(pair_side_masks[key] | side);
            ++assignment_count;
        }
    }
}

}  // namespace

std::int64_t floor_div(std::int64_t value, std::int64_t divisor) {
    if (divisor <= 0) {
        throw std::invalid_argument("divisor must be positive");
    }
    std::int64_t quotient = value / divisor;
    const std::int64_t remainder = value % divisor;
    if (remainder != 0 && ((remainder < 0) != (divisor < 0))) {
        --quotient;
    }
    return quotient;
}

BBox bbox_tile_range(const BBox& bbox, std::int64_t tile_size) {
    if (tile_size <= 0) {
        throw std::invalid_argument("tile_size must be positive");
    }
    return BBox{
        floor_div(bbox.x0, tile_size),
        floor_div(bbox.y0, tile_size),
        floor_div(bbox.x1, tile_size),
        floor_div(bbox.y1, tile_size),
    };
}

std::string canonical_blob(const std::vector<Point>& points, bool valid) {
    const std::vector<Point> encoded_points = valid ? canonical_valid_points(points) : points;
    const std::uint32_t validity = valid ? kValidityValid : kValidityConservativeRaw;
    if (encoded_points.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("polygon point count exceeds encoding limit");
    }
    std::string out;
    out.reserve(12 + encoded_points.size() * 16);
    append_u32_be(out, kEncodingVersion);
    append_u32_be(out, validity);
    append_u32_be(out, static_cast<std::uint32_t>(encoded_points.size()));
    for (const Point& point : encoded_points) {
        append_i64_be(out, point.x);
        append_i64_be(out, point.y);
    }
    return out;
}

std::vector<std::uint64_t> intern_canonical_blobs(const std::vector<std::string>& blobs) {
    std::set<std::string> unique_set(blobs.begin(), blobs.end());
    std::vector<std::string> unique(unique_set.begin(), unique_set.end());
    std::map<std::string, std::uint64_t> id_by_blob;
    for (std::size_t i = 0; i < unique.size(); ++i) {
        id_by_blob.emplace(unique[i], static_cast<std::uint64_t>(i));
    }
    std::vector<std::uint64_t> ids;
    ids.reserve(blobs.size());
    for (const std::string& blob : blobs) {
        ids.push_back(id_by_blob.at(blob));
    }
    return ids;
}

TilePrefilterResult run_tile_prefilter(
    const std::vector<PolygonRecord>& old_polygons,
    const std::vector<std::uint64_t>& old_canonical_ids,
    const std::vector<PolygonRecord>& new_polygons,
    const std::vector<std::uint64_t>& new_canonical_ids,
    const TileConfig& config) {
    if (config.tile_size <= 0 || config.max_tiles_per_polygon == 0) {
        throw std::invalid_argument("invalid tile config");
    }
    if (old_polygons.size() != old_canonical_ids.size() || new_polygons.size() != new_canonical_ids.size()) {
        throw std::invalid_argument("canonical id vector length must match polygon count");
    }

    std::map<PairKey, std::uint8_t> pair_side_masks;
    std::map<std::tuple<LayerSpec, std::uint64_t, BBox>, std::uint8_t> oversized_side_masks;
    TilePrefilterResult result;

    for (std::size_t i = 0; i < old_polygons.size(); ++i) {
        add_side(pair_side_masks, oversized_side_masks, old_polygons[i], old_canonical_ids[i], kOldSide, config, result.assignment_count);
    }
    for (std::size_t i = 0; i < new_polygons.size(); ++i) {
        add_side(pair_side_masks, oversized_side_masks, new_polygons[i], new_canonical_ids[i], kNewSide, config, result.assignment_count);
    }

    for (const auto& [key, side_mask] : pair_side_masks) {
        if (side_mask != kBothSides) {
            result.candidate_tiles.push_back(key.first);
        }
    }
    std::sort(result.candidate_tiles.begin(), result.candidate_tiles.end());
    result.candidate_tiles.erase(std::unique(result.candidate_tiles.begin(), result.candidate_tiles.end()), result.candidate_tiles.end());
    result.collapsed_pair_count = pair_side_masks.size();

    for (const auto& [key, side_mask] : oversized_side_masks) {
        if (side_mask == kBothSides) {
            ++result.oversized_matched_count;
        } else {
            result.oversized_candidates.push_back(
                OversizedPolygon{std::get<0>(key), std::get<1>(key), side_mask, std::get<2>(key)});
        }
    }
    std::sort(result.oversized_candidates.begin(), result.oversized_candidates.end());
    return result;
}

}  // namespace gds_gpu_compare
