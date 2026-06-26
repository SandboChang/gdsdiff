#pragma once

#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace gdsdiff {

constexpr std::uint32_t kEncodingVersion = 1;
constexpr std::uint32_t kValidityValid = 0;
constexpr std::uint32_t kValidityConservativeRaw = 1;
constexpr std::uint8_t kOldSide = 1;
constexpr std::uint8_t kNewSide = 2;
constexpr std::uint8_t kBothSides = kOldSide | kNewSide;

struct Point {
    std::int64_t x = 0;
    std::int64_t y = 0;

    friend bool operator==(const Point&, const Point&) = default;
    friend auto operator<=>(const Point&, const Point&) = default;
};

struct LayerSpec {
    std::int32_t layer = 0;
    std::int32_t datatype = 0;

    friend bool operator==(const LayerSpec&, const LayerSpec&) = default;
    friend auto operator<=>(const LayerSpec&, const LayerSpec&) = default;
};

struct BBox {
    std::int64_t x0 = 0;
    std::int64_t y0 = 0;
    std::int64_t x1 = 0;
    std::int64_t y1 = 0;

    friend bool operator==(const BBox&, const BBox&) = default;
    friend auto operator<=>(const BBox&, const BBox&) = default;
};

struct PolygonRecord {
    LayerSpec layer;
    BBox bbox;
    std::vector<Point> points;
    bool valid = true;
};

struct TileConfig {
    std::int64_t tile_size = 1024;
    std::uint64_t max_tiles_per_polygon = 4096;
};

struct TileKey {
    LayerSpec layer;
    std::int64_t tx = 0;
    std::int64_t ty = 0;

    friend bool operator==(const TileKey&, const TileKey&) = default;
    friend auto operator<=>(const TileKey&, const TileKey&) = default;
};

struct OversizedPolygon {
    LayerSpec layer;
    std::uint64_t canonical_id = 0;
    std::uint8_t side_mask = 0;
    BBox tile_bbox;

    friend bool operator==(const OversizedPolygon&, const OversizedPolygon&) = default;
    friend auto operator<=>(const OversizedPolygon&, const OversizedPolygon&) = default;
};

struct TilePrefilterResult {
    std::vector<TileKey> candidate_tiles;
    std::vector<OversizedPolygon> oversized_candidates;
    std::uint64_t assignment_count = 0;
    std::uint64_t collapsed_pair_count = 0;
    std::uint64_t oversized_matched_count = 0;
};

std::int64_t floor_div(std::int64_t value, std::int64_t divisor);
BBox bbox_tile_range(const BBox& bbox, std::int64_t tile_size);
std::string canonical_blob(const std::vector<Point>& points, bool valid);
std::vector<std::uint64_t> intern_canonical_blobs(const std::vector<std::string>& blobs);
TilePrefilterResult run_tile_prefilter(
    const std::vector<PolygonRecord>& old_polygons,
    const std::vector<std::uint64_t>& old_canonical_ids,
    const std::vector<PolygonRecord>& new_polygons,
    const std::vector<std::uint64_t>& new_canonical_ids,
    const TileConfig& config);

}  // namespace gdsdiff
