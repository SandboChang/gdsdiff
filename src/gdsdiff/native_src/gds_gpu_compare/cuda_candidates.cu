#include "cuda_candidates.cuh"

#include "cuda_assignment.cuh"

#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>

namespace gds_gpu_compare {
namespace {

struct AssignmentKey {
    std::int32_t layer;
    std::int32_t datatype;
    std::int64_t tx;
    std::int64_t ty;
    std::uint64_t canonical_id;

    __host__ __device__ bool operator<(const AssignmentKey& other) const {
        return thrust::make_tuple(layer, datatype, tx, ty, canonical_id)
            < thrust::make_tuple(other.layer, other.datatype, other.tx, other.ty, other.canonical_id);
    }

    __host__ __device__ bool operator==(const AssignmentKey& other) const {
        return layer == other.layer && datatype == other.datatype && tx == other.tx && ty == other.ty
            && canonical_id == other.canonical_id;
    }
};

struct AssignmentKeyEqual {
    __host__ __device__ bool operator()(const AssignmentKey& left, const AssignmentKey& right) const {
        return left == right;
    }
};

struct SideOr {
    __host__ __device__ std::uint8_t operator()(std::uint8_t left, std::uint8_t right) const {
        return static_cast<std::uint8_t>(left | right);
    }
};

void merge_oversized(
    std::map<std::tuple<LayerSpec, std::uint64_t, BBox>, std::uint8_t>& out,
    const std::vector<OversizedPolygon>& items) {
    for (const auto& item : items) {
        auto key = std::make_tuple(item.layer, item.canonical_id, item.tile_bbox);
        out[key] = static_cast<std::uint8_t>(out[key] | item.side_mask);
    }
}

}  // namespace

TilePrefilterResult run_cuda_candidate_compaction(
    const std::vector<PolygonRecord>& old_polygons,
    const std::vector<std::uint64_t>& old_canonical_ids,
    const std::vector<PolygonRecord>& new_polygons,
    const std::vector<std::uint64_t>& new_canonical_ids,
    const TileConfig& config) {
    const auto old_assignments = run_cuda_assignment(old_polygons, old_canonical_ids, kOldSide, config);
    const auto new_assignments = run_cuda_assignment(new_polygons, new_canonical_ids, kNewSide, config);

    std::vector<AssignmentKey> host_keys;
    std::vector<std::uint8_t> host_sides;
    host_keys.reserve(old_assignments.assignments.size() + new_assignments.assignments.size());
    host_sides.reserve(host_keys.capacity());
    for (const auto& item : old_assignments.assignments) {
        host_keys.push_back(AssignmentKey{item.layer.layer, item.layer.datatype, item.tx, item.ty, item.canonical_id});
        host_sides.push_back(item.side);
    }
    for (const auto& item : new_assignments.assignments) {
        host_keys.push_back(AssignmentKey{item.layer.layer, item.layer.datatype, item.tx, item.ty, item.canonical_id});
        host_sides.push_back(item.side);
    }

    TilePrefilterResult result;
    result.assignment_count = host_keys.size();

    if (!host_keys.empty()) {
        thrust::device_vector<AssignmentKey> keys = host_keys;
        thrust::device_vector<std::uint8_t> sides = host_sides;
        thrust::sort_by_key(keys.begin(), keys.end(), sides.begin());

        thrust::device_vector<AssignmentKey> reduced_keys(keys.size());
        thrust::device_vector<std::uint8_t> reduced_sides(sides.size());
        auto ends = thrust::reduce_by_key(
            keys.begin(),
            keys.end(),
            sides.begin(),
            reduced_keys.begin(),
            reduced_sides.begin(),
            AssignmentKeyEqual{},
            SideOr{});
        const auto reduced_count = static_cast<std::size_t>(ends.first - reduced_keys.begin());
        result.collapsed_pair_count = reduced_count;
        reduced_keys.resize(reduced_count);
        reduced_sides.resize(reduced_count);

        thrust::host_vector<AssignmentKey> final_keys = reduced_keys;
        thrust::host_vector<std::uint8_t> final_sides = reduced_sides;
        std::set<TileKey> candidate_tiles;
        for (std::size_t i = 0; i < reduced_count; ++i) {
            if (final_sides[i] != kBothSides) {
                candidate_tiles.insert(TileKey{LayerSpec{final_keys[i].layer, final_keys[i].datatype}, final_keys[i].tx, final_keys[i].ty});
            }
        }
        result.candidate_tiles.assign(candidate_tiles.begin(), candidate_tiles.end());
    }

    std::map<std::tuple<LayerSpec, std::uint64_t, BBox>, std::uint8_t> oversized_side_masks;
    merge_oversized(oversized_side_masks, old_assignments.oversized);
    merge_oversized(oversized_side_masks, new_assignments.oversized);
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
