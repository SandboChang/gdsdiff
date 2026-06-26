#include "cuda_memory.cuh"

#include "cuda_candidates.cuh"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace gdsdiff {
namespace {

struct IndexedPolygon {
    PolygonRecord polygon;
    std::uint64_t canonical_id;
    BBox tile_bbox;
};

std::vector<IndexedPolygon> index_polygons(const std::vector<PolygonRecord>& polygons, const std::vector<std::uint64_t>& ids, const TileConfig& config) {
    if (polygons.size() != ids.size()) {
        throw std::invalid_argument("canonical id vector length must match polygon count");
    }
    std::vector<IndexedPolygon> indexed;
    indexed.reserve(polygons.size());
    for (std::size_t i = 0; i < polygons.size(); ++i) {
        indexed.push_back(IndexedPolygon{polygons[i], ids[i], bbox_tile_range(polygons[i].bbox, config.tile_size)});
    }
    return indexed;
}

std::uint64_t tile_count(const BBox& tile_bbox) {
    const std::int64_t span_x = tile_bbox.x1 - tile_bbox.x0 + 1;
    const std::int64_t span_y = tile_bbox.y1 - tile_bbox.y0 + 1;
    if (span_x <= 0 || span_y <= 0) {
        throw std::overflow_error("invalid tile span");
    }
    const auto span_x_u = static_cast<std::uint64_t>(span_x);
    const auto span_y_u = static_cast<std::uint64_t>(span_y);
    if (span_y_u != 0 && span_x_u > std::numeric_limits<std::uint64_t>::max() / span_y_u) {
        throw std::overflow_error("tile span overflow");
    }
    return span_x_u * span_y_u;
}

std::uint64_t estimate_assignment_bytes(
    const std::vector<IndexedPolygon>& old_indexed,
    const std::vector<IndexedPolygon>& new_indexed,
    const TileConfig& config,
    const CudaMemoryBudget& budget,
    std::int64_t row0,
    std::int64_t row1) {
    std::uint64_t assignments = 0;
    auto add = [&](const std::vector<IndexedPolygon>& polygons) {
        for (const auto& item : polygons) {
            if (item.tile_bbox.y1 < row0 || item.tile_bbox.y0 > row1) {
                continue;
            }
            const std::uint64_t count = tile_count(item.tile_bbox);
            if (count <= config.max_tiles_per_polygon) {
                if (assignments > std::numeric_limits<std::uint64_t>::max() - count) {
                    throw std::overflow_error("assignment estimate overflow");
                }
                assignments += count;
            }
        }
    };
    add(old_indexed);
    add(new_indexed);
    if (budget.bytes_per_assignment != 0 && assignments > std::numeric_limits<std::uint64_t>::max() / budget.bytes_per_assignment) {
        throw std::overflow_error("assignment byte estimate overflow");
    }
    return assignments * budget.bytes_per_assignment;
}

void subset_for_rows(
    const std::vector<IndexedPolygon>& indexed,
    std::int64_t row0,
    std::int64_t row1,
    std::vector<PolygonRecord>& polygons,
    std::vector<std::uint64_t>& ids) {
    for (const auto& item : indexed) {
        if (item.tile_bbox.y1 < row0 || item.tile_bbox.y0 > row1) {
            continue;
        }
        polygons.push_back(item.polygon);
        ids.push_back(item.canonical_id);
    }
}

}  // namespace

CudaChunkedCandidateResult run_cuda_candidate_compaction_chunked(
    const std::vector<PolygonRecord>& old_polygons,
    const std::vector<std::uint64_t>& old_canonical_ids,
    const std::vector<PolygonRecord>& new_polygons,
    const std::vector<std::uint64_t>& new_canonical_ids,
    const TileConfig& config,
    const CudaMemoryBudget& budget) {
    if (budget.bytes_per_assignment == 0 || budget.max_assignment_bytes == 0) {
        throw std::invalid_argument("CUDA memory budget values must be positive");
    }
    const auto old_indexed = index_polygons(old_polygons, old_canonical_ids, config);
    const auto new_indexed = index_polygons(new_polygons, new_canonical_ids, config);
    if (old_indexed.empty() && new_indexed.empty()) {
        CudaChunkedCandidateResult result;
        result.stats.chunk_count = 1;
        return result;
    }

    std::int64_t min_row = std::numeric_limits<std::int64_t>::max();
    std::int64_t max_row = std::numeric_limits<std::int64_t>::min();
    auto expand = [&](const std::vector<IndexedPolygon>& indexed) {
        for (const auto& item : indexed) {
            min_row = std::min(min_row, item.tile_bbox.y0);
            max_row = std::max(max_row, item.tile_bbox.y1);
        }
    };
    expand(old_indexed);
    expand(new_indexed);

    const std::uint64_t whole_estimate = estimate_assignment_bytes(old_indexed, new_indexed, config, budget, min_row, max_row);
    if (whole_estimate <= budget.max_assignment_bytes) {
        CudaChunkedCandidateResult result;
        result.prefilter = run_cuda_candidate_compaction(old_polygons, old_canonical_ids, new_polygons, new_canonical_ids, config);
        result.stats.chunk_count = 1;
        result.stats.peak_estimated_assignment_bytes = whole_estimate;
        result.stats.used_chunking = false;
        return result;
    }

    std::set<TileKey> candidate_tiles;
    std::map<std::tuple<LayerSpec, std::uint64_t, BBox>, std::uint8_t> oversized_side_masks;
    TilePrefilterResult merged;
    CudaChunkStats stats;
    stats.used_chunking = true;

    std::int64_t row = min_row;
    while (row <= max_row) {
        std::int64_t chunk_end = row;
        std::uint64_t estimate = estimate_assignment_bytes(old_indexed, new_indexed, config, budget, row, chunk_end);
        while (chunk_end < max_row) {
            const std::uint64_t next_estimate = estimate_assignment_bytes(old_indexed, new_indexed, config, budget, row, chunk_end + 1);
            if (next_estimate > budget.max_assignment_bytes && chunk_end >= row) {
                break;
            }
            estimate = next_estimate;
            ++chunk_end;
            if (estimate >= budget.max_assignment_bytes) {
                break;
            }
        }
        stats.peak_estimated_assignment_bytes = std::max(stats.peak_estimated_assignment_bytes, estimate);
        ++stats.chunk_count;

        std::vector<PolygonRecord> old_subset;
        std::vector<PolygonRecord> new_subset;
        std::vector<std::uint64_t> old_ids;
        std::vector<std::uint64_t> new_ids;
        subset_for_rows(old_indexed, row, chunk_end, old_subset, old_ids);
        subset_for_rows(new_indexed, row, chunk_end, new_subset, new_ids);
        auto chunk = run_cuda_candidate_compaction(old_subset, old_ids, new_subset, new_ids, config);
        merged.assignment_count += chunk.assignment_count;
        merged.collapsed_pair_count += chunk.collapsed_pair_count;
        for (const auto& tile : chunk.candidate_tiles) {
            if (tile.ty >= row && tile.ty <= chunk_end) {
                candidate_tiles.insert(tile);
            }
        }
        for (const auto& item : chunk.oversized_candidates) {
            auto key = std::make_tuple(item.layer, item.canonical_id, item.tile_bbox);
            oversized_side_masks[key] = static_cast<std::uint8_t>(oversized_side_masks[key] | item.side_mask);
        }
        row = chunk_end + 1;
    }

    merged.candidate_tiles.assign(candidate_tiles.begin(), candidate_tiles.end());
    for (const auto& [key, side_mask] : oversized_side_masks) {
        if (side_mask == kBothSides) {
            ++merged.oversized_matched_count;
        } else {
            merged.oversized_candidates.push_back(
                OversizedPolygon{std::get<0>(key), std::get<1>(key), side_mask, std::get<2>(key)});
        }
    }
    std::sort(merged.oversized_candidates.begin(), merged.oversized_candidates.end());

    CudaChunkedCandidateResult result;
    result.prefilter = std::move(merged);
    result.stats = stats;
    return result;
}

}  // namespace gdsdiff
