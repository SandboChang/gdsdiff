#pragma once

#include "core.hpp"

#include <cstdint>
#include <vector>

namespace gdsdiff {

TilePrefilterResult run_cuda_candidate_compaction(
    const std::vector<PolygonRecord>& old_polygons,
    const std::vector<std::uint64_t>& old_canonical_ids,
    const std::vector<PolygonRecord>& new_polygons,
    const std::vector<std::uint64_t>& new_canonical_ids,
    const TileConfig& config);

}  // namespace gdsdiff
