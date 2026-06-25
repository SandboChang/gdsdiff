#pragma once

#include "core.hpp"

#include <cstdint>
#include <vector>

namespace gds_gpu_compare {

struct TileAssignment {
    LayerSpec layer;
    std::int64_t tx = 0;
    std::int64_t ty = 0;
    std::uint64_t canonical_id = 0;
    std::uint8_t side = 0;

    friend bool operator==(const TileAssignment&, const TileAssignment&) = default;
    friend auto operator<=>(const TileAssignment&, const TileAssignment&) = default;
};

struct CudaAssignmentResult {
    std::vector<TileAssignment> assignments;
    std::vector<OversizedPolygon> oversized;
    std::uint64_t total_assignment_count = 0;
};

CudaAssignmentResult run_cuda_assignment(
    const std::vector<PolygonRecord>& polygons,
    const std::vector<std::uint64_t>& canonical_ids,
    std::uint8_t side,
    const TileConfig& config);

}  // namespace gds_gpu_compare
