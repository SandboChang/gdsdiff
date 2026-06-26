#pragma once

#include "core.hpp"

#include <cstdint>
#include <vector>

namespace gdsdiff {

struct CudaMemoryBudget {
    std::uint64_t max_assignment_bytes = 256ull * 1024ull * 1024ull;
    std::uint64_t bytes_per_assignment = 64;
};

struct CudaChunkStats {
    std::uint64_t chunk_count = 0;
    std::uint64_t peak_estimated_assignment_bytes = 0;
    bool used_chunking = false;
};

struct CudaChunkedCandidateResult {
    TilePrefilterResult prefilter;
    CudaChunkStats stats;
};

CudaChunkedCandidateResult run_cuda_candidate_compaction_chunked(
    const std::vector<PolygonRecord>& old_polygons,
    const std::vector<std::uint64_t>& old_canonical_ids,
    const std::vector<PolygonRecord>& new_polygons,
    const std::vector<std::uint64_t>& new_canonical_ids,
    const TileConfig& config,
    const CudaMemoryBudget& budget);

}  // namespace gdsdiff
