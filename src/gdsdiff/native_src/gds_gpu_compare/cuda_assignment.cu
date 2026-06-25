#include "cuda_assignment.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>

namespace gds_gpu_compare {
namespace {

struct DevicePolygon {
    std::int64_t x0;
    std::int64_t y0;
    std::int64_t x1;
    std::int64_t y1;
    std::int32_t layer;
    std::int32_t datatype;
    std::uint64_t canonical_id;
};

struct DeviceTileAssignment {
    std::int32_t layer;
    std::int32_t datatype;
    std::int64_t tx;
    std::int64_t ty;
    std::uint64_t canonical_id;
    std::uint8_t side;
};

struct DeviceOversized {
    std::int32_t layer;
    std::int32_t datatype;
    std::uint64_t canonical_id;
    std::uint8_t side;
    std::int64_t tx0;
    std::int64_t ty0;
    std::int64_t tx1;
    std::int64_t ty1;
};

__device__ std::int64_t device_floor_div(std::int64_t value, std::int64_t divisor) {
    std::int64_t quotient = value / divisor;
    const std::int64_t remainder = value % divisor;
    if (remainder != 0 && ((remainder < 0) != (divisor < 0))) {
        --quotient;
    }
    return quotient;
}

__global__ void count_kernel(
    const DevicePolygon* polygons,
    std::uint64_t n,
    std::int64_t tile_size,
    std::uint64_t max_tiles_per_polygon,
    std::uint64_t* counts,
    std::uint8_t* oversized_flags,
    DeviceOversized* oversized,
    std::uint8_t side,
    std::uint32_t* error_flags) {
    const std::uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }
    const DevicePolygon polygon = polygons[index];
    const std::int64_t tx0 = device_floor_div(polygon.x0, tile_size);
    const std::int64_t ty0 = device_floor_div(polygon.y0, tile_size);
    const std::int64_t tx1 = device_floor_div(polygon.x1, tile_size);
    const std::int64_t ty1 = device_floor_div(polygon.y1, tile_size);
    const std::int64_t span_x = tx1 - tx0 + 1;
    const std::int64_t span_y = ty1 - ty0 + 1;
    if (span_x <= 0 || span_y <= 0) {
        error_flags[index] = 1;
        counts[index] = 0;
        oversized_flags[index] = 0;
        return;
    }
    const auto span_x_u = static_cast<std::uint64_t>(span_x);
    const auto span_y_u = static_cast<std::uint64_t>(span_y);
    if (span_y_u != 0 && span_x_u > UINT64_MAX / span_y_u) {
        error_flags[index] = 2;
        counts[index] = 0;
        oversized_flags[index] = 0;
        return;
    }
    const std::uint64_t count = span_x_u * span_y_u;
    if (count > max_tiles_per_polygon) {
        counts[index] = 0;
        oversized_flags[index] = 1;
        oversized[index] = DeviceOversized{
            polygon.layer,
            polygon.datatype,
            polygon.canonical_id,
            side,
            tx0,
            ty0,
            tx1,
            ty1,
        };
    } else {
        counts[index] = count;
        oversized_flags[index] = 0;
    }
}

__global__ void emit_kernel(
    const DevicePolygon* polygons,
    std::uint64_t n,
    std::int64_t tile_size,
    const std::uint64_t* offsets,
    const std::uint8_t* oversized_flags,
    DeviceTileAssignment* assignments,
    std::uint8_t side) {
    const std::uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n || oversized_flags[index]) {
        return;
    }
    const DevicePolygon polygon = polygons[index];
    const std::int64_t tx0 = device_floor_div(polygon.x0, tile_size);
    const std::int64_t ty0 = device_floor_div(polygon.y0, tile_size);
    const std::int64_t tx1 = device_floor_div(polygon.x1, tile_size);
    const std::int64_t ty1 = device_floor_div(polygon.y1, tile_size);
    std::uint64_t out = offsets[index];
    for (std::int64_t tx = tx0; tx <= tx1; ++tx) {
        for (std::int64_t ty = ty0; ty <= ty1; ++ty) {
            assignments[out++] = DeviceTileAssignment{
                polygon.layer,
                polygon.datatype,
                tx,
                ty,
                polygon.canonical_id,
                side,
            };
        }
    }
}

void check_cuda(cudaError_t status, const char* action) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(status));
    }
}

template <class T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        if (count_) {
            check_cuda(cudaMalloc(&ptr_, sizeof(T) * count_), "cudaMalloc");
        }
    }

    ~DeviceBuffer() {
        if (ptr_) {
            (void)cudaFree(ptr_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }

private:
    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

std::vector<DevicePolygon> pack_polygons(const std::vector<PolygonRecord>& polygons, const std::vector<std::uint64_t>& ids) {
    std::vector<DevicePolygon> out;
    out.reserve(polygons.size());
    for (std::size_t i = 0; i < polygons.size(); ++i) {
        out.push_back(DevicePolygon{
            polygons[i].bbox.x0,
            polygons[i].bbox.y0,
            polygons[i].bbox.x1,
            polygons[i].bbox.y1,
            polygons[i].layer.layer,
            polygons[i].layer.datatype,
            ids[i],
        });
    }
    return out;
}

}  // namespace

CudaAssignmentResult run_cuda_assignment(
    const std::vector<PolygonRecord>& polygons,
    const std::vector<std::uint64_t>& canonical_ids,
    std::uint8_t side,
    const TileConfig& config) {
    if (config.tile_size <= 0 || config.max_tiles_per_polygon == 0) {
        throw std::invalid_argument("invalid tile config");
    }
    if (polygons.size() != canonical_ids.size()) {
        throw std::invalid_argument("canonical id vector length must match polygon count");
    }
    if (polygons.empty()) {
        return {};
    }

    const std::uint64_t n = polygons.size();
    const auto packed = pack_polygons(polygons, canonical_ids);
    DeviceBuffer<DevicePolygon> d_polygons(n);
    DeviceBuffer<std::uint64_t> d_counts(n);
    DeviceBuffer<std::uint8_t> d_oversized_flags(n);
    DeviceBuffer<DeviceOversized> d_oversized(n);
    DeviceBuffer<std::uint32_t> d_errors(n);
    check_cuda(cudaMemcpy(d_polygons.get(), packed.data(), sizeof(DevicePolygon) * packed.size(), cudaMemcpyHostToDevice), "copy polygons");
    check_cuda(cudaMemset(d_errors.get(), 0, sizeof(std::uint32_t) * n), "clear errors");

    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    count_kernel<<<blocks, threads>>>(
        d_polygons.get(),
        n,
        config.tile_size,
        config.max_tiles_per_polygon,
        d_counts.get(),
        d_oversized_flags.get(),
        d_oversized.get(),
        side,
        d_errors.get());
    check_cuda(cudaGetLastError(), "launch count kernel");
    check_cuda(cudaDeviceSynchronize(), "run count kernel");

    std::vector<std::uint32_t> errors(n);
    check_cuda(cudaMemcpy(errors.data(), d_errors.get(), sizeof(std::uint32_t) * n, cudaMemcpyDeviceToHost), "copy errors");
    for (std::uint32_t error : errors) {
        if (error) {
            throw std::overflow_error("CUDA assignment count overflow or invalid span");
        }
    }

    std::vector<std::uint64_t> counts(n);
    std::vector<std::uint8_t> oversized_flags(n);
    std::vector<DeviceOversized> oversized_raw(n);
    check_cuda(cudaMemcpy(counts.data(), d_counts.get(), sizeof(std::uint64_t) * n, cudaMemcpyDeviceToHost), "copy counts");
    check_cuda(cudaMemcpy(oversized_flags.data(), d_oversized_flags.get(), sizeof(std::uint8_t) * n, cudaMemcpyDeviceToHost), "copy oversized flags");
    check_cuda(cudaMemcpy(oversized_raw.data(), d_oversized.get(), sizeof(DeviceOversized) * n, cudaMemcpyDeviceToHost), "copy oversized");

    std::vector<std::uint64_t> offsets(n);
    std::uint64_t total = 0;
    for (std::size_t i = 0; i < counts.size(); ++i) {
        offsets[i] = total;
        if (counts[i] > UINT64_MAX - total) {
            throw std::overflow_error("total assignment count overflow");
        }
        total += counts[i];
    }

    CudaAssignmentResult result;
    result.total_assignment_count = total;
    for (std::size_t i = 0; i < oversized_flags.size(); ++i) {
        if (oversized_flags[i]) {
            const auto item = oversized_raw[i];
            result.oversized.push_back(
                OversizedPolygon{
                    LayerSpec{item.layer, item.datatype},
                    item.canonical_id,
                    item.side,
                    BBox{item.tx0, item.ty0, item.tx1, item.ty1},
                });
        }
    }

    if (total == 0) {
        return result;
    }

    DeviceBuffer<std::uint64_t> d_offsets(n);
    DeviceBuffer<DeviceTileAssignment> d_assignments(total);
    check_cuda(cudaMemcpy(d_offsets.get(), offsets.data(), sizeof(std::uint64_t) * n, cudaMemcpyHostToDevice), "copy offsets");
    emit_kernel<<<blocks, threads>>>(
        d_polygons.get(),
        n,
        config.tile_size,
        d_offsets.get(),
        d_oversized_flags.get(),
        d_assignments.get(),
        side);
    check_cuda(cudaGetLastError(), "launch emit kernel");
    check_cuda(cudaDeviceSynchronize(), "run emit kernel");

    std::vector<DeviceTileAssignment> raw_assignments(total);
    check_cuda(cudaMemcpy(raw_assignments.data(), d_assignments.get(), sizeof(DeviceTileAssignment) * total, cudaMemcpyDeviceToHost), "copy assignments");
    result.assignments.reserve(raw_assignments.size());
    for (const auto item : raw_assignments) {
        result.assignments.push_back(
            TileAssignment{
                LayerSpec{item.layer, item.datatype},
                item.tx,
                item.ty,
                item.canonical_id,
                item.side,
            });
    }
    return result;
}

}  // namespace gds_gpu_compare
