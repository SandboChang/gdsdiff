#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/remove.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

constexpr std::uint8_t kOldSide = 1;
constexpr std::uint8_t kNewSide = 2;
constexpr std::uint8_t kBothSides = kOldSide | kNewSide;
constexpr int kMaxRectSetRects = 16;
constexpr int kMaxRectSetCoords = kMaxRectSetRects * 4;
constexpr int kMaxPolygonComponentEdges = 384;
constexpr int kMaxPolygonComponentBreaks = 2048;
constexpr int kMaxPolygonComponentIntervals = 384;

thread_local std::string g_last_error;

struct DevicePolygon {
    std::int64_t x0;
    std::int64_t y0;
    std::int64_t x1;
    std::int64_t y1;
    std::int32_t layer;
    std::int32_t datatype;
    std::uint64_t canonical_id;
};

struct DeviceAssignment {
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

struct DeviceAssignmentToKey {
    __host__ __device__ AssignmentKey operator()(const DeviceAssignment& item) const {
        return AssignmentKey{item.layer, item.datatype, item.tx, item.ty, item.canonical_id};
    }
};

struct DeviceAssignmentToSide {
    __host__ __device__ std::uint8_t operator()(const DeviceAssignment& item) const {
        return item.side;
    }
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

struct SideAssignments {
    thrust::device_vector<DeviceAssignment> assignments;
    std::vector<OversizedItem> oversized_items;
    std::uint64_t assignment_count = 0;
};

struct AxisRawEdge {
    std::int64_t x0 = 0;
    std::int64_t y0 = 0;
    std::int64_t x1 = 0;
    std::int64_t y1 = 0;
    std::int64_t line = 0;
    std::int64_t p0 = 0;
    std::int64_t p1 = 0;
    std::uint8_t vertical = 0;
    std::uint8_t valid = 0;
};

struct AxisEndpointKey {
    std::uint8_t vertical = 0;
    std::int64_t line = 0;
    std::int64_t projection = 0;

    __host__ __device__ bool operator<(const AxisEndpointKey& other) const {
        return thrust::make_tuple(vertical, line, projection)
            < thrust::make_tuple(other.vertical, other.line, other.projection);
    }

    __host__ __device__ bool operator==(const AxisEndpointKey& other) const {
        return vertical == other.vertical && line == other.line && projection == other.projection;
    }
};

struct AxisEndpointInvalid {
    __host__ __device__ bool operator()(const AxisEndpointKey& key) const {
        return key.vertical > 1;
    }
};

struct EdgeKey {
    std::int64_t x0 = 0;
    std::int64_t y0 = 0;
    std::int64_t x1 = 0;
    std::int64_t y1 = 0;

    __host__ __device__ bool operator<(const EdgeKey& other) const {
        return thrust::make_tuple(x0, y0, x1, y1) < thrust::make_tuple(other.x0, other.y0, other.x1, other.y1);
    }

    __host__ __device__ bool operator==(const EdgeKey& other) const {
        return x0 == other.x0 && y0 == other.y0 && x1 == other.x1 && y1 == other.y1;
    }
};

struct EdgeInvalid {
    __host__ __device__ bool operator()(const EdgeKey& edge) const {
        return edge.x0 == edge.x1 && edge.y0 == edge.y1;
    }
};

template <class T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        if (count_) {
            auto status = cudaMalloc(&ptr_, sizeof(T) * count_);
            if (status != cudaSuccess) {
                throw std::runtime_error(std::string("cudaMalloc: ") + cudaGetErrorString(status));
            }
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

private:
    T* ptr_ = nullptr;
    std::size_t count_ = 0;
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
        oversized[index] = DeviceOversized{polygon.layer, polygon.datatype, polygon.canonical_id, side, tx0, ty0, tx1, ty1};
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
    DeviceAssignment* assignments,
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
            assignments[out++] = DeviceAssignment{polygon.layer, polygon.datatype, tx, ty, polygon.canonical_id, side};
        }
    }
}

__global__ void byte_compare_kernel(const std::uint8_t* old_data, const std::uint8_t* new_data, std::uint64_t n, std::uint32_t* mismatch) {
    const std::uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n || *mismatch) {
        return;
    }
    if (old_data[index] != new_data[index]) {
        atomicExch(mismatch, 1u);
    }
}

struct PolygonRecordKey {
    std::int32_t layer;
    std::int32_t datatype;
    std::uint32_t flags;
    std::uint64_t vertex_count;
    std::int64_t x0;
    std::int64_t y0;
    std::int64_t x1;
    std::int64_t y1;
    std::uint64_t hash0;
    std::uint64_t hash1;

    __host__ __device__ bool operator<(const PolygonRecordKey& other) const {
        if (layer != other.layer) return layer < other.layer;
        if (datatype != other.datatype) return datatype < other.datatype;
        if (flags != other.flags) return flags < other.flags;
        if (vertex_count != other.vertex_count) return vertex_count < other.vertex_count;
        if (x0 != other.x0) return x0 < other.x0;
        if (y0 != other.y0) return y0 < other.y0;
        if (x1 != other.x1) return x1 < other.x1;
        if (y1 != other.y1) return y1 < other.y1;
        if (hash0 != other.hash0) return hash0 < other.hash0;
        return hash1 < other.hash1;
    }

    __host__ __device__ bool operator==(const PolygonRecordKey& other) const {
        return layer == other.layer && datatype == other.datatype && flags == other.flags
            && vertex_count == other.vertex_count && x0 == other.x0 && y0 == other.y0 && x1 == other.x1
            && y1 == other.y1 && hash0 == other.hash0 && hash1 == other.hash1;
    }
};

struct PolygonSideRecord {
    PolygonRecordKey key;
    std::uint8_t side;
    std::uint64_t index;
};

struct PolygonCountPair {
    std::uint64_t old_count;
    std::uint64_t new_count;
    std::uint64_t old_index;
    std::uint64_t new_index;
};

struct PolygonRecordToKey {
    __host__ __device__ PolygonRecordKey operator()(const PolygonSideRecord& item) const {
        return item.key;
    }
};

struct PolygonSideRecordLess {
    __host__ __device__ bool operator()(const PolygonSideRecord& left, const PolygonSideRecord& right) const {
        return left.key < right.key;
    }
};

struct PolygonRecordToCountPair {
    __host__ __device__ PolygonCountPair operator()(const PolygonSideRecord& item) const {
        constexpr std::uint64_t none = UINT64_MAX;
        return item.side == kOldSide ? PolygonCountPair{1, 0, item.index, none} : PolygonCountPair{0, 1, none, item.index};
    }
};

struct PolygonKeyEqual {
    __host__ __device__ bool operator()(const PolygonRecordKey& left, const PolygonRecordKey& right) const {
        return left == right;
    }
};

struct PolygonCountPairAdd {
    __host__ __device__ PolygonCountPair operator()(const PolygonCountPair& left, const PolygonCountPair& right) const {
        return PolygonCountPair{
            left.old_count + right.old_count,
            left.new_count + right.new_count,
            min(left.old_index, right.old_index),
            min(left.new_index, right.new_index),
        };
    }
};

__device__ std::uint64_t fnv1a_u64_device(std::uint64_t hash, std::uint64_t value) {
    constexpr std::uint64_t prime = 1099511628211ull;
    for (int i = 0; i < 8; ++i) {
        hash ^= (value >> (i * 8)) & 0xffu;
        hash *= prime;
    }
    return hash;
}

__device__ bool point_less_device(
    const std::int64_t* vertices_xy,
    std::uint64_t left_index,
    std::uint64_t right_index) {
    const std::int64_t lx = vertices_xy[left_index * 2 + 0];
    const std::int64_t ly = vertices_xy[left_index * 2 + 1];
    const std::int64_t rx = vertices_xy[right_index * 2 + 0];
    const std::int64_t ry = vertices_xy[right_index * 2 + 1];
    return lx < rx || (lx == rx && ly < ry);
}

__device__ std::uint64_t sequence_index_device(
    std::uint64_t start,
    std::uint64_t count,
    std::uint64_t candidate,
    std::uint64_t step,
    bool forward) {
    if (forward) {
        return start + ((candidate + step) % count);
    }
    return start + ((candidate + count - (step % count)) % count);
}

__device__ bool sequence_less_device(
    const std::int64_t* vertices_xy,
    std::uint64_t start,
    std::uint64_t count,
    std::uint64_t left_candidate,
    bool left_forward,
    std::uint64_t right_candidate,
    bool right_forward) {
    for (std::uint64_t step = 0; step < count; ++step) {
        const std::uint64_t left_index = sequence_index_device(start, count, left_candidate, step, left_forward);
        const std::uint64_t right_index = sequence_index_device(start, count, right_candidate, step, right_forward);
        if (point_less_device(vertices_xy, left_index, right_index)) {
            return true;
        }
        if (point_less_device(vertices_xy, right_index, left_index)) {
            return false;
        }
    }
    return false;
}

__device__ bool is_axis_aligned_bbox_rectangle_device(
    const std::int64_t* vertices_xy,
    std::uint64_t start,
    std::uint64_t count,
    std::int64_t x0,
    std::int64_t y0,
    std::int64_t x1,
    std::int64_t y1) {
    if (count != 4 || x0 >= x1 || y0 >= y1) {
        return false;
    }
    bool corners[4] = {false, false, false, false};
    for (std::uint64_t offset = 0; offset < count; ++offset) {
        const auto x = vertices_xy[(start + offset) * 2 + 0];
        const auto y = vertices_xy[(start + offset) * 2 + 1];
        int corner = -1;
        if (x == x0 && y == y0) corner = 0;
        if (x == x0 && y == y1) corner = 1;
        if (x == x1 && y == y0) corner = 2;
        if (x == x1 && y == y1) corner = 3;
        if (corner < 0 || corners[corner]) {
            return false;
        }
        corners[corner] = true;
    }
    return corners[0] && corners[1] && corners[2] && corners[3];
}

__device__ void canonical_polygon_hash_device(
    const std::int64_t* vertices_xy,
    std::uint64_t start,
    std::uint64_t stop,
    std::uint32_t flags,
    std::int64_t x0,
    std::int64_t y0,
    std::int64_t x1,
    std::int64_t y1,
    std::uint64_t* hash0,
    std::uint64_t* hash1) {
    const std::uint64_t count = stop - start;
    std::uint64_t h0 = 1469598103934665603ull;
    std::uint64_t h1 = 1099511628211ull;
    if (flags == 0 && is_axis_aligned_bbox_rectangle_device(vertices_xy, start, count, x0, y0, x1, y1)) {
        h0 = fnv1a_u64_device(fnv1a_u64_device(h0, 2), 0);
        h0 = fnv1a_u64_device(fnv1a_u64_device(h0, static_cast<std::uint64_t>(x0)), static_cast<std::uint64_t>(y0));
        h0 = fnv1a_u64_device(fnv1a_u64_device(h0, static_cast<std::uint64_t>(x1)), static_cast<std::uint64_t>(y1));
        h1 = fnv1a_u64_device(fnv1a_u64_device(h1, static_cast<std::uint64_t>(y1)), static_cast<std::uint64_t>(x1));
        h1 = fnv1a_u64_device(fnv1a_u64_device(h1, static_cast<std::uint64_t>(y0)), static_cast<std::uint64_t>(x0));
        *hash0 = h0;
        *hash1 = h1;
        return;
    }
    std::uint64_t best_candidate = 0;
    bool best_forward = true;
    if (flags == 0 && count > 0) {
        for (std::uint64_t candidate = 1; candidate < count; ++candidate) {
            if (sequence_less_device(vertices_xy, start, count, candidate, true, best_candidate, best_forward)) {
                best_candidate = candidate;
                best_forward = true;
            }
        }
        for (std::uint64_t candidate = 0; candidate < count; ++candidate) {
            if (sequence_less_device(vertices_xy, start, count, candidate, false, best_candidate, best_forward)) {
                best_candidate = candidate;
                best_forward = false;
            }
        }
    }
    for (std::uint64_t step = 0; step < count; ++step) {
        const std::uint64_t vertex = flags == 0
            ? sequence_index_device(start, count, best_candidate, step, best_forward)
            : start + step;
        const auto x = static_cast<std::uint64_t>(vertices_xy[vertex * 2 + 0]);
        const auto y = static_cast<std::uint64_t>(vertices_xy[vertex * 2 + 1]);
        h0 = fnv1a_u64_device(fnv1a_u64_device(h0, x), y);
        h1 = fnv1a_u64_device(fnv1a_u64_device(h1, y), x);
    }
    *hash0 = h0;
    *hash1 = h1;
}

__global__ void polygon_record_key_kernel(
    const std::int64_t* vertices_xy,
    const std::uint64_t* polygon_offsets,
    const std::int32_t* layers,
    const std::int32_t* datatypes,
    const std::int64_t* bboxes,
    const std::uint32_t* flags,
    std::uint64_t count,
    std::uint8_t side,
    std::uint64_t out_offset,
    PolygonSideRecord* out_records) {
    const std::uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const std::uint64_t start = polygon_offsets[index];
    const std::uint64_t stop = polygon_offsets[index + 1];
    std::uint64_t hash0 = 0;
    std::uint64_t hash1 = 0;
    canonical_polygon_hash_device(
        vertices_xy,
        start,
        stop,
        flags[index],
        bboxes[index * 4 + 0],
        bboxes[index * 4 + 1],
        bboxes[index * 4 + 2],
        bboxes[index * 4 + 3],
        &hash0,
        &hash1);
    out_records[out_offset + index] = PolygonSideRecord{
        PolygonRecordKey{
            layers[index],
            datatypes[index],
            flags[index],
            stop - start,
            bboxes[index * 4 + 0],
            bboxes[index * 4 + 1],
            bboxes[index * 4 + 2],
            bboxes[index * 4 + 3],
            hash0,
            hash1,
        },
        side,
        index,
    };
}

__global__ void polygon_record_key_from_hash_kernel(
    const std::uint64_t* polygon_offsets,
    const std::int32_t* layers,
    const std::int32_t* datatypes,
    const std::int64_t* bboxes,
    const std::uint32_t* flags,
    const std::uint64_t* hashes,
    std::uint64_t count,
    std::uint8_t side,
    std::uint64_t out_offset,
    PolygonSideRecord* out_records) {
    const std::uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const std::uint64_t start = polygon_offsets[index];
    const std::uint64_t stop = polygon_offsets[index + 1];
    out_records[out_offset + index] = PolygonSideRecord{
        PolygonRecordKey{
            layers[index],
            datatypes[index],
            flags[index],
            stop - start,
            bboxes[index * 4 + 0],
            bboxes[index * 4 + 1],
            bboxes[index * 4 + 2],
            bboxes[index * 4 + 3],
            hashes[index * 2 + 0],
            hashes[index * 2 + 1],
        },
        side,
        index,
    };
}

struct PolygonSweepEdge {
    double x0;
    double y0;
    double x1;
    double y1;
    std::uint8_t side;
    int polygon_slot;
};

struct PolygonSweepCrossing {
    double y;
    int edge_index;
};

struct PolygonSweepInterval {
    double low_y;
    double high_y;
    int low_edge_index;
    int high_edge_index;
};

__device__ double edge_y_at(const PolygonSweepEdge& edge, double x) {
    const double dx = edge.x1 - edge.x0;
    if (fabs(dx) < 1e-12) {
        return edge.y0;
    }
    const double t = (x - edge.x0) / dx;
    return edge.y0 + t * (edge.y1 - edge.y0);
}

__device__ bool add_breakpoint(double* breaks, int* break_count, double value) {
    if (*break_count >= kMaxPolygonComponentBreaks || !isfinite(value)) {
        return false;
    }
    for (int i = 0; i < *break_count; ++i) {
        if (fabs(breaks[i] - value) <= 1e-7) {
            return true;
        }
    }
    breaks[(*break_count)++] = value;
    return true;
}

__device__ void sort_breakpoints(double* breaks, int count) {
    for (int i = 1; i < count; ++i) {
        const double value = breaks[i];
        int j = i - 1;
        while (j >= 0 && breaks[j] > value) {
            breaks[j + 1] = breaks[j];
            --j;
        }
        breaks[j + 1] = value;
    }
}

__device__ void sort_crossings(PolygonSweepCrossing* crossings, int count) {
    for (int i = 1; i < count; ++i) {
        const auto value = crossings[i];
        int j = i - 1;
        while (j >= 0 && crossings[j].y > value.y) {
            crossings[j + 1] = crossings[j];
            --j;
        }
        crossings[j + 1] = value;
    }
}

__device__ void sort_intervals(PolygonSweepInterval* intervals, int count) {
    for (int i = 1; i < count; ++i) {
        const auto value = intervals[i];
        int j = i - 1;
        while (j >= 0 && intervals[j].low_y > value.low_y) {
            intervals[j + 1] = intervals[j];
            --j;
        }
        intervals[j + 1] = value;
    }
}

__device__ bool edge_intersection_x(const PolygonSweepEdge& a, const PolygonSweepEdge& b, double* out_x) {
    const double ax = a.x1 - a.x0;
    const double ay = a.y1 - a.y0;
    const double bx = b.x1 - b.x0;
    const double by = b.y1 - b.y0;
    const double denom = ax * by - ay * bx;
    if (fabs(denom) < 1e-12) {
        return false;
    }
    const double cx = b.x0 - a.x0;
    const double cy = b.y0 - a.y0;
    const double ta = (cx * by - cy * bx) / denom;
    const double tb = (cx * ay - cy * ax) / denom;
    if (ta < -1e-10 || ta > 1.0 + 1e-10 || tb < -1e-10 || tb > 1.0 + 1e-10) {
        return false;
    }
    *out_x = a.x0 + ta * ax;
    return true;
}

__device__ bool build_union_intervals(
    const PolygonSweepEdge* edges,
    int edge_count,
    int polygon_count,
    std::uint8_t side,
    double x_mid,
    PolygonSweepCrossing* crossings,
    PolygonSweepInterval* intervals,
    int* interval_count) {
    int raw_interval_count = 0;
    for (int polygon_slot = 0; polygon_slot < polygon_count; ++polygon_slot) {
        int crossing_count = 0;
        for (int edge_index = 0; edge_index < edge_count; ++edge_index) {
            const auto& edge = edges[edge_index];
            if (edge.side != side || edge.polygon_slot != polygon_slot) {
                continue;
            }
            const double min_x = fmin(edge.x0, edge.x1);
            const double max_x = fmax(edge.x0, edge.x1);
            if (x_mid <= min_x + 1e-9 || x_mid >= max_x - 1e-9) {
                continue;
            }
            if (crossing_count >= kMaxPolygonComponentIntervals) {
                return false;
            }
            crossings[crossing_count++] = PolygonSweepCrossing{edge_y_at(edge, x_mid), edge_index};
        }
        sort_crossings(crossings, crossing_count);
        for (int i = 0; i + 1 < crossing_count; i += 2) {
            PolygonSweepInterval next{
                crossings[i].y,
                crossings[i + 1].y,
                crossings[i].edge_index,
                crossings[i + 1].edge_index,
            };
            if (next.high_y <= next.low_y + 1e-9) {
                continue;
            }
            if (raw_interval_count >= kMaxPolygonComponentIntervals) {
                return false;
            }
            intervals[raw_interval_count++] = next;
        }
    }
    sort_intervals(intervals, raw_interval_count);
    *interval_count = 0;
    for (int i = 0; i < raw_interval_count; ++i) {
        const auto next = intervals[i];
        if (*interval_count == 0 || next.low_y > intervals[*interval_count - 1].high_y + 1e-9) {
            intervals[(*interval_count)++] = next;
        } else if (next.high_y > intervals[*interval_count - 1].high_y) {
            intervals[*interval_count - 1].high_y = next.high_y;
            intervals[*interval_count - 1].high_edge_index = next.high_edge_index;
        }
    }
    return true;
}

__device__ double interval_area(
    const PolygonSweepEdge* edges,
    const PolygonSweepInterval& interval,
    double x_left,
    double x_right) {
    const double low_left = edge_y_at(edges[interval.low_edge_index], x_left);
    const double low_right = edge_y_at(edges[interval.low_edge_index], x_right);
    const double high_left = edge_y_at(edges[interval.high_edge_index], x_left);
    const double high_right = edge_y_at(edges[interval.high_edge_index], x_right);
    const double height_sum = (high_left - low_left) + (high_right - low_right);
    const double area = (x_right - x_left) * height_sum * 0.5;
    return area > 0.0 ? area : 0.0;
}

__device__ double interval_difference_area(
    const PolygonSweepEdge* edges,
    const PolygonSweepInterval* keep,
    int keep_count,
    const PolygonSweepInterval* cut,
    int cut_count,
    double x_left,
    double x_right,
    double x_mid) {
    double area = 0.0;
    for (int i = 0; i < keep_count; ++i) {
        double cursor_y = keep[i].low_y;
        int cursor_edge = keep[i].low_edge_index;
        for (int j = 0; j < cut_count; ++j) {
            if (cut[j].high_y <= cursor_y + 1e-9 || cut[j].low_y >= keep[i].high_y - 1e-9) {
                continue;
            }
            if (cut[j].low_y > cursor_y + 1e-9) {
                PolygonSweepInterval piece{cursor_y, fmin(cut[j].low_y, keep[i].high_y), cursor_edge, cut[j].low_edge_index};
                area += interval_area(edges, piece, x_left, x_right);
            }
            if (cut[j].high_y > cursor_y) {
                cursor_y = fmax(cursor_y, cut[j].high_y);
                cursor_edge = cut[j].high_edge_index;
            }
            if (cursor_y >= keep[i].high_y - 1e-9) {
                break;
            }
        }
        if (cursor_y < keep[i].high_y - 1e-9) {
            PolygonSweepInterval piece{cursor_y, keep[i].high_y, cursor_edge, keep[i].high_edge_index};
            area += interval_area(edges, piece, x_left, x_right);
        }
    }
    return area;
}

__device__ bool emit_interval_fragment(
    const PolygonSweepEdge* edges,
    const PolygonSweepInterval& interval,
    double x_left,
    double x_right,
    std::int64_t* out_quads,
    std::int64_t* out_fragment_twice_areas,
    std::uint16_t* out_count,
    std::int64_t* out_twice_area,
    std::uint16_t max_fragments) {
    const double low_left = edge_y_at(edges[interval.low_edge_index], x_left);
    const double low_right = edge_y_at(edges[interval.low_edge_index], x_right);
    const double high_left = edge_y_at(edges[interval.high_edge_index], x_left);
    const double high_right = edge_y_at(edges[interval.high_edge_index], x_right);
    const double height_sum = (high_left - low_left) + (high_right - low_right);
    const double area = (x_right - x_left) * height_sum * 0.5;
    if (area <= 1e-9) {
        return true;
    }
    if (*out_count >= max_fragments) {
        return false;
    }
    const std::uint16_t index = (*out_count)++;
    const auto fragment_twice_area = static_cast<std::int64_t>(llround(area * 2.0));
    std::int64_t* quad = out_quads + static_cast<std::uint64_t>(index) * 8;
    quad[0] = static_cast<std::int64_t>(llround(x_left));
    quad[1] = static_cast<std::int64_t>(llround(low_left));
    quad[2] = static_cast<std::int64_t>(llround(x_right));
    quad[3] = static_cast<std::int64_t>(llround(low_right));
    quad[4] = static_cast<std::int64_t>(llround(x_right));
    quad[5] = static_cast<std::int64_t>(llround(high_right));
    quad[6] = static_cast<std::int64_t>(llround(x_left));
    quad[7] = static_cast<std::int64_t>(llround(high_left));
    out_fragment_twice_areas[index] = fragment_twice_area;
    *out_twice_area += fragment_twice_area;
    return true;
}

__device__ bool emit_interval_difference_fragments(
    const PolygonSweepEdge* edges,
    const PolygonSweepInterval* keep,
    int keep_count,
    const PolygonSweepInterval* cut,
    int cut_count,
    double x_left,
    double x_right,
    std::int64_t* out_quads,
    std::int64_t* out_fragment_twice_areas,
    std::uint16_t* out_count,
    std::int64_t* out_twice_area,
    std::uint16_t max_fragments) {
    for (int i = 0; i < keep_count; ++i) {
        double cursor_y = keep[i].low_y;
        int cursor_edge = keep[i].low_edge_index;
        for (int j = 0; j < cut_count; ++j) {
            if (cut[j].high_y <= cursor_y + 1e-9 || cut[j].low_y >= keep[i].high_y - 1e-9) {
                continue;
            }
            if (cut[j].low_y > cursor_y + 1e-9) {
                PolygonSweepInterval piece{cursor_y, fmin(cut[j].low_y, keep[i].high_y), cursor_edge, cut[j].low_edge_index};
                if (!emit_interval_fragment(edges, piece, x_left, x_right, out_quads, out_fragment_twice_areas, out_count, out_twice_area, max_fragments)) {
                    return false;
                }
            }
            if (cut[j].high_y > cursor_y) {
                cursor_y = fmax(cursor_y, cut[j].high_y);
                cursor_edge = cut[j].high_edge_index;
            }
            if (cursor_y >= keep[i].high_y - 1e-9) {
                break;
            }
        }
        if (cursor_y < keep[i].high_y - 1e-9) {
            PolygonSweepInterval piece{cursor_y, keep[i].high_y, cursor_edge, keep[i].high_edge_index};
            if (!emit_interval_fragment(edges, piece, x_left, x_right, out_quads, out_fragment_twice_areas, out_count, out_twice_area, max_fragments)) {
                return false;
            }
        }
    }
    return true;
}

__global__ void polygon_component_area_kernel(
    const std::int64_t* vertices_xy,
    const std::uint64_t* polygon_offsets,
    const std::uint8_t* polygon_sides,
    const std::uint64_t* component_offsets,
    std::uint64_t component_count,
    std::int64_t* out_twice_areas,
    std::uint32_t* out_errors) {
    const std::uint64_t component_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (component_index >= component_count) {
        return;
    }
    PolygonSweepEdge edges[kMaxPolygonComponentEdges];
    double breaks[kMaxPolygonComponentBreaks];
    PolygonSweepCrossing crossings[kMaxPolygonComponentIntervals];
    PolygonSweepInterval old_intervals[kMaxPolygonComponentIntervals];
    PolygonSweepInterval new_intervals[kMaxPolygonComponentIntervals];
    int edge_count = 0;
    int break_count = 0;

    const std::uint64_t polygon_start = component_offsets[component_index];
    const std::uint64_t polygon_stop = component_offsets[component_index + 1];
    for (std::uint64_t polygon_index = polygon_start; polygon_index < polygon_stop; ++polygon_index) {
        const int polygon_slot = static_cast<int>(polygon_index - polygon_start);
        const std::uint64_t vertex_start = polygon_offsets[polygon_index];
        const std::uint64_t vertex_stop = polygon_offsets[polygon_index + 1];
        const std::uint64_t vertex_count = vertex_stop - vertex_start;
        if (vertex_count < 3) {
            continue;
        }
        for (std::uint64_t local = 0; local < vertex_count; ++local) {
            const std::uint64_t a = vertex_start + local;
            const std::uint64_t b = vertex_start + ((local + 1) % vertex_count);
            const double x0 = static_cast<double>(vertices_xy[a * 2 + 0]);
            const double y0 = static_cast<double>(vertices_xy[a * 2 + 1]);
            const double x1 = static_cast<double>(vertices_xy[b * 2 + 0]);
            const double y1 = static_cast<double>(vertices_xy[b * 2 + 1]);
            if (!add_breakpoint(breaks, &break_count, x0) || !add_breakpoint(breaks, &break_count, x1)) {
                out_errors[component_index] = 1;
                return;
            }
            if (fabs(x1 - x0) <= 1e-12) {
                continue;
            }
            if (edge_count >= kMaxPolygonComponentEdges) {
                out_errors[component_index] = 2;
                return;
            }
            edges[edge_count++] = PolygonSweepEdge{x0, y0, x1, y1, polygon_sides[polygon_index], polygon_slot};
        }
    }
    for (int i = 0; i < edge_count; ++i) {
        for (int j = i + 1; j < edge_count; ++j) {
            double x = 0.0;
            if (edge_intersection_x(edges[i], edges[j], &x)) {
                const double min_x = fmax(fmin(edges[i].x0, edges[i].x1), fmin(edges[j].x0, edges[j].x1));
                const double max_x = fmin(fmax(edges[i].x0, edges[i].x1), fmax(edges[j].x0, edges[j].x1));
                if (x > min_x + 1e-7 && x < max_x - 1e-7 && !add_breakpoint(breaks, &break_count, x)) {
                    out_errors[component_index] = 1;
                    return;
                }
            }
        }
    }
    sort_breakpoints(breaks, break_count);
    double old_minus_new = 0.0;
    double new_minus_old = 0.0;
    for (int slab = 0; slab + 1 < break_count; ++slab) {
        const double x_left = breaks[slab];
        const double x_right = breaks[slab + 1];
        if (x_right <= x_left + 1e-9) {
            continue;
        }
        const double x_mid = (x_left + x_right) * 0.5;
        int old_count = 0;
        int new_count = 0;
        const int component_polygon_count = static_cast<int>(polygon_stop - polygon_start);
        if (!build_union_intervals(edges, edge_count, component_polygon_count, kOldSide, x_mid, crossings, old_intervals, &old_count)
            || !build_union_intervals(edges, edge_count, component_polygon_count, kNewSide, x_mid, crossings, new_intervals, &new_count)) {
            out_errors[component_index] = 3;
            return;
        }
        old_minus_new += interval_difference_area(edges, old_intervals, old_count, new_intervals, new_count, x_left, x_right, x_mid);
        new_minus_old += interval_difference_area(edges, new_intervals, new_count, old_intervals, old_count, x_left, x_right, x_mid);
    }
    const auto old_twice = static_cast<std::int64_t>(llround(old_minus_new * 2.0));
    const auto new_twice = static_cast<std::int64_t>(llround(new_minus_old * 2.0));
    out_twice_areas[component_index * 3 + 0] = old_twice;
    out_twice_areas[component_index * 3 + 1] = new_twice;
    out_twice_areas[component_index * 3 + 2] = old_twice + new_twice;
    out_errors[component_index] = 0;
}

__global__ void polygon_component_fragment_kernel(
    const std::int64_t* vertices_xy,
    const std::uint64_t* polygon_offsets,
    const std::uint8_t* polygon_sides,
    const std::uint64_t* component_offsets,
    std::uint64_t component_count,
    std::uint16_t max_fragments,
    std::int64_t* out_quads,
    std::int64_t* out_fragment_twice_areas,
    std::uint16_t* out_counts,
    std::int64_t* out_twice_areas,
    std::uint32_t* out_errors) {
    const std::uint64_t component_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (component_index >= component_count) {
        return;
    }
    PolygonSweepEdge edges[kMaxPolygonComponentEdges];
    double breaks[kMaxPolygonComponentBreaks];
    PolygonSweepCrossing crossings[kMaxPolygonComponentIntervals];
    PolygonSweepInterval old_intervals[kMaxPolygonComponentIntervals];
    PolygonSweepInterval new_intervals[kMaxPolygonComponentIntervals];
    int edge_count = 0;
    int break_count = 0;

    const std::uint64_t polygon_start = component_offsets[component_index];
    const std::uint64_t polygon_stop = component_offsets[component_index + 1];
    for (std::uint64_t polygon_index = polygon_start; polygon_index < polygon_stop; ++polygon_index) {
        const int polygon_slot = static_cast<int>(polygon_index - polygon_start);
        const std::uint64_t vertex_start = polygon_offsets[polygon_index];
        const std::uint64_t vertex_stop = polygon_offsets[polygon_index + 1];
        const std::uint64_t vertex_count = vertex_stop - vertex_start;
        if (vertex_count < 3) {
            continue;
        }
        for (std::uint64_t local = 0; local < vertex_count; ++local) {
            const std::uint64_t a = vertex_start + local;
            const std::uint64_t b = vertex_start + ((local + 1) % vertex_count);
            const double x0 = static_cast<double>(vertices_xy[a * 2 + 0]);
            const double y0 = static_cast<double>(vertices_xy[a * 2 + 1]);
            const double x1 = static_cast<double>(vertices_xy[b * 2 + 0]);
            const double y1 = static_cast<double>(vertices_xy[b * 2 + 1]);
            if (!add_breakpoint(breaks, &break_count, x0) || !add_breakpoint(breaks, &break_count, x1)) {
                out_errors[component_index] = 1;
                return;
            }
            if (fabs(x1 - x0) <= 1e-12) {
                continue;
            }
            if (edge_count >= kMaxPolygonComponentEdges) {
                out_errors[component_index] = 2;
                return;
            }
            edges[edge_count++] = PolygonSweepEdge{x0, y0, x1, y1, polygon_sides[polygon_index], polygon_slot};
        }
    }
    for (int i = 0; i < edge_count; ++i) {
        for (int j = i + 1; j < edge_count; ++j) {
            double x = 0.0;
            if (edge_intersection_x(edges[i], edges[j], &x)) {
                const double min_x = fmax(fmin(edges[i].x0, edges[i].x1), fmin(edges[j].x0, edges[j].x1));
                const double max_x = fmin(fmax(edges[i].x0, edges[i].x1), fmax(edges[j].x0, edges[j].x1));
                if (x > min_x + 1e-7 && x < max_x - 1e-7 && !add_breakpoint(breaks, &break_count, x)) {
                    out_errors[component_index] = 1;
                    return;
                }
            }
        }
    }
    sort_breakpoints(breaks, break_count);
    std::uint16_t old_count_out = 0;
    std::uint16_t new_count_out = 0;
    std::int64_t old_twice = 0;
    std::int64_t new_twice = 0;
    double old_area_total = 0.0;
    double new_area_total = 0.0;
    std::int64_t* old_quads = out_quads + component_index * 2 * static_cast<std::uint64_t>(max_fragments) * 8;
    std::int64_t* new_quads = old_quads + static_cast<std::uint64_t>(max_fragments) * 8;
    std::int64_t* old_fragment_areas = out_fragment_twice_areas + component_index * 2 * static_cast<std::uint64_t>(max_fragments);
    std::int64_t* new_fragment_areas = old_fragment_areas + static_cast<std::uint64_t>(max_fragments);
    for (int slab = 0; slab + 1 < break_count; ++slab) {
        const double x_left = breaks[slab];
        const double x_right = breaks[slab + 1];
        if (x_right <= x_left + 1e-9) {
            continue;
        }
        const double x_mid = (x_left + x_right) * 0.5;
        int old_count = 0;
        int new_count = 0;
        const int component_polygon_count = static_cast<int>(polygon_stop - polygon_start);
        if (!build_union_intervals(edges, edge_count, component_polygon_count, kOldSide, x_mid, crossings, old_intervals, &old_count)
            || !build_union_intervals(edges, edge_count, component_polygon_count, kNewSide, x_mid, crossings, new_intervals, &new_count)) {
            out_errors[component_index] = 3;
            return;
        }
        old_area_total += interval_difference_area(edges, old_intervals, old_count, new_intervals, new_count, x_left, x_right, x_mid);
        new_area_total += interval_difference_area(edges, new_intervals, new_count, old_intervals, old_count, x_left, x_right, x_mid);
        if (!emit_interval_difference_fragments(
                edges,
                old_intervals,
                old_count,
                new_intervals,
                new_count,
                x_left,
                x_right,
                old_quads,
                old_fragment_areas,
                &old_count_out,
                &old_twice,
                max_fragments)
            || !emit_interval_difference_fragments(
                edges,
                new_intervals,
                new_count,
                old_intervals,
                old_count,
                x_left,
                x_right,
                new_quads,
                new_fragment_areas,
                &new_count_out,
                &new_twice,
                max_fragments)) {
            out_errors[component_index] = 4;
            return;
        }
    }
    old_twice = static_cast<std::int64_t>(llround(old_area_total * 2.0));
    new_twice = static_cast<std::int64_t>(llround(new_area_total * 2.0));
    out_counts[component_index * 2 + 0] = old_count_out;
    out_counts[component_index * 2 + 1] = new_count_out;
    out_twice_areas[component_index * 3 + 0] = old_twice;
    out_twice_areas[component_index * 3 + 1] = new_twice;
    out_twice_areas[component_index * 3 + 2] = old_twice + new_twice;
    out_errors[component_index] = 0;
}

__device__ bool point_in_bbox_double(double x, double y, const std::int64_t* bbox) {
    return x >= static_cast<double>(bbox[0]) && x <= static_cast<double>(bbox[2])
        && y >= static_cast<double>(bbox[1]) && y <= static_cast<double>(bbox[3]);
}

__device__ bool point_in_polygon_double(
    double x,
    double y,
    const std::int64_t* vertices_xy,
    const std::uint64_t* polygon_offsets,
    std::uint64_t polygon_index) {
    const std::uint64_t start = polygon_offsets[polygon_index];
    const std::uint64_t stop = polygon_offsets[polygon_index + 1];
    const std::uint64_t vertex_count = stop - start;
    if (vertex_count < 3) {
        return false;
    }
    bool inside = false;
    for (std::uint64_t local = 0; local < vertex_count; ++local) {
        const std::uint64_t a = start + local;
        const std::uint64_t b = start + ((local + 1) % vertex_count);
        const double x0 = static_cast<double>(vertices_xy[a * 2 + 0]);
        const double y0 = static_cast<double>(vertices_xy[a * 2 + 1]);
        const double x1 = static_cast<double>(vertices_xy[b * 2 + 0]);
        const double y1 = static_cast<double>(vertices_xy[b * 2 + 1]);
        if ((y0 > y) != (y1 > y)) {
            const double x_intersection = x0 + (y - y0) * (x1 - x0) / (y1 - y0);
            if (x_intersection > x) {
                inside = !inside;
            }
        }
    }
    return inside;
}

__device__ bool union_contains_double(
    double x,
    double y,
    const std::int64_t* vertices_xy,
    const std::uint64_t* polygon_offsets,
    const std::int64_t* bboxes,
    std::uint64_t polygon_count) {
    for (std::uint64_t polygon_index = 0; polygon_index < polygon_count; ++polygon_index) {
        const std::int64_t* bbox = bboxes + polygon_index * 4;
        if (point_in_bbox_double(x, y, bbox)
            && point_in_polygon_double(x, y, vertices_xy, polygon_offsets, polygon_index)) {
            return true;
        }
    }
    return false;
}

__device__ bool union_contains_side_double(
    double x,
    double y,
    std::uint8_t side,
    const std::int64_t* vertices_xy,
    const std::uint64_t* polygon_offsets,
    const std::int64_t* bboxes,
    const std::uint8_t* polygon_sides,
    std::uint64_t polygon_count) {
    for (std::uint64_t polygon_index = 0; polygon_index < polygon_count; ++polygon_index) {
        if (polygon_sides[polygon_index] != side) {
            continue;
        }
        const std::int64_t* bbox = bboxes + polygon_index * 4;
        if (point_in_bbox_double(x, y, bbox)
            && point_in_polygon_double(x, y, vertices_xy, polygon_offsets, polygon_index)) {
            return true;
        }
    }
    return false;
}

__device__ bool directional_target_contains_double(
    double x,
    double y,
    std::uint8_t direction,
    const std::int64_t* vertices_xy,
    const std::uint64_t* polygon_offsets,
    const std::int64_t* bboxes,
    const std::uint8_t* polygon_sides,
    std::uint64_t polygon_count) {
    const bool old_inside = union_contains_side_double(x, y, kOldSide, vertices_xy, polygon_offsets, bboxes, polygon_sides, polygon_count);
    const bool new_inside = union_contains_side_double(x, y, kNewSide, vertices_xy, polygon_offsets, bboxes, polygon_sides, polygon_count);
    if (direction == kOldSide) {
        return old_inside && !new_inside;
    }
    return new_inside && !old_inside;
}

__global__ void mixed_boundary_edge_kernel(
    const std::int64_t* vertices_xy,
    const std::uint64_t* polygon_offsets,
    const std::int64_t* bboxes,
    const std::uint8_t* polygon_sides,
    std::uint64_t polygon_count,
    const double* segment_points,
    std::uint64_t segment_count,
    std::uint8_t direction,
    double* out_edges,
    std::uint8_t* out_keep) {
    const std::uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= segment_count) {
        return;
    }
    out_keep[index] = 0;
    const double x0 = segment_points[index * 4 + 0];
    const double y0 = segment_points[index * 4 + 1];
    const double x1 = segment_points[index * 4 + 2];
    const double y1 = segment_points[index * 4 + 3];
    const double dx = x1 - x0;
    const double dy = y1 - y0;
    const double length = sqrt(dx * dx + dy * dy);
    if (length <= 1e-12) {
        return;
    }
    const double normal_x = -dy / length;
    const double normal_y = dx / length;
    const double mid_x = (x0 + x1) * 0.5;
    const double mid_y = (y0 + y1) * 0.5;
    const bool left_inside = directional_target_contains_double(
        mid_x + normal_x * 1e-5,
        mid_y + normal_y * 1e-5,
        direction,
        vertices_xy,
        polygon_offsets,
        bboxes,
        polygon_sides,
        polygon_count);
    const bool right_inside = directional_target_contains_double(
        mid_x - normal_x * 1e-5,
        mid_y - normal_y * 1e-5,
        direction,
        vertices_xy,
        polygon_offsets,
        bboxes,
        polygon_sides,
        polygon_count);
    if (left_inside && !right_inside) {
        out_edges[index * 4 + 0] = x0;
        out_edges[index * 4 + 1] = y0;
        out_edges[index * 4 + 2] = x1;
        out_edges[index * 4 + 3] = y1;
        out_keep[index] = 1;
    } else if (right_inside && !left_inside) {
        out_edges[index * 4 + 0] = x1;
        out_edges[index * 4 + 1] = y1;
        out_edges[index * 4 + 2] = x0;
        out_edges[index * 4 + 3] = y0;
        out_keep[index] = 1;
    }
}

__global__ void one_sided_boundary_edge_kernel(
    const std::int64_t* vertices_xy,
    const std::uint64_t* polygon_offsets,
    const std::int64_t* bboxes,
    const std::int64_t* signed_twice_areas,
    std::uint64_t polygon_count,
    const double* segment_points,
    const std::uint32_t* segment_polygon_indices,
    std::uint64_t segment_count,
    double* out_edges,
    std::uint8_t* out_keep) {
    const std::uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= segment_count) {
        return;
    }
    out_keep[index] = 0;
    const auto polygon_index = static_cast<std::uint64_t>(segment_polygon_indices[index]);
    if (polygon_index >= polygon_count) {
        return;
    }
    const std::int64_t signed_area = signed_twice_areas[polygon_index];
    if (signed_area == 0) {
        return;
    }
    const double x0 = segment_points[index * 4 + 0];
    const double y0 = segment_points[index * 4 + 1];
    const double x1 = segment_points[index * 4 + 2];
    const double y1 = segment_points[index * 4 + 3];
    const double dx = x1 - x0;
    const double dy = y1 - y0;
    const double length = sqrt(dx * dx + dy * dy);
    if (length <= 1e-12) {
        return;
    }
    const double normal_x = -dy / length;
    const double normal_y = dx / length;
    const double mid_x = (x0 + x1) * 0.5;
    const double mid_y = (y0 + y1) * 0.5;
    const double sample_x = signed_area > 0 ? mid_x - normal_x * 1e-5 : mid_x + normal_x * 1e-5;
    const double sample_y = signed_area > 0 ? mid_y - normal_y * 1e-5 : mid_y + normal_y * 1e-5;
    if (union_contains_double(sample_x, sample_y, vertices_xy, polygon_offsets, bboxes, polygon_count)) {
        return;
    }
    if (signed_area > 0) {
        out_edges[index * 4 + 0] = x0;
        out_edges[index * 4 + 1] = y0;
        out_edges[index * 4 + 2] = x1;
        out_edges[index * 4 + 3] = y1;
    } else {
        out_edges[index * 4 + 0] = x1;
        out_edges[index * 4 + 1] = y1;
        out_edges[index * 4 + 2] = x0;
        out_edges[index * 4 + 3] = y0;
    }
    out_keep[index] = 1;
}

__device__ bool nearly_equal_double(double left, double right) {
    return fabs(left - right) <= 1e-9;
}

__device__ bool point_equal_double(double ax, double ay, double bx, double by) {
    return nearly_equal_double(ax, bx) && nearly_equal_double(ay, by);
}

__device__ bool strict_between_zero_one(double value) {
    return value > 1e-9 && value < 1.0 - 1e-9;
}

__device__ bool segment_pairs_require_split(const double* left, const double* right) {
    const double ax0 = left[0];
    const double ay0 = left[1];
    const double ax1 = left[2];
    const double ay1 = left[3];
    const double bx0 = right[0];
    const double by0 = right[1];
    const double bx1 = right[2];
    const double by1 = right[3];
    if (fmax(fmin(ax0, ax1), fmin(bx0, bx1)) > fmin(fmax(ax0, ax1), fmax(bx0, bx1)) + 1e-9
        || fmax(fmin(ay0, ay1), fmin(by0, by1)) > fmin(fmax(ay0, ay1), fmax(by0, by1)) + 1e-9) {
        return false;
    }
    const double adx = ax1 - ax0;
    const double ady = ay1 - ay0;
    const double bdx = bx1 - bx0;
    const double bdy = by1 - by0;
    const double denom = adx * bdy - ady * bdx;
    const double cx = bx0 - ax0;
    const double cy = by0 - ay0;
    if (fabs(denom) <= 1e-12) {
        if (fabs(cx * ady - cy * adx) > 1e-6) {
            return false;
        }
        const double alen2 = adx * adx + ady * ady;
        const double blen2 = bdx * bdx + bdy * bdy;
        if (alen2 <= 1e-12 || blen2 <= 1e-12) {
            return false;
        }
        const double bt0 = ((bx0 - ax0) * adx + (by0 - ay0) * ady) / alen2;
        const double bt1 = ((bx1 - ax0) * adx + (by1 - ay0) * ady) / alen2;
        const double at0 = ((ax0 - bx0) * bdx + (ay0 - by0) * bdy) / blen2;
        const double at1 = ((ax1 - bx0) * bdx + (ay1 - by0) * bdy) / blen2;
        return strict_between_zero_one(bt0) || strict_between_zero_one(bt1)
            || strict_between_zero_one(at0) || strict_between_zero_one(at1);
    }
    const double t = (cx * bdy - cy * bdx) / denom;
    const double u = (cx * ady - cy * adx) / denom;
    if (t < -1e-9 || t > 1.0 + 1e-9 || u < -1e-9 || u > 1.0 + 1e-9) {
        return false;
    }
    const bool a_interior = strict_between_zero_one(t);
    const bool b_interior = strict_between_zero_one(u);
    if (a_interior || b_interior) {
        return true;
    }
    const double ix_a = ax0 + t * adx;
    const double iy_a = ay0 + t * ady;
    return !(point_equal_double(ix_a, iy_a, ax0, ay0)
             || point_equal_double(ix_a, iy_a, ax1, ay1))
        || !(point_equal_double(ix_a, iy_a, bx0, by0)
             || point_equal_double(ix_a, iy_a, bx1, by1));
}

__global__ void one_sided_split_required_kernel(
    const double* segment_points,
    std::uint64_t segment_count,
    std::uint32_t* split_required) {
    const std::uint64_t left_index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint64_t right_index = blockIdx.y * blockDim.y + threadIdx.y;
    if (left_index >= segment_count || right_index >= segment_count || right_index <= left_index || *split_required) {
        return;
    }
    if (segment_pairs_require_split(segment_points + left_index * 4, segment_points + right_index * 4)) {
        atomicExch(split_required, 1u);
    }
}

__device__ std::int64_t rect_area_twice(std::int64_t x0, std::int64_t y0, std::int64_t x1, std::int64_t y1) {
    const std::int64_t width = x1 - x0;
    const std::int64_t height = y1 - y0;
    if (width <= 0 || height <= 0) {
        return 0;
    }
    return width * height * 2;
}

__device__ std::int64_t rect_minus_rect_twice_area(const std::int64_t* rect, const std::int64_t* cutter, bool has_cutter) {
    if (!has_cutter) {
        return rect_area_twice(rect[0], rect[1], rect[2], rect[3]);
    }
    const std::int64_t ix0 = max(rect[0], cutter[0]);
    const std::int64_t iy0 = max(rect[1], cutter[1]);
    const std::int64_t ix1 = min(rect[2], cutter[2]);
    const std::int64_t iy1 = min(rect[3], cutter[3]);
    if (ix0 >= ix1 || iy0 >= iy1) {
        return rect_area_twice(rect[0], rect[1], rect[2], rect[3]);
    }
    return rect_area_twice(rect[0], rect[1], ix0, rect[3])
        + rect_area_twice(ix1, rect[1], rect[2], rect[3])
        + rect_area_twice(ix0, rect[1], ix1, iy0)
        + rect_area_twice(ix0, iy1, ix1, rect[3]);
}

__device__ std::uint8_t emit_rect_minus_rect(
    const std::int64_t* rect,
    const std::int64_t* cutter,
    bool has_cutter,
    std::int64_t* out_rects) {
    if (!has_cutter) {
        out_rects[0] = rect[0];
        out_rects[1] = rect[1];
        out_rects[2] = rect[2];
        out_rects[3] = rect[3];
        return rect_area_twice(rect[0], rect[1], rect[2], rect[3]) ? 1 : 0;
    }
    const std::int64_t ix0 = max(rect[0], cutter[0]);
    const std::int64_t iy0 = max(rect[1], cutter[1]);
    const std::int64_t ix1 = min(rect[2], cutter[2]);
    const std::int64_t iy1 = min(rect[3], cutter[3]);
    if (ix0 >= ix1 || iy0 >= iy1) {
        out_rects[0] = rect[0];
        out_rects[1] = rect[1];
        out_rects[2] = rect[2];
        out_rects[3] = rect[3];
        return rect_area_twice(rect[0], rect[1], rect[2], rect[3]) ? 1 : 0;
    }
    const std::int64_t pieces[4][4] = {
        {rect[0], rect[1], ix0, rect[3]},
        {ix1, rect[1], rect[2], rect[3]},
        {ix0, rect[1], ix1, iy0},
        {ix0, iy1, ix1, rect[3]},
    };
    std::uint8_t count = 0;
    for (int i = 0; i < 4; ++i) {
        if (rect_area_twice(pieces[i][0], pieces[i][1], pieces[i][2], pieces[i][3])) {
            out_rects[count * 4 + 0] = pieces[i][0];
            out_rects[count * 4 + 1] = pieces[i][1];
            out_rects[count * 4 + 2] = pieces[i][2];
            out_rects[count * 4 + 3] = pieces[i][3];
            ++count;
        }
    }
    return count;
}

__global__ void rect_exact_area_kernel(
    const std::int64_t* old_rects,
    const std::int64_t* new_rects,
    const std::uint8_t* old_present,
    const std::uint8_t* new_present,
    std::uint64_t n,
    std::int64_t* out_twice_areas) {
    const std::uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }
    const bool has_old = old_present[index] != 0;
    const bool has_new = new_present[index] != 0;
    const std::int64_t* old_rect = old_rects + index * 4;
    const std::int64_t* new_rect = new_rects + index * 4;
    const std::int64_t old_minus = has_old ? rect_minus_rect_twice_area(old_rect, new_rect, has_new) : 0;
    const std::int64_t new_minus = has_new ? rect_minus_rect_twice_area(new_rect, old_rect, has_old) : 0;
    out_twice_areas[index * 3 + 0] = old_minus;
    out_twice_areas[index * 3 + 1] = new_minus;
    out_twice_areas[index * 3 + 2] = old_minus + new_minus;
}

__global__ void rect_exact_fragments_kernel(
    const std::int64_t* old_rects,
    const std::int64_t* new_rects,
    const std::uint8_t* old_present,
    const std::uint8_t* new_present,
    std::uint64_t n,
    std::int64_t* out_rects,
    std::uint8_t* out_counts,
    std::int64_t* out_twice_areas) {
    const std::uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }
    const bool has_old = old_present[index] != 0;
    const bool has_new = new_present[index] != 0;
    const std::int64_t* old_rect = old_rects + index * 4;
    const std::int64_t* new_rect = new_rects + index * 4;
    std::int64_t* old_out = out_rects + index * 8 * 4;
    std::int64_t* new_out = old_out + 4 * 4;
    std::uint8_t old_count = 0;
    std::uint8_t new_count = 0;
    if (has_old) {
        old_count = emit_rect_minus_rect(old_rect, new_rect, has_new, old_out);
    }
    if (has_new) {
        new_count = emit_rect_minus_rect(new_rect, old_rect, has_old, new_out);
    }
    out_counts[index * 2 + 0] = old_count;
    out_counts[index * 2 + 1] = new_count;
    std::int64_t old_area = 0;
    std::int64_t new_area = 0;
    for (std::uint8_t i = 0; i < old_count; ++i) {
        old_area += rect_area_twice(old_out[i * 4 + 0], old_out[i * 4 + 1], old_out[i * 4 + 2], old_out[i * 4 + 3]);
    }
    for (std::uint8_t i = 0; i < new_count; ++i) {
        new_area += rect_area_twice(new_out[i * 4 + 0], new_out[i * 4 + 1], new_out[i * 4 + 2], new_out[i * 4 + 3]);
    }
    out_twice_areas[index * 3 + 0] = old_area;
    out_twice_areas[index * 3 + 1] = new_area;
    out_twice_areas[index * 3 + 2] = old_area + new_area;
}

__device__ void insert_coord_sorted(std::int64_t value, std::int64_t* coords, std::uint8_t* count) {
    for (std::uint8_t i = 0; i < *count; ++i) {
        if (coords[i] == value) {
            return;
        }
    }
    std::uint8_t pos = *count;
    while (pos > 0 && coords[pos - 1] > value) {
        coords[pos] = coords[pos - 1];
        --pos;
    }
    coords[pos] = value;
    ++(*count);
}

__device__ bool rect_contains_cell_device(const std::int64_t* rect, std::int64_t x0, std::int64_t y0, std::int64_t x1, std::int64_t y1) {
    return rect[0] <= x0 && x1 <= rect[2] && rect[1] <= y0 && y1 <= rect[3];
}

__device__ bool any_rect_contains_cell_device(
    const std::int64_t* rects,
    std::uint8_t count,
    std::int64_t x0,
    std::int64_t y0,
    std::int64_t x1,
    std::int64_t y1) {
    for (std::uint8_t i = 0; i < count; ++i) {
        if (rect_contains_cell_device(rects + i * 4, x0, y0, x1, y1)) {
            return true;
        }
    }
    return false;
}

__device__ std::uint16_t emit_rect_set_minus_rect_set_device(
    const std::int64_t* source,
    std::uint8_t source_count,
    const std::int64_t* cutters,
    std::uint8_t cutter_count,
    std::uint16_t max_fragments,
    std::int64_t* out_rects,
    std::int64_t* out_twice_area,
    std::uint32_t* error_flag) {
    std::int64_t xs[kMaxRectSetCoords];
    std::int64_t ys[kMaxRectSetCoords];
    std::uint8_t x_count = 0;
    std::uint8_t y_count = 0;
    for (std::uint8_t i = 0; i < source_count; ++i) {
        const std::int64_t* rect = source + i * 4;
        insert_coord_sorted(rect[0], xs, &x_count);
        insert_coord_sorted(rect[2], xs, &x_count);
        insert_coord_sorted(rect[1], ys, &y_count);
        insert_coord_sorted(rect[3], ys, &y_count);
    }
    for (std::uint8_t i = 0; i < cutter_count; ++i) {
        const std::int64_t* rect = cutters + i * 4;
        insert_coord_sorted(rect[0], xs, &x_count);
        insert_coord_sorted(rect[2], xs, &x_count);
        insert_coord_sorted(rect[1], ys, &y_count);
        insert_coord_sorted(rect[3], ys, &y_count);
    }
    std::uint16_t out_count = 0;
    std::int64_t area = 0;
    for (std::uint8_t xi = 0; xi + 1 < x_count; ++xi) {
        const std::int64_t x0 = xs[xi];
        const std::int64_t x1 = xs[xi + 1];
        if (x0 == x1) {
            continue;
        }
        for (std::uint8_t yi = 0; yi + 1 < y_count; ++yi) {
            const std::int64_t y0 = ys[yi];
            const std::int64_t y1 = ys[yi + 1];
            if (y0 == y1) {
                continue;
            }
            if (!any_rect_contains_cell_device(source, source_count, x0, y0, x1, y1)) {
                continue;
            }
            if (any_rect_contains_cell_device(cutters, cutter_count, x0, y0, x1, y1)) {
                continue;
            }
            if (out_count >= max_fragments) {
                *error_flag = 2;
                *out_twice_area = area;
                return out_count;
            }
            std::int64_t* out = out_rects + out_count * 4;
            out[0] = x0;
            out[1] = y0;
            out[2] = x1;
            out[3] = y1;
            area += rect_area_twice(x0, y0, x1, y1);
            ++out_count;
        }
    }
    *out_twice_area = area;
    return out_count;
}

__global__ void rect_set_exact_fragments_kernel(
    const std::int64_t* old_rects,
    const std::int64_t* new_rects,
    const std::uint8_t* old_counts,
    const std::uint8_t* new_counts,
    std::uint64_t task_count,
    std::uint16_t max_rects,
    std::uint16_t max_fragments,
    std::int64_t* out_rects,
    std::uint16_t* out_counts,
    std::int64_t* out_twice_areas,
    std::uint32_t* error_flags) {
    const std::uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= task_count) {
        return;
    }
    const std::uint8_t old_count = old_counts[index];
    const std::uint8_t new_count = new_counts[index];
    if (old_count > max_rects || new_count > max_rects || max_rects > kMaxRectSetRects) {
        error_flags[index] = 1;
        return;
    }
    const std::int64_t* old_task = old_rects + index * max_rects * 4;
    const std::int64_t* new_task = new_rects + index * max_rects * 4;
    std::int64_t* old_out = out_rects + index * 2 * max_fragments * 4;
    std::int64_t* new_out = old_out + max_fragments * 4;
    std::int64_t old_area = 0;
    std::int64_t new_area = 0;
    const std::uint16_t old_out_count = emit_rect_set_minus_rect_set_device(
        old_task,
        old_count,
        new_task,
        new_count,
        max_fragments,
        old_out,
        &old_area,
        error_flags + index);
    const std::uint16_t new_out_count = emit_rect_set_minus_rect_set_device(
        new_task,
        new_count,
        old_task,
        old_count,
        max_fragments,
        new_out,
        &new_area,
        error_flags + index);
    out_counts[index * 2 + 0] = old_out_count;
    out_counts[index * 2 + 1] = new_out_count;
    out_twice_areas[index * 3 + 0] = old_area;
    out_twice_areas[index * 3 + 1] = new_area;
    out_twice_areas[index * 3 + 2] = old_area + new_area;
}

struct DeviceLoopTraceStats {
    std::uint64_t fragment_count;
    std::uint64_t vertex_count;
    std::uint64_t dropped_zero_area_loop_count;
    std::uint32_t error;
};

__device__ bool loop_point_equal(const std::int64_t* points, std::uint64_t left, std::uint64_t right) {
    return points[left * 2] == points[right * 2] && points[left * 2 + 1] == points[right * 2 + 1];
}

__device__ bool loop_edge_less(const std::int64_t* edges, std::uint64_t left, std::uint64_t right) {
    for (int axis = 0; axis < 4; ++axis) {
        const auto left_value = edges[left * 4 + axis];
        const auto right_value = edges[right * 4 + axis];
        if (left_value != right_value) {
            return left_value < right_value;
        }
    }
    return false;
}

__device__ bool loop_point_less_xy(std::int64_t ax, std::int64_t ay, std::int64_t bx, std::int64_t by) {
    return ax < bx || (ax == bx && ay < by);
}

__device__ bool loop_edge_start_matches(const std::int64_t* edges, std::uint64_t edge_index, std::int64_t x, std::int64_t y) {
    return edges[edge_index * 4] == x && edges[edge_index * 4 + 1] == y;
}

__device__ bool loop_edge_stop_matches(const std::int64_t* edges, std::uint64_t edge_index, std::int64_t x, std::int64_t y) {
    return edges[edge_index * 4 + 2] == x && edges[edge_index * 4 + 3] == y;
}

__device__ bool loop_collinear(std::int64_t ax, std::int64_t ay, std::int64_t bx, std::int64_t by, std::int64_t cx, std::int64_t cy) {
    const long long left = static_cast<long long>(bx - ax) * static_cast<long long>(cy - by);
    const long long right = static_cast<long long>(by - ay) * static_cast<long long>(cx - bx);
    return left == right;
}

__device__ void loop_turn_key(
    std::int64_t px,
    std::int64_t py,
    std::int64_t cx,
    std::int64_t cy,
    std::int64_t nx,
    std::int64_t ny,
    int* turn_class,
    unsigned long long* abs_cross,
    long long* dot) {
    const long long in_x = static_cast<long long>(cx - px);
    const long long in_y = static_cast<long long>(cy - py);
    const long long out_x = static_cast<long long>(nx - cx);
    const long long out_y = static_cast<long long>(ny - cy);
    const long long cross = in_x * out_y - in_y * out_x;
    *dot = in_x * out_x + in_y * out_y;
    if (cross < 0) {
        *turn_class = 0;
        *abs_cross = static_cast<unsigned long long>(-cross);
    } else if (cross == 0 && *dot > 0) {
        *turn_class = 1;
        *abs_cross = 0;
    } else if (cross > 0) {
        *turn_class = 2;
        *abs_cross = static_cast<unsigned long long>(cross);
    } else {
        *turn_class = 3;
        *abs_cross = 0;
    }
}

__device__ bool loop_candidate_better(
    std::int64_t px,
    std::int64_t py,
    std::int64_t cx,
    std::int64_t cy,
    std::int64_t candidate_x,
    std::int64_t candidate_y,
    std::int64_t best_x,
    std::int64_t best_y) {
    int candidate_class = 0;
    int best_class = 0;
    unsigned long long candidate_abs_cross = 0;
    unsigned long long best_abs_cross = 0;
    long long candidate_dot = 0;
    long long best_dot = 0;
    loop_turn_key(px, py, cx, cy, candidate_x, candidate_y, &candidate_class, &candidate_abs_cross, &candidate_dot);
    loop_turn_key(px, py, cx, cy, best_x, best_y, &best_class, &best_abs_cross, &best_dot);
    if (candidate_class != best_class) {
        return candidate_class < best_class;
    }
    if (candidate_abs_cross != best_abs_cross) {
        return candidate_abs_cross > best_abs_cross;
    }
    if (candidate_dot != best_dot) {
        return candidate_dot > best_dot;
    }
    return loop_point_less_xy(candidate_x, candidate_y, best_x, best_y);
}

__device__ std::uint64_t loop_clean_duplicates(
    const std::int64_t* input_points,
    std::uint64_t input_count,
    std::int64_t* output_points) {
    std::uint64_t out_count = 0;
    for (std::uint64_t index = 0; index < input_count; ++index) {
        const auto x = input_points[index * 2];
        const auto y = input_points[index * 2 + 1];
        if (out_count == 0 || output_points[(out_count - 1) * 2] != x || output_points[(out_count - 1) * 2 + 1] != y) {
            output_points[out_count * 2] = x;
            output_points[out_count * 2 + 1] = y;
            ++out_count;
        }
    }
    if (out_count > 1 && output_points[0] == output_points[(out_count - 1) * 2] && output_points[1] == output_points[(out_count - 1) * 2 + 1]) {
        --out_count;
    }
    return out_count;
}

__device__ std::uint64_t loop_clean_collinear_once(
    const std::int64_t* input_points,
    std::uint64_t input_count,
    std::int64_t* output_points,
    bool* changed) {
    *changed = false;
    if (input_count < 3) {
        for (std::uint64_t index = 0; index < input_count; ++index) {
            output_points[index * 2] = input_points[index * 2];
            output_points[index * 2 + 1] = input_points[index * 2 + 1];
        }
        return input_count;
    }
    std::uint64_t out_count = 0;
    for (std::uint64_t index = 0; index < input_count; ++index) {
        const std::uint64_t previous = (index + input_count - 1) % input_count;
        const std::uint64_t following = (index + 1) % input_count;
        const auto ax = input_points[previous * 2];
        const auto ay = input_points[previous * 2 + 1];
        const auto bx = input_points[index * 2];
        const auto by = input_points[index * 2 + 1];
        const auto cx = input_points[following * 2];
        const auto cy = input_points[following * 2 + 1];
        if (loop_collinear(ax, ay, bx, by, cx, cy)) {
            *changed = true;
            continue;
        }
        output_points[out_count * 2] = bx;
        output_points[out_count * 2 + 1] = by;
        ++out_count;
    }
    return out_count;
}

__device__ long long loop_twice_area(const std::int64_t* points, std::uint64_t count) {
    long long area = 0;
    for (std::uint64_t index = 0; index < count; ++index) {
        const auto next = (index + 1) % count;
        area += static_cast<long long>(points[index * 2]) * static_cast<long long>(points[next * 2 + 1])
            - static_cast<long long>(points[next * 2]) * static_cast<long long>(points[index * 2 + 1]);
    }
    return area;
}

__global__ void boundary_loop_next_kernel(
    const std::int64_t* edges,
    std::uint64_t edge_count,
    std::uint64_t* next_indices) {
    const std::uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= edge_count) {
        return;
    }
    const auto previous_x = edges[index * 4];
    const auto previous_y = edges[index * 4 + 1];
    const auto current_x = edges[index * 4 + 2];
    const auto current_y = edges[index * 4 + 3];
    bool have_best = false;
    std::uint64_t best_edge = UINT64_MAX;
    std::int64_t best_x = 0;
    std::int64_t best_y = 0;
    for (std::uint64_t candidate = 0; candidate < edge_count; ++candidate) {
        if (!loop_edge_start_matches(edges, candidate, current_x, current_y)) {
            continue;
        }
        const auto candidate_x = edges[candidate * 4 + 2];
        const auto candidate_y = edges[candidate * 4 + 3];
        if (!have_best || loop_candidate_better(previous_x, previous_y, current_x, current_y, candidate_x, candidate_y, best_x, best_y)) {
            best_edge = candidate;
            best_x = candidate_x;
            best_y = candidate_y;
            have_best = true;
        }
    }
    next_indices[index] = have_best ? best_edge : UINT64_MAX;
}

__global__ void boundary_loop_trace_kernel(
    const std::int64_t* edges,
    std::uint64_t edge_count,
    const std::uint64_t* next_indices,
    std::uint8_t* remaining,
    std::int64_t* scratch_a,
    std::int64_t* scratch_b,
    std::int64_t* out_vertices,
    std::uint64_t* out_offsets,
    std::int64_t* out_bboxes,
    std::int64_t* out_twice_areas,
    DeviceLoopTraceStats* stats) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    stats->fragment_count = 0;
    stats->vertex_count = 0;
    stats->dropped_zero_area_loop_count = 0;
    stats->error = 0;
    out_offsets[0] = 0;

    while (true) {
        bool found_seed = false;
        std::uint64_t seed = 0;
        for (std::uint64_t index = 0; index < edge_count; ++index) {
            if (!remaining[index]) {
                continue;
            }
            if (!found_seed || loop_edge_less(edges, index, seed)) {
                seed = index;
                found_seed = true;
            }
        }
        if (!found_seed) {
            break;
        }

        remaining[seed] = 0;
        const auto start_x = edges[seed * 4];
        const auto start_y = edges[seed * 4 + 1];
        std::int64_t previous_x = start_x;
        std::int64_t previous_y = start_y;
        std::int64_t current_x = edges[seed * 4 + 2];
        std::int64_t current_y = edges[seed * 4 + 3];
        std::uint64_t current_edge = seed;
        std::uint64_t raw_count = 0;
        scratch_a[raw_count * 2] = start_x;
        scratch_a[raw_count * 2 + 1] = start_y;
        ++raw_count;
        scratch_a[raw_count * 2] = current_x;
        scratch_a[raw_count * 2 + 1] = current_y;
        ++raw_count;

        while (!(current_x == start_x && current_y == start_y)) {
            const std::uint64_t best_edge = next_indices[current_edge];
            if (best_edge == UINT64_MAX || !remaining[best_edge]) {
                break;
            }
            const auto best_x = edges[best_edge * 4 + 2];
            const auto best_y = edges[best_edge * 4 + 3];
            remaining[best_edge] = 0;
            if (!(best_x == start_x && best_y == start_y)) {
                if (raw_count >= edge_count) {
                    stats->error = 1;
                    return;
                }
                scratch_a[raw_count * 2] = best_x;
                scratch_a[raw_count * 2 + 1] = best_y;
                ++raw_count;
            }
            previous_x = current_x;
            previous_y = current_y;
            current_x = best_x;
            current_y = best_y;
            current_edge = best_edge;
        }

        std::uint64_t clean_count = loop_clean_duplicates(scratch_a, raw_count, scratch_b);
        bool changed = true;
        bool scratch_b_current = true;
        while (changed && clean_count >= 3) {
            changed = false;
            if (scratch_b_current) {
                clean_count = loop_clean_collinear_once(scratch_b, clean_count, scratch_a, &changed);
                scratch_b_current = false;
            } else {
                clean_count = loop_clean_collinear_once(scratch_a, clean_count, scratch_b, &changed);
                scratch_b_current = true;
            }
        }
        const std::int64_t* clean_points = scratch_b_current ? scratch_b : scratch_a;
        if (clean_count < 3) {
            ++stats->dropped_zero_area_loop_count;
            continue;
        }
        const auto area = loop_twice_area(clean_points, clean_count);
        if (area == 0) {
            ++stats->dropped_zero_area_loop_count;
            continue;
        }
        const std::uint64_t fragment_index = stats->fragment_count;
        if (fragment_index >= edge_count || stats->vertex_count + clean_count > edge_count) {
            stats->error = 2;
            return;
        }
        std::int64_t x0 = clean_points[0];
        std::int64_t y0 = clean_points[1];
        std::int64_t x1 = clean_points[0];
        std::int64_t y1 = clean_points[1];
        for (std::uint64_t index = 0; index < clean_count; ++index) {
            const auto x = clean_points[index * 2];
            const auto y = clean_points[index * 2 + 1];
            out_vertices[(stats->vertex_count + index) * 2] = x;
            out_vertices[(stats->vertex_count + index) * 2 + 1] = y;
            x0 = min(x0, x);
            y0 = min(y0, y);
            x1 = max(x1, x);
            y1 = max(y1, y);
        }
        out_bboxes[fragment_index * 4] = x0;
        out_bboxes[fragment_index * 4 + 1] = y0;
        out_bboxes[fragment_index * 4 + 2] = x1;
        out_bboxes[fragment_index * 4 + 3] = y1;
        out_twice_areas[fragment_index] = area;
        stats->vertex_count += clean_count;
        ++stats->fragment_count;
        out_offsets[stats->fragment_count] = stats->vertex_count;
    }
}

__global__ void axis_raw_edge_kernel(
    const std::int64_t* vertices_xy,
    const std::uint64_t* fragment_offsets,
    std::uint64_t fragment_count,
    AxisRawEdge* raw_edges,
    AxisEndpointKey* endpoints,
    std::uint32_t* error) {
    const std::uint64_t stride = static_cast<std::uint64_t>(blockDim.x) * static_cast<std::uint64_t>(gridDim.x);
    for (std::uint64_t fragment_index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         fragment_index < fragment_count;
         fragment_index += stride) {
        const std::uint64_t start = fragment_offsets[fragment_index];
        const std::uint64_t stop = fragment_offsets[fragment_index + 1];
        if (stop <= start + 2) {
            continue;
        }
        for (std::uint64_t vertex_index = start; vertex_index < stop; ++vertex_index) {
            const std::uint64_t next_index = vertex_index + 1 == stop ? start : vertex_index + 1;
            const auto x0 = vertices_xy[vertex_index * 2];
            const auto y0 = vertices_xy[vertex_index * 2 + 1];
            const auto x1 = vertices_xy[next_index * 2];
            const auto y1 = vertices_xy[next_index * 2 + 1];
            AxisRawEdge edge{};
            AxisEndpointKey first_endpoint{2, 0, 0};
            AxisEndpointKey second_endpoint{2, 0, 0};
            if (x0 == x1 && y0 == y1) {
                raw_edges[vertex_index] = edge;
                endpoints[vertex_index * 2] = first_endpoint;
                endpoints[vertex_index * 2 + 1] = second_endpoint;
                continue;
            }
            if (x0 == x1) {
                edge = AxisRawEdge{x0, y0, x1, y1, x0, y0, y1, 1, 1};
                first_endpoint = AxisEndpointKey{1, x0, y0};
                second_endpoint = AxisEndpointKey{1, x0, y1};
            } else if (y0 == y1) {
                edge = AxisRawEdge{x0, y0, x1, y1, y0, x0, x1, 0, 1};
                first_endpoint = AxisEndpointKey{0, y0, x0};
                second_endpoint = AxisEndpointKey{0, y0, x1};
            } else {
                atomicOr(error, 1u);
            }
            raw_edges[vertex_index] = edge;
            endpoints[vertex_index * 2] = first_endpoint;
            endpoints[vertex_index * 2 + 1] = second_endpoint;
        }
    }
}

__device__ bool axis_endpoint_in_edge(const AxisEndpointKey& endpoint, const AxisRawEdge& edge, std::int64_t low, std::int64_t high) {
    return endpoint.vertical == edge.vertical
        && endpoint.line == edge.line
        && low <= endpoint.projection
        && endpoint.projection <= high;
}

__global__ void axis_split_count_kernel(
    const AxisRawEdge* raw_edges,
    std::uint64_t raw_edge_count,
    const AxisEndpointKey* endpoints,
    std::uint64_t endpoint_count,
    std::uint64_t* split_counts) {
    const std::uint64_t stride = static_cast<std::uint64_t>(blockDim.x) * static_cast<std::uint64_t>(gridDim.x);
    for (std::uint64_t edge_index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         edge_index < raw_edge_count;
         edge_index += stride) {
        const auto edge = raw_edges[edge_index];
        if (!edge.valid) {
            split_counts[edge_index] = 0;
            continue;
        }
        const auto low = min(edge.p0, edge.p1);
        const auto high = max(edge.p0, edge.p1);
        std::uint64_t breakpoint_count = 0;
        for (std::uint64_t endpoint_index = 0; endpoint_index < endpoint_count; ++endpoint_index) {
            if (axis_endpoint_in_edge(endpoints[endpoint_index], edge, low, high)) {
                ++breakpoint_count;
            }
        }
        split_counts[edge_index] = breakpoint_count > 1 ? breakpoint_count - 1 : 0;
    }
}

__device__ EdgeKey axis_edge_from_line_piece(std::uint8_t vertical, std::int64_t line, std::int64_t a, std::int64_t b, bool reverse) {
    if (vertical) {
        return reverse ? EdgeKey{line, b, line, a} : EdgeKey{line, a, line, b};
    }
    return reverse ? EdgeKey{b, line, a, line} : EdgeKey{a, line, b, line};
}

__global__ void axis_split_emit_kernel(
    const AxisRawEdge* raw_edges,
    std::uint64_t raw_edge_count,
    const AxisEndpointKey* endpoints,
    std::uint64_t endpoint_count,
    const std::uint64_t* split_offsets,
    EdgeKey* split_edges) {
    const std::uint64_t stride = static_cast<std::uint64_t>(blockDim.x) * static_cast<std::uint64_t>(gridDim.x);
    for (std::uint64_t edge_index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         edge_index < raw_edge_count;
         edge_index += stride) {
        const auto edge = raw_edges[edge_index];
        if (!edge.valid) {
            continue;
        }
        const auto low = min(edge.p0, edge.p1);
        const auto high = max(edge.p0, edge.p1);
        const bool reverse = edge.p0 > edge.p1;
        bool have_previous = false;
        std::int64_t previous = 0;
        std::uint64_t out_index = split_offsets[edge_index];
        for (std::uint64_t endpoint_index = 0; endpoint_index < endpoint_count; ++endpoint_index) {
            const auto endpoint = endpoints[endpoint_index];
            if (!axis_endpoint_in_edge(endpoint, edge, low, high)) {
                continue;
            }
            if (!have_previous) {
                previous = endpoint.projection;
                have_previous = true;
                continue;
            }
            if (previous != endpoint.projection) {
                split_edges[out_index++] = axis_edge_from_line_piece(edge.vertical, edge.line, previous, endpoint.projection, reverse);
            }
            previous = endpoint.projection;
        }
    }
}

__device__ EdgeKey edge_reverse_key(const EdgeKey& edge) {
    return EdgeKey{edge.x1, edge.y1, edge.x0, edge.y0};
}

__device__ bool edge_less_device(const EdgeKey& left, const EdgeKey& right) {
    if (left.x0 != right.x0) return left.x0 < right.x0;
    if (left.y0 != right.y0) return left.y0 < right.y0;
    if (left.x1 != right.x1) return left.x1 < right.x1;
    return left.y1 < right.y1;
}

__device__ bool edge_equal_device(const EdgeKey& left, const EdgeKey& right) {
    return left.x0 == right.x0 && left.y0 == right.y0 && left.x1 == right.x1 && left.y1 == right.y1;
}

__device__ std::uint64_t edge_lower_bound_device(const EdgeKey* edges, std::uint64_t count, const EdgeKey& key) {
    std::uint64_t left = 0;
    std::uint64_t right = count;
    while (left < right) {
        const std::uint64_t middle = left + (right - left) / 2;
        if (edge_less_device(edges[middle], key)) {
            left = middle + 1;
        } else {
            right = middle;
        }
    }
    return left;
}

__global__ void edge_surplus_count_kernel(
    const EdgeKey* unique_edges,
    const std::uint64_t* counts,
    std::uint64_t unique_count,
    std::uint64_t* surplus_counts) {
    const std::uint64_t stride = static_cast<std::uint64_t>(blockDim.x) * static_cast<std::uint64_t>(gridDim.x);
    for (std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < unique_count;
         index += stride) {
        const auto reverse = edge_reverse_key(unique_edges[index]);
        const auto reverse_index = edge_lower_bound_device(unique_edges, unique_count, reverse);
        std::uint64_t reverse_count = 0;
        if (reverse_index < unique_count && edge_equal_device(unique_edges[reverse_index], reverse)) {
            reverse_count = counts[reverse_index];
        }
        surplus_counts[index] = counts[index] > reverse_count ? counts[index] - reverse_count : 0;
    }
}

__global__ void edge_surplus_emit_kernel(
    const EdgeKey* unique_edges,
    const std::uint64_t* surplus_counts,
    const std::uint64_t* surplus_offsets,
    std::uint64_t unique_count,
    std::int64_t* out_edges) {
    const std::uint64_t stride = static_cast<std::uint64_t>(blockDim.x) * static_cast<std::uint64_t>(gridDim.x);
    for (std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < unique_count;
         index += stride) {
        const auto edge = unique_edges[index];
        const auto count = surplus_counts[index];
        const auto start = surplus_offsets[index];
        for (std::uint64_t repeat = 0; repeat < count; ++repeat) {
            const std::uint64_t out_index = start + repeat;
            out_edges[out_index * 4] = edge.x0;
            out_edges[out_index * 4 + 1] = edge.y0;
            out_edges[out_index * 4 + 2] = edge.x1;
            out_edges[out_index * 4 + 3] = edge.y1;
        }
    }
}

__global__ void split_free_raw_edge_kernel(
    const std::int64_t* vertices_xy,
    const std::uint64_t* fragment_offsets,
    std::uint64_t fragment_count,
    EdgeKey* raw_edges) {
    const std::uint64_t stride = static_cast<std::uint64_t>(blockDim.x) * static_cast<std::uint64_t>(gridDim.x);
    for (std::uint64_t fragment_index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         fragment_index < fragment_count;
         fragment_index += stride) {
        const std::uint64_t start = fragment_offsets[fragment_index];
        const std::uint64_t stop = fragment_offsets[fragment_index + 1];
        if (stop <= start + 2) {
            continue;
        }
        for (std::uint64_t vertex_index = start; vertex_index < stop; ++vertex_index) {
            const std::uint64_t next_index = vertex_index + 1 == stop ? start : vertex_index + 1;
            raw_edges[vertex_index] = EdgeKey{
                vertices_xy[vertex_index * 2],
                vertices_xy[vertex_index * 2 + 1],
                vertices_xy[next_index * 2],
                vertices_xy[next_index * 2 + 1],
            };
        }
    }
}

__device__ long long edge_cross_64(long long ax, long long ay, long long bx, long long by) {
    return ax * by - ay * bx;
}

__device__ bool edge_same_undirected(const EdgeKey& left, const EdgeKey& right) {
    return (left.x0 == right.x0 && left.y0 == right.y0 && left.x1 == right.x1 && left.y1 == right.y1)
        || (left.x0 == right.x1 && left.y0 == right.y1 && left.x1 == right.x0 && left.y1 == right.y0);
}

__device__ bool edge_collinear(const EdgeKey& left, const EdgeKey& right) {
    const long long ldx = static_cast<long long>(left.x1 - left.x0);
    const long long ldy = static_cast<long long>(left.y1 - left.y0);
    const long long rdx = static_cast<long long>(right.x1 - right.x0);
    const long long rdy = static_cast<long long>(right.y1 - right.y0);
    if (edge_cross_64(ldx, ldy, rdx, rdy) != 0) {
        return false;
    }
    const long long cx = static_cast<long long>(right.x0 - left.x0);
    const long long cy = static_cast<long long>(right.y0 - left.y0);
    return edge_cross_64(cx, cy, ldx, ldy) == 0;
}

__device__ bool edge_positive_projection_overlap(const EdgeKey& left, const EdgeKey& right) {
    const auto ldx = left.x1 >= left.x0 ? left.x1 - left.x0 : left.x0 - left.x1;
    const auto ldy = left.y1 >= left.y0 ? left.y1 - left.y0 : left.y0 - left.y1;
    std::int64_t left_a = 0;
    std::int64_t left_b = 0;
    std::int64_t right_a = 0;
    std::int64_t right_b = 0;
    if (ldx >= ldy) {
        left_a = min(left.x0, left.x1);
        left_b = max(left.x0, left.x1);
        right_a = min(right.x0, right.x1);
        right_b = max(right.x0, right.x1);
    } else {
        left_a = min(left.y0, left.y1);
        left_b = max(left.y0, left.y1);
        right_a = min(right.y0, right.y1);
        right_b = max(right.y0, right.y1);
    }
    return max(left_a, right_a) < min(left_b, right_b);
}

__global__ void split_free_required_kernel(
    const EdgeKey* raw_edges,
    std::uint64_t edge_count,
    std::uint32_t* split_required) {
    const std::uint64_t total_pairs = edge_count * edge_count;
    const std::uint64_t stride = static_cast<std::uint64_t>(blockDim.x) * static_cast<std::uint64_t>(gridDim.x);
    for (std::uint64_t linear = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total_pairs;
         linear += stride) {
        if (*split_required) {
            return;
        }
        const std::uint64_t left_index = linear / edge_count;
        const std::uint64_t right_index = linear - left_index * edge_count;
        if (right_index <= left_index) {
            continue;
        }
        const auto left = raw_edges[left_index];
        const auto right = raw_edges[right_index];
        if (EdgeInvalid{}(left) || EdgeInvalid{}(right)) {
            continue;
        }
        if (!edge_collinear(left, right) || !edge_positive_projection_overlap(left, right)) {
            continue;
        }
        if (!edge_same_undirected(left, right)) {
            atomicOr(split_required, 1u);
            return;
        }
    }
}

__global__ void context_overlap_flag_kernel(
    const std::int64_t* component_bboxes,
    const std::int32_t* component_layers,
    const std::int32_t* component_datatypes,
    std::uint64_t component_count,
    const std::int64_t* polygon_bboxes,
    const std::int32_t* polygon_layers,
    const std::int32_t* polygon_datatypes,
    std::uint64_t polygon_count,
    std::uint64_t total_pairs,
    std::uint64_t* flags) {
    const std::uint64_t stride = static_cast<std::uint64_t>(blockDim.x) * static_cast<std::uint64_t>(gridDim.x);
    for (std::uint64_t linear = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total_pairs;
         linear += stride) {
        const std::uint64_t component_index = linear / polygon_count;
        const std::uint64_t polygon_index = linear - component_index * polygon_count;
        const auto cx0 = component_bboxes[component_index * 4];
        const auto cy0 = component_bboxes[component_index * 4 + 1];
        const auto cx1 = component_bboxes[component_index * 4 + 2];
        const auto cy1 = component_bboxes[component_index * 4 + 3];
        const auto px0 = polygon_bboxes[polygon_index * 4];
        const auto py0 = polygon_bboxes[polygon_index * 4 + 1];
        const auto px1 = polygon_bboxes[polygon_index * 4 + 2];
        const auto py1 = polygon_bboxes[polygon_index * 4 + 3];
        const bool same_layer = component_layers[component_index] == polygon_layers[polygon_index]
            && component_datatypes[component_index] == polygon_datatypes[polygon_index];
        const bool overlaps = px1 > cx0 && cx1 > px0 && max(py0, cy0) < min(py1, cy1);
        flags[linear] = same_layer && overlaps ? 1 : 0;
    }
}

__global__ void context_overlap_emit_kernel(
    const std::uint64_t* flags,
    const std::uint64_t* offsets,
    std::uint64_t polygon_count,
    std::uint64_t total_pairs,
    std::uint64_t* out_component_indices,
    std::uint64_t* out_polygon_indices) {
    const std::uint64_t stride = static_cast<std::uint64_t>(blockDim.x) * static_cast<std::uint64_t>(gridDim.x);
    for (std::uint64_t linear = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total_pairs;
         linear += stride) {
        if (!flags[linear]) {
            continue;
        }
        const std::uint64_t component_index = linear / polygon_count;
        const std::uint64_t polygon_index = linear - component_index * polygon_count;
        const std::uint64_t out_index = offsets[linear];
        out_component_indices[out_index] = component_index;
        out_polygon_indices[out_index] = polygon_index;
    }
}

void check_cuda(cudaError_t status, const char* action) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(status));
    }
}

std::vector<DevicePolygon> pack_polygons(
    const std::int64_t* bboxes,
    const std::int32_t* layers,
    const std::int32_t* datatypes,
    const std::uint64_t* canonical_ids,
    std::uint64_t count) {
    std::vector<DevicePolygon> out;
    out.reserve(count);
    for (std::uint64_t i = 0; i < count; ++i) {
        out.push_back(DevicePolygon{
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

std::vector<std::uint8_t> read_binary_file(const char* path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error(std::string("unable to open file: ") + path);
    }
    const auto end = input.tellg();
    if (end < 0) {
        throw std::runtime_error(std::string("unable to determine file size: ") + path);
    }
    std::vector<std::uint8_t> data(static_cast<std::size_t>(end));
    input.seekg(0, std::ios::beg);
    if (!data.empty()) {
        input.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
        if (!input) {
            throw std::runtime_error(std::string("unable to read file: ") + path);
        }
    }
    return data;
}

SideAssignments build_side_assignments(
    const std::vector<DevicePolygon>& polygons,
    std::int64_t tile_size,
    std::uint64_t max_tiles_per_polygon,
    std::uint8_t side) {
    SideAssignments result;
    if (polygons.empty()) {
        return result;
    }
    const std::uint64_t n = polygons.size();
    thrust::device_vector<DevicePolygon> d_polygons(polygons.begin(), polygons.end());
    thrust::device_vector<std::uint64_t> d_counts(n);
    thrust::device_vector<std::uint8_t> d_oversized_flags(n);
    thrust::device_vector<DeviceOversized> d_oversized(n);
    thrust::device_vector<std::uint32_t> d_errors(n, 0);

    auto* d_polygons_ptr = thrust::raw_pointer_cast(d_polygons.data());
    auto* d_counts_ptr = thrust::raw_pointer_cast(d_counts.data());
    auto* d_oversized_flags_ptr = thrust::raw_pointer_cast(d_oversized_flags.data());
    auto* d_oversized_ptr = thrust::raw_pointer_cast(d_oversized.data());
    auto* d_errors_ptr = thrust::raw_pointer_cast(d_errors.data());

    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    count_kernel<<<blocks, threads>>>(
        d_polygons_ptr,
        n,
        tile_size,
        max_tiles_per_polygon,
        d_counts_ptr,
        d_oversized_flags_ptr,
        d_oversized_ptr,
        side,
        d_errors_ptr);
    check_cuda(cudaGetLastError(), "launch count kernel");
    check_cuda(cudaDeviceSynchronize(), "run count kernel");

    const auto max_error = thrust::reduce(d_errors.begin(), d_errors.end(), std::uint32_t{0}, thrust::maximum<std::uint32_t>{});
    if (max_error) {
        throw std::overflow_error("CUDA assignment count overflow or invalid span");
    }
    std::vector<std::uint8_t> oversized_flags(n);
    std::vector<DeviceOversized> oversized(n);
    check_cuda(
        cudaMemcpy(oversized_flags.data(), d_oversized_flags_ptr, sizeof(std::uint8_t) * n, cudaMemcpyDeviceToHost),
        "copy oversized flags");
    check_cuda(cudaMemcpy(oversized.data(), d_oversized_ptr, sizeof(DeviceOversized) * n, cudaMemcpyDeviceToHost), "copy oversized");

    thrust::device_vector<std::uint64_t> d_offsets(n);
    thrust::exclusive_scan(d_counts.begin(), d_counts.end(), d_offsets.begin());

    std::uint64_t last_offset = 0;
    std::uint64_t last_count = 0;
    check_cuda(
        cudaMemcpy(&last_offset, thrust::raw_pointer_cast(d_offsets.data()) + (n - 1), sizeof(std::uint64_t), cudaMemcpyDeviceToHost),
        "copy last offset");
    check_cuda(cudaMemcpy(&last_count, d_counts_ptr + (n - 1), sizeof(std::uint64_t), cudaMemcpyDeviceToHost), "copy last count");
    if (last_count > std::numeric_limits<std::uint64_t>::max() - last_offset) {
        throw std::overflow_error("total assignment count overflow");
    }
    const std::uint64_t total = last_offset + last_count;
    result.assignment_count = total;
    for (std::size_t i = 0; i < oversized_flags.size(); ++i) {
        if (oversized_flags[i]) {
            const DeviceOversized& item = oversized[i];
            result.oversized_items.push_back(OversizedItem{
                OversizedKey{item.layer, item.datatype, item.canonical_id, item.tx0, item.ty0, item.tx1, item.ty1},
                item.side,
            });
        }
    }
    if (total == 0) {
        return result;
    }
    result.assignments.resize(total);
    emit_kernel<<<blocks, threads>>>(
        d_polygons_ptr,
        n,
        tile_size,
        thrust::raw_pointer_cast(d_offsets.data()),
        d_oversized_flags_ptr,
        thrust::raw_pointer_cast(result.assignments.data()),
        side);
    check_cuda(cudaGetLastError(), "launch emit kernel");
    check_cuda(cudaDeviceSynchronize(), "run emit kernel");
    return result;
}

}  // namespace

extern "C" {

struct GdsCudaPrefilterResult {
    std::int64_t* candidate_tiles;
    std::uint64_t candidate_count;
    std::int64_t* oversized;
    std::uint64_t oversized_count;
    std::uint64_t assignment_count;
    std::uint64_t collapsed_pair_count;
    std::uint64_t oversized_matched_count;
};

struct GdsCudaByteCompareResult {
    std::uint64_t old_size;
    std::uint64_t new_size;
    std::uint8_t same;
};

struct GdsCudaPolygonRecordDeltaResult {
    std::uint64_t record_count;
    std::int32_t* layers;
    std::int32_t* datatypes;
    std::uint8_t* directions;
    std::uint64_t* counts;
    std::int64_t* bboxes;
    std::uint64_t* vertex_counts;
    std::uint32_t* flags;
    std::uint64_t* hashes;
    std::uint64_t* representative_indices;
    std::uint64_t old_input_count;
    std::uint64_t new_input_count;
    std::uint64_t matched_record_count;
    std::uint64_t reduced_key_count;
    std::uint64_t cuda_kernel_count;
    float cuda_event_elapsed_ms;
};

struct GdsCudaBoundaryLoopResult {
    std::uint64_t fragment_count;
    std::uint64_t vertex_count;
    std::int64_t* vertices_xy;
    std::uint64_t* fragment_offsets;
    std::int64_t* bboxes;
    std::int64_t* twice_areas;
    std::uint64_t dropped_zero_area_loop_count;
    std::uint64_t cuda_kernel_count;
    float cuda_event_elapsed_ms;
};

struct GdsCudaBoundaryEdgesResult {
    std::uint64_t edge_count;
    std::int64_t* edges;
    std::uint64_t cuda_kernel_count;
    float cuda_event_elapsed_ms;
};

struct GdsCudaContextOverlapPairsResult {
    std::uint64_t pair_count;
    std::uint64_t* component_indices;
    std::uint64_t* polygon_indices;
    std::uint64_t cuda_kernel_count;
    float cuda_event_elapsed_ms;
};

const char* gds_cuda_last_error() {
    return g_last_error.c_str();
}

int gds_cuda_release_device_memory() {
    const auto sync_status = cudaDeviceSynchronize();
    if (sync_status != cudaSuccess && sync_status != cudaErrorCudartUnloading) {
        g_last_error = std::string("cudaDeviceSynchronize before release: ") + cudaGetErrorString(sync_status);
        return 1;
    }
#if CUDART_VERSION >= 11020
    cudaMemPool_t pool = nullptr;
    const auto pool_status = cudaDeviceGetDefaultMemPool(&pool, 0);
    if (pool_status == cudaSuccess && pool) {
        const auto trim_status = cudaMemPoolTrimTo(pool, 0);
        if (trim_status != cudaSuccess && trim_status != cudaErrorCudartUnloading) {
            g_last_error = std::string("cudaMemPoolTrimTo: ") + cudaGetErrorString(trim_status);
            return 1;
        }
    } else if (pool_status != cudaErrorNotSupported && pool_status != cudaErrorCudartUnloading) {
        g_last_error = std::string("cudaDeviceGetDefaultMemPool: ") + cudaGetErrorString(pool_status);
        return 1;
    }
#endif
    g_last_error.clear();
    return 0;
}

void gds_cuda_free_result(GdsCudaPrefilterResult* result) {
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

void gds_cuda_free_polygon_record_delta_result(GdsCudaPolygonRecordDeltaResult* result) {
    if (!result) {
        return;
    }
    std::free(result->layers);
    std::free(result->datatypes);
    std::free(result->directions);
    std::free(result->counts);
    std::free(result->bboxes);
    std::free(result->vertex_counts);
    std::free(result->flags);
    std::free(result->hashes);
    std::free(result->representative_indices);
    *result = GdsCudaPolygonRecordDeltaResult{};
}

void gds_cuda_free_boundary_loop_result(GdsCudaBoundaryLoopResult* result) {
    if (!result) {
        return;
    }
    std::free(result->vertices_xy);
    std::free(result->fragment_offsets);
    std::free(result->bboxes);
    std::free(result->twice_areas);
    *result = GdsCudaBoundaryLoopResult{};
}

void gds_cuda_free_boundary_edges_result(GdsCudaBoundaryEdgesResult* result) {
    if (!result) {
        return;
    }
    std::free(result->edges);
    *result = GdsCudaBoundaryEdgesResult{};
}

void gds_cuda_free_context_overlap_pairs_result(GdsCudaContextOverlapPairsResult* result) {
    if (!result) {
        return;
    }
    std::free(result->component_indices);
    std::free(result->polygon_indices);
    *result = GdsCudaContextOverlapPairsResult{};
}

int gds_cuda_compare_files(const char* old_path, const char* new_path, GdsCudaByteCompareResult* result) {
    try {
        if (!old_path || !new_path || !result) {
            throw std::invalid_argument("file compare received null pointer");
        }
        *result = GdsCudaByteCompareResult{};
        auto old_data = read_binary_file(old_path);
        auto new_data = read_binary_file(new_path);
        result->old_size = old_data.size();
        result->new_size = new_data.size();
        if (old_data.size() != new_data.size()) {
            result->same = 0;
            g_last_error.clear();
            return 0;
        }
        if (old_data.empty()) {
            result->same = 1;
            g_last_error.clear();
            return 0;
        }
        DeviceBuffer<std::uint8_t> d_old(old_data.size());
        DeviceBuffer<std::uint8_t> d_new(new_data.size());
        DeviceBuffer<std::uint32_t> d_mismatch(1);
        check_cuda(cudaMemcpy(d_old.get(), old_data.data(), old_data.size(), cudaMemcpyHostToDevice), "copy old file bytes");
        check_cuda(cudaMemcpy(d_new.get(), new_data.data(), new_data.size(), cudaMemcpyHostToDevice), "copy new file bytes");
        check_cuda(cudaMemset(d_mismatch.get(), 0, sizeof(std::uint32_t)), "clear byte compare mismatch");
        const int threads = 256;
        const int blocks = static_cast<int>((old_data.size() + threads - 1) / threads);
        byte_compare_kernel<<<blocks, threads>>>(d_old.get(), d_new.get(), old_data.size(), d_mismatch.get());
        check_cuda(cudaGetLastError(), "launch byte compare kernel");
        check_cuda(cudaDeviceSynchronize(), "run byte compare kernel");
        std::uint32_t mismatch = 0;
        check_cuda(cudaMemcpy(&mismatch, d_mismatch.get(), sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "copy byte compare mismatch");
        result->same = mismatch ? 0 : 1;
        g_last_error.clear();
        return 0;
    } catch (const std::exception& exc) {
        g_last_error = exc.what();
        return 1;
    } catch (...) {
        g_last_error = "unknown CUDA byte compare error";
        return 1;
    }
}

int gds_cuda_polygon_record_delta(
    const std::int64_t* old_vertices_xy,
    std::uint64_t old_vertex_count,
    const std::uint64_t* old_offsets,
    const std::int32_t* old_layers,
    const std::int32_t* old_datatypes,
    const std::int64_t* old_bboxes,
    const std::uint32_t* old_flags,
    std::uint64_t old_count,
    const std::int64_t* new_vertices_xy,
    std::uint64_t new_vertex_count,
    const std::uint64_t* new_offsets,
    const std::int32_t* new_layers,
    const std::int32_t* new_datatypes,
    const std::int64_t* new_bboxes,
    const std::uint32_t* new_flags,
    std::uint64_t new_count,
    GdsCudaPolygonRecordDeltaResult* result) {
    cudaEvent_t event_start = nullptr;
    cudaEvent_t event_stop = nullptr;
    try {
        if (!result) {
            throw std::invalid_argument("polygon record delta result pointer is null");
        }
        *result = GdsCudaPolygonRecordDeltaResult{};
        result->old_input_count = old_count;
        result->new_input_count = new_count;
        const std::uint64_t total_count = old_count + new_count;
        if (total_count == 0) {
            g_last_error.clear();
            return 0;
        }
        if (total_count < old_count) {
            throw std::overflow_error("polygon record count overflow");
        }
        if (old_count && (!old_offsets || !old_layers || !old_datatypes || !old_bboxes || !old_flags || (old_vertex_count && !old_vertices_xy))) {
            throw std::invalid_argument("old polygon record delta input contains null pointer");
        }
        if (new_count && (!new_offsets || !new_layers || !new_datatypes || !new_bboxes || !new_flags || (new_vertex_count && !new_vertices_xy))) {
            throw std::invalid_argument("new polygon record delta input contains null pointer");
        }

        check_cuda(cudaEventCreate(&event_start), "create polygon delta start event");
        check_cuda(cudaEventCreate(&event_stop), "create polygon delta stop event");
        check_cuda(cudaEventRecord(event_start), "record polygon delta start event");

        DeviceBuffer<std::int64_t> d_old_vertices(old_vertex_count * 2);
        DeviceBuffer<std::uint64_t> d_old_offsets(old_count ? old_count + 1 : 0);
        DeviceBuffer<std::int32_t> d_old_layers(old_count);
        DeviceBuffer<std::int32_t> d_old_datatypes(old_count);
        DeviceBuffer<std::int64_t> d_old_bboxes(old_count * 4);
        DeviceBuffer<std::uint32_t> d_old_flags(old_count);
        DeviceBuffer<std::int64_t> d_new_vertices(new_vertex_count * 2);
        DeviceBuffer<std::uint64_t> d_new_offsets(new_count ? new_count + 1 : 0);
        DeviceBuffer<std::int32_t> d_new_layers(new_count);
        DeviceBuffer<std::int32_t> d_new_datatypes(new_count);
        DeviceBuffer<std::int64_t> d_new_bboxes(new_count * 4);
        DeviceBuffer<std::uint32_t> d_new_flags(new_count);

        if (old_count) {
            if (old_vertex_count) {
                check_cuda(cudaMemcpy(d_old_vertices.get(), old_vertices_xy, sizeof(std::int64_t) * old_vertex_count * 2, cudaMemcpyHostToDevice), "copy old polygon vertices");
            }
            check_cuda(cudaMemcpy(d_old_offsets.get(), old_offsets, sizeof(std::uint64_t) * (old_count + 1), cudaMemcpyHostToDevice), "copy old polygon offsets");
            check_cuda(cudaMemcpy(d_old_layers.get(), old_layers, sizeof(std::int32_t) * old_count, cudaMemcpyHostToDevice), "copy old polygon layers");
            check_cuda(cudaMemcpy(d_old_datatypes.get(), old_datatypes, sizeof(std::int32_t) * old_count, cudaMemcpyHostToDevice), "copy old polygon datatypes");
            check_cuda(cudaMemcpy(d_old_bboxes.get(), old_bboxes, sizeof(std::int64_t) * old_count * 4, cudaMemcpyHostToDevice), "copy old polygon bboxes");
            check_cuda(cudaMemcpy(d_old_flags.get(), old_flags, sizeof(std::uint32_t) * old_count, cudaMemcpyHostToDevice), "copy old polygon flags");
        }
        if (new_count) {
            if (new_vertex_count) {
                check_cuda(cudaMemcpy(d_new_vertices.get(), new_vertices_xy, sizeof(std::int64_t) * new_vertex_count * 2, cudaMemcpyHostToDevice), "copy new polygon vertices");
            }
            check_cuda(cudaMemcpy(d_new_offsets.get(), new_offsets, sizeof(std::uint64_t) * (new_count + 1), cudaMemcpyHostToDevice), "copy new polygon offsets");
            check_cuda(cudaMemcpy(d_new_layers.get(), new_layers, sizeof(std::int32_t) * new_count, cudaMemcpyHostToDevice), "copy new polygon layers");
            check_cuda(cudaMemcpy(d_new_datatypes.get(), new_datatypes, sizeof(std::int32_t) * new_count, cudaMemcpyHostToDevice), "copy new polygon datatypes");
            check_cuda(cudaMemcpy(d_new_bboxes.get(), new_bboxes, sizeof(std::int64_t) * new_count * 4, cudaMemcpyHostToDevice), "copy new polygon bboxes");
            check_cuda(cudaMemcpy(d_new_flags.get(), new_flags, sizeof(std::uint32_t) * new_count, cudaMemcpyHostToDevice), "copy new polygon flags");
        }

        thrust::device_vector<PolygonSideRecord> records(total_count);
        const int threads = 256;
        if (old_count) {
            const int blocks = static_cast<int>((old_count + threads - 1) / threads);
            polygon_record_key_kernel<<<blocks, threads>>>(
                d_old_vertices.get(),
                d_old_offsets.get(),
                d_old_layers.get(),
                d_old_datatypes.get(),
                d_old_bboxes.get(),
                d_old_flags.get(),
                old_count,
                kOldSide,
                0,
                thrust::raw_pointer_cast(records.data()));
            check_cuda(cudaGetLastError(), "launch old polygon record key kernel");
        }
        if (new_count) {
            const int blocks = static_cast<int>((new_count + threads - 1) / threads);
            polygon_record_key_kernel<<<blocks, threads>>>(
                d_new_vertices.get(),
                d_new_offsets.get(),
                d_new_layers.get(),
                d_new_datatypes.get(),
                d_new_bboxes.get(),
                d_new_flags.get(),
                new_count,
                kNewSide,
                old_count,
                thrust::raw_pointer_cast(records.data()));
            check_cuda(cudaGetLastError(), "launch new polygon record key kernel");
        }
        check_cuda(cudaDeviceSynchronize(), "run polygon record key kernels");
        result->cuda_kernel_count = (old_count ? 1 : 0) + (new_count ? 1 : 0);

        thrust::sort(records.begin(), records.end(), PolygonSideRecordLess{});
        thrust::device_vector<PolygonRecordKey> keys(total_count);
        thrust::device_vector<PolygonCountPair> pairs(total_count);
        thrust::transform(records.begin(), records.end(), keys.begin(), PolygonRecordToKey{});
        thrust::transform(records.begin(), records.end(), pairs.begin(), PolygonRecordToCountPair{});
        thrust::device_vector<PolygonRecordKey> reduced_keys(total_count);
        thrust::device_vector<PolygonCountPair> reduced_pairs(total_count);
        auto ends = thrust::reduce_by_key(
            keys.begin(),
            keys.end(),
            pairs.begin(),
            reduced_keys.begin(),
            reduced_pairs.begin(),
            PolygonKeyEqual{},
            PolygonCountPairAdd{});
        const auto reduced_count = static_cast<std::size_t>(ends.first - reduced_keys.begin());
        reduced_keys.resize(reduced_count);
        reduced_pairs.resize(reduced_count);
        result->reduced_key_count = reduced_count;

        check_cuda(cudaEventRecord(event_stop), "record polygon delta stop event");
        check_cuda(cudaEventSynchronize(event_stop), "synchronize polygon delta stop event");
        check_cuda(cudaEventElapsedTime(&result->cuda_event_elapsed_ms, event_start, event_stop), "measure polygon delta elapsed time");

        thrust::host_vector<PolygonRecordKey> host_keys = reduced_keys;
        thrust::host_vector<PolygonCountPair> host_pairs = reduced_pairs;
        struct HostDelta {
            PolygonRecordKey key;
            std::uint8_t direction;
            std::uint64_t count;
            std::uint64_t representative_index;
        };
        std::vector<HostDelta> deltas;
        for (std::size_t i = 0; i < host_keys.size(); ++i) {
            const auto old_side_count = host_pairs[i].old_count;
            const auto new_side_count = host_pairs[i].new_count;
            result->matched_record_count += std::min(old_side_count, new_side_count);
            if (old_side_count > new_side_count) {
                deltas.push_back(HostDelta{host_keys[i], kOldSide, old_side_count - new_side_count, host_pairs[i].old_index});
            } else if (new_side_count > old_side_count) {
                deltas.push_back(HostDelta{host_keys[i], kNewSide, new_side_count - old_side_count, host_pairs[i].new_index});
            }
        }
        result->record_count = deltas.size();
        if (!deltas.empty()) {
            result->layers = static_cast<std::int32_t*>(std::malloc(sizeof(std::int32_t) * deltas.size()));
            result->datatypes = static_cast<std::int32_t*>(std::malloc(sizeof(std::int32_t) * deltas.size()));
            result->directions = static_cast<std::uint8_t*>(std::malloc(sizeof(std::uint8_t) * deltas.size()));
            result->counts = static_cast<std::uint64_t*>(std::malloc(sizeof(std::uint64_t) * deltas.size()));
            result->bboxes = static_cast<std::int64_t*>(std::malloc(sizeof(std::int64_t) * deltas.size() * 4));
            result->vertex_counts = static_cast<std::uint64_t*>(std::malloc(sizeof(std::uint64_t) * deltas.size()));
            result->flags = static_cast<std::uint32_t*>(std::malloc(sizeof(std::uint32_t) * deltas.size()));
            result->hashes = static_cast<std::uint64_t*>(std::malloc(sizeof(std::uint64_t) * deltas.size() * 2));
            result->representative_indices = static_cast<std::uint64_t*>(std::malloc(sizeof(std::uint64_t) * deltas.size()));
            if (!result->layers || !result->datatypes || !result->directions || !result->counts || !result->bboxes
                || !result->vertex_counts || !result->flags || !result->hashes || !result->representative_indices) {
                throw std::bad_alloc();
            }
            for (std::size_t i = 0; i < deltas.size(); ++i) {
                const auto& delta = deltas[i];
                result->layers[i] = delta.key.layer;
                result->datatypes[i] = delta.key.datatype;
                result->directions[i] = delta.direction;
                result->counts[i] = delta.count;
                result->bboxes[i * 4 + 0] = delta.key.x0;
                result->bboxes[i * 4 + 1] = delta.key.y0;
                result->bboxes[i * 4 + 2] = delta.key.x1;
                result->bboxes[i * 4 + 3] = delta.key.y1;
                result->vertex_counts[i] = delta.key.vertex_count;
                result->flags[i] = delta.key.flags;
                result->hashes[i * 2 + 0] = delta.key.hash0;
                result->hashes[i * 2 + 1] = delta.key.hash1;
                result->representative_indices[i] = delta.representative_index;
            }
        }
        cudaEventDestroy(event_start);
        cudaEventDestroy(event_stop);
        g_last_error.clear();
        return 0;
    } catch (const std::exception& exc) {
        if (event_start) {
            cudaEventDestroy(event_start);
        }
        if (event_stop) {
            cudaEventDestroy(event_stop);
        }
        g_last_error = exc.what();
        if (result) {
            gds_cuda_free_polygon_record_delta_result(result);
        }
        return 1;
    } catch (...) {
        if (event_start) {
            cudaEventDestroy(event_start);
        }
        if (event_stop) {
            cudaEventDestroy(event_stop);
        }
        g_last_error = "unknown CUDA polygon record delta error";
        if (result) {
            gds_cuda_free_polygon_record_delta_result(result);
        }
        return 1;
    }
}

int gds_cuda_polygon_record_delta_from_hashes(
    const std::uint64_t* old_offsets,
    const std::int32_t* old_layers,
    const std::int32_t* old_datatypes,
    const std::int64_t* old_bboxes,
    const std::uint32_t* old_flags,
    const std::uint64_t* old_hashes,
    std::uint64_t old_count,
    const std::uint64_t* new_offsets,
    const std::int32_t* new_layers,
    const std::int32_t* new_datatypes,
    const std::int64_t* new_bboxes,
    const std::uint32_t* new_flags,
    const std::uint64_t* new_hashes,
    std::uint64_t new_count,
    GdsCudaPolygonRecordDeltaResult* result) {
    cudaEvent_t event_start = nullptr;
    cudaEvent_t event_stop = nullptr;
    try {
        if (!result) {
            throw std::invalid_argument("polygon record delta result pointer is null");
        }
        *result = GdsCudaPolygonRecordDeltaResult{};
        result->old_input_count = old_count;
        result->new_input_count = new_count;
        const std::uint64_t total_count = old_count + new_count;
        if (total_count == 0) {
            g_last_error.clear();
            return 0;
        }
        if (total_count < old_count) {
            throw std::overflow_error("polygon record count overflow");
        }
        if (old_count && (!old_offsets || !old_layers || !old_datatypes || !old_bboxes || !old_flags || !old_hashes)) {
            throw std::invalid_argument("old polygon hash delta input contains null pointer");
        }
        if (new_count && (!new_offsets || !new_layers || !new_datatypes || !new_bboxes || !new_flags || !new_hashes)) {
            throw std::invalid_argument("new polygon hash delta input contains null pointer");
        }

        check_cuda(cudaEventCreate(&event_start), "create polygon hash delta start event");
        check_cuda(cudaEventCreate(&event_stop), "create polygon hash delta stop event");
        check_cuda(cudaEventRecord(event_start), "record polygon hash delta start event");

        DeviceBuffer<std::uint64_t> d_old_offsets(old_count ? old_count + 1 : 0);
        DeviceBuffer<std::int32_t> d_old_layers(old_count);
        DeviceBuffer<std::int32_t> d_old_datatypes(old_count);
        DeviceBuffer<std::int64_t> d_old_bboxes(old_count * 4);
        DeviceBuffer<std::uint32_t> d_old_flags(old_count);
        DeviceBuffer<std::uint64_t> d_old_hashes(old_count * 2);
        DeviceBuffer<std::uint64_t> d_new_offsets(new_count ? new_count + 1 : 0);
        DeviceBuffer<std::int32_t> d_new_layers(new_count);
        DeviceBuffer<std::int32_t> d_new_datatypes(new_count);
        DeviceBuffer<std::int64_t> d_new_bboxes(new_count * 4);
        DeviceBuffer<std::uint32_t> d_new_flags(new_count);
        DeviceBuffer<std::uint64_t> d_new_hashes(new_count * 2);

        if (old_count) {
            check_cuda(cudaMemcpy(d_old_offsets.get(), old_offsets, sizeof(std::uint64_t) * (old_count + 1), cudaMemcpyHostToDevice), "copy old polygon offsets");
            check_cuda(cudaMemcpy(d_old_layers.get(), old_layers, sizeof(std::int32_t) * old_count, cudaMemcpyHostToDevice), "copy old polygon layers");
            check_cuda(cudaMemcpy(d_old_datatypes.get(), old_datatypes, sizeof(std::int32_t) * old_count, cudaMemcpyHostToDevice), "copy old polygon datatypes");
            check_cuda(cudaMemcpy(d_old_bboxes.get(), old_bboxes, sizeof(std::int64_t) * old_count * 4, cudaMemcpyHostToDevice), "copy old polygon bboxes");
            check_cuda(cudaMemcpy(d_old_flags.get(), old_flags, sizeof(std::uint32_t) * old_count, cudaMemcpyHostToDevice), "copy old polygon flags");
            check_cuda(cudaMemcpy(d_old_hashes.get(), old_hashes, sizeof(std::uint64_t) * old_count * 2, cudaMemcpyHostToDevice), "copy old polygon hashes");
        }
        if (new_count) {
            check_cuda(cudaMemcpy(d_new_offsets.get(), new_offsets, sizeof(std::uint64_t) * (new_count + 1), cudaMemcpyHostToDevice), "copy new polygon offsets");
            check_cuda(cudaMemcpy(d_new_layers.get(), new_layers, sizeof(std::int32_t) * new_count, cudaMemcpyHostToDevice), "copy new polygon layers");
            check_cuda(cudaMemcpy(d_new_datatypes.get(), new_datatypes, sizeof(std::int32_t) * new_count, cudaMemcpyHostToDevice), "copy new polygon datatypes");
            check_cuda(cudaMemcpy(d_new_bboxes.get(), new_bboxes, sizeof(std::int64_t) * new_count * 4, cudaMemcpyHostToDevice), "copy new polygon bboxes");
            check_cuda(cudaMemcpy(d_new_flags.get(), new_flags, sizeof(std::uint32_t) * new_count, cudaMemcpyHostToDevice), "copy new polygon flags");
            check_cuda(cudaMemcpy(d_new_hashes.get(), new_hashes, sizeof(std::uint64_t) * new_count * 2, cudaMemcpyHostToDevice), "copy new polygon hashes");
        }

        thrust::device_vector<PolygonSideRecord> records(total_count);
        const int threads = 256;
        if (old_count) {
            const int blocks = static_cast<int>((old_count + threads - 1) / threads);
            polygon_record_key_from_hash_kernel<<<blocks, threads>>>(
                d_old_offsets.get(),
                d_old_layers.get(),
                d_old_datatypes.get(),
                d_old_bboxes.get(),
                d_old_flags.get(),
                d_old_hashes.get(),
                old_count,
                kOldSide,
                0,
                thrust::raw_pointer_cast(records.data()));
            check_cuda(cudaGetLastError(), "launch old polygon hash record key kernel");
        }
        if (new_count) {
            const int blocks = static_cast<int>((new_count + threads - 1) / threads);
            polygon_record_key_from_hash_kernel<<<blocks, threads>>>(
                d_new_offsets.get(),
                d_new_layers.get(),
                d_new_datatypes.get(),
                d_new_bboxes.get(),
                d_new_flags.get(),
                d_new_hashes.get(),
                new_count,
                kNewSide,
                old_count,
                thrust::raw_pointer_cast(records.data()));
            check_cuda(cudaGetLastError(), "launch new polygon hash record key kernel");
        }
        check_cuda(cudaDeviceSynchronize(), "run polygon hash record key kernels");
        result->cuda_kernel_count = (old_count ? 1 : 0) + (new_count ? 1 : 0);

        thrust::sort(records.begin(), records.end(), PolygonSideRecordLess{});
        thrust::device_vector<PolygonRecordKey> keys(total_count);
        thrust::device_vector<PolygonCountPair> pairs(total_count);
        thrust::transform(records.begin(), records.end(), keys.begin(), PolygonRecordToKey{});
        thrust::transform(records.begin(), records.end(), pairs.begin(), PolygonRecordToCountPair{});
        thrust::device_vector<PolygonRecordKey> reduced_keys(total_count);
        thrust::device_vector<PolygonCountPair> reduced_pairs(total_count);
        auto ends = thrust::reduce_by_key(
            keys.begin(),
            keys.end(),
            pairs.begin(),
            reduced_keys.begin(),
            reduced_pairs.begin(),
            PolygonKeyEqual{},
            PolygonCountPairAdd{});
        const auto reduced_count = static_cast<std::size_t>(ends.first - reduced_keys.begin());
        reduced_keys.resize(reduced_count);
        reduced_pairs.resize(reduced_count);
        result->reduced_key_count = reduced_count;

        check_cuda(cudaEventRecord(event_stop), "record polygon hash delta stop event");
        check_cuda(cudaEventSynchronize(event_stop), "synchronize polygon hash delta stop event");
        check_cuda(cudaEventElapsedTime(&result->cuda_event_elapsed_ms, event_start, event_stop), "measure polygon hash delta elapsed time");

        thrust::host_vector<PolygonRecordKey> host_keys = reduced_keys;
        thrust::host_vector<PolygonCountPair> host_pairs = reduced_pairs;
        struct HostDelta {
            PolygonRecordKey key;
            std::uint8_t direction;
            std::uint64_t count;
            std::uint64_t representative_index;
        };
        std::vector<HostDelta> deltas;
        for (std::size_t i = 0; i < host_keys.size(); ++i) {
            const auto old_side_count = host_pairs[i].old_count;
            const auto new_side_count = host_pairs[i].new_count;
            result->matched_record_count += std::min(old_side_count, new_side_count);
            if (old_side_count > new_side_count) {
                deltas.push_back(HostDelta{host_keys[i], kOldSide, old_side_count - new_side_count, host_pairs[i].old_index});
            } else if (new_side_count > old_side_count) {
                deltas.push_back(HostDelta{host_keys[i], kNewSide, new_side_count - old_side_count, host_pairs[i].new_index});
            }
        }
        result->record_count = deltas.size();
        if (!deltas.empty()) {
            result->layers = static_cast<std::int32_t*>(std::malloc(sizeof(std::int32_t) * deltas.size()));
            result->datatypes = static_cast<std::int32_t*>(std::malloc(sizeof(std::int32_t) * deltas.size()));
            result->directions = static_cast<std::uint8_t*>(std::malloc(sizeof(std::uint8_t) * deltas.size()));
            result->counts = static_cast<std::uint64_t*>(std::malloc(sizeof(std::uint64_t) * deltas.size()));
            result->bboxes = static_cast<std::int64_t*>(std::malloc(sizeof(std::int64_t) * deltas.size() * 4));
            result->vertex_counts = static_cast<std::uint64_t*>(std::malloc(sizeof(std::uint64_t) * deltas.size()));
            result->flags = static_cast<std::uint32_t*>(std::malloc(sizeof(std::uint32_t) * deltas.size()));
            result->hashes = static_cast<std::uint64_t*>(std::malloc(sizeof(std::uint64_t) * deltas.size() * 2));
            result->representative_indices = static_cast<std::uint64_t*>(std::malloc(sizeof(std::uint64_t) * deltas.size()));
            if (!result->layers || !result->datatypes || !result->directions || !result->counts || !result->bboxes
                || !result->vertex_counts || !result->flags || !result->hashes || !result->representative_indices) {
                throw std::bad_alloc();
            }
            for (std::size_t i = 0; i < deltas.size(); ++i) {
                const auto& delta = deltas[i];
                result->layers[i] = delta.key.layer;
                result->datatypes[i] = delta.key.datatype;
                result->directions[i] = delta.direction;
                result->counts[i] = delta.count;
                result->bboxes[i * 4 + 0] = delta.key.x0;
                result->bboxes[i * 4 + 1] = delta.key.y0;
                result->bboxes[i * 4 + 2] = delta.key.x1;
                result->bboxes[i * 4 + 3] = delta.key.y1;
                result->vertex_counts[i] = delta.key.vertex_count;
                result->flags[i] = delta.key.flags;
                result->hashes[i * 2 + 0] = delta.key.hash0;
                result->hashes[i * 2 + 1] = delta.key.hash1;
                result->representative_indices[i] = delta.representative_index;
            }
        }
        cudaEventDestroy(event_start);
        cudaEventDestroy(event_stop);
        g_last_error.clear();
        return 0;
    } catch (const std::exception& exc) {
        if (event_start) {
            cudaEventDestroy(event_start);
        }
        if (event_stop) {
            cudaEventDestroy(event_stop);
        }
        g_last_error = exc.what();
        if (result) {
            gds_cuda_free_polygon_record_delta_result(result);
        }
        return 1;
    } catch (...) {
        if (event_start) {
            cudaEventDestroy(event_start);
        }
        if (event_stop) {
            cudaEventDestroy(event_stop);
        }
        g_last_error = "unknown CUDA polygon hash record delta error";
        if (result) {
            gds_cuda_free_polygon_record_delta_result(result);
        }
        return 1;
    }
}

int gds_cuda_prefilter(
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
    GdsCudaPrefilterResult* result) {
    try {
        if (!result) {
            throw std::invalid_argument("result pointer is null");
        }
        *result = GdsCudaPrefilterResult{};
        if (tile_size <= 0 || max_tiles_per_polygon == 0) {
            throw std::invalid_argument("invalid tile config");
        }
        std::vector<OversizedItem> oversized_items;
        auto old_polygons = pack_polygons(old_bboxes, old_layers, old_datatypes, old_ids, old_count);
        auto new_polygons = pack_polygons(new_bboxes, new_layers, new_datatypes, new_ids, new_count);
        auto old_assignments = build_side_assignments(old_polygons, tile_size, max_tiles_per_polygon, kOldSide);
        auto new_assignments = build_side_assignments(new_polygons, tile_size, max_tiles_per_polygon, kNewSide);
        result->assignment_count = old_assignments.assignment_count + new_assignments.assignment_count;
        oversized_items.reserve(old_assignments.oversized_items.size() + new_assignments.oversized_items.size());
        oversized_items.insert(oversized_items.end(), old_assignments.oversized_items.begin(), old_assignments.oversized_items.end());
        oversized_items.insert(oversized_items.end(), new_assignments.oversized_items.begin(), new_assignments.oversized_items.end());

        std::vector<CandidateTile> candidate_tiles;
        if (result->assignment_count != 0) {
            thrust::device_vector<DeviceAssignment> assignments;
            assignments.reserve(result->assignment_count);
            assignments.insert(assignments.end(), old_assignments.assignments.begin(), old_assignments.assignments.end());
            assignments.insert(assignments.end(), new_assignments.assignments.begin(), new_assignments.assignments.end());

            thrust::device_vector<AssignmentKey> keys(assignments.size());
            thrust::device_vector<std::uint8_t> sides(assignments.size());
            thrust::transform(assignments.begin(), assignments.end(), keys.begin(), DeviceAssignmentToKey{});
            thrust::transform(assignments.begin(), assignments.end(), sides.begin(), DeviceAssignmentToSide{});
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
            result->collapsed_pair_count = reduced_count;
            reduced_keys.resize(reduced_count);
            reduced_sides.resize(reduced_count);

            thrust::host_vector<AssignmentKey> final_keys = reduced_keys;
            thrust::host_vector<std::uint8_t> final_sides = reduced_sides;
            for (std::size_t i = 0; i < reduced_count; ++i) {
                if (final_sides[i] != kBothSides) {
                    const AssignmentKey key = final_keys[i];
                    candidate_tiles.push_back(CandidateTile{key.layer, key.datatype, key.tx, key.ty});
                }
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
            } while (
                i < oversized_items.size()
                && !(key < oversized_items[i].key)
                && !(oversized_items[i].key < key));
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
            gds_cuda_free_result(result);
        }
        return 1;
    } catch (...) {
        g_last_error = "unknown CUDA prefilter error";
        if (result) {
            gds_cuda_free_result(result);
        }
        return 1;
    }
}

int gds_cuda_rect_exact_areas(
    const std::int64_t* old_rects,
    const std::int64_t* new_rects,
    const std::uint8_t* old_present,
    const std::uint8_t* new_present,
    std::uint64_t count,
    std::int64_t* out_twice_areas) {
    try {
        if (!out_twice_areas) {
            throw std::invalid_argument("rect exact output pointer is null");
        }
        if (count == 0) {
            g_last_error.clear();
            return 0;
        }
        if (!old_rects || !new_rects || !old_present || !new_present) {
            throw std::invalid_argument("rect exact received null input pointer");
        }
        DeviceBuffer<std::int64_t> d_old_rects(count * 4);
        DeviceBuffer<std::int64_t> d_new_rects(count * 4);
        DeviceBuffer<std::uint8_t> d_old_present(count);
        DeviceBuffer<std::uint8_t> d_new_present(count);
        DeviceBuffer<std::int64_t> d_out(count * 3);
        check_cuda(cudaMemcpy(d_old_rects.get(), old_rects, sizeof(std::int64_t) * count * 4, cudaMemcpyHostToDevice), "copy old rects");
        check_cuda(cudaMemcpy(d_new_rects.get(), new_rects, sizeof(std::int64_t) * count * 4, cudaMemcpyHostToDevice), "copy new rects");
        check_cuda(cudaMemcpy(d_old_present.get(), old_present, sizeof(std::uint8_t) * count, cudaMemcpyHostToDevice), "copy old present");
        check_cuda(cudaMemcpy(d_new_present.get(), new_present, sizeof(std::uint8_t) * count, cudaMemcpyHostToDevice), "copy new present");
        const int threads = 256;
        const int blocks = static_cast<int>((count + threads - 1) / threads);
        rect_exact_area_kernel<<<blocks, threads>>>(
            d_old_rects.get(),
            d_new_rects.get(),
            d_old_present.get(),
            d_new_present.get(),
            count,
            d_out.get());
        check_cuda(cudaGetLastError(), "launch rect exact area kernel");
        check_cuda(cudaDeviceSynchronize(), "run rect exact area kernel");
        check_cuda(cudaMemcpy(out_twice_areas, d_out.get(), sizeof(std::int64_t) * count * 3, cudaMemcpyDeviceToHost), "copy rect exact areas");
        g_last_error.clear();
        return 0;
    } catch (const std::exception& exc) {
        g_last_error = exc.what();
        return 1;
    } catch (...) {
        g_last_error = "unknown CUDA rect exact error";
        return 1;
    }
}

int gds_cuda_rect_exact_fragments(
    const std::int64_t* old_rects,
    const std::int64_t* new_rects,
    const std::uint8_t* old_present,
    const std::uint8_t* new_present,
    std::uint64_t count,
    std::int64_t* out_rects,
    std::uint8_t* out_counts,
    std::int64_t* out_twice_areas) {
    try {
        if (!out_rects || !out_counts || !out_twice_areas) {
            throw std::invalid_argument("rect exact fragment output pointer is null");
        }
        if (count == 0) {
            g_last_error.clear();
            return 0;
        }
        if (!old_rects || !new_rects || !old_present || !new_present) {
            throw std::invalid_argument("rect exact fragments received null input pointer");
        }
        DeviceBuffer<std::int64_t> d_old_rects(count * 4);
        DeviceBuffer<std::int64_t> d_new_rects(count * 4);
        DeviceBuffer<std::uint8_t> d_old_present(count);
        DeviceBuffer<std::uint8_t> d_new_present(count);
        DeviceBuffer<std::int64_t> d_out_rects(count * 8 * 4);
        DeviceBuffer<std::uint8_t> d_out_counts(count * 2);
        DeviceBuffer<std::int64_t> d_out_areas(count * 3);
        check_cuda(cudaMemcpy(d_old_rects.get(), old_rects, sizeof(std::int64_t) * count * 4, cudaMemcpyHostToDevice), "copy old rects");
        check_cuda(cudaMemcpy(d_new_rects.get(), new_rects, sizeof(std::int64_t) * count * 4, cudaMemcpyHostToDevice), "copy new rects");
        check_cuda(cudaMemcpy(d_old_present.get(), old_present, sizeof(std::uint8_t) * count, cudaMemcpyHostToDevice), "copy old present");
        check_cuda(cudaMemcpy(d_new_present.get(), new_present, sizeof(std::uint8_t) * count, cudaMemcpyHostToDevice), "copy new present");
        check_cuda(cudaMemset(d_out_rects.get(), 0, sizeof(std::int64_t) * count * 8 * 4), "clear rect fragment output");
        check_cuda(cudaMemset(d_out_counts.get(), 0, sizeof(std::uint8_t) * count * 2), "clear rect fragment counts");
        check_cuda(cudaMemset(d_out_areas.get(), 0, sizeof(std::int64_t) * count * 3), "clear rect fragment areas");
        const int threads = 256;
        const int blocks = static_cast<int>((count + threads - 1) / threads);
        rect_exact_fragments_kernel<<<blocks, threads>>>(
            d_old_rects.get(),
            d_new_rects.get(),
            d_old_present.get(),
            d_new_present.get(),
            count,
            d_out_rects.get(),
            d_out_counts.get(),
            d_out_areas.get());
        check_cuda(cudaGetLastError(), "launch rect exact fragment kernel");
        check_cuda(cudaDeviceSynchronize(), "run rect exact fragment kernel");
        check_cuda(cudaMemcpy(out_rects, d_out_rects.get(), sizeof(std::int64_t) * count * 8 * 4, cudaMemcpyDeviceToHost), "copy rect exact fragments");
        check_cuda(cudaMemcpy(out_counts, d_out_counts.get(), sizeof(std::uint8_t) * count * 2, cudaMemcpyDeviceToHost), "copy rect exact fragment counts");
        check_cuda(cudaMemcpy(out_twice_areas, d_out_areas.get(), sizeof(std::int64_t) * count * 3, cudaMemcpyDeviceToHost), "copy rect exact fragment areas");
        g_last_error.clear();
        return 0;
    } catch (const std::exception& exc) {
        g_last_error = exc.what();
        return 1;
    } catch (...) {
        g_last_error = "unknown CUDA rect exact fragment error";
        return 1;
    }
}

int gds_cuda_polygon_component_areas(
    const std::int64_t* vertices_xy,
    std::uint64_t vertex_count,
    const std::uint64_t* polygon_offsets,
    const std::uint8_t* polygon_sides,
    std::uint64_t polygon_count,
    const std::uint64_t* component_offsets,
    std::uint64_t component_count,
    std::int64_t* out_twice_areas) {
    try {
        if (!out_twice_areas) {
            throw std::invalid_argument("polygon component area output pointer is null");
        }
        if (component_count == 0) {
            g_last_error.clear();
            return 0;
        }
        if (!polygon_offsets || !polygon_sides || !component_offsets || (vertex_count && !vertices_xy)) {
            throw std::invalid_argument("polygon component area received null input pointer");
        }
        DeviceBuffer<std::int64_t> d_vertices(vertex_count * 2);
        DeviceBuffer<std::uint64_t> d_polygon_offsets(polygon_count + 1);
        DeviceBuffer<std::uint8_t> d_polygon_sides(polygon_count);
        DeviceBuffer<std::uint64_t> d_component_offsets(component_count + 1);
        DeviceBuffer<std::int64_t> d_out(component_count * 3);
        DeviceBuffer<std::uint32_t> d_errors(component_count);
        if (vertex_count) {
            check_cuda(cudaMemcpy(d_vertices.get(), vertices_xy, sizeof(std::int64_t) * vertex_count * 2, cudaMemcpyHostToDevice), "copy polygon component vertices");
        }
        check_cuda(cudaMemcpy(d_polygon_offsets.get(), polygon_offsets, sizeof(std::uint64_t) * (polygon_count + 1), cudaMemcpyHostToDevice), "copy polygon component offsets");
        check_cuda(cudaMemcpy(d_polygon_sides.get(), polygon_sides, sizeof(std::uint8_t) * polygon_count, cudaMemcpyHostToDevice), "copy polygon component sides");
        check_cuda(cudaMemcpy(d_component_offsets.get(), component_offsets, sizeof(std::uint64_t) * (component_count + 1), cudaMemcpyHostToDevice), "copy polygon component offsets");
        check_cuda(cudaMemset(d_out.get(), 0, sizeof(std::int64_t) * component_count * 3), "clear polygon component area output");
        check_cuda(cudaMemset(d_errors.get(), 0, sizeof(std::uint32_t) * component_count), "clear polygon component errors");
        const int threads = 64;
        const int blocks = static_cast<int>((component_count + threads - 1) / threads);
        polygon_component_area_kernel<<<blocks, threads>>>(
            d_vertices.get(),
            d_polygon_offsets.get(),
            d_polygon_sides.get(),
            d_component_offsets.get(),
            component_count,
            d_out.get(),
            d_errors.get());
        check_cuda(cudaGetLastError(), "launch polygon component area kernel");
        check_cuda(cudaDeviceSynchronize(), "run polygon component area kernel");
        const auto max_error = thrust::reduce(
            thrust::device_pointer_cast(d_errors.get()),
            thrust::device_pointer_cast(d_errors.get()) + component_count,
            std::uint32_t{0},
            thrust::maximum<std::uint32_t>{});
        if (max_error) {
            throw std::runtime_error("polygon component area task exceeds supported limits");
        }
        check_cuda(cudaMemcpy(out_twice_areas, d_out.get(), sizeof(std::int64_t) * component_count * 3, cudaMemcpyDeviceToHost), "copy polygon component areas");
        g_last_error.clear();
        return 0;
    } catch (const std::exception& exc) {
        g_last_error = exc.what();
        return 1;
    } catch (...) {
        g_last_error = "unknown CUDA polygon component area error";
        return 1;
    }
}

int gds_cuda_polygon_component_fragments(
    const std::int64_t* vertices_xy,
    std::uint64_t vertex_count,
    const std::uint64_t* polygon_offsets,
    const std::uint8_t* polygon_sides,
    std::uint64_t polygon_count,
    const std::uint64_t* component_offsets,
    std::uint64_t component_count,
    std::uint16_t max_fragments,
    std::int64_t* out_quads,
    std::int64_t* out_fragment_twice_areas,
    std::uint16_t* out_counts,
    std::int64_t* out_twice_areas) {
    try {
        if (!out_quads || !out_fragment_twice_areas || !out_counts || !out_twice_areas) {
            throw std::invalid_argument("polygon component fragment output pointer is null");
        }
        if (component_count == 0) {
            g_last_error.clear();
            return 0;
        }
        if (!polygon_offsets || !polygon_sides || !component_offsets || (vertex_count && !vertices_xy)) {
            throw std::invalid_argument("polygon component fragments received null input pointer");
        }
        if (max_fragments == 0) {
            throw std::invalid_argument("polygon component max_fragments must be positive");
        }
        DeviceBuffer<std::int64_t> d_vertices(vertex_count * 2);
        DeviceBuffer<std::uint64_t> d_polygon_offsets(polygon_count + 1);
        DeviceBuffer<std::uint8_t> d_polygon_sides(polygon_count);
        DeviceBuffer<std::uint64_t> d_component_offsets(component_count + 1);
        DeviceBuffer<std::int64_t> d_out_quads(component_count * 2 * max_fragments * 8);
        DeviceBuffer<std::int64_t> d_out_fragment_areas(component_count * 2 * max_fragments);
        DeviceBuffer<std::uint16_t> d_out_counts(component_count * 2);
        DeviceBuffer<std::int64_t> d_out_areas(component_count * 3);
        DeviceBuffer<std::uint32_t> d_errors(component_count);
        if (vertex_count) {
            check_cuda(cudaMemcpy(d_vertices.get(), vertices_xy, sizeof(std::int64_t) * vertex_count * 2, cudaMemcpyHostToDevice), "copy polygon component fragment vertices");
        }
        check_cuda(cudaMemcpy(d_polygon_offsets.get(), polygon_offsets, sizeof(std::uint64_t) * (polygon_count + 1), cudaMemcpyHostToDevice), "copy polygon component fragment offsets");
        check_cuda(cudaMemcpy(d_polygon_sides.get(), polygon_sides, sizeof(std::uint8_t) * polygon_count, cudaMemcpyHostToDevice), "copy polygon component fragment sides");
        check_cuda(cudaMemcpy(d_component_offsets.get(), component_offsets, sizeof(std::uint64_t) * (component_count + 1), cudaMemcpyHostToDevice), "copy polygon component fragment component offsets");
        check_cuda(cudaMemset(d_out_quads.get(), 0, sizeof(std::int64_t) * component_count * 2 * max_fragments * 8), "clear polygon component fragments");
        check_cuda(cudaMemset(d_out_fragment_areas.get(), 0, sizeof(std::int64_t) * component_count * 2 * max_fragments), "clear polygon component fragment areas");
        check_cuda(cudaMemset(d_out_counts.get(), 0, sizeof(std::uint16_t) * component_count * 2), "clear polygon component fragment counts");
        check_cuda(cudaMemset(d_out_areas.get(), 0, sizeof(std::int64_t) * component_count * 3), "clear polygon component fragment areas");
        check_cuda(cudaMemset(d_errors.get(), 0, sizeof(std::uint32_t) * component_count), "clear polygon component fragment errors");
        const int threads = 64;
        const int blocks = static_cast<int>((component_count + threads - 1) / threads);
        polygon_component_fragment_kernel<<<blocks, threads>>>(
            d_vertices.get(),
            d_polygon_offsets.get(),
            d_polygon_sides.get(),
            d_component_offsets.get(),
            component_count,
            max_fragments,
            d_out_quads.get(),
            d_out_fragment_areas.get(),
            d_out_counts.get(),
            d_out_areas.get(),
            d_errors.get());
        check_cuda(cudaGetLastError(), "launch polygon component fragment kernel");
        check_cuda(cudaDeviceSynchronize(), "run polygon component fragment kernel");
        const auto max_error = thrust::reduce(
            thrust::device_pointer_cast(d_errors.get()),
            thrust::device_pointer_cast(d_errors.get()) + component_count,
            std::uint32_t{0},
            thrust::maximum<std::uint32_t>{});
        if (max_error) {
            throw std::runtime_error(max_error == 4 ? "polygon component fragment task exceeds fragment limit" : "polygon component fragment task exceeds supported limits");
        }
        check_cuda(cudaMemcpy(out_quads, d_out_quads.get(), sizeof(std::int64_t) * component_count * 2 * max_fragments * 8, cudaMemcpyDeviceToHost), "copy polygon component fragments");
        check_cuda(cudaMemcpy(out_fragment_twice_areas, d_out_fragment_areas.get(), sizeof(std::int64_t) * component_count * 2 * max_fragments, cudaMemcpyDeviceToHost), "copy polygon component fragment areas");
        check_cuda(cudaMemcpy(out_counts, d_out_counts.get(), sizeof(std::uint16_t) * component_count * 2, cudaMemcpyDeviceToHost), "copy polygon component fragment counts");
        check_cuda(cudaMemcpy(out_twice_areas, d_out_areas.get(), sizeof(std::int64_t) * component_count * 3, cudaMemcpyDeviceToHost), "copy polygon component fragment areas");
        g_last_error.clear();
        return 0;
    } catch (const std::exception& exc) {
        g_last_error = exc.what();
        return 1;
    } catch (...) {
        g_last_error = "unknown CUDA polygon component fragment error";
        return 1;
    }
}

int gds_cuda_one_sided_boundary_edges(
    const std::int64_t* vertices_xy,
    std::uint64_t vertex_count,
    const std::uint64_t* polygon_offsets,
    const std::int64_t* bboxes,
    const std::int64_t* signed_twice_areas,
    std::uint64_t polygon_count,
    const double* segment_points,
    const std::uint32_t* segment_polygon_indices,
    std::uint64_t segment_count,
    double* out_edges,
    std::uint8_t* out_keep) {
    try {
        if (!out_edges || !out_keep) {
            throw std::invalid_argument("one-sided boundary edge output pointer is null");
        }
        if (segment_count == 0) {
            g_last_error.clear();
            return 0;
        }
        if (!polygon_offsets || !bboxes || !signed_twice_areas || !segment_points || !segment_polygon_indices || (vertex_count && !vertices_xy)) {
            throw std::invalid_argument("one-sided boundary edge received null input pointer");
        }
        DeviceBuffer<std::int64_t> d_vertices(vertex_count * 2);
        DeviceBuffer<std::uint64_t> d_polygon_offsets(polygon_count + 1);
        DeviceBuffer<std::int64_t> d_bboxes(polygon_count * 4);
        DeviceBuffer<std::int64_t> d_signed_twice_areas(polygon_count);
        DeviceBuffer<double> d_segment_points(segment_count * 4);
        DeviceBuffer<std::uint32_t> d_segment_polygon_indices(segment_count);
        DeviceBuffer<double> d_out_edges(segment_count * 4);
        DeviceBuffer<std::uint8_t> d_out_keep(segment_count);
        if (vertex_count) {
            check_cuda(cudaMemcpy(d_vertices.get(), vertices_xy, sizeof(std::int64_t) * vertex_count * 2, cudaMemcpyHostToDevice), "copy one-sided boundary vertices");
        }
        check_cuda(cudaMemcpy(d_polygon_offsets.get(), polygon_offsets, sizeof(std::uint64_t) * (polygon_count + 1), cudaMemcpyHostToDevice), "copy one-sided boundary offsets");
        check_cuda(cudaMemcpy(d_bboxes.get(), bboxes, sizeof(std::int64_t) * polygon_count * 4, cudaMemcpyHostToDevice), "copy one-sided boundary bboxes");
        check_cuda(cudaMemcpy(d_signed_twice_areas.get(), signed_twice_areas, sizeof(std::int64_t) * polygon_count, cudaMemcpyHostToDevice), "copy one-sided boundary signed areas");
        check_cuda(cudaMemcpy(d_segment_points.get(), segment_points, sizeof(double) * segment_count * 4, cudaMemcpyHostToDevice), "copy one-sided boundary segment points");
        check_cuda(cudaMemcpy(d_segment_polygon_indices.get(), segment_polygon_indices, sizeof(std::uint32_t) * segment_count, cudaMemcpyHostToDevice), "copy one-sided boundary segment polygon indices");
        check_cuda(cudaMemset(d_out_edges.get(), 0, sizeof(double) * segment_count * 4), "clear one-sided boundary edges");
        check_cuda(cudaMemset(d_out_keep.get(), 0, sizeof(std::uint8_t) * segment_count), "clear one-sided boundary keep flags");
        const int threads = 256;
        const int blocks = static_cast<int>((segment_count + threads - 1) / threads);
        one_sided_boundary_edge_kernel<<<blocks, threads>>>(
            d_vertices.get(),
            d_polygon_offsets.get(),
            d_bboxes.get(),
            d_signed_twice_areas.get(),
            polygon_count,
            d_segment_points.get(),
            d_segment_polygon_indices.get(),
            segment_count,
            d_out_edges.get(),
            d_out_keep.get());
        check_cuda(cudaGetLastError(), "launch one-sided boundary edge kernel");
        check_cuda(cudaDeviceSynchronize(), "run one-sided boundary edge kernel");
        check_cuda(cudaMemcpy(out_edges, d_out_edges.get(), sizeof(double) * segment_count * 4, cudaMemcpyDeviceToHost), "copy one-sided boundary edges");
        check_cuda(cudaMemcpy(out_keep, d_out_keep.get(), sizeof(std::uint8_t) * segment_count, cudaMemcpyDeviceToHost), "copy one-sided boundary keep flags");
        g_last_error.clear();
        return 0;
    } catch (const std::exception& exc) {
        g_last_error = exc.what();
        return 1;
    } catch (...) {
        g_last_error = "unknown CUDA one-sided boundary edge error";
        return 1;
    }
}

int gds_cuda_mixed_boundary_edges(
    const std::int64_t* vertices_xy,
    std::uint64_t vertex_count,
    const std::uint64_t* polygon_offsets,
    const std::int64_t* bboxes,
    const std::uint8_t* polygon_sides,
    std::uint64_t polygon_count,
    const double* segment_points,
    std::uint64_t segment_count,
    std::uint8_t direction,
    double* out_edges,
    std::uint8_t* out_keep) {
    try {
        if (!out_edges || !out_keep) {
            throw std::invalid_argument("mixed boundary edge output pointer is null");
        }
        if (segment_count == 0) {
            g_last_error.clear();
            return 0;
        }
        if (!polygon_offsets || !bboxes || !polygon_sides || !segment_points || (vertex_count && !vertices_xy)) {
            throw std::invalid_argument("mixed boundary edge received null input pointer");
        }
        if (direction != kOldSide && direction != kNewSide) {
            throw std::invalid_argument("mixed boundary edge direction must be old or new side");
        }
        DeviceBuffer<std::int64_t> d_vertices(vertex_count * 2);
        DeviceBuffer<std::uint64_t> d_polygon_offsets(polygon_count + 1);
        DeviceBuffer<std::int64_t> d_bboxes(polygon_count * 4);
        DeviceBuffer<std::uint8_t> d_polygon_sides(polygon_count);
        DeviceBuffer<double> d_segment_points(segment_count * 4);
        DeviceBuffer<double> d_out_edges(segment_count * 4);
        DeviceBuffer<std::uint8_t> d_out_keep(segment_count);
        if (vertex_count) {
            check_cuda(cudaMemcpy(d_vertices.get(), vertices_xy, sizeof(std::int64_t) * vertex_count * 2, cudaMemcpyHostToDevice), "copy mixed boundary vertices");
        }
        check_cuda(cudaMemcpy(d_polygon_offsets.get(), polygon_offsets, sizeof(std::uint64_t) * (polygon_count + 1), cudaMemcpyHostToDevice), "copy mixed boundary offsets");
        check_cuda(cudaMemcpy(d_bboxes.get(), bboxes, sizeof(std::int64_t) * polygon_count * 4, cudaMemcpyHostToDevice), "copy mixed boundary bboxes");
        check_cuda(cudaMemcpy(d_polygon_sides.get(), polygon_sides, sizeof(std::uint8_t) * polygon_count, cudaMemcpyHostToDevice), "copy mixed boundary sides");
        check_cuda(cudaMemcpy(d_segment_points.get(), segment_points, sizeof(double) * segment_count * 4, cudaMemcpyHostToDevice), "copy mixed boundary segment points");
        check_cuda(cudaMemset(d_out_edges.get(), 0, sizeof(double) * segment_count * 4), "clear mixed boundary edges");
        check_cuda(cudaMemset(d_out_keep.get(), 0, sizeof(std::uint8_t) * segment_count), "clear mixed boundary keep flags");
        const int threads = 256;
        const int blocks = static_cast<int>((segment_count + threads - 1) / threads);
        mixed_boundary_edge_kernel<<<blocks, threads>>>(
            d_vertices.get(),
            d_polygon_offsets.get(),
            d_bboxes.get(),
            d_polygon_sides.get(),
            polygon_count,
            d_segment_points.get(),
            segment_count,
            direction,
            d_out_edges.get(),
            d_out_keep.get());
        check_cuda(cudaGetLastError(), "launch mixed boundary edge kernel");
        check_cuda(cudaDeviceSynchronize(), "run mixed boundary edge kernel");
        check_cuda(cudaMemcpy(out_edges, d_out_edges.get(), sizeof(double) * segment_count * 4, cudaMemcpyDeviceToHost), "copy mixed boundary edges");
        check_cuda(cudaMemcpy(out_keep, d_out_keep.get(), sizeof(std::uint8_t) * segment_count, cudaMemcpyDeviceToHost), "copy mixed boundary keep flags");
        g_last_error.clear();
        return 0;
    } catch (const std::exception& exc) {
        g_last_error = exc.what();
        return 1;
    } catch (...) {
        g_last_error = "unknown CUDA mixed boundary edge error";
        return 1;
    }
}

int gds_cuda_one_sided_boundary_edges_unsplit_if_safe(
    const std::int64_t* vertices_xy,
    std::uint64_t vertex_count,
    const std::uint64_t* polygon_offsets,
    const std::int64_t* bboxes,
    const std::int64_t* signed_twice_areas,
    std::uint64_t polygon_count,
    const double* segment_points,
    const std::uint32_t* segment_polygon_indices,
    std::uint64_t segment_count,
    double* out_edges,
    std::uint8_t* out_keep,
    std::uint8_t* out_safe) {
    try {
        if (!out_edges || !out_keep || !out_safe) {
            throw std::invalid_argument("one-sided unsplit boundary output pointer is null");
        }
        *out_safe = 0;
        if (segment_count == 0) {
            *out_safe = 1;
            g_last_error.clear();
            return 0;
        }
        if (!polygon_offsets || !bboxes || !signed_twice_areas || !segment_points || !segment_polygon_indices || (vertex_count && !vertices_xy)) {
            throw std::invalid_argument("one-sided unsplit boundary received null input pointer");
        }
        DeviceBuffer<std::int64_t> d_vertices(vertex_count * 2);
        DeviceBuffer<std::uint64_t> d_polygon_offsets(polygon_count + 1);
        DeviceBuffer<std::int64_t> d_bboxes(polygon_count * 4);
        DeviceBuffer<std::int64_t> d_signed_twice_areas(polygon_count);
        DeviceBuffer<double> d_segment_points(segment_count * 4);
        DeviceBuffer<std::uint32_t> d_segment_polygon_indices(segment_count);
        DeviceBuffer<double> d_out_edges(segment_count * 4);
        DeviceBuffer<std::uint8_t> d_out_keep(segment_count);
        DeviceBuffer<std::uint32_t> d_split_required(1);
        if (vertex_count) {
            check_cuda(cudaMemcpy(d_vertices.get(), vertices_xy, sizeof(std::int64_t) * vertex_count * 2, cudaMemcpyHostToDevice), "copy one-sided unsplit boundary vertices");
        }
        check_cuda(cudaMemcpy(d_polygon_offsets.get(), polygon_offsets, sizeof(std::uint64_t) * (polygon_count + 1), cudaMemcpyHostToDevice), "copy one-sided unsplit boundary offsets");
        check_cuda(cudaMemcpy(d_bboxes.get(), bboxes, sizeof(std::int64_t) * polygon_count * 4, cudaMemcpyHostToDevice), "copy one-sided unsplit boundary bboxes");
        check_cuda(cudaMemcpy(d_signed_twice_areas.get(), signed_twice_areas, sizeof(std::int64_t) * polygon_count, cudaMemcpyHostToDevice), "copy one-sided unsplit boundary signed areas");
        check_cuda(cudaMemcpy(d_segment_points.get(), segment_points, sizeof(double) * segment_count * 4, cudaMemcpyHostToDevice), "copy one-sided unsplit boundary segment points");
        check_cuda(cudaMemcpy(d_segment_polygon_indices.get(), segment_polygon_indices, sizeof(std::uint32_t) * segment_count, cudaMemcpyHostToDevice), "copy one-sided unsplit boundary segment polygon indices");
        check_cuda(cudaMemset(d_out_edges.get(), 0, sizeof(double) * segment_count * 4), "clear one-sided unsplit boundary edges");
        check_cuda(cudaMemset(d_out_keep.get(), 0, sizeof(std::uint8_t) * segment_count), "clear one-sided unsplit boundary keep flags");
        check_cuda(cudaMemset(d_split_required.get(), 0, sizeof(std::uint32_t)), "clear one-sided unsplit split flag");
        dim3 threads2(16, 16);
        dim3 blocks2(
            static_cast<unsigned int>((segment_count + threads2.x - 1) / threads2.x),
            static_cast<unsigned int>((segment_count + threads2.y - 1) / threads2.y));
        one_sided_split_required_kernel<<<blocks2, threads2>>>(
            d_segment_points.get(),
            segment_count,
            d_split_required.get());
        check_cuda(cudaGetLastError(), "launch one-sided split-required kernel");
        check_cuda(cudaDeviceSynchronize(), "run one-sided split-required kernel");
        std::uint32_t split_required = 0;
        check_cuda(cudaMemcpy(&split_required, d_split_required.get(), sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "copy one-sided split-required flag");
        if (split_required) {
            check_cuda(cudaMemset(d_out_keep.get(), 0, sizeof(std::uint8_t) * segment_count), "clear one-sided fallback keep flags");
            check_cuda(cudaMemcpy(out_keep, d_out_keep.get(), sizeof(std::uint8_t) * segment_count, cudaMemcpyDeviceToHost), "copy one-sided fallback keep flags");
            *out_safe = 0;
            g_last_error.clear();
            return 0;
        }
        const int threads = 256;
        const int blocks = static_cast<int>((segment_count + threads - 1) / threads);
        one_sided_boundary_edge_kernel<<<blocks, threads>>>(
            d_vertices.get(),
            d_polygon_offsets.get(),
            d_bboxes.get(),
            d_signed_twice_areas.get(),
            polygon_count,
            d_segment_points.get(),
            d_segment_polygon_indices.get(),
            segment_count,
            d_out_edges.get(),
            d_out_keep.get());
        check_cuda(cudaGetLastError(), "launch one-sided unsplit boundary edge kernel");
        check_cuda(cudaDeviceSynchronize(), "run one-sided unsplit boundary edge kernel");
        check_cuda(cudaMemcpy(out_edges, d_out_edges.get(), sizeof(double) * segment_count * 4, cudaMemcpyDeviceToHost), "copy one-sided unsplit boundary edges");
        check_cuda(cudaMemcpy(out_keep, d_out_keep.get(), sizeof(std::uint8_t) * segment_count, cudaMemcpyDeviceToHost), "copy one-sided unsplit boundary keep flags");
        *out_safe = 1;
        g_last_error.clear();
        return 0;
    } catch (const std::exception& exc) {
        g_last_error = exc.what();
        return 1;
    } catch (...) {
        g_last_error = "unknown CUDA one-sided unsplit boundary edge error";
        return 1;
    }
}

int gds_cuda_boundary_loops_from_edges(
    const std::int64_t* edges,
    std::uint64_t edge_count,
    GdsCudaBoundaryLoopResult* result) {
    cudaEvent_t event_start = nullptr;
    cudaEvent_t event_stop = nullptr;
    try {
        if (!result) {
            throw std::invalid_argument("CUDA boundary loop output pointer is null");
        }
        *result = GdsCudaBoundaryLoopResult{};
        if (edge_count == 0) {
            g_last_error.clear();
            return 0;
        }
        if (!edges) {
            throw std::invalid_argument("CUDA boundary loop edge pointer is null");
        }
        DeviceBuffer<std::int64_t> d_edges(edge_count * 4);
        DeviceBuffer<std::uint64_t> d_next_indices(edge_count);
        DeviceBuffer<std::uint8_t> d_remaining(edge_count);
        DeviceBuffer<std::int64_t> d_scratch_a(edge_count * 2);
        DeviceBuffer<std::int64_t> d_scratch_b(edge_count * 2);
        DeviceBuffer<std::int64_t> d_vertices(edge_count * 2);
        DeviceBuffer<std::uint64_t> d_offsets(edge_count + 1);
        DeviceBuffer<std::int64_t> d_bboxes(edge_count * 4);
        DeviceBuffer<std::int64_t> d_areas(edge_count);
        DeviceBuffer<DeviceLoopTraceStats> d_stats(1);
        check_cuda(cudaMemcpy(d_edges.get(), edges, sizeof(std::int64_t) * edge_count * 4, cudaMemcpyHostToDevice), "copy boundary loop edges");
        check_cuda(cudaMemset(d_next_indices.get(), 0xff, sizeof(std::uint64_t) * edge_count), "clear boundary loop next indices");
        check_cuda(cudaMemset(d_remaining.get(), 1, sizeof(std::uint8_t) * edge_count), "initialize boundary loop remaining flags");
        check_cuda(cudaMemset(d_vertices.get(), 0, sizeof(std::int64_t) * edge_count * 2), "clear boundary loop vertices");
        check_cuda(cudaMemset(d_offsets.get(), 0, sizeof(std::uint64_t) * (edge_count + 1)), "clear boundary loop offsets");
        check_cuda(cudaMemset(d_bboxes.get(), 0, sizeof(std::int64_t) * edge_count * 4), "clear boundary loop bboxes");
        check_cuda(cudaMemset(d_areas.get(), 0, sizeof(std::int64_t) * edge_count), "clear boundary loop areas");
        check_cuda(cudaMemset(d_stats.get(), 0, sizeof(DeviceLoopTraceStats)), "clear boundary loop stats");
        check_cuda(cudaEventCreate(&event_start), "create boundary loop start event");
        check_cuda(cudaEventCreate(&event_stop), "create boundary loop stop event");
        check_cuda(cudaEventRecord(event_start), "record boundary loop start event");
        {
            const int threads = 256;
            const int blocks = static_cast<int>((edge_count + threads - 1) / threads);
            boundary_loop_next_kernel<<<blocks, threads>>>(d_edges.get(), edge_count, d_next_indices.get());
            check_cuda(cudaGetLastError(), "launch boundary loop next kernel");
        }
        boundary_loop_trace_kernel<<<1, 1>>>(
            d_edges.get(),
            edge_count,
            d_next_indices.get(),
            d_remaining.get(),
            d_scratch_a.get(),
            d_scratch_b.get(),
            d_vertices.get(),
            d_offsets.get(),
            d_bboxes.get(),
            d_areas.get(),
            d_stats.get());
        check_cuda(cudaGetLastError(), "launch boundary loop trace kernel");
        check_cuda(cudaEventRecord(event_stop), "record boundary loop stop event");
        check_cuda(cudaEventSynchronize(event_stop), "synchronize boundary loop stop event");
        DeviceLoopTraceStats stats{};
        check_cuda(cudaMemcpy(&stats, d_stats.get(), sizeof(DeviceLoopTraceStats), cudaMemcpyDeviceToHost), "copy boundary loop stats");
        if (stats.error) {
            throw std::runtime_error(stats.error == 1 ? "CUDA boundary loop trace exceeded raw vertex limit" : "CUDA boundary loop trace exceeded output limit");
        }
        result->fragment_count = stats.fragment_count;
        result->vertex_count = stats.vertex_count;
        result->dropped_zero_area_loop_count = stats.dropped_zero_area_loop_count;
        result->cuda_kernel_count = 1;
        check_cuda(cudaEventElapsedTime(&result->cuda_event_elapsed_ms, event_start, event_stop), "measure boundary loop elapsed time");
        if (stats.fragment_count) {
            result->fragment_offsets = static_cast<std::uint64_t*>(std::malloc(sizeof(std::uint64_t) * (stats.fragment_count + 1)));
            result->vertices_xy = static_cast<std::int64_t*>(std::malloc(sizeof(std::int64_t) * stats.vertex_count * 2));
            result->bboxes = static_cast<std::int64_t*>(std::malloc(sizeof(std::int64_t) * stats.fragment_count * 4));
            result->twice_areas = static_cast<std::int64_t*>(std::malloc(sizeof(std::int64_t) * stats.fragment_count));
            if (!result->fragment_offsets || !result->vertices_xy || !result->bboxes || !result->twice_areas) {
                throw std::runtime_error("CUDA boundary loop host allocation failed");
            }
            check_cuda(cudaMemcpy(result->fragment_offsets, d_offsets.get(), sizeof(std::uint64_t) * (stats.fragment_count + 1), cudaMemcpyDeviceToHost), "copy boundary loop offsets");
            check_cuda(cudaMemcpy(result->vertices_xy, d_vertices.get(), sizeof(std::int64_t) * stats.vertex_count * 2, cudaMemcpyDeviceToHost), "copy boundary loop vertices");
            check_cuda(cudaMemcpy(result->bboxes, d_bboxes.get(), sizeof(std::int64_t) * stats.fragment_count * 4, cudaMemcpyDeviceToHost), "copy boundary loop bboxes");
            check_cuda(cudaMemcpy(result->twice_areas, d_areas.get(), sizeof(std::int64_t) * stats.fragment_count, cudaMemcpyDeviceToHost), "copy boundary loop areas");
        } else {
            result->fragment_offsets = static_cast<std::uint64_t*>(std::malloc(sizeof(std::uint64_t)));
            if (!result->fragment_offsets) {
                throw std::runtime_error("CUDA boundary loop empty offset allocation failed");
            }
            result->fragment_offsets[0] = 0;
        }
        cudaEventDestroy(event_start);
        cudaEventDestroy(event_stop);
        g_last_error.clear();
        return 0;
    } catch (const std::exception& exc) {
        if (event_start) {
            cudaEventDestroy(event_start);
        }
        if (event_stop) {
            cudaEventDestroy(event_stop);
        }
        g_last_error = exc.what();
        if (result) {
            gds_cuda_free_boundary_loop_result(result);
        }
        return 1;
    } catch (...) {
        if (event_start) {
            cudaEventDestroy(event_start);
        }
        if (event_stop) {
            cudaEventDestroy(event_stop);
        }
        g_last_error = "unknown CUDA boundary loop error";
        if (result) {
            gds_cuda_free_boundary_loop_result(result);
        }
        return 1;
    }
}

int gds_cuda_axis_aligned_boundary_edges_from_polygons(
    const std::int64_t* vertices_xy,
    const std::uint64_t* fragment_offsets,
    std::uint64_t fragment_count,
    GdsCudaBoundaryEdgesResult* result) {
    cudaEvent_t event_start = nullptr;
    cudaEvent_t event_stop = nullptr;
    try {
        if (!result) {
            throw std::invalid_argument("axis-aligned boundary edge result pointer is null");
        }
        *result = GdsCudaBoundaryEdgesResult{};
        if (fragment_count == 0) {
            g_last_error.clear();
            return 0;
        }
        if (!vertices_xy || !fragment_offsets) {
            throw std::invalid_argument("axis-aligned boundary edge input contains null pointer");
        }
        const std::uint64_t vertex_count = fragment_offsets[fragment_count];
        if (vertex_count == 0) {
            g_last_error.clear();
            return 0;
        }

        check_cuda(cudaEventCreate(&event_start), "create axis boundary edge start event");
        check_cuda(cudaEventCreate(&event_stop), "create axis boundary edge stop event");
        check_cuda(cudaEventRecord(event_start), "record axis boundary edge start event");

        DeviceBuffer<std::int64_t> d_vertices(vertex_count * 2);
        DeviceBuffer<std::uint64_t> d_offsets(fragment_count + 1);
        DeviceBuffer<AxisRawEdge> d_raw_edges(vertex_count);
        DeviceBuffer<AxisEndpointKey> d_endpoint_storage(vertex_count * 2);
        DeviceBuffer<std::uint32_t> d_error(1);
        check_cuda(cudaMemcpy(d_vertices.get(), vertices_xy, sizeof(std::int64_t) * vertex_count * 2, cudaMemcpyHostToDevice), "copy axis boundary vertices");
        check_cuda(cudaMemcpy(d_offsets.get(), fragment_offsets, sizeof(std::uint64_t) * (fragment_count + 1), cudaMemcpyHostToDevice), "copy axis boundary offsets");
        check_cuda(cudaMemset(d_raw_edges.get(), 0, sizeof(AxisRawEdge) * vertex_count), "clear axis raw edges");
        check_cuda(cudaMemset(d_error.get(), 0, sizeof(std::uint32_t)), "clear axis boundary error");

        const int threads = 256;
        const int fragment_blocks = static_cast<int>(std::min<std::uint64_t>((fragment_count + threads - 1) / threads, 65535));
        axis_raw_edge_kernel<<<fragment_blocks, threads>>>(
            d_vertices.get(),
            d_offsets.get(),
            fragment_count,
            d_raw_edges.get(),
            d_endpoint_storage.get(),
            d_error.get());
        check_cuda(cudaGetLastError(), "launch axis raw edge kernel");

        std::uint32_t error = 0;
        check_cuda(cudaMemcpy(&error, d_error.get(), sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "copy axis raw edge error");
        if (error) {
            throw std::runtime_error("axis-aligned boundary edge CUDA path received non-axis-aligned edge");
        }

        thrust::device_vector<AxisEndpointKey> d_endpoints(vertex_count * 2);
        check_cuda(cudaMemcpy(thrust::raw_pointer_cast(d_endpoints.data()), d_endpoint_storage.get(), sizeof(AxisEndpointKey) * vertex_count * 2, cudaMemcpyDeviceToDevice), "copy axis endpoints to thrust vector");
        auto valid_end = thrust::remove_if(d_endpoints.begin(), d_endpoints.end(), AxisEndpointInvalid{});
        d_endpoints.erase(valid_end, d_endpoints.end());
        if (d_endpoints.empty()) {
            check_cuda(cudaEventRecord(event_stop), "record empty axis boundary edge stop event");
            check_cuda(cudaEventSynchronize(event_stop), "synchronize empty axis boundary edge stop event");
            check_cuda(cudaEventElapsedTime(&result->cuda_event_elapsed_ms, event_start, event_stop), "measure empty axis boundary edge elapsed time");
            result->cuda_kernel_count = 1;
            cudaEventDestroy(event_start);
            cudaEventDestroy(event_stop);
            g_last_error.clear();
            return 0;
        }
        thrust::sort(d_endpoints.begin(), d_endpoints.end());
        auto unique_end = thrust::unique(d_endpoints.begin(), d_endpoints.end());
        d_endpoints.erase(unique_end, d_endpoints.end());
        const std::uint64_t endpoint_count = d_endpoints.size();

        DeviceBuffer<std::uint64_t> d_split_counts(vertex_count);
        DeviceBuffer<std::uint64_t> d_split_offsets(vertex_count);
        const int edge_blocks = static_cast<int>(std::min<std::uint64_t>((vertex_count + threads - 1) / threads, 65535));
        axis_split_count_kernel<<<edge_blocks, threads>>>(
            d_raw_edges.get(),
            vertex_count,
            thrust::raw_pointer_cast(d_endpoints.data()),
            endpoint_count,
            d_split_counts.get());
        check_cuda(cudaGetLastError(), "launch axis split count kernel");
        thrust::device_ptr<std::uint64_t> split_counts_begin(d_split_counts.get());
        thrust::device_ptr<std::uint64_t> split_counts_end(d_split_counts.get() + vertex_count);
        thrust::device_ptr<std::uint64_t> split_offsets_begin(d_split_offsets.get());
        thrust::exclusive_scan(thrust::device, split_counts_begin, split_counts_end, split_offsets_begin);

        std::uint64_t last_offset = 0;
        std::uint64_t last_count = 0;
        check_cuda(cudaMemcpy(&last_offset, d_split_offsets.get() + (vertex_count - 1), sizeof(std::uint64_t), cudaMemcpyDeviceToHost), "copy axis split last offset");
        check_cuda(cudaMemcpy(&last_count, d_split_counts.get() + (vertex_count - 1), sizeof(std::uint64_t), cudaMemcpyDeviceToHost), "copy axis split last count");
        if (last_count > std::numeric_limits<std::uint64_t>::max() - last_offset) {
            throw std::overflow_error("axis split edge count overflow");
        }
        const std::uint64_t split_edge_count = last_offset + last_count;
        result->cuda_kernel_count = 4;
        if (split_edge_count == 0) {
            check_cuda(cudaEventRecord(event_stop), "record no-split axis boundary edge stop event");
            check_cuda(cudaEventSynchronize(event_stop), "synchronize no-split axis boundary edge stop event");
            check_cuda(cudaEventElapsedTime(&result->cuda_event_elapsed_ms, event_start, event_stop), "measure no-split axis boundary edge elapsed time");
            cudaEventDestroy(event_start);
            cudaEventDestroy(event_stop);
            g_last_error.clear();
            return 0;
        }

        DeviceBuffer<EdgeKey> d_split_edges(split_edge_count);
        axis_split_emit_kernel<<<edge_blocks, threads>>>(
            d_raw_edges.get(),
            vertex_count,
            thrust::raw_pointer_cast(d_endpoints.data()),
            endpoint_count,
            d_split_offsets.get(),
            d_split_edges.get());
        check_cuda(cudaGetLastError(), "launch axis split emit kernel");

        thrust::device_ptr<EdgeKey> split_begin(d_split_edges.get());
        thrust::device_ptr<EdgeKey> split_end(d_split_edges.get() + split_edge_count);
        thrust::sort(thrust::device, split_begin, split_end);
        thrust::device_vector<EdgeKey> d_unique_edges(split_edge_count);
        thrust::device_vector<std::uint64_t> d_unique_counts(split_edge_count);
        auto reduce_end = thrust::reduce_by_key(
            thrust::device,
            split_begin,
            split_end,
            thrust::make_constant_iterator<std::uint64_t>(1),
            d_unique_edges.begin(),
            d_unique_counts.begin());
        const std::uint64_t unique_count = reduce_end.first - d_unique_edges.begin();
        if (unique_count == 0) {
            check_cuda(cudaEventRecord(event_stop), "record no-unique axis boundary edge stop event");
            check_cuda(cudaEventSynchronize(event_stop), "synchronize no-unique axis boundary edge stop event");
            check_cuda(cudaEventElapsedTime(&result->cuda_event_elapsed_ms, event_start, event_stop), "measure no-unique axis boundary edge elapsed time");
            cudaEventDestroy(event_start);
            cudaEventDestroy(event_stop);
            g_last_error.clear();
            return 0;
        }
        d_unique_edges.resize(unique_count);
        d_unique_counts.resize(unique_count);

        thrust::device_vector<std::uint64_t> d_surplus_counts(unique_count);
        const int unique_blocks = static_cast<int>(std::min<std::uint64_t>((unique_count + threads - 1) / threads, 65535));
        edge_surplus_count_kernel<<<unique_blocks, threads>>>(
            thrust::raw_pointer_cast(d_unique_edges.data()),
            thrust::raw_pointer_cast(d_unique_counts.data()),
            unique_count,
            thrust::raw_pointer_cast(d_surplus_counts.data()));
        check_cuda(cudaGetLastError(), "launch axis edge surplus count kernel");
        thrust::device_vector<std::uint64_t> d_surplus_offsets(unique_count);
        thrust::exclusive_scan(d_surplus_counts.begin(), d_surplus_counts.end(), d_surplus_offsets.begin());
        check_cuda(cudaMemcpy(&last_offset, thrust::raw_pointer_cast(d_surplus_offsets.data()) + (unique_count - 1), sizeof(std::uint64_t), cudaMemcpyDeviceToHost), "copy axis surplus last offset");
        check_cuda(cudaMemcpy(&last_count, thrust::raw_pointer_cast(d_surplus_counts.data()) + (unique_count - 1), sizeof(std::uint64_t), cudaMemcpyDeviceToHost), "copy axis surplus last count");
        if (last_count > std::numeric_limits<std::uint64_t>::max() - last_offset) {
            throw std::overflow_error("axis surplus edge count overflow");
        }
        result->edge_count = last_offset + last_count;
        if (result->edge_count) {
            DeviceBuffer<std::int64_t> d_out_edges(result->edge_count * 4);
            edge_surplus_emit_kernel<<<unique_blocks, threads>>>(
                thrust::raw_pointer_cast(d_unique_edges.data()),
                thrust::raw_pointer_cast(d_surplus_counts.data()),
                thrust::raw_pointer_cast(d_surplus_offsets.data()),
                unique_count,
                d_out_edges.get());
            check_cuda(cudaGetLastError(), "launch axis edge surplus emit kernel");
            check_cuda(cudaEventRecord(event_stop), "record axis boundary edge stop event");
            check_cuda(cudaEventSynchronize(event_stop), "synchronize axis boundary edge stop event");
            check_cuda(cudaEventElapsedTime(&result->cuda_event_elapsed_ms, event_start, event_stop), "measure axis boundary edge elapsed time");
            result->edges = static_cast<std::int64_t*>(std::malloc(sizeof(std::int64_t) * result->edge_count * 4));
            if (!result->edges) {
                throw std::runtime_error("axis boundary edge output allocation failed");
            }
            check_cuda(cudaMemcpy(result->edges, d_out_edges.get(), sizeof(std::int64_t) * result->edge_count * 4, cudaMemcpyDeviceToHost), "copy axis boundary edges");
        } else {
            check_cuda(cudaEventRecord(event_stop), "record empty-surplus axis boundary edge stop event");
            check_cuda(cudaEventSynchronize(event_stop), "synchronize empty-surplus axis boundary edge stop event");
            check_cuda(cudaEventElapsedTime(&result->cuda_event_elapsed_ms, event_start, event_stop), "measure empty-surplus axis boundary edge elapsed time");
        }
        result->cuda_kernel_count = 6;
        cudaEventDestroy(event_start);
        cudaEventDestroy(event_stop);
        g_last_error.clear();
        return 0;
    } catch (const std::exception& exc) {
        if (event_start) {
            cudaEventDestroy(event_start);
        }
        if (event_stop) {
            cudaEventDestroy(event_stop);
        }
        g_last_error = exc.what();
        if (result) {
            gds_cuda_free_boundary_edges_result(result);
        }
        return 1;
    } catch (...) {
        if (event_start) {
            cudaEventDestroy(event_start);
        }
        if (event_stop) {
            cudaEventDestroy(event_stop);
        }
        g_last_error = "unknown CUDA axis boundary edge error";
        if (result) {
            gds_cuda_free_boundary_edges_result(result);
        }
        return 1;
    }
}

int gds_cuda_split_free_boundary_edges_from_polygons(
    const std::int64_t* vertices_xy,
    const std::uint64_t* fragment_offsets,
    std::uint64_t fragment_count,
    GdsCudaBoundaryEdgesResult* result) {
    cudaEvent_t event_start = nullptr;
    cudaEvent_t event_stop = nullptr;
    try {
        if (!result) {
            throw std::invalid_argument("split-free boundary edge result pointer is null");
        }
        *result = GdsCudaBoundaryEdgesResult{};
        if (fragment_count == 0) {
            g_last_error.clear();
            return 0;
        }
        if (!vertices_xy || !fragment_offsets) {
            throw std::invalid_argument("split-free boundary edge input contains null pointer");
        }
        const std::uint64_t vertex_count = fragment_offsets[fragment_count];
        if (vertex_count == 0) {
            g_last_error.clear();
            return 0;
        }
        if (vertex_count > 100000) {
            throw std::overflow_error("split-free boundary edge input is too large for quadratic safety check");
        }
        if (vertex_count > std::numeric_limits<std::uint64_t>::max() / vertex_count) {
            throw std::overflow_error("split-free boundary edge pair count overflow");
        }

        check_cuda(cudaEventCreate(&event_start), "create split-free boundary edge start event");
        check_cuda(cudaEventCreate(&event_stop), "create split-free boundary edge stop event");
        check_cuda(cudaEventRecord(event_start), "record split-free boundary edge start event");

        DeviceBuffer<std::int64_t> d_vertices(vertex_count * 2);
        DeviceBuffer<std::uint64_t> d_offsets(fragment_count + 1);
        DeviceBuffer<EdgeKey> d_raw_edges(vertex_count);
        DeviceBuffer<std::uint32_t> d_split_required(1);
        check_cuda(cudaMemcpy(d_vertices.get(), vertices_xy, sizeof(std::int64_t) * vertex_count * 2, cudaMemcpyHostToDevice), "copy split-free boundary vertices");
        check_cuda(cudaMemcpy(d_offsets.get(), fragment_offsets, sizeof(std::uint64_t) * (fragment_count + 1), cudaMemcpyHostToDevice), "copy split-free boundary offsets");
        check_cuda(cudaMemset(d_raw_edges.get(), 0, sizeof(EdgeKey) * vertex_count), "clear split-free raw edges");
        check_cuda(cudaMemset(d_split_required.get(), 0, sizeof(std::uint32_t)), "clear split-free required flag");

        const int threads = 256;
        const int fragment_blocks = static_cast<int>(std::min<std::uint64_t>((fragment_count + threads - 1) / threads, 65535));
        split_free_raw_edge_kernel<<<fragment_blocks, threads>>>(
            d_vertices.get(),
            d_offsets.get(),
            fragment_count,
            d_raw_edges.get());
        check_cuda(cudaGetLastError(), "launch split-free raw edge kernel");

        const std::uint64_t total_pairs = vertex_count * vertex_count;
        const int pair_blocks = static_cast<int>(std::min<std::uint64_t>((total_pairs + threads - 1) / threads, 65535));
        split_free_required_kernel<<<pair_blocks, threads>>>(
            d_raw_edges.get(),
            vertex_count,
            d_split_required.get());
        check_cuda(cudaGetLastError(), "launch split-free required kernel");
        std::uint32_t split_required = 0;
        check_cuda(cudaMemcpy(&split_required, d_split_required.get(), sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "copy split-free required flag");
        if (split_required) {
            throw std::runtime_error("split-free boundary edge CUDA path requires collinear splitting");
        }

        thrust::device_vector<EdgeKey> d_edges(vertex_count);
        check_cuda(cudaMemcpy(thrust::raw_pointer_cast(d_edges.data()), d_raw_edges.get(), sizeof(EdgeKey) * vertex_count, cudaMemcpyDeviceToDevice), "copy split-free raw edges to thrust vector");
        auto valid_end = thrust::remove_if(d_edges.begin(), d_edges.end(), EdgeInvalid{});
        d_edges.erase(valid_end, d_edges.end());
        if (d_edges.empty()) {
            check_cuda(cudaEventRecord(event_stop), "record empty split-free boundary edge stop event");
            check_cuda(cudaEventSynchronize(event_stop), "synchronize empty split-free boundary edge stop event");
            check_cuda(cudaEventElapsedTime(&result->cuda_event_elapsed_ms, event_start, event_stop), "measure empty split-free boundary edge elapsed time");
            result->cuda_kernel_count = 2;
            cudaEventDestroy(event_start);
            cudaEventDestroy(event_stop);
            g_last_error.clear();
            return 0;
        }
        thrust::sort(d_edges.begin(), d_edges.end());
        const std::uint64_t edge_count = d_edges.size();
        thrust::device_vector<EdgeKey> d_unique_edges(edge_count);
        thrust::device_vector<std::uint64_t> d_unique_counts(edge_count);
        auto reduce_end = thrust::reduce_by_key(
            thrust::device,
            d_edges.begin(),
            d_edges.end(),
            thrust::make_constant_iterator<std::uint64_t>(1),
            d_unique_edges.begin(),
            d_unique_counts.begin());
        const std::uint64_t unique_count = reduce_end.first - d_unique_edges.begin();
        if (unique_count == 0) {
            check_cuda(cudaEventRecord(event_stop), "record no-unique split-free boundary edge stop event");
            check_cuda(cudaEventSynchronize(event_stop), "synchronize no-unique split-free boundary edge stop event");
            check_cuda(cudaEventElapsedTime(&result->cuda_event_elapsed_ms, event_start, event_stop), "measure no-unique split-free boundary edge elapsed time");
            result->cuda_kernel_count = 2;
            cudaEventDestroy(event_start);
            cudaEventDestroy(event_stop);
            g_last_error.clear();
            return 0;
        }
        d_unique_edges.resize(unique_count);
        d_unique_counts.resize(unique_count);

        thrust::device_vector<std::uint64_t> d_surplus_counts(unique_count);
        const int unique_blocks = static_cast<int>(std::min<std::uint64_t>((unique_count + threads - 1) / threads, 65535));
        edge_surplus_count_kernel<<<unique_blocks, threads>>>(
            thrust::raw_pointer_cast(d_unique_edges.data()),
            thrust::raw_pointer_cast(d_unique_counts.data()),
            unique_count,
            thrust::raw_pointer_cast(d_surplus_counts.data()));
        check_cuda(cudaGetLastError(), "launch split-free edge surplus count kernel");
        thrust::device_vector<std::uint64_t> d_surplus_offsets(unique_count);
        thrust::exclusive_scan(d_surplus_counts.begin(), d_surplus_counts.end(), d_surplus_offsets.begin());
        std::uint64_t last_offset = 0;
        std::uint64_t last_count = 0;
        check_cuda(cudaMemcpy(&last_offset, thrust::raw_pointer_cast(d_surplus_offsets.data()) + (unique_count - 1), sizeof(std::uint64_t), cudaMemcpyDeviceToHost), "copy split-free surplus last offset");
        check_cuda(cudaMemcpy(&last_count, thrust::raw_pointer_cast(d_surplus_counts.data()) + (unique_count - 1), sizeof(std::uint64_t), cudaMemcpyDeviceToHost), "copy split-free surplus last count");
        if (last_count > std::numeric_limits<std::uint64_t>::max() - last_offset) {
            throw std::overflow_error("split-free surplus edge count overflow");
        }
        result->edge_count = last_offset + last_count;
        if (result->edge_count) {
            DeviceBuffer<std::int64_t> d_out_edges(result->edge_count * 4);
            edge_surplus_emit_kernel<<<unique_blocks, threads>>>(
                thrust::raw_pointer_cast(d_unique_edges.data()),
                thrust::raw_pointer_cast(d_surplus_counts.data()),
                thrust::raw_pointer_cast(d_surplus_offsets.data()),
                unique_count,
                d_out_edges.get());
            check_cuda(cudaGetLastError(), "launch split-free edge surplus emit kernel");
            check_cuda(cudaEventRecord(event_stop), "record split-free boundary edge stop event");
            check_cuda(cudaEventSynchronize(event_stop), "synchronize split-free boundary edge stop event");
            check_cuda(cudaEventElapsedTime(&result->cuda_event_elapsed_ms, event_start, event_stop), "measure split-free boundary edge elapsed time");
            result->edges = static_cast<std::int64_t*>(std::malloc(sizeof(std::int64_t) * result->edge_count * 4));
            if (!result->edges) {
                throw std::runtime_error("split-free boundary edge output allocation failed");
            }
            check_cuda(cudaMemcpy(result->edges, d_out_edges.get(), sizeof(std::int64_t) * result->edge_count * 4, cudaMemcpyDeviceToHost), "copy split-free boundary edges");
        } else {
            check_cuda(cudaEventRecord(event_stop), "record empty-surplus split-free boundary edge stop event");
            check_cuda(cudaEventSynchronize(event_stop), "synchronize empty-surplus split-free boundary edge stop event");
            check_cuda(cudaEventElapsedTime(&result->cuda_event_elapsed_ms, event_start, event_stop), "measure empty-surplus split-free boundary edge elapsed time");
        }
        result->cuda_kernel_count = 5;
        cudaEventDestroy(event_start);
        cudaEventDestroy(event_stop);
        g_last_error.clear();
        return 0;
    } catch (const std::exception& exc) {
        if (event_start) {
            cudaEventDestroy(event_start);
        }
        if (event_stop) {
            cudaEventDestroy(event_stop);
        }
        g_last_error = exc.what();
        if (result) {
            gds_cuda_free_boundary_edges_result(result);
        }
        return 1;
    } catch (...) {
        if (event_start) {
            cudaEventDestroy(event_start);
        }
        if (event_stop) {
            cudaEventDestroy(event_stop);
        }
        g_last_error = "unknown CUDA split-free boundary edge error";
        if (result) {
            gds_cuda_free_boundary_edges_result(result);
        }
        return 1;
    }
}

int gds_cuda_context_overlap_pairs(
    const std::int64_t* component_bboxes,
    const std::int32_t* component_layers,
    const std::int32_t* component_datatypes,
    std::uint64_t component_count,
    const std::int64_t* polygon_bboxes,
    const std::int32_t* polygon_layers,
    const std::int32_t* polygon_datatypes,
    std::uint64_t polygon_count,
    GdsCudaContextOverlapPairsResult* result) {
    cudaEvent_t event_start = nullptr;
    cudaEvent_t event_stop = nullptr;
    try {
        if (!result) {
            throw std::invalid_argument("context overlap result pointer is null");
        }
        *result = GdsCudaContextOverlapPairsResult{};
        if (component_count == 0 || polygon_count == 0) {
            g_last_error.clear();
            return 0;
        }
        if (!component_bboxes || !component_layers || !component_datatypes || !polygon_bboxes || !polygon_layers || !polygon_datatypes) {
            throw std::invalid_argument("context overlap input contains null pointer");
        }
        if (component_count > std::numeric_limits<std::uint64_t>::max() / polygon_count) {
            throw std::overflow_error("context overlap pair search count overflow");
        }
        const std::uint64_t total_pairs = component_count * polygon_count;
        check_cuda(cudaEventCreate(&event_start), "create context overlap start event");
        check_cuda(cudaEventCreate(&event_stop), "create context overlap stop event");
        check_cuda(cudaEventRecord(event_start), "record context overlap start event");

        DeviceBuffer<std::int64_t> d_component_bboxes(component_count * 4);
        DeviceBuffer<std::int32_t> d_component_layers(component_count);
        DeviceBuffer<std::int32_t> d_component_datatypes(component_count);
        DeviceBuffer<std::int64_t> d_polygon_bboxes(polygon_count * 4);
        DeviceBuffer<std::int32_t> d_polygon_layers(polygon_count);
        DeviceBuffer<std::int32_t> d_polygon_datatypes(polygon_count);
        DeviceBuffer<std::uint64_t> d_flags(total_pairs);

        check_cuda(cudaMemcpy(d_component_bboxes.get(), component_bboxes, sizeof(std::int64_t) * component_count * 4, cudaMemcpyHostToDevice), "copy context component bboxes");
        check_cuda(cudaMemcpy(d_component_layers.get(), component_layers, sizeof(std::int32_t) * component_count, cudaMemcpyHostToDevice), "copy context component layers");
        check_cuda(cudaMemcpy(d_component_datatypes.get(), component_datatypes, sizeof(std::int32_t) * component_count, cudaMemcpyHostToDevice), "copy context component datatypes");
        check_cuda(cudaMemcpy(d_polygon_bboxes.get(), polygon_bboxes, sizeof(std::int64_t) * polygon_count * 4, cudaMemcpyHostToDevice), "copy context polygon bboxes");
        check_cuda(cudaMemcpy(d_polygon_layers.get(), polygon_layers, sizeof(std::int32_t) * polygon_count, cudaMemcpyHostToDevice), "copy context polygon layers");
        check_cuda(cudaMemcpy(d_polygon_datatypes.get(), polygon_datatypes, sizeof(std::int32_t) * polygon_count, cudaMemcpyHostToDevice), "copy context polygon datatypes");

        const int threads = 256;
        const int blocks = static_cast<int>(std::min<std::uint64_t>((total_pairs + threads - 1) / threads, 65535));
        context_overlap_flag_kernel<<<blocks, threads>>>(
            d_component_bboxes.get(),
            d_component_layers.get(),
            d_component_datatypes.get(),
            component_count,
            d_polygon_bboxes.get(),
            d_polygon_layers.get(),
            d_polygon_datatypes.get(),
            polygon_count,
            total_pairs,
            d_flags.get());
        check_cuda(cudaGetLastError(), "launch context overlap flag kernel");

        thrust::device_ptr<std::uint64_t> flags_begin(d_flags.get());
        thrust::device_ptr<std::uint64_t> flags_end(d_flags.get() + total_pairs);
        thrust::device_vector<std::uint64_t> d_offsets(total_pairs);
        thrust::exclusive_scan(thrust::device, flags_begin, flags_end, d_offsets.begin());

        std::uint64_t last_offset = 0;
        std::uint64_t last_flag = 0;
        check_cuda(cudaMemcpy(&last_offset, thrust::raw_pointer_cast(d_offsets.data()) + (total_pairs - 1), sizeof(std::uint64_t), cudaMemcpyDeviceToHost), "copy context overlap last offset");
        check_cuda(cudaMemcpy(&last_flag, d_flags.get() + (total_pairs - 1), sizeof(std::uint64_t), cudaMemcpyDeviceToHost), "copy context overlap last flag");
        if (last_flag > std::numeric_limits<std::uint64_t>::max() - last_offset) {
            throw std::overflow_error("context overlap pair count overflow");
        }
        result->pair_count = last_offset + last_flag;
        result->cuda_kernel_count = 3;

        if (result->pair_count) {
            DeviceBuffer<std::uint64_t> d_component_indices(result->pair_count);
            DeviceBuffer<std::uint64_t> d_polygon_indices(result->pair_count);
            context_overlap_emit_kernel<<<blocks, threads>>>(
                d_flags.get(),
                thrust::raw_pointer_cast(d_offsets.data()),
                polygon_count,
                total_pairs,
                d_component_indices.get(),
                d_polygon_indices.get());
            check_cuda(cudaGetLastError(), "launch context overlap emit kernel");
            check_cuda(cudaEventRecord(event_stop), "record context overlap stop event");
            check_cuda(cudaEventSynchronize(event_stop), "synchronize context overlap stop event");
            check_cuda(cudaEventElapsedTime(&result->cuda_event_elapsed_ms, event_start, event_stop), "measure context overlap elapsed time");

            result->component_indices = static_cast<std::uint64_t*>(std::malloc(sizeof(std::uint64_t) * result->pair_count));
            result->polygon_indices = static_cast<std::uint64_t*>(std::malloc(sizeof(std::uint64_t) * result->pair_count));
            if (!result->component_indices || !result->polygon_indices) {
                throw std::runtime_error("context overlap output allocation failed");
            }
            check_cuda(cudaMemcpy(result->component_indices, d_component_indices.get(), sizeof(std::uint64_t) * result->pair_count, cudaMemcpyDeviceToHost), "copy context overlap component indices");
            check_cuda(cudaMemcpy(result->polygon_indices, d_polygon_indices.get(), sizeof(std::uint64_t) * result->pair_count, cudaMemcpyDeviceToHost), "copy context overlap polygon indices");
        } else {
            check_cuda(cudaEventRecord(event_stop), "record empty context overlap stop event");
            check_cuda(cudaEventSynchronize(event_stop), "synchronize empty context overlap stop event");
            check_cuda(cudaEventElapsedTime(&result->cuda_event_elapsed_ms, event_start, event_stop), "measure empty context overlap elapsed time");
        }
        cudaEventDestroy(event_start);
        cudaEventDestroy(event_stop);
        g_last_error.clear();
        return 0;
    } catch (const std::exception& exc) {
        if (event_start) {
            cudaEventDestroy(event_start);
        }
        if (event_stop) {
            cudaEventDestroy(event_stop);
        }
        g_last_error = exc.what();
        if (result) {
            gds_cuda_free_context_overlap_pairs_result(result);
        }
        return 1;
    } catch (...) {
        if (event_start) {
            cudaEventDestroy(event_start);
        }
        if (event_stop) {
            cudaEventDestroy(event_stop);
        }
        g_last_error = "unknown CUDA context overlap error";
        if (result) {
            gds_cuda_free_context_overlap_pairs_result(result);
        }
        return 1;
    }
}

int gds_cuda_rect_set_exact_fragments(
    const std::int64_t* old_rects,
    const std::int64_t* new_rects,
    const std::uint8_t* old_counts,
    const std::uint8_t* new_counts,
    std::uint64_t task_count,
    std::uint16_t max_rects,
    std::uint16_t max_fragments,
    std::int64_t* out_rects,
    std::uint16_t* out_counts,
    std::int64_t* out_twice_areas) {
    try {
        if (!out_rects || !out_counts || !out_twice_areas) {
            throw std::invalid_argument("rect set exact fragment output pointer is null");
        }
        if (task_count == 0) {
            g_last_error.clear();
            return 0;
        }
        if (!old_rects || !new_rects || !old_counts || !new_counts) {
            throw std::invalid_argument("rect set exact fragments received null input pointer");
        }
        if (max_rects == 0 || max_rects > kMaxRectSetRects || max_fragments == 0) {
            throw std::invalid_argument("invalid rect set exact limits");
        }
        DeviceBuffer<std::int64_t> d_old_rects(task_count * max_rects * 4);
        DeviceBuffer<std::int64_t> d_new_rects(task_count * max_rects * 4);
        DeviceBuffer<std::uint8_t> d_old_counts(task_count);
        DeviceBuffer<std::uint8_t> d_new_counts(task_count);
        DeviceBuffer<std::int64_t> d_out_rects(task_count * 2 * max_fragments * 4);
        DeviceBuffer<std::uint16_t> d_out_counts(task_count * 2);
        DeviceBuffer<std::int64_t> d_out_areas(task_count * 3);
        DeviceBuffer<std::uint32_t> d_errors(task_count);
        check_cuda(cudaMemcpy(d_old_rects.get(), old_rects, sizeof(std::int64_t) * task_count * max_rects * 4, cudaMemcpyHostToDevice), "copy old rect set rects");
        check_cuda(cudaMemcpy(d_new_rects.get(), new_rects, sizeof(std::int64_t) * task_count * max_rects * 4, cudaMemcpyHostToDevice), "copy new rect set rects");
        check_cuda(cudaMemcpy(d_old_counts.get(), old_counts, sizeof(std::uint8_t) * task_count, cudaMemcpyHostToDevice), "copy old rect set counts");
        check_cuda(cudaMemcpy(d_new_counts.get(), new_counts, sizeof(std::uint8_t) * task_count, cudaMemcpyHostToDevice), "copy new rect set counts");
        check_cuda(cudaMemset(d_out_rects.get(), 0, sizeof(std::int64_t) * task_count * 2 * max_fragments * 4), "clear rect set fragments");
        check_cuda(cudaMemset(d_out_counts.get(), 0, sizeof(std::uint16_t) * task_count * 2), "clear rect set counts");
        check_cuda(cudaMemset(d_out_areas.get(), 0, sizeof(std::int64_t) * task_count * 3), "clear rect set areas");
        check_cuda(cudaMemset(d_errors.get(), 0, sizeof(std::uint32_t) * task_count), "clear rect set errors");
        const int threads = 128;
        const int blocks = static_cast<int>((task_count + threads - 1) / threads);
        rect_set_exact_fragments_kernel<<<blocks, threads>>>(
            d_old_rects.get(),
            d_new_rects.get(),
            d_old_counts.get(),
            d_new_counts.get(),
            task_count,
            max_rects,
            max_fragments,
            d_out_rects.get(),
            d_out_counts.get(),
            d_out_areas.get(),
            d_errors.get());
        check_cuda(cudaGetLastError(), "launch rect set exact fragment kernel");
        check_cuda(cudaDeviceSynchronize(), "run rect set exact fragment kernel");
        const auto max_error = thrust::reduce(
            thrust::device_pointer_cast(d_errors.get()),
            thrust::device_pointer_cast(d_errors.get()) + task_count,
            std::uint32_t{0},
            thrust::maximum<std::uint32_t>{});
        if (max_error) {
            throw std::runtime_error(max_error == 1 ? "rect set exact task exceeds supported rectangle limit" : "rect set exact task exceeds fragment limit");
        }
        check_cuda(cudaMemcpy(out_rects, d_out_rects.get(), sizeof(std::int64_t) * task_count * 2 * max_fragments * 4, cudaMemcpyDeviceToHost), "copy rect set exact fragments");
        check_cuda(cudaMemcpy(out_counts, d_out_counts.get(), sizeof(std::uint16_t) * task_count * 2, cudaMemcpyDeviceToHost), "copy rect set exact counts");
        check_cuda(cudaMemcpy(out_twice_areas, d_out_areas.get(), sizeof(std::int64_t) * task_count * 3, cudaMemcpyDeviceToHost), "copy rect set exact areas");
        g_last_error.clear();
        return 0;
    } catch (const std::exception& exc) {
        g_last_error = exc.what();
        return 1;
    } catch (...) {
        g_last_error = "unknown CUDA rect set exact fragment error";
        return 1;
    }
}

}  // extern "C"
