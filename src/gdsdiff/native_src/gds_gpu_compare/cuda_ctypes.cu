#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <algorithm>
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

const char* gds_cuda_last_error() {
    return g_last_error.c_str();
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
