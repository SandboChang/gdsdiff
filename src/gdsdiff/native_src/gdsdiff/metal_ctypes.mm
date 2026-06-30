#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <atomic>
#include <algorithm>
#include <cstdint>
#include <exception>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

thread_local std::string g_last_error;
std::mutex g_mutex;
id<MTLDevice> g_device = nil;
id<MTLCommandQueue> g_queue = nil;
id<MTLLibrary> g_library = nil;
id<MTLComputePipelineState> g_byte_compare_pipeline = nil;
id<MTLComputePipelineState> g_tile_count_pipeline = nil;
id<MTLComputePipelineState> g_tile_emit_pipeline = nil;
id<MTLComputePipelineState> g_rect_set_pipeline = nil;
id<MTLComputePipelineState> g_record_delta_pipeline_old = nil;
id<MTLComputePipelineState> g_record_delta_pipeline_new = nil;
id<MTLComputePipelineState> g_record_key_from_hash_pipeline = nil;
id<MTLComputePipelineState> g_context_overlap_pipeline = nil;
id<MTLComputePipelineState> g_one_sided_boundary_pipeline = nil;
id<MTLComputePipelineState> g_mixed_boundary_pipeline = nil;
id<MTLComputePipelineState> g_boundary_loop_next_pipeline = nil;
id<MTLComputePipelineState> g_boundary_loop_trace_pipeline = nil;

constexpr const char* kMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct ByteCompareParams {
    ulong byte_count;
};

static inline long metal_floor_div(long value, long divisor) {
    return static_cast<long>(floor(static_cast<float>(value) / static_cast<float>(divisor)));
}

kernel void byte_compare(
    device const uchar* old_data [[buffer(0)]],
    device const uchar* new_data [[buffer(1)]],
    device atomic_uint* mismatch [[buffer(2)]],
    constant ByteCompareParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    ulong index = gid;
    if (index >= params.byte_count) {
        return;
    }
    if (old_data[index] != new_data[index]) {
        atomic_store_explicit(mismatch, 1u, memory_order_relaxed);
    }
}

struct TileCountParams {
    ulong polygon_count;
    long tile_size;
    ulong max_tiles_per_polygon;
    uint side;
};

kernel void tile_count(
    device const long* bboxes [[buffer(0)]],
    device const int* layers [[buffer(1)]],
    device const int* datatypes [[buffer(2)]],
    device const ulong* canonical_ids [[buffer(3)]],
    device ulong* out_counts [[buffer(4)]],
    device uchar* out_oversized_flags [[buffer(5)]],
    device int* out_oversized_layers [[buffer(6)]],
    device int* out_oversized_datatypes [[buffer(7)]],
    device ulong* out_oversized_canonical_ids [[buffer(8)]],
    device long* out_oversized_tile_bboxes [[buffer(9)]],
    device uint* out_errors [[buffer(10)]],
    constant TileCountParams& params [[buffer(11)]],
    uint gid [[thread_position_in_grid]]
) {
    ulong index = gid;
    if (index >= params.polygon_count) {
        return;
    }
    long x0 = bboxes[index * 4];
    long y0 = bboxes[index * 4 + 1];
    long x1 = bboxes[index * 4 + 2];
    long y1 = bboxes[index * 4 + 3];
    long tx0 = metal_floor_div(x0, params.tile_size);
    long ty0 = metal_floor_div(y0, params.tile_size);
    long tx1 = metal_floor_div(x1, params.tile_size);
    long ty1 = metal_floor_div(y1, params.tile_size);
    long span_x = tx1 - tx0 + 1;
    long span_y = ty1 - ty0 + 1;
    if (span_x <= 0 || span_y <= 0) {
        out_errors[index] = 1;
        out_counts[index] = 0;
        out_oversized_flags[index] = 0;
        return;
    }
    ulong span_x_u = static_cast<ulong>(span_x);
    ulong span_y_u = static_cast<ulong>(span_y);
    ulong max_ulong = ~static_cast<ulong>(0);
    if (span_y_u != 0 && span_x_u > max_ulong / span_y_u) {
        out_errors[index] = 2;
        out_counts[index] = 0;
        out_oversized_flags[index] = 0;
        return;
    }
    ulong count = span_x_u * span_y_u;
    if (count > params.max_tiles_per_polygon) {
        out_counts[index] = 0;
        out_oversized_flags[index] = 1;
        out_oversized_layers[index] = layers[index];
        out_oversized_datatypes[index] = datatypes[index];
        out_oversized_canonical_ids[index] = canonical_ids[index];
        out_oversized_tile_bboxes[index * 4] = tx0;
        out_oversized_tile_bboxes[index * 4 + 1] = ty0;
        out_oversized_tile_bboxes[index * 4 + 2] = tx1;
        out_oversized_tile_bboxes[index * 4 + 3] = ty1;
    } else {
        out_counts[index] = count;
        out_oversized_flags[index] = 0;
    }
}

struct TileEmitParams {
    ulong polygon_count;
    long tile_size;
    uint side;
};

kernel void tile_emit(
    device const long* bboxes [[buffer(0)]],
    device const int* layers [[buffer(1)]],
    device const int* datatypes [[buffer(2)]],
    device const ulong* canonical_ids [[buffer(3)]],
    device const ulong* offsets [[buffer(4)]],
    device const uchar* oversized_flags [[buffer(5)]],
    device int* out_layers [[buffer(6)]],
    device int* out_datatypes [[buffer(7)]],
    device long* out_tx [[buffer(8)]],
    device long* out_ty [[buffer(9)]],
    device ulong* out_canonical_ids [[buffer(10)]],
    device uchar* out_sides [[buffer(11)]],
    constant TileEmitParams& params [[buffer(12)]],
    uint gid [[thread_position_in_grid]]
) {
    ulong index = gid;
    if (index >= params.polygon_count || oversized_flags[index]) {
        return;
    }
    long tx0 = metal_floor_div(bboxes[index * 4], params.tile_size);
    long ty0 = metal_floor_div(bboxes[index * 4 + 1], params.tile_size);
    long tx1 = metal_floor_div(bboxes[index * 4 + 2], params.tile_size);
    long ty1 = metal_floor_div(bboxes[index * 4 + 3], params.tile_size);
    ulong out = offsets[index];
    for (long tx = tx0; tx <= tx1; ++tx) {
        for (long ty = ty0; ty <= ty1; ++ty) {
            out_layers[out] = layers[index];
            out_datatypes[out] = datatypes[index];
            out_tx[out] = tx;
            out_ty[out] = ty;
            out_canonical_ids[out] = canonical_ids[index];
            out_sides[out] = static_cast<uchar>(params.side);
            out += 1;
        }
    }
}

struct Params {
    ulong task_count;
    ushort max_rects;
    ushort max_fragments;
};

static inline void add_break(thread long* values, thread ushort& count, long value) {
    values[count] = value;
    count += 1;
}

static inline ushort sort_unique(thread long* values, ushort count) {
    for (ushort i = 1; i < count; ++i) {
        long item = values[i];
        ushort j = i;
        while (j > 0 && values[j - 1] > item) {
            values[j] = values[j - 1];
            j -= 1;
        }
        values[j] = item;
    }
    if (count == 0) {
        return 0;
    }
    ushort out = 1;
    for (ushort i = 1; i < count; ++i) {
        if (values[i] != values[out - 1]) {
            values[out] = values[i];
            out += 1;
        }
    }
    return out;
}

static inline bool valid_rect(device const long* rects, ulong base) {
    return rects[base] < rects[base + 2] && rects[base + 1] < rects[base + 3];
}

static inline bool cell_inside_any(
    long x0,
    long y0,
    long x1,
    long y1,
    device const long* rects,
    ushort count,
    ushort max_rects,
    ulong task
) {
    for (ushort i = 0; i < count; ++i) {
        ulong base = (task * max_rects + i) * 4;
        if (!valid_rect(rects, base)) {
            continue;
        }
        if (x0 >= rects[base] && x1 <= rects[base + 2] && y0 >= rects[base + 1] && y1 <= rects[base + 3]) {
            return true;
        }
    }
    return false;
}

static inline void emit_cell(
    long x0,
    long y0,
    long x1,
    long y1,
    ulong task,
    ushort side,
    ushort max_fragments,
    device long* out_rects,
    thread ushort& count,
    thread long& twice_area,
    device atomic_uint* error
) {
    long area = (x1 - x0) * (y1 - y0) * 2;
    twice_area += area;
    if (count >= max_fragments) {
        atomic_store_explicit(error, 1u, memory_order_relaxed);
        return;
    }
    ulong base = ((task * 2 + side) * max_fragments + count) * 4;
    out_rects[base] = x0;
    out_rects[base + 1] = y0;
    out_rects[base + 2] = x1;
    out_rects[base + 3] = y1;
    count += 1;
}

static inline void diff_side(
    device const long* source_rects,
    device const long* cutter_rects,
    ushort source_count,
    ushort cutter_count,
    ushort max_rects,
    ushort max_fragments,
    ulong task,
    ushort side,
    device long* out_rects,
    device ushort* out_counts,
    device long* out_areas,
    device atomic_uint* error
) {
    thread long xs[64];
    thread long ys[64];
    ushort x_count = 0;
    ushort y_count = 0;
    for (ushort i = 0; i < source_count; ++i) {
        ulong base = (task * max_rects + i) * 4;
        if (!valid_rect(source_rects, base)) {
            continue;
        }
        add_break(xs, x_count, source_rects[base]);
        add_break(xs, x_count, source_rects[base + 2]);
        add_break(ys, y_count, source_rects[base + 1]);
        add_break(ys, y_count, source_rects[base + 3]);
    }
    for (ushort i = 0; i < cutter_count; ++i) {
        ulong base = (task * max_rects + i) * 4;
        if (!valid_rect(cutter_rects, base)) {
            continue;
        }
        add_break(xs, x_count, cutter_rects[base]);
        add_break(xs, x_count, cutter_rects[base + 2]);
        add_break(ys, y_count, cutter_rects[base + 1]);
        add_break(ys, y_count, cutter_rects[base + 3]);
    }
    x_count = sort_unique(xs, x_count);
    y_count = sort_unique(ys, y_count);
    ushort emitted = 0;
    long twice_area = 0;
    if (x_count >= 2 && y_count >= 2) {
        for (ushort xi = 0; xi + 1 < x_count; ++xi) {
            long x0 = xs[xi];
            long x1 = xs[xi + 1];
            if (x0 == x1) {
                continue;
            }
            for (ushort yi = 0; yi + 1 < y_count; ++yi) {
                long y0 = ys[yi];
                long y1 = ys[yi + 1];
                if (y0 == y1) {
                    continue;
                }
                bool in_source = cell_inside_any(x0, y0, x1, y1, source_rects, source_count, max_rects, task);
                bool in_cutter = cell_inside_any(x0, y0, x1, y1, cutter_rects, cutter_count, max_rects, task);
                if (in_source && !in_cutter) {
                    emit_cell(x0, y0, x1, y1, task, side, max_fragments, out_rects, emitted, twice_area, error);
                }
            }
        }
    }
    out_counts[task * 2 + side] = emitted;
    out_areas[task * 3 + side] = twice_area;
}

kernel void rect_set_exact(
    device const long* old_rects [[buffer(0)]],
    device const long* new_rects [[buffer(1)]],
    device const uchar* old_counts [[buffer(2)]],
    device const uchar* new_counts [[buffer(3)]],
    device long* out_rects [[buffer(4)]],
    device ushort* out_counts [[buffer(5)]],
    device long* out_areas [[buffer(6)]],
    device atomic_uint* error [[buffer(7)]],
    constant Params& params [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    ulong task = gid;
    if (task >= params.task_count) {
        return;
    }
    ushort old_count = min(static_cast<ushort>(old_counts[task]), params.max_rects);
    ushort new_count = min(static_cast<ushort>(new_counts[task]), params.max_rects);
    diff_side(old_rects, new_rects, old_count, new_count, params.max_rects, params.max_fragments, task, 0, out_rects, out_counts, out_areas, error);
    diff_side(new_rects, old_rects, new_count, old_count, params.max_rects, params.max_fragments, task, 1, out_rects, out_counts, out_areas, error);
    out_areas[task * 3 + 2] = out_areas[task * 3] + out_areas[task * 3 + 1];
}

struct RecordParams {
    ulong old_count;
    ulong new_count;
};

static inline ulong vertex_count(device const ulong* offsets, ulong index) {
    return offsets[index + 1] - offsets[index];
}

static inline bool record_key_equal(
    device const ulong* left_offsets,
    device const int* left_layers,
    device const int* left_datatypes,
    device const long* left_bboxes,
    device const uint* left_flags,
    device const ulong* left_hashes,
    ulong left_index,
    device const ulong* right_offsets,
    device const int* right_layers,
    device const int* right_datatypes,
    device const long* right_bboxes,
    device const uint* right_flags,
    device const ulong* right_hashes,
    ulong right_index
) {
    if (left_layers[left_index] != right_layers[right_index]) {
        return false;
    }
    if (left_datatypes[left_index] != right_datatypes[right_index]) {
        return false;
    }
    if (left_flags[left_index] != right_flags[right_index]) {
        return false;
    }
    if (vertex_count(left_offsets, left_index) != vertex_count(right_offsets, right_index)) {
        return false;
    }
    for (ushort offset = 0; offset < 4; ++offset) {
        if (left_bboxes[left_index * 4 + offset] != right_bboxes[right_index * 4 + offset]) {
            return false;
        }
    }
    return left_hashes[left_index * 2] == right_hashes[right_index * 2]
        && left_hashes[left_index * 2 + 1] == right_hashes[right_index * 2 + 1];
}

static inline bool first_in_side(
    device const ulong* offsets,
    device const int* layers,
    device const int* datatypes,
    device const long* bboxes,
    device const uint* flags,
    device const ulong* hashes,
    ulong index
) {
    for (ulong other = 0; other < index; ++other) {
        if (record_key_equal(offsets, layers, datatypes, bboxes, flags, hashes, index,
                             offsets, layers, datatypes, bboxes, flags, hashes, other)) {
            return false;
        }
    }
    return true;
}

static inline ulong count_same(
    device const ulong* source_offsets,
    device const int* source_layers,
    device const int* source_datatypes,
    device const long* source_bboxes,
    device const uint* source_flags,
    device const ulong* source_hashes,
    ulong source_index,
    device const ulong* target_offsets,
    device const int* target_layers,
    device const int* target_datatypes,
    device const long* target_bboxes,
    device const uint* target_flags,
    device const ulong* target_hashes,
    ulong target_count
) {
    ulong total = 0;
    for (ulong other = 0; other < target_count; ++other) {
        if (record_key_equal(source_offsets, source_layers, source_datatypes, source_bboxes, source_flags, source_hashes, source_index,
                             target_offsets, target_layers, target_datatypes, target_bboxes, target_flags, target_hashes, other)) {
            total += 1;
        }
    }
    return total;
}

static inline void emit_record_delta(
    uchar direction,
    ulong surplus,
    ulong representative,
    device const ulong* offsets,
    device const int* layers,
    device const int* datatypes,
    device const long* bboxes,
    device const uint* flags,
    device const ulong* hashes,
    device atomic_uint* record_counter,
    device int* out_layers,
    device int* out_datatypes,
    device uchar* out_directions,
    device ulong* out_counts,
    device long* out_bboxes,
    device ulong* out_vertex_counts,
    device uint* out_flags,
    device ulong* out_hashes,
    device ulong* out_representative_indices
) {
    uint row = atomic_fetch_add_explicit(record_counter, 1u, memory_order_relaxed);
    out_layers[row] = layers[representative];
    out_datatypes[row] = datatypes[representative];
    out_directions[row] = direction;
    out_counts[row] = surplus;
    for (ushort offset = 0; offset < 4; ++offset) {
        out_bboxes[row * 4 + offset] = bboxes[representative * 4 + offset];
    }
    out_vertex_counts[row] = vertex_count(offsets, representative);
    out_flags[row] = flags[representative];
    out_hashes[row * 2] = hashes[representative * 2];
    out_hashes[row * 2 + 1] = hashes[representative * 2 + 1];
    out_representative_indices[row] = representative;
}

kernel void record_delta_old(
    device const ulong* old_offsets [[buffer(0)]],
    device const int* old_layers [[buffer(1)]],
    device const int* old_datatypes [[buffer(2)]],
    device const long* old_bboxes [[buffer(3)]],
    device const uint* old_flags [[buffer(4)]],
    device const ulong* old_hashes [[buffer(5)]],
    device const ulong* new_offsets [[buffer(6)]],
    device const int* new_layers [[buffer(7)]],
    device const int* new_datatypes [[buffer(8)]],
    device const long* new_bboxes [[buffer(9)]],
    device const uint* new_flags [[buffer(10)]],
    device const ulong* new_hashes [[buffer(11)]],
    device int* out_layers [[buffer(12)]],
    device int* out_datatypes [[buffer(13)]],
    device uchar* out_directions [[buffer(14)]],
    device ulong* out_counts [[buffer(15)]],
    device long* out_bboxes [[buffer(16)]],
    device ulong* out_vertex_counts [[buffer(17)]],
    device uint* out_flags [[buffer(18)]],
    device ulong* out_hashes [[buffer(19)]],
    device ulong* out_representative_indices [[buffer(20)]],
    device atomic_uint* record_counter [[buffer(21)]],
    device atomic_uint* matched_counter [[buffer(22)]],
    device atomic_uint* reduced_counter [[buffer(23)]],
    constant RecordParams& params [[buffer(24)]],
    uint gid [[thread_position_in_grid]]
) {
    ulong index = gid;
    if (index >= params.old_count) {
        return;
    }
    if (!first_in_side(old_offsets, old_layers, old_datatypes, old_bboxes, old_flags, old_hashes, index)) {
        return;
    }
    ulong old_total = count_same(old_offsets, old_layers, old_datatypes, old_bboxes, old_flags, old_hashes, index,
                                 old_offsets, old_layers, old_datatypes, old_bboxes, old_flags, old_hashes, params.old_count);
    ulong new_total = count_same(old_offsets, old_layers, old_datatypes, old_bboxes, old_flags, old_hashes, index,
                                 new_offsets, new_layers, new_datatypes, new_bboxes, new_flags, new_hashes, params.new_count);
    atomic_fetch_add_explicit(reduced_counter, 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(matched_counter, static_cast<uint>(min(old_total, new_total)), memory_order_relaxed);
    if (old_total > new_total) {
        emit_record_delta(1, old_total - new_total, index, old_offsets, old_layers, old_datatypes, old_bboxes, old_flags, old_hashes,
                          record_counter, out_layers, out_datatypes, out_directions, out_counts, out_bboxes,
                          out_vertex_counts, out_flags, out_hashes, out_representative_indices);
    }
}

kernel void record_delta_new(
    device const ulong* old_offsets [[buffer(0)]],
    device const int* old_layers [[buffer(1)]],
    device const int* old_datatypes [[buffer(2)]],
    device const long* old_bboxes [[buffer(3)]],
    device const uint* old_flags [[buffer(4)]],
    device const ulong* old_hashes [[buffer(5)]],
    device const ulong* new_offsets [[buffer(6)]],
    device const int* new_layers [[buffer(7)]],
    device const int* new_datatypes [[buffer(8)]],
    device const long* new_bboxes [[buffer(9)]],
    device const uint* new_flags [[buffer(10)]],
    device const ulong* new_hashes [[buffer(11)]],
    device int* out_layers [[buffer(12)]],
    device int* out_datatypes [[buffer(13)]],
    device uchar* out_directions [[buffer(14)]],
    device ulong* out_counts [[buffer(15)]],
    device long* out_bboxes [[buffer(16)]],
    device ulong* out_vertex_counts [[buffer(17)]],
    device uint* out_flags [[buffer(18)]],
    device ulong* out_hashes [[buffer(19)]],
    device ulong* out_representative_indices [[buffer(20)]],
    device atomic_uint* record_counter [[buffer(21)]],
    device atomic_uint* matched_counter [[buffer(22)]],
    device atomic_uint* reduced_counter [[buffer(23)]],
    constant RecordParams& params [[buffer(24)]],
    uint gid [[thread_position_in_grid]]
) {
    ulong index = gid;
    if (index >= params.new_count) {
        return;
    }
    if (!first_in_side(new_offsets, new_layers, new_datatypes, new_bboxes, new_flags, new_hashes, index)) {
        return;
    }
    ulong new_total = count_same(new_offsets, new_layers, new_datatypes, new_bboxes, new_flags, new_hashes, index,
                                 new_offsets, new_layers, new_datatypes, new_bboxes, new_flags, new_hashes, params.new_count);
    ulong old_total = count_same(new_offsets, new_layers, new_datatypes, new_bboxes, new_flags, new_hashes, index,
                                 old_offsets, old_layers, old_datatypes, old_bboxes, old_flags, old_hashes, params.old_count);
    if (old_total == 0) {
        atomic_fetch_add_explicit(reduced_counter, 1u, memory_order_relaxed);
    }
    if (new_total > old_total) {
        emit_record_delta(2, new_total - old_total, index, new_offsets, new_layers, new_datatypes, new_bboxes, new_flags, new_hashes,
                          record_counter, out_layers, out_datatypes, out_directions, out_counts, out_bboxes,
                          out_vertex_counts, out_flags, out_hashes, out_representative_indices);
    }
}

struct KeyParams {
    ulong count;
    ulong output_offset;
    uint direction;
};

kernel void record_key_from_hash(
    device const ulong* offsets [[buffer(0)]],
    device const int* layers [[buffer(1)]],
    device const int* datatypes [[buffer(2)]],
    device const long* bboxes [[buffer(3)]],
    device const uint* flags [[buffer(4)]],
    device const ulong* hashes [[buffer(5)]],
    device int* out_layers [[buffer(6)]],
    device int* out_datatypes [[buffer(7)]],
    device uchar* out_directions [[buffer(8)]],
    device long* out_bboxes [[buffer(9)]],
    device ulong* out_vertex_counts [[buffer(10)]],
    device uint* out_flags [[buffer(11)]],
    device ulong* out_hashes [[buffer(12)]],
    device ulong* out_representative_indices [[buffer(13)]],
    constant KeyParams& params [[buffer(14)]],
    uint gid [[thread_position_in_grid]]
) {
    ulong index = gid;
    if (index >= params.count) {
        return;
    }
    ulong row = params.output_offset + index;
    out_layers[row] = layers[index];
    out_datatypes[row] = datatypes[index];
    out_directions[row] = static_cast<uchar>(params.direction);
    for (ushort offset = 0; offset < 4; ++offset) {
        out_bboxes[row * 4 + offset] = bboxes[index * 4 + offset];
    }
    out_vertex_counts[row] = vertex_count(offsets, index);
    out_flags[row] = flags[index];
    out_hashes[row * 2] = hashes[index * 2];
    out_hashes[row * 2 + 1] = hashes[index * 2 + 1];
    out_representative_indices[row] = index;
}

struct ContextParams {
    ulong component_count;
    ulong polygon_count;
    ulong total_pairs;
};

kernel void context_overlap_flags(
    device const long* component_bboxes [[buffer(0)]],
    device const int* component_layers [[buffer(1)]],
    device const int* component_datatypes [[buffer(2)]],
    device const long* polygon_bboxes [[buffer(3)]],
    device const int* polygon_layers [[buffer(4)]],
    device const int* polygon_datatypes [[buffer(5)]],
    device uchar* out_flags [[buffer(6)]],
    constant ContextParams& params [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    ulong linear = gid;
    if (linear >= params.total_pairs) {
        return;
    }
    ulong component_index = linear / params.polygon_count;
    ulong polygon_index = linear - component_index * params.polygon_count;
    long cx0 = component_bboxes[component_index * 4];
    long cy0 = component_bboxes[component_index * 4 + 1];
    long cx1 = component_bboxes[component_index * 4 + 2];
    long cy1 = component_bboxes[component_index * 4 + 3];
    long px0 = polygon_bboxes[polygon_index * 4];
    long py0 = polygon_bboxes[polygon_index * 4 + 1];
    long px1 = polygon_bboxes[polygon_index * 4 + 2];
    long py1 = polygon_bboxes[polygon_index * 4 + 3];
    bool same_layer = component_layers[component_index] == polygon_layers[polygon_index]
        && component_datatypes[component_index] == polygon_datatypes[polygon_index];
    bool overlaps = px1 > cx0 && cx1 > px0 && max(py0, cy0) < min(py1, cy1);
    out_flags[linear] = same_layer && overlaps ? 1 : 0;
}

struct BoundaryEdgeParams {
    ulong vertex_count;
    ulong polygon_count;
    ulong segment_count;
    uint direction;
};

static inline bool point_in_bbox_float(float x, float y, device const long* bbox) {
    return x >= static_cast<float>(bbox[0]) && x <= static_cast<float>(bbox[2])
        && y >= static_cast<float>(bbox[1]) && y <= static_cast<float>(bbox[3]);
}

static inline bool point_in_polygon_float(
    float x,
    float y,
    device const long* vertices_xy,
    device const ulong* polygon_offsets,
    ulong polygon_index
) {
    ulong start = polygon_offsets[polygon_index];
    ulong stop = polygon_offsets[polygon_index + 1];
    ulong vertex_count = stop - start;
    if (vertex_count < 3) {
        return false;
    }
    bool inside = false;
    for (ulong local = 0; local < vertex_count; ++local) {
        ulong a = start + local;
        ulong b = start + ((local + 1) % vertex_count);
        float x0 = static_cast<float>(vertices_xy[a * 2]);
        float y0 = static_cast<float>(vertices_xy[a * 2 + 1]);
        float x1 = static_cast<float>(vertices_xy[b * 2]);
        float y1 = static_cast<float>(vertices_xy[b * 2 + 1]);
        if ((y0 > y) != (y1 > y)) {
            float x_intersection = x0 + (y - y0) * (x1 - x0) / (y1 - y0);
            if (x_intersection > x) {
                inside = !inside;
            }
        }
    }
    return inside;
}

static inline bool union_contains_float(
    float x,
    float y,
    device const long* vertices_xy,
    device const ulong* polygon_offsets,
    device const long* bboxes,
    ulong polygon_count
) {
    for (ulong polygon_index = 0; polygon_index < polygon_count; ++polygon_index) {
        device const long* bbox = bboxes + polygon_index * 4;
        if (point_in_bbox_float(x, y, bbox)
            && point_in_polygon_float(x, y, vertices_xy, polygon_offsets, polygon_index)) {
            return true;
        }
    }
    return false;
}

static inline bool union_contains_side_float(
    float x,
    float y,
    uchar side,
    device const long* vertices_xy,
    device const ulong* polygon_offsets,
    device const long* bboxes,
    device const uchar* polygon_sides,
    ulong polygon_count
) {
    for (ulong polygon_index = 0; polygon_index < polygon_count; ++polygon_index) {
        if (polygon_sides[polygon_index] != side) {
            continue;
        }
        device const long* bbox = bboxes + polygon_index * 4;
        if (point_in_bbox_float(x, y, bbox)
            && point_in_polygon_float(x, y, vertices_xy, polygon_offsets, polygon_index)) {
            return true;
        }
    }
    return false;
}

static inline bool directional_target_contains_float(
    float x,
    float y,
    uchar direction,
    device const long* vertices_xy,
    device const ulong* polygon_offsets,
    device const long* bboxes,
    device const uchar* polygon_sides,
    ulong polygon_count
) {
    bool old_inside = union_contains_side_float(x, y, 1, vertices_xy, polygon_offsets, bboxes, polygon_sides, polygon_count);
    bool new_inside = union_contains_side_float(x, y, 2, vertices_xy, polygon_offsets, bboxes, polygon_sides, polygon_count);
    if (direction == 1) {
        return old_inside && !new_inside;
    }
    return new_inside && !old_inside;
}

static inline float boundary_sample_epsilon(float x, float y) {
    return max(1e-3f, max(fabs(x), fabs(y)) * 1e-6f);
}

kernel void one_sided_boundary_edges(
    device const long* vertices_xy [[buffer(0)]],
    device const ulong* polygon_offsets [[buffer(1)]],
    device const long* bboxes [[buffer(2)]],
    device const long* signed_twice_areas [[buffer(3)]],
    device const float* segment_points [[buffer(4)]],
    device const uint* segment_polygon_indices [[buffer(5)]],
    device float* out_edges [[buffer(6)]],
    device uchar* out_keep [[buffer(7)]],
    constant BoundaryEdgeParams& params [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    ulong index = gid;
    if (index >= params.segment_count) {
        return;
    }
    out_keep[index] = 0;
    ulong polygon_index = static_cast<ulong>(segment_polygon_indices[index]);
    if (polygon_index >= params.polygon_count) {
        return;
    }
    long signed_area = signed_twice_areas[polygon_index];
    if (signed_area == 0) {
        return;
    }
    float x0 = segment_points[index * 4];
    float y0 = segment_points[index * 4 + 1];
    float x1 = segment_points[index * 4 + 2];
    float y1 = segment_points[index * 4 + 3];
    float dx = x1 - x0;
    float dy = y1 - y0;
    float length = sqrt(dx * dx + dy * dy);
    if (length <= 1e-12f) {
        return;
    }
    float normal_x = -dy / length;
    float normal_y = dx / length;
    float mid_x = (x0 + x1) * 0.5f;
    float mid_y = (y0 + y1) * 0.5f;
    float epsilon = boundary_sample_epsilon(mid_x, mid_y);
    float sample_x = signed_area > 0 ? mid_x - normal_x * epsilon : mid_x + normal_x * epsilon;
    float sample_y = signed_area > 0 ? mid_y - normal_y * epsilon : mid_y + normal_y * epsilon;
    if (union_contains_float(sample_x, sample_y, vertices_xy, polygon_offsets, bboxes, params.polygon_count)) {
        return;
    }
    if (signed_area > 0) {
        out_edges[index * 4] = x0;
        out_edges[index * 4 + 1] = y0;
        out_edges[index * 4 + 2] = x1;
        out_edges[index * 4 + 3] = y1;
    } else {
        out_edges[index * 4] = x1;
        out_edges[index * 4 + 1] = y1;
        out_edges[index * 4 + 2] = x0;
        out_edges[index * 4 + 3] = y0;
    }
    out_keep[index] = 1;
}

kernel void mixed_boundary_edges(
    device const long* vertices_xy [[buffer(0)]],
    device const ulong* polygon_offsets [[buffer(1)]],
    device const long* bboxes [[buffer(2)]],
    device const uchar* polygon_sides [[buffer(3)]],
    device const float* segment_points [[buffer(4)]],
    device float* out_edges [[buffer(5)]],
    device uchar* out_keep [[buffer(6)]],
    constant BoundaryEdgeParams& params [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    ulong index = gid;
    if (index >= params.segment_count) {
        return;
    }
    out_keep[index] = 0;
    float x0 = segment_points[index * 4];
    float y0 = segment_points[index * 4 + 1];
    float x1 = segment_points[index * 4 + 2];
    float y1 = segment_points[index * 4 + 3];
    float dx = x1 - x0;
    float dy = y1 - y0;
    float length = sqrt(dx * dx + dy * dy);
    if (length <= 1e-12f) {
        return;
    }
    float normal_x = -dy / length;
    float normal_y = dx / length;
    float mid_x = (x0 + x1) * 0.5f;
    float mid_y = (y0 + y1) * 0.5f;
    float epsilon = boundary_sample_epsilon(mid_x, mid_y);
    bool left_inside = directional_target_contains_float(
        mid_x + normal_x * epsilon,
        mid_y + normal_y * epsilon,
        static_cast<uchar>(params.direction),
        vertices_xy,
        polygon_offsets,
        bboxes,
        polygon_sides,
        params.polygon_count);
    bool right_inside = directional_target_contains_float(
        mid_x - normal_x * epsilon,
        mid_y - normal_y * epsilon,
        static_cast<uchar>(params.direction),
        vertices_xy,
        polygon_offsets,
        bboxes,
        polygon_sides,
        params.polygon_count);
    if (left_inside && !right_inside) {
        out_edges[index * 4] = x0;
        out_edges[index * 4 + 1] = y0;
        out_edges[index * 4 + 2] = x1;
        out_edges[index * 4 + 3] = y1;
        out_keep[index] = 1;
    } else if (right_inside && !left_inside) {
        out_edges[index * 4] = x1;
        out_edges[index * 4 + 1] = y1;
        out_edges[index * 4 + 2] = x0;
        out_edges[index * 4 + 3] = y0;
        out_keep[index] = 1;
    }
}

struct DeviceLoopTraceStats {
    ulong fragment_count;
    ulong vertex_count;
    ulong dropped_zero_area_loop_count;
    uint error;
};

static inline bool loop_edge_less(device const long* edges, ulong left, ulong right) {
    for (ushort axis = 0; axis < 4; ++axis) {
        long left_value = edges[left * 4 + axis];
        long right_value = edges[right * 4 + axis];
        if (left_value != right_value) {
            return left_value < right_value;
        }
    }
    return false;
}

static inline bool loop_point_less_xy(long ax, long ay, long bx, long by) {
    return ax < bx || (ax == bx && ay < by);
}

static inline bool loop_edge_start_matches(device const long* edges, ulong edge_index, long x, long y) {
    return edges[edge_index * 4] == x && edges[edge_index * 4 + 1] == y;
}

static inline bool loop_collinear(long ax, long ay, long bx, long by, long cx, long cy) {
    long left = (bx - ax) * (cy - by);
    long right = (by - ay) * (cx - bx);
    return left == right;
}

static inline void loop_turn_key(
    long px,
    long py,
    long cx,
    long cy,
    long nx,
    long ny,
    thread int& turn_class,
    thread ulong& abs_cross,
    thread long& dot
) {
    long in_x = cx - px;
    long in_y = cy - py;
    long out_x = nx - cx;
    long out_y = ny - cy;
    long cross = in_x * out_y - in_y * out_x;
    dot = in_x * out_x + in_y * out_y;
    if (cross < 0) {
        turn_class = 0;
        abs_cross = static_cast<ulong>(-cross);
    } else if (cross == 0 && dot > 0) {
        turn_class = 1;
        abs_cross = 0;
    } else if (cross > 0) {
        turn_class = 2;
        abs_cross = static_cast<ulong>(cross);
    } else {
        turn_class = 3;
        abs_cross = 0;
    }
}

static inline bool loop_candidate_better(
    long px,
    long py,
    long cx,
    long cy,
    long candidate_x,
    long candidate_y,
    long best_x,
    long best_y
) {
    int candidate_class = 0;
    int best_class = 0;
    ulong candidate_abs_cross = 0;
    ulong best_abs_cross = 0;
    long candidate_dot = 0;
    long best_dot = 0;
    loop_turn_key(px, py, cx, cy, candidate_x, candidate_y, candidate_class, candidate_abs_cross, candidate_dot);
    loop_turn_key(px, py, cx, cy, best_x, best_y, best_class, best_abs_cross, best_dot);
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

static inline ulong loop_clean_duplicates(device const long* input_points, ulong input_count, device long* output_points) {
    ulong out_count = 0;
    for (ulong index = 0; index < input_count; ++index) {
        long x = input_points[index * 2];
        long y = input_points[index * 2 + 1];
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

static inline ulong loop_clean_collinear_once(
    device const long* input_points,
    ulong input_count,
    device long* output_points,
    thread bool& changed
) {
    changed = false;
    if (input_count < 3) {
        for (ulong index = 0; index < input_count; ++index) {
            output_points[index * 2] = input_points[index * 2];
            output_points[index * 2 + 1] = input_points[index * 2 + 1];
        }
        return input_count;
    }
    ulong out_count = 0;
    for (ulong index = 0; index < input_count; ++index) {
        ulong previous = (index + input_count - 1) % input_count;
        ulong following = (index + 1) % input_count;
        long ax = input_points[previous * 2];
        long ay = input_points[previous * 2 + 1];
        long bx = input_points[index * 2];
        long by = input_points[index * 2 + 1];
        long cx = input_points[following * 2];
        long cy = input_points[following * 2 + 1];
        if (loop_collinear(ax, ay, bx, by, cx, cy)) {
            changed = true;
            continue;
        }
        output_points[out_count * 2] = bx;
        output_points[out_count * 2 + 1] = by;
        ++out_count;
    }
    return out_count;
}

static inline long loop_twice_area(device const long* points, ulong count) {
    long area = 0;
    for (ulong index = 0; index < count; ++index) {
        ulong next = (index + 1) % count;
        area += points[index * 2] * points[next * 2 + 1] - points[next * 2] * points[index * 2 + 1];
    }
    return area;
}

kernel void boundary_loop_next(
    device const long* edges [[buffer(0)]],
    device ulong* next_indices [[buffer(1)]],
    constant ulong& edge_count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    ulong index = gid;
    if (index >= edge_count) {
        return;
    }
    long previous_x = edges[index * 4];
    long previous_y = edges[index * 4 + 1];
    long current_x = edges[index * 4 + 2];
    long current_y = edges[index * 4 + 3];
    bool have_best = false;
    ulong best_edge = ~static_cast<ulong>(0);
    long best_x = 0;
    long best_y = 0;
    for (ulong candidate = 0; candidate < edge_count; ++candidate) {
        if (!loop_edge_start_matches(edges, candidate, current_x, current_y)) {
            continue;
        }
        long candidate_x = edges[candidate * 4 + 2];
        long candidate_y = edges[candidate * 4 + 3];
        if (!have_best || loop_candidate_better(previous_x, previous_y, current_x, current_y, candidate_x, candidate_y, best_x, best_y)) {
            best_edge = candidate;
            best_x = candidate_x;
            best_y = candidate_y;
            have_best = true;
        }
    }
    next_indices[index] = have_best ? best_edge : ~static_cast<ulong>(0);
}

kernel void boundary_loop_trace(
    device const long* edges [[buffer(0)]],
    device const ulong* next_indices [[buffer(1)]],
    device uchar* remaining [[buffer(2)]],
    device long* scratch_a [[buffer(3)]],
    device long* scratch_b [[buffer(4)]],
    device long* out_vertices [[buffer(5)]],
    device ulong* out_offsets [[buffer(6)]],
    device long* out_bboxes [[buffer(7)]],
    device long* out_twice_areas [[buffer(8)]],
    device DeviceLoopTraceStats* stats [[buffer(9)]],
    constant ulong& edge_count [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) {
        return;
    }
    stats->fragment_count = 0;
    stats->vertex_count = 0;
    stats->dropped_zero_area_loop_count = 0;
    stats->error = 0;
    out_offsets[0] = 0;

    while (true) {
        bool found_seed = false;
        ulong seed = 0;
        for (ulong index = 0; index < edge_count; ++index) {
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
        long start_x = edges[seed * 4];
        long start_y = edges[seed * 4 + 1];
        long current_x = edges[seed * 4 + 2];
        long current_y = edges[seed * 4 + 3];
        ulong current_edge = seed;
        ulong raw_count = 0;
        scratch_a[raw_count * 2] = start_x;
        scratch_a[raw_count * 2 + 1] = start_y;
        ++raw_count;
        scratch_a[raw_count * 2] = current_x;
        scratch_a[raw_count * 2 + 1] = current_y;
        ++raw_count;

        while (!(current_x == start_x && current_y == start_y)) {
            ulong best_edge = next_indices[current_edge];
            if (best_edge == ~static_cast<ulong>(0) || !remaining[best_edge]) {
                break;
            }
            long best_x = edges[best_edge * 4 + 2];
            long best_y = edges[best_edge * 4 + 3];
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
            current_x = best_x;
            current_y = best_y;
            current_edge = best_edge;
        }

        ulong clean_count = loop_clean_duplicates(scratch_a, raw_count, scratch_b);
        bool changed = true;
        bool scratch_b_current = true;
        while (changed && clean_count >= 3) {
            changed = false;
            if (scratch_b_current) {
                clean_count = loop_clean_collinear_once(scratch_b, clean_count, scratch_a, changed);
                scratch_b_current = false;
            } else {
                clean_count = loop_clean_collinear_once(scratch_a, clean_count, scratch_b, changed);
                scratch_b_current = true;
            }
        }
        device const long* clean_points = scratch_b_current ? scratch_b : scratch_a;
        if (clean_count < 3) {
            ++stats->dropped_zero_area_loop_count;
            continue;
        }
        long area = loop_twice_area(clean_points, clean_count);
        if (area == 0) {
            ++stats->dropped_zero_area_loop_count;
            continue;
        }
        ulong fragment_index = stats->fragment_count;
        if (fragment_index >= edge_count || stats->vertex_count + clean_count > edge_count) {
            stats->error = 2;
            return;
        }
        long x0 = clean_points[0];
        long y0 = clean_points[1];
        long x1 = clean_points[0];
        long y1 = clean_points[1];
        for (ulong index = 0; index < clean_count; ++index) {
            long x = clean_points[index * 2];
            long y = clean_points[index * 2 + 1];
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
)METAL";

struct Params {
    std::uint64_t task_count;
    std::uint16_t max_rects;
    std::uint16_t max_fragments;
};

struct ByteCompareParamsHost {
    std::uint64_t byte_count;
};

struct TileCountParamsHost {
    std::uint64_t polygon_count;
    std::int64_t tile_size;
    std::uint64_t max_tiles_per_polygon;
    std::uint32_t side;
};

struct TileEmitParamsHost {
    std::uint64_t polygon_count;
    std::int64_t tile_size;
    std::uint32_t side;
};

struct RecordParamsHost {
    std::uint64_t old_count;
    std::uint64_t new_count;
};

struct KeyParamsHost {
    std::uint64_t count;
    std::uint64_t output_offset;
    std::uint32_t direction;
};

struct ContextParamsHost {
    std::uint64_t component_count;
    std::uint64_t polygon_count;
    std::uint64_t total_pairs;
};

struct BoundaryEdgeParamsHost {
    std::uint64_t vertex_count;
    std::uint64_t polygon_count;
    std::uint64_t segment_count;
    std::uint32_t direction;
};

struct DeviceLoopTraceStatsHost {
    std::uint64_t fragment_count;
    std::uint64_t vertex_count;
    std::uint64_t dropped_zero_area_loop_count;
    std::uint32_t error;
};

struct HostRecordDelta {
    std::int32_t layer;
    std::int32_t datatype;
    std::uint8_t direction;
    std::uint64_t count;
    std::int64_t bbox[4];
    std::uint64_t vertex_count;
    std::uint32_t flags;
    std::uint64_t hash[2];
    std::uint64_t representative_index;
};

struct HostTileAssignment {
    std::int32_t layer;
    std::int32_t datatype;
    std::int64_t tx;
    std::int64_t ty;
    std::uint64_t canonical_id;
    std::uint8_t side;
};

struct HostCandidateTile {
    std::int32_t layer;
    std::int32_t datatype;
    std::int64_t tx;
    std::int64_t ty;
};

struct HostOversizedTile {
    std::int32_t layer;
    std::int32_t datatype;
    std::uint64_t canonical_id;
    std::uint8_t side;
    std::int64_t tx0;
    std::int64_t ty0;
    std::int64_t tx1;
    std::int64_t ty1;
};

void set_error(const std::string& message) {
    g_last_error = message;
}

std::vector<std::uint8_t> read_binary_file(const char* path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error(std::string("could not open file: ") + path);
    }
    const auto size = stream.tellg();
    if (size < 0) {
        throw std::runtime_error(std::string("could not determine file size: ") + path);
    }
    std::vector<std::uint8_t> data(static_cast<std::size_t>(size));
    stream.seekg(0, std::ios::beg);
    if (!data.empty()) {
        stream.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
        if (!stream) {
            throw std::runtime_error(std::string("could not read file: ") + path);
        }
    }
    return data;
}

void ensure_library_locked() {
    if (g_device && g_queue && g_library) {
        return;
    }
    if (!g_device) {
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
            if ([devices count] > 0) {
                g_device = [devices objectAtIndex:0];
            }
        }
    }
    if (!g_device) {
        throw std::runtime_error("Metal device is not available");
    }
    if (!g_queue) {
        g_queue = [g_device newCommandQueue];
    }
    if (!g_queue) {
        throw std::runtime_error("Metal command queue could not be created");
    }
    if (g_library) {
        return;
    }
    NSError* error = nil;
    NSString* source = [NSString stringWithUTF8String:kMetalSource];
    g_library = [g_device newLibraryWithSource:source options:nil error:&error];
    if (!g_library) {
        NSString* message = error ? [error localizedDescription] : @"unknown Metal library compile failure";
        throw std::runtime_error([[message description] UTF8String]);
    }
}

id<MTLComputePipelineState> make_named_pipeline_locked(const char* function_name, const char* label) {
    ensure_library_locked();
    NSError* error = nil;
    NSString* name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> function = [g_library newFunctionWithName:name];
    if (!function) {
        throw std::runtime_error(std::string("Metal function ") + function_name + " is unavailable");
    }
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) {
        NSString* message = error ? [error localizedDescription] : @"unknown Metal pipeline creation failure";
        throw std::runtime_error(std::string("unknown Metal ") + label + " pipeline creation failure: " + [[message description] UTF8String]);
    }
    return pipeline;
}

void ensure_pipeline_locked() {
    if (g_byte_compare_pipeline && g_rect_set_pipeline && g_record_delta_pipeline_old && g_record_delta_pipeline_new && g_record_key_from_hash_pipeline && g_context_overlap_pipeline && g_one_sided_boundary_pipeline && g_mixed_boundary_pipeline && g_boundary_loop_next_pipeline && g_boundary_loop_trace_pipeline) {
        return;
    }
    if (!g_byte_compare_pipeline) g_byte_compare_pipeline = make_named_pipeline_locked("byte_compare", "byte-compare");
    if (!g_rect_set_pipeline) g_rect_set_pipeline = make_named_pipeline_locked("rect_set_exact", "rectangle exact");
    if (!g_record_delta_pipeline_old) g_record_delta_pipeline_old = make_named_pipeline_locked("record_delta_old", "old record-delta");
    if (!g_record_delta_pipeline_new) g_record_delta_pipeline_new = make_named_pipeline_locked("record_delta_new", "new record-delta");
    if (!g_record_key_from_hash_pipeline) g_record_key_from_hash_pipeline = make_named_pipeline_locked("record_key_from_hash", "record-key");
    if (!g_context_overlap_pipeline) g_context_overlap_pipeline = make_named_pipeline_locked("context_overlap_flags", "context-overlap");
    if (!g_one_sided_boundary_pipeline) g_one_sided_boundary_pipeline = make_named_pipeline_locked("one_sided_boundary_edges", "one-sided boundary");
    if (!g_mixed_boundary_pipeline) g_mixed_boundary_pipeline = make_named_pipeline_locked("mixed_boundary_edges", "mixed boundary");
    if (!g_boundary_loop_next_pipeline) g_boundary_loop_next_pipeline = make_named_pipeline_locked("boundary_loop_next", "boundary loop next");
    if (!g_boundary_loop_trace_pipeline) g_boundary_loop_trace_pipeline = make_named_pipeline_locked("boundary_loop_trace", "boundary loop trace");
}

void ensure_tile_pipelines_locked() {
    if (!g_tile_count_pipeline) g_tile_count_pipeline = make_named_pipeline_locked("tile_count", "tile-count");
    if (!g_tile_emit_pipeline) g_tile_emit_pipeline = make_named_pipeline_locked("tile_emit", "tile-emit");
}

id<MTLBuffer> make_input_buffer(const void* data, std::size_t byte_count) {
    static const std::uint8_t zero = 0;
    if (byte_count == 0) {
        byte_count = 1;
        data = &zero;
    }
    id<MTLBuffer> buffer = [g_device newBufferWithBytes:data length:byte_count options:MTLResourceStorageModeShared];
    if (!buffer) {
        throw std::runtime_error("Metal input buffer allocation failed");
    }
    return buffer;
}

id<MTLBuffer> make_output_buffer(std::size_t byte_count) {
    if (byte_count == 0) {
        byte_count = 1;
    }
    id<MTLBuffer> buffer = [g_device newBufferWithLength:byte_count options:MTLResourceStorageModeShared];
    if (!buffer) {
        throw std::runtime_error("Metal output buffer allocation failed");
    }
    std::memset([buffer contents], 0, byte_count);
    return buffer;
}

struct TileSideRun {
    std::vector<HostTileAssignment> assignments;
    std::vector<HostOversizedTile> oversized;
    std::uint64_t assignment_count = 0;
    std::uint64_t kernel_count = 0;
};

TileSideRun run_tile_side(
    const std::int64_t* bboxes,
    const std::int32_t* layers,
    const std::int32_t* datatypes,
    const std::uint64_t* canonical_ids,
    std::uint64_t count,
    std::int64_t tile_size,
    std::uint64_t max_tiles_per_polygon,
    std::uint8_t side
) {
    TileSideRun run;
    if (count == 0) {
        return run;
    }
    ensure_tile_pipelines_locked();
    const std::size_t count_size = static_cast<std::size_t>(count);
    id<MTLBuffer> bboxes_buffer = make_input_buffer(bboxes, count_size * 4 * sizeof(std::int64_t));
    id<MTLBuffer> layers_buffer = make_input_buffer(layers, count_size * sizeof(std::int32_t));
    id<MTLBuffer> datatypes_buffer = make_input_buffer(datatypes, count_size * sizeof(std::int32_t));
    id<MTLBuffer> canonical_ids_buffer = make_input_buffer(canonical_ids, count_size * sizeof(std::uint64_t));
    id<MTLBuffer> counts_buffer = make_output_buffer(count_size * sizeof(std::uint64_t));
    id<MTLBuffer> oversized_flags_buffer = make_output_buffer(count_size * sizeof(std::uint8_t));
    id<MTLBuffer> oversized_layers_buffer = make_output_buffer(count_size * sizeof(std::int32_t));
    id<MTLBuffer> oversized_datatypes_buffer = make_output_buffer(count_size * sizeof(std::int32_t));
    id<MTLBuffer> oversized_canonical_ids_buffer = make_output_buffer(count_size * sizeof(std::uint64_t));
    id<MTLBuffer> oversized_tile_bboxes_buffer = make_output_buffer(count_size * 4 * sizeof(std::int64_t));
    id<MTLBuffer> errors_buffer = make_output_buffer(count_size * sizeof(std::uint32_t));
    TileCountParamsHost count_params{count, tile_size, max_tiles_per_polygon, side};
    id<MTLBuffer> count_params_buffer = make_input_buffer(&count_params, sizeof(count_params));

    id<MTLCommandBuffer> count_command = [g_queue commandBuffer];
    if (!count_command) {
        throw std::runtime_error("Metal tile count command buffer allocation failed");
    }
    id<MTLComputeCommandEncoder> count_encoder = [count_command computeCommandEncoder];
    if (!count_encoder) {
        throw std::runtime_error("Metal tile count encoder allocation failed");
    }
    [count_encoder setComputePipelineState:g_tile_count_pipeline];
    [count_encoder setBuffer:bboxes_buffer offset:0 atIndex:0];
    [count_encoder setBuffer:layers_buffer offset:0 atIndex:1];
    [count_encoder setBuffer:datatypes_buffer offset:0 atIndex:2];
    [count_encoder setBuffer:canonical_ids_buffer offset:0 atIndex:3];
    [count_encoder setBuffer:counts_buffer offset:0 atIndex:4];
    [count_encoder setBuffer:oversized_flags_buffer offset:0 atIndex:5];
    [count_encoder setBuffer:oversized_layers_buffer offset:0 atIndex:6];
    [count_encoder setBuffer:oversized_datatypes_buffer offset:0 atIndex:7];
    [count_encoder setBuffer:oversized_canonical_ids_buffer offset:0 atIndex:8];
    [count_encoder setBuffer:oversized_tile_bboxes_buffer offset:0 atIndex:9];
    [count_encoder setBuffer:errors_buffer offset:0 atIndex:10];
    [count_encoder setBuffer:count_params_buffer offset:0 atIndex:11];
    NSUInteger count_group_width = g_tile_count_pipeline.maxTotalThreadsPerThreadgroup;
    if (count_group_width > 256) {
        count_group_width = 256;
    }
    if (count_group_width == 0) {
        count_group_width = 1;
    }
    [count_encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(count), 1, 1) threadsPerThreadgroup:MTLSizeMake(count_group_width, 1, 1)];
    [count_encoder endEncoding];
    [count_command commit];
    [count_command waitUntilCompleted];
    if (count_command.status == MTLCommandBufferStatusError) {
        NSString* message = count_command.error ? [count_command.error localizedDescription] : @"Metal tile count command failed";
        throw std::runtime_error([[message description] UTF8String]);
    }
    run.kernel_count += 1;

    const auto* errors = reinterpret_cast<const std::uint32_t*>([errors_buffer contents]);
    const auto* counts = reinterpret_cast<const std::uint64_t*>([counts_buffer contents]);
    const auto* oversized_flags = reinterpret_cast<const std::uint8_t*>([oversized_flags_buffer contents]);
    const auto* oversized_layers = reinterpret_cast<const std::int32_t*>([oversized_layers_buffer contents]);
    const auto* oversized_datatypes = reinterpret_cast<const std::int32_t*>([oversized_datatypes_buffer contents]);
    const auto* oversized_canonical_ids = reinterpret_cast<const std::uint64_t*>([oversized_canonical_ids_buffer contents]);
    const auto* oversized_tile_bboxes = reinterpret_cast<const std::int64_t*>([oversized_tile_bboxes_buffer contents]);
    std::vector<std::uint64_t> offsets(count_size);
    std::uint64_t total = 0;
    for (std::size_t index = 0; index < count_size; ++index) {
        if (errors[index]) {
            throw std::overflow_error("Metal assignment count overflow or invalid span");
        }
        offsets[index] = total;
        if (counts[index] > std::numeric_limits<std::uint64_t>::max() - total) {
            throw std::overflow_error("Metal total assignment count overflow");
        }
        total += counts[index];
        if (oversized_flags[index]) {
            run.oversized.push_back(HostOversizedTile{
                oversized_layers[index],
                oversized_datatypes[index],
                oversized_canonical_ids[index],
                side,
                oversized_tile_bboxes[index * 4],
                oversized_tile_bboxes[index * 4 + 1],
                oversized_tile_bboxes[index * 4 + 2],
                oversized_tile_bboxes[index * 4 + 3],
            });
        }
    }
    run.assignment_count = total;
    if (total == 0) {
        return run;
    }
    if (total > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error("Metal total assignment count exceeds host addressable size");
    }
    id<MTLBuffer> offsets_buffer = make_input_buffer(offsets.data(), offsets.size() * sizeof(std::uint64_t));
    const std::size_t total_size = static_cast<std::size_t>(total);
    id<MTLBuffer> assignment_layers_buffer = make_output_buffer(total_size * sizeof(std::int32_t));
    id<MTLBuffer> assignment_datatypes_buffer = make_output_buffer(total_size * sizeof(std::int32_t));
    id<MTLBuffer> assignment_tx_buffer = make_output_buffer(total_size * sizeof(std::int64_t));
    id<MTLBuffer> assignment_ty_buffer = make_output_buffer(total_size * sizeof(std::int64_t));
    id<MTLBuffer> assignment_canonical_ids_buffer = make_output_buffer(total_size * sizeof(std::uint64_t));
    id<MTLBuffer> assignment_sides_buffer = make_output_buffer(total_size * sizeof(std::uint8_t));
    TileEmitParamsHost emit_params{count, tile_size, side};
    id<MTLBuffer> emit_params_buffer = make_input_buffer(&emit_params, sizeof(emit_params));

    id<MTLCommandBuffer> emit_command = [g_queue commandBuffer];
    if (!emit_command) {
        throw std::runtime_error("Metal tile emit command buffer allocation failed");
    }
    id<MTLComputeCommandEncoder> emit_encoder = [emit_command computeCommandEncoder];
    if (!emit_encoder) {
        throw std::runtime_error("Metal tile emit encoder allocation failed");
    }
    [emit_encoder setComputePipelineState:g_tile_emit_pipeline];
    [emit_encoder setBuffer:bboxes_buffer offset:0 atIndex:0];
    [emit_encoder setBuffer:layers_buffer offset:0 atIndex:1];
    [emit_encoder setBuffer:datatypes_buffer offset:0 atIndex:2];
    [emit_encoder setBuffer:canonical_ids_buffer offset:0 atIndex:3];
    [emit_encoder setBuffer:offsets_buffer offset:0 atIndex:4];
    [emit_encoder setBuffer:oversized_flags_buffer offset:0 atIndex:5];
    [emit_encoder setBuffer:assignment_layers_buffer offset:0 atIndex:6];
    [emit_encoder setBuffer:assignment_datatypes_buffer offset:0 atIndex:7];
    [emit_encoder setBuffer:assignment_tx_buffer offset:0 atIndex:8];
    [emit_encoder setBuffer:assignment_ty_buffer offset:0 atIndex:9];
    [emit_encoder setBuffer:assignment_canonical_ids_buffer offset:0 atIndex:10];
    [emit_encoder setBuffer:assignment_sides_buffer offset:0 atIndex:11];
    [emit_encoder setBuffer:emit_params_buffer offset:0 atIndex:12];
    NSUInteger emit_group_width = g_tile_emit_pipeline.maxTotalThreadsPerThreadgroup;
    if (emit_group_width > 256) {
        emit_group_width = 256;
    }
    if (emit_group_width == 0) {
        emit_group_width = 1;
    }
    [emit_encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(count), 1, 1) threadsPerThreadgroup:MTLSizeMake(emit_group_width, 1, 1)];
    [emit_encoder endEncoding];
    [emit_command commit];
    [emit_command waitUntilCompleted];
    if (emit_command.status == MTLCommandBufferStatusError) {
        NSString* message = emit_command.error ? [emit_command.error localizedDescription] : @"Metal tile emit command failed";
        throw std::runtime_error([[message description] UTF8String]);
    }
    run.kernel_count += 1;

    const auto* assignment_layers = reinterpret_cast<const std::int32_t*>([assignment_layers_buffer contents]);
    const auto* assignment_datatypes = reinterpret_cast<const std::int32_t*>([assignment_datatypes_buffer contents]);
    const auto* assignment_tx = reinterpret_cast<const std::int64_t*>([assignment_tx_buffer contents]);
    const auto* assignment_ty = reinterpret_cast<const std::int64_t*>([assignment_ty_buffer contents]);
    const auto* assignment_canonical_ids = reinterpret_cast<const std::uint64_t*>([assignment_canonical_ids_buffer contents]);
    const auto* assignment_sides = reinterpret_cast<const std::uint8_t*>([assignment_sides_buffer contents]);
    run.assignments.reserve(total_size);
    for (std::size_t index = 0; index < total_size; ++index) {
        run.assignments.push_back(HostTileAssignment{
            assignment_layers[index],
            assignment_datatypes[index],
            assignment_tx[index],
            assignment_ty[index],
            assignment_canonical_ids[index],
            assignment_sides[index],
        });
    }
    return run;
}

}  // namespace

extern "C" {

struct GdsMetalPolygonRecordDeltaResult {
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
    std::uint64_t metal_kernel_count;
    float metal_event_elapsed_ms;
};

struct GdsMetalByteCompareResult {
    std::uint64_t old_size;
    std::uint64_t new_size;
    std::uint8_t same;
};

struct GdsMetalBoundaryLoopResult {
    std::uint64_t fragment_count;
    std::uint64_t vertex_count;
    std::int64_t* vertices_xy;
    std::uint64_t* fragment_offsets;
    std::int64_t* bboxes;
    std::int64_t* twice_areas;
    std::uint64_t dropped_zero_area_loop_count;
    std::uint64_t metal_kernel_count;
    float metal_event_elapsed_ms;
};

struct GdsMetalPrefilterResult {
    std::int64_t* candidate_tiles;
    std::uint64_t candidate_count;
    std::int64_t* oversized;
    std::uint64_t oversized_count;
    std::uint64_t assignment_count;
    std::uint64_t collapsed_pair_count;
    std::uint64_t oversized_matched_count;
    std::uint64_t metal_kernel_count;
    float metal_event_elapsed_ms;
};

struct GdsMetalContextOverlapPairsResult {
    std::uint64_t pair_count;
    std::uint64_t* component_indices;
    std::uint64_t* polygon_indices;
    std::uint64_t metal_kernel_count;
    float metal_event_elapsed_ms;
};

const char* gds_metal_last_error() {
    return g_last_error.c_str();
}

void gds_metal_free_polygon_record_delta_result(GdsMetalPolygonRecordDeltaResult* result) {
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
    *result = GdsMetalPolygonRecordDeltaResult{};
}

void gds_metal_free_prefilter_result(GdsMetalPrefilterResult* result) {
    if (!result) {
        return;
    }
    std::free(result->candidate_tiles);
    std::free(result->oversized);
    *result = GdsMetalPrefilterResult{};
}

void gds_metal_free_boundary_loop_result(GdsMetalBoundaryLoopResult* result) {
    if (!result) {
        return;
    }
    std::free(result->vertices_xy);
    std::free(result->fragment_offsets);
    std::free(result->bboxes);
    std::free(result->twice_areas);
    *result = GdsMetalBoundaryLoopResult{};
}

void gds_metal_free_context_overlap_pairs_result(GdsMetalContextOverlapPairsResult* result) {
    if (!result) {
        return;
    }
    std::free(result->component_indices);
    std::free(result->polygon_indices);
    *result = GdsMetalContextOverlapPairsResult{};
}

int gds_metal_available() {
    @autoreleasepool {
        try {
            std::lock_guard<std::mutex> lock(g_mutex);
            ensure_pipeline_locked();
            return 1;
        } catch (const std::exception& exc) {
            set_error(exc.what());
            return 0;
        } catch (...) {
            set_error("unknown Metal availability probe failure");
            return 0;
        }
    }
}

int gds_metal_release_device_memory() {
    @autoreleasepool {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_byte_compare_pipeline = nil;
        g_tile_count_pipeline = nil;
        g_tile_emit_pipeline = nil;
        g_rect_set_pipeline = nil;
        g_record_delta_pipeline_old = nil;
        g_record_delta_pipeline_new = nil;
        g_record_key_from_hash_pipeline = nil;
        g_context_overlap_pipeline = nil;
        g_one_sided_boundary_pipeline = nil;
        g_mixed_boundary_pipeline = nil;
        g_boundary_loop_next_pipeline = nil;
        g_boundary_loop_trace_pipeline = nil;
        g_library = nil;
        g_queue = nil;
        g_device = nil;
        return 0;
    }
}

int gds_metal_compare_files(const char* old_path, const char* new_path, GdsMetalByteCompareResult* result) {
    @autoreleasepool {
        try {
            if (!old_path || !new_path || !result) {
                throw std::runtime_error("file compare received null pointer");
            }
            *result = GdsMetalByteCompareResult{};
            auto old_data = read_binary_file(old_path);
            auto new_data = read_binary_file(new_path);
            result->old_size = old_data.size();
            result->new_size = new_data.size();
            if (old_data.size() != new_data.size()) {
                result->same = 0;
                return 0;
            }
            if (old_data.empty()) {
                result->same = 1;
                return 0;
            }
            std::lock_guard<std::mutex> lock(g_mutex);
            ensure_pipeline_locked();
            id<MTLBuffer> old_buffer = make_input_buffer(old_data.data(), old_data.size());
            id<MTLBuffer> new_buffer = make_input_buffer(new_data.data(), new_data.size());
            std::uint32_t zero = 0;
            id<MTLBuffer> mismatch_buffer = make_input_buffer(&zero, sizeof(zero));
            ByteCompareParamsHost params{static_cast<std::uint64_t>(old_data.size())};
            id<MTLBuffer> params_buffer = make_input_buffer(&params, sizeof(params));

            id<MTLCommandBuffer> command_buffer = [g_queue commandBuffer];
            if (!command_buffer) {
                throw std::runtime_error("Metal command buffer allocation failed");
            }
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            if (!encoder) {
                throw std::runtime_error("Metal compute encoder allocation failed");
            }
            [encoder setComputePipelineState:g_byte_compare_pipeline];
            [encoder setBuffer:old_buffer offset:0 atIndex:0];
            [encoder setBuffer:new_buffer offset:0 atIndex:1];
            [encoder setBuffer:mismatch_buffer offset:0 atIndex:2];
            [encoder setBuffer:params_buffer offset:0 atIndex:3];
            NSUInteger group_width = g_byte_compare_pipeline.maxTotalThreadsPerThreadgroup;
            if (group_width > 256) {
                group_width = 256;
            }
            if (group_width == 0) {
                group_width = 1;
            }
            [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(old_data.size()), 1, 1) threadsPerThreadgroup:MTLSizeMake(group_width, 1, 1)];
            [encoder endEncoding];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
            if (command_buffer.status == MTLCommandBufferStatusError) {
                NSString* message = command_buffer.error ? [command_buffer.error localizedDescription] : @"Metal command failed";
                throw std::runtime_error([[message description] UTF8String]);
            }
            const std::uint32_t mismatch = *reinterpret_cast<std::uint32_t*>([mismatch_buffer contents]);
            result->same = mismatch ? 0 : 1;
            return 0;
        } catch (const std::exception& exc) {
            set_error(exc.what());
            return 1;
        } catch (...) {
            set_error("unknown Metal byte compare failure");
            return 1;
        }
    }
}

int gds_metal_prefilter(
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
    GdsMetalPrefilterResult* result
) {
    @autoreleasepool {
        try {
            if (!result) {
                throw std::runtime_error("prefilter result pointer is null");
            }
            *result = GdsMetalPrefilterResult{};
            if (tile_size <= 0 || max_tiles_per_polygon == 0) {
                throw std::runtime_error("invalid tile config");
            }
            if (old_count && (!old_bboxes || !old_layers || !old_datatypes || !old_ids)) {
                throw std::runtime_error("old tile prefilter input contains null pointer");
            }
            if (new_count && (!new_bboxes || !new_layers || !new_datatypes || !new_ids)) {
                throw std::runtime_error("new tile prefilter input contains null pointer");
            }

            TileSideRun old_run;
            TileSideRun new_run;
            {
                std::lock_guard<std::mutex> lock(g_mutex);
                ensure_pipeline_locked();
                old_run = run_tile_side(old_bboxes, old_layers, old_datatypes, old_ids, old_count, tile_size, max_tiles_per_polygon, 1);
                new_run = run_tile_side(new_bboxes, new_layers, new_datatypes, new_ids, new_count, tile_size, max_tiles_per_polygon, 2);
            }
            result->assignment_count = old_run.assignment_count + new_run.assignment_count;
            result->metal_kernel_count = old_run.kernel_count + new_run.kernel_count;
            result->metal_event_elapsed_ms = 0.0f;

            std::vector<HostTileAssignment> assignments;
            assignments.reserve(old_run.assignments.size() + new_run.assignments.size());
            assignments.insert(assignments.end(), old_run.assignments.begin(), old_run.assignments.end());
            assignments.insert(assignments.end(), new_run.assignments.begin(), new_run.assignments.end());
            std::sort(assignments.begin(), assignments.end(), [](const HostTileAssignment& left, const HostTileAssignment& right) {
                return std::make_tuple(left.layer, left.datatype, left.tx, left.ty, left.canonical_id)
                    < std::make_tuple(right.layer, right.datatype, right.tx, right.ty, right.canonical_id);
            });
            std::vector<HostCandidateTile> candidate_tiles;
            std::size_t index = 0;
            while (index < assignments.size()) {
                const auto& key = assignments[index];
                std::uint8_t side = 0;
                std::size_t next = index;
                while (
                    next < assignments.size()
                    && assignments[next].layer == key.layer
                    && assignments[next].datatype == key.datatype
                    && assignments[next].tx == key.tx
                    && assignments[next].ty == key.ty
                    && assignments[next].canonical_id == key.canonical_id
                ) {
                    side = static_cast<std::uint8_t>(side | assignments[next].side);
                    next += 1;
                }
                result->collapsed_pair_count += 1;
                if (side != 3) {
                    candidate_tiles.push_back(HostCandidateTile{key.layer, key.datatype, key.tx, key.ty});
                }
                index = next;
            }
            std::sort(candidate_tiles.begin(), candidate_tiles.end(), [](const HostCandidateTile& left, const HostCandidateTile& right) {
                return std::make_tuple(left.layer, left.datatype, left.tx, left.ty)
                    < std::make_tuple(right.layer, right.datatype, right.tx, right.ty);
            });
            candidate_tiles.erase(
                std::unique(
                    candidate_tiles.begin(),
                    candidate_tiles.end(),
                    [](const HostCandidateTile& left, const HostCandidateTile& right) {
                        return left.layer == right.layer && left.datatype == right.datatype && left.tx == right.tx && left.ty == right.ty;
                    }
                ),
                candidate_tiles.end()
            );
            result->candidate_count = candidate_tiles.size();
            if (!candidate_tiles.empty()) {
                result->candidate_tiles = static_cast<std::int64_t*>(std::malloc(sizeof(std::int64_t) * candidate_tiles.size() * 4));
                if (!result->candidate_tiles) {
                    throw std::bad_alloc();
                }
                for (std::size_t row = 0; row < candidate_tiles.size(); ++row) {
                    result->candidate_tiles[row * 4] = candidate_tiles[row].layer;
                    result->candidate_tiles[row * 4 + 1] = candidate_tiles[row].datatype;
                    result->candidate_tiles[row * 4 + 2] = candidate_tiles[row].tx;
                    result->candidate_tiles[row * 4 + 3] = candidate_tiles[row].ty;
                }
            }

            std::vector<HostOversizedTile> oversized;
            oversized.reserve(old_run.oversized.size() + new_run.oversized.size());
            oversized.insert(oversized.end(), old_run.oversized.begin(), old_run.oversized.end());
            oversized.insert(oversized.end(), new_run.oversized.begin(), new_run.oversized.end());
            std::sort(oversized.begin(), oversized.end(), [](const HostOversizedTile& left, const HostOversizedTile& right) {
                return std::make_tuple(left.layer, left.datatype, left.canonical_id, left.tx0, left.ty0, left.tx1, left.ty1)
                    < std::make_tuple(right.layer, right.datatype, right.canonical_id, right.tx0, right.ty0, right.tx1, right.ty1);
            });
            std::vector<HostOversizedTile> oversized_out;
            index = 0;
            while (index < oversized.size()) {
                HostOversizedTile item = oversized[index];
                std::uint8_t side = 0;
                std::size_t next = index;
                while (
                    next < oversized.size()
                    && oversized[next].layer == item.layer
                    && oversized[next].datatype == item.datatype
                    && oversized[next].canonical_id == item.canonical_id
                    && oversized[next].tx0 == item.tx0
                    && oversized[next].ty0 == item.ty0
                    && oversized[next].tx1 == item.tx1
                    && oversized[next].ty1 == item.ty1
                ) {
                    side = static_cast<std::uint8_t>(side | oversized[next].side);
                    next += 1;
                }
                if (side == 3) {
                    result->oversized_matched_count += 1;
                } else {
                    item.side = side;
                    oversized_out.push_back(item);
                }
                index = next;
            }
            result->oversized_count = oversized_out.size();
            if (!oversized_out.empty()) {
                result->oversized = static_cast<std::int64_t*>(std::malloc(sizeof(std::int64_t) * oversized_out.size() * 8));
                if (!result->oversized) {
                    throw std::bad_alloc();
                }
                for (std::size_t row = 0; row < oversized_out.size(); ++row) {
                    result->oversized[row * 8] = oversized_out[row].layer;
                    result->oversized[row * 8 + 1] = oversized_out[row].datatype;
                    result->oversized[row * 8 + 2] = static_cast<std::int64_t>(oversized_out[row].canonical_id);
                    result->oversized[row * 8 + 3] = oversized_out[row].side;
                    result->oversized[row * 8 + 4] = oversized_out[row].tx0;
                    result->oversized[row * 8 + 5] = oversized_out[row].ty0;
                    result->oversized[row * 8 + 6] = oversized_out[row].tx1;
                    result->oversized[row * 8 + 7] = oversized_out[row].ty1;
                }
            }
            return 0;
        } catch (const std::exception& exc) {
            set_error(exc.what());
            if (result) {
                gds_metal_free_prefilter_result(result);
            }
            return 1;
        } catch (...) {
            set_error("unknown Metal tile prefilter failure");
            if (result) {
                gds_metal_free_prefilter_result(result);
            }
            return 1;
        }
    }
}

int gds_metal_rect_set_exact_fragments(
    const std::int64_t* old_rects,
    const std::int64_t* new_rects,
    const std::uint8_t* old_counts,
    const std::uint8_t* new_counts,
    std::uint64_t task_count,
    std::uint16_t max_rects,
    std::uint16_t max_fragments,
    std::int64_t* out_rects,
    std::uint16_t* out_counts,
    std::int64_t* out_areas
) {
    @autoreleasepool {
        try {
            if (!old_rects || !new_rects || !old_counts || !new_counts || !out_rects || !out_counts || !out_areas) {
                throw std::runtime_error("invalid null argument");
            }
            if (max_rects == 0 || max_rects > 16) {
                throw std::runtime_error("max_rects must be between 1 and 16");
            }
            if (max_fragments == 0 || max_fragments > 4096) {
                throw std::runtime_error("max_fragments must be between 1 and 4096");
            }
            if (task_count == 0) {
                return 0;
            }
            std::lock_guard<std::mutex> lock(g_mutex);
            ensure_pipeline_locked();
            const std::size_t rect_input_bytes = static_cast<std::size_t>(task_count) * max_rects * 4 * sizeof(std::int64_t);
            const std::size_t counts_input_bytes = static_cast<std::size_t>(task_count) * sizeof(std::uint8_t);
            const std::size_t out_rect_bytes = static_cast<std::size_t>(task_count) * 2 * max_fragments * 4 * sizeof(std::int64_t);
            const std::size_t out_count_bytes = static_cast<std::size_t>(task_count) * 2 * sizeof(std::uint16_t);
            const std::size_t out_area_bytes = static_cast<std::size_t>(task_count) * 3 * sizeof(std::int64_t);
            id<MTLBuffer> old_buffer = make_input_buffer(old_rects, rect_input_bytes);
            id<MTLBuffer> new_buffer = make_input_buffer(new_rects, rect_input_bytes);
            id<MTLBuffer> old_count_buffer = make_input_buffer(old_counts, counts_input_bytes);
            id<MTLBuffer> new_count_buffer = make_input_buffer(new_counts, counts_input_bytes);
            id<MTLBuffer> rect_output = make_output_buffer(out_rect_bytes);
            id<MTLBuffer> count_output = make_output_buffer(out_count_bytes);
            id<MTLBuffer> area_output = make_output_buffer(out_area_bytes);
            std::uint32_t zero = 0;
            id<MTLBuffer> error_buffer = make_input_buffer(&zero, sizeof(zero));
            Params params{task_count, max_rects, max_fragments};
            id<MTLBuffer> params_buffer = make_input_buffer(&params, sizeof(params));

            id<MTLCommandBuffer> command_buffer = [g_queue commandBuffer];
            if (!command_buffer) {
                throw std::runtime_error("Metal command buffer allocation failed");
            }
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            if (!encoder) {
                throw std::runtime_error("Metal compute encoder allocation failed");
            }
            [encoder setComputePipelineState:g_rect_set_pipeline];
            [encoder setBuffer:old_buffer offset:0 atIndex:0];
            [encoder setBuffer:new_buffer offset:0 atIndex:1];
            [encoder setBuffer:old_count_buffer offset:0 atIndex:2];
            [encoder setBuffer:new_count_buffer offset:0 atIndex:3];
            [encoder setBuffer:rect_output offset:0 atIndex:4];
            [encoder setBuffer:count_output offset:0 atIndex:5];
            [encoder setBuffer:area_output offset:0 atIndex:6];
            [encoder setBuffer:error_buffer offset:0 atIndex:7];
            [encoder setBuffer:params_buffer offset:0 atIndex:8];
            const NSUInteger width = static_cast<NSUInteger>(task_count);
            NSUInteger group_width = g_rect_set_pipeline.maxTotalThreadsPerThreadgroup;
            if (group_width > 256) {
                group_width = 256;
            }
            if (group_width == 0) {
                group_width = 1;
            }
            MTLSize grid = MTLSizeMake(width, 1, 1);
            MTLSize group = MTLSizeMake(group_width, 1, 1);
            [encoder dispatchThreads:grid threadsPerThreadgroup:group];
            [encoder endEncoding];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
            if (command_buffer.status == MTLCommandBufferStatusError) {
                NSString* message = command_buffer.error ? [command_buffer.error localizedDescription] : @"Metal command failed";
                throw std::runtime_error([[message description] UTF8String]);
            }
            std::uint32_t error_flag = *reinterpret_cast<std::uint32_t*>([error_buffer contents]);
            if (error_flag != 0) {
                throw std::runtime_error("Metal rectangle set output exceeded max_fragments");
            }
            std::memcpy(out_rects, [rect_output contents], out_rect_bytes);
            std::memcpy(out_counts, [count_output contents], out_count_bytes);
            std::memcpy(out_areas, [area_output contents], out_area_bytes);
            return 0;
        } catch (const std::exception& exc) {
            set_error(exc.what());
            return 1;
        } catch (...) {
            set_error("unknown Metal rectangle set exact failure");
            return 1;
        }
    }
}

int gds_metal_polygon_record_delta_from_hashes(
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
    GdsMetalPolygonRecordDeltaResult* result
) {
    @autoreleasepool {
        try {
            if (!result || !old_offsets || !new_offsets || (!old_layers && old_count) || (!old_datatypes && old_count)
                || (!old_bboxes && old_count) || (!old_flags && old_count) || (!old_hashes && old_count)
                || (!new_layers && new_count) || (!new_datatypes && new_count) || (!new_bboxes && new_count)
                || (!new_flags && new_count) || (!new_hashes && new_count)) {
                throw std::runtime_error("invalid null argument");
            }
            *result = GdsMetalPolygonRecordDeltaResult{};
            const std::uint64_t max_records_u64 = old_count + new_count;
            if (max_records_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
                throw std::runtime_error("Metal record delta exceeds uint32 record counter capacity");
            }
            if (max_records_u64 == 0) {
                result->old_input_count = old_count;
                result->new_input_count = new_count;
                return 0;
            }
            std::lock_guard<std::mutex> lock(g_mutex);
            ensure_pipeline_locked();
            const std::size_t old_count_size = static_cast<std::size_t>(old_count);
            const std::size_t new_count_size = static_cast<std::size_t>(new_count);
            const std::size_t max_records = static_cast<std::size_t>(max_records_u64);
            id<MTLBuffer> old_offsets_buffer = make_input_buffer(old_offsets, (old_count_size + 1) * sizeof(std::uint64_t));
            id<MTLBuffer> old_layers_buffer = make_input_buffer(old_layers, old_count_size * sizeof(std::int32_t));
            id<MTLBuffer> old_datatypes_buffer = make_input_buffer(old_datatypes, old_count_size * sizeof(std::int32_t));
            id<MTLBuffer> old_bboxes_buffer = make_input_buffer(old_bboxes, old_count_size * 4 * sizeof(std::int64_t));
            id<MTLBuffer> old_flags_buffer = make_input_buffer(old_flags, old_count_size * sizeof(std::uint32_t));
            id<MTLBuffer> old_hashes_buffer = make_input_buffer(old_hashes, old_count_size * 2 * sizeof(std::uint64_t));
            id<MTLBuffer> new_offsets_buffer = make_input_buffer(new_offsets, (new_count_size + 1) * sizeof(std::uint64_t));
            id<MTLBuffer> new_layers_buffer = make_input_buffer(new_layers, new_count_size * sizeof(std::int32_t));
            id<MTLBuffer> new_datatypes_buffer = make_input_buffer(new_datatypes, new_count_size * sizeof(std::int32_t));
            id<MTLBuffer> new_bboxes_buffer = make_input_buffer(new_bboxes, new_count_size * 4 * sizeof(std::int64_t));
            id<MTLBuffer> new_flags_buffer = make_input_buffer(new_flags, new_count_size * sizeof(std::uint32_t));
            id<MTLBuffer> new_hashes_buffer = make_input_buffer(new_hashes, new_count_size * 2 * sizeof(std::uint64_t));
            id<MTLBuffer> out_layers_buffer = make_output_buffer(max_records * sizeof(std::int32_t));
            id<MTLBuffer> out_datatypes_buffer = make_output_buffer(max_records * sizeof(std::int32_t));
            id<MTLBuffer> out_directions_buffer = make_output_buffer(max_records * sizeof(std::uint8_t));
            id<MTLBuffer> out_bboxes_buffer = make_output_buffer(max_records * 4 * sizeof(std::int64_t));
            id<MTLBuffer> out_vertex_counts_buffer = make_output_buffer(max_records * sizeof(std::uint64_t));
            id<MTLBuffer> out_flags_buffer = make_output_buffer(max_records * sizeof(std::uint32_t));
            id<MTLBuffer> out_hashes_buffer = make_output_buffer(max_records * 2 * sizeof(std::uint64_t));
            id<MTLBuffer> out_representative_indices_buffer = make_output_buffer(max_records * sizeof(std::uint64_t));
            KeyParamsHost old_params{old_count, 0, 1};
            KeyParamsHost new_params{new_count, old_count, 2};
            id<MTLBuffer> old_params_buffer = make_input_buffer(&old_params, sizeof(old_params));
            id<MTLBuffer> new_params_buffer = make_input_buffer(&new_params, sizeof(new_params));

            id<MTLCommandBuffer> command_buffer = [g_queue commandBuffer];
            if (!command_buffer) {
                throw std::runtime_error("Metal command buffer allocation failed");
            }
            auto encode = [&](
                id<MTLBuffer> offsets_buffer,
                id<MTLBuffer> layers_buffer,
                id<MTLBuffer> datatypes_buffer,
                id<MTLBuffer> bboxes_buffer,
                id<MTLBuffer> flags_buffer,
                id<MTLBuffer> hashes_buffer,
                id<MTLBuffer> params_buffer,
                NSUInteger width
            ) {
                if (width == 0) {
                    return;
                }
                id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
                if (!encoder) {
                    throw std::runtime_error("Metal compute encoder allocation failed");
                }
                [encoder setComputePipelineState:g_record_key_from_hash_pipeline];
                [encoder setBuffer:offsets_buffer offset:0 atIndex:0];
                [encoder setBuffer:layers_buffer offset:0 atIndex:1];
                [encoder setBuffer:datatypes_buffer offset:0 atIndex:2];
                [encoder setBuffer:bboxes_buffer offset:0 atIndex:3];
                [encoder setBuffer:flags_buffer offset:0 atIndex:4];
                [encoder setBuffer:hashes_buffer offset:0 atIndex:5];
                [encoder setBuffer:out_layers_buffer offset:0 atIndex:6];
                [encoder setBuffer:out_datatypes_buffer offset:0 atIndex:7];
                [encoder setBuffer:out_directions_buffer offset:0 atIndex:8];
                [encoder setBuffer:out_bboxes_buffer offset:0 atIndex:9];
                [encoder setBuffer:out_vertex_counts_buffer offset:0 atIndex:10];
                [encoder setBuffer:out_flags_buffer offset:0 atIndex:11];
                [encoder setBuffer:out_hashes_buffer offset:0 atIndex:12];
                [encoder setBuffer:out_representative_indices_buffer offset:0 atIndex:13];
                [encoder setBuffer:params_buffer offset:0 atIndex:14];
                NSUInteger group_width = g_record_key_from_hash_pipeline.maxTotalThreadsPerThreadgroup;
                if (group_width > 256) {
                    group_width = 256;
                }
                if (group_width == 0) {
                    group_width = 1;
                }
                [encoder dispatchThreads:MTLSizeMake(width, 1, 1) threadsPerThreadgroup:MTLSizeMake(group_width, 1, 1)];
                [encoder endEncoding];
            };
            encode(old_offsets_buffer, old_layers_buffer, old_datatypes_buffer, old_bboxes_buffer, old_flags_buffer, old_hashes_buffer, old_params_buffer, static_cast<NSUInteger>(old_count));
            encode(new_offsets_buffer, new_layers_buffer, new_datatypes_buffer, new_bboxes_buffer, new_flags_buffer, new_hashes_buffer, new_params_buffer, static_cast<NSUInteger>(new_count));
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
            if (command_buffer.status == MTLCommandBufferStatusError) {
                NSString* message = command_buffer.error ? [command_buffer.error localizedDescription] : @"Metal command failed";
                throw std::runtime_error([[message description] UTF8String]);
            }

            const auto* out_layers = reinterpret_cast<const std::int32_t*>([out_layers_buffer contents]);
            const auto* out_datatypes = reinterpret_cast<const std::int32_t*>([out_datatypes_buffer contents]);
            const auto* out_directions = reinterpret_cast<const std::uint8_t*>([out_directions_buffer contents]);
            const auto* out_bboxes = reinterpret_cast<const std::int64_t*>([out_bboxes_buffer contents]);
            const auto* out_vertex_counts = reinterpret_cast<const std::uint64_t*>([out_vertex_counts_buffer contents]);
            const auto* out_flags = reinterpret_cast<const std::uint32_t*>([out_flags_buffer contents]);
            const auto* out_hashes = reinterpret_cast<const std::uint64_t*>([out_hashes_buffer contents]);
            const auto* out_reps = reinterpret_cast<const std::uint64_t*>([out_representative_indices_buffer contents]);
            std::vector<HostRecordDelta> source_records;
            source_records.reserve(max_records);
            for (std::size_t row = 0; row < max_records; ++row) {
                HostRecordDelta record{};
                record.layer = out_layers[row];
                record.datatype = out_datatypes[row];
                record.direction = out_directions[row];
                record.count = 1;
                for (int offset = 0; offset < 4; ++offset) {
                    record.bbox[offset] = out_bboxes[row * 4 + offset];
                }
                record.vertex_count = out_vertex_counts[row];
                record.flags = out_flags[row];
                record.hash[0] = out_hashes[row * 2];
                record.hash[1] = out_hashes[row * 2 + 1];
                record.representative_index = out_reps[row];
                source_records.push_back(record);
            }
            auto key_less = [](const HostRecordDelta& left, const HostRecordDelta& right) {
                return std::make_tuple(left.layer, left.datatype, left.bbox[0], left.bbox[1], left.bbox[2], left.bbox[3], left.vertex_count, left.flags, left.hash[0], left.hash[1])
                    < std::make_tuple(right.layer, right.datatype, right.bbox[0], right.bbox[1], right.bbox[2], right.bbox[3], right.vertex_count, right.flags, right.hash[0], right.hash[1]);
            };
            auto key_equal = [](const HostRecordDelta& left, const HostRecordDelta& right) {
                return left.layer == right.layer
                    && left.datatype == right.datatype
                    && left.bbox[0] == right.bbox[0]
                    && left.bbox[1] == right.bbox[1]
                    && left.bbox[2] == right.bbox[2]
                    && left.bbox[3] == right.bbox[3]
                    && left.vertex_count == right.vertex_count
                    && left.flags == right.flags
                    && left.hash[0] == right.hash[0]
                    && left.hash[1] == right.hash[1];
            };
            std::sort(source_records.begin(), source_records.end(), key_less);
            std::vector<HostRecordDelta> records;
            std::size_t index = 0;
            while (index < source_records.size()) {
                const auto& key_record = source_records[index];
                std::uint64_t old_total = 0;
                std::uint64_t new_total = 0;
                std::uint64_t old_rep = 0;
                std::uint64_t new_rep = 0;
                std::size_t next = index;
                while (next < source_records.size() && key_equal(key_record, source_records[next])) {
                    if (source_records[next].direction == 1) {
                        if (old_total == 0) {
                            old_rep = source_records[next].representative_index;
                        }
                        old_total += 1;
                    } else {
                        if (new_total == 0) {
                            new_rep = source_records[next].representative_index;
                        }
                        new_total += 1;
                    }
                    next += 1;
                }
                result->reduced_key_count += 1;
                result->matched_record_count += std::min(old_total, new_total);
                if (old_total > new_total) {
                    HostRecordDelta delta = key_record;
                    delta.direction = 1;
                    delta.count = old_total - new_total;
                    delta.representative_index = old_rep;
                    records.push_back(delta);
                } else if (new_total > old_total) {
                    HostRecordDelta delta = key_record;
                    delta.direction = 2;
                    delta.count = new_total - old_total;
                    delta.representative_index = new_rep;
                    records.push_back(delta);
                }
                index = next;
            }
            std::sort(records.begin(), records.end(), [](const HostRecordDelta& left, const HostRecordDelta& right) {
                return std::make_tuple(left.direction, left.layer, left.datatype, left.bbox[0], left.bbox[1], left.bbox[2], left.bbox[3], left.hash[0], left.hash[1])
                    < std::make_tuple(right.direction, right.layer, right.datatype, right.bbox[0], right.bbox[1], right.bbox[2], right.bbox[3], right.hash[0], right.hash[1]);
            });

            const std::size_t record_count = records.size();
            result->record_count = record_count;
            result->old_input_count = old_count;
            result->new_input_count = new_count;
            result->metal_kernel_count = (old_count ? 1 : 0) + (new_count ? 1 : 0);
            result->metal_event_elapsed_ms = 0.0f;
            if (record_count == 0) {
                return 0;
            }
            result->layers = static_cast<std::int32_t*>(std::malloc(sizeof(std::int32_t) * record_count));
            result->datatypes = static_cast<std::int32_t*>(std::malloc(sizeof(std::int32_t) * record_count));
            result->directions = static_cast<std::uint8_t*>(std::malloc(sizeof(std::uint8_t) * record_count));
            result->counts = static_cast<std::uint64_t*>(std::malloc(sizeof(std::uint64_t) * record_count));
            result->bboxes = static_cast<std::int64_t*>(std::malloc(sizeof(std::int64_t) * record_count * 4));
            result->vertex_counts = static_cast<std::uint64_t*>(std::malloc(sizeof(std::uint64_t) * record_count));
            result->flags = static_cast<std::uint32_t*>(std::malloc(sizeof(std::uint32_t) * record_count));
            result->hashes = static_cast<std::uint64_t*>(std::malloc(sizeof(std::uint64_t) * record_count * 2));
            result->representative_indices = static_cast<std::uint64_t*>(std::malloc(sizeof(std::uint64_t) * record_count));
            if (!result->layers || !result->datatypes || !result->directions || !result->counts || !result->bboxes
                || !result->vertex_counts || !result->flags || !result->hashes || !result->representative_indices) {
                throw std::bad_alloc();
            }
            for (std::size_t row = 0; row < record_count; ++row) {
                const auto& record = records[row];
                result->layers[row] = record.layer;
                result->datatypes[row] = record.datatype;
                result->directions[row] = record.direction;
                result->counts[row] = record.count;
                for (int offset = 0; offset < 4; ++offset) {
                    result->bboxes[row * 4 + offset] = record.bbox[offset];
                }
                result->vertex_counts[row] = record.vertex_count;
                result->flags[row] = record.flags;
                result->hashes[row * 2] = record.hash[0];
                result->hashes[row * 2 + 1] = record.hash[1];
                result->representative_indices[row] = record.representative_index;
            }
            return 0;
        } catch (const std::exception& exc) {
            set_error(exc.what());
            if (result) {
                gds_metal_free_polygon_record_delta_result(result);
            }
            return 1;
        } catch (...) {
            set_error("unknown Metal polygon record delta failure");
            if (result) {
                gds_metal_free_polygon_record_delta_result(result);
            }
            return 1;
        }
    }
}

int gds_metal_context_overlap_pairs(
    const std::int64_t* component_bboxes,
    const std::int32_t* component_layers,
    const std::int32_t* component_datatypes,
    std::uint64_t component_count,
    const std::int64_t* polygon_bboxes,
    const std::int32_t* polygon_layers,
    const std::int32_t* polygon_datatypes,
    std::uint64_t polygon_count,
    GdsMetalContextOverlapPairsResult* result
) {
    @autoreleasepool {
        try {
            if (!result) {
                throw std::runtime_error("context overlap result pointer is null");
            }
            *result = GdsMetalContextOverlapPairsResult{};
            if (component_count == 0 || polygon_count == 0) {
                return 0;
            }
            if (!component_bboxes || !component_layers || !component_datatypes || !polygon_bboxes || !polygon_layers || !polygon_datatypes) {
                throw std::runtime_error("context overlap input contains null pointer");
            }
            if (component_count > std::numeric_limits<std::uint64_t>::max() / polygon_count) {
                throw std::overflow_error("context overlap pair search count overflow");
            }
            const std::uint64_t total_pairs = component_count * polygon_count;
            if (total_pairs > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
                throw std::overflow_error("context overlap pair count exceeds host address space");
            }
            std::lock_guard<std::mutex> lock(g_mutex);
            ensure_pipeline_locked();
            const std::size_t component_count_size = static_cast<std::size_t>(component_count);
            const std::size_t polygon_count_size = static_cast<std::size_t>(polygon_count);
            const std::size_t total_pairs_size = static_cast<std::size_t>(total_pairs);
            id<MTLBuffer> component_bboxes_buffer = make_input_buffer(component_bboxes, component_count_size * 4 * sizeof(std::int64_t));
            id<MTLBuffer> component_layers_buffer = make_input_buffer(component_layers, component_count_size * sizeof(std::int32_t));
            id<MTLBuffer> component_datatypes_buffer = make_input_buffer(component_datatypes, component_count_size * sizeof(std::int32_t));
            id<MTLBuffer> polygon_bboxes_buffer = make_input_buffer(polygon_bboxes, polygon_count_size * 4 * sizeof(std::int64_t));
            id<MTLBuffer> polygon_layers_buffer = make_input_buffer(polygon_layers, polygon_count_size * sizeof(std::int32_t));
            id<MTLBuffer> polygon_datatypes_buffer = make_input_buffer(polygon_datatypes, polygon_count_size * sizeof(std::int32_t));
            id<MTLBuffer> flags_buffer = make_output_buffer(total_pairs_size * sizeof(std::uint8_t));
            ContextParamsHost params{component_count, polygon_count, total_pairs};
            id<MTLBuffer> params_buffer = make_input_buffer(&params, sizeof(params));

            id<MTLCommandBuffer> command_buffer = [g_queue commandBuffer];
            if (!command_buffer) {
                throw std::runtime_error("Metal command buffer allocation failed");
            }
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            if (!encoder) {
                throw std::runtime_error("Metal compute encoder allocation failed");
            }
            [encoder setComputePipelineState:g_context_overlap_pipeline];
            [encoder setBuffer:component_bboxes_buffer offset:0 atIndex:0];
            [encoder setBuffer:component_layers_buffer offset:0 atIndex:1];
            [encoder setBuffer:component_datatypes_buffer offset:0 atIndex:2];
            [encoder setBuffer:polygon_bboxes_buffer offset:0 atIndex:3];
            [encoder setBuffer:polygon_layers_buffer offset:0 atIndex:4];
            [encoder setBuffer:polygon_datatypes_buffer offset:0 atIndex:5];
            [encoder setBuffer:flags_buffer offset:0 atIndex:6];
            [encoder setBuffer:params_buffer offset:0 atIndex:7];
            NSUInteger group_width = g_context_overlap_pipeline.maxTotalThreadsPerThreadgroup;
            if (group_width > 256) {
                group_width = 256;
            }
            if (group_width == 0) {
                group_width = 1;
            }
            [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(total_pairs), 1, 1) threadsPerThreadgroup:MTLSizeMake(group_width, 1, 1)];
            [encoder endEncoding];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
            if (command_buffer.status == MTLCommandBufferStatusError) {
                NSString* message = command_buffer.error ? [command_buffer.error localizedDescription] : @"Metal command failed";
                throw std::runtime_error([[message description] UTF8String]);
            }

            const auto* flags = reinterpret_cast<const std::uint8_t*>([flags_buffer contents]);
            std::uint64_t pair_count = 0;
            for (std::size_t linear = 0; linear < total_pairs_size; ++linear) {
                pair_count += flags[linear] ? 1 : 0;
            }
            result->pair_count = pair_count;
            result->metal_kernel_count = 1;
            result->metal_event_elapsed_ms = 0.0f;
            if (pair_count == 0) {
                return 0;
            }
            result->component_indices = static_cast<std::uint64_t*>(std::malloc(sizeof(std::uint64_t) * pair_count));
            result->polygon_indices = static_cast<std::uint64_t*>(std::malloc(sizeof(std::uint64_t) * pair_count));
            if (!result->component_indices || !result->polygon_indices) {
                throw std::bad_alloc();
            }
            std::uint64_t out_index = 0;
            for (std::uint64_t linear = 0; linear < total_pairs; ++linear) {
                if (!flags[static_cast<std::size_t>(linear)]) {
                    continue;
                }
                const std::uint64_t component_index = linear / polygon_count;
                const std::uint64_t polygon_index = linear - component_index * polygon_count;
                result->component_indices[out_index] = component_index;
                result->polygon_indices[out_index] = polygon_index;
                out_index += 1;
            }
            return 0;
        } catch (const std::exception& exc) {
            set_error(exc.what());
            if (result) {
                gds_metal_free_context_overlap_pairs_result(result);
            }
            return 1;
        } catch (...) {
            set_error("unknown Metal context overlap failure");
            if (result) {
                gds_metal_free_context_overlap_pairs_result(result);
            }
            return 1;
        }
    }
}

int gds_metal_boundary_loops_from_edges(
    const std::int64_t* edges,
    std::uint64_t edge_count,
    GdsMetalBoundaryLoopResult* result
) {
    @autoreleasepool {
        try {
            if (!result) {
                throw std::runtime_error("boundary loop result pointer is null");
            }
            *result = GdsMetalBoundaryLoopResult{};
            if (edge_count == 0) {
                result->fragment_offsets = static_cast<std::uint64_t*>(std::malloc(sizeof(std::uint64_t)));
                if (!result->fragment_offsets) {
                    throw std::bad_alloc();
                }
                result->fragment_offsets[0] = 0;
                return 0;
            }
            if (!edges) {
                throw std::runtime_error("boundary loop edge pointer is null");
            }
            if (edge_count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
                throw std::overflow_error("boundary loop edge count exceeds host address space");
            }
            std::lock_guard<std::mutex> lock(g_mutex);
            ensure_pipeline_locked();
            const std::size_t edge_count_size = static_cast<std::size_t>(edge_count);
            id<MTLBuffer> edges_buffer = make_input_buffer(edges, edge_count_size * 4 * sizeof(std::int64_t));
            id<MTLBuffer> next_indices_buffer = make_output_buffer(edge_count_size * sizeof(std::uint64_t));
            std::vector<std::uint8_t> remaining(edge_count_size, 1);
            id<MTLBuffer> remaining_buffer = make_input_buffer(remaining.data(), remaining.size() * sizeof(std::uint8_t));
            id<MTLBuffer> scratch_a_buffer = make_output_buffer(edge_count_size * 2 * sizeof(std::int64_t));
            id<MTLBuffer> scratch_b_buffer = make_output_buffer(edge_count_size * 2 * sizeof(std::int64_t));
            id<MTLBuffer> vertices_buffer = make_output_buffer(edge_count_size * 2 * sizeof(std::int64_t));
            id<MTLBuffer> offsets_buffer = make_output_buffer((edge_count_size + 1) * sizeof(std::uint64_t));
            id<MTLBuffer> bboxes_buffer = make_output_buffer(edge_count_size * 4 * sizeof(std::int64_t));
            id<MTLBuffer> areas_buffer = make_output_buffer(edge_count_size * sizeof(std::int64_t));
            id<MTLBuffer> stats_buffer = make_output_buffer(sizeof(DeviceLoopTraceStatsHost));
            id<MTLBuffer> edge_count_buffer = make_input_buffer(&edge_count, sizeof(edge_count));

            id<MTLCommandBuffer> command_buffer = [g_queue commandBuffer];
            if (!command_buffer) {
                throw std::runtime_error("Metal command buffer allocation failed");
            }
            id<MTLComputeCommandEncoder> next_encoder = [command_buffer computeCommandEncoder];
            if (!next_encoder) {
                throw std::runtime_error("Metal boundary loop next encoder allocation failed");
            }
            [next_encoder setComputePipelineState:g_boundary_loop_next_pipeline];
            [next_encoder setBuffer:edges_buffer offset:0 atIndex:0];
            [next_encoder setBuffer:next_indices_buffer offset:0 atIndex:1];
            [next_encoder setBuffer:edge_count_buffer offset:0 atIndex:2];
            NSUInteger next_group_width = g_boundary_loop_next_pipeline.maxTotalThreadsPerThreadgroup;
            if (next_group_width > 256) {
                next_group_width = 256;
            }
            if (next_group_width == 0) {
                next_group_width = 1;
            }
            [next_encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(edge_count), 1, 1) threadsPerThreadgroup:MTLSizeMake(next_group_width, 1, 1)];
            [next_encoder endEncoding];

            id<MTLComputeCommandEncoder> trace_encoder = [command_buffer computeCommandEncoder];
            if (!trace_encoder) {
                throw std::runtime_error("Metal boundary loop trace encoder allocation failed");
            }
            [trace_encoder setComputePipelineState:g_boundary_loop_trace_pipeline];
            [trace_encoder setBuffer:edges_buffer offset:0 atIndex:0];
            [trace_encoder setBuffer:next_indices_buffer offset:0 atIndex:1];
            [trace_encoder setBuffer:remaining_buffer offset:0 atIndex:2];
            [trace_encoder setBuffer:scratch_a_buffer offset:0 atIndex:3];
            [trace_encoder setBuffer:scratch_b_buffer offset:0 atIndex:4];
            [trace_encoder setBuffer:vertices_buffer offset:0 atIndex:5];
            [trace_encoder setBuffer:offsets_buffer offset:0 atIndex:6];
            [trace_encoder setBuffer:bboxes_buffer offset:0 atIndex:7];
            [trace_encoder setBuffer:areas_buffer offset:0 atIndex:8];
            [trace_encoder setBuffer:stats_buffer offset:0 atIndex:9];
            [trace_encoder setBuffer:edge_count_buffer offset:0 atIndex:10];
            [trace_encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [trace_encoder endEncoding];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
            if (command_buffer.status == MTLCommandBufferStatusError) {
                NSString* message = command_buffer.error ? [command_buffer.error localizedDescription] : @"Metal boundary loop command failed";
                throw std::runtime_error([[message description] UTF8String]);
            }
            const auto* stats = reinterpret_cast<const DeviceLoopTraceStatsHost*>([stats_buffer contents]);
            if (stats->error) {
                throw std::runtime_error(stats->error == 1 ? "Metal boundary loop trace exceeded raw vertex limit" : "Metal boundary loop trace exceeded output limit");
            }
            result->fragment_count = stats->fragment_count;
            result->vertex_count = stats->vertex_count;
            result->dropped_zero_area_loop_count = stats->dropped_zero_area_loop_count;
            result->metal_kernel_count = 2;
            result->metal_event_elapsed_ms = 0.0f;
            if (result->fragment_count) {
                result->fragment_offsets = static_cast<std::uint64_t*>(std::malloc(sizeof(std::uint64_t) * (result->fragment_count + 1)));
                result->vertices_xy = static_cast<std::int64_t*>(std::malloc(sizeof(std::int64_t) * result->vertex_count * 2));
                result->bboxes = static_cast<std::int64_t*>(std::malloc(sizeof(std::int64_t) * result->fragment_count * 4));
                result->twice_areas = static_cast<std::int64_t*>(std::malloc(sizeof(std::int64_t) * result->fragment_count));
                if (!result->fragment_offsets || !result->vertices_xy || !result->bboxes || !result->twice_areas) {
                    throw std::bad_alloc();
                }
                std::memcpy(result->fragment_offsets, [offsets_buffer contents], sizeof(std::uint64_t) * (result->fragment_count + 1));
                std::memcpy(result->vertices_xy, [vertices_buffer contents], sizeof(std::int64_t) * result->vertex_count * 2);
                std::memcpy(result->bboxes, [bboxes_buffer contents], sizeof(std::int64_t) * result->fragment_count * 4);
                std::memcpy(result->twice_areas, [areas_buffer contents], sizeof(std::int64_t) * result->fragment_count);
            } else {
                result->fragment_offsets = static_cast<std::uint64_t*>(std::malloc(sizeof(std::uint64_t)));
                if (!result->fragment_offsets) {
                    throw std::bad_alloc();
                }
                result->fragment_offsets[0] = 0;
            }
            return 0;
        } catch (const std::exception& exc) {
            set_error(exc.what());
            if (result) {
                gds_metal_free_boundary_loop_result(result);
            }
            return 1;
        } catch (...) {
            set_error("unknown Metal boundary loop failure");
            if (result) {
                gds_metal_free_boundary_loop_result(result);
            }
            return 1;
        }
    }
}

int gds_metal_one_sided_boundary_edges(
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
    std::uint8_t* out_keep
) {
    @autoreleasepool {
        try {
            if (!out_edges || !out_keep) {
                throw std::runtime_error("one-sided boundary output pointer is null");
            }
            if (segment_count == 0) {
                return 0;
            }
            if (!polygon_offsets || !bboxes || !signed_twice_areas || !segment_points || !segment_polygon_indices || (vertex_count && !vertices_xy)) {
                throw std::runtime_error("one-sided boundary input contains null pointer");
            }
            std::lock_guard<std::mutex> lock(g_mutex);
            ensure_pipeline_locked();
            const std::size_t vertex_count_size = static_cast<std::size_t>(vertex_count);
            const std::size_t polygon_count_size = static_cast<std::size_t>(polygon_count);
            const std::size_t segment_count_size = static_cast<std::size_t>(segment_count);
            std::vector<float> segment_points_float(segment_count_size * 4);
            for (std::size_t index = 0; index < segment_points_float.size(); ++index) {
                segment_points_float[index] = static_cast<float>(segment_points[index]);
            }
            id<MTLBuffer> vertices_buffer = make_input_buffer(vertices_xy, vertex_count_size * 2 * sizeof(std::int64_t));
            id<MTLBuffer> offsets_buffer = make_input_buffer(polygon_offsets, (polygon_count_size + 1) * sizeof(std::uint64_t));
            id<MTLBuffer> bboxes_buffer = make_input_buffer(bboxes, polygon_count_size * 4 * sizeof(std::int64_t));
            id<MTLBuffer> signed_buffer = make_input_buffer(signed_twice_areas, polygon_count_size * sizeof(std::int64_t));
            id<MTLBuffer> segment_buffer = make_input_buffer(segment_points_float.data(), segment_points_float.size() * sizeof(float));
            id<MTLBuffer> index_buffer = make_input_buffer(segment_polygon_indices, segment_count_size * sizeof(std::uint32_t));
            id<MTLBuffer> out_edges_buffer = make_output_buffer(segment_count_size * 4 * sizeof(float));
            id<MTLBuffer> out_keep_buffer = make_output_buffer(segment_count_size * sizeof(std::uint8_t));
            BoundaryEdgeParamsHost params{vertex_count, polygon_count, segment_count, 0};
            id<MTLBuffer> params_buffer = make_input_buffer(&params, sizeof(params));

            id<MTLCommandBuffer> command_buffer = [g_queue commandBuffer];
            if (!command_buffer) {
                throw std::runtime_error("Metal command buffer allocation failed");
            }
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            if (!encoder) {
                throw std::runtime_error("Metal compute encoder allocation failed");
            }
            [encoder setComputePipelineState:g_one_sided_boundary_pipeline];
            [encoder setBuffer:vertices_buffer offset:0 atIndex:0];
            [encoder setBuffer:offsets_buffer offset:0 atIndex:1];
            [encoder setBuffer:bboxes_buffer offset:0 atIndex:2];
            [encoder setBuffer:signed_buffer offset:0 atIndex:3];
            [encoder setBuffer:segment_buffer offset:0 atIndex:4];
            [encoder setBuffer:index_buffer offset:0 atIndex:5];
            [encoder setBuffer:out_edges_buffer offset:0 atIndex:6];
            [encoder setBuffer:out_keep_buffer offset:0 atIndex:7];
            [encoder setBuffer:params_buffer offset:0 atIndex:8];
            NSUInteger group_width = g_one_sided_boundary_pipeline.maxTotalThreadsPerThreadgroup;
            if (group_width > 256) {
                group_width = 256;
            }
            if (group_width == 0) {
                group_width = 1;
            }
            [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(segment_count), 1, 1) threadsPerThreadgroup:MTLSizeMake(group_width, 1, 1)];
            [encoder endEncoding];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
            if (command_buffer.status == MTLCommandBufferStatusError) {
                NSString* message = command_buffer.error ? [command_buffer.error localizedDescription] : @"Metal one-sided boundary command failed";
                throw std::runtime_error([[message description] UTF8String]);
            }
            const auto* out_edges_float = reinterpret_cast<const float*>([out_edges_buffer contents]);
            for (std::size_t index = 0; index < segment_count_size * 4; ++index) {
                out_edges[index] = static_cast<double>(out_edges_float[index]);
            }
            std::memcpy(out_keep, [out_keep_buffer contents], segment_count_size * sizeof(std::uint8_t));
            return 0;
        } catch (const std::exception& exc) {
            set_error(exc.what());
            return 1;
        } catch (...) {
            set_error("unknown Metal one-sided boundary edge failure");
            return 1;
        }
    }
}

int gds_metal_mixed_boundary_edges(
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
    std::uint8_t* out_keep
) {
    @autoreleasepool {
        try {
            if (!out_edges || !out_keep) {
                throw std::runtime_error("mixed boundary output pointer is null");
            }
            if (direction != 1 && direction != 2) {
                throw std::runtime_error("mixed boundary direction must be 1 or 2");
            }
            if (segment_count == 0) {
                return 0;
            }
            if (!polygon_offsets || !bboxes || !polygon_sides || !segment_points || (vertex_count && !vertices_xy)) {
                throw std::runtime_error("mixed boundary input contains null pointer");
            }
            std::lock_guard<std::mutex> lock(g_mutex);
            ensure_pipeline_locked();
            const std::size_t vertex_count_size = static_cast<std::size_t>(vertex_count);
            const std::size_t polygon_count_size = static_cast<std::size_t>(polygon_count);
            const std::size_t segment_count_size = static_cast<std::size_t>(segment_count);
            std::vector<float> segment_points_float(segment_count_size * 4);
            for (std::size_t index = 0; index < segment_points_float.size(); ++index) {
                segment_points_float[index] = static_cast<float>(segment_points[index]);
            }
            id<MTLBuffer> vertices_buffer = make_input_buffer(vertices_xy, vertex_count_size * 2 * sizeof(std::int64_t));
            id<MTLBuffer> offsets_buffer = make_input_buffer(polygon_offsets, (polygon_count_size + 1) * sizeof(std::uint64_t));
            id<MTLBuffer> bboxes_buffer = make_input_buffer(bboxes, polygon_count_size * 4 * sizeof(std::int64_t));
            id<MTLBuffer> sides_buffer = make_input_buffer(polygon_sides, polygon_count_size * sizeof(std::uint8_t));
            id<MTLBuffer> segment_buffer = make_input_buffer(segment_points_float.data(), segment_points_float.size() * sizeof(float));
            id<MTLBuffer> out_edges_buffer = make_output_buffer(segment_count_size * 4 * sizeof(float));
            id<MTLBuffer> out_keep_buffer = make_output_buffer(segment_count_size * sizeof(std::uint8_t));
            BoundaryEdgeParamsHost params{vertex_count, polygon_count, segment_count, direction};
            id<MTLBuffer> params_buffer = make_input_buffer(&params, sizeof(params));

            id<MTLCommandBuffer> command_buffer = [g_queue commandBuffer];
            if (!command_buffer) {
                throw std::runtime_error("Metal command buffer allocation failed");
            }
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            if (!encoder) {
                throw std::runtime_error("Metal compute encoder allocation failed");
            }
            [encoder setComputePipelineState:g_mixed_boundary_pipeline];
            [encoder setBuffer:vertices_buffer offset:0 atIndex:0];
            [encoder setBuffer:offsets_buffer offset:0 atIndex:1];
            [encoder setBuffer:bboxes_buffer offset:0 atIndex:2];
            [encoder setBuffer:sides_buffer offset:0 atIndex:3];
            [encoder setBuffer:segment_buffer offset:0 atIndex:4];
            [encoder setBuffer:out_edges_buffer offset:0 atIndex:5];
            [encoder setBuffer:out_keep_buffer offset:0 atIndex:6];
            [encoder setBuffer:params_buffer offset:0 atIndex:7];
            NSUInteger group_width = g_mixed_boundary_pipeline.maxTotalThreadsPerThreadgroup;
            if (group_width > 256) {
                group_width = 256;
            }
            if (group_width == 0) {
                group_width = 1;
            }
            [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(segment_count), 1, 1) threadsPerThreadgroup:MTLSizeMake(group_width, 1, 1)];
            [encoder endEncoding];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
            if (command_buffer.status == MTLCommandBufferStatusError) {
                NSString* message = command_buffer.error ? [command_buffer.error localizedDescription] : @"Metal mixed boundary command failed";
                throw std::runtime_error([[message description] UTF8String]);
            }
            const auto* out_edges_float = reinterpret_cast<const float*>([out_edges_buffer contents]);
            for (std::size_t index = 0; index < segment_count_size * 4; ++index) {
                out_edges[index] = static_cast<double>(out_edges_float[index]);
            }
            std::memcpy(out_keep, [out_keep_buffer contents], segment_count_size * sizeof(std::uint8_t));
            return 0;
        } catch (const std::exception& exc) {
            set_error(exc.what());
            return 1;
        } catch (...) {
            set_error("unknown Metal mixed boundary edge failure");
            return 1;
        }
    }
}

}  // extern "C"
