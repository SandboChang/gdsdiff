#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

thread_local std::string g_last_error;

struct LayerKey {
    std::int32_t layer = 0;
    std::int32_t datatype = 0;

    bool operator<(const LayerKey& other) const {
        return layer < other.layer || (layer == other.layer && datatype < other.datatype);
    }
};

struct BBox {
    std::int64_t x0 = 0;
    std::int64_t y0 = 0;
    std::int64_t x1 = 0;
    std::int64_t y1 = 0;
    bool valid = false;
};

struct LayerSummary {
    std::uint64_t count = 0;
    BBox bbox;
};

struct Reference {
    std::string child;
    std::uint64_t columns = 1;
    std::uint64_t rows = 1;
    std::int64_t ox = 0;
    std::int64_t oy = 0;
    std::int64_t cx = 0;
    std::int64_t cy = 0;
    std::int64_t rx = 0;
    std::int64_t ry = 0;
    bool has_xy = false;
    std::uint64_t raw_hash = 0;
};

struct Cell {
    std::map<LayerKey, LayerSummary> layers;
    std::vector<Reference> references;
    std::uint64_t local_shape_count = 0;
    std::uint64_t local_geometry_hash = 1469598103934665603ull;
};

struct FlattenedSummary {
    std::map<LayerKey, LayerSummary> layers;
    std::uint64_t shape_count = 0;
    std::uint64_t geometry_hash = 1469598103934665603ull;
};

std::uint16_t read_u16(const unsigned char* p) {
    return static_cast<std::uint16_t>((static_cast<std::uint16_t>(p[0]) << 8) | p[1]);
}

std::int16_t read_i16(const unsigned char* p) {
    return static_cast<std::int16_t>(read_u16(p));
}

std::int32_t read_i32(const unsigned char* p) {
    return (static_cast<std::int32_t>(p[0]) << 24) | (static_cast<std::int32_t>(p[1]) << 16)
        | (static_cast<std::int32_t>(p[2]) << 8) | static_cast<std::int32_t>(p[3]);
}

std::string decode_string(const unsigned char* data, std::size_t size) {
    while (size && data[size - 1] == 0) {
        --size;
    }
    return std::string(reinterpret_cast<const char*>(data), size);
}

std::uint64_t fnv1a_update(std::uint64_t hash, const unsigned char* data, std::size_t size) {
    constexpr std::uint64_t prime = 1099511628211ull;
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= static_cast<std::uint64_t>(data[i]);
        hash *= prime;
    }
    return hash;
}

std::uint64_t fnv1a_u64(std::uint64_t hash, std::uint64_t value) {
    unsigned char bytes[8];
    for (int i = 0; i < 8; ++i) {
        bytes[i] = static_cast<unsigned char>((value >> (56 - i * 8)) & 0xff);
    }
    return fnv1a_update(hash, bytes, sizeof(bytes));
}

std::uint64_t fnv1a_i64(std::uint64_t hash, std::int64_t value) {
    return fnv1a_u64(hash, static_cast<std::uint64_t>(value));
}

std::vector<unsigned char> read_file(const char* path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("unable to open GDS file");
    }
    const auto size = in.tellg();
    if (size < 0) {
        throw std::runtime_error("unable to determine GDS file size");
    }
    std::vector<unsigned char> data(static_cast<std::size_t>(size));
    in.seekg(0, std::ios::beg);
    if (!data.empty()) {
        in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
        if (!in) {
            throw std::runtime_error("unable to read GDS file");
        }
    }
    return data;
}

void bbox_include(BBox& target, const BBox& source) {
    if (!source.valid) {
        return;
    }
    if (!target.valid) {
        target = source;
        return;
    }
    target.x0 = std::min(target.x0, source.x0);
    target.y0 = std::min(target.y0, source.y0);
    target.x1 = std::max(target.x1, source.x1);
    target.y1 = std::max(target.y1, source.y1);
}

BBox shifted_bbox(const BBox& bbox, std::int64_t dx, std::int64_t dy) {
    if (!bbox.valid) {
        return bbox;
    }
    return BBox{bbox.x0 + dx, bbox.y0 + dy, bbox.x1 + dx, bbox.y1 + dy, true};
}

BBox bbox_from_xy(const std::vector<std::pair<std::int64_t, std::int64_t>>& xy) {
    BBox bbox;
    for (const auto& point : xy) {
        BBox p{point.first, point.second, point.first, point.second, true};
        bbox_include(bbox, p);
    }
    return bbox;
}

std::vector<std::pair<std::int64_t, std::int64_t>> parse_xy(const unsigned char* payload, std::size_t payload_size) {
    if (payload_size % 8 != 0) {
        throw std::runtime_error("invalid XY payload length");
    }
    std::vector<std::pair<std::int64_t, std::int64_t>> points;
    points.reserve(payload_size / 8);
    for (std::size_t i = 0; i < payload_size; i += 8) {
        points.push_back({read_i32(payload + i), read_i32(payload + i + 4)});
    }
    return points;
}

void add_layer_summary(std::map<LayerKey, LayerSummary>& layers, LayerKey key, std::uint64_t count, const BBox& bbox) {
    if (count == 0) {
        return;
    }
    auto& summary = layers[key];
    if (summary.count > std::numeric_limits<std::uint64_t>::max() - count) {
        throw std::overflow_error("layer shape count overflow");
    }
    summary.count += count;
    bbox_include(summary.bbox, bbox);
}

std::map<std::string, Cell> parse_gds(const char* path, bool detail) {
    const auto data = read_file(path);
    std::map<std::string, Cell> cells;
    std::size_t offset = 0;
    std::string current_name;
    Cell current_cell;
    int element_kind = -1;
    std::size_t element_start = 0;
    LayerKey pending_layer;
    std::string pending_sname;
    Reference pending_reference;

    while (offset + 4 <= data.size()) {
        const std::uint16_t record_len = read_u16(data.data() + offset);
        const int record_type = data[offset + 2];
        if (record_len < 4 || offset + record_len > data.size()) {
            throw std::runtime_error("invalid GDS record length");
        }
        const unsigned char* payload = data.data() + offset + 4;
        const std::size_t payload_size = record_len - 4;
        offset += record_len;

        switch (record_type) {
        case 0x05:  // BGNSTR
            current_name.clear();
            current_cell = Cell{};
            break;
        case 0x06:  // STRNAME
            current_name = decode_string(payload, payload_size);
            break;
        case 0x08:  // BOUNDARY
        case 0x09:  // PATH
            element_kind = record_type;
            element_start = offset - record_len;
            pending_layer = LayerKey{};
            break;
        case 0x0A:  // SREF
        case 0x0B:  // AREF
            element_kind = record_type;
            element_start = offset - record_len;
            pending_reference = Reference{};
            pending_sname.clear();
            break;
        case 0x0D:  // LAYER
            if ((element_kind == 0x08 || element_kind == 0x09) && payload_size >= 2) {
                pending_layer.layer = read_i16(payload);
            }
            break;
        case 0x0E:  // DATATYPE
            if ((element_kind == 0x08 || element_kind == 0x09) && payload_size >= 2) {
                pending_layer.datatype = read_i16(payload);
            }
            break;
        case 0x12:  // SNAME
            if (element_kind == 0x0A || element_kind == 0x0B) {
                pending_sname = decode_string(payload, payload_size);
            }
            break;
        case 0x13:  // COLROW
            if (element_kind == 0x0B) {
                if (payload_size < 4) {
                    throw std::runtime_error("invalid COLROW record");
                }
                const auto columns = read_i16(payload);
                const auto rows = read_i16(payload + 2);
                if (columns <= 0 || rows <= 0) {
                    throw std::runtime_error("invalid AREF repetition");
                }
                pending_reference.columns = static_cast<std::uint64_t>(columns);
                pending_reference.rows = static_cast<std::uint64_t>(rows);
            }
            break;
        case 0x10: {  // XY
            if (element_kind == 0x08 || element_kind == 0x09) {
                ++current_cell.local_shape_count;
                if (detail) {
                    const auto xy = parse_xy(payload, payload_size);
                    const BBox bbox = bbox_from_xy(xy);
                    add_layer_summary(current_cell.layers, pending_layer, 1, bbox);
                    current_cell.local_geometry_hash = fnv1a_u64(current_cell.local_geometry_hash, static_cast<std::uint64_t>(pending_layer.layer));
                    current_cell.local_geometry_hash = fnv1a_u64(current_cell.local_geometry_hash, static_cast<std::uint64_t>(pending_layer.datatype));
                    current_cell.local_geometry_hash = fnv1a_i64(current_cell.local_geometry_hash, bbox.x0);
                    current_cell.local_geometry_hash = fnv1a_i64(current_cell.local_geometry_hash, bbox.y0);
                    current_cell.local_geometry_hash = fnv1a_i64(current_cell.local_geometry_hash, bbox.x1);
                    current_cell.local_geometry_hash = fnv1a_i64(current_cell.local_geometry_hash, bbox.y1);
                    current_cell.local_geometry_hash = fnv1a_update(current_cell.local_geometry_hash, data.data() + element_start, offset - element_start);
                }
            } else if (element_kind == 0x0A || element_kind == 0x0B) {
                if (detail) {
                    const auto xy = parse_xy(payload, payload_size);
                    if (!xy.empty()) {
                        pending_reference.ox = xy[0].first;
                        pending_reference.oy = xy[0].second;
                        pending_reference.cx = xy.size() > 1 ? xy[1].first : xy[0].first;
                        pending_reference.cy = xy.size() > 1 ? xy[1].second : xy[0].second;
                        pending_reference.rx = xy.size() > 2 ? xy[2].first : xy[0].first;
                        pending_reference.ry = xy.size() > 2 ? xy[2].second : xy[0].second;
                        pending_reference.has_xy = true;
                    }
                }
            }
            break;
        }
        case 0x11:  // ENDEL
            if (element_kind == 0x0A || element_kind == 0x0B) {
                if (pending_sname.empty()) {
                    throw std::runtime_error("reference missing SNAME");
                }
                pending_reference.child = pending_sname;
                pending_reference.raw_hash = fnv1a_update(1469598103934665603ull, data.data() + element_start, offset - element_start);
                current_cell.references.push_back(pending_reference);
            }
            element_kind = -1;
            pending_sname.clear();
            break;
        case 0x07:  // ENDSTR
            if (current_name.empty()) {
                throw std::runtime_error("cell missing STRNAME");
            }
            cells[current_name] = current_cell;
            current_name.clear();
            current_cell = Cell{};
            break;
        default:
            break;
        }
    }
    if (offset != data.size()) {
        throw std::runtime_error("trailing partial GDS record");
    }
    return cells;
}

std::string select_top(const std::map<std::string, Cell>& cells, const char* requested_top) {
    if (requested_top && requested_top[0]) {
        if (!cells.count(requested_top)) {
            throw std::runtime_error("requested top cell not found");
        }
        return requested_top;
    }
    std::set<std::string> referenced;
    for (const auto& item : cells) {
        for (const auto& reference : item.second.references) {
            referenced.insert(reference.child);
        }
    }
    std::vector<std::string> tops;
    for (const auto& item : cells) {
        if (!referenced.count(item.first)) {
            tops.push_back(item.first);
        }
    }
    if (tops.size() != 1) {
        throw std::runtime_error("expected exactly one top cell");
    }
    return tops.front();
}

std::int64_t div_floor(std::int64_t numerator, std::uint64_t denominator) {
    if (denominator == 0) {
        throw std::runtime_error("zero denominator");
    }
    if (numerator >= 0) {
        return numerator / static_cast<std::int64_t>(denominator);
    }
    return -static_cast<std::int64_t>((static_cast<std::uint64_t>(-numerator) + denominator - 1) / denominator);
}

FlattenedSummary flatten_cell(
    const std::string& name,
    const std::map<std::string, Cell>& cells,
    std::map<std::string, FlattenedSummary>& memo,
    std::set<std::string>& visiting) {
    if (memo.count(name)) {
        return memo.at(name);
    }
    if (visiting.count(name)) {
        throw std::runtime_error("cell reference cycle detected");
    }
    const auto found = cells.find(name);
    if (found == cells.end()) {
        throw std::runtime_error("unresolved reference");
    }
    visiting.insert(name);
    FlattenedSummary summary;
    for (const auto& item : found->second.layers) {
        add_layer_summary(summary.layers, item.first, item.second.count, item.second.bbox);
        summary.shape_count += item.second.count;
    }
    if (found->second.layers.empty()) {
        summary.shape_count = found->second.local_shape_count;
    }
    summary.geometry_hash = fnv1a_u64(summary.geometry_hash, found->second.local_shape_count);
    summary.geometry_hash = fnv1a_u64(summary.geometry_hash, found->second.local_geometry_hash);
    for (const auto& reference : found->second.references) {
        const auto child = flatten_cell(reference.child, cells, memo, visiting);
        const std::uint64_t columns = std::max<std::uint64_t>(1, reference.columns);
        const std::uint64_t rows = std::max<std::uint64_t>(1, reference.rows);
        const std::int64_t col_dx = div_floor(reference.cx - reference.ox, columns);
        const std::int64_t col_dy = div_floor(reference.cy - reference.oy, columns);
        const std::int64_t row_dx = div_floor(reference.rx - reference.ox, rows);
        const std::int64_t row_dy = div_floor(reference.ry - reference.oy, rows);
        for (std::uint64_t column = 0; column < columns; ++column) {
            for (std::uint64_t row = 0; row < rows; ++row) {
                const std::int64_t dx = reference.ox + static_cast<std::int64_t>(column) * col_dx + static_cast<std::int64_t>(row) * row_dx;
                const std::int64_t dy = reference.oy + static_cast<std::int64_t>(column) * col_dy + static_cast<std::int64_t>(row) * row_dy;
                for (const auto& item : child.layers) {
                    add_layer_summary(summary.layers, item.first, item.second.count, shifted_bbox(item.second.bbox, dx, dy));
                }
                if (summary.shape_count > std::numeric_limits<std::uint64_t>::max() - child.shape_count) {
                    throw std::overflow_error("flattened shape count overflow");
                }
                summary.shape_count += child.shape_count;
            }
        }
        summary.geometry_hash = fnv1a_update(summary.geometry_hash, reinterpret_cast<const unsigned char*>(reference.child.data()), reference.child.size());
        summary.geometry_hash = fnv1a_u64(summary.geometry_hash, reference.raw_hash);
        summary.geometry_hash = fnv1a_u64(summary.geometry_hash, child.geometry_hash);
    }
    visiting.erase(name);
    memo[name] = summary;
    return summary;
}

}  // namespace

extern "C" {

struct GdsGeometryLayerResult {
    std::int32_t layer;
    std::int32_t datatype;
    std::uint64_t count;
    std::int64_t x0;
    std::int64_t y0;
    std::int64_t x1;
    std::int64_t y1;
};

struct GdsGeometrySummaryResult {
    std::uint64_t flattened_shape_count;
    std::uint64_t layer_count;
    std::uint64_t geometry_hash;
    GdsGeometryLayerResult* layers;
};

struct GdsGeometryCompareResult {
    std::uint64_t old_flattened_shape_count;
    std::uint64_t new_flattened_shape_count;
    std::uint64_t old_layer_count;
    std::uint64_t new_layer_count;
    std::uint64_t old_geometry_hash;
    std::uint64_t new_geometry_hash;
    std::uint8_t summaries_match;
    std::uint8_t detail_computed;
    GdsGeometryLayerResult* old_layers;
    GdsGeometryLayerResult* new_layers;
};

const char* gds_geometry_last_error() {
    return g_last_error.c_str();
}

void gds_geometry_free_summary(GdsGeometrySummaryResult* result) {
    if (!result) {
        return;
    }
    std::free(result->layers);
    result->layers = nullptr;
    result->layer_count = 0;
    result->flattened_shape_count = 0;
    result->geometry_hash = 0;
}

void gds_geometry_free_compare(GdsGeometryCompareResult* result) {
    if (!result) {
        return;
    }
    std::free(result->old_layers);
    std::free(result->new_layers);
    result->old_layers = nullptr;
    result->new_layers = nullptr;
    result->old_layer_count = 0;
    result->new_layer_count = 0;
}

void fill_layer_array(const std::map<LayerKey, LayerSummary>& layers, GdsGeometryLayerResult*& out_layers, std::uint64_t& out_count) {
    out_count = layers.size();
    out_layers = nullptr;
    if (layers.empty()) {
        return;
    }
    out_layers = static_cast<GdsGeometryLayerResult*>(std::malloc(sizeof(GdsGeometryLayerResult) * layers.size()));
    if (!out_layers) {
        throw std::bad_alloc();
    }
    std::size_t index = 0;
    for (const auto& item : layers) {
        out_layers[index++] = GdsGeometryLayerResult{
            item.first.layer,
            item.first.datatype,
            item.second.count,
            item.second.bbox.x0,
            item.second.bbox.y0,
            item.second.bbox.x1,
            item.second.bbox.y1,
        };
    }
}

int gds_geometry_summary_impl(const char* path, const char* top_name, GdsGeometrySummaryResult* out, bool detail) {
    try {
        if (!path || !out) {
            throw std::runtime_error("invalid arguments");
        }
        *out = GdsGeometrySummaryResult{};
        const auto cells = parse_gds(path, detail);
        if (cells.empty()) {
            throw std::runtime_error("no cells found");
        }
        const auto top = select_top(cells, top_name);
        std::map<std::string, FlattenedSummary> memo;
        std::set<std::string> visiting;
        const auto summary = flatten_cell(top, cells, memo, visiting);
        out->flattened_shape_count = summary.shape_count;
        out->layer_count = detail ? summary.layers.size() : 0;
        out->geometry_hash = detail ? summary.geometry_hash : 0;
        if (detail) {
            fill_layer_array(summary.layers, out->layers, out->layer_count);
        }
        g_last_error.clear();
        return 0;
    } catch (const std::exception& exc) {
        g_last_error = exc.what();
        if (out) {
            gds_geometry_free_summary(out);
        }
        return 1;
    } catch (...) {
        g_last_error = "unknown native geometry summary error";
        if (out) {
            gds_geometry_free_summary(out);
        }
        return 1;
    }
}

int gds_geometry_summary(const char* path, const char* top_name, GdsGeometrySummaryResult* out) {
    return gds_geometry_summary_impl(path, top_name, out, true);
}

int gds_geometry_summary_counts_only(const char* path, const char* top_name, GdsGeometrySummaryResult* out) {
    return gds_geometry_summary_impl(path, top_name, out, false);
}

int gds_geometry_compare(
    const char* old_path,
    const char* new_path,
    const char* old_top_name,
    const char* new_top_name,
    GdsGeometryCompareResult* out) {
    try {
        if (!old_path || !new_path || !out) {
            throw std::runtime_error("invalid arguments");
        }
        *out = GdsGeometryCompareResult{};

        const auto old_count_cells = parse_gds(old_path, false);
        const auto new_count_cells = parse_gds(new_path, false);
        const auto old_top = select_top(old_count_cells, old_top_name);
        const auto new_top = select_top(new_count_cells, new_top_name);
        std::map<std::string, FlattenedSummary> old_count_memo;
        std::map<std::string, FlattenedSummary> new_count_memo;
        std::set<std::string> old_count_visiting;
        std::set<std::string> new_count_visiting;
        const auto old_count_summary = flatten_cell(old_top, old_count_cells, old_count_memo, old_count_visiting);
        const auto new_count_summary = flatten_cell(new_top, new_count_cells, new_count_memo, new_count_visiting);
        out->old_flattened_shape_count = old_count_summary.shape_count;
        out->new_flattened_shape_count = new_count_summary.shape_count;
        if (old_count_summary.shape_count != new_count_summary.shape_count) {
            out->summaries_match = 0;
            out->detail_computed = 0;
            g_last_error.clear();
            return 0;
        }

        const auto old_detail_cells = parse_gds(old_path, true);
        const auto new_detail_cells = parse_gds(new_path, true);
        const auto old_detail_top = select_top(old_detail_cells, old_top_name);
        const auto new_detail_top = select_top(new_detail_cells, new_top_name);
        std::map<std::string, FlattenedSummary> old_detail_memo;
        std::map<std::string, FlattenedSummary> new_detail_memo;
        std::set<std::string> old_detail_visiting;
        std::set<std::string> new_detail_visiting;
        const auto old_detail = flatten_cell(old_detail_top, old_detail_cells, old_detail_memo, old_detail_visiting);
        const auto new_detail = flatten_cell(new_detail_top, new_detail_cells, new_detail_memo, new_detail_visiting);
        out->old_flattened_shape_count = old_detail.shape_count;
        out->new_flattened_shape_count = new_detail.shape_count;
        out->old_geometry_hash = old_detail.geometry_hash;
        out->new_geometry_hash = new_detail.geometry_hash;
        out->detail_computed = 1;
        out->summaries_match = (old_detail.shape_count == new_detail.shape_count && old_detail.geometry_hash == new_detail.geometry_hash) ? 1 : 0;
        fill_layer_array(old_detail.layers, out->old_layers, out->old_layer_count);
        fill_layer_array(new_detail.layers, out->new_layers, out->new_layer_count);
        g_last_error.clear();
        return 0;
    } catch (const std::exception& exc) {
        g_last_error = exc.what();
        if (out) {
            gds_geometry_free_compare(out);
        }
        return 1;
    } catch (...) {
        g_last_error = "unknown native geometry compare error";
        if (out) {
            gds_geometry_free_compare(out);
        }
        return 1;
    }
}

}  // extern "C"
