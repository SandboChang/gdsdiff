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

struct Point {
    std::int64_t x = 0;
    std::int64_t y = 0;
};

struct BBox {
    std::int64_t x0 = 0;
    std::int64_t y0 = 0;
    std::int64_t x1 = 0;
    std::int64_t y1 = 0;
};

struct Polygon {
    std::int32_t layer = 0;
    std::int32_t datatype = 0;
    std::vector<Point> points;
    BBox bbox;
    std::uint32_t flags = 0;
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
};

struct Cell {
    std::vector<Polygon> polygons;
    std::vector<Reference> references;
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

std::vector<Point> parse_xy(const unsigned char* payload, std::size_t payload_size) {
    if (payload_size % 8 != 0) {
        throw std::runtime_error("invalid XY payload length");
    }
    std::vector<Point> points;
    points.reserve(payload_size / 8);
    for (std::size_t i = 0; i < payload_size; i += 8) {
        points.push_back(Point{read_i32(payload + i), read_i32(payload + i + 4)});
    }
    return points;
}

std::vector<Point> normalize_points(std::vector<Point> points) {
    if (points.size() > 1 && points.front().x == points.back().x && points.front().y == points.back().y) {
        points.pop_back();
    }
    std::vector<Point> normalized;
    normalized.reserve(points.size());
    for (const auto& point : points) {
        if (normalized.empty() || normalized.back().x != point.x || normalized.back().y != point.y) {
            normalized.push_back(point);
        }
    }
    if (normalized.size() > 1 && normalized.front().x == normalized.back().x && normalized.front().y == normalized.back().y) {
        normalized.pop_back();
    }
    return normalized;
}

BBox bbox_from_points(const std::vector<Point>& points) {
    if (points.empty()) {
        throw std::runtime_error("cannot compute bbox of empty polygon");
    }
    BBox bbox{points[0].x, points[0].y, points[0].x, points[0].y};
    for (const auto& point : points) {
        bbox.x0 = std::min(bbox.x0, point.x);
        bbox.y0 = std::min(bbox.y0, point.y);
        bbox.x1 = std::max(bbox.x1, point.x);
        bbox.y1 = std::max(bbox.y1, point.y);
    }
    return bbox;
}

std::uint32_t polygon_flags(const std::vector<Point>& points) {
    std::uint32_t flags = 0;
    std::set<std::pair<std::int64_t, std::int64_t>> unique_points;
    for (const auto& point : points) {
        unique_points.insert({point.x, point.y});
    }
    if (unique_points.size() < 3) {
        flags |= 1u;
    }
    __int128 twice_area = 0;
    if (points.size() >= 3) {
        for (std::size_t index = 0; index < points.size(); ++index) {
            const auto& point = points[index];
            const auto& next = points[(index + 1) % points.size()];
            twice_area += static_cast<__int128>(point.x) * static_cast<__int128>(next.y)
                - static_cast<__int128>(next.x) * static_cast<__int128>(point.y);
        }
    }
    if (twice_area == 0) {
        flags |= 2u;
    }
    return flags;
}

std::map<std::string, Cell> parse_gds(const char* path) {
    const auto data = read_file(path);
    std::map<std::string, Cell> cells;
    std::size_t offset = 0;
    std::string current_name;
    Cell current_cell;
    int element_kind = -1;
    std::int32_t pending_layer = 0;
    std::int32_t pending_datatype = 0;
    std::string pending_sname;
    Reference pending_reference;
    std::vector<Point> pending_xy;

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
            element_kind = record_type;
            pending_layer = 0;
            pending_datatype = 0;
            pending_xy.clear();
            break;
        case 0x09:  // PATH
            throw std::runtime_error("PATH records are not supported by native polygon extraction");
        case 0x0A:  // SREF
        case 0x0B:  // AREF
            element_kind = record_type;
            pending_reference = Reference{};
            pending_sname.clear();
            pending_xy.clear();
            break;
        case 0x0D:  // LAYER
            if (element_kind == 0x08 && payload_size >= 2) {
                pending_layer = read_i16(payload);
            }
            break;
        case 0x0E:  // DATATYPE
            if (element_kind == 0x08 && payload_size >= 2) {
                pending_datatype = read_i16(payload);
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
        case 0x10:  // XY
            pending_xy = parse_xy(payload, payload_size);
            break;
        case 0x1A:  // STRANS
        case 0x1B:  // MAG
        case 0x1C:  // ANGLE
            if (element_kind == 0x0A || element_kind == 0x0B) {
                throw std::runtime_error("transformed references are not supported by native polygon extraction");
            }
            break;
        case 0x11:  // ENDEL
            if (element_kind == 0x08) {
                auto normalized = normalize_points(pending_xy);
                if (!normalized.empty()) {
                    current_cell.polygons.push_back(Polygon{
                        pending_layer,
                        pending_datatype,
                        normalized,
                        bbox_from_points(normalized),
                        polygon_flags(normalized),
                    });
                }
            } else if (element_kind == 0x0A || element_kind == 0x0B) {
                if (pending_sname.empty()) {
                    throw std::runtime_error("reference missing SNAME");
                }
                if (pending_xy.empty()) {
                    throw std::runtime_error("reference missing XY");
                }
                pending_reference.child = pending_sname;
                pending_reference.ox = pending_xy[0].x;
                pending_reference.oy = pending_xy[0].y;
                pending_reference.cx = pending_xy.size() > 1 ? pending_xy[1].x : pending_xy[0].x;
                pending_reference.cy = pending_xy.size() > 1 ? pending_xy[1].y : pending_xy[0].y;
                pending_reference.rx = pending_xy.size() > 2 ? pending_xy[2].x : pending_xy[0].x;
                pending_reference.ry = pending_xy.size() > 2 ? pending_xy[2].y : pending_xy[0].y;
                current_cell.references.push_back(pending_reference);
            }
            element_kind = -1;
            pending_sname.clear();
            pending_xy.clear();
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

void append_shifted_polygon(std::vector<Polygon>& out, const Polygon& polygon, std::int64_t dx, std::int64_t dy) {
    Polygon shifted;
    shifted.layer = polygon.layer;
    shifted.datatype = polygon.datatype;
    shifted.flags = polygon.flags;
    shifted.points.reserve(polygon.points.size());
    for (const auto& point : polygon.points) {
        shifted.points.push_back(Point{point.x + dx, point.y + dy});
    }
    shifted.bbox = BBox{polygon.bbox.x0 + dx, polygon.bbox.y0 + dy, polygon.bbox.x1 + dx, polygon.bbox.y1 + dy};
    out.push_back(std::move(shifted));
}

void flatten_cell(
    const std::string& name,
    const std::map<std::string, Cell>& cells,
    std::vector<Polygon>& out,
    std::set<std::string>& visiting,
    std::int64_t dx,
    std::int64_t dy) {
    if (visiting.count(name)) {
        throw std::runtime_error("cell reference cycle detected");
    }
    const auto found = cells.find(name);
    if (found == cells.end()) {
        throw std::runtime_error("unresolved reference");
    }
    visiting.insert(name);
    for (const auto& polygon : found->second.polygons) {
        append_shifted_polygon(out, polygon, dx, dy);
    }
    for (const auto& reference : found->second.references) {
        const std::uint64_t columns = std::max<std::uint64_t>(1, reference.columns);
        const std::uint64_t rows = std::max<std::uint64_t>(1, reference.rows);
        const std::int64_t col_dx = div_floor(reference.cx - reference.ox, columns);
        const std::int64_t col_dy = div_floor(reference.cy - reference.oy, columns);
        const std::int64_t row_dx = div_floor(reference.rx - reference.ox, rows);
        const std::int64_t row_dy = div_floor(reference.ry - reference.oy, rows);
        for (std::uint64_t column = 0; column < columns; ++column) {
            for (std::uint64_t row = 0; row < rows; ++row) {
                const std::int64_t child_dx = dx + reference.ox + static_cast<std::int64_t>(column) * col_dx
                    + static_cast<std::int64_t>(row) * row_dx;
                const std::int64_t child_dy = dy + reference.oy + static_cast<std::int64_t>(column) * col_dy
                    + static_cast<std::int64_t>(row) * row_dy;
                flatten_cell(reference.child, cells, out, visiting, child_dx, child_dy);
            }
        }
    }
    visiting.erase(name);
}

}  // namespace

extern "C" {

struct GdsPolygonExtractResult {
    std::uint64_t polygon_count;
    std::uint64_t vertex_count;
    std::int64_t* vertices_xy;
    std::uint64_t* polygon_offsets;
    std::int32_t* layers;
    std::int32_t* datatypes;
    std::int64_t* bboxes;
    std::uint32_t* flags;
    std::uint64_t* source_ids;
};

const char* gds_polygon_extract_last_error() {
    return g_last_error.c_str();
}

void gds_polygon_extract_free(GdsPolygonExtractResult* result) {
    if (!result) {
        return;
    }
    std::free(result->vertices_xy);
    std::free(result->polygon_offsets);
    std::free(result->layers);
    std::free(result->datatypes);
    std::free(result->bboxes);
    std::free(result->flags);
    std::free(result->source_ids);
    *result = GdsPolygonExtractResult{};
}

int gds_polygon_extract(const char* path, const char* top_name, GdsPolygonExtractResult* out) {
    try {
        if (!path || !out) {
            throw std::runtime_error("invalid arguments");
        }
        *out = GdsPolygonExtractResult{};
        const auto cells = parse_gds(path);
        if (cells.empty()) {
            throw std::runtime_error("no cells found");
        }
        const auto top = select_top(cells, top_name);
        std::vector<Polygon> polygons;
        std::set<std::string> visiting;
        flatten_cell(top, cells, polygons, visiting, 0, 0);

        std::uint64_t vertex_count = 0;
        for (const auto& polygon : polygons) {
            if (vertex_count > std::numeric_limits<std::uint64_t>::max() - polygon.points.size()) {
                throw std::overflow_error("vertex count overflow");
            }
            vertex_count += static_cast<std::uint64_t>(polygon.points.size());
        }
        out->polygon_count = static_cast<std::uint64_t>(polygons.size());
        out->vertex_count = vertex_count;
        out->polygon_offsets = static_cast<std::uint64_t*>(std::malloc(sizeof(std::uint64_t) * (polygons.size() + 1)));
        out->vertices_xy = static_cast<std::int64_t*>(std::malloc(sizeof(std::int64_t) * vertex_count * 2));
        out->layers = static_cast<std::int32_t*>(std::malloc(sizeof(std::int32_t) * polygons.size()));
        out->datatypes = static_cast<std::int32_t*>(std::malloc(sizeof(std::int32_t) * polygons.size()));
        out->bboxes = static_cast<std::int64_t*>(std::malloc(sizeof(std::int64_t) * polygons.size() * 4));
        out->flags = static_cast<std::uint32_t*>(std::malloc(sizeof(std::uint32_t) * polygons.size()));
        out->source_ids = static_cast<std::uint64_t*>(std::malloc(sizeof(std::uint64_t) * polygons.size()));
        if (!out->polygon_offsets || (!out->vertices_xy && vertex_count) || (!out->layers && !polygons.empty())
            || (!out->datatypes && !polygons.empty()) || (!out->bboxes && !polygons.empty())
            || (!out->flags && !polygons.empty()) || (!out->source_ids && !polygons.empty())) {
            throw std::bad_alloc();
        }
        std::uint64_t vertex_offset = 0;
        out->polygon_offsets[0] = 0;
        for (std::size_t index = 0; index < polygons.size(); ++index) {
            const auto& polygon = polygons[index];
            out->layers[index] = polygon.layer;
            out->datatypes[index] = polygon.datatype;
            out->bboxes[index * 4 + 0] = polygon.bbox.x0;
            out->bboxes[index * 4 + 1] = polygon.bbox.y0;
            out->bboxes[index * 4 + 2] = polygon.bbox.x1;
            out->bboxes[index * 4 + 3] = polygon.bbox.y1;
            out->flags[index] = polygon.flags;
            out->source_ids[index] = static_cast<std::uint64_t>(index);
            for (const auto& point : polygon.points) {
                out->vertices_xy[vertex_offset * 2 + 0] = point.x;
                out->vertices_xy[vertex_offset * 2 + 1] = point.y;
                ++vertex_offset;
            }
            out->polygon_offsets[index + 1] = vertex_offset;
        }
        g_last_error.clear();
        return 0;
    } catch (const std::exception& exc) {
        g_last_error = exc.what();
        if (out) {
            gds_polygon_extract_free(out);
        }
        return 1;
    } catch (...) {
        g_last_error = "unknown native polygon extract error";
        if (out) {
            gds_polygon_extract_free(out);
        }
        return 1;
    }
}

}  // extern "C"
