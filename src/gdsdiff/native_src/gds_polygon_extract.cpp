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
#include <utility>
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
    unsigned quarter_turns = 0;
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
    const auto value = (static_cast<std::uint32_t>(p[0]) << 24) | (static_cast<std::uint32_t>(p[1]) << 16)
        | (static_cast<std::uint32_t>(p[2]) << 8) | static_cast<std::uint32_t>(p[3]);
    return static_cast<std::int32_t>(value & 0x80000000u ? static_cast<std::int64_t>(value) - 0x100000000ll : value);
}

unsigned read_quarter_turns(const unsigned char* p) {
    std::uint64_t mantissa = 0;
    for (int i = 1; i < 8; ++i) {
        mantissa = (mantissa << 8) | p[i];
    }
    if (mantissa == 0 && p[0] == 0) return 0;
    if (p[1] < 0x10) throw std::runtime_error("invalid ANGLE REAL8 encoding");

    // REAL8 is a 56-bit fraction with a hexadecimal exponent. Reduce the
    // integer value modulo 360 without rounding nearly orthogonal angles.
    const int shift = 4 * (static_cast<int>(p[0] & 0x7f) - 64) - 56;
    unsigned degrees = 0;
    if (shift < 0) {
        if (shift <= -56 || (mantissa & ((1ull << -shift) - 1))) {
            throw std::runtime_error("nonorthogonal reference ANGLE is not supported by native polygon extraction");
        }
        degrees = static_cast<unsigned>((mantissa >> -shift) % 360);
    } else {
        degrees = static_cast<unsigned>(mantissa % 360);
        for (int i = 0; i < shift; ++i) degrees = (degrees * 2) % 360;
    }
    if (degrees % 90 != 0) {
        throw std::runtime_error("nonorthogonal reference ANGLE is not supported by native polygon extraction");
    }
    return ((p[0] & 0x80) ? (360 - degrees) % 360 : degrees) / 90;
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

std::uint64_t fnv1a_u64(std::uint64_t hash, std::uint64_t value) {
    constexpr std::uint64_t prime = 1099511628211ull;
    for (int i = 0; i < 8; ++i) {
        hash ^= (value >> (i * 8)) & 0xffu;
        hash *= prime;
    }
    return hash;
}

bool point_less(const Point& left, const Point& right) {
    return left.x < right.x || (left.x == right.x && left.y < right.y);
}

std::size_t sequence_index(std::size_t count, std::size_t candidate, std::size_t step, bool forward) {
    if (forward) {
        return (candidate + step) % count;
    }
    return (candidate + count - (step % count)) % count;
}

bool sequence_less(
    const std::vector<Point>& points,
    std::size_t left_candidate,
    bool left_forward,
    std::size_t right_candidate,
    bool right_forward) {
    const std::size_t count = points.size();
    for (std::size_t step = 0; step < count; ++step) {
        const auto& left = points[sequence_index(count, left_candidate, step, left_forward)];
        const auto& right = points[sequence_index(count, right_candidate, step, right_forward)];
        if (point_less(left, right)) {
            return true;
        }
        if (point_less(right, left)) {
            return false;
        }
    }
    return false;
}

bool is_axis_aligned_bbox_rectangle(const Polygon& polygon) {
    if (polygon.points.size() != 4 || polygon.bbox.x0 >= polygon.bbox.x1 || polygon.bbox.y0 >= polygon.bbox.y1) {
        return false;
    }
    bool corners[4] = {false, false, false, false};
    for (const auto& point : polygon.points) {
        int corner = -1;
        if (point.x == polygon.bbox.x0 && point.y == polygon.bbox.y0) corner = 0;
        if (point.x == polygon.bbox.x0 && point.y == polygon.bbox.y1) corner = 1;
        if (point.x == polygon.bbox.x1 && point.y == polygon.bbox.y0) corner = 2;
        if (point.x == polygon.bbox.x1 && point.y == polygon.bbox.y1) corner = 3;
        if (corner < 0 || corners[corner]) {
            return false;
        }
        corners[corner] = true;
    }
    return corners[0] && corners[1] && corners[2] && corners[3];
}

std::pair<std::uint64_t, std::uint64_t> canonical_polygon_hash(const Polygon& polygon) {
    std::uint64_t h0 = 1469598103934665603ull;
    std::uint64_t h1 = 1099511628211ull;
    const auto& points = polygon.points;
    if (polygon.flags == 0 && is_axis_aligned_bbox_rectangle(polygon)) {
        h0 = fnv1a_u64(fnv1a_u64(h0, 2), 0);
        h0 = fnv1a_u64(fnv1a_u64(h0, static_cast<std::uint64_t>(polygon.bbox.x0)), static_cast<std::uint64_t>(polygon.bbox.y0));
        h0 = fnv1a_u64(fnv1a_u64(h0, static_cast<std::uint64_t>(polygon.bbox.x1)), static_cast<std::uint64_t>(polygon.bbox.y1));
        h1 = fnv1a_u64(fnv1a_u64(h1, static_cast<std::uint64_t>(polygon.bbox.y1)), static_cast<std::uint64_t>(polygon.bbox.x1));
        h1 = fnv1a_u64(fnv1a_u64(h1, static_cast<std::uint64_t>(polygon.bbox.y0)), static_cast<std::uint64_t>(polygon.bbox.x0));
        return {h0, h1};
    }
    std::size_t best_candidate = 0;
    bool best_forward = true;
    if (polygon.flags == 0 && !points.empty()) {
        for (std::size_t candidate = 1; candidate < points.size(); ++candidate) {
            if (sequence_less(points, candidate, true, best_candidate, best_forward)) {
                best_candidate = candidate;
                best_forward = true;
            }
        }
        for (std::size_t candidate = 0; candidate < points.size(); ++candidate) {
            if (sequence_less(points, candidate, false, best_candidate, best_forward)) {
                best_candidate = candidate;
                best_forward = false;
            }
        }
    }
    for (std::size_t step = 0; step < points.size(); ++step) {
        const std::size_t index = polygon.flags == 0 ? sequence_index(points.size(), best_candidate, step, best_forward) : step;
        const auto& point = points[index];
        const auto x = static_cast<std::uint64_t>(point.x);
        const auto y = static_cast<std::uint64_t>(point.y);
        h0 = fnv1a_u64(fnv1a_u64(h0, x), y);
        h1 = fnv1a_u64(fnv1a_u64(h1, y), x);
    }
    return {h0, h1};
}

std::map<std::string, Cell> parse_gds(const char* path, bool& oracle_reuse_eligible) {
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
    unsigned pending_transform_records = 0;
    std::vector<Point> pending_xy;
    bool saw_endlib = false;

    while (offset + 4 <= data.size()) {
        const std::uint16_t record_len = read_u16(data.data() + offset);
        const int record_type = data[offset + 2];
        const int data_type = data[offset + 3];
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
            // The legacy AREF reader applies integer floor to each pitch.
            if (record_type == 0x0B) oracle_reuse_eligible = false;
            element_kind = record_type;
            pending_reference = Reference{};
            pending_transform_records = 0;
            pending_sname.clear();
            pending_xy.clear();
            break;
        case 0x0D:  // LAYER
            if (element_kind == 0x08 && payload_size >= 2) {
                pending_layer = read_i16(payload);
                if (data_type != 0x02 || payload_size != 2) oracle_reuse_eligible = false;
            }
            break;
        case 0x0E:  // DATATYPE
            if (element_kind == 0x08 && payload_size >= 2) {
                pending_datatype = read_i16(payload);
                if (data_type != 0x02 || payload_size != 2) oracle_reuse_eligible = false;
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
            if (data_type != 0x03) oracle_reuse_eligible = false;
            pending_xy = parse_xy(payload, payload_size);
            break;
        case 0x1A:  // STRANS
        case 0x1B:  // MAG
        case 0x1C:  // ANGLE
            if (element_kind == 0x0A || element_kind == 0x0B) {
                const bool is_strans = record_type == 0x1A;
                const unsigned bit = 1u << (record_type - 0x1A);
                const char* name = is_strans ? "STRANS" : record_type == 0x1B ? "MAG" : "ANGLE";
                if (data_type != (is_strans ? 0x01 : 0x05) || payload_size != (is_strans ? 2u : 8u)) {
                    throw std::runtime_error(std::string("invalid ") + name + " record type or length");
                }
                if (pending_transform_records & bit) {
                    throw std::runtime_error(std::string("duplicate reference ") + name + " record");
                }
                pending_transform_records |= bit;
                if (is_strans) {
                    if (read_u16(payload) != 0) {
                        throw std::runtime_error("nonzero reference STRANS flags are not supported by native polygon extraction");
                    }
                } else if (record_type == 0x1B) {
                    constexpr unsigned char unit_magnification[8] = {0x41, 0x10, 0, 0, 0, 0, 0, 0};
                    if (std::memcmp(payload, unit_magnification, sizeof(unit_magnification)) != 0) {
                        throw std::runtime_error("non-unit reference MAG is not supported by native polygon extraction");
                    }
                } else {
                    pending_reference.quarter_turns = read_quarter_turns(payload);
                    if ((payload[0] & 0x7f) > 0x43) oracle_reuse_eligible = false;
                    if (element_kind == 0x0B && pending_reference.quarter_turns != 0) {
                        throw std::runtime_error("rotated AREF is not supported by native polygon extraction");
                    }
                }
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
                if (pending_xy.size() < 4 || pending_xy.front().x != pending_xy.back().x ||
                    pending_xy.front().y != pending_xy.back().y) {
                    oracle_reuse_eligible = false;
                }
            } else if (element_kind == 0x0A || element_kind == 0x0B) {
                if (pending_sname.empty()) {
                    throw std::runtime_error("reference missing SNAME");
                }
                if (pending_xy.empty()) {
                    throw std::runtime_error("reference missing XY");
                }
                if (element_kind == 0x0A && pending_xy.size() != 1) oracle_reuse_eligible = false;
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
            if (cells.count(current_name)) oracle_reuse_eligible = false;
            cells[current_name] = current_cell;
            current_name.clear();
            current_cell = Cell{};
            break;
        case 0x04:  // ENDLIB
            saw_endlib = true;
            if (offset != data.size()) oracle_reuse_eligible = false;
            break;
        case 0x00:  // HEADER
        case 0x01:  // BGNLIB
        case 0x02:  // LIBNAME
        case 0x03:  // UNITS
        case 0x0C:  // TEXT
        case 0x16:  // TEXTTYPE
        case 0x17:  // PRESENTATION
        case 0x19:  // STRING
        case 0x26:  // ELFLAGS
        case 0x2B:  // PROPATTR
        case 0x2C:  // PROPVALUE
        case 0x2F:  // PLEX
            break;
        default:
            // An ignored record can carry geometry in another reader (e.g. BOX).
            oracle_reuse_eligible = false;
            break;
        }
    }
    if (offset != data.size()) {
        throw std::runtime_error("trailing partial GDS record");
    }
    if (!saw_endlib) oracle_reuse_eligible = false;
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

std::int64_t checked_coordinate(__int128 value) {
    if (value < std::numeric_limits<std::int64_t>::min() || value > std::numeric_limits<std::int64_t>::max()) {
        throw std::overflow_error("transformed coordinate exceeds signed 64-bit DBU range");
    }
    return static_cast<std::int64_t>(value);
}

struct Placement {
    unsigned quarter_turns = 0;
    Point origin;

    Point apply(Point point) const {
        const auto x = static_cast<__int128>(point.x);
        const auto y = static_cast<__int128>(point.y);
        switch (quarter_turns) {
        case 1: return {checked_coordinate(origin.x - y), checked_coordinate(origin.y + x)};
        case 2: return {checked_coordinate(origin.x - x), checked_coordinate(origin.y - y)};
        case 3: return {checked_coordinate(origin.x + y), checked_coordinate(origin.y - x)};
        default: return {checked_coordinate(origin.x + x), checked_coordinate(origin.y + y)};
        }
    }
};

void append_placed_polygon(std::vector<Polygon>& out, const Polygon& polygon, const Placement& placement) {
    // A rotation/translation preserves vertex distinctness and signed area.
    Polygon placed{polygon.layer, polygon.datatype, {}, {}, polygon.flags};
    placed.points.reserve(polygon.points.size());
    for (const auto& point : polygon.points) {
        placed.points.push_back(placement.apply(point));
    }
    const auto low = placement.apply({polygon.bbox.x0, polygon.bbox.y0});
    const auto high = placement.apply({polygon.bbox.x1, polygon.bbox.y1});
    placed.bbox = BBox{std::min(low.x, high.x), std::min(low.y, high.y), std::max(low.x, high.x), std::max(low.y, high.y)};
    out.push_back(std::move(placed));
}

void flatten_cell(
    const std::string& name,
    const std::map<std::string, Cell>& cells,
    std::vector<Polygon>& out,
    std::set<std::string>& visiting,
    const Placement& placement) {
    if (visiting.count(name)) {
        throw std::runtime_error("cell reference cycle detected");
    }
    const auto found = cells.find(name);
    if (found == cells.end()) {
        throw std::runtime_error("unresolved reference");
    }
    visiting.insert(name);
    for (const auto& polygon : found->second.polygons) {
        append_placed_polygon(out, polygon, placement);
    }
    for (const auto& reference : found->second.references) {
        const std::uint64_t columns = std::max<std::uint64_t>(1, reference.columns);
        const std::uint64_t rows = std::max<std::uint64_t>(1, reference.rows);
        if (placement.quarter_turns != 0 &&
            ((columns > 1 && ((reference.cx - reference.ox) % static_cast<std::int64_t>(columns) != 0 ||
                             (reference.cy - reference.oy) % static_cast<std::int64_t>(columns) != 0)) ||
             (rows > 1 && ((reference.rx - reference.ox) % static_cast<std::int64_t>(rows) != 0 ||
                          (reference.ry - reference.oy) % static_cast<std::int64_t>(rows) != 0)))) {
            throw std::runtime_error("nonintegral AREF pitch under a rotated ancestor is not supported by native polygon extraction");
        }
        const std::int64_t col_dx = div_floor(reference.cx - reference.ox, columns);
        const std::int64_t col_dy = div_floor(reference.cy - reference.oy, columns);
        const std::int64_t row_dx = div_floor(reference.rx - reference.ox, rows);
        const std::int64_t row_dy = div_floor(reference.ry - reference.oy, rows);
        for (std::uint64_t column = 0; column < columns; ++column) {
            for (std::uint64_t row = 0; row < rows; ++row) {
                const std::int64_t child_x = reference.ox + static_cast<std::int64_t>(column) * col_dx
                    + static_cast<std::int64_t>(row) * row_dx;
                const std::int64_t child_y = reference.oy + static_cast<std::int64_t>(column) * col_dy
                    + static_cast<std::int64_t>(row) * row_dy;
                const Placement child{(placement.quarter_turns + reference.quarter_turns) % 4, placement.apply({child_x, child_y})};
                flatten_cell(reference.child, cells, out, visiting, child);
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
    std::uint64_t* hashes;
    std::uint32_t oracle_reuse_eligible;
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
    std::free(result->hashes);
    *result = GdsPolygonExtractResult{};
}

int gds_polygon_extract(const char* path, const char* top_name, GdsPolygonExtractResult* out) {
    try {
        if (!path || !out) {
            throw std::runtime_error("invalid arguments");
        }
        *out = GdsPolygonExtractResult{};
        bool oracle_reuse_eligible = true;
        const auto cells = parse_gds(path, oracle_reuse_eligible);
        if (cells.empty()) {
            throw std::runtime_error("no cells found");
        }
        const auto top = select_top(cells, top_name);
        std::vector<Polygon> polygons;
        std::set<std::string> visiting;
        flatten_cell(top, cells, polygons, visiting, Placement{});

        std::uint64_t vertex_count = 0;
        for (const auto& polygon : polygons) {
            if (vertex_count > std::numeric_limits<std::uint64_t>::max() - polygon.points.size()) {
                throw std::overflow_error("vertex count overflow");
            }
            vertex_count += static_cast<std::uint64_t>(polygon.points.size());
            constexpr std::int64_t max_oracle_reuse_coordinate = 1ll << 40;
            if (polygon.bbox.x0 < -max_oracle_reuse_coordinate || polygon.bbox.y0 < -max_oracle_reuse_coordinate ||
                polygon.bbox.x1 > max_oracle_reuse_coordinate || polygon.bbox.y1 > max_oracle_reuse_coordinate) {
                oracle_reuse_eligible = false;
            }
        }
        out->oracle_reuse_eligible = oracle_reuse_eligible ? 1 : 0;
        out->polygon_count = static_cast<std::uint64_t>(polygons.size());
        out->vertex_count = vertex_count;
        out->polygon_offsets = static_cast<std::uint64_t*>(std::malloc(sizeof(std::uint64_t) * (polygons.size() + 1)));
        out->vertices_xy = static_cast<std::int64_t*>(std::malloc(sizeof(std::int64_t) * vertex_count * 2));
        out->layers = static_cast<std::int32_t*>(std::malloc(sizeof(std::int32_t) * polygons.size()));
        out->datatypes = static_cast<std::int32_t*>(std::malloc(sizeof(std::int32_t) * polygons.size()));
        out->bboxes = static_cast<std::int64_t*>(std::malloc(sizeof(std::int64_t) * polygons.size() * 4));
        out->flags = static_cast<std::uint32_t*>(std::malloc(sizeof(std::uint32_t) * polygons.size()));
        out->source_ids = static_cast<std::uint64_t*>(std::malloc(sizeof(std::uint64_t) * polygons.size()));
        out->hashes = static_cast<std::uint64_t*>(std::malloc(sizeof(std::uint64_t) * polygons.size() * 2));
        if (!out->polygon_offsets || (!out->vertices_xy && vertex_count) || (!out->layers && !polygons.empty())
            || (!out->datatypes && !polygons.empty()) || (!out->bboxes && !polygons.empty())
            || (!out->flags && !polygons.empty()) || (!out->source_ids && !polygons.empty())
            || (!out->hashes && !polygons.empty())) {
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
            const auto hash = canonical_polygon_hash(polygon);
            out->hashes[index * 2 + 0] = hash.first;
            out->hashes[index * 2 + 1] = hash.second;
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
