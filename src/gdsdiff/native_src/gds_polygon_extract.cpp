#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

thread_local std::string g_last_error;
using Clock = std::chrono::steady_clock;

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

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

using Layer = std::pair<std::int32_t, std::int32_t>;
constexpr auto kUnselectedLayer = std::numeric_limits<std::uint32_t>::max();

struct LocalPoint {
    std::int32_t x;
    std::int32_t y;
};

bool operator==(LocalPoint a, LocalPoint b) {
    return a.x == b.x && a.y == b.y;
}

struct Polygon {
    std::size_t xy_offset;
    std::size_t point_offset;
    std::uint32_t point_count;
    Layer layer;
    std::uint32_t layer_id = kUnselectedLayer;
    std::uint32_t flags = 0;
    LocalPoint low{};
    LocalPoint high{};
};

struct PolygonBlock {
    std::size_t begin;
    std::size_t end;
    std::uint64_t polygon_count = 0;
    std::uint64_t vertex_count = 0;
};

struct Cell;

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
    Cell* target = nullptr;
};

struct Cell {
    std::vector<Polygon> polygons;
    std::vector<Reference> references;
    std::vector<PolygonBlock> blocks;
    BBox bbox;
    bool visiting = false;
    bool visited = false;
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

struct FileBytes {
    std::unique_ptr<unsigned char[]> data;
    std::size_t size;
};

FileBytes read_file(const char* path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("unable to open GDS file");
    }
    const std::streamoff size = in.tellg();
    if (size < 0) {
        throw std::runtime_error("unable to determine GDS file size");
    }
    if (static_cast<std::uintmax_t>(size) > std::numeric_limits<std::size_t>::max() ||
        size > std::numeric_limits<std::streamsize>::max()) {
        throw std::overflow_error("GDS file is too large");
    }
    FileBytes bytes{std::unique_ptr<unsigned char[]>(new unsigned char[static_cast<std::size_t>(size)]),
                    static_cast<std::size_t>(size)};
    in.seekg(0, std::ios::beg);
    if (size) {
        in.read(reinterpret_cast<char*>(bytes.data.get()), static_cast<std::streamsize>(size));
        if (!in) {
            throw std::runtime_error("unable to read GDS file");
        }
    }
    return bytes;
}

LocalPoint read_xy(const unsigned char* xy) {
    return {read_i32(xy), read_i32(xy + 4)};
}

__int128 cross(LocalPoint a, LocalPoint b) {
    return static_cast<__int128>(static_cast<std::int64_t>(a.x) * b.y)
        - static_cast<std::int64_t>(b.x) * a.y;
}

void normalize_polygon(Polygon& polygon, const unsigned char* data, LocalPoint* points) {
    const auto* xy = data + polygon.xy_offset;
    auto* out = points + polygon.point_offset;
    const auto first = read_xy(xy);
    auto prev = first;
    LocalPoint low = first, high = first, second = first;
    unsigned distinct = 1;
    std::uint32_t count = 1;
    __int128 twice_area = 0;
    out[0] = first;
    for (std::uint32_t i = 1; i < polygon.point_count; ++i) {
        const auto point = read_xy(xy + i * 8);
        if (point == prev) {
            continue;
        }
        out[count++] = point;
        low = {std::min(low.x, point.x), std::min(low.y, point.y)};
        high = {std::max(high.x, point.x), std::max(high.y, point.y)};
        if (distinct == 1) {
            second = point;
            distinct = 2;
        } else if (distinct == 2 && !(point == first) && !(point == second)) {
            distinct = 3;
        }
        twice_area += cross(prev, point);
        prev = point;
    }
    twice_area += cross(prev, first);
    if (count > 1 && prev == first) {
        --count;
    }
    polygon.point_count = count;
    polygon.low = low;
    polygon.high = high;
    polygon.flags = (distinct < 3 ? 1u : 0u) | (twice_area == 0 ? 2u : 0u);
}

template<class Work>
void parallel_ranges(std::size_t count, std::size_t grain, unsigned workers, Work work) {
    const auto lanes = std::min<std::size_t>(workers, count / grain + (count % grain != 0));
    if (lanes <= 1) {
        work(0, count);
        return;
    }
    // Input validation is done before these allocation-free worker callbacks.
    const auto run = [&](std::size_t lane) {
        const auto stride = count / lanes;
        work(lane * stride, lane + 1 == lanes ? count : (lane + 1) * stride);
    };
    std::vector<std::thread> threads;
    threads.reserve(lanes - 1);
    try {
        for (std::size_t lane = 1; lane < lanes; ++lane) {
            threads.emplace_back(run, lane);
        }
        run(0);
    } catch (...) {
        for (auto& thread : threads) thread.join();
        throw;
    }
    for (auto& thread : threads) thread.join();
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

struct PointSequence {
    const std::int64_t* xy;
    std::size_t count;

    Point operator[](std::size_t index) const {
        return {xy[2 * index], xy[2 * index + 1]};
    }
};

bool sequence_less(
    PointSequence points,
    std::size_t left_candidate,
    bool left_forward,
    std::size_t right_candidate,
    bool right_forward) {
    const auto count = points.count;
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

bool is_axis_aligned_bbox_rectangle(PointSequence points, BBox bbox) {
    if (points.count != 4 || bbox.x0 >= bbox.x1 || bbox.y0 >= bbox.y1) {
        return false;
    }
    bool corners[4] = {false, false, false, false};
    for (std::size_t index = 0; index < 4; ++index) {
        const auto point = points[index];
        int corner = -1;
        if (point.x == bbox.x0 && point.y == bbox.y0) corner = 0;
        if (point.x == bbox.x0 && point.y == bbox.y1) corner = 1;
        if (point.x == bbox.x1 && point.y == bbox.y0) corner = 2;
        if (point.x == bbox.x1 && point.y == bbox.y1) corner = 3;
        if (corner < 0 || corners[corner]) {
            return false;
        }
        corners[corner] = true;
    }
    return corners[0] && corners[1] && corners[2] && corners[3];
}

std::pair<std::uint64_t, std::uint64_t> canonical_polygon_hash(PointSequence points, BBox bbox, std::uint32_t flags) {
    std::uint64_t h0 = 1469598103934665603ull;
    std::uint64_t h1 = 1099511628211ull;
    if (flags == 0 && is_axis_aligned_bbox_rectangle(points, bbox)) {
        h0 = fnv1a_u64(fnv1a_u64(h0, 2), 0);
        h0 = fnv1a_u64(fnv1a_u64(h0, static_cast<std::uint64_t>(bbox.x0)), static_cast<std::uint64_t>(bbox.y0));
        h0 = fnv1a_u64(fnv1a_u64(h0, static_cast<std::uint64_t>(bbox.x1)), static_cast<std::uint64_t>(bbox.y1));
        h1 = fnv1a_u64(fnv1a_u64(h1, static_cast<std::uint64_t>(bbox.y1)), static_cast<std::uint64_t>(bbox.x1));
        h1 = fnv1a_u64(fnv1a_u64(h1, static_cast<std::uint64_t>(bbox.y0)), static_cast<std::uint64_t>(bbox.x0));
        return {h0, h1};
    }
    std::size_t best_candidate = 0;
    bool best_forward = true;
    if (flags == 0 && points.count) {
        for (std::size_t candidate = 1; candidate < points.count; ++candidate) {
            if (sequence_less(points, candidate, true, best_candidate, best_forward)) {
                best_candidate = candidate;
                best_forward = true;
            }
        }
        for (std::size_t candidate = 0; candidate < points.count; ++candidate) {
            if (sequence_less(points, candidate, false, best_candidate, best_forward)) {
                best_candidate = candidate;
                best_forward = false;
            }
        }
    }
    for (std::size_t step = 0; step < points.count; ++step) {
        const auto index = flags == 0 ? sequence_index(points.count, best_candidate, step, best_forward) : step;
        const auto& point = points[index];
        const auto x = static_cast<std::uint64_t>(point.x);
        const auto y = static_cast<std::uint64_t>(point.y);
        h0 = fnv1a_u64(fnv1a_u64(h0, x), y);
        h1 = fnv1a_u64(fnv1a_u64(h1, y), x);
    }
    return {h0, h1};
}

std::map<std::string, Cell> parse_gds(const FileBytes& file, bool& oracle_reuse_eligible, std::size_t& point_capacity) {
    const auto* data = file.data.get();
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
    const unsigned char* pending_xy = nullptr;
    std::uint32_t pending_point_count = 0;
    bool saw_endlib = false;

    while (file.size - offset >= 4) {
        const std::uint16_t record_len = read_u16(data + offset);
        const int record_type = data[offset + 2];
        const int data_type = data[offset + 3];
        if (record_len < 4 || record_len > file.size - offset) {
            throw std::runtime_error("invalid GDS record length");
        }
        const unsigned char* payload = data + offset + 4;
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
            pending_point_count = 0;
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
            pending_point_count = 0;
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
            if (payload_size % 8 != 0) {
                throw std::runtime_error("invalid XY payload length");
            }
            pending_xy = payload;
            pending_point_count = static_cast<std::uint32_t>(payload_size / 8);
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
                if (pending_point_count) {
                    if (point_capacity > std::numeric_limits<std::size_t>::max() - pending_point_count) {
                        throw std::overflow_error("stored vertex count overflow");
                    }
                    current_cell.polygons.push_back(Polygon{
                        static_cast<std::size_t>(pending_xy - data), point_capacity, pending_point_count,
                        {pending_layer, pending_datatype},
                    });
                    point_capacity += pending_point_count;
                }
                if (pending_point_count < 4 || std::memcmp(pending_xy, pending_xy + 8 * (pending_point_count - 1), 8) != 0) {
                    oracle_reuse_eligible = false;
                }
            } else if (element_kind == 0x0A || element_kind == 0x0B) {
                if (pending_sname.empty()) {
                    throw std::runtime_error("reference missing SNAME");
                }
                if (!pending_point_count) {
                    throw std::runtime_error("reference missing XY");
                }
                if (element_kind == 0x0A && pending_point_count != 1) oracle_reuse_eligible = false;
                pending_reference.child = pending_sname;
                const auto origin = read_xy(pending_xy);
                const auto column = pending_point_count > 1 ? read_xy(pending_xy + 8) : origin;
                const auto row = pending_point_count > 2 ? read_xy(pending_xy + 16) : origin;
                pending_reference.ox = origin.x;
                pending_reference.oy = origin.y;
                pending_reference.cx = column.x;
                pending_reference.cy = column.y;
                pending_reference.rx = row.x;
                pending_reference.ry = row.y;
                current_cell.references.push_back(std::move(pending_reference));
            }
            element_kind = -1;
            pending_sname.clear();
            pending_point_count = 0;
            break;
        case 0x07:  // ENDSTR
            if (current_name.empty()) {
                throw std::runtime_error("cell missing STRNAME");
            }
            if (cells.count(current_name)) oracle_reuse_eligible = false;
            cells.insert_or_assign(current_name, std::move(current_cell));
            current_name.clear();
            current_cell = Cell{};
            break;
        case 0x04:  // ENDLIB
            saw_endlib = true;
            if (offset != file.size) oracle_reuse_eligible = false;
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
    if (offset != file.size) {
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

void resolve_reachable_cells(Cell& cell, std::map<std::string, Cell>& cells, std::vector<Cell*>& reachable) {
    if (cell.visiting) {
        throw std::runtime_error("cell reference cycle detected");
    }
    if (cell.visited) {
        return;
    }
    cell.visiting = true;
    reachable.push_back(&cell);
    for (auto& reference : cell.references) {
        const auto found = cells.find(reference.child);
        if (found == cells.end()) {
            throw std::runtime_error("unresolved reference");
        }
        reference.target = &found->second;
        resolve_reachable_cells(*reference.target, cells, reachable);
    }
    cell.visiting = false;
    cell.visited = true;
}

struct PolygonSpan {
    Polygon* polygons;
    std::size_t count;
};

struct EmitTile {
    const Polygon* polygons;
    std::size_t count;
    Placement placement;
    std::uint64_t polygon_start;
    std::uint64_t vertex_start;
    std::uint64_t source_start;
};

struct PackingPlan {
    std::map<std::string, Cell> cells;
    std::unique_ptr<LocalPoint[]> points;
    std::vector<Layer> layers;
    std::vector<EmitTile> tiles;
    std::uint64_t polygon_count = 0;
    std::uint64_t vertex_count = 0;
    std::uint64_t source_count = 0;
    bool oracle_reuse_eligible = true;
    unsigned workers = 1;
};

constexpr std::size_t kPolygonBlockSize = 2048;

void normalize_cells(PackingPlan& plan, const std::vector<Cell*>& reachable, const FileBytes& file, std::size_t capacity) {
    plan.points.reset(new LocalPoint[capacity]);
    std::vector<PolygonSpan> spans;
    for (auto* cell : reachable) {
        for (std::size_t start = 0; start < cell->polygons.size(); start += kPolygonBlockSize) {
            const auto count = std::min(kPolygonBlockSize, cell->polygons.size() - start);
            spans.push_back({cell->polygons.data() + start, count});
            cell->blocks.push_back({start, start + count});
        }
    }
    parallel_ranges(spans.size(), 2, plan.workers, [&](std::size_t begin, std::size_t end) {
        for (auto i = begin; i < end; ++i) {
            for (std::size_t j = 0; j < spans[i].count; ++j) {
                normalize_polygon(spans[i].polygons[j], file.data.get(), plan.points.get());
            }
        }
    });
}

void assign_layer_ids(
    PackingPlan& plan, const std::vector<Cell*>& reachable, const std::vector<Layer>& selected, bool all_layers) {
    std::map<Layer, std::uint32_t> ids;
    for (const auto* cell : reachable) {
        Layer previous{};
        bool have_previous = false;
        for (const auto& polygon : cell->polygons) {
            if (!have_previous || polygon.layer != previous) {
                previous = polygon.layer;
                have_previous = true;
                if (all_layers || std::binary_search(selected.begin(), selected.end(), polygon.layer)) {
                    ids.emplace(polygon.layer, 0);
                }
            }
        }
    }
    for (auto& [layer, id] : ids) {
        id = static_cast<std::uint32_t>(plan.layers.size());
        plan.layers.push_back(layer);
    }

    parallel_ranges(reachable.size(), 2, plan.workers, [&](std::size_t begin, std::size_t end) {
        const auto& layer_ids = ids;
        for (auto i = begin; i < end; ++i) {
            auto& cell = *reachable[i];
            if (cell.polygons.empty()) continue;
            const auto& first = cell.polygons.front();
            auto bounds = BBox{first.low.x, first.low.y, first.high.x, first.high.y};
            Layer previous = first.layer;
            auto layer = layer_ids.find(previous);
            for (auto& block : cell.blocks) {
                for (auto j = block.begin; j < block.end; ++j) {
                    auto& polygon = cell.polygons[j];
                    if (polygon.layer != previous) {
                        previous = polygon.layer;
                        layer = layer_ids.find(previous);
                    }
                    if (layer != layer_ids.end()) {
                        polygon.layer_id = layer->second;
                        ++block.polygon_count;
                        block.vertex_count += polygon.point_count;
                    }
                    bounds.x0 = std::min<std::int64_t>(bounds.x0, polygon.low.x);
                    bounds.y0 = std::min<std::int64_t>(bounds.y0, polygon.low.y);
                    bounds.x1 = std::max<std::int64_t>(bounds.x1, polygon.high.x);
                    bounds.y1 = std::max<std::int64_t>(bounds.y1, polygon.high.y);
                }
            }
            cell.bbox = bounds;
        }
    });
}

void add_count(std::uint64_t& count, std::uint64_t extra) {
    if (extra > std::numeric_limits<std::uint64_t>::max() - count) {
        throw std::overflow_error("flattened geometry count overflow");
    }
    count += extra;
}

void plan_placement(PackingPlan& plan, const Cell& cell, const Placement& placement) {
    if (!cell.polygons.empty()) {
        // Checking the cell extrema also proves each vertex can be transformed in int64.
        const auto low = placement.apply({cell.bbox.x0, cell.bbox.y0});
        const auto high = placement.apply({cell.bbox.x1, cell.bbox.y1});
        constexpr std::int64_t limit = 1ll << 40;
        if (std::min({low.x, low.y, high.x, high.y}) < -limit ||
            std::max({low.x, low.y, high.x, high.y}) > limit) {
            plan.oracle_reuse_eligible = false;
        }
        const auto source_start = plan.source_count;
        add_count(plan.source_count, cell.polygons.size());
        for (const auto& block : cell.blocks) {
            if (block.polygon_count) {
                plan.tiles.push_back({
                    cell.polygons.data() + block.begin, block.end - block.begin, placement,
                    plan.polygon_count, plan.vertex_count, source_start + block.begin,
                });
                add_count(plan.polygon_count, block.polygon_count);
                add_count(plan.vertex_count, block.vertex_count);
            }
        }
    }
    for (const auto& reference : cell.references) {
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
                plan_placement(plan, *reference.target, child);
            }
        }
    }
}

struct GdsPolygonExtractPlanInfo {
    std::uint64_t polygon_count;
    std::uint64_t vertex_count;
    std::uint64_t layer_count;
    double read_s;
    double scan_s;
    double normalize_s;
    double layer_table_s;
    double placement_s;
    std::uint32_t oracle_reuse_eligible;
};

struct GdsPolygonExtractArrays {
    std::int64_t* vertices_xy;
    std::uint64_t* polygon_offsets;
    std::uint32_t* layer_ids;
    std::int64_t* bboxes;
    std::uint32_t* flags;
    std::uint64_t* source_ids;
    std::uint64_t* hashes;
    std::int32_t* layer_pairs;
};

template<unsigned Turns>
Point transform(LocalPoint point, Point origin) {
    if constexpr (Turns == 1) return {origin.x - point.y, origin.y + point.x};
    if constexpr (Turns == 2) return {origin.x - point.x, origin.y - point.y};
    if constexpr (Turns == 3) return {origin.x + point.y, origin.y - point.x};
    return {origin.x + point.x, origin.y + point.y};
}

template<unsigned Turns>
void emit_tile(
    const EmitTile& tile, const LocalPoint* points, const GdsPolygonExtractArrays& out) {
    auto index = tile.polygon_start;
    auto offset = tile.vertex_start;
    for (std::size_t i = 0; i < tile.count; ++i) {
        const auto& polygon = tile.polygons[i];
        if (polygon.layer_id == kUnselectedLayer) continue;
        const auto a = transform<Turns>(polygon.low, tile.placement.origin);
        const auto b = transform<Turns>(polygon.high, tile.placement.origin);
        const BBox bbox{std::min(a.x, b.x), std::min(a.y, b.y), std::max(a.x, b.x), std::max(a.y, b.y)};
        out.polygon_offsets[index] = offset;
        out.layer_ids[index] = polygon.layer_id;
        out.bboxes[index * 4 + 0] = bbox.x0;
        out.bboxes[index * 4 + 1] = bbox.y0;
        out.bboxes[index * 4 + 2] = bbox.x1;
        out.bboxes[index * 4 + 3] = bbox.y1;
        out.flags[index] = polygon.flags;
        out.source_ids[index] = tile.source_start + i;
        auto* xy = out.vertices_xy + 2 * offset;
        const auto* local = points + polygon.point_offset;
        for (std::uint32_t j = 0; j < polygon.point_count; ++j) {
            const auto vertex = transform<Turns>(local[j], tile.placement.origin);
            xy[2 * j] = vertex.x;
            xy[2 * j + 1] = vertex.y;
        }
        if (out.hashes) {
            const auto hash = canonical_polygon_hash({xy, polygon.point_count}, bbox, polygon.flags);
            out.hashes[2 * index] = hash.first;
            out.hashes[2 * index + 1] = hash.second;
        }
        offset += polygon.point_count;
        ++index;
    }
}

}  // namespace

extern "C" {

const char* gds_polygon_extract_last_error() {
    return g_last_error.c_str();
}

void gds_polygon_extract_plan_free(void* plan) {
    delete static_cast<PackingPlan*>(plan);
}

int gds_polygon_extract_plan(
    const char* path, const char* top_name, const std::int32_t* selected_pairs, std::uint64_t selected_count,
    bool all_layers, unsigned workers, GdsPolygonExtractPlanInfo* info, void** out) {
    try {
        if (!path || !info || !out || (selected_count && !selected_pairs)) {
            throw std::runtime_error("invalid arguments");
        }
        *out = nullptr;
        *info = GdsPolygonExtractPlanInfo{};
        auto plan = std::make_unique<PackingPlan>();
        plan->workers = std::max(1u, workers);

        auto start = Clock::now();
        const auto file = read_file(path);
        info->read_s = elapsed(start);
        start = Clock::now();
        std::size_t point_capacity = 0;
        plan->cells = parse_gds(file, plan->oracle_reuse_eligible, point_capacity);
        if (plan->cells.empty()) {
            throw std::runtime_error("no cells found");
        }
        const auto top_name_resolved = select_top(plan->cells, top_name);
        auto& top = plan->cells.at(top_name_resolved);
        std::vector<Cell*> reachable;
        resolve_reachable_cells(top, plan->cells, reachable);
        info->scan_s = elapsed(start);

        start = Clock::now();
        normalize_cells(*plan, reachable, file, point_capacity);
        info->normalize_s = elapsed(start);

        start = Clock::now();
        std::vector<Layer> selected;
        for (std::uint64_t i = 0; i < selected_count; ++i) {
            selected.emplace_back(selected_pairs[2 * i], selected_pairs[2 * i + 1]);
        }
        assign_layer_ids(*plan, reachable, selected, all_layers);
        info->layer_table_s = elapsed(start);

        start = Clock::now();
        plan_placement(*plan, top, Placement{});
        const auto max_bytes = std::numeric_limits<std::ptrdiff_t>::max();
        if (plan->vertex_count > max_bytes / (2 * sizeof(std::int64_t)) ||
            plan->polygon_count > max_bytes / (4 * sizeof(std::int64_t))) {
            throw std::overflow_error("flattened geometry arrays are too large");
        }
        info->placement_s = elapsed(start);
        info->polygon_count = plan->polygon_count;
        info->vertex_count = plan->vertex_count;
        info->layer_count = plan->layers.size();
        info->oracle_reuse_eligible = plan->oracle_reuse_eligible;
        *out = plan.release();
        g_last_error.clear();
        return 0;
    } catch (const std::exception& exc) {
        g_last_error = exc.what();
        return 1;
    } catch (...) {
        g_last_error = "unknown native polygon extract error";
        return 1;
    }
}

int gds_polygon_extract_emit(const void* handle, const GdsPolygonExtractArrays* out, double* pack_s) {
    try {
        if (!handle || !out || !pack_s) {
            throw std::runtime_error("invalid arguments");
        }
        const auto& plan = *static_cast<const PackingPlan*>(handle);
        if (!out->polygon_offsets ||
            (plan.polygon_count && (!out->layer_ids || !out->bboxes || !out->flags || !out->source_ids)) ||
            (plan.vertex_count && !out->vertices_xy) || (!plan.layers.empty() && !out->layer_pairs)) {
            throw std::runtime_error("missing polygon output array");
        }
        const auto start = Clock::now();
        for (std::size_t i = 0; i < plan.layers.size(); ++i) {
            out->layer_pairs[2 * i] = plan.layers[i].first;
            out->layer_pairs[2 * i + 1] = plan.layers[i].second;
        }
        // Tiles own disjoint output ranges, including each polygon's start offset.
        parallel_ranges(plan.tiles.size(), 2, plan.workers, [&](std::size_t begin, std::size_t end) {
            for (auto i = begin; i < end; ++i) {
                const auto& tile = plan.tiles[i];
                switch (tile.placement.quarter_turns) {
                case 1: emit_tile<1>(tile, plan.points.get(), *out); break;
                case 2: emit_tile<2>(tile, plan.points.get(), *out); break;
                case 3: emit_tile<3>(tile, plan.points.get(), *out); break;
                default: emit_tile<0>(tile, plan.points.get(), *out); break;
                }
            }
        });
        out->polygon_offsets[plan.polygon_count] = plan.vertex_count;
        *pack_s = elapsed(start);
        g_last_error.clear();
        return 0;
    } catch (const std::exception& exc) {
        g_last_error = exc.what();
        return 1;
    } catch (...) {
        g_last_error = "unknown native polygon packing error";
        return 1;
    }
}

}  // extern "C"
