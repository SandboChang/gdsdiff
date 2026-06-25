#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

thread_local std::string g_last_error;

struct Point {
    std::int64_t x = 0;
    std::int64_t y = 0;

    bool operator<(const Point& other) const {
        return std::tie(x, y) < std::tie(other.x, other.y);
    }

    bool operator==(const Point& other) const {
        return x == other.x && y == other.y;
    }
};

struct Edge {
    Point start;
    Point stop;

    bool operator<(const Edge& other) const {
        return std::tie(start.x, start.y, stop.x, stop.y)
            < std::tie(other.start.x, other.start.y, other.stop.x, other.stop.y);
    }
};

struct LineKey {
    std::int64_t ux = 0;
    std::int64_t uy = 0;
    __int128 c = 0;

    bool operator<(const LineKey& other) const {
        if (ux != other.ux) return ux < other.ux;
        if (uy != other.uy) return uy < other.uy;
        return c < other.c;
    }
};

struct Fragment {
    std::vector<Point> points;
    std::int64_t bbox[4] = {0, 0, 0, 0};
    std::int64_t twice_area = 0;
};

std::int64_t cross(Point a, Point b, Point c) {
    return (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
}

bool collinear(Point a, Point b, Point c) {
    return cross(a, b, c) == 0;
}

std::int64_t abs_i64(std::int64_t value) {
    return value < 0 ? -value : value;
}

LineKey line_key(const Edge& edge) {
    const std::int64_t dx = edge.stop.x - edge.start.x;
    const std::int64_t dy = edge.stop.y - edge.start.y;
    const auto scale = std::gcd(abs_i64(dx), abs_i64(dy));
    if (scale == 0) {
        throw std::runtime_error("zero-length segment has no line key");
    }
    std::int64_t ux = dx / scale;
    std::int64_t uy = dy / scale;
    if (ux < 0 || (ux == 0 && uy < 0)) {
        ux = -ux;
        uy = -uy;
    }
    const __int128 c = -static_cast<__int128>(uy) * edge.start.x + static_cast<__int128>(ux) * edge.start.y;
    return LineKey{ux, uy, c};
}

__int128 projection(Point point, const LineKey& key) {
    return static_cast<__int128>(point.x) * key.ux + static_cast<__int128>(point.y) * key.uy;
}

std::vector<Edge> split_collinear_edges(const std::vector<Edge>& edges) {
    std::map<LineKey, std::vector<Edge>> by_line;
    std::map<LineKey, std::map<__int128, Point>> points_by_line;
    for (const auto& edge : edges) {
        if (edge.start == edge.stop) {
            continue;
        }
        const auto key = line_key(edge);
        by_line[key].push_back(edge);
        points_by_line[key][projection(edge.start, key)] = edge.start;
        points_by_line[key][projection(edge.stop, key)] = edge.stop;
    }

    std::vector<Edge> split_edges;
    for (const auto& item : by_line) {
        const auto& key = item.first;
        const auto& line_edges = item.second;
        std::vector<__int128> line_points;
        line_points.reserve(points_by_line[key].size());
        for (const auto& point_item : points_by_line[key]) {
            line_points.push_back(point_item.first);
        }
        for (const auto& edge : line_edges) {
            const auto start_t = projection(edge.start, key);
            const auto end_t = projection(edge.stop, key);
            const auto low = std::min(start_t, end_t);
            const auto high = std::max(start_t, end_t);
            std::vector<__int128> breakpoints;
            for (const auto value : line_points) {
                if (low <= value && value <= high) {
                    breakpoints.push_back(value);
                }
            }
            if (start_t > end_t) {
                std::reverse(breakpoints.begin(), breakpoints.end());
            }
            for (std::size_t index = 0; index + 1 < breakpoints.size(); ++index) {
                if (breakpoints[index] == breakpoints[index + 1]) {
                    continue;
                }
                split_edges.push_back(Edge{points_by_line[key][breakpoints[index]], points_by_line[key][breakpoints[index + 1]]});
            }
        }
    }
    std::sort(split_edges.begin(), split_edges.end());
    return split_edges;
}

std::vector<Edge> boundary_edges_from_polygons(
    const std::int64_t* vertices_xy,
    const std::uint64_t* fragment_offsets,
    std::uint64_t fragment_count
) {
    std::vector<Edge> raw_edges;
    for (std::uint64_t fragment_index = 0; fragment_index < fragment_count; ++fragment_index) {
        const std::uint64_t start = fragment_offsets[fragment_index];
        const std::uint64_t stop = fragment_offsets[fragment_index + 1];
        if (stop <= start + 2) {
            continue;
        }
        for (std::uint64_t vertex_index = start; vertex_index < stop; ++vertex_index) {
            const std::uint64_t next_index = vertex_index + 1 == stop ? start : vertex_index + 1;
            Edge edge{
                Point{vertices_xy[vertex_index * 2], vertices_xy[vertex_index * 2 + 1]},
                Point{vertices_xy[next_index * 2], vertices_xy[next_index * 2 + 1]},
            };
            if (!(edge.start == edge.stop)) {
                raw_edges.push_back(edge);
            }
        }
    }

    std::map<Edge, std::uint64_t> counts;
    for (const auto& edge : split_collinear_edges(raw_edges)) {
        const Edge reverse{edge.stop, edge.start};
        auto found = counts.find(reverse);
        if (found != counts.end() && found->second > 0) {
            --found->second;
            if (found->second == 0) {
                counts.erase(found);
            }
        } else {
            ++counts[edge];
        }
    }

    std::vector<Edge> boundary;
    for (const auto& item : counts) {
        for (std::uint64_t repeat = 0; repeat < item.second; ++repeat) {
            boundary.push_back(item.first);
        }
    }
    std::sort(boundary.begin(), boundary.end());
    return boundary;
}

std::pair<int, __int128> turn_key(Point previous, Point current, Point next) {
    const __int128 in_x = static_cast<__int128>(current.x) - previous.x;
    const __int128 in_y = static_cast<__int128>(current.y) - previous.y;
    const __int128 out_x = static_cast<__int128>(next.x) - current.x;
    const __int128 out_y = static_cast<__int128>(next.y) - current.y;
    const __int128 cross_value = in_x * out_y - in_y * out_x;
    const __int128 dot = in_x * out_x + in_y * out_y;
    int turn_class = 3;
    if (cross_value < 0) {
        turn_class = 0;
    } else if (cross_value == 0 && dot > 0) {
        turn_class = 1;
    } else if (cross_value > 0) {
        turn_class = 2;
    }
    const __int128 abs_cross = cross_value < 0 ? -cross_value : cross_value;
    return {turn_class, -abs_cross * 2 - dot};
}

std::vector<std::vector<Point>> trace_loops(const std::vector<Edge>& edges) {
    std::map<Point, std::vector<Point>> outgoing;
    std::set<Edge> remaining;
    for (const auto& edge : edges) {
        outgoing[edge.start].push_back(edge.stop);
        remaining.insert(edge);
    }
    for (auto& item : outgoing) {
        std::sort(item.second.begin(), item.second.end());
    }

    std::vector<std::vector<Point>> loops;
    while (!remaining.empty()) {
        const auto seed = *remaining.begin();
        remaining.erase(seed);
        std::vector<Point> loop{seed.start, seed.stop};
        Point previous = seed.start;
        Point current = seed.stop;
        while (!(current == seed.start)) {
            const auto found = outgoing.find(current);
            if (found == outgoing.end()) {
                break;
            }
            Point best{};
            bool have_best = false;
            for (const auto& candidate : found->second) {
                Edge next_edge{current, candidate};
                if (remaining.find(next_edge) == remaining.end()) {
                    continue;
                }
                if (!have_best) {
                    best = candidate;
                    have_best = true;
                    continue;
                }
                const auto left_key = turn_key(previous, current, candidate);
                const auto right_key = turn_key(previous, current, best);
                if (left_key < right_key || (left_key == right_key && candidate < best)) {
                    best = candidate;
                }
            }
            if (!have_best) {
                break;
            }
            remaining.erase(Edge{current, best});
            loop.push_back(best);
            previous = current;
            current = best;
        }
        if (!loop.empty() && loop.back() == seed.start) {
            loop.pop_back();
        }
        loops.push_back(std::move(loop));
    }
    std::sort(loops.begin(), loops.end());
    return loops;
}

std::vector<Point> clean_loop(const std::vector<Point>& input) {
    std::vector<Point> cleaned;
    cleaned.reserve(input.size());
    for (const auto& point : input) {
        if (cleaned.empty() || !(cleaned.back() == point)) {
            cleaned.push_back(point);
        }
    }
    if (cleaned.size() > 1 && cleaned.front() == cleaned.back()) {
        cleaned.pop_back();
    }
    bool changed = true;
    while (changed && cleaned.size() >= 3) {
        changed = false;
        std::vector<Point> next;
        next.reserve(cleaned.size());
        for (std::size_t index = 0; index < cleaned.size(); ++index) {
            const auto& previous = cleaned[(index + cleaned.size() - 1) % cleaned.size()];
            const auto& point = cleaned[index];
            const auto& following = cleaned[(index + 1) % cleaned.size()];
            if (collinear(previous, point, following)) {
                changed = true;
                continue;
            }
            next.push_back(point);
        }
        cleaned = std::move(next);
    }
    return cleaned;
}

std::int64_t twice_area(const std::vector<Point>& points) {
    __int128 area = 0;
    for (std::size_t index = 0; index < points.size(); ++index) {
        const auto& point = points[index];
        const auto& next = points[(index + 1) % points.size()];
        area += static_cast<__int128>(point.x) * next.y - static_cast<__int128>(next.x) * point.y;
    }
    if (area > std::numeric_limits<std::int64_t>::max() || area < std::numeric_limits<std::int64_t>::min()) {
        throw std::runtime_error("boundary loop area overflow");
    }
    if (area < 0) {
        area = -area;
    }
    return static_cast<std::int64_t>(area);
}

void fill_bbox(Fragment& fragment) {
    fragment.bbox[0] = fragment.points[0].x;
    fragment.bbox[1] = fragment.points[0].y;
    fragment.bbox[2] = fragment.points[0].x;
    fragment.bbox[3] = fragment.points[0].y;
    for (const auto& point : fragment.points) {
        fragment.bbox[0] = std::min(fragment.bbox[0], point.x);
        fragment.bbox[1] = std::min(fragment.bbox[1], point.y);
        fragment.bbox[2] = std::max(fragment.bbox[2], point.x);
        fragment.bbox[3] = std::max(fragment.bbox[3], point.y);
    }
}

std::vector<Fragment> fragments_from_edges(const std::int64_t* raw_edges, std::uint64_t edge_count, std::uint64_t* dropped) {
    std::vector<Edge> edges;
    edges.reserve(static_cast<std::size_t>(edge_count));
    for (std::uint64_t index = 0; index < edge_count; ++index) {
        const auto base = index * 4;
        Edge edge{Point{raw_edges[base], raw_edges[base + 1]}, Point{raw_edges[base + 2], raw_edges[base + 3]}};
        if (!(edge.start == edge.stop)) {
            edges.push_back(edge);
        }
    }
    auto loops = trace_loops(edges);
    std::vector<Fragment> fragments;
    for (const auto& loop : loops) {
        auto points = clean_loop(loop);
        if (points.size() < 3) {
            ++(*dropped);
            continue;
        }
        auto area = twice_area(points);
        if (area == 0) {
            ++(*dropped);
            continue;
        }
        Fragment fragment;
        fragment.points = std::move(points);
        fragment.twice_area = area;
        fill_bbox(fragment);
        fragments.push_back(std::move(fragment));
    }
    std::sort(fragments.begin(), fragments.end(), [](const Fragment& left, const Fragment& right) {
        return std::tie(left.bbox[0], left.bbox[1], left.bbox[2], left.bbox[3], left.twice_area, left.points)
            < std::tie(right.bbox[0], right.bbox[1], right.bbox[2], right.bbox[3], right.twice_area, right.points);
    });
    return fragments;
}

}  // namespace

extern "C" {

struct GdsBoundaryEdgesResult {
    std::uint64_t edge_count;
    std::int64_t* edges;
};

struct GdsBoundaryLoopResult {
    std::uint64_t fragment_count;
    std::uint64_t vertex_count;
    std::int64_t* vertices_xy;
    std::uint64_t* fragment_offsets;
    std::int64_t* bboxes;
    std::int64_t* twice_areas;
    std::uint64_t dropped_zero_area_loop_count;
};

const char* gds_boundary_loops_last_error() {
    return g_last_error.c_str();
}

void gds_boundary_edges_free(GdsBoundaryEdgesResult* result) {
    if (!result) {
        return;
    }
    std::free(result->edges);
    std::memset(result, 0, sizeof(GdsBoundaryEdgesResult));
}

void gds_boundary_loops_free(GdsBoundaryLoopResult* result) {
    if (!result) {
        return;
    }
    std::free(result->vertices_xy);
    std::free(result->fragment_offsets);
    std::free(result->bboxes);
    std::free(result->twice_areas);
    std::memset(result, 0, sizeof(GdsBoundaryLoopResult));
}

int gds_boundary_edges_from_polygons(
    const std::int64_t* vertices_xy,
    const std::uint64_t* fragment_offsets,
    std::uint64_t fragment_count,
    GdsBoundaryEdgesResult* out
) {
    try {
        if (!out) {
            throw std::invalid_argument("boundary edge output pointer is null");
        }
        std::memset(out, 0, sizeof(GdsBoundaryEdgesResult));
        if (fragment_count && (!vertices_xy || !fragment_offsets)) {
            throw std::invalid_argument("boundary edge polygon input pointer is null");
        }
        if (!fragment_count) {
            return 0;
        }
        const auto edges = boundary_edges_from_polygons(vertices_xy, fragment_offsets, fragment_count);
        out->edge_count = static_cast<std::uint64_t>(edges.size());
        out->edges = static_cast<std::int64_t*>(std::calloc(edges.size() * 4, sizeof(std::int64_t)));
        if (edges.size() && !out->edges) {
            throw std::runtime_error("boundary edge allocation failed");
        }
        for (std::size_t index = 0; index < edges.size(); ++index) {
            const auto& edge = edges[index];
            out->edges[index * 4] = edge.start.x;
            out->edges[index * 4 + 1] = edge.start.y;
            out->edges[index * 4 + 2] = edge.stop.x;
            out->edges[index * 4 + 3] = edge.stop.y;
        }
        return 0;
    } catch (const std::exception& exc) {
        g_last_error = exc.what();
        if (out) {
            gds_boundary_edges_free(out);
        }
        return 1;
    } catch (...) {
        g_last_error = "unknown boundary edge error";
        if (out) {
            gds_boundary_edges_free(out);
        }
        return 1;
    }
}

int gds_boundary_loops_from_edges(const std::int64_t* edges, std::uint64_t edge_count, GdsBoundaryLoopResult* out) {
    try {
        if (!out) {
            throw std::invalid_argument("boundary loop output pointer is null");
        }
        std::memset(out, 0, sizeof(GdsBoundaryLoopResult));
        if (edge_count && !edges) {
            throw std::invalid_argument("boundary loop edge pointer is null");
        }
        std::uint64_t dropped = 0;
        const auto fragments = fragments_from_edges(edges, edge_count, &dropped);
        std::uint64_t vertex_count = 0;
        for (const auto& fragment : fragments) {
            vertex_count += static_cast<std::uint64_t>(fragment.points.size());
        }
        out->fragment_count = static_cast<std::uint64_t>(fragments.size());
        out->vertex_count = vertex_count;
        out->dropped_zero_area_loop_count = dropped;
        out->fragment_offsets = static_cast<std::uint64_t*>(std::calloc(fragments.size() + 1, sizeof(std::uint64_t)));
        out->vertices_xy = static_cast<std::int64_t*>(std::calloc(vertex_count * 2, sizeof(std::int64_t)));
        out->bboxes = static_cast<std::int64_t*>(std::calloc(fragments.size() * 4, sizeof(std::int64_t)));
        out->twice_areas = static_cast<std::int64_t*>(std::calloc(fragments.size(), sizeof(std::int64_t)));
        if (!out->fragment_offsets || (vertex_count && !out->vertices_xy) || (fragments.size() && (!out->bboxes || !out->twice_areas))) {
            throw std::runtime_error("boundary loop allocation failed");
        }
        std::uint64_t vertex_offset = 0;
        for (std::size_t index = 0; index < fragments.size(); ++index) {
            const auto& fragment = fragments[index];
            out->fragment_offsets[index] = vertex_offset;
            for (const auto& point : fragment.points) {
                out->vertices_xy[vertex_offset * 2] = point.x;
                out->vertices_xy[vertex_offset * 2 + 1] = point.y;
                ++vertex_offset;
            }
            for (int axis = 0; axis < 4; ++axis) {
                out->bboxes[index * 4 + axis] = fragment.bbox[axis];
            }
            out->twice_areas[index] = fragment.twice_area;
        }
        out->fragment_offsets[fragments.size()] = vertex_offset;
        return 0;
    } catch (const std::exception& exc) {
        g_last_error = exc.what();
        if (out) {
            gds_boundary_loops_free(out);
        }
        return 1;
    } catch (...) {
        g_last_error = "unknown boundary loop error";
        if (out) {
            gds_boundary_loops_free(out);
        }
        return 1;
    }
}

}
