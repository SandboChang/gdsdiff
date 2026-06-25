#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

thread_local std::string g_last_error;

struct Reference {
    std::string child;
    std::uint64_t multiplier = 1;
    std::uint64_t raw_hash = 0;
};

struct Cell {
    std::uint64_t local_shape_count = 0;
    std::uint64_t local_shape_hash = 1469598103934665603ull;
    std::vector<Reference> references;
};

struct ParseResult {
    std::map<std::string, Cell> cells;
};

std::uint16_t read_u16(const unsigned char* p) {
    return static_cast<std::uint16_t>((static_cast<std::uint16_t>(p[0]) << 8) | p[1]);
}

std::int16_t read_i16(const unsigned char* p) {
    return static_cast<std::int16_t>(read_u16(p));
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

std::vector<unsigned char> read_file(const char* path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("unable to open GDS file");
    }
    in.seekg(0, std::ios::end);
    const auto size = in.tellg();
    if (size < 0) {
        throw std::runtime_error("unable to determine GDS file size");
    }
    in.seekg(0, std::ios::beg);
    std::vector<unsigned char> data(static_cast<std::size_t>(size));
    if (!data.empty()) {
        in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
        if (!in) {
            throw std::runtime_error("unable to read GDS file");
        }
    }
    return data;
}

ParseResult parse_gds(const char* path, bool compute_hash) {
    const auto data = read_file(path);
    ParseResult result;
    std::size_t offset = 0;
    std::string current_name;
    std::uint64_t local_shape_count = 0;
    std::uint64_t local_shape_hash = 1469598103934665603ull;
    std::vector<Reference> references;
    int element_kind = -1;
    std::size_t element_start = 0;
    std::string pending_sname;
    std::uint64_t pending_multiplier = 1;

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
            local_shape_count = 0;
            local_shape_hash = 1469598103934665603ull;
            references.clear();
            break;
        case 0x06:  // STRNAME
            current_name = decode_string(payload, payload_size);
            break;
        case 0x08:  // BOUNDARY
        case 0x09:  // PATH
            element_kind = record_type;
            element_start = offset - record_len;
            break;
        case 0x0A:  // SREF
        case 0x0B:  // AREF
            element_kind = record_type;
            element_start = offset - record_len;
            pending_sname.clear();
            pending_multiplier = 1;
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
                pending_multiplier = static_cast<std::uint64_t>(columns) * static_cast<std::uint64_t>(rows);
            }
            break;
        case 0x11:  // ENDEL
            if (element_kind == 0x08 || element_kind == 0x09) {
                ++local_shape_count;
                if (compute_hash) {
                    local_shape_hash = fnv1a_u64(local_shape_hash, static_cast<std::uint64_t>(record_type));
                    local_shape_hash = fnv1a_update(local_shape_hash, data.data() + element_start, offset - element_start);
                }
            } else if (element_kind == 0x0A || element_kind == 0x0B) {
                if (pending_sname.empty()) {
                    throw std::runtime_error("reference missing SNAME");
                }
                const std::uint64_t raw_hash = compute_hash
                    ? fnv1a_update(1469598103934665603ull, data.data() + element_start, offset - element_start)
                    : 0;
                references.push_back(Reference{pending_sname, pending_multiplier, raw_hash});
            }
            element_kind = -1;
            pending_sname.clear();
            pending_multiplier = 1;
            break;
        case 0x07:  // ENDSTR
            if (current_name.empty()) {
                throw std::runtime_error("cell missing STRNAME");
            }
            result.cells[current_name] = Cell{local_shape_count, local_shape_hash, references};
            current_name.clear();
            local_shape_count = 0;
            local_shape_hash = 1469598103934665603ull;
            references.clear();
            break;
        default:
            break;
        }
    }
    if (offset != data.size()) {
        throw std::runtime_error("trailing partial GDS record");
    }
    return result;
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

std::uint64_t flatten_count(
    const std::string& name,
    const std::map<std::string, Cell>& cells,
    std::map<std::string, std::uint64_t>& memo,
    std::set<std::string>& visiting,
    std::set<std::string>& seen) {
    if (memo.count(name)) {
        seen.insert(name);
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
    seen.insert(name);
    std::uint64_t total = found->second.local_shape_count;
    for (const auto& reference : found->second.references) {
        const std::uint64_t child_count = flatten_count(reference.child, cells, memo, visiting, seen);
        if (reference.multiplier != 0 && child_count > UINT64_MAX / reference.multiplier) {
            throw std::overflow_error("flattened shape count overflow");
        }
        const std::uint64_t add = child_count * reference.multiplier;
        if (total > UINT64_MAX - add) {
            throw std::overflow_error("flattened shape count overflow");
        }
        total += add;
    }
    visiting.erase(name);
    memo[name] = total;
    return total;
}

std::uint64_t structural_hash(
    const std::string& name,
    const std::map<std::string, Cell>& cells,
    std::map<std::string, std::uint64_t>& memo,
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
    std::uint64_t hash = 1469598103934665603ull;
    hash = fnv1a_u64(hash, found->second.local_shape_count);
    hash = fnv1a_u64(hash, found->second.local_shape_hash);
    hash = fnv1a_u64(hash, found->second.references.size());
    for (const auto& reference : found->second.references) {
        hash = fnv1a_u64(hash, reference.multiplier);
        hash = fnv1a_u64(hash, reference.raw_hash);
        hash = fnv1a_update(hash, reinterpret_cast<const unsigned char*>(reference.child.data()), reference.child.size());
        hash = fnv1a_u64(hash, structural_hash(reference.child, cells, memo, visiting));
    }
    visiting.erase(name);
    memo[name] = hash;
    return hash;
}

}  // namespace

extern "C" {

struct GdsStructuralResult {
    std::uint64_t flattened_shape_count;
    std::uint64_t cell_count;
    std::uint64_t reachable_cell_count;
    std::uint64_t structural_hash;
};

const char* gds_structural_last_error() {
    return g_last_error.c_str();
}

int gds_structural_summary_impl(const char* path, const char* top_name, GdsStructuralResult* out, bool compute_hash) {
    try {
        if (!path || !out) {
            throw std::runtime_error("invalid arguments");
        }
        const auto parsed = parse_gds(path, compute_hash);
        if (parsed.cells.empty()) {
            throw std::runtime_error("no cells found");
        }
        const std::string top = select_top(parsed.cells, top_name);
        std::map<std::string, std::uint64_t> memo;
        std::set<std::string> visiting;
        std::set<std::string> seen;
        const std::uint64_t count = flatten_count(top, parsed.cells, memo, visiting, seen);
        std::uint64_t hash = 0;
        if (compute_hash) {
            std::map<std::string, std::uint64_t> hash_memo;
            std::set<std::string> hash_visiting;
            hash = structural_hash(top, parsed.cells, hash_memo, hash_visiting);
        }
        out->flattened_shape_count = count;
        out->cell_count = parsed.cells.size();
        out->reachable_cell_count = seen.size();
        out->structural_hash = hash;
        return 0;
    } catch (const std::exception& exc) {
        g_last_error = exc.what();
        return 1;
    }
}

int gds_structural_summary(const char* path, const char* top_name, GdsStructuralResult* out) {
    return gds_structural_summary_impl(path, top_name, out, true);
}

int gds_structural_summary_counts_only(const char* path, const char* top_name, GdsStructuralResult* out) {
    return gds_structural_summary_impl(path, top_name, out, false);
}

}  // extern "C"
