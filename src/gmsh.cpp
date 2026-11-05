#include "idmesh/mesh.hpp"
#include <algorithm>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace idmesh {
namespace {
struct Element {
    CellType type;
    std::vector<std::uint64_t> nodes;
    int physical = 0;
};
CellType element_type(unsigned type)
{
    switch (type) {
    case 1: return CellType::segment;
    case 2: return CellType::triangle;
    case 3: return CellType::quadrilateral;
    case 4: return CellType::tetrahedron;
    case 5: return CellType::hexahedron;
    case 6: return CellType::prism;
    case 7: return CellType::pyramid;
    default: throw std::runtime_error("unsupported Gmsh element type: only first-order elements are supported");
    }
}
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}
} // namespace

Mesh read_gmsh(const std::string& filename, TopologyOptions options)
{
    std::ifstream in(filename);
    if (!in) throw std::runtime_error("cannot open Gmsh file: " + filename);
    std::string section;
    double version = 0.0;
    std::map<std::uint64_t, std::array<double, 3>> nodes;
    std::vector<Element> elements;
    std::map<std::pair<unsigned, std::uint64_t>, int> physical_tags;
    while (in >> section) {
        if (section == "$MeshFormat") {
            unsigned file_type = 0, data_size = 0;
            in >> version >> file_type >> data_size;
            require(in.good() && file_type == 0
                    && ((version >= 2.2 && version < 2.3) || (version >= 4.1 && version < 4.2)),
                "only ASCII Gmsh 2.2 and 4.1 are supported");
        } else if (section == "$Entities") {
            require(version >= 4.1, "unexpected Entities section");
            std::uint64_t counts[4] {};
            for (auto& n : counts) in >> n;
            std::string line;
            std::getline(in, line);
            for (unsigned dim = 0; dim < 4; ++dim) {
                for (std::uint64_t i = 0; i < counts[dim]; ++i) {
                    require(static_cast<bool>(std::getline(in, line)), "truncated Entities");
                    std::istringstream record(line);
                    std::uint64_t tag;
                    double ignored;
                    record >> tag;
                    for (unsigned j = 0; j < (dim == 0 ? 3U : 6U); ++j) record >> ignored;
                    std::uint64_t count = 0;
                    record >> count;
                    require(static_cast<bool>(record), "malformed Entities record");
                    for (std::uint64_t p = 0; p < count; ++p) {
                        int value;
                        record >> value;
                        require(static_cast<bool>(record), "malformed physical tag");
                        if (p == 0) physical_tags[{dim, tag}] = value;
                    }
                }
            }
        } else if (section == "$Nodes") {
            require(version != 0, "missing MeshFormat");
            if (version < 4) {
                std::uint64_t count;
                in >> count;
                for (std::uint64_t i = 0; i < count; ++i) {
                    std::uint64_t tag;
                    std::array<double, 3> xyz;
                    in >> tag >> xyz[0] >> xyz[1] >> xyz[2];
                    require(nodes.emplace(tag, xyz).second, "duplicate node tag");
                }
            } else {
                std::uint64_t blocks, total, min_tag, max_tag;
                in >> blocks >> total >> min_tag >> max_tag;
                for (std::uint64_t b = 0; b < blocks; ++b) {
                    unsigned dim, parametric;
                    std::uint64_t entity, count;
                    in >> dim >> entity >> parametric >> count;
                    require(dim <= 3 && parametric <= 1, "invalid node block");
                    std::vector<std::uint64_t> tags(count);
                    for (auto& tag : tags) in >> tag;
                    for (auto tag : tags) {
                        std::array<double, 3> xyz;
                        in >> xyz[0] >> xyz[1] >> xyz[2];
                        double unused;
                        if (parametric) for (unsigned i = 0; i < dim; ++i) in >> unused;
                        require(nodes.emplace(tag, xyz).second, "duplicate node tag");
                    }
                }
                require(nodes.size() == total, "Gmsh node count mismatch");
            }
        } else if (section == "$Elements") {
            require(version != 0, "missing MeshFormat");
            if (version < 4) {
                std::uint64_t count;
                in >> count;
                for (std::uint64_t i = 0; i < count; ++i) {
                    std::uint64_t id;
                    unsigned kind, tag_count;
                    in >> id >> kind >> tag_count;
                    int physical = 0;
                    for (unsigned t = 0; t < tag_count; ++t) {
                        int tag;
                        in >> tag;
                        if (t == 0) physical = tag;
                    }
                    if (kind == 15) {
                        std::uint64_t vertex;
                        in >> vertex;
                        continue;
                    }
                    auto type = element_type(kind);
                    Element element {type, {}, physical};
                    element.nodes.resize(cell_vertex_count(type));
                    for (auto& node : element.nodes) in >> node;
                    elements.push_back(std::move(element));
                }
            } else {
                std::uint64_t blocks, total, min_tag, max_tag;
                in >> blocks >> total >> min_tag >> max_tag;
                for (std::uint64_t b = 0; b < blocks; ++b) {
                    unsigned dim, kind;
                    std::uint64_t entity, count;
                    in >> dim >> entity >> kind >> count;
                    if (kind == 15) {
                        for (std::uint64_t i = 0; i < count; ++i) {
                            std::uint64_t id, node;
                            in >> id >> node;
                        }
                        continue;
                    }
                    auto type = element_type(kind);
                    require(cell_dimension(type) == dim, "element block dimension mismatch");
                    auto physical = physical_tags.find({dim, entity});
                    for (std::uint64_t i = 0; i < count; ++i) {
                        std::uint64_t id;
                        in >> id;
                        Element element {type, {}, physical == physical_tags.end() ? 0 : physical->second};
                        element.nodes.resize(cell_vertex_count(type));
                        for (auto& node : element.nodes) in >> node;
                        elements.push_back(std::move(element));
                    }
                }
            }
        } else if (!section.empty() && section[0] == '$' && section.rfind("$End", 0) != 0) {
            const std::string end = "$End" + section.substr(1);
            std::string line;
            std::getline(in, line);
            while (std::getline(in, line) && line != end) {}
            require(line == end, "unterminated Gmsh section");
        }
        require(!in.bad(), "Gmsh read failure");
    }
    require(!nodes.empty() && !elements.empty(), "empty Gmsh mesh");
    Index dimension = 0;
    for (const auto& element : elements) dimension = std::max(dimension, cell_dimension(element.type));
    MeshBuilder builder(dimension, 3);
    std::map<std::uint64_t, Index> local;
    for (const auto& [tag, xyz] : nodes) local.emplace(tag, builder.add_vertex(xyz));
    for (const auto& element : elements) {
        if (cell_dimension(element.type) != dimension) continue;
        std::vector<Index> vertices;
        for (auto tag : element.nodes) {
            const auto it = local.find(tag);
            require(it != local.end(), "element refers to unknown vertex");
            vertices.push_back(it->second);
        }
        builder.add_cell(element.type, vertices, element.physical);
    }
    return std::move(builder).finalize(options);
}
} // namespace idmesh
