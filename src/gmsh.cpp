#include "idmesh/mesh.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace idmesh {
    namespace {

        void require(bool condition, const char* reason)
        {
            if (!condition)
                throw std::runtime_error(reason);
        }

        struct ElementInfo {
            CellType type;
            Index dimension;
            Index corners;
            Index nodes;
            bool point = false;
        };

        ElementInfo element_info(unsigned kind)
        {
            switch (kind) {
            case 1:
                return {CellType::segment, 1, 2, 2};
            case 2:
                return {CellType::triangle, 2, 3, 3};
            case 3:
                return {CellType::quadrilateral, 2, 4, 4};
            case 4:
                return {CellType::tetrahedron, 3, 4, 4};
            case 5:
                return {CellType::hexahedron, 3, 8, 8};
            case 6:
                return {CellType::prism, 3, 6, 6};
            case 7:
                return {CellType::pyramid, 3, 5, 5};
            case 8:
                return {CellType::segment, 1, 2, 3};
            case 9:
                return {CellType::triangle, 2, 3, 6};
            case 10:
                return {CellType::quadrilateral, 2, 4, 9};
            case 11:
                return {CellType::tetrahedron, 3, 4, 10};
            case 12:
                return {CellType::hexahedron, 3, 8, 27};
            case 13:
                return {CellType::prism, 3, 6, 18};
            case 14:
                return {CellType::pyramid, 3, 5, 14};
            case 15:
                return {CellType::segment, 0, 1, 1, true};
            case 16:
                return {CellType::quadrilateral, 2, 4, 8};
            case 17:
                return {CellType::hexahedron, 3, 8, 20};
            case 18:
                return {CellType::prism, 3, 6, 15};
            case 19:
                return {CellType::pyramid, 3, 5, 13};
            default:
                throw std::runtime_error("unsupported Gmsh element type (supported: order 1 and 2)");
            }
        }

        struct Element {
            ElementInfo info;
            std::vector<std::uint64_t> nodes;
            std::vector<int> physical;
        };

        using EntityKey = std::pair<unsigned, std::uint64_t>;

        void expect_end(std::istream& input, const std::string& section)
        {
            std::string marker;
            require(static_cast<bool>(input >> marker) && marker == "$End" + section.substr(1),
                "missing or incorrect Gmsh section terminator");
        }

    } // namespace

    Mesh read_gmsh(const std::string& filename, BuildOptions options)
    {
        std::ifstream in(filename);
        if (!in)
            throw std::runtime_error("cannot open Gmsh file: " + filename);

        double version = 0;
        bool seen_format = false, seen_nodes = false, seen_elements = false, seen_entities = false;
        std::map<std::uint64_t, std::array<double, 3>> nodes;
        std::vector<Element> elements;
        std::map<EntityKey, std::vector<int>> entity_physical;
        std::map<std::pair<Index, int>, std::string> physical_names;

        std::string section;
        while (in >> section) {
            if (section == "$MeshFormat") {
                require(!seen_format, "duplicate MeshFormat section");
                seen_format = true;
                unsigned format = 0, data_size = 0;
                require(static_cast<bool>(in >> version >> format >> data_size),
                    "invalid MeshFormat header");
                require(format == 0 && data_size == 8 && ((version >= 2.2 && version < 2.3) || (version >= 4.1 && version < 4.2)),
                    "only ASCII Gmsh MSH 2.2 and 4.1 are supported");
                expect_end(in, section);
            }
            else if (section == "$PhysicalNames") {
                require(seen_format, "MeshFormat must precede PhysicalNames");
                std::uint64_t count = 0;
                require(static_cast<bool>(in >> count), "invalid PhysicalNames header");
                for (std::uint64_t i = 0; i < count; ++i) {
                    Index dim = 0;
                    int tag = 0;
                    std::string name;
                    require(static_cast<bool>(in >> dim >> tag >> std::quoted(name)),
                        "invalid PhysicalNames entry");
                    require(dim <= 3, "invalid PhysicalNames dimension");
                    physical_names[{dim, tag}] = std::move(name);
                }
                expect_end(in, section);
            }
            else if (section == "$Entities") {
                require(version >= 4.1 && !seen_entities, "invalid duplicate Entities section");
                seen_entities = true;
                std::array<std::uint64_t, 4> counts {};
                for (auto& count : counts)
                    require(static_cast<bool>(in >> count), "invalid Entities header");
                std::string line;
                std::getline(in, line);
                for (unsigned dim = 0; dim <= 3; ++dim) {
                    for (std::uint64_t i = 0; i < counts[dim]; ++i) {
                        require(static_cast<bool>(std::getline(in, line)), "truncated Entities");
                        std::istringstream record(line);
                        std::uint64_t id = 0;
                        double ignored = 0;
                        require(static_cast<bool>(record >> id), "invalid entity tag");
                        for (unsigned component = 0; component < (dim == 0 ? 3U : 6U); ++component)
                            require(static_cast<bool>(record >> ignored), "invalid entity bounds");
                        std::uint64_t group_count = 0;
                        require(static_cast<bool>(record >> group_count), "invalid physical group count");
                        std::vector<int> groups(group_count);
                        for (auto& group : groups)
                            require(static_cast<bool>(record >> group), "invalid physical group");
                        require(entity_physical.emplace(EntityKey {dim, id}, std::move(groups)).second,
                            "duplicate geometrical entity tag");
                    }
                }
                expect_end(in, section);
            }
            else if (section == "$Nodes") {
                require(seen_format && !seen_nodes, "missing format or duplicate Nodes");
                seen_nodes = true;
                if (version < 4) {
                    std::uint64_t count = 0;
                    require(static_cast<bool>(in >> count), "invalid Nodes header");
                    for (std::uint64_t i = 0; i < count; ++i) {
                        std::uint64_t id;
                        std::array<double, 3> position;
                        require(static_cast<bool>(in >> id >> position[0] >> position[1] >> position[2]),
                            "truncated Gmsh nodes");
                        require(nodes.emplace(id, position).second, "duplicate Gmsh node tag");
                    }
                }
                else {
                    std::uint64_t blocks, total, min_tag, max_tag;
                    require(static_cast<bool>(in >> blocks >> total >> min_tag >> max_tag),
                        "invalid Nodes block header");
                    for (std::uint64_t block = 0; block < blocks; ++block) {
                        unsigned dim = 0, parametric = 0;
                        std::uint64_t entity = 0, count = 0;
                        require(static_cast<bool>(in >> dim >> entity >> parametric >> count),
                            "invalid Nodes block");
                        require(dim <= 3 && parametric <= 1, "invalid Gmsh node classification");
                        std::vector<std::uint64_t> tags(count);
                        for (auto& tag : tags)
                            require(static_cast<bool>(in >> tag), "missing node tag");
                        for (auto tag : tags) {
                            std::array<double, 3> position;
                            require(static_cast<bool>(in >> position[0] >> position[1] >> position[2]),
                                "missing Gmsh node coordinate");
                            for (unsigned i = 0; i < (parametric ? dim : 0); ++i) {
                                double parameter;
                                require(static_cast<bool>(in >> parameter), "missing parametric coordinate");
                            }
                            require(nodes.emplace(tag, position).second, "duplicate Gmsh node tag");
                        }
                    }
                    require(nodes.size() == total, "Gmsh node count mismatch");
                }
                expect_end(in, section);
            }
            else if (section == "$Elements") {
                require(seen_format && !seen_elements, "missing format or duplicate Elements");
                seen_elements = true;
                if (version < 4) {
                    std::uint64_t count = 0;
                    require(static_cast<bool>(in >> count), "invalid Elements header");
                    for (std::uint64_t i = 0; i < count; ++i) {
                        std::uint64_t id = 0;
                        unsigned kind = 0, n_tags = 0;
                        require(static_cast<bool>(in >> id >> kind >> n_tags), "invalid Gmsh element");
                        const auto info = element_info(kind);
                        Element element {info, {}, {}};
                        for (unsigned j = 0; j < n_tags; ++j) {
                            int tag = 0;
                            require(static_cast<bool>(in >> tag), "truncated Gmsh element tags");
                            if (j == 0 && tag != 0)
                                element.physical.push_back(tag);
                        }
                        element.nodes.resize(info.nodes);
                        for (auto& node : element.nodes)
                            require(static_cast<bool>(in >> node), "truncated element connectivity");
                        elements.push_back(std::move(element));
                    }
                }
                else {
                    std::uint64_t blocks, count, min_tag, max_tag;
                    require(static_cast<bool>(in >> blocks >> count >> min_tag >> max_tag),
                        "invalid Elements block header");
                    std::uint64_t read = 0;
                    for (std::uint64_t block = 0; block < blocks; ++block) {
                        unsigned dim = 0, kind = 0;
                        std::uint64_t entity = 0, n = 0;
                        require(static_cast<bool>(in >> dim >> entity >> kind >> n),
                            "invalid Elements block");
                        const auto info = element_info(kind);
                        require(info.dimension == dim, "Gmsh element dimension mismatch");
                        const auto found = entity_physical.find({dim, entity});
                        const std::vector<int> groups = found == entity_physical.end()
                            ? std::vector<int> {}
                            : found->second;
                        for (std::uint64_t j = 0; j < n; ++j) {
                            std::uint64_t id = 0;
                            require(static_cast<bool>(in >> id), "truncated element tag");
                            Element element {info, {}, groups};
                            element.nodes.resize(info.nodes);
                            for (auto& node : element.nodes)
                                require(static_cast<bool>(in >> node), "truncated element connectivity");
                            elements.push_back(std::move(element));
                        }
                        read += n;
                    }
                    require(read == count, "Gmsh element count mismatch");
                }
                expect_end(in, section);
            }
            else if (!section.empty() && section.front() == '$' && section.rfind("$End", 0) != 0) {
                const auto end = "$End" + section.substr(1);
                std::string line;
                std::getline(in, line);
                while (std::getline(in, line) && line != end) { }
                require(line == end, "unterminated Gmsh section");
            }
            else {
                throw std::runtime_error("unexpected Gmsh section delimiter: " + section);
            }
        }
        require(seen_format && seen_nodes && seen_elements, "missing required Gmsh section");
        require(!nodes.empty() && !elements.empty(), "empty Gmsh mesh");
        Index dimension = 0;
        for (const auto& element : elements)
            dimension = std::max(dimension, element.info.dimension);
        require(dimension >= 1 && dimension <= 3, "mesh needs at least one 1D-3D element");

        // High-order geometric nodes never become topological mesh vertices. Even
        // isolated lower-dimensional physical entities keep their corner vertices
        // for a lossless physical-label key.
        std::set<std::uint64_t> corner_tags;
        for (const auto& element : elements) {
            for (Index j = 0; j < element.info.corners; ++j) {
                require(nodes.contains(element.nodes[j]), "element references missing Gmsh node");
                corner_tags.insert(element.nodes[j]);
            }
            for (auto tag : element.nodes)
                require(nodes.contains(tag), "element references missing Gmsh node");
        }
        MeshBuilder builder(dimension, 3);
        std::map<std::uint64_t, Index> corners;
        for (auto node : corner_tags)
            corners.emplace(node, builder.add_vertex(nodes.at(node)));
        std::map<std::uint64_t, Index> geometry;
        for (const auto& element : elements) {
            if (element.info.dimension != dimension || element.info.point)
                continue;
            std::vector<Index> vertices;
            for (Index i = 0; i < element.info.corners; ++i)
                vertices.push_back(corners.at(element.nodes[i]));
            const int first = element.physical.empty() ? 0 : element.physical.front();
            Index cell = invalid_index;
            if (element.info.nodes == element.info.corners) {
                cell = builder.add_cell(element.info.type, vertices, first);
            }
            else {
                std::vector<Index> extras;
                for (Index i = element.info.corners; i < element.info.nodes; ++i) {
                    const auto tag = element.nodes[i];
                    auto [found, inserted] = geometry.try_emplace(tag, invalid_index);
                    if (inserted)
                        found->second = builder.add_geometry_node(nodes.at(tag));
                    extras.push_back(found->second);
                }
                cell = builder.add_quadratic_cell(element.info.type, vertices, extras, first);
            }
            for (std::size_t i = 1; i < element.physical.size(); ++i)
                builder.add_cell_physical_tag(cell, element.physical[i]);
        }
        for (const auto& element : elements) {
            if (element.info.dimension >= dimension)
                continue;
            if (element.physical.empty())
                continue;
            std::vector<Index> vertices;
            for (Index i = 0; i < element.info.corners; ++i)
                vertices.push_back(corners.at(element.nodes[i]));
            for (const int physical : element.physical)
                builder.add_physical_tag(element.info.dimension, vertices, physical);
        }
        for (auto& [key, name] : physical_names)
            if (key.first <= dimension)
                builder.set_physical_name(key.first, key.second, std::move(name));
        return std::move(builder).finalize(options);
    }

} // namespace idmesh
