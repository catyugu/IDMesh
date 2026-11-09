#include "idmesh/io.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace idmesh {
    namespace {

        struct OutputElement {
            unsigned kind = 0;
            Index dimension = 0;
            Index cell = invalid_index;
            std::vector<Index> lower_vertices;
            std::vector<int> physical;
        };

        unsigned gmsh_kind(CellType type, std::size_t count)
        {
            switch (type) {
            case CellType::segment:
                if (count == 2) return 1;
                if (count == 3) return 8;
                break;
            case CellType::triangle:
                if (count == 3) return 2;
                if (count == 6) return 9;
                break;
            case CellType::quadrilateral:
                if (count == 4) return 3;
                if (count == 8) return 16;
                if (count == 9) return 10;
                break;
            case CellType::tetrahedron:
                if (count == 4) return 4;
                if (count == 10) return 11;
                break;
            case CellType::hexahedron:
                if (count == 8) return 5;
                if (count == 20) return 17;
                if (count == 27) return 12;
                break;
            case CellType::prism:
                if (count == 6) return 6;
                if (count == 15) return 18;
                if (count == 18) return 13;
                break;
            case CellType::pyramid:
                if (count == 5) return 7;
                if (count == 13) return 19;
                if (count == 14) return 14;
                break;
            }
            throw std::runtime_error("unsupported Gmsh element geometry order");
        }

        unsigned lower_kind(Index dimension, std::size_t count)
        {
            if (dimension == 0 && count == 1) return 15;
            if (dimension == 1 && count == 2) return 1;
            if (dimension == 2 && count == 3) return 2;
            if (dimension == 2 && count == 4) return 3;
            throw std::runtime_error("unsupported lower-dimensional Gmsh label");
        }

        std::span<const Index> element_nodes(const Mesh& mesh, const OutputElement& element)
        {
            if (element.cell != invalid_index)
                return mesh.cell_geometry_nodes(element.cell);
            return element.lower_vertices;
        }

        void write_number(std::ostream& out, std::uint64_t value, unsigned bytes)
        {
            for (unsigned i = 0; i < bytes; ++i)
                out.put(static_cast<char>((value >> (8 * i)) & 0xff));
        }

        void write_int(std::ostream& out, std::int32_t value)
        {
            write_number(out, std::bit_cast<std::uint32_t>(value), 4);
        }

        void write_double(std::ostream& out, double value)
        {
            write_number(out, std::bit_cast<std::uint64_t>(value), 8);
        }

        std::int32_t bounded_int(std::uint64_t value)
        {
            if (value > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()))
                throw std::length_error("Gmsh 2.2 integer tag exceeds INT32_MAX");
            return static_cast<std::int32_t>(value);
        }

        void write_physical_names(std::ostream& out, const Mesh& mesh)
        {
            std::map<std::pair<Index, int>, std::string> names;
            for (const auto& label : mesh.labels()) {
                if (label.value <= 0)
                    throw std::runtime_error("Gmsh physical tags must be positive");
                const auto name = mesh.label_name(label.dimension, label.value);
                if (!name.empty())
                    names.emplace(std::make_pair(label.dimension, label.value), name);
            }
            if (names.empty())
                return;
            out << "$PhysicalNames\n" << names.size() << '\n';
            for (const auto& [key, name] : names) {
                if (name.size() > 127 || name.find('"') != std::string::npos)
                    throw std::runtime_error("Gmsh physical name exceeds 127 bytes or contains quotes");
                out << key.first << ' ' << key.second << ' ' << std::quoted(name) << '\n';
            }
            out << "$EndPhysicalNames\n";
        }

        std::vector<OutputElement> collect_elements(const Mesh& mesh, GmshVersion version)
        {
            const Index dimension = mesh.dimension();
            std::vector<OutputElement> result;
            result.reserve(mesh.entity_count(dimension) + mesh.labels().size());

            for (Index c = 0; c < mesh.entity_count(dimension); ++c) {
                OutputElement element;
                element.kind = gmsh_kind(mesh.cell_types()[c], mesh.cell_geometry_nodes(c).size());
                element.dimension = dimension;
                element.cell = c;
                for (const auto& label : mesh.labels(dimension, c)) {
                    if (label.value <= 0)
                        throw std::runtime_error("Gmsh physical tags must be positive");
                    element.physical.push_back(label.value);
                }
                std::sort(element.physical.begin(), element.physical.end());
                element.physical.erase(std::unique(element.physical.begin(), element.physical.end()),
                    element.physical.end());
                if (version == GmshVersion::v22 && element.physical.size() > 1)
                    throw std::runtime_error("Gmsh 2.2 cannot represent multiple physical groups on one cell");
                result.push_back(std::move(element));
            }

            // Coalesce multiple physical labels on the same lower-dimensional
            // entity; in 4.1 the resulting entity has all physical memberships.
            struct Labeled {
                std::vector<Index> vertices;
                std::vector<int> physical;
            };
            std::map<std::pair<Index, std::vector<Index>>, Labeled> lower;
            for (const auto& label : mesh.labels()) {
                if (label.dimension >= dimension)
                    continue;
                if (label.value <= 0)
                    throw std::runtime_error("Gmsh physical tags must be positive");
                std::vector<Index> key(label.vertices.begin(), label.vertices.end());
                std::sort(key.begin(), key.end());
                auto& record = lower[{label.dimension, key}];
                if (record.vertices.empty())
                    record.vertices.assign(label.vertices.begin(), label.vertices.end());
                record.physical.push_back(label.value);
            }
            for (auto& [key, record] : lower) {
                std::sort(record.physical.begin(), record.physical.end());
                record.physical.erase(std::unique(record.physical.begin(), record.physical.end()),
                    record.physical.end());
                const auto kind = lower_kind(key.first, record.vertices.size());
                if (version == GmshVersion::v22) {
                    for (const int physical : record.physical) {
                        OutputElement element;
                        element.kind = kind;
                        element.dimension = key.first;
                        element.lower_vertices = record.vertices;
                        element.physical.push_back(physical);
                        result.push_back(std::move(element));
                    }
                }
                else {
                    OutputElement element;
                    element.kind = kind;
                    element.dimension = key.first;
                    element.lower_vertices = std::move(record.vertices);
                    element.physical = std::move(record.physical);
                    result.push_back(std::move(element));
                }
            }
            return result;
        }

        struct Group {
            Index dimension = 0;
            unsigned kind = 0;
            std::int32_t entity = 0;
            std::vector<int> physical;
            std::vector<std::size_t> element_indices;
            std::array<double, 3> lower {};
            std::array<double, 3> upper {};
        };

        std::vector<Group> group_elements(const Mesh& mesh, const std::vector<OutputElement>& elements)
        {
            using Key = std::tuple<Index, unsigned, std::vector<int>, Index>;
            std::map<Key, std::size_t> indices;
            std::vector<Group> groups;
            std::array<std::int32_t, 4> next_tag {1, 1, 1, 1};
            for (std::size_t i = 0; i < elements.size(); ++i) {
                const auto& element = elements[i];
                // One point per point entity in Gmsh's geometric model.
                const Index point = element.dimension == 0 ? element.lower_vertices.front() : 0;
                auto [it, inserted] = indices.emplace(
                    Key {element.dimension, element.kind, element.physical, point}, groups.size());
                if (inserted) {
                    auto& next = next_tag.at(element.dimension);
                    if (next == std::numeric_limits<std::int32_t>::max())
                        throw std::length_error("too many Gmsh geometrical entities");
                    Group group;
                    group.dimension = element.dimension;
                    group.kind = element.kind;
                    group.entity = next++;
                    group.physical = element.physical;
                    group.lower.fill(std::numeric_limits<double>::infinity());
                    group.upper.fill(-std::numeric_limits<double>::infinity());
                    groups.push_back(std::move(group));
                }
                auto& group = groups[it->second];
                group.element_indices.push_back(i);
                for (const auto node : element_nodes(mesh, element)) {
                    const auto p = mesh.geometry_node_position(node);
                    for (unsigned axis = 0; axis < 3; ++axis) {
                        group.lower[axis] = std::min(group.lower[axis], p[axis]);
                        group.upper[axis] = std::max(group.upper[axis], p[axis]);
                    }
                }
            }
            return groups;
        }

        void write_entities(std::ostream& out, const std::vector<Group>& groups, bool binary)
        {
            std::array<std::uint64_t, 4> counts {};
            for (const auto& group : groups)
                ++counts.at(group.dimension);
            out << "$Entities\n";
            if (binary) {
                for (auto count : counts)
                    write_number(out, count, 8);
            }
            else
                out << counts[0] << ' ' << counts[1] << ' '
                    << counts[2] << ' ' << counts[3] << '\n';

            for (Index dim = 0; dim < 4; ++dim) {
                for (const auto& group : groups) {
                    if (group.dimension != dim)
                        continue;
                    if (binary) {
                        write_int(out, group.entity);
                        for (const auto coordinate : group.lower)
                            write_double(out, coordinate);
                        if (dim > 0)
                            for (const auto coordinate : group.upper)
                                write_double(out, coordinate);
                        write_number(out, group.physical.size(), 8);
                        for (const int tag : group.physical)
                            write_int(out, tag);
                        if (dim > 0)
                            write_number(out, 0, 8); // no BRep boundary links
                    }
                    else {
                        out << group.entity;
                        for (const auto coordinate : group.lower)
                            out << ' ' << coordinate;
                        if (dim > 0)
                            for (const auto coordinate : group.upper)
                                out << ' ' << coordinate;
                        out << ' ' << group.physical.size();
                        for (const int tag : group.physical)
                            out << ' ' << tag;
                        if (dim > 0)
                            out << " 0";
                        out << '\n';
                    }
                }
            }
            out << "\n$EndEntities\n";
        }

        void write_nodes(std::ostream& out, const Mesh& mesh, GmshVersion version,
            bool binary, std::int32_t node_entity)
        {
            const std::uint64_t count = mesh.geometry_node_count();
            if (version == GmshVersion::v22) {
                if (count > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()))
                    throw std::length_error("too many nodes for Gmsh 2.2");
                out << "$Nodes\n" << count << '\n';
                for (std::uint64_t node = 0; node < count; ++node) {
                    const auto p = mesh.geometry_node_position(static_cast<Index>(node));
                    if (binary) {
                        write_int(out, bounded_int(node + 1));
                        for (const auto value : p)
                            write_double(out, value);
                    }
                    else {
                        out << node + 1;
                        for (const auto value : p)
                            out << ' ' << value;
                        out << '\n';
                    }
                }
            }
            else {
                out << "$Nodes\n";
                if (binary) {
                    for (auto header : {std::uint64_t {1}, count, std::uint64_t {1}, count})
                        write_number(out, header, 8);
                    write_int(out, static_cast<std::int32_t>(mesh.dimension()));
                    write_int(out, node_entity);
                    write_int(out, 0); // no parametric coordinates
                    write_number(out, count, 8);
                    for (std::uint64_t node = 1; node <= count; ++node)
                        write_number(out, node, 8);
                    for (std::uint64_t node = 0; node < count; ++node) {
                        for (auto coordinate : mesh.geometry_node_position(static_cast<Index>(node)))
                            write_double(out, coordinate);
                    }
                }
                else {
                    out << "1 " << count << " 1 " << count << '\n'
                        << mesh.dimension() << ' ' << node_entity << " 0 " << count << '\n';
                    for (std::uint64_t node = 1; node <= count; ++node)
                        out << node << '\n';
                    for (std::uint64_t node = 0; node < count; ++node) {
                        const auto p = mesh.geometry_node_position(static_cast<Index>(node));
                        out << p[0] << ' ' << p[1] << ' ' << p[2] << '\n';
                    }
                }
            }
            out << "\n$EndNodes\n";
        }

        void write_elements22(std::ostream& out, const Mesh& mesh,
            const std::vector<OutputElement>& elements, bool binary)
        {
            const auto count = elements.size();
            if (count > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
                throw std::length_error("too many elements for Gmsh 2.2");
            out << "$Elements\n" << count << '\n';
            std::vector<std::size_t> sorted(count);
            std::iota(sorted.begin(), sorted.end(), 0);
            std::stable_sort(sorted.begin(), sorted.end(), [&](std::size_t a, std::size_t b) {
                return elements[a].kind < elements[b].kind;
            });
            std::uint64_t tag = 0;
            for (std::size_t start = 0; start < count;) {
                auto finish = start + 1;
                const auto kind = elements[sorted[start]].kind;
                while (finish < count && elements[sorted[finish]].kind == kind)
                    ++finish;
                if (binary) {
                    write_int(out, static_cast<std::int32_t>(kind));
                    write_int(out, bounded_int(finish - start));
                    write_int(out, 2);
                }
                for (auto i = start; i < finish; ++i) {
                    const auto& element = elements[sorted[i]];
                    const int physical = element.physical.empty() ? 0 : element.physical.front();
                    if (binary) {
                        write_int(out, bounded_int(++tag));
                        write_int(out, physical);
                        write_int(out, 1); // synthethic elementary entity
                        for (const auto node : element_nodes(mesh, element))
                            write_int(out, bounded_int(std::uint64_t(node) + 1));
                    }
                    else {
                        out << ++tag << ' ' << kind << " 2 " << physical << " 1";
                        for (const auto node : element_nodes(mesh, element))
                            out << ' ' << (std::uint64_t(node) + 1);
                        out << '\n';
                    }
                }
                start = finish;
            }
            out << "\n$EndElements\n";
        }

        void write_elements41(std::ostream& out, const Mesh& mesh,
            const std::vector<OutputElement>& elements, const std::vector<Group>& groups,
            bool binary)
        {
            const auto count = elements.size();
            out << "$Elements\n";
            if (binary) {
                for (const auto value : {std::uint64_t(groups.size()),
                         std::uint64_t(count), std::uint64_t {1}, std::uint64_t(count)})
                    write_number(out, value, 8);
            }
            else
                out << groups.size() << ' ' << count << " 1 " << count << '\n';
            std::uint64_t tag = 0;
            for (const auto& group : groups) {
                if (binary) {
                    write_int(out, static_cast<std::int32_t>(group.dimension));
                    write_int(out, group.entity);
                    write_int(out, static_cast<std::int32_t>(group.kind));
                    write_number(out, group.element_indices.size(), 8);
                }
                else
                    out << group.dimension << ' ' << group.entity << ' '
                        << group.kind << ' ' << group.element_indices.size() << '\n';
                for (const auto index : group.element_indices) {
                    if (binary) {
                        write_number(out, ++tag, 8);
                        for (const auto node : element_nodes(mesh, elements[index]))
                            write_number(out, std::uint64_t(node) + 1, 8);
                    }
                    else {
                        out << ++tag;
                        for (const auto node : element_nodes(mesh, elements[index]))
                            out << ' ' << (std::uint64_t(node) + 1);
                        out << '\n';
                    }
                }
            }
            out << "\n$EndElements\n";
        }

    } // namespace

    void write_gmsh(const Mesh& mesh, const std::string& filename, GmshWriteOptions options)
    {
        if (options.version != GmshVersion::v22 && options.version != GmshVersion::v41)
            throw std::invalid_argument("unsupported Gmsh writer version");
        if (options.encoding != GmshEncoding::ascii && options.encoding != GmshEncoding::binary)
            throw std::invalid_argument("unsupported Gmsh writer encoding");
        if (mesh.dimension() < 1 || mesh.dimension() > 3
            || mesh.entity_count(mesh.dimension()) == 0)
            throw std::invalid_argument("Gmsh writer requires a nonempty 1D-3D mesh");

        const bool binary = options.encoding == GmshEncoding::binary;
        const auto elements = collect_elements(mesh, options.version);
        const auto groups = options.version == GmshVersion::v41
            ? group_elements(mesh, elements)
            : std::vector<Group> {};
        const auto top = std::find_if(groups.begin(), groups.end(), [&](const Group& group) {
            return group.dimension == mesh.dimension();
        });
        const auto node_entity = top == groups.end() ? 1 : top->entity;
        std::ofstream out(filename, std::ios::binary);
        if (!out)
            throw std::runtime_error("cannot open Gmsh output file: " + filename);
        out << std::setprecision(std::numeric_limits<double>::max_digits10);
        out << "$MeshFormat\n" << (options.version == GmshVersion::v41 ? "4.1" : "2.2")
            << ' ' << (binary ? 1 : 0) << " 8\n";
        if (binary)
            write_int(out, 1); // little-endian marker
        out << "\n$EndMeshFormat\n";
        write_physical_names(out, mesh);
        if (options.version == GmshVersion::v41)
            write_entities(out, groups, binary);
        write_nodes(out, mesh, options.version, binary, node_entity);
        if (options.version == GmshVersion::v41)
            write_elements41(out, mesh, elements, groups, binary);
        else
            write_elements22(out, mesh, elements, binary);
        out.flush();
        if (!out)
            throw std::runtime_error("failed to write Gmsh mesh: " + filename);
    }

} // namespace idmesh
