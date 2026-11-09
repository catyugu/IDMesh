#include "contracts.hpp"
#include "idmesh/io.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace idmesh {
    namespace {

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


        // Binary 2.2 uses 32-bit signed tags. Binary 4.1 uses 32-bit
        // classification tags and data-size-wide unsigned node/element tags.
        // Decode explicitly: files are portable across host byte orders.
        struct BinaryReader {
            std::istream& input;
            bool little_endian;
            unsigned size_width;

            std::uint64_t unsigned_number(unsigned width)
            {
                detail::require_runtime(width == 4 || width == 8,
                    "unsupported Gmsh binary integer width");
                std::array<unsigned char, 8> bytes {};
                input.read(reinterpret_cast<char*>(bytes.data()), width);
                detail::require_runtime(static_cast<bool>(input), "truncated Gmsh binary payload");
                std::uint64_t number = 0;
                if (little_endian) {
                    for (unsigned i = 0; i < width; ++i)
                        number |= std::uint64_t(bytes[i]) << (8 * i);
                }
                else {
                    for (unsigned i = 0; i < width; ++i)
                        number = (number << 8) | bytes[i];
                }
                return number;
            }

            std::int32_t integer()
            {
                return std::bit_cast<std::int32_t>(
                    static_cast<std::uint32_t>(unsigned_number(4)));
            }

            std::uint64_t size() { return unsigned_number(size_width); }

            double real() { return std::bit_cast<double>(unsigned_number(8)); }
        };

        void expect_binary_newline(std::istream& input)
        {
            if (input.peek() == '\r')
                input.get();
            detail::require_runtime(input.get() == '\n',
                "missing newline before Gmsh binary data");
        }

        std::uint64_t positive_id(std::int32_t value)
        {
            detail::require_runtime(value > 0, "invalid Gmsh binary tag");
            return static_cast<std::uint64_t>(value);
        }

        std::uint64_t positive_size_id(std::uint64_t value)
        {
            detail::require_runtime(value != 0, "invalid Gmsh binary tag");
            return value;
        }

        void expect_end(std::istream& input, const std::string& section)
        {
            std::string marker;
            detail::require_runtime(static_cast<bool>(input >> marker) && marker == "$End" + section.substr(1),
                "missing or incorrect Gmsh section terminator");
        }

    } // namespace

    Mesh read_gmsh(const std::string& filename, BuildOptions options)
    {
        std::ifstream in(filename, std::ios::binary);
        if (!in)
            throw std::runtime_error("cannot open Gmsh file: " + filename);

        double version = 0;
        bool binary = false;
        std::optional<BinaryReader> reader;
        bool seen_format = false, seen_nodes = false, seen_elements = false, seen_entities = false;
        std::map<std::uint64_t, std::array<double, 3>> nodes;
        std::vector<Element> elements;
        std::map<EntityKey, std::vector<int>> entity_physical;
        std::map<std::pair<Index, int>, std::string> physical_names;

        std::string section;
        while (in >> section) {
            if (section == "$MeshFormat") {
                detail::require_runtime(!seen_format, "duplicate MeshFormat section");
                seen_format = true;
                unsigned format = 0, data_size = 0;
                detail::require_runtime(static_cast<bool>(in >> version >> format >> data_size),
                    "invalid MeshFormat header");
                detail::require_runtime((format == 0 || format == 1)
                        && ((version == 2.2 && data_size == 8)
                            || (version == 4.1 && (data_size == 4 || data_size == 8))),
                    "only Gmsh MSH 2.2 and 4.1 ASCII/binary are supported");
                binary = format == 1;
                if (binary) {
                    expect_binary_newline(in);
                    std::array<unsigned char, 4> marker {};
                    in.read(reinterpret_cast<char*>(marker.data()), marker.size());
                    detail::require_runtime(static_cast<bool>(in),
                        "truncated Gmsh binary endian marker");
                    const bool little = marker == std::array<unsigned char, 4> {1, 0, 0, 0};
                    const bool big = marker == std::array<unsigned char, 4> {0, 0, 0, 1};
                    detail::require_runtime(little || big, "invalid Gmsh binary endian marker");
                    reader.emplace(BinaryReader {in, little, data_size});
                }
                expect_end(in, section);
            }
            else if (section == "$PhysicalNames") {
                detail::require_runtime(seen_format, "MeshFormat must precede PhysicalNames");
                std::uint64_t count = 0;
                detail::require_runtime(static_cast<bool>(in >> count), "invalid PhysicalNames header");
                for (std::uint64_t i = 0; i < count; ++i) {
                    Index dim = 0;
                    int tag = 0;
                    std::string name;
                    detail::require_runtime(static_cast<bool>(in >> dim >> tag >> std::quoted(name)),
                        "invalid PhysicalNames entry");
                    detail::require_runtime(dim <= 3, "invalid PhysicalNames dimension");
                    physical_names[{dim, tag}] = std::move(name);
                }
                expect_end(in, section);
            }
            else if (section == "$Entities") {
                detail::require_runtime(version >= 4.1 && !seen_entities, "invalid duplicate Entities section");
                seen_entities = true;
                std::array<std::uint64_t, 4> counts {};
                if (binary) {
                    expect_binary_newline(in);
                    for (auto& count : counts)
                        count = reader->size();
                    for (unsigned dim = 0; dim <= 3; ++dim) {
                        for (std::uint64_t i = 0; i < counts[dim]; ++i) {
                            const auto id = positive_id(reader->integer());
                            for (unsigned axis = 0; axis < (dim == 0 ? 3U : 6U); ++axis)
                                (void)reader->real();
                            const auto group_count = reader->size();
                            detail::require_runtime(group_count <= std::numeric_limits<Index>::max(),
                                "Gmsh physical group count exceeds supported range");
                            std::vector<int> groups;
                            groups.reserve(static_cast<std::size_t>(group_count));
                            for (std::uint64_t j = 0; j < group_count; ++j)
                                groups.push_back(reader->integer());
                            if (dim != 0) {
                                const auto bounds = reader->size();
                                detail::require_runtime(bounds <= std::numeric_limits<Index>::max(),
                                    "Gmsh bounding entity count exceeds supported range");
                                for (std::uint64_t j = 0; j < bounds; ++j)
                                    (void)reader->integer();
                            }
                            detail::require_runtime(
                                entity_physical.emplace(EntityKey {dim, id}, std::move(groups)).second,
                                "duplicate geometrical entity tag");
                        }
                    }
                }
                else {
                    for (auto& count : counts)
                        detail::require_runtime(static_cast<bool>(in >> count), "invalid Entities header");
                    std::string line;
                    std::getline(in, line);
                    for (unsigned dim = 0; dim <= 3; ++dim) {
                        for (std::uint64_t i = 0; i < counts[dim]; ++i) {
                            detail::require_runtime(static_cast<bool>(std::getline(in, line)), "truncated Entities");
                            std::istringstream record(line);
                            std::uint64_t id = 0;
                            double ignored = 0;
                            detail::require_runtime(static_cast<bool>(record >> id), "invalid entity tag");
                            for (unsigned component = 0; component < (dim == 0 ? 3U : 6U); ++component)
                                detail::require_runtime(static_cast<bool>(record >> ignored), "invalid entity bounds");
                            std::uint64_t group_count = 0;
                            detail::require_runtime(static_cast<bool>(record >> group_count), "invalid physical group count");
                            std::vector<int> groups(group_count);
                            for (auto& group : groups)
                                detail::require_runtime(static_cast<bool>(record >> group), "invalid physical group");
                            detail::require_runtime(entity_physical.emplace(EntityKey {dim, id}, std::move(groups)).second,
                                "duplicate geometrical entity tag");
                        }
                    }
                }
                expect_end(in, section);
            }
            else if (section == "$Nodes") {
                detail::require_runtime(seen_format && !seen_nodes, "missing format or duplicate Nodes");
                seen_nodes = true;
                if (version < 4) {
                    std::uint64_t count = 0;
                    detail::require_runtime(static_cast<bool>(in >> count), "invalid Nodes header");
                    detail::require_runtime(count < invalid_index,
                        "Gmsh node count exceeds supported range");
                    if (binary)
                        expect_binary_newline(in);
                    for (std::uint64_t i = 0; i < count; ++i) {
                        std::uint64_t id = 0;
                        std::array<double, 3> position {};
                        if (binary) {
                            id = positive_id(reader->integer());
                            for (auto& coordinate : position)
                                coordinate = reader->real();
                        }
                        else {
                            detail::require_runtime(static_cast<bool>(in >> id >> position[0]
                                    >> position[1] >> position[2]),
                                "truncated Gmsh nodes");
                        }
                        detail::require_runtime(nodes.emplace(id, position).second, "duplicate Gmsh node tag");
                    }
                }
                else {
                    std::uint64_t blocks = 0, total = 0, min_tag = 0, max_tag = 0;
                    if (binary) {
                        expect_binary_newline(in);
                        blocks = reader->size();
                        total = reader->size();
                        min_tag = reader->size();
                        max_tag = reader->size();
                    }
                    else {
                        detail::require_runtime(static_cast<bool>(in >> blocks >> total >> min_tag >> max_tag),
                            "invalid Nodes block header");
                    }
                    detail::require_runtime(total < invalid_index && blocks <= total,
                        "invalid Gmsh node count");
                    for (std::uint64_t block = 0; block < blocks; ++block) {
                        unsigned dim = 0, parametric = 0;
                        std::uint64_t entity = 0, count = 0;
                        if (binary) {
                            const auto raw_dim = reader->integer();
                            const auto raw_entity = reader->integer();
                            const auto raw_parametric = reader->integer();
                            detail::require_runtime(raw_dim >= 0 && raw_entity > 0
                                    && (raw_parametric == 0 || raw_parametric == 1),
                                "invalid Gmsh node classification");
                            dim = static_cast<unsigned>(raw_dim);
                            entity = static_cast<std::uint64_t>(raw_entity);
                            parametric = static_cast<unsigned>(raw_parametric);
                            count = reader->size();
                        }
                        else {
                            detail::require_runtime(static_cast<bool>(in >> dim >> entity >> parametric >> count),
                                "invalid Nodes block");
                        }
                        detail::require_runtime(dim <= 3 && parametric <= 1 && count <= total - nodes.size(),
                            "invalid Gmsh node classification or block count");
                        std::vector<std::uint64_t> tags(static_cast<std::size_t>(count));
                        for (auto& tag : tags) {
                            if (binary)
                                tag = positive_size_id(reader->size());
                            else
                                detail::require_runtime(static_cast<bool>(in >> tag), "missing node tag");
                        }
                        for (auto tag : tags) {
                            std::array<double, 3> position {};
                            for (auto& coordinate : position) {
                                if (binary)
                                    coordinate = reader->real();
                                else
                                    detail::require_runtime(static_cast<bool>(in >> coordinate), "missing Gmsh node coordinate");
                            }
                            for (unsigned i = 0; i < (parametric ? dim : 0); ++i) {
                                if (binary)
                                    (void)reader->real();
                                else {
                                    double parameter = 0;
                                    detail::require_runtime(static_cast<bool>(in >> parameter), "missing parametric coordinate");
                                }
                            }
                            detail::require_runtime(nodes.emplace(tag, position).second, "duplicate Gmsh node tag");
                        }
                    }
                    detail::require_runtime(nodes.size() == total, "Gmsh node count mismatch");
                }
                expect_end(in, section);
            }
            else if (section == "$Elements") {
                detail::require_runtime(seen_format && !seen_elements, "missing format or duplicate Elements");
                seen_elements = true;
                if (version < 4) {
                    std::uint64_t count = 0;
                    detail::require_runtime(static_cast<bool>(in >> count), "invalid Elements header");
                    detail::require_runtime(count < invalid_index,
                        "Gmsh element count exceeds supported range");
                    if (binary)
                        expect_binary_newline(in);
                    std::uint64_t read = 0;
                    if (binary) {
                        while (read < count) {
                            const auto kind = reader->integer();
                            const auto block_count = reader->integer();
                            const auto n_tags = reader->integer();
                            detail::require_runtime(kind > 0 && block_count > 0 && n_tags >= 0
                                    && static_cast<std::uint64_t>(block_count) <= count - read
                                    && n_tags <= 1024,
                                "invalid Gmsh binary element block");
                            const auto info = element_info(static_cast<unsigned>(kind));
                            for (int j = 0; j < block_count; ++j) {
                                (void)positive_id(reader->integer());
                                Element element {info, {}, {}};
                                for (int k = 0; k < n_tags; ++k) {
                                    const auto tag = reader->integer();
                                    if (k == 0 && tag != 0)
                                        element.physical.push_back(tag);
                                }
                                element.nodes.resize(info.nodes);
                                for (auto& node : element.nodes)
                                    node = positive_id(reader->integer());
                                elements.push_back(std::move(element));
                            }
                            read += static_cast<std::uint64_t>(block_count);
                        }
                    }
                    else {
                        for (; read < count; ++read) {
                            std::uint64_t id = 0;
                            unsigned kind = 0, n_tags = 0;
                            detail::require_runtime(static_cast<bool>(in >> id >> kind >> n_tags),
                                "invalid Gmsh element");
                            const auto info = element_info(kind);
                            Element element {info, {}, {}};
                            for (unsigned j = 0; j < n_tags; ++j) {
                                int tag = 0;
                                detail::require_runtime(static_cast<bool>(in >> tag), "truncated Gmsh element tags");
                                if (j == 0 && tag != 0)
                                    element.physical.push_back(tag);
                            }
                            element.nodes.resize(info.nodes);
                            for (auto& node : element.nodes)
                                detail::require_runtime(static_cast<bool>(in >> node), "truncated element connectivity");
                            elements.push_back(std::move(element));
                        }
                    }
                }
                else {
                    std::uint64_t blocks = 0, count = 0, min_tag = 0, max_tag = 0;
                    if (binary) {
                        expect_binary_newline(in);
                        blocks = reader->size();
                        count = reader->size();
                        min_tag = reader->size();
                        max_tag = reader->size();
                    }
                    else {
                        detail::require_runtime(static_cast<bool>(in >> blocks >> count >> min_tag >> max_tag),
                            "invalid Elements block header");
                    }
                    detail::require_runtime(count < invalid_index && blocks <= count,
                        "invalid Gmsh element count");
                    std::uint64_t read = 0;
                    for (std::uint64_t block = 0; block < blocks; ++block) {
                        unsigned dim = 0, kind = 0;
                        std::uint64_t entity = 0, n = 0;
                        if (binary) {
                            const auto raw_dim = reader->integer();
                            const auto raw_entity = reader->integer();
                            const auto raw_kind = reader->integer();
                            detail::require_runtime(raw_dim >= 0 && raw_entity > 0 && raw_kind > 0,
                                "invalid Gmsh binary element block header");
                            dim = static_cast<unsigned>(raw_dim);
                            entity = static_cast<std::uint64_t>(raw_entity);
                            kind = static_cast<unsigned>(raw_kind);
                            n = reader->size();
                        }
                        else {
                            detail::require_runtime(static_cast<bool>(in >> dim >> entity >> kind >> n),
                                "invalid Elements block");
                        }
                        detail::require_runtime(n <= count - read, "invalid Gmsh element block count");
                        const auto info = element_info(kind);
                        detail::require_runtime(info.dimension == dim, "Gmsh element dimension mismatch");
                        const auto found = entity_physical.find({dim, entity});
                        const std::vector<int> groups = found == entity_physical.end()
                            ? std::vector<int> {}
                            : found->second;
                        for (std::uint64_t j = 0; j < n; ++j) {
                            std::uint64_t id = 0;
                            if (binary)
                                id = positive_size_id(reader->size());
                            else
                                detail::require_runtime(static_cast<bool>(in >> id), "truncated element tag");
                            Element element {info, {}, groups};
                            element.nodes.resize(info.nodes);
                            for (auto& node : element.nodes) {
                                if (binary)
                                    node = positive_size_id(reader->size());
                                else
                                    detail::require_runtime(static_cast<bool>(in >> node),
                                        "truncated element connectivity");
                            }
                            elements.push_back(std::move(element));
                        }
                        read += n;
                    }
                    detail::require_runtime(read == count, "Gmsh element count mismatch");
                }
                expect_end(in, section);
            }
            else if (!section.empty() && section.front() == '$' && section.rfind("$End", 0) != 0) {
                const auto end = "$End" + section.substr(1);
                std::string line;
                std::getline(in, line);
                while (std::getline(in, line) && line != end) { }
                detail::require_runtime(line == end, "unterminated Gmsh section");
            }
            else {
                throw std::runtime_error("unexpected Gmsh section delimiter: " + section);
            }
        }
        detail::require_runtime(seen_format && seen_nodes && seen_elements, "missing required Gmsh section");
        detail::require_runtime(!nodes.empty() && !elements.empty(), "empty Gmsh mesh");
        Index dimension = 0;
        for (const auto& element : elements)
            dimension = std::max(dimension, element.info.dimension);
        detail::require_runtime(dimension >= 1 && dimension <= 3, "mesh needs at least one 1D-3D element");

        // High-order geometric nodes never become topological mesh vertices. Even
        // isolated lower-dimensional physical entities keep their corner vertices
        // for a lossless physical-label key.
        std::set<std::uint64_t> corner_tags;
        for (const auto& element : elements) {
            for (Index j = 0; j < element.info.corners; ++j) {
                detail::require_runtime(nodes.contains(element.nodes[j]), "element references missing Gmsh node");
                corner_tags.insert(element.nodes[j]);
            }
            for (auto tag : element.nodes)
                detail::require_runtime(nodes.contains(tag), "element references missing Gmsh node");
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
            Index cell = invalid_index;
            if (element.info.nodes == element.info.corners) {
                cell = builder.add_cell(element.info.type, vertices);
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
                cell = builder.add_quadratic_cell(element.info.type, vertices, extras);
            }
            for (const int physical : element.physical)
                builder.add_cell_label(cell, physical);
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
                builder.add_label(element.info.dimension, vertices, physical);
        }
        for (auto& [key, name] : physical_names)
            if (key.first <= dimension)
                builder.set_label_name(key.first, key.second, std::move(name));
        return std::move(builder).finalize(options);
    }

} // namespace idmesh
