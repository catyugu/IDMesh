#include "idmesh/mesh.hpp"

#include <algorithm>
#include <array>
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
        struct Record {
            CellType type;
            Index dimension;
            std::vector<Index> nodes;
            int tag;
        };
        struct Kind {
            CellType type;
            Index nodes;
        };
        Kind keyword_type(const std::string& name)
        {
            if (name == "Edges")
                return {CellType::segment, 2};
            if (name == "Triangles")
                return {CellType::triangle, 3};
            if (name == "Quadrilaterals")
                return {CellType::quadrilateral, 4};
            if (name == "Tetrahedra")
                return {CellType::tetrahedron, 4};
            if (name == "Hexahedra" || name == "Hexaedra")
                return {CellType::hexahedron, 8};
            if (name == "Prisms" || name == "Pentahedra")
                return {CellType::prism, 6};
            if (name == "Pyramids")
                return {CellType::pyramid, 5};
            if (name == "EdgesP2")
                return {CellType::segment, 3};
            if (name == "TrianglesP2")
                return {CellType::triangle, 6};
            if (name == "TetrahedraP2")
                return {CellType::tetrahedron, 10};
            throw std::runtime_error("unsupported Medit element section: " + name);
        }
        const char* keyword_name(CellType type, Index nodes)
        {
            switch (type) {
            case CellType::segment:
                if (nodes == 2)
                    return "Edges";
                if (nodes == 3)
                    return "EdgesP2";
                break;
            case CellType::triangle:
                if (nodes == 3)
                    return "Triangles";
                if (nodes == 6)
                    return "TrianglesP2";
                break;
            case CellType::quadrilateral:
                if (nodes == 4)
                    return "Quadrilaterals";
                break;
            case CellType::tetrahedron:
                if (nodes == 4)
                    return "Tetrahedra";
                if (nodes == 10)
                    return "TetrahedraP2";
                break;
            case CellType::hexahedron:
                if (nodes == 8)
                    return "Hexahedra";
                break;
            case CellType::prism:
                if (nodes == 6)
                    return "Prisms";
                break;
            case CellType::pyramid:
                if (nodes == 5)
                    return "Pyramids";
                break;
            }
            throw std::runtime_error("Medit .mesh has no compatible P2 section for this cell");
        }
        std::istringstream sanitized_input(const std::string& filename)
        {
            std::ifstream f(filename);
            require(bool(f), "cannot open Medit input");
            std::ostringstream out;
            std::string line;
            while (std::getline(f, line)) {
                if (auto i = line.find('#'); i != std::string::npos)
                    line.resize(i);
                out << line << '\n';
            }
            return std::istringstream(out.str());
        }
    } // namespace

    Mesh read_medit(const std::string& filename, BuildOptions options)
    {
        auto in = sanitized_input(filename);
        std::string key;
        Index space = 0;
        int version = 0;
        bool seen_vertices = false;
        std::vector<std::array<double, 3>> positions;
        std::vector<int> vertex_tags;
        std::vector<Record> records;
        while (in >> key) {
            if (key == "MeshVersionFormatted") {
                require(bool(in >> version) && version >= 1 && version <= 2,
                    "unsupported Medit version");
            }
            else if (key == "Dimension") {
                require(bool(in >> space) && (space == 2 || space == 3),
                    "Medit geometric dimension must be 2 or 3");
            }
            else if (key == "Vertices") {
                require(version != 0 && space != 0 && !seen_vertices, "invalid Medit vertex section");
                seen_vertices = true;
                std::size_t count = 0;
                require(bool(in >> count) && count < invalid_index, "invalid Medit vertex count");
                positions.resize(count);
                vertex_tags.resize(count);
                for (std::size_t i = 0; i < count; ++i) {
                    for (Index j = 0; j < space; ++j)
                        require(bool(in >> positions[i][j]), "invalid Medit vertex coordinate");
                    require(bool(in >> vertex_tags[i]), "invalid Medit vertex tag");
                }
            }
            else if (key == "End") {
                break;
            }
            else {
                require(seen_vertices, "Medit elements must follow Vertices");
                const auto kind = keyword_type(key);
                std::size_t count = 0;
                require(bool(in >> count), "invalid Medit element count");
                for (std::size_t j = 0; j < count; ++j) {
                    Record element {kind.type, cell_dimension(kind.type), {}, 0};
                    element.nodes.resize(kind.nodes);
                    for (auto& id : element.nodes) {
                        std::uint64_t vertex = 0;
                        require(bool(in >> vertex) && vertex > 0 && vertex <= positions.size(),
                            "invalid Medit node index");
                        id = static_cast<Index>(vertex - 1);
                    }
                    require(bool(in >> element.tag), "invalid Medit element reference");
                    records.push_back(std::move(element));
                }
            }
        }
        require(version != 0 && seen_vertices && !records.empty(), "incomplete Medit mesh");
        Index dimension = 0;
        for (auto& r : records)
            dimension = std::max(dimension, r.dimension);
        require(dimension <= space, "topological dimension exceeds geometry dimension");
        std::set<Index> corner_points;
        for (auto& r : records)
            for (Index j = 0; j < cell_vertex_count(r.type); ++j)
                corner_points.insert(r.nodes[j]);
        // Vertex reference labels apply to any topological corner.
        for (Index i = 0; i < vertex_tags.size(); ++i)
            if (vertex_tags[i] != 0)
                corner_points.insert(i);
        MeshBuilder builder(dimension, space);
        std::map<Index, Index> corners, extras;
        for (Index id : corner_points)
            corners[id] = builder.add_vertex(positions[id]);
        for (auto& r : records) {
            if (r.dimension != dimension)
                continue;
            std::vector<Index> v, extra;
            const auto nc = cell_vertex_count(r.type);
            for (Index j = 0; j < nc; ++j)
                v.push_back(corners.at(r.nodes[j]));
            for (Index j = nc; j < r.nodes.size(); ++j) {
                auto [it, inserted] = extras.try_emplace(r.nodes[j], invalid_index);
                if (inserted)
                    it->second = builder.add_geometry_node(positions[r.nodes[j]]);
                extra.push_back(it->second);
            }
            if (extra.empty())
                builder.add_cell(r.type, v, r.tag);
            else
                builder.add_quadratic_cell(r.type, v, extra, r.tag);
        }
        for (auto& r : records) {
            if (r.dimension >= dimension || r.tag == 0)
                continue;
            std::vector<Index> v;
            for (Index j = 0; j < cell_vertex_count(r.type); ++j)
                v.push_back(corners.at(r.nodes[j]));
            builder.add_physical_tag(r.dimension, v, r.tag);
        }
        for (Index i = 0; i < vertex_tags.size(); ++i)
            if (vertex_tags[i] != 0) {
                const std::array<Index, 1> v {corners.at(i)};
                if (dimension > 0)
                    builder.add_physical_tag(0, v, vertex_tags[i]);
            }
        return std::move(builder).finalize(options);
    }

    void write_medit(const Mesh& mesh, const std::string& filename)
    {
        struct OutputCell {
            CellType type;
            std::vector<Index> nodes;
            int tag;
        };
        std::vector<OutputCell> cells;
        const Index d = mesh.dimension();
        for (Index c = 0; c < mesh.entity_count(d); ++c) {
            const auto tags = mesh.physical_tags(d, c);
            require(tags.size() <= 1, "Medit cannot store multiple cell physical groups");
            const auto ids = mesh.cell_geometry_nodes(c);
            // Reject before touching the output file.
            (void)keyword_name(mesh.cell_types()[c], static_cast<Index>(ids.size()));
            cells.push_back({mesh.cell_types()[c],
                std::vector<Index>(ids.begin(), ids.end()), tags.empty() ? 0 : tags.front()});
        }
        for (const auto& label : mesh.physical_labels()) {
            if (label.dimension == 0 || label.dimension >= d || label.tag == 0)
                continue;
            CellType type = CellType::segment;
            if (label.dimension == 2)
                type = label.vertices.size() == 3 ? CellType::triangle : CellType::quadrilateral;
            cells.push_back({type,
                std::vector<Index>(label.vertices.begin(), label.vertices.end()), label.tag});
        }
        const Index embedding = std::max<Index>(2, mesh.geometric_dimension());
        std::ofstream out(filename);
        require(bool(out), "cannot open Medit output");
        out << "MeshVersionFormatted 2\nDimension " << embedding << "\n\n";
        out << "Vertices\n"
            << mesh.geometry_node_count() << '\n'
            << std::setprecision(17);
        for (Index v = 0; v < mesh.geometry_node_count(); ++v) {
            const auto p = mesh.geometry_node_position(v);
            for (Index j = 0; j < embedding; ++j)
                out << p[j] << ' ';
            if (v < mesh.entity_count(0)) {
                const auto tags = mesh.physical_tags(0, v);
                require(tags.size() <= 1, "Medit cannot store multiple vertex physical groups");
                out << (tags.empty() ? 0 : tags.front()) << '\n';
            }
            else
                out << "0\n";
        }
        for (const auto type : {CellType::segment, CellType::triangle, CellType::quadrilateral,
                 CellType::tetrahedron, CellType::hexahedron, CellType::prism,
                 CellType::pyramid}) {
            for (auto nodes : {2u, 3u, 4u, 6u, 8u, 10u}) {
                std::size_t count = 0;
                for (auto& c : cells)
                    if (c.type == type && c.nodes.size() == nodes)
                        ++count;
                if (count == 0)
                    continue;
                out << '\n'
                    << keyword_name(type, nodes) << '\n'
                    << count << '\n';
                for (auto& c : cells) {
                    if (c.type != type || c.nodes.size() != nodes)
                        continue;
                    for (auto vertex : c.nodes)
                        out << vertex + 1 << ' ';
                    out << c.tag << '\n';
                }
            }
        }
        out << "\nEnd\n";
        require(bool(out), "Medit write failure");
    }
} // namespace idmesh
