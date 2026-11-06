#include "idmesh/mesh.hpp"
#include "second_order.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
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
        struct CellRecord {
            CellType type;
            Index dimension;
            std::vector<Index> nodes; // VTK point indices in VTK ordering
            int tag = 0;
        };
        std::pair<CellType, Index> vtk_type(int code)
        {
            switch (code) {
            case 1:
                return {CellType::segment, 0};
            case 3:
            case 21:
                return {CellType::segment, 1};
            case 5:
            case 22:
                return {CellType::triangle, 2};
            case 9:
            case 23:
            case 28:
                return {CellType::quadrilateral, 2};
            case 10:
            case 24:
                return {CellType::tetrahedron, 3};
            case 12:
            case 25:
            case 29:
                return {CellType::hexahedron, 3};
            case 13:
            case 26:
            case 32:
                return {CellType::prism, 3};
            case 14:
            case 27:
                return {CellType::pyramid, 3};
            default:
                throw std::runtime_error("unsupported VTK cell type");
            }
        }
        Index vtk_node_count(int code)
        {
            switch (code) {
            case 1:
                return 1;
            case 3:
                return 2;
            case 5:
                return 3;
            case 9:
                return 4;
            case 10:
                return 4;
            case 12:
                return 8;
            case 13:
                return 6;
            case 14:
                return 5;
            case 21:
                return 3;
            case 22:
                return 6;
            case 23:
                return 8;
            case 24:
                return 10;
            case 25:
                return 20;
            case 26:
                return 15;
            case 27:
                return 13;
            case 28:
                return 9;
            case 29:
                return 27;
            case 32:
                return 18;
            default:
                throw std::runtime_error("unsupported VTK cell arity");
            }
        }
        int vtk_type(CellType type, Index count)
        {
            switch (type) {
            case CellType::segment:
                if (count == 2)
                    return 3;
                if (count == 3)
                    return 21;
                break;
            case CellType::triangle:
                if (count == 3)
                    return 5;
                if (count == 6)
                    return 22;
                break;
            case CellType::quadrilateral:
                if (count == 4)
                    return 9;
                if (count == 8)
                    return 23;
                if (count == 9)
                    return 28;
                break;
            case CellType::tetrahedron:
                if (count == 4)
                    return 10;
                if (count == 10)
                    return 24;
                break;
            case CellType::hexahedron:
                if (count == 8)
                    return 12;
                if (count == 20)
                    return 25;
                if (count == 27)
                    return 29;
                break;
            case CellType::prism:
                if (count == 6)
                    return 13;
                if (count == 15)
                    return 26;
                if (count == 18)
                    return 32;
                break;
            case CellType::pyramid:
                if (count == 5)
                    return 14;
                if (count == 13)
                    return 27;
                break;
            }
            throw std::runtime_error("no compatible quadratic VTK legacy cell type");
        }
    } // namespace

    Mesh read_vtk_legacy(const std::string& filename, TopologyOptions options)
    {
        std::ifstream in(filename);
        require(bool(in), "cannot open VTK legacy file");
        std::string line;
        require(bool(std::getline(in, line)) && line.rfind("# vtk DataFile", 0) == 0,
            "invalid VTK legacy header");
        require(bool(std::getline(in, line)), "missing VTK title");
        require(bool(std::getline(in, line)) && line == "ASCII",
            "only ASCII VTK legacy files are supported");
        require(bool(std::getline(in, line)) && line == "DATASET UNSTRUCTURED_GRID",
            "expected VTK UNSTRUCTURED_GRID");
        std::string keyword, type;
        std::size_t point_count = 0;
        require(bool(in >> keyword >> point_count >> type) && keyword == "POINTS" && point_count < invalid_index, "invalid VTK POINTS");
        std::vector<std::array<double, 3>> points(point_count);
        for (auto& p : points)
            for (auto& x : p)
                require(bool(in >> x), "truncated VTK POINTS");
        std::size_t cell_count = 0, words = 0;
        require(bool(in >> keyword >> cell_count >> words) && keyword == "CELLS" && cell_count < invalid_index, "invalid VTK CELLS");
        std::vector<CellRecord> cells(cell_count);
        std::size_t actual = 0;
        for (auto& cell : cells) {
            std::size_t n = 0;
            require(bool(in >> n) && n <= 27, "invalid VTK cell arity");
            actual += n + 1;
            cell.nodes.resize(n);
            for (auto& id : cell.nodes) {
                std::uint64_t value = 0;
                require(bool(in >> value) && value < points.size(), "invalid VTK point ID");
                id = static_cast<Index>(value);
            }
        }
        require(words == actual, "VTK CELLS word count mismatch");
        std::size_t type_count = 0;
        require(bool(in >> keyword >> type_count) && keyword == "CELL_TYPES" && type_count == cells.size(), "invalid VTK CELL_TYPES");
        for (auto& cell : cells) {
            int code = 0;
            require(bool(in >> code), "truncated VTK CELL_TYPES");
            const auto [kind, dim] = vtk_type(code);
            cell.type = kind;
            cell.dimension = dim;
            require(vtk_node_count(code) == cell.nodes.size(), "invalid VTK cell node count");
        }
        enum class Attribute { none,
            cells,
            points };
        Attribute domain = Attribute::none;
        while (in >> keyword) {
            if (keyword == "CELL_DATA" || keyword == "POINT_DATA") {
                std::size_t count = 0;
                require(bool(in >> count), "invalid VTK data header");
                domain = keyword == "CELL_DATA" ? Attribute::cells : Attribute::points;
                require(count == (domain == Attribute::cells ? cells.size() : points.size()),
                    "VTK data size mismatch");
            }
            else if (keyword == "SCALARS") {
                require(domain != Attribute::none, "SCALARS outside data section");
                std::string name, kind, rest, lookup, table;
                require(bool(in >> name >> kind), "invalid VTK SCALARS");
                std::getline(in, rest);
                std::size_t components = 1;
                if (auto pos = rest.find_first_not_of(" \t\r"); pos != std::string::npos)
                    components = std::stoul(rest.substr(pos));
                require(components >= 1 && components <= 32, "unsupported VTK scalar components");
                require(bool(in >> lookup >> table) && lookup == "LOOKUP_TABLE",
                    "missing VTK scalar lookup table");
                const auto count = domain == Attribute::cells ? cells.size() : points.size();
                for (std::size_t i = 0; i < count; ++i) {
                    for (std::size_t c = 0; c < components; ++c) {
                        double value = 0;
                        require(bool(in >> value), "truncated VTK scalar array");
                        if (domain == Attribute::cells && name == "physical_tag" && components == 1)
                            cells[i].tag = static_cast<int>(value);
                    }
                }
            }
            else
                throw std::runtime_error("unsupported VTK attribute section");
        }
        Index dimension = 0;
        for (auto& cell : cells)
            dimension = std::max(dimension, cell.dimension);
        require(dimension >= 1, "VTK has no mesh cells");

        // Only topological corners are mesh vertices. Shared quadratic points
        // retain their identity in the independent geometry-node pool.
        std::set<Index> corner_points;
        for (auto& cell : cells) {
            const Index n = cell.dimension == 0 ? 1 : cell_vertex_count(cell.type);
            for (Index j = 0; j < n; ++j)
                corner_points.insert(cell.nodes[j]);
        }
        MeshBuilder builder(dimension, 3);
        std::map<Index, Index> corner_ids, extra_ids;
        for (Index id : corner_points)
            corner_ids[id] = builder.add_vertex(points[id]);

        for (auto& cell : cells) {
            if (cell.dimension != dimension)
                continue;
            std::vector<Index> gmsh = cell.nodes;
            const auto permutation = detail::vtk_from_gmsh(cell.type, static_cast<Index>(gmsh.size()));
            if (!permutation.empty())
                for (Index i = 0; i < gmsh.size(); ++i)
                    gmsh[permutation[i]] = cell.nodes[i];
            std::vector<Index> vertices, extras;
            for (Index i = 0; i < cell_vertex_count(cell.type); ++i)
                vertices.push_back(corner_ids.at(gmsh[i]));
            for (Index i = cell_vertex_count(cell.type); i < gmsh.size(); ++i) {
                auto [it, inserted] = extra_ids.try_emplace(gmsh[i], invalid_index);
                if (inserted)
                    it->second = builder.add_geometry_node(points[gmsh[i]]);
                extras.push_back(it->second);
            }
            if (extras.empty())
                builder.add_cell(cell.type, vertices, cell.tag);
            else
                builder.add_quadratic_cell(cell.type, vertices, extras, cell.tag);
        }
        for (auto& cell : cells) {
            if (cell.dimension >= dimension || cell.tag == 0)
                continue;
            std::vector<Index> vertices;
            for (Index i = 0; i < cell.nodes.size() && i < (cell.dimension == 0 ? 1u : cell_vertex_count(cell.type)); ++i)
                vertices.push_back(corner_ids.at(cell.nodes[i]));
            builder.add_physical_tag(cell.dimension, vertices, cell.tag);
        }
        return std::move(builder).finalize(options);
    }

    void write_vtk_legacy(const Mesh& mesh, const std::string& filename)
    {
        struct OutputCell {
            int code;
            std::vector<Index> nodes;
            int tag;
        };
        std::vector<OutputCell> cells;
        const Index d = mesh.dimension();
        for (Index c = 0; c < mesh.entity_count(d); ++c) {
            const auto groups = mesh.physical_tags(d, c);
            require(groups.size() <= 1, "VTK legacy cannot store multiple cell physical groups");
            const auto ids = mesh.cell_geometry_nodes(c);
            const auto type = mesh.cell_types()[c];
            const int code = vtk_type(type, static_cast<Index>(ids.size()));
            const auto permutation = detail::vtk_from_gmsh(type, static_cast<Index>(ids.size()));
            std::vector<Index> nodes;
            nodes.reserve(ids.size());
            for (Index i = 0; i < ids.size(); ++i)
                nodes.push_back(ids[permutation.empty() ? i : permutation[i]]);
            cells.push_back({code, std::move(nodes), groups.empty() ? 0 : groups.front()});
        }
        for (const auto& label : mesh.physical_labels()) {
            if (label.dimension >= d || label.tag == 0)
                continue;
            int code = 0;
            if (label.dimension == 0)
                code = 1;
            else if (label.dimension == 1)
                code = 3;
            else if (label.dimension == 2)
                code = label.vertices.size() == 3 ? 5 : 9;
            else
                throw std::runtime_error("unsupported physical dimension");
            cells.push_back({code,
                std::vector<Index>(label.vertices.begin(), label.vertices.end()), label.tag});
        }
        std::ofstream out(filename);
        require(bool(out), "cannot open VTK output");
        out << "# vtk DataFile Version 3.0\nIDMesh\nASCII\nDATASET UNSTRUCTURED_GRID\n"
            << std::setprecision(17);
        out << "POINTS " << mesh.geometry_node_count() << " double\n";
        for (Index id = 0; id < mesh.geometry_node_count(); ++id) {
            const auto p = mesh.geometry_node_position(id);
            out << p[0] << ' ' << p[1] << ' ' << p[2] << '\n';
        }
        std::size_t words = 0;
        for (auto& cell : cells)
            words += cell.nodes.size() + 1;
        out << "CELLS " << cells.size() << ' ' << words << '\n';
        for (auto& cell : cells) {
            out << cell.nodes.size();
            for (auto id : cell.nodes)
                out << ' ' << id;
            out << '\n';
        }
        out << "CELL_TYPES " << cells.size() << '\n';
        for (auto& cell : cells)
            out << cell.code << '\n';
        out << "CELL_DATA " << cells.size() << "\nSCALARS physical_tag int 1\n"
                                               "LOOKUP_TABLE default\n";
        for (auto& cell : cells)
            out << cell.tag << '\n';
        require(bool(out), "VTK output failure");
    }
} // namespace idmesh
