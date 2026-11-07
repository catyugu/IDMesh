#include "contracts.hpp"
#include "idmesh/mesh.hpp"
#include "vtk_common.hpp"

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
        struct CellRecord {
            CellType type;
            Index dimension;
            std::vector<Index> nodes; // VTK point indices in VTK ordering
            int tag = 0;
        };
    } // namespace

    Mesh read_vtk_legacy(const std::string& filename, BuildOptions options)
    {
        std::ifstream in(filename);
        detail::require_runtime(bool(in), "cannot open VTK legacy file");
        std::string line;
        detail::require_runtime(bool(std::getline(in, line)) && line.rfind("# vtk DataFile", 0) == 0,
            "invalid VTK legacy header");
        detail::require_runtime(bool(std::getline(in, line)), "missing VTK title");
        detail::require_runtime(bool(std::getline(in, line)) && line == "ASCII",
            "only ASCII VTK legacy files are supported");
        detail::require_runtime(bool(std::getline(in, line)) && line == "DATASET UNSTRUCTURED_GRID",
            "expected VTK UNSTRUCTURED_GRID");
        std::string keyword, type;
        std::size_t point_count = 0;
        detail::require_runtime(bool(in >> keyword >> point_count >> type) && keyword == "POINTS" && point_count < invalid_index, "invalid VTK POINTS");
        std::vector<std::array<double, 3>> points(point_count);
        for (auto& p : points)
            for (auto& x : p)
                detail::require_runtime(bool(in >> x), "truncated VTK POINTS");
        std::size_t cell_count = 0, words = 0;
        detail::require_runtime(bool(in >> keyword >> cell_count >> words) && keyword == "CELLS" && cell_count < invalid_index, "invalid VTK CELLS");
        std::vector<CellRecord> cells(cell_count);
        std::size_t actual = 0;
        for (auto& cell : cells) {
            std::size_t n = 0;
            detail::require_runtime(bool(in >> n) && n <= 27, "invalid VTK cell arity");
            actual += n + 1;
            cell.nodes.resize(n);
            for (auto& id : cell.nodes) {
                std::uint64_t value = 0;
                detail::require_runtime(bool(in >> value) && value < points.size(), "invalid VTK point ID");
                id = static_cast<Index>(value);
            }
        }
        detail::require_runtime(words == actual, "VTK CELLS word count mismatch");
        std::size_t type_count = 0;
        detail::require_runtime(bool(in >> keyword >> type_count) && keyword == "CELL_TYPES" && type_count == cells.size(), "invalid VTK CELL_TYPES");
        for (auto& cell : cells) {
            int code = 0;
            detail::require_runtime(bool(in >> code), "truncated VTK CELL_TYPES");
            const auto info = detail::vtk::cell_info(code);
            cell.type = info.type;
            cell.dimension = info.dimension;
            detail::require_runtime(
                info.nodes == cell.nodes.size(),
                "invalid VTK cell node count");
        }
        enum class Attribute { none,
            cells,
            points };
        Attribute domain = Attribute::none;
        while (in >> keyword) {
            if (keyword == "CELL_DATA" || keyword == "POINT_DATA") {
                std::size_t count = 0;
                detail::require_runtime(bool(in >> count), "invalid VTK data header");
                domain = keyword == "CELL_DATA" ? Attribute::cells : Attribute::points;
                detail::require_runtime(count == (domain == Attribute::cells ? cells.size() : points.size()),
                    "VTK data size mismatch");
            }
            else if (keyword == "SCALARS") {
                detail::require_runtime(domain != Attribute::none, "SCALARS outside data section");
                std::string name, kind, rest, lookup, table;
                detail::require_runtime(bool(in >> name >> kind), "invalid VTK SCALARS");
                std::getline(in, rest);
                std::size_t components = 1;
                if (auto pos = rest.find_first_not_of(" \t\r"); pos != std::string::npos)
                    components = std::stoul(rest.substr(pos));
                detail::require_runtime(components >= 1 && components <= 32, "unsupported VTK scalar components");
                detail::require_runtime(bool(in >> lookup >> table) && lookup == "LOOKUP_TABLE",
                    "missing VTK scalar lookup table");
                const auto count = domain == Attribute::cells ? cells.size() : points.size();
                for (std::size_t i = 0; i < count; ++i) {
                    for (std::size_t c = 0; c < components; ++c) {
                        double value = 0;
                        detail::require_runtime(bool(in >> value), "truncated VTK scalar array");
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
        detail::require_runtime(dimension >= 1, "VTK has no mesh cells");

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
            const auto gmsh
                = detail::vtk::to_gmsh_order(
                    cell.type, cell.nodes);
            std::vector<Index> vertices, extras;
            for (Index i = 0; i < cell_vertex_count(cell.type); ++i)
                vertices.push_back(corner_ids.at(gmsh[i]));
            for (Index i = cell_vertex_count(cell.type); i < gmsh.size(); ++i) {
                auto [it, inserted] = extra_ids.try_emplace(gmsh[i], invalid_index);
                if (inserted)
                    it->second = builder.add_geometry_node(points[gmsh[i]]);
                extras.push_back(it->second);
            }
            const Index id = extras.empty()
                ? builder.add_cell(cell.type, vertices)
                : builder.add_quadratic_cell(cell.type, vertices, extras);
            if (cell.tag != 0)
                builder.add_cell_label(id, cell.tag);
        }
        for (auto& cell : cells) {
            if (cell.dimension >= dimension || cell.tag == 0)
                continue;
            std::vector<Index> vertices;
            for (Index i = 0; i < cell.nodes.size() && i < (cell.dimension == 0 ? 1u : cell_vertex_count(cell.type)); ++i)
                vertices.push_back(corner_ids.at(cell.nodes[i]));
            builder.add_label(cell.dimension, vertices, cell.tag);
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
        for (const auto& label : mesh.labels())
            detail::require_runtime(label.value != 0,
                "VTK legacy uses zero as the unlabeled sentinel");
        for (Index c = 0; c < mesh.entity_count(d); ++c) {
            const auto groups = mesh.labels(d, c);
            detail::require_runtime(groups.size() <= 1, "VTK legacy cannot store multiple labels on one cell");
            auto encoded = detail::vtk::encode_cell(mesh, c);
            cells.push_back({encoded.code, std::move(encoded.points),
                groups.empty() ? 0 : groups.front().value});
        }
        for (const auto& label : mesh.labels()) {
            if (label.dimension >= d)
                continue;
            const int code = detail::vtk::entity_code(
                label.dimension,
                static_cast<Index>(label.vertices.size()));
            const auto numbering
                = detail::vtk::point_numbering(mesh);
            std::vector<Index> points;
            points.reserve(label.vertices.size());
            for (Index vertex : label.vertices)
                points.push_back(
                    numbering.point_for_geometry(vertex));
            cells.push_back({code, std::move(points), label.value});
        }
        std::ofstream out(filename);
        detail::require_runtime(bool(out), "cannot open VTK output");
        out << "# vtk DataFile Version 3.0\nIDMesh\nASCII\nDATASET UNSTRUCTURED_GRID\n"
            << std::setprecision(17);
        const auto numbering = detail::vtk::point_numbering(mesh);
        out << "POINTS " << numbering.count << " double\n";
        for (Index point = 0; point < numbering.count; ++point) {
            const auto p = mesh.geometry_node_position(
                numbering.geometry_for_point(point));
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
        detail::require_runtime(bool(out), "VTK output failure");
    }
} // namespace idmesh
