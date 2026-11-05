#include "idmesh/mesh.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace idmesh {
namespace {

void require(bool condition, const char* reason)
{
    if (!condition) throw std::runtime_error(reason);
}

struct VtkCell {
    CellType type = CellType::segment;
    Index dimension = 0;
    std::vector<Index> vertices;
    int tag = 0;
};

std::pair<CellType, Index> vtk_type(int code)
{
    switch (code) {
    case 1: return {CellType::segment, 0}; // labeled point
    case 3: return {CellType::segment, 1};
    case 5: return {CellType::triangle, 2};
    case 9: return {CellType::quadrilateral, 2};
    case 10: return {CellType::tetrahedron, 3};
    case 12: return {CellType::hexahedron, 3};
    case 13: return {CellType::prism, 3};
    case 14: return {CellType::pyramid, 3};
    default: throw std::runtime_error("unsupported VTK legacy cell type");
    }
}

int vtk_type(CellType type)
{
    switch (type) {
    case CellType::segment: return 3;
    case CellType::triangle: return 5;
    case CellType::quadrilateral: return 9;
    case CellType::tetrahedron: return 10;
    case CellType::hexahedron: return 12;
    case CellType::prism: return 13;
    case CellType::pyramid: return 14;
    }
    throw std::runtime_error("unsupported VTK cell type");
}
} // namespace

Mesh read_vtk_legacy(const std::string& filename, TopologyOptions options)
{
    std::ifstream in(filename);
    require(static_cast<bool>(in), "cannot open VTK legacy file");
    std::string line;
    require(static_cast<bool>(std::getline(in, line)) &&
            line.rfind("# vtk DataFile", 0) == 0, "invalid VTK legacy header");
    require(static_cast<bool>(std::getline(in, line)), "missing VTK title");
    require(static_cast<bool>(std::getline(in, line)) && line == "ASCII",
            "only ASCII VTK legacy files are supported");
    require(static_cast<bool>(std::getline(in, line)) &&
            line == "DATASET UNSTRUCTURED_GRID", "expected VTK UNSTRUCTURED_GRID");

    std::string keyword, numeric_type;
    std::size_t point_count = 0;
    require(static_cast<bool>(in >> keyword >> point_count >> numeric_type) &&
            keyword == "POINTS", "missing VTK POINTS");
    require(point_count < invalid_index, "too many VTK points");
    std::vector<std::array<double, 3>> points(point_count);
    for (auto& position : points)
        for (double& coordinate : position)
            require(static_cast<bool>(in >> coordinate), "truncated VTK points");

    std::size_t cell_count = 0, connectivity_size = 0;
    require(static_cast<bool>(in >> keyword >> cell_count >> connectivity_size) &&
            keyword == "CELLS", "missing VTK CELLS");
    require(cell_count < invalid_index, "too many VTK cells");
    std::vector<VtkCell> cells(cell_count);
    std::size_t actual_size = 0;
    for (auto& cell : cells) {
        std::size_t n = 0;
        require(static_cast<bool>(in >> n) && n <= 8, "invalid VTK cell arity");
        ++actual_size;
        cell.vertices.resize(n);
        for (auto& node : cell.vertices) {
            std::uint64_t index = 0;
            require(static_cast<bool>(in >> index) && index < points.size(),
                    "invalid VTK connectivity");
            node = static_cast<Index>(index);
            ++actual_size;
        }
    }
    require(actual_size == connectivity_size, "VTK CELLS size mismatch");
    std::size_t types_count = 0;
    require(static_cast<bool>(in >> keyword >> types_count) &&
            keyword == "CELL_TYPES" && types_count == cells.size(),
            "missing or invalid VTK CELL_TYPES");
    for (auto& cell : cells) {
        int code = 0;
        require(static_cast<bool>(in >> code), "truncated VTK cell type");
        const auto [type, dim] = vtk_type(code);
        cell.type = type;
        cell.dimension = dim;
        require(cell.vertices.size() == (dim == 0 ? 1u : cell_vertex_count(type)),
                "VTK cell arity does not match type");
    }

    // Other scalar arrays are parsed and ignored. The physical_tag array is
    // explicitly mapped to our dimension-independent physical-group model.
    enum class AttributeDomain { none, cells, points };
    AttributeDomain domain = AttributeDomain::none;
    while (in >> keyword) {
        if (keyword == "CELL_DATA" || keyword == "POINT_DATA") {
            std::size_t count = 0;
            require(static_cast<bool>(in >> count), "invalid VTK data size");
            domain = keyword == "CELL_DATA" ? AttributeDomain::cells : AttributeDomain::points;
            require(count == (domain == AttributeDomain::cells ? cells.size() : points.size()),
                    "VTK attribute count mismatch");
        } else if (keyword == "SCALARS") {
            require(domain != AttributeDomain::none, "scalar array without domain");
            std::string name, type, tail;
            require(static_cast<bool>(in >> name >> type), "invalid scalar header");
            std::getline(in, tail);
            std::size_t components = 1;
            if (!tail.empty()) {
                const auto first = tail.find_first_not_of(" \t\r");
                if (first != std::string::npos)
                    components = static_cast<std::size_t>(std::stoul(tail.substr(first)));
            }
            require(components >= 1 && components <= 32, "invalid scalar components");
            std::string lookup, table;
            require(static_cast<bool>(in >> lookup >> table) && lookup == "LOOKUP_TABLE",
                    "missing VTK scalar lookup table");
            const auto count = domain == AttributeDomain::cells ? cells.size() : points.size();
            for (std::size_t i = 0; i < count; ++i) {
                for (std::size_t c = 0; c < components; ++c) {
                    double value = 0;
                    require(static_cast<bool>(in >> value), "truncated VTK scalar data");
                    if (domain == AttributeDomain::cells && name == "physical_tag" &&
                        components == 1)
                        cells[i].tag = static_cast<int>(value);
                }
            }
        } else {
            throw std::runtime_error("unsupported VTK attribute section");
        }
    }

    Index dimension = 0;
    for (const auto& cell : cells) dimension = std::max(dimension, cell.dimension);
    require(dimension >= 1, "VTK file has no supported mesh cells");
    MeshBuilder builder(dimension, 3);
    for (auto position : points) builder.add_vertex(position);
    for (const auto& cell : cells)
        if (cell.dimension == dimension)
            builder.add_cell(cell.type, cell.vertices, cell.tag);
    for (const auto& cell : cells)
        if (cell.dimension < dimension && cell.tag != 0)
            builder.add_physical_tag(cell.dimension, cell.vertices, cell.tag);
    return std::move(builder).finalize(options);
}

void write_vtk_legacy(const Mesh& mesh, const std::string& filename)
{
    struct OutputCell {
        int code;
        std::vector<Index> vertices;
        int tag;
    };
    std::vector<OutputCell> cells;
    const Index d = mesh.dimension();
    for (Index c = 0; c < mesh.entity_count(d); ++c) {
        if (mesh.cell_order(c) != 1)
            throw std::runtime_error("VTK legacy writer supports linear cells only");
        const auto groups = mesh.physical_tags(d, c);
        require(groups.size() <= 1, "VTK legacy cannot store multiple top-cell physical groups");
        const auto v = mesh.entity_vertices(d, c);
        cells.push_back({vtk_type(mesh.cell_types()[c]),
            std::vector<Index>(v.begin(), v.end()),
            groups.empty() ? 0 : groups.front()});
    }
    for (const auto& label : mesh.physical_labels()) {
        if (label.dimension >= d || label.tag == 0) continue;
        int code = 0;
        if (label.dimension == 0) code = 1;
        else if (label.dimension == 1) code = 3;
        else if (label.dimension == 2)
            code = label.vertices.size() == 3 ? 5 : 9;
        else throw std::runtime_error("unsupported physical entity");
        cells.push_back({code, label.vertices, label.tag});
    }
    std::ofstream out(filename);
    require(static_cast<bool>(out), "cannot open VTK legacy output file");
    out << "# vtk DataFile Version 3.0\nIDMesh\nASCII\nDATASET UNSTRUCTURED_GRID\n";
    out << std::setprecision(17);
    out << "POINTS " << mesh.coordinates().size() << " double\n";
    for (const auto& point : mesh.coordinates())
        out << point[0] << ' ' << point[1] << ' ' << point[2] << '\n';
    std::size_t size = 0;
    for (const auto& cell : cells) size += cell.vertices.size() + 1;
    out << "CELLS " << cells.size() << ' ' << size << '\n';
    for (const auto& cell : cells) {
        out << cell.vertices.size();
        for (auto vertex : cell.vertices) out << ' ' << vertex;
        out << '\n';
    }
    out << "CELL_TYPES " << cells.size() << '\n';
    for (const auto& cell : cells) out << cell.code << '\n';
    out << "CELL_DATA " << cells.size() << "\nSCALARS physical_tag int 1\nLOOKUP_TABLE default\n";
    for (const auto& cell : cells) out << cell.tag << '\n';
    require(static_cast<bool>(out), "VTK legacy write failure");
}

} // namespace idmesh
