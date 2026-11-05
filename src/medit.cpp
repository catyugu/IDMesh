#include "idmesh/mesh.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace idmesh {
namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

struct MeditCell {
    CellType type;
    std::vector<Index> vertices;
    int tag;
};

struct MeditKind {
    CellType type;
    Index arity;
};

MeditKind medit_kind(const std::string& name)
{
    if (name == "Edges") return {CellType::segment, 2};
    if (name == "Triangles") return {CellType::triangle, 3};
    if (name == "Quadrilaterals") return {CellType::quadrilateral, 4};
    if (name == "Tetrahedra") return {CellType::tetrahedron, 4};
    if (name == "Hexahedra" || name == "Hexaedra") return {CellType::hexahedron, 8};
    if (name == "Prisms" || name == "Pentahedra") return {CellType::prism, 6};
    if (name == "Pyramids") return {CellType::pyramid, 5};
    throw std::runtime_error("unsupported Medit element section: " + name);
}

const char* medit_name(CellType type)
{
    switch (type) {
    case CellType::segment: return "Edges";
    case CellType::triangle: return "Triangles";
    case CellType::quadrilateral: return "Quadrilaterals";
    case CellType::tetrahedron: return "Tetrahedra";
    case CellType::hexahedron: return "Hexahedra";
    case CellType::prism: return "Prisms";
    case CellType::pyramid: return "Pyramids";
    }
    throw std::runtime_error("unknown Medit cell type");
}

std::istringstream sanitized_input(const std::string& filename)
{
    std::ifstream file(filename);
    require(static_cast<bool>(file), "cannot open Medit file");
    std::ostringstream cleaned;
    std::string line;
    while (std::getline(file, line)) {
        const auto comment = line.find('#');
        if (comment != std::string::npos) line.resize(comment);
        cleaned << line << '\n';
    }
    return std::istringstream(cleaned.str());
}

} // namespace

Mesh read_medit(const std::string& filename, TopologyOptions options)
{
    auto in = sanitized_input(filename);
    std::string keyword;
    Index space_dimension = 0;
    int version = 0;
    bool seen_vertices = false;
    std::vector<std::array<double, 3>> positions;
    std::vector<int> vertex_tags;
    std::vector<MeditCell> elements;
    while (in >> keyword) {
        if (keyword == "MeshVersionFormatted") {
            require(static_cast<bool>(in >> version) && version >= 1 && version <= 2,
                    "unsupported Medit version");
        } else if (keyword == "Dimension") {
            require(static_cast<bool>(in >> space_dimension)
                    && (space_dimension == 2 || space_dimension == 3),
                    "Medit supports geometric dimension 2 or 3");
        } else if (keyword == "Vertices") {
            require(version != 0 && space_dimension != 0 && !seen_vertices,
                    "invalid Medit vertex section");
            seen_vertices = true;
            std::size_t count = 0;
            require(static_cast<bool>(in >> count) && count < invalid_index,
                    "invalid Medit vertex count");
            positions.resize(count);
            vertex_tags.resize(count);
            for (std::size_t i = 0; i < count; ++i) {
                for (Index j = 0; j < space_dimension; ++j)
                    require(static_cast<bool>(in >> positions[i][j]),
                            "invalid Medit vertex coordinates");
                require(static_cast<bool>(in >> vertex_tags[i]), "missing Medit vertex reference");
            }
        } else if (keyword == "End") {
            break;
        } else {
            const auto kind = medit_kind(keyword);
            std::size_t count = 0;
            require(static_cast<bool>(in >> count), "invalid Medit element count");
            for (std::size_t j = 0; j < count; ++j) {
                MeditCell element {kind.type, {}, 0};
                for (Index i = 0; i < kind.arity; ++i) {
                    std::uint64_t vertex = 0;
                    require(static_cast<bool>(in >> vertex) && vertex > 0 &&
                            vertex <= positions.size(), "invalid Medit element vertex");
                    element.vertices.push_back(static_cast<Index>(vertex - 1));
                }
                require(static_cast<bool>(in >> element.tag), "missing Medit element reference");
                elements.push_back(std::move(element));
            }
        }
    }
    require(version != 0 && seen_vertices && !elements.empty(), "incomplete Medit mesh");
    Index dimension = 0;
    for (const auto& element : elements)
        dimension = std::max(dimension, cell_dimension(element.type));
    require(dimension <= space_dimension, "Medit element exceeds geometric dimension");

    MeshBuilder builder(dimension, space_dimension);
    for (const auto& position : positions) builder.add_vertex(position);
    for (const auto& element : elements)
        if (cell_dimension(element.type) == dimension)
            builder.add_cell(element.type, element.vertices, element.tag);
    for (const auto& element : elements)
        if (cell_dimension(element.type) < dimension && element.tag != 0)
            builder.add_physical_tag(cell_dimension(element.type), element.vertices, element.tag);
    for (Index i = 0; i < vertex_tags.size(); ++i)
        if (vertex_tags[i] != 0) {
            const std::array<Index, 1> vertex {i};
            builder.add_physical_tag(0, vertex, vertex_tags[i]);
        }
    return std::move(builder).finalize(options);
}

void write_medit(const Mesh& mesh, const std::string& filename)
{
    struct OutputCell {
        CellType type;
        std::vector<Index> vertices;
        int tag;
    };
    const Index dimension = mesh.dimension();
    std::vector<OutputCell> output;
    for (Index c = 0; c < mesh.entity_count(dimension); ++c) {
        if (mesh.cell_order(c) != 1)
            throw std::runtime_error("Medit writer supports linear cells only");
        const auto tags = mesh.physical_tags(dimension, c);
        require(tags.size() <= 1, "Medit cannot store multiple physical groups on one cell");
        const auto nodes = mesh.entity_vertices(dimension, c);
        output.push_back({mesh.cell_types()[c],
            std::vector<Index>(nodes.begin(), nodes.end()),
            tags.empty() ? 0 : tags.front()});
    }
    for (const auto& label : mesh.physical_labels()) {
        if (label.dimension == 0 || label.dimension >= dimension || label.tag == 0) continue;
        CellType type = CellType::segment;
        if (label.dimension == 2)
            type = label.vertices.size() == 3 ? CellType::triangle : CellType::quadrilateral;
        output.push_back({type, label.vertices, label.tag});
    }
    std::ofstream out(filename);
    require(static_cast<bool>(out), "cannot open Medit output");
    const Index embedding = std::max<Index>(2, mesh.geometric_dimension());
    out << "MeshVersionFormatted 2\nDimension " << embedding << "\n\n";
    out << "Vertices\n" << mesh.coordinates().size() << '\n' << std::setprecision(17);
    for (Index v = 0; v < mesh.coordinates().size(); ++v) {
        const auto& p = mesh.coordinates()[v];
        for (Index j = 0; j < embedding; ++j) out << p[j] << ' ';
        const auto tags = mesh.physical_tags(0, v);
        require(tags.size() <= 1, "Medit cannot store multiple physical groups on one vertex");
        out << (tags.empty() ? 0 : tags.front()) << '\n';
    }
    for (const auto type : {CellType::segment, CellType::triangle,
                            CellType::quadrilateral, CellType::tetrahedron,
                            CellType::hexahedron, CellType::prism, CellType::pyramid}) {
        std::size_t count = 0;
        for (const auto& element : output)
            if (element.type == type) ++count;
        if (count == 0) continue;
        out << '\n' << medit_name(type) << '\n' << count << '\n';
        for (const auto& element : output) {
            if (element.type != type) continue;
            for (auto vertex : element.vertices) out << vertex + 1 << ' ';
            out << element.tag << '\n';
        }
    }
    out << "\nEnd\n";
    require(static_cast<bool>(out), "Medit write failure");
}

} // namespace idmesh
