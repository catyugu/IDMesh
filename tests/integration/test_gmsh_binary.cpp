#include "test.hpp"

#include "idmesh/geometry.hpp"
#include "idmesh/io.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
    using idmesh::Index;
    using Point = std::array<double, 3>;

    std::filesystem::path fixture(const std::string& name)
    {
        return std::filesystem::temp_directory_path() / ("idmesh-gmsh-" + name + ".msh");
    }

    void raw_number(std::ostream& out, std::uint64_t value, unsigned bytes, bool little)
    {
        for (unsigned i = 0; i < bytes; ++i) {
            const unsigned shift = little ? i : bytes - 1 - i;
            out.put(static_cast<char>((value >> (8 * shift)) & 0xff));
        }
    }
    void raw_int(std::ostream& out, std::int32_t value, bool little)
    {
        raw_number(out, std::bit_cast<std::uint32_t>(value), 4, little);
    }
    void raw_double(std::ostream& out, double value, bool little)
    {
        raw_number(out, std::bit_cast<std::uint64_t>(value), 8, little);
    }

    constexpr std::array<Point, 10> tetra_nodes {{
        {0., 0., 0.}, {1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.},
        {0.5, 0., 0.}, {0.5, 0.5, 0.}, {0., 0.5, 0.},
        {0., 0., 0.5}, {0., 0.5, 0.5}, {0.5, 0., 0.5}
    }};

    void handrolled22(const std::filesystem::path& filename, bool little)
    {
        std::ofstream out(filename, std::ios::binary);
        out << "$MeshFormat\n2.2 1 8\n";
        raw_int(out, 1, little);
        out << "\n$EndMeshFormat\n$PhysicalNames\n4\n"
            << "0 101 \"Point\"\n1 102 \"Edge\"\n2 103 \"Face\"\n3 104 \"Body\"\n"
            << "$EndPhysicalNames\n$Nodes\n10\n";
        for (std::int32_t i = 0; i < 10; ++i) {
            raw_int(out, i + 1, little);
            for (double x : tetra_nodes[static_cast<std::size_t>(i)])
                raw_double(out, x, little);
        }
        out << "\n$EndNodes\n$Elements\n4\n";
        // Gmsh 2.2 binary elements are explicitly separated into kind blocks.
        const std::array<std::pair<int, std::vector<int>>, 4> elements {{
            {15, {1}}, {8, {1, 2, 5}}, {9, {1, 2, 3, 5, 6, 7}},
            {11, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}}
        }};
        for (std::int32_t i = 0; i < 4; ++i) {
            raw_int(out, elements[i].first, little);
            raw_int(out, 1, little);
            raw_int(out, 2, little);
            raw_int(out, i + 1, little);
            raw_int(out, 101 + i, little);
            raw_int(out, 1, little);
            for (int node : elements[i].second)
                raw_int(out, node, little);
        }
        out << "\n$EndElements\n";
    }

    void handrolled41(const std::filesystem::path& filename, bool little, unsigned size_width)
    {
        std::ofstream out(filename, std::ios::binary);
        out << "$MeshFormat\n4.1 1 " << size_width << '\n';
        raw_int(out, 1, little);
        out << "\n$EndMeshFormat\n"
            << "$PhysicalNames\n2\n2 203 \"Surface\"\n3 204 \"Volume\"\n$EndPhysicalNames\n"
            << "$Entities\n";
        for (std::uint64_t n : {0u, 0u, 1u, 1u})
            raw_number(out, n, size_width, little);
        for (int dim : {2, 3}) {
            raw_int(out, dim + 1, little);
            for (double x : {0., 0., 0., 1., 1., 1.})
                raw_double(out, x, little);
            raw_number(out, 3, size_width, little);
            raw_int(out, dim == 2 ? 203 : 204, little);
            raw_int(out, 0, little); // reserved unassigned physical tag
            raw_int(out, dim == 2 ? 213 : 214, little);
            raw_number(out, 0, size_width, little);
        }
        out << "\n$EndEntities\n$Nodes\n";
        for (std::uint64_t n : {1u, 10u, 1u, 10u})
            raw_number(out, n, size_width, little);
        raw_int(out, 3, little);
        raw_int(out, 4, little);
        raw_int(out, 1, little); // parametric classification
        raw_number(out, 10, size_width, little);
        for (std::uint64_t node = 1; node <= 10; ++node)
            raw_number(out, node, size_width, little);
        for (const auto& p : tetra_nodes) {
            for (double x : p)
                raw_double(out, x, little);
            for (double u : {0.1, 0.2, 0.3})
                raw_double(out, u, little);
        }
        out << "\n$EndNodes\n$Elements\n";
        for (std::uint64_t n : {2u, 2u, 1u, 2u})
            raw_number(out, n, size_width, little);
        for (int block = 0; block < 2; ++block) {
            raw_int(out, block == 0 ? 2 : 3, little);
            raw_int(out, block == 0 ? 3 : 4, little);
            raw_int(out, block == 0 ? 9 : 11, little);
            raw_number(out, 1, size_width, little);
            raw_number(out, block + 1, size_width, little);
            const int nodes = block == 0 ? 6 : 10;
            for (int i = 1; i <= nodes; ++i)
                raw_number(out, i, size_width, little);
        }
        out << "\n$EndElements\n";
    }

    bool has_group(const idmesh::Mesh& mesh, Index dim, int label)
    {
        return std::any_of(mesh.labels().begin(), mesh.labels().end(),
            [&](const idmesh::Label& x) { return x.dimension == dim && x.value == label; });
    }

    void manually_encoded_import()
    {
        for (bool little : {true, false}) {
            const auto p22 = fixture(little ? "little22" : "big22");
            handrolled22(p22, little);
            const auto m22 = idmesh::read_gmsh(p22.string());
            std::filesystem::remove(p22);
            check(m22.entity_count(3) == 1 && m22.entity_count(0) == 4,
                "independent Gmsh 2.2 binary counts");
            check(m22.cell_geometry_nodes(0).size() == 10, "Gmsh 2.2 P2 lost");
            for (int dim = 0; dim <= 3; ++dim)
                check(has_group(m22, dim, 101 + dim), "Gmsh 2.2 physical tag lost");
            check(m22.label_name(3, 104) == "Body", "Gmsh 2.2 group name lost");

            for (unsigned width : {4u, 8u}) {
                const auto p41 = fixture(std::string(little ? "little41" : "big41")
                    + std::to_string(width));
                handrolled41(p41, little, width);
                const auto m41 = idmesh::read_gmsh(p41.string());
                std::filesystem::remove(p41);
                check(m41.entity_count(3) == 1 && m41.entity_count(0) == 4,
                    "independent Gmsh 4.1 binary counts");
                check(m41.cell_geometry_nodes(0).size() == 10,
                    "Gmsh 4.1 parametric/P2 coordinates lost");
                check(m41.labels(3, 0).size() == 2
                        && has_group(m41, 2, 203) && has_group(m41, 2, 213)
                        && has_group(m41, 3, 204) && has_group(m41, 3, 214),
                    "Gmsh 4.1 physical groups lost");
                check(m41.label_name(2, 203) == "Surface",
                    "Gmsh 4.1 physical group name lost");
            }
        }
    }

    idmesh::Mesh quadratic_mesh(idmesh::CellType type, Index nodes)
    {
        const auto positions = idmesh::reference_geometry_nodes(type, nodes);
        idmesh::MeshBuilder builder(idmesh::cell_dimension(type), 3);
        std::vector<Index> corners, extra;
        for (Index i = 0; i < idmesh::cell_vertex_count(type); ++i)
            corners.push_back(builder.add_vertex(positions[i]));
        for (Index i = static_cast<Index>(corners.size()); i < nodes; ++i)
            extra.push_back(builder.add_geometry_node(positions[i]));
        const auto cell = extra.empty() ? builder.add_cell(type, corners)
                                        : builder.add_quadratic_cell(type, corners, extra);
        builder.add_cell_label(cell, 121);
        builder.set_label_name(idmesh::cell_dimension(type), 121, "material");
        if (idmesh::cell_dimension(type) > 1) {
            const auto facet = idmesh::reference_facet_vertices(type, 0);
            std::vector<Index> v;
            for (auto local : facet)
                v.push_back(corners[local]);
            builder.add_label(idmesh::cell_dimension(type) - 1, v, 212);
            builder.set_label_name(idmesh::cell_dimension(type) - 1, 212, "interface");
        }
        return std::move(builder).finalize();
    }

    void all_cell_type_roundtrips()
    {
        const std::array shapes {
            std::pair {idmesh::CellType::segment, 3u},
            std::pair {idmesh::CellType::triangle, 6u},
            std::pair {idmesh::CellType::quadrilateral, 8u},
            std::pair {idmesh::CellType::quadrilateral, 9u},
            std::pair {idmesh::CellType::tetrahedron, 10u},
            std::pair {idmesh::CellType::hexahedron, 20u},
            std::pair {idmesh::CellType::hexahedron, 27u},
            std::pair {idmesh::CellType::prism, 15u},
            std::pair {idmesh::CellType::prism, 18u},
            std::pair {idmesh::CellType::pyramid, 13u},
            std::pair {idmesh::CellType::pyramid, 14u}
        };
        for (const auto [type, count] : shapes) {
            const auto original = quadratic_mesh(type, count);
            for (const auto version : {idmesh::GmshVersion::v22, idmesh::GmshVersion::v41}) {
                for (const auto encoding : {idmesh::GmshEncoding::ascii,
                         idmesh::GmshEncoding::binary}) {
                    const auto filename = fixture("round-trip");
                    idmesh::write_gmsh(original, filename.string(), {version, encoding});
                    const auto actual = idmesh::read_gmsh(filename.string(),
                        {idmesh::TopologyLayout::cell_facets,
                            idmesh::RelationDirection::bidirectional});
                    std::filesystem::remove(filename);
                    check(actual.cell_types()[0] == type, "Gmsh cell type changed");
                    check(actual.cell_geometry_nodes(0).size() == count,
                        "Gmsh geometry order changed");
                    const auto before = original.cell_geometry_nodes(0);
                    const auto after = actual.cell_geometry_nodes(0);
                    for (std::size_t i = 0; i < before.size(); ++i) {
                        const auto a = original.geometry_node_position(before[i]);
                        const auto b = actual.geometry_node_position(after[i]);
                        for (int axis = 0; axis < 3; ++axis)
                            check(std::abs(a[axis] - b[axis]) < 1e-12,
                                "Gmsh P2 geometry node order or coordinate changed");
                    }
                    check(has_group(actual, actual.dimension(), 121), "Gmsh cell tag lost");
                    check(actual.label_name(actual.dimension(), 121) == "material",
                        "Gmsh group name lost");
                    if (actual.dimension() > 1)
                        check(has_group(actual, actual.dimension() - 1, 212),
                            "Gmsh lower-dimensional label lost");
                }
            }
        }
    }

    void multimaterial41_and_22_rejection()
    {
        idmesh::MeshBuilder b(3, 3);
        for (const auto p : std::array<Point, 4> {
                 Point {0., 0., 0.}, Point {1., 0., 0.},
                 Point {0., 1., 0.}, Point {0., 0., 1.}})
            b.add_vertex(p);
        const std::array<Index, 4> tetra {0, 1, 2, 3};
        const std::array<Index, 3> surface {0, 1, 2};
        b.add_cell(idmesh::CellType::tetrahedron, tetra);
        b.add_cell_label(0, 4);
        b.add_cell_label(0, 14);
        b.add_label(2, surface, 7);
        b.add_label(2, surface, 17);
        b.set_label_name(3, 14, "second material");
        const auto mesh = std::move(b).finalize(
            {idmesh::TopologyLayout::cell_facets, idmesh::RelationDirection::bidirectional});
        const auto filename = fixture("multigroup");
        for (const auto encoding : {idmesh::GmshEncoding::ascii, idmesh::GmshEncoding::binary}) {
            idmesh::write_gmsh(mesh, filename.string(), {idmesh::GmshVersion::v41, encoding});
            const auto result = idmesh::read_gmsh(filename.string());
            check(result.labels(3, 0).size() == 2, "4.1 cell multi-membership lost");
            check(result.label_name(3, 14) == "second material", "4.1 name lost");
            check(has_group(result, 2, 7) && has_group(result, 2, 17),
                "4.1 facet multi-membership lost");
        }
        expect_error<std::runtime_error>([&] {
            idmesh::write_gmsh(mesh, filename.string(), {idmesh::GmshVersion::v22,
                idmesh::GmshEncoding::binary});
        }, "2.2 silently lost multiple physical groups");
        std::filesystem::remove(filename);
    }

    void malformed_binary_rejected()
    {
        const auto filename = fixture("bad-marker");
        {
            std::ofstream out(filename, std::ios::binary);
            out << "$MeshFormat\n4.1 1 8\n";
            raw_int(out, 2, true);
            out << "\n$EndMeshFormat\n";
        }
        expect_error<std::runtime_error>([&] { (void)idmesh::read_gmsh(filename.string()); },
            "Gmsh invalid endian marker");
        {
            std::ofstream out(filename, std::ios::binary);
            out << "$MeshFormat\n2.2 1 8\n";
            raw_int(out, 1, true);
            out << "\n$EndMeshFormat\n$Nodes\n1\n";
            raw_int(out, 1, true);
            raw_int(out, 0, true); // Only half the first coordinate.
        }
        expect_error<std::runtime_error>([&] { (void)idmesh::read_gmsh(filename.string()); },
            "Gmsh truncated payload");
        std::filesystem::remove(filename);
    }

} // namespace

int main()
{
    return idmesh::test::run({
        {"hand-encoded endian, width and parametric data", manually_encoded_import},
        {"all supported P2 cell families, ASCII/binary 2.2/4.1", all_cell_type_roundtrips},
        {"multiple physical groups and explicit 2.2 rejection", multimaterial41_and_22_rejection},
        {"invalid and truncated binary input", malformed_binary_rejected},
    });
}
