#include "idmesh/mesh.hpp"
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
void check(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

void test_reference_cells()
{
    constexpr std::array types {
        idmesh::CellType::segment, idmesh::CellType::triangle,
        idmesh::CellType::quadrilateral, idmesh::CellType::tetrahedron,
        idmesh::CellType::hexahedron, idmesh::CellType::prism,
        idmesh::CellType::pyramid
    };
    constexpr std::array<unsigned, 7> facets {2, 3, 4, 4, 6, 5, 5};
    for (std::size_t k = 0; k < types.size(); ++k) {
        auto dim = idmesh::cell_dimension(types[k]);
        idmesh::MeshBuilder builder(dim, dim);
        std::vector<idmesh::Index> vertices;
        for (unsigned i = 0; i < idmesh::cell_vertex_count(types[k]); ++i)
            vertices.push_back(builder.add_vertex({double(i), double(i % 3), double(i % 2)}));
        builder.add_cell(types[k], vertices, 42);
        auto mesh = std::move(builder).finalize();
        mesh.validate();
        check(mesh.entity_count(dim) == 1, "missing reference cell");
        check(mesh.cell_tags().front() == 42, "cell tag lost");
        if (dim > 1) {
            check(mesh.entity_count(dim - 1) == facets[k], "wrong facet count");
            check(mesh.relation(dim, dim - 1).row(0).size() == facets[k], "wrong facet incidence");
        }
        idmesh::MeshBuilder minimal(dim, dim);
        for (unsigned i = 0; i < vertices.size(); ++i) minimal.add_vertex({double(i), 0, 0});
        minimal.add_cell(types[k], vertices);
        auto reduced = std::move(minimal).finalize(idmesh::TopologyOptions::cells_and_vertices());
        check(reduced.has_relation(dim, 0), "missing cell-vertex relation");
        if (dim > 1) check(!reduced.has_relation(dim, dim - 1), "extra materialization");
    }
}

void test_shared_orientations()
{
    idmesh::MeshBuilder builder(3, 3);
    for (const auto& xyz : std::array<std::array<double, 3>, 5>{{
        {0,0,0}, {1,0,0}, {0,1,0}, {0,0,1}, {0,0,-1}
    }}) builder.add_vertex(xyz);
    std::array<idmesh::Index, 4> first {0,1,2,3}, second {0,2,1,4};
    builder.add_cell(idmesh::CellType::tetrahedron, first);
    builder.add_cell(idmesh::CellType::tetrahedron, second);
    auto mesh = std::move(builder).finalize();
    check(mesh.entity_count(2) == 7, "shared face was not deduplicated");
    auto faces = mesh.relation(3, 2);
    auto support = mesh.relation(2, 3);
    bool found = false;
    for (idmesh::Index f = 0; f < mesh.entity_count(2); ++f) {
        if (support.row(f).size() != 2) continue;
        found = true;
        int sign[2] {};
        for (idmesh::Index c = 0; c < 2; ++c) {
            const auto row = faces.row(c);
            const auto signs = faces.row_signs(c);
            for (std::size_t i = 0; i < row.size(); ++i)
                if (row[i] == f) sign[c] = signs[i];
        }
        check(sign[0] == -sign[1] && sign[0] != 0, "incorrect shared face orientation");
    }
    check(found, "shared face not found");
}

void test_finite_volume()
{
    idmesh::MeshBuilder builder(3, 3);
    for (unsigned i = 0; i < 4; ++i) builder.add_vertex({double(i),0,0});
    const std::array<idmesh::Index, 4> tetra {0,1,2,3};
    builder.add_cell(idmesh::CellType::tetrahedron, tetra);
    auto mesh = std::move(builder).finalize(idmesh::TopologyOptions::finite_volume());
    check(mesh.entity_count(2) == 4, "missing FV facets");
    check(mesh.entity_count(1) == 0, "FV unnecessarily materializes edges");
    check(mesh.has_relation(3,2) && mesh.has_relation(2,3), "missing FV adjacency");
}

void test_gmsh()
{
    auto path = std::filesystem::temp_directory_path() / "idmesh-test-2.2.msh";
    {
        std::ofstream out(path);
        out << "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
            << "$Nodes\n3\n10 0 0 0\n20 1 0 0\n30 0 1 0\n$EndNodes\n"
            << "$Elements\n1\n1 2 2 17 1 10 20 30\n$EndElements\n";
    }
    auto mesh = idmesh::read_gmsh(path.string());
    check(mesh.dimension() == 2 && mesh.entity_count(2) == 1, "Gmsh v2 load failed");
    check(mesh.cell_tags()[0] == 17, "Gmsh v2 tag lost");
    std::filesystem::remove(path);
}


void test_gmsh_v41()
{
    auto path = std::filesystem::temp_directory_path() / "idmesh-test-4.1.msh";
    {
        std::ofstream out(path);
        out << "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n"
            << "$Entities\n0 0 1 0\n1 0 0 0 1 1 0 1 23 0\n$EndEntities\n"
            << "$Nodes\n1 3 1 3\n2 1 0 3\n1 2 3\n"
            << "0 0 0\n1 0 0\n0 1 0\n$EndNodes\n"
            << "$Elements\n1 1 1 1\n2 1 2 1\n1 1 2 3\n$EndElements\n";
    }
    auto mesh = idmesh::read_gmsh(path.string());
    check(mesh.dimension() == 2 && mesh.entity_count(2) == 1, "Gmsh v4 load failed");
    check(mesh.cell_tags()[0] == 23, "Gmsh v4 physical tag lost");
    std::filesystem::remove(path);
}

void test_invalid()
{
    idmesh::MeshBuilder builder(2, 2);
    builder.add_vertex({0,0,0});
    builder.add_vertex({1,0,0});
    bool rejected = false;
    const std::array<idmesh::Index, 3> invalid {0,1,1};
    try { builder.add_cell(idmesh::CellType::triangle, invalid); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "duplicate vertex was accepted");
}
}

int main()
{
    try {
        test_reference_cells();
        test_shared_orientations();
        test_finite_volume();
        test_gmsh();
        test_gmsh_v41();
        test_invalid();
        std::cout << "IDMesh topology tests passed\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "IDMesh test failure: " << ex.what() << '\n';
        return 1;
    }
}
