#include "test.hpp"

#include <array>
#include <cmath>
#include <filesystem>
#include <idmesh/geometry.hpp>
#include <idmesh/io.hpp>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
    using idmesh::Index;
    idmesh::Mesh mesh_for(idmesh::CellType type, Index count)
    {
        auto points = idmesh::reference_geometry_nodes(type, count);
        idmesh::MeshBuilder b(idmesh::cell_dimension(type), 3);
        std::vector<Index> corners, extras;
        const auto n = idmesh::cell_vertex_count(type);
        for (Index i = 0; i < n; ++i)
            corners.push_back(b.add_vertex(points[i]));
        for (Index i = n; i < count; ++i)
            extras.push_back(b.add_geometry_node(points[i]));
        const auto cell = extras.empty()
            ? b.add_cell(type, corners)
            : b.add_quadratic_cell(type, corners, extras);
        b.add_cell_label(cell, 11);
        b.add_cell_label(cell, 11); // duplicate must remain unique
        return std::move(b).finalize();
    }
    void compare(const idmesh::Mesh& original, const idmesh::Mesh& reread)
    {
        check(original.dimension() == reread.dimension(), "dimension changed in IO");
        check(original.cell_types()[0] == reread.cell_types()[0], "cell type changed in IO");
        check(original.cell_order(0) == reread.cell_order(0), "geometry order changed in IO");
        auto a = original.cell_geometry_nodes(0), b = reread.cell_geometry_nodes(0);
        check(a.size() == b.size(), "geometry nodes changed in IO");
        for (std::size_t i = 0; i < a.size(); ++i) {
            auto pa = original.geometry_node_position(a[i]);
            auto pb = reread.geometry_node_position(b[i]);
            for (int j = 0; j < 3; ++j)
                check(std::abs(pa[j] - pb[j]) < 1e-12,
                    "geometry node permutation changed in IO");
        }
        const auto labels = reread.labels(reread.dimension(), 0);
        check(labels.size() == 1 && labels.front().value == 11,
            "label missing in IO");
    }
    void vtk_cases()
    {
        const std::array cases {
            std::pair {idmesh::CellType::segment, 3u},
            std::pair {idmesh::CellType::triangle, 6u},
            std::pair {idmesh::CellType::quadrilateral, 8u},
            std::pair {idmesh::CellType::quadrilateral, 9u},
            std::pair {idmesh::CellType::tetrahedron, 10u},
            std::pair {idmesh::CellType::hexahedron, 20u},
            std::pair {idmesh::CellType::hexahedron, 27u},
            std::pair {idmesh::CellType::prism, 15u},
            std::pair {idmesh::CellType::prism, 18u},
            std::pair {idmesh::CellType::pyramid, 13u}};
        const auto path = std::filesystem::temp_directory_path() / "idmesh-quadratic.vtk";
        for (auto [type, n] : cases) {
            const auto original = mesh_for(type, n);
            idmesh::write_vtk_legacy(original, path.string());
            const auto reread = idmesh::read_vtk_legacy(path.string());
            compare(original, reread);
        }
        std::filesystem::remove(path);
        bool rejected = false;
        try {
            idmesh::write_vtk_legacy(mesh_for(idmesh::CellType::pyramid, 14), path.string());
        }
        catch (const std::runtime_error&) {
            rejected = true;
        }
        check(rejected, "unsupported 14-node VTK pyramid silently written");
    }
    void medit_cases()
    {
        const auto path = std::filesystem::temp_directory_path() / "idmesh-quadratic.mesh";
        for (auto [type, n] : std::array {
                 std::pair {idmesh::CellType::segment, 3u},
                 std::pair {idmesh::CellType::triangle, 6u},
                 std::pair {idmesh::CellType::tetrahedron, 10u}}) {
            const auto original = mesh_for(type, n);
            idmesh::write_medit(original, path.string());
            const auto reread = idmesh::read_medit(path.string());
            compare(original, reread);
        }
        std::filesystem::remove(path);
        bool rejected = false;
        try {
            idmesh::write_medit(mesh_for(idmesh::CellType::hexahedron, 20), path.string());
        }
        catch (const std::runtime_error&) {
            rejected = true;
        }
        check(rejected, "unsupported Medit hexa20 silently written");
    }
}
int main()
{
    return idmesh::test::run({
        {"VTK quadratic IO", vtk_cases},
        {"Medit quadratic IO", medit_cases},
    });
}
