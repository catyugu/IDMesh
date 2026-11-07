#include "test.hpp"

#include "idmesh/mesh.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
    using idmesh::Index;

    void has_group(const idmesh::Mesh& mesh, Index dimension, int tag, bool bound)
    {
        const auto labels = mesh.labels();
        const auto found = std::find_if(labels.begin(), labels.end(), [&](const idmesh::Label& label) {
            return label.dimension == dimension && label.value == tag;
        });
        check(found != labels.end(), "missing label membership");
        if (bound)
            check(found->entity != idmesh::invalid_index, "unbound label");
        else
            check(found->entity == idmesh::invalid_index, "unexpected materialized entity");
    }

    std::filesystem::path temporary(const char* name)
    {
        return std::filesystem::temp_directory_path() / name;
    }

    void quadratic_variants()
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
            std::pair {idmesh::CellType::pyramid, 13u},
            std::pair {idmesh::CellType::pyramid, 14u}};
        for (auto [type, node_count] : cases) {
            idmesh::MeshBuilder builder(idmesh::cell_dimension(type), 3);
            std::vector<Index> corners, extra;
            const Index n = idmesh::cell_vertex_count(type);
            for (Index i = 0; i < n; ++i)
                corners.push_back(builder.add_vertex({double(i), 0., 0.}));
            for (Index i = n; i < node_count; ++i)
                extra.push_back(builder.add_geometry_node({double(i), 1., 0.}));
            {
                const auto cell = builder.add_quadratic_cell(type, corners, extra);
                builder.add_cell_label(cell, 81);
            }
            auto mesh = std::move(builder).finalize();
            mesh.validate();
            check(mesh.cell_order(0) == 2, "wrong quadratic order");
            check(mesh.entity_count(0) == n, "quadratic nodes became topology vertices");
            check(mesh.geometry_node_count() == node_count, "geometry node count");
            const auto geometry = mesh.cell_geometry_nodes(0);
            check(geometry.size() == node_count, "cell-local geometry node count");
            for (Index j = 0; j < node_count; ++j)
                check(mesh.geometry_node_position(geometry[j])[0] == double(j),
                    "quadratic geometry connectivity ordering");
        }
    }

    void quadratic_gmsh22_all_groups()
    {
        const auto path = temporary("idmesh-order2-groups22.msh");
        {
            std::ofstream out(path);
            out << "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n";
            out << "$PhysicalNames\n4\n0 101 \"corner\"\n1 102 \"edge\"\n"
                   "2 103 \"surface\"\n3 104 \"volume\"\n$EndPhysicalNames\n";
            out << "$Nodes\n10\n";
            for (int n = 1; n <= 10; ++n)
                out << n << ' ' << double(n) << " 0 0\n";
            out << "$EndNodes\n$Elements\n4\n"
                   "1 15 2 101 1 1\n"
                   "2 8 2 102 1 1 2 5\n"
                   "3 9 2 103 1 1 2 3 5 6 7\n"
                   "4 11 2 104 1 1 2 3 4 5 6 7 8 9 10\n"
                   "$EndElements\n";
        }
        const auto mesh = idmesh::read_gmsh(path.string());
        check(mesh.dimension() == 3 && mesh.entity_count(0) == 4, "quadratic Gmsh vertices");
        check(mesh.geometry_node_count() == 10 && mesh.cell_geometry_nodes(0).size() == 10,
            "Gmsh quadratic geometry lost");
        for (Index d = 0; d < 4; ++d)
            has_group(mesh, d, 101 + int(d), true);
        check(mesh.label_name(2, 103) == "surface", "Gmsh label name lost");
        auto reduced = idmesh::read_gmsh(path.string(), idmesh::BuildOptions {.topology = idmesh::TopologyLayout::cell_facets});
        check(reduced.entity_count(1) == 0 && reduced.entity_count(2) == 4,
            "quadratic reduced topology");
        has_group(reduced, 1, 102, false);
        has_group(reduced, 0, 101, true);
        has_group(reduced, 2, 103, true);
        has_group(reduced, 3, 104, true);
        std::filesystem::remove(path);
    }

    void quadratic_gmsh41_multigroup()
    {
        const auto path = temporary("idmesh-order2-groups41.msh");
        {
            std::ofstream out(path);
            out << "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n"
                   "$PhysicalNames\n2\n2 203 \"Surface Group\"\n3 204 \"Volume Group\"\n$EndPhysicalNames\n"
                   "$Entities\n1 1 1 1\n"
                   "1 0 0 0 1 201\n"
                   "2 0 0 0 1 1 1 2 202 212 0\n"
                   "3 0 0 0 1 1 1 2 203 213 0\n"
                   "4 0 0 0 1 1 1 2 204 214 0\n"
                   "$EndEntities\n"
                   "$Nodes\n1 10 1 10\n3 4 0 10\n";
            for (int n = 1; n <= 10; ++n)
                out << n << '\n';
            for (int n = 1; n <= 10; ++n)
                out << double(n) << " 0 0\n";
            out << "$EndNodes\n"
                   "$Elements\n4 4 1 4\n"
                   "0 1 15 1\n1 1\n"
                   "1 2 8 1\n2 1 2 5\n"
                   "2 3 9 1\n3 1 2 3 5 6 7\n"
                   "3 4 11 1\n4 1 2 3 4 5 6 7 8 9 10\n"
                   "$EndElements\n";
        }
        const auto mesh = idmesh::read_gmsh(path.string());
        std::filesystem::remove(path);
        check(mesh.cell_order(0) == 2 && mesh.geometry_node_count() == 10, "Gmsh 4.1 order two");
        has_group(mesh, 0, 201, true);
        has_group(mesh, 1, 202, true);
        has_group(mesh, 1, 212, true);
        has_group(mesh, 2, 203, true);
        has_group(mesh, 2, 213, true);
        has_group(mesh, 3, 204, true);
        has_group(mesh, 3, 214, true);
        const auto groups = mesh.labels(3, 0);
        check(groups.size() == 2 && groups[0].value == 204 && groups[1].value == 214,
            "Gmsh 4.1 multiple cell labels");
        check(mesh.label_name(3, 204) == "Volume Group", "Gmsh 4.1 group name");
    }

    idmesh::Mesh labeled_linear()
    {
        idmesh::MeshBuilder builder(2, 2);
        for (auto position : std::array<std::array<double, 3>, 4> {{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}}})
            builder.add_vertex(position);
        const std::array<Index, 3> a {0, 1, 2}, b {0, 2, 3};
        const auto first = builder.add_cell(idmesh::CellType::triangle, a);
        const auto second = builder.add_cell(idmesh::CellType::triangle, b);
        builder.add_cell_label(first, 31);
        builder.add_cell_label(second, 32);
        const std::array<Index, 2> edge {0, 1};
        const std::array<Index, 1> point {0};
        builder.add_label(1, edge, 40);
        builder.add_label(0, point, 41);
        return std::move(builder).finalize();
    }

    void legacy_vtk_round_trip()
    {
        const auto path = temporary("idmesh-roundtrip.vtk");
        idmesh::write_vtk_legacy(labeled_linear(), path.string());
        const auto mesh = idmesh::read_vtk_legacy(path.string());
        std::filesystem::remove(path);
        check(mesh.dimension() == 2 && mesh.entity_count(2) == 2, "VTK roundtrip cell count");
        check(mesh.labels(2, 0).front().value == 31 && mesh.labels(2, 1).front().value == 32,
            "VTK cell labels");
        has_group(mesh, 0, 41, true);
        has_group(mesh, 1, 40, true);
        has_group(mesh, 2, 31, true);
    }

    void medit_round_trip()
    {
        const auto path = temporary("idmesh-roundtrip.mesh");
        idmesh::write_medit(labeled_linear(), path.string());
        const auto mesh = idmesh::read_medit(path.string());
        std::filesystem::remove(path);
        check(mesh.dimension() == 2 && mesh.entity_count(2) == 2, "Medit roundtrip cell count");
        has_group(mesh, 0, 41, true);
        has_group(mesh, 1, 40, true);
        has_group(mesh, 2, 31, true);
    }

    void zero_label_io_rejected()
    {
        idmesh::MeshBuilder builder(1, 1);
        builder.add_vertex({0, 0, 0});
        builder.add_vertex({1, 0, 0});
        const std::array<Index, 2> vertices {0, 1};
        const auto cell = builder.add_cell(idmesh::CellType::segment, vertices);
        builder.add_cell_label(cell, 0);
        const auto mesh = std::move(builder).finalize();

        for (const auto& path : {temporary("idmesh-zero-label.vtk"),
                 temporary("idmesh-zero-label.mesh")}) {
            bool rejected = false;
            try {
                if (path.extension() == ".vtk")
                    idmesh::write_vtk_legacy(mesh, path.string());
                else
                    idmesh::write_medit(mesh, path.string());
            }
            catch (const std::runtime_error&) {
                rejected = true;
            }
            std::filesystem::remove(path);
            check(rejected, "zero label silently lost by legacy IO");
        }
    }

} // namespace

void mixed_geometry_orders()
{
    idmesh::MeshBuilder builder(2, 2);
    for (auto p : std::array<std::array<double, 3>, 4> {{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}}})
        builder.add_vertex(p);
    const std::array<Index, 3> lower {0, 1, 2}, upper {0, 2, 3};
    const auto linear = builder.add_cell(idmesh::CellType::triangle, lower);
    builder.add_cell_label(linear, 5);
    std::vector<Index> extra;
    for (unsigned i = 0; i < 3; ++i)
        extra.push_back(builder.add_geometry_node({double(i), 0.5, 0}));
    const auto quadratic = builder.add_quadratic_cell(idmesh::CellType::triangle, upper, extra);
    builder.add_cell_label(quadratic, 6);
    auto mesh = std::move(builder).finalize();
    mesh.validate();
    check(mesh.cell_order(0) == 1 && mesh.cell_order(1) == 2, "mixed geometry orders");
    check(mesh.cell_geometry_nodes(0).size() == 3 && mesh.cell_geometry_nodes(1).size() == 6, "mixed local geometry nodes");
}

void omitted_layer_labels()
{
    auto make = [] {
        idmesh::MeshBuilder builder(2, 2);
        builder.add_vertex({0, 0, 0});
        builder.add_vertex({1, 0, 0});
        builder.add_vertex({0, 1, 0});
        const std::array<Index, 3> triangle {0, 1, 2};
        const std::array<Index, 2> edge {0, 1};
        const auto cell = builder.add_cell(idmesh::CellType::triangle, triangle);
        builder.add_cell_label(cell, 21);
        builder.add_label(1, edge, 42);
        builder.add_label(1, edge, 43);
        return std::move(builder).finalize(idmesh::BuildOptions {.topology = idmesh::TopologyLayout::cell_vertices, .relations = idmesh::RelationDirection::downward});
    };
    // Labels are borrowed from the Mesh: copying the Mesh must preserve
    // storage ownership, including keys on omitted dimensions.
    const auto mesh = make();
    const auto shared = mesh;
    check(shared.labels().size() == 3, "omitted label multiplicity");
    for (const auto& label : shared.labels()) {
        if (label.dimension == 1) {
            check(label.entity == idmesh::invalid_index, "omitted edge materialized");
            check(label.vertices.size() == 2 && label.vertices[0] == 0
                    && label.vertices[1] == 1,
                "omitted label vertex key changed");
        }
        else {
            check(label.dimension == 2 && label.entity == 0,
                "top-dimensional label lost");
            const auto actual = shared.entity_vertices(2, 0);
            check(std::equal(actual.begin(), actual.end(),
                      label.vertices.begin(), label.vertices.end()),
                "top-dimensional label vertex key changed");
        }
    }
    shared.validate();
}

void many_labels()
{
    idmesh::MeshBuilder builder(1, 1);
    constexpr Index n = 5000;
    for (Index i = 0; i <= n; ++i)
        builder.add_vertex({double(i), 0, 0});
    for (Index i = 0; i < n; ++i) {
        const std::array<Index, 2> v {i, i + 1};
        const auto cell = builder.add_cell(idmesh::CellType::segment, v);
        builder.add_cell_label(cell, 100 + int(i % 5));
        if (i % 11 == 0) {
            builder.add_cell_label(cell, 120);
            builder.add_cell_label(cell, 120);
        }
    }
    const auto mesh = std::move(builder).finalize(idmesh::BuildOptions {.topology = idmesh::TopologyLayout::cell_vertices, .relations = idmesh::RelationDirection::downward});
    check(mesh.labels().size() == n + (n + 10) / 11,
        "label deduplication or entity count");
    for (const auto& label : mesh.labels()) {
        const auto key = mesh.entity_vertices(label.dimension, label.entity);
        check(std::equal(key.begin(), key.end(), label.vertices.begin(), label.vertices.end()),
            "materialized label did not reuse entity connectivity");
    }
    for (Index c = 0; c < n; ++c) {
        const auto labels = mesh.labels(1, c);
        check(labels.size() == (c % 11 == 0 ? 2u : 1u), "label membership count");
        check(labels.front().value == 100 + int(c % 5), "incorrect label at cell");
        if (c % 11 == 0)
            check(labels.back().value == 120, "second membership lost");
    }
}

int main()
{
    return idmesh::test::run({
        {"quadratic variants", quadratic_variants},
        {"Gmsh 2.2 groups", quadratic_gmsh22_all_groups},
        {"Gmsh 4.1 groups", quadratic_gmsh41_multigroup},
        {"legacy VTK round-trip", legacy_vtk_round_trip},
        {"Medit round-trip", medit_round_trip},
        {"zero labels rejected", zero_label_io_rejected},
        {"mixed geometry orders", mixed_geometry_orders},
        {"many labels", many_labels},
        {"omitted-layer labels", omitted_layer_labels},
    });
}
