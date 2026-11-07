#include "test.hpp"

#include "io/vtk/vtk_common.hpp"

#include <idmesh/data.hpp>
#include <idmesh/mesh.hpp>

#include <array>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

    using idmesh::Index;
    namespace data = idmesh::data;
    namespace vtk = idmesh::detail::vtk;

    idmesh::Mesh make_cell(
        idmesh::CellType type, Index nodes)
    {
        idmesh::MeshBuilder builder(
            idmesh::cell_dimension(type), 3);
        std::vector<Index> corners;
        std::vector<Index> extras;
        const Index corner_count
            = idmesh::cell_vertex_count(type);
        for (Index i = 0; i < corner_count; ++i)
            corners.push_back(
                builder.add_vertex(
                    {double(i), 0.0, 0.0}));
        for (Index i = corner_count; i < nodes; ++i)
            extras.push_back(
                builder.add_geometry_node(
                    {double(i), 1.0, 0.0}));
        if (extras.empty())
            builder.add_cell(type, corners);
        else
            builder.add_quadratic_cell(
                type, corners, extras);
        return std::move(builder).finalize();
    }

    void cell_mappings()
    {
        struct Case {
            idmesh::CellType type;
            Index nodes;
            int code;
            Index dimension;
        };
        const std::array cases {
            Case {idmesh::CellType::segment, 2, 3, 1},
            Case {idmesh::CellType::segment, 3, 21, 1},
            Case {idmesh::CellType::triangle, 3, 5, 2},
            Case {idmesh::CellType::triangle, 6, 22, 2},
            Case {idmesh::CellType::quadrilateral, 4, 9, 2},
            Case {idmesh::CellType::quadrilateral, 8, 23, 2},
            Case {idmesh::CellType::quadrilateral, 9, 28, 2},
            Case {idmesh::CellType::tetrahedron, 4, 10, 3},
            Case {idmesh::CellType::tetrahedron, 10, 24, 3},
            Case {idmesh::CellType::hexahedron, 8, 12, 3},
            Case {idmesh::CellType::hexahedron, 20, 25, 3},
            Case {idmesh::CellType::hexahedron, 27, 29, 3},
            Case {idmesh::CellType::prism, 6, 13, 3},
            Case {idmesh::CellType::prism, 15, 26, 3},
            Case {idmesh::CellType::prism, 18, 32, 3},
            Case {idmesh::CellType::pyramid, 5, 14, 3},
            Case {idmesh::CellType::pyramid, 13, 27, 3}};

        for (const auto& test : cases) {
            const auto info = vtk::cell_info(test.code);
            check(info.type == test.type
                    && info.nodes == test.nodes
                    && info.dimension == test.dimension,
                "VTK cell-info mapping mismatch");
            check(vtk::cell_code(test.type, test.nodes)
                    == test.code,
                "VTK reverse cell-code mapping mismatch");

            const auto mesh = make_cell(
                test.type, test.nodes);
            const auto encoded = vtk::encode_cell(mesh, 0);
            check(encoded.code == test.code,
                "encoded VTK cell code mismatch");
            const auto decoded = vtk::to_gmsh_order(
                test.type, encoded.points);
            const auto source = mesh.cell_geometry_nodes(0);
            check(decoded.size() == source.size(),
                "VTK decoded geometry size mismatch");
            for (Index i = 0; i < source.size(); ++i)
                check(decoded[i] == source[i],
                    "VTK high-order permutation round-trip failed");
        }

        check(vtk::entity_code(0, 1) == 1,
            "VTK vertex entity code");
        check(vtk::entity_code(1, 2) == 3,
            "VTK edge entity code");
        check(vtk::entity_code(2, 3) == 5,
            "VTK triangle entity code");
        check(vtk::entity_code(2, 4) == 9,
            "VTK quad entity code");

        expect_error<std::runtime_error>(
            [] {
                (void)vtk::cell_code(
                    idmesh::CellType::pyramid, 14);
            },
            "Pyramid14 was silently mapped to VTK");
        expect_error<std::runtime_error>(
            [] { (void)vtk::cell_info(999); },
            "unknown VTK cell code accepted");
        expect_error<std::runtime_error>(
            [] { (void)vtk::entity_code(2, 5); },
            "unsupported polygon entity was mapped to VTK");
    }

    void data_association()
    {
        const auto linear = make_cell(
            idmesh::CellType::triangle, 3);

        const std::array<double, 3> point_values {
            1, 2, 3};
        data::MeshDataView vertices {
            "vertex",
            data::entities(0),
            data::all(),
            data::make_scalar_array(
                std::span(point_values))};
        check(vtk::attribute_association(linear, vertices)
                == vtk::AttributeAssociation::point,
            "linear vertex data did not map to PointData");

        data::MeshDataView geometry {
            "geometry",
            data::geometry_nodes(),
            data::all(),
            data::make_scalar_array(
                std::span(point_values))};
        check(vtk::attribute_association(linear, geometry)
                == vtk::AttributeAssociation::point,
            "geometry-node data did not map to PointData");

        const std::array<double, 1> cell_value {7};
        data::MeshDataView cells {
            "cell",
            data::entities(2),
            data::all(),
            data::make_scalar_array(
                std::span(cell_value))};
        check(vtk::attribute_association(linear, cells)
                == vtk::AttributeAssociation::cell,
            "cell data did not map to CellData");

        data::MeshDataView global {
            "global",
            data::whole_mesh(),
            data::all(),
            data::make_scalar_array(
                std::span(cell_value))};
        check(vtk::attribute_association(linear, global)
                == vtk::AttributeAssociation::field,
            "mesh data did not map to FieldData");

        const std::array<double, 3> edge_values {
            1, 2, 3};
        data::MeshDataView edges {
            "edge",
            data::entities(1),
            data::all(),
            data::make_scalar_array(
                std::span(edge_values))};
        expect_error<std::invalid_argument>(
            [&] {
                (void)vtk::attribute_association(
                    linear, edges);
            },
            "intermediate entity data silently mapped to VTK");

        const std::array<Index, 1> subset {0};
        data::MeshDataView sparse {
            "sparse",
            data::entities(2),
            data::select(subset),
            data::make_scalar_array(
                std::span(cell_value))};
        expect_error<std::invalid_argument>(
            [&] {
                (void)vtk::attribute_association(
                    linear, sparse);
            },
            "sparse data silently mapped to VTK");

        const auto quadratic = make_cell(
            idmesh::CellType::triangle, 6);
        data::MeshDataView quadratic_vertices {
            "vertex",
            data::entities(0),
            data::all(),
            data::make_scalar_array(
                std::span(point_values))};
        expect_error<std::invalid_argument>(
            [&] {
                (void)vtk::attribute_association(
                    quadratic,
                    quadratic_vertices);
            },
            "vertex-only quadratic data silently padded to PointData");

        const std::array<double, 6> geometry_values {
            1, 2, 3, 4, 5, 6};
        data::MeshDataView quadratic_geometry {
            "geometry",
            data::geometry_nodes(),
            data::all(),
            data::make_scalar_array(
                std::span(geometry_values))};
        check(vtk::attribute_association(
                  quadratic, quadratic_geometry)
                == vtk::AttributeAssociation::point,
            "quadratic geometry data did not map to PointData");
    }

} // namespace

int main()
{
    return idmesh::test::run({
        {"cell mappings", cell_mappings},
        {"data association", data_association},
    });
}
