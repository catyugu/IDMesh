#include "vtk_common.hpp"

#include "contracts.hpp"
#include "second_order.hpp"

#include <stdexcept>

namespace idmesh::detail::vtk {

    CellInfo cell_info(int code)
    {
        switch (code) {
        case 1:
            return {CellType::segment, 0, 1};
        case 3:
            return {CellType::segment, 1, 2};
        case 5:
            return {CellType::triangle, 2, 3};
        case 9:
            return {CellType::quadrilateral, 2, 4};
        case 10:
            return {CellType::tetrahedron, 3, 4};
        case 12:
            return {CellType::hexahedron, 3, 8};
        case 13:
            return {CellType::prism, 3, 6};
        case 14:
            return {CellType::pyramid, 3, 5};
        case 21:
            return {CellType::segment, 1, 3};
        case 22:
            return {CellType::triangle, 2, 6};
        case 23:
            return {CellType::quadrilateral, 2, 8};
        case 24:
            return {CellType::tetrahedron, 3, 10};
        case 25:
            return {CellType::hexahedron, 3, 20};
        case 26:
            return {CellType::prism, 3, 15};
        case 27:
            return {CellType::pyramid, 3, 13};
        case 28:
            return {CellType::quadrilateral, 2, 9};
        case 29:
            return {CellType::hexahedron, 3, 27};
        case 32:
            return {CellType::prism, 3, 18};
        default:
            throw std::runtime_error("unsupported VTK cell type");
        }
    }

    int cell_code(CellType type, Index nodes)
    {
        switch (type) {
        case CellType::segment:
            if (nodes == 2)
                return 3;
            if (nodes == 3)
                return 21;
            break;
        case CellType::triangle:
            if (nodes == 3)
                return 5;
            if (nodes == 6)
                return 22;
            break;
        case CellType::quadrilateral:
            if (nodes == 4)
                return 9;
            if (nodes == 8)
                return 23;
            if (nodes == 9)
                return 28;
            break;
        case CellType::tetrahedron:
            if (nodes == 4)
                return 10;
            if (nodes == 10)
                return 24;
            break;
        case CellType::hexahedron:
            if (nodes == 8)
                return 12;
            if (nodes == 20)
                return 25;
            if (nodes == 27)
                return 29;
            break;
        case CellType::prism:
            if (nodes == 6)
                return 13;
            if (nodes == 15)
                return 26;
            if (nodes == 18)
                return 32;
            break;
        case CellType::pyramid:
            if (nodes == 5)
                return 14;
            if (nodes == 13)
                return 27;
            break;
        }
        throw std::runtime_error(
            "no exact VTK cell type for IDMesh geometry");
    }

    int entity_code(Index dimension, Index arity)
    {
        if (dimension == 0 && arity == 1)
            return 1;
        if (dimension == 1 && arity == 2)
            return 3;
        if (dimension == 2 && arity == 3)
            return 5;
        if (dimension == 2 && arity == 4)
            return 9;
        throw std::runtime_error(
            "no exact VTK cell type for labeled entity");
    }

    std::vector<Index> to_vtk_order(
        CellType type, std::span<const Index> gmsh_nodes)
    {
        const auto permutation = detail::vtk_from_gmsh(
            type, static_cast<Index>(gmsh_nodes.size()));
        if (permutation.empty())
            return {gmsh_nodes.begin(), gmsh_nodes.end()};

        ::idmesh::detail::require_runtime(
            permutation.size() == gmsh_nodes.size(),
            "invalid VTK node permutation");
        std::vector<Index> result(gmsh_nodes.size());
        for (Index i = 0; i < gmsh_nodes.size(); ++i)
            result[i] = gmsh_nodes[permutation[i]];
        return result;
    }

    std::vector<Index> to_gmsh_order(
        CellType type, std::span<const Index> vtk_nodes)
    {
        const auto permutation = detail::vtk_from_gmsh(
            type, static_cast<Index>(vtk_nodes.size()));
        if (permutation.empty())
            return {vtk_nodes.begin(), vtk_nodes.end()};

        ::idmesh::detail::require_runtime(
            permutation.size() == vtk_nodes.size(),
            "invalid VTK node permutation");
        std::vector<Index> result(vtk_nodes.size());
        for (Index i = 0; i < vtk_nodes.size(); ++i)
            result[permutation[i]] = vtk_nodes[i];
        return result;
    }

    Index PointNumbering::point_for_geometry(
        Index geometry_node) const
    {
        ::idmesh::detail::require_range(
            geometry_node < count,
            "VTK geometry node out of range");
        return geometry_node;
    }

    Index PointNumbering::geometry_for_point(Index point) const
    {
        ::idmesh::detail::require_range(
            point < count, "VTK point out of range");
        return point;
    }

    PointNumbering point_numbering(const Mesh& mesh)
    {
        return {mesh.geometry_node_count()};
    }

    EncodedCell encode_cell(const Mesh& mesh, Index cell)
    {
        ::idmesh::detail::require_range(
            cell < mesh.entity_count(mesh.dimension()),
            "VTK cell index out of range");

        const auto geometry = mesh.cell_geometry_nodes(cell);
        EncodedCell result;
        result.code = cell_code(
            mesh.cell_types()[cell],
            static_cast<Index>(geometry.size()));
        result.points = to_vtk_order(
            mesh.cell_types()[cell], geometry);

        const auto numbering = point_numbering(mesh);
        for (auto& point : result.points)
            point = numbering.point_for_geometry(point);
        return result;
    }

    AttributeAssociation attribute_association(
        const Mesh& mesh, const data::MeshDataView& field)
    {
        data::validate(mesh, field);
        ::idmesh::detail::require_argument(
            field.selection.kind
                == data::SelectionKind::all,
            "VTK attributes require dense all-location data");

        switch (field.location.kind) {
        case data::LocationKind::geometry_node:
            return AttributeAssociation::point;

        case data::LocationKind::mesh:
            return AttributeAssociation::field;

        case data::LocationKind::entity:
            if (field.location.dimension
                == mesh.dimension())
                return AttributeAssociation::cell;
            if (field.location.dimension == 0) {
                ::idmesh::detail::require_argument(
                    mesh.entity_count(0)
                        == mesh.geometry_node_count(),
                    "VTK PointData cannot represent vertex-only data "
                    "when high-order geometry nodes are present");
                return AttributeAssociation::point;
            }
            throw std::invalid_argument(
                "VTK attributes cannot represent intermediate "
                "entity-dimension data");
        }
        throw std::invalid_argument("invalid VTK data location");
    }

} // namespace idmesh::detail::vtk
