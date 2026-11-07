#include "vtk_common.hpp"

#include "contracts.hpp"
#include "node_order.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
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
        const auto permutation = ::idmesh::detail::vtk_from_gmsh(
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
        const auto permutation = ::idmesh::detail::vtk_from_gmsh(
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

        for (const Index point : result.points)
            ::idmesh::detail::require_range(
                point < mesh.geometry_node_count(),
                "VTK geometry node out of range");
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

    ImportedMesh build_imported_mesh(
        std::vector<std::array<double, 3>> points,
        std::span<const ImportedCell> cells,
        BuildOptions options)
    {
        Index dimension = 0;
        for (const auto& cell : cells)
            dimension = std::max(
                dimension, cell.dimension);
        ::idmesh::detail::require_runtime(
            dimension >= 1, "VTK has no mesh cells");

        std::set<Index> corner_points;
        for (const auto& cell : cells) {
            const Index corners
                = cell.dimension == 0
                ? 1
                : cell_vertex_count(cell.type);
            ::idmesh::detail::require_runtime(
                cell.points.size() >= corners,
                "VTK cell has too few corner points");
            for (Index i = 0; i < corners; ++i)
                corner_points.insert(cell.points[i]);
        }

        MeshBuilder builder(dimension, 3);
        std::map<Index, Index> corner_ids;
        for (Index point : corner_points) {
            ::idmesh::detail::require_runtime(
                point < points.size(),
                "VTK point ID out of range");
            corner_ids[point]
                = builder.add_vertex(points[point]);
        }

        const Index corner_count
            = static_cast<Index>(corner_ids.size());
        std::map<Index, Index> extra_ids;
        std::vector<Index> point_to_geometry(
            points.size(), invalid_index);
        for (const auto& [point, vertex] : corner_ids)
            point_to_geometry[point] = vertex;

        for (const auto& cell : cells) {
            if (cell.dimension != dimension)
                continue;
            const auto gmsh = to_gmsh_order(
                cell.type, cell.points);
            std::vector<Index> vertices, extras;
            for (Index i = 0;
                 i < cell_vertex_count(cell.type); ++i)
                vertices.push_back(
                    corner_ids.at(gmsh[i]));
            for (Index i = cell_vertex_count(cell.type);
                 i < gmsh.size(); ++i) {
                const Index point = gmsh[i];
                ::idmesh::detail::require_runtime(
                    point < points.size(),
                    "VTK geometry point out of range");
                auto [it, inserted]
                    = extra_ids.try_emplace(
                        point, invalid_index);
                if (inserted) {
                    it->second
                        = builder.add_geometry_node(
                            points[point]);
                    point_to_geometry[point]
                        = corner_count + it->second;
                }
                extras.push_back(it->second);
            }
            const Index id = extras.empty()
                ? builder.add_cell(
                      cell.type, vertices)
                : builder.add_quadratic_cell(
                      cell.type, vertices, extras);
            if (cell.tag != 0)
                builder.add_cell_label(id, cell.tag);
        }

        bool lower = false;
        for (const auto& cell : cells) {
            if (cell.dimension >= dimension)
                continue;
            lower = true;
            if (cell.tag == 0)
                continue;
            std::vector<Index> vertices;
            const Index corners
                = cell.dimension == 0
                ? 1
                : cell_vertex_count(cell.type);
            for (Index i = 0; i < corners; ++i)
                vertices.push_back(
                    corner_ids.at(cell.points[i]));
            builder.add_label(
                cell.dimension, vertices, cell.tag);
        }

        auto mesh = std::move(builder).finalize(options);
        ::idmesh::detail::require_runtime(
            mesh.geometry_node_count() == points.size(),
            "VTK contains unused or unrepresentable points");
        return {std::move(mesh),
            std::move(point_to_geometry), lower};
    }

    data::Array reorder_point_array(
        data::Array input,
        std::span<const Index> point_to_geometry)
    {
        ::idmesh::detail::require_runtime(
            input.tuples == point_to_geometry.size(),
            "VTK PointData tuple count mismatch");
        const auto width
            = data::scalar_size(input.type);
        const auto tuple_bytes
            = static_cast<std::size_t>(
                  input.components) * width;
        data::Array output {
            input.type, input.tuples,
            input.components,
            std::vector<std::byte>(
                input.values.size())};
        std::vector<bool> seen(input.tuples);
        for (Index point = 0;
             point < point_to_geometry.size(); ++point) {
            const Index target
                = point_to_geometry[point];
            ::idmesh::detail::require_runtime(
                target < output.tuples
                    && !seen[target],
                "VTK point-to-geometry mapping is not bijective");
            seen[target] = true;
            std::copy_n(
                input.values.data()
                    + static_cast<std::size_t>(point)
                        * tuple_bytes,
                tuple_bytes,
                output.values.data()
                    + static_cast<std::size_t>(target)
                        * tuple_bytes);
        }
        ::idmesh::detail::require_runtime(
            std::all_of(
                seen.begin(), seen.end(),
                [](bool value) { return value; }),
            "VTK has points not represented by IDMesh geometry");
        return output;
    }


} // namespace idmesh::detail::vtk
