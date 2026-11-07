#pragma once

#include "idmesh/data.hpp"
#include "idmesh/mesh.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace idmesh::detail::vtk {

    struct CellInfo {
        CellType type = CellType::segment;
        Index dimension = 0;
        Index nodes = 0;
    };

    // Classical VTK unstructured-grid cell codes shared by legacy VTK and VTU.
    CellInfo cell_info(int code);
    int cell_code(CellType type, Index nodes);
    int entity_code(Index dimension, Index arity);

    // IDMesh geometry connectivity uses Gmsh ordering. These conversions are
    // exact permutations; unsupported VTK cell families are rejected by
    // cell_code/cell_info rather than approximated.
    std::vector<Index> to_vtk_order(
        CellType type, std::span<const Index> gmsh_nodes);
    std::vector<Index> to_gmsh_order(
        CellType type, std::span<const Index> vtk_nodes);

    // VTK points are emitted in IDMesh global geometry-node order. Making this
    // mapping explicit keeps PointData tuple order identical across VTK writers.
    struct PointNumbering {
        Index count = 0;

        Index point_for_geometry(Index geometry_node) const;
        Index geometry_for_point(Index point) const;
    };

    PointNumbering point_numbering(const Mesh& mesh);

    struct EncodedCell {
        int code = 0;
        std::vector<Index> points;
    };

    EncodedCell encode_cell(const Mesh& mesh, Index cell);

    enum class AttributeAssociation : std::uint8_t {
        point,
        cell,
        field
    };

    // Classifies only representations that VTK can preserve exactly as a
    // dense PointData/CellData/FieldData array. Sparse selections and
    // intermediate entity dimensions are rejected.
    AttributeAssociation attribute_association(
        const Mesh& mesh, const data::MeshDataView& field);

    struct ImportedCell {
        CellType type = CellType::segment;
        Index dimension = 0;
        std::vector<Index> points;
        int tag = 0;
    };

    struct ImportedMesh {
        Mesh mesh;
        std::vector<Index> point_to_geometry;
        bool has_lower_cells = false;
    };

    ImportedMesh build_imported_mesh(
        std::vector<std::array<double, 3>> points,
        std::span<const ImportedCell> cells,
        BuildOptions options);

    data::Array reorder_point_array(
        data::Array input,
        std::span<const Index> point_to_geometry);

} // namespace idmesh::detail::vtk
