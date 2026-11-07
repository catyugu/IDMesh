#pragma once

#include "idmesh/export.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>

namespace idmesh {

    using Index = std::uint32_t;
    constexpr Index invalid_index = ~Index {0};

    enum class CellType : std::uint8_t {
        segment = 1,
        triangle,
        quadrilateral,
        tetrahedron,
        hexahedron,
        prism,
        pyramid
    };

    enum class TopologyLayout : std::uint8_t {
        cell_vertices,
        cell_facets,
        complete
    };

    enum class RelationDirection : std::uint8_t {
        downward,
        bidirectional
    };

    struct BuildOptions {
        TopologyLayout topology = TopologyLayout::complete;
        RelationDirection relations = RelationDirection::bidirectional;
    };

    // The canonical sequence is local[(rotation + i * (reversed ? -1 : 1)) mod n].
    // Sign alone is insufficient for higher-order facet transformations.
    struct Orientation {
        std::uint16_t rotation = 0;
        bool reversed = false;
        constexpr int sign() const { return reversed ? -1 : 1; }
    };

    // Generic integer metadata attached to mesh entities. If a topology layer is
    // omitted, entity is invalid_index and vertices keep the canonical entity key.
    struct Label {
        Index dimension = 0;
        Index entity = invalid_index;
        std::span<const Index> vertices;
        int value = 0;
    };

    struct RelationRow {
        std::span<const Index> targets;
        std::span<const std::int8_t> coefficients;
        std::span<const Orientation> orientations;

        bool oriented() const noexcept { return !coefficients.empty(); }
    };

    struct IDMESH_API RelationView {
        std::span<const Index> offsets;
        std::span<const Index> targets;
        std::span<const std::int8_t> coefficients;
        std::span<const Orientation> orientations;
        Index rows = 0;
        Index stride = 0;

        Index size() const noexcept { return rows; }
        bool oriented() const noexcept { return !coefficients.empty(); }
        RelationRow operator[](Index row) const;
    };

    class IDMESH_API Mesh {
    public:
        Index dimension() const;
        Index geometric_dimension() const;
        Index entity_count(Index dimension) const;
        bool has_entities(Index dimension) const;
        bool has_relation(Index from, Index to) const;
        RelationView relation(Index from, Index to) const;
        std::span<const Index> entity_vertices(Index dimension, Index id) const;
        std::span<const std::array<double, 3>> coordinates() const;
        std::span<const CellType> cell_types() const;
        std::span<const Label> labels() const;
        std::span<const Label> labels(Index dimension, Index entity) const;
        std::string_view label_name(Index dimension, int value) const;

        // Geometry node IDs are separate from topological vertex IDs:
        // 0..num_vertices-1 are corners, following IDs are higher-order nodes.
        std::uint8_t cell_order(Index cell) const;
        std::span<const Index> cell_geometry_nodes(Index cell) const;
        std::array<double, 3> geometry_node_position(Index node) const;
        Index geometry_node_count() const;
        void validate() const;

    private:
        struct Data;
        std::shared_ptr<const Data> data_;
        explicit Mesh(std::shared_ptr<const Data> data);
        void validate_structure() const;
        friend class MeshBuilder;
    };

    class IDMESH_API MeshBuilder {
    public:
        MeshBuilder(Index dimension, Index geometric_dimension);
        ~MeshBuilder();
        MeshBuilder(MeshBuilder&&) noexcept;
        MeshBuilder& operator=(MeshBuilder&&) noexcept;
        MeshBuilder(const MeshBuilder&) = delete;
        MeshBuilder& operator=(const MeshBuilder&) = delete;

        Index add_vertex(std::array<double, 3> position);
        Index add_cell(CellType type, std::span<const Index> vertices);
        // Returns an index in the extra geometry-node pool, not a vertex ID.
        Index add_geometry_node(std::array<double, 3> position);
        Index add_quadratic_cell(CellType type, std::span<const Index> vertices,
            std::span<const Index> extra_nodes);
        void add_cell_label(Index cell, int value);
        void add_label(Index dimension, std::span<const Index> vertices, int value);
        void set_label_name(Index dimension, int value, std::string name);
        Mesh finalize(BuildOptions options = {}) &&;

    private:
        struct Data;
        std::unique_ptr<Data> data_;
    };

    // Gmsh ASCII MSH 2.2 / 4.1: first- and second-order standard cells,
    // geometry-node connectivity, physical groups and names.
    IDMESH_API Mesh read_gmsh(const std::string& filename, BuildOptions options = {});
    // Mesh-only compatibility wrappers. The reader accepts ASCII/BINARY legacy
    // UNSTRUCTURED_GRID; the writer emits ASCII. Field/state and VTU APIs live
    // in idmesh/vtk.hpp.
    IDMESH_API Mesh read_vtk_legacy(const std::string& filename,
        BuildOptions options = {});
    IDMESH_API void write_vtk_legacy(const Mesh& mesh, const std::string& filename);
    // ASCII Medit .mesh, linear cells and native Edge/Triangle/Tetra P2;
    // one nonzero integer label/reference per exported entity.
    IDMESH_API Mesh read_medit(const std::string& filename,
        BuildOptions options = {});
    IDMESH_API void write_medit(const Mesh& mesh, const std::string& filename);
    IDMESH_API Index cell_dimension(CellType type);
    IDMESH_API Index cell_vertex_count(CellType type);
    // Local vertex indices of the oriented reference-cell facet.
    IDMESH_API Index cell_facet_count(CellType type);
    IDMESH_API std::span<const std::uint8_t> reference_facet_vertices(CellType type, Index facet);

} // namespace idmesh
