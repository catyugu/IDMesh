#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32) && defined(IDMESH_SHARED)
#if defined(IDMESH_EXPORTS)
#define IDMESH_API __declspec(dllexport)
#else
#define IDMESH_API __declspec(dllimport)
#endif
#elif defined(IDMESH_SHARED) && (defined(__GNUC__) || defined(__clang__))
#define IDMESH_API __attribute__((visibility("default")))
#else
#define IDMESH_API
#endif

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

    // Physical group membership survives even when its entity dimension is omitted.
    // An invalid entity ID identifies a vertex-keyed, unmaterialized physical element.
    struct PhysicalLabel {
        Index dimension = 0;
        Index entity = invalid_index;
        // Borrowed, immutable vertex sequence. Valid while the owning Mesh lives.
        // Materialized entities reuse the mesh connectivity; omitted layers use
        // a compact internal backing array.
        std::span<const Index> vertices;
        int tag = 0;
    };

    struct RelationRow {
        std::span<const Index> targets;
        std::span<const std::int8_t> coefficients;
        std::span<const Orientation> orientations;

        bool oriented() const noexcept { return !coefficients.empty(); }
    };

    struct RelationView {
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
        Mesh();
        Index dimension() const;
        Index geometric_dimension() const;
        Index entity_count(Index dimension) const;
        bool has_entities(Index dimension) const;
        bool has_relation(Index from, Index to) const;
        RelationView relation(Index from, Index to) const;
        std::span<const Index> entity_vertices(Index dimension, Index id) const;
        std::span<const std::array<double, 3>> coordinates() const;
        std::span<const CellType> cell_types() const;
        std::span<const int> cell_tags() const;
        // Sparse optional physical tags, indexed by facet ID; empty if untagged.
        std::span<const int> facet_tags() const;
        int facet_tag(Index facet) const;
        std::span<const PhysicalLabel> physical_labels() const;
        std::vector<int> physical_tags(Index dimension, Index entity) const;
        std::string physical_name(Index dimension, int tag) const;

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
        Index add_vertex(std::array<double, 3> position);
        Index add_cell(CellType type, std::span<const Index> vertices, int tag = 0);
        // Returns an index in the *extra* geometry-node pool, not a vertex ID.
        Index add_geometry_node(std::array<double, 3> position);
        Index add_quadratic_cell(CellType type, std::span<const Index> vertices,
            std::span<const Index> extra_nodes, int tag = 0);
        void add_cell_physical_tag(Index cell, int tag);
        void add_physical_tag(Index dimension, std::span<const Index> vertices, int tag);
        void add_facet_tag(std::span<const Index> vertices, int tag);
        void set_physical_name(Index dimension, int tag, std::string name);
        Mesh finalize(BuildOptions options = {}) &&;

    private:
        Index dimension_;
        Index geometric_dimension_;
        std::vector<std::array<double, 3>> coordinates_;
        std::vector<CellType> types_;
        std::vector<Index> cell_offsets_;
        std::vector<Index> cell_vertices_;
        std::vector<int> tags_;
        std::vector<std::array<double, 3>> extra_geometry_nodes_;
        std::vector<Index> cell_extra_offsets_;
        std::vector<Index> cell_extra_nodes_;
        std::vector<std::uint8_t> cell_orders_;
        struct PhysicalAssignment {
            Index dimension;
            Index begin;
            Index end;
            int tag;
        };
        std::vector<Index> physical_assignment_vertices_;
        std::vector<PhysicalAssignment> physical_assignments_;
        std::vector<std::pair<Index, int>> cell_physical_tags_;
        std::map<std::pair<Index, int>, std::string> physical_names_;
    };

    // Gmsh ASCII MSH 2.2 / 4.1: first- and second-order standard cells,
    // geometry-node connectivity, physical groups and names.
    IDMESH_API Mesh read_gmsh(const std::string& filename, BuildOptions options = {});
    // ASCII legacy VTK UNSTRUCTURED_GRID with standard linear and supported
    // quadratic cell types; one physical_tag attribute per exported cell.
    IDMESH_API Mesh read_vtk_legacy(const std::string& filename,
        BuildOptions options = {});
    IDMESH_API void write_vtk_legacy(const Mesh& mesh, const std::string& filename);
    // ASCII Medit .mesh, linear cells and native Edge/Triangle/Tetra P2;
    // one integer physical reference per exported cell.
    IDMESH_API Mesh read_medit(const std::string& filename,
        BuildOptions options = {});
    IDMESH_API void write_medit(const Mesh& mesh, const std::string& filename);
    IDMESH_API Index cell_dimension(CellType type);
    IDMESH_API Index cell_vertex_count(CellType type);
    // Local vertex indices of the oriented reference-cell facet.
    IDMESH_API Index cell_facet_count(CellType type);
    IDMESH_API std::span<const std::uint8_t> reference_facet_vertices(CellType type, Index facet);

} // namespace idmesh
