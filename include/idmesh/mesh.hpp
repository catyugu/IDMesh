#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#if defined(_WIN32) && defined(IDMESH_SHARED)
#  if defined(IDMESH_EXPORTS)
#    define IDMESH_API __declspec(dllexport)
#  else
#    define IDMESH_API __declspec(dllimport)
#  endif
#elif defined(IDMESH_SHARED) && (defined(__GNUC__) || defined(__clang__))
#  define IDMESH_API __attribute__((visibility("default")))
#else
#  define IDMESH_API
#endif

namespace idmesh {

using Index = std::uint32_t;
constexpr Index invalid_index = ~Index{0};

enum class CellType : std::uint8_t {
    segment = 1, triangle, quadrilateral, tetrahedron, hexahedron, prism, pyramid
};

struct TopologyOptions {
    bool edges = true;
    bool faces = true;
    bool facets = true;
    bool support = true;

    static TopologyOptions full() { return {}; }
    static TopologyOptions finite_volume() { return {false, false, true, true}; }
    static TopologyOptions cells_and_vertices() { return {false, false, false, false}; }
};

// The canonical sequence is local[(rotation + i * (reversed ? -1 : 1)) mod n].
// Sign alone is insufficient for higher-order facet transformations.
struct Orientation {
    std::uint16_t rotation = 0;
    bool reversed = false;
    constexpr int sign() const { return reversed ? -1 : 1; }
};

struct Incidence {
    Index entity = invalid_index;
    std::int8_t sign = 0;
    Orientation orientation {};
};

struct RelationView {
    std::span<const Index> offsets;
    std::span<const Index> indices;
    std::span<const std::int8_t> signs;
    std::span<const Orientation> orientations;
    Index rows = 0;
    Index stride = 0;

    std::span<const Index> row(Index index) const;
    std::span<const std::int8_t> row_signs(Index index) const;
    std::span<const Orientation> row_orientations(Index index) const;
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
    void validate() const;

private:
    struct Data;
    std::shared_ptr<const Data> data_;
    explicit Mesh(std::shared_ptr<const Data> data);
    friend class MeshBuilder;
};

class IDMESH_API MeshBuilder {
public:
    MeshBuilder(Index dimension, Index geometric_dimension);
    Index add_vertex(std::array<double, 3> position);
    Index add_cell(CellType type, std::span<const Index> vertices, int tag = 0);
    Mesh finalize(TopologyOptions options = TopologyOptions::full()) &&;

private:
    Index dimension_;
    Index geometric_dimension_;
    std::vector<std::array<double, 3>> coordinates_;
    std::vector<CellType> types_;
    std::vector<std::vector<Index>> cells_;
    std::vector<int> tags_;
};

// Gmsh first-order ASCII MSH 2.2 / 4.1; higher-order elements are rejected.
IDMESH_API Mesh read_gmsh(const std::string& filename, TopologyOptions options = TopologyOptions::full());
IDMESH_API Index cell_dimension(CellType type);
IDMESH_API Index cell_vertex_count(CellType type);

} // namespace idmesh
