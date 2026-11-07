#pragma once

#include "idmesh/export.hpp"
#include "idmesh/mesh.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>

namespace idmesh::kokkos {
    // The application owns Kokkos initialization/finalization. Destroy every
    // device mirror before calling Kokkos::finalize().
    // Persistent mirrors own memory in Kokkos::DefaultExecutionSpace::memory_space
    // without exposing Kokkos types in the installed SDK ABI. Returned pointers are
    // device-space addresses and may only be dereferenced when that memory space is
    // host-accessible or wrapped by a backend-aware view.
    class IDMESH_KOKKOS_API DeviceMesh {
    public:
        explicit DeviceMesh(const Mesh& mesh);

        Index dimension() const noexcept;
        Index geometric_dimension() const noexcept;
        Index cell_count() const noexcept;
        Index geometry_node_count() const noexcept;
        std::size_t cell_vertex_entry_count() const noexcept;
        std::size_t high_order_geometry_entry_count() const noexcept;

        const CellType* cell_types() const noexcept;
        const std::uint8_t* cell_orders() const noexcept;
        const Index* cell_vertex_offsets() const noexcept;
        const Index* cell_vertices() const noexcept;
        // Rows are empty for linear cells; order-two rows store the full
        // geometry-node sequence, including corners.
        const Index* high_order_geometry_offsets() const noexcept;
        const Index* high_order_geometry_nodes() const noexcept;
        const double* geometry_x() const noexcept;
        const double* geometry_y() const noexcept;
        const double* geometry_z() const noexcept;

    private:
        struct Data;
        std::shared_ptr<Data> data_;
    };

    class IDMESH_KOKKOS_API DeviceRelation {
    public:
        DeviceRelation(const Mesh& mesh, Index from, Index to);

        Index row_count() const noexcept;
        std::size_t entry_count() const noexcept;
        bool oriented() const noexcept;
        const Index* offsets() const noexcept;
        const Index* targets() const noexcept;
        const std::int8_t* coefficients() const noexcept;
        const Orientation* orientations() const noexcept;

    private:
        struct Data;
        std::shared_ptr<Data> data_;
    };
} // namespace idmesh::kokkos
