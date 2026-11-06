#pragma once
#include "idmesh/mesh.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#if defined(_WIN32)
#if defined(IDMESH_KOKKOS_EXPORTS)
#define IDMESH_KOKKOS_API __declspec(dllexport)
#elif defined(IDMESH_KOKKOS_SHARED)
#define IDMESH_KOKKOS_API __declspec(dllimport)
#else
#define IDMESH_KOKKOS_API
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define IDMESH_KOKKOS_API __attribute__((visibility("default")))
#else
#define IDMESH_KOKKOS_API
#endif

namespace idmesh::kokkos {
    // A simple device-ready topology traversal. No Kokkos types escape the SDK ABI.
    IDMESH_KOKKOS_API std::uint64_t incidence_count(const Mesh& mesh, Index from, Index to);

    // Persistent mirror of one materialized topology relation in the default
    // Kokkos memory space. No Kokkos header or template is exposed in this ABI.
    // Construct only after Kokkos::initialize(), destroy before finalize().
    // Persistent mirror of topological cell corners plus all geometry nodes.
    // Device storage uses SoA coordinates and CSR cell connectivity.
    // Batched numerical reductions run on the default Kokkos execution space.
    class IDMESH_KOKKOS_API DeviceMesh {
    public:
        explicit DeviceMesh(const Mesh& mesh);
        Index cells() const noexcept;
        Index geometry_nodes() const noexcept;
        std::array<double, 3> centroid_sum() const;
        std::uint64_t geometry_connectivity_checksum() const;

    private:
        struct Data;
        std::shared_ptr<Data> data_;
    };

    // Kokkos-owned device field storage. Pointers are device-space addresses:
    // never dereference them on Host unless that memory space is host accessible.
    // All fields and kernels require an initialized Kokkos runtime.
    class IDMESH_KOKKOS_API DeviceCellField {
    public:
        explicit DeviceCellField(Index cells, Index components = 1);
        Index cells() const noexcept;
        Index components() const noexcept;
        void upload(std::span<const double> values);
        std::vector<double> download() const;
        void* device_data() noexcept;
        const void* device_data() const noexcept;

    private:
        struct Data;
        std::shared_ptr<Data> data_;
    };

    class IDMESH_KOKKOS_API DeviceRelation {
    public:
        DeviceRelation(const Mesh& mesh, Index from, Index to);
        Index rows() const noexcept;
        std::uint64_t incidence_count() const;
        std::uint64_t weighted_index_sum() const;
        std::int64_t boundary_coefficient_sum() const;

    private:
        struct Data;
        std::shared_ptr<Data> data_;
    };
} // namespace idmesh::kokkos
