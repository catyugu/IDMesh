#pragma once

#include "idmesh/kokkos.hpp"
#include "idmesh/mpi.hpp"
#include <cstdint>
#include <memory>

#if defined(_WIN32)
#if defined(IDMESH_MPI_KOKKOS_EXPORTS)
#define IDMESH_MPI_KOKKOS_API __declspec(dllexport)
#elif defined(IDMESH_MPI_KOKKOS_SHARED)
#define IDMESH_MPI_KOKKOS_API __declspec(dllimport)
#else
#define IDMESH_MPI_KOKKOS_API
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define IDMESH_MPI_KOKKOS_API __attribute__((visibility("default")))
#else
#define IDMESH_MPI_KOKKOS_API
#endif

namespace idmesh::mpi::kokkos {

    // Auto chooses direct GPU-aware MPI when explicitly detected, otherwise
    // Host Staging. RequireGpuAware fails rather than passing unsupported
    // device pointers to an MPI implementation.
    enum class DeviceTransport : std::uint8_t {
        Automatic,
        HostStaging,
        RequireGpuAware
    };

    class IDMESH_MPI_KOKKOS_API DeviceHaloPlan {
    public:
        DeviceHaloPlan(const ExchangePlan& plan, Index components = 1);
        Index components() const noexcept;
        bool gpu_aware_available() const;
        // Blocking collective. Device packing/unpacking use Kokkos kernels.
        // Call after MPI_Init and Kokkos::initialize and before finalization.
        void exchange(idmesh::kokkos::DeviceCellField& field,
            DeviceTransport mode = DeviceTransport::Automatic) const;

    private:
        struct Data;
        std::shared_ptr<Data> data_;
    };

} // namespace idmesh::mpi::kokkos
