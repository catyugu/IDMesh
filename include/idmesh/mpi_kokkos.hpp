#pragma once

#include "idmesh/export.hpp"

#include "idmesh/kokkos.hpp"
#include "idmesh/mpi.hpp"
#include <cstdint>
#include <memory>

namespace idmesh::mpi::kokkos {

    // automatic chooses direct GPU-aware MPI when explicitly detected,
    // otherwise host staging. require_gpu_aware fails before unsupported
    // device pointers reach the MPI implementation.
    enum class DeviceTransport : std::uint8_t {
        automatic,
        host_staging,
        require_gpu_aware
    };

    class IDMESH_MPI_KOKKOS_API DeviceHaloPlan {
    public:
        DeviceHaloPlan(const ExchangePlan& plan, Index components = 1);
        Index component_count() const noexcept;
        bool direct_device_transport_available() const;
        // Blocking collective. Device packing/unpacking use Kokkos kernels.
        // Call after MPI_Init and Kokkos::initialize and before finalization.
        void exchange(idmesh::kokkos::DeviceField& field,
            DeviceTransport mode = DeviceTransport::automatic) const;

    private:
        struct Data;
        std::shared_ptr<Data> data_;
    };

} // namespace idmesh::mpi::kokkos
