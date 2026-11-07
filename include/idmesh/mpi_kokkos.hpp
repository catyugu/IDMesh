#pragma once

#include "idmesh/export.hpp"

#include "idmesh/kokkos.hpp"
#include "idmesh/mpi.hpp"
#include <cstddef>
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

    // MPI and Kokkos must remain active for the lifetime of this object.
    class IDMESH_MPI_KOKKOS_API DeviceHaloPlan {
    public:
        DeviceHaloPlan(
            const ExchangePlan& plan, std::size_t bytes_per_entity);
        ~DeviceHaloPlan();
        DeviceHaloPlan(DeviceHaloPlan&&) noexcept;
        DeviceHaloPlan& operator=(DeviceHaloPlan&&) noexcept;
        DeviceHaloPlan(const DeviceHaloPlan&) = delete;
        DeviceHaloPlan& operator=(const DeviceHaloPlan&) = delete;

        std::size_t bytes_per_entity() const noexcept;
        bool direct_device_transport_available() const;

        // Packs fixed-width records from caller-owned device memory, performs
        // any required D2H staging, then starts the sparse MPI neighborhood exchange. The device
        // storage must remain alive until wait() or destruction completes.
        // Destroying an active plan waits for MPI but discards received ghosts;
        // only wait() unpacks into caller-owned device storage.
        void start(void* device_data, std::size_t byte_size,
            DeviceTransport mode = DeviceTransport::automatic);
        // Completes MPI, performs any required H2D staging, and unpacks ghosts.
        // Calling wait() with no active exchange is a no-op.
        void wait();

    private:
        struct Data;
        std::unique_ptr<Data> data_;
    };

} // namespace idmesh::mpi::kokkos
