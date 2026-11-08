#include <idmesh/mpi_kokkos.hpp>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>

static_assert(std::is_default_constructible_v<idmesh::mpi::ExchangePlan>);
static_assert(std::is_move_constructible_v<idmesh::mpi::kokkos::DeviceHaloPlan>);
static_assert(!std::is_copy_constructible_v<idmesh::mpi::kokkos::DeviceHaloPlan>);

void link_device_halo_api(idmesh::mpi::kokkos::DeviceHaloPlan& plan,
    void* device_data, std::size_t byte_size)
{
    plan.start(device_data, byte_size,
        idmesh::mpi::kokkos::DeviceTransport::host_staging);
    plan.wait();
}

int main()
{
    // Verify that the installed combined adapter is loadable and callable.
    try {
        const idmesh::mpi::ExchangePlan empty;
        idmesh::mpi::kokkos::DeviceHaloPlan device(empty, sizeof(double));
    }
    catch (const std::logic_error&) {
        return 0; // MPI/Kokkos are deliberately uninitialized.
    }
    return 1;
}
