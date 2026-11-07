#include <idmesh/mpi_kokkos.hpp>
#include <cstddef>
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

int main() { return 0; }
