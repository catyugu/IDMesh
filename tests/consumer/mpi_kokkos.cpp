#include <idmesh/mpi_kokkos.hpp>
#include <type_traits>
#include <utility>

static_assert(std::is_default_constructible_v<idmesh::mpi::ExchangePlan>);
static_assert(std::is_move_constructible_v<idmesh::mpi::kokkos::DeviceHaloPlan>);
static_assert(!std::is_copy_constructible_v<idmesh::mpi::kokkos::DeviceHaloPlan>);

void link_device_halo_api(idmesh::mpi::kokkos::DeviceHaloPlan& plan,
    idmesh::kokkos::DeviceField& field)
{
    plan.start(field, idmesh::mpi::kokkos::DeviceTransport::host_staging);
    plan.wait();
}

int main() { return 0; }
