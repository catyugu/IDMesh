#include <idmesh/mpi_kokkos.hpp>
#include <type_traits>

static_assert(std::is_default_constructible_v<idmesh::mpi::ExchangePlan>);
static_assert(std::is_same_v<decltype(idmesh::mpi::gpu_aware_support(
    idmesh::mpi::DeviceMemoryKind::Host)), bool>);

int main() { return 0; }
