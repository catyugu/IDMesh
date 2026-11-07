#include <idmesh/kokkos.hpp>
#include <type_traits>
#include <utility>

static_assert(std::is_same_v<
    decltype(std::declval<const idmesh::kokkos::DeviceRelation&>().targets()),
    const idmesh::Index*>);
static_assert(std::is_same_v<
    decltype(std::declval<const idmesh::kokkos::DeviceMesh&>().geometry_x()),
    const double*>);

int main() { return 0; }
