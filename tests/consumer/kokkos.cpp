#include <idmesh/kokkos.hpp>
#include <type_traits>
static_assert(std::is_same_v<decltype(&idmesh::kokkos::incidence_count),
    std::uint64_t(*)(const idmesh::Mesh&,idmesh::Index,idmesh::Index)>);
int main(){return 0;}
