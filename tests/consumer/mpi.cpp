#include <idmesh/mpi.hpp>
#include <type_traits>

static_assert(!std::is_default_constructible_v<idmesh::mpi::Context>);
static_assert(std::is_copy_constructible_v<idmesh::mpi::Context>);
static_assert(!std::is_default_constructible_v<idmesh::mpi::PartitionedMesh>);
static_assert(std::is_copy_constructible_v<idmesh::mpi::PartitionedMesh>);

int main() { return 0; }
