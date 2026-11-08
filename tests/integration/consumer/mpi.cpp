#include <idmesh/mpi.hpp>
#include <stdexcept>
#include <type_traits>

static_assert(!std::is_default_constructible_v<idmesh::mpi::Context>);
static_assert(std::is_copy_constructible_v<idmesh::mpi::Context>);
static_assert(!std::is_default_constructible_v<idmesh::mpi::PartitionedMesh>);
static_assert(std::is_copy_constructible_v<idmesh::mpi::PartitionedMesh>);

int main()
{
    // Exercises the exported MPI adapter without exposing mpi.h to this client.
    try {
        (void)idmesh::mpi::Context::world();
    }
    catch (const std::logic_error&) {
        return 0; // The caller has not initialized MPI.
    }
    return 1;
}
