#include <idmesh/mpi.hpp>
int main() {
    idmesh::mpi::PartitionedMesh partition;
    return partition.owned_cells == 0 ? 0 : 1;
}
