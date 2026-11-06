#include <Kokkos_Core.hpp>
#include <idmesh/mpi_kokkos.hpp>
#include <mpi.h>

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

    idmesh::Mesh create_source()
    {
        idmesh::MeshBuilder builder(2, 2);
        for (const auto& point : std::array<std::array<double, 3>, 9> {{{0, 0, 0}, {1, 0, 0}, {2, 0, 0},
                 {0, 1, 0}, {1, 1, 0}, {2, 1, 0},
                 {0, 2, 0}, {1, 2, 0}, {2, 2, 0}}})
            builder.add_vertex(point);
        const std::array<std::array<idmesh::Index, 4>, 4> cells {{{0, 1, 4, 3}, {1, 2, 5, 4}, {3, 4, 7, 6}, {4, 5, 8, 7}}};
        for (const auto& cell : cells)
            builder.add_cell(idmesh::CellType::quadrilateral, cell);
        return std::move(builder).finalize(
            idmesh::BuildOptions {.topology = idmesh::TopologyLayout::cell_facets});
    }

    void check_exchange(const idmesh::mpi::PartitionedMesh& partition,
        idmesh::mpi::kokkos::DeviceTransport mode)
    {
        const auto count = partition.mesh().entity_count(partition.mesh().dimension());
        idmesh::kokkos::DeviceField field(count, 2);
        std::vector<double> initial(std::size_t(count) * 2, -19);
        for (idmesh::Index i = 0; i < partition.owned_cell_count(); ++i) {
            const auto id = partition.global_ids(partition.mesh().dimension())[i];
            initial[i * 2] = double(2 * id + 1);
            initial[i * 2 + 1] = double(3 * id + 7);
        }
        field.upload(initial);
        idmesh::mpi::kokkos::DeviceHaloPlan plan(partition.exchange_plan(), 2);
        plan.exchange(field, mode);
        const auto values = field.download();
        for (idmesh::Index i = 0; i < count; ++i) {
            const auto id = partition.global_ids(partition.mesh().dimension())[i];
            if (values[i * 2] != double(2 * id + 1) || values[i * 2 + 1] != double(3 * id + 7))
                throw std::runtime_error("Kokkos device halo exchange mismatch");
        }
    }

} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);
    int rc = 0;
    try {
        const auto context = idmesh::mpi::Context::world();
        const auto source = create_source();
        const auto partition = idmesh::mpi::partition_replicated(context, source, 1);
        check_exchange(partition, idmesh::mpi::kokkos::DeviceTransport::host_staging);
        check_exchange(partition, idmesh::mpi::kokkos::DeviceTransport::automatic);
        idmesh::mpi::kokkos::DeviceHaloPlan probe(partition.exchange_plan(), 2);
        if (probe.direct_device_transport_available()) {
            // On a GPU-aware MPI runner this exercises direct device buffers.
            check_exchange(partition, idmesh::mpi::kokkos::DeviceTransport::require_gpu_aware);
        }
        else {
            idmesh::kokkos::DeviceField field(partition.mesh().entity_count(
                                                  partition.mesh().dimension()),
                2);
            bool rejected = false;
            try {
                probe.exchange(field, idmesh::mpi::kokkos::DeviceTransport::require_gpu_aware);
            }
            catch (const std::runtime_error&) {
                rejected = true;
            }
            if (!rejected)
                throw std::runtime_error("unsupported direct GPU-aware MPI was accepted");
        }
        if (context.rank() == 0)
            std::cout << "MPI+Kokkos: packed device field halo (staging/auto) OK\n";
    }
    catch (const std::exception& e) {
        std::cerr << "MPI+Kokkos test failure: " << e.what() << '\n';
        rc = 1;
    }
    Kokkos::finalize();
    if (rc)
        MPI_Abort(MPI_COMM_WORLD, rc);
    MPI_Finalize();
    return rc;
}
