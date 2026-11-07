#include <Kokkos_Core.hpp>
#include <idmesh/mpi_kokkos.hpp>
#include <idmesh/partition.hpp>
#include <mpi.h>

#include <array>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

    using idmesh::Index;

    idmesh::Mesh create_source()
    {
        idmesh::MeshBuilder builder(2, 2);
        for (const auto& point : std::array<std::array<double, 3>, 9> {{{0, 0, 0}, {1, 0, 0}, {2, 0, 0},
                 {0, 1, 0}, {1, 1, 0}, {2, 1, 0},
                 {0, 2, 0}, {1, 2, 0}, {2, 2, 0}}})
            builder.add_vertex(point);
        const std::array<std::array<Index, 4>, 4> cells {{{0, 1, 4, 3}, {1, 2, 5, 4}, {3, 4, 7, 6}, {4, 5, 8, 7}}};
        for (const auto& cell : cells)
            builder.add_cell(idmesh::CellType::quadrilateral, cell);
        return std::move(builder).finalize(
            idmesh::BuildOptions {.topology = idmesh::TopologyLayout::cell_facets});
    }

    void check_exchange(const idmesh::mpi::PartitionedMesh& partition,
        Index dimension, idmesh::mpi::kokkos::DeviceTransport mode)
    {
        const auto count = partition.mesh().entity_count(dimension);
        idmesh::kokkos::DeviceField field(count, 2);
        std::vector<double> initial(std::size_t(count) * 2, -19);
        for (Index entity : partition.owned_entities(dimension)) {
            const auto id = partition.global_ids(dimension)[entity];
            initial[entity * 2] = double(2 * id + dimension + 1);
            initial[entity * 2 + 1] = double(3 * id + dimension + 7);
        }
        field.upload(initial);
        idmesh::mpi::kokkos::DeviceHaloPlan plan(
            partition.exchange_plan(dimension), 2);
        plan.start(field, mode);
        plan.wait();
        plan.start(field, mode);
        plan.wait();
        const auto values = field.download();
        for (Index entity = 0; entity < count; ++entity) {
            const auto id = partition.global_ids(dimension)[entity];
            if (values[entity * 2] != double(2 * id + dimension + 1)
                || values[entity * 2 + 1] != double(3 * id + dimension + 7))
                throw std::runtime_error("Kokkos device entity halo exchange mismatch");
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
        const int rank = context.rank();
        const int size = context.size();
        const auto prefix = (std::filesystem::temp_directory_path()
            / ("idmesh-mpi-kokkos-" + std::to_string(size)))
                                .string();

        if (rank == 0) {
            const auto source = create_source();
            const auto owners = idmesh::partition_graph(source, size);
            idmesh::write_partition_shards(source, owners, 1, prefix);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        auto partition = idmesh::mpi::read_partition_shard(
            context, prefix + "." + std::to_string(rank) + ".idm");
        for (Index dimension = 0; dimension <= partition.mesh().dimension();
             ++dimension) {
            if (!partition.mesh().has_entities(dimension))
                continue;
            check_exchange(partition, dimension,
                idmesh::mpi::kokkos::DeviceTransport::host_staging);
            check_exchange(partition, dimension,
                idmesh::mpi::kokkos::DeviceTransport::automatic);
        }

        const Index dimension = partition.mesh().dimension();
        idmesh::mpi::kokkos::DeviceHaloPlan probe(
            partition.exchange_plan(dimension), 2);
        {
            idmesh::kokkos::DeviceField field(
                partition.mesh().entity_count(dimension), 2);
            probe.start(field, idmesh::mpi::kokkos::DeviceTransport::host_staging);
            bool rejected = false;
            try {
                probe.start(field,
                    idmesh::mpi::kokkos::DeviceTransport::host_staging);
            }
            catch (const std::logic_error&) {
                rejected = true;
            }
            if (!rejected)
                throw std::runtime_error(
                    "second active device halo start was accepted");
            probe.wait();
        }
        if (probe.direct_device_transport_available()) {
            check_exchange(partition, dimension,
                idmesh::mpi::kokkos::DeviceTransport::require_gpu_aware);
        }
        else {
            idmesh::kokkos::DeviceField field(
                partition.mesh().entity_count(dimension), 2);
            bool rejected = false;
            try {
                probe.start(field,
                    idmesh::mpi::kokkos::DeviceTransport::require_gpu_aware);
            }
            catch (const std::runtime_error&) {
                rejected = true;
            }
            if (!rejected)
                throw std::runtime_error(
                    "unsupported direct GPU-aware MPI was accepted");
        }

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0)
            for (int peer = 0; peer < size; ++peer)
                std::filesystem::remove(
                    prefix + "." + std::to_string(peer) + ".idm");
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0)
            std::cout << "MPI+Kokkos: all-dimension device halo exchange OK\n";
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
