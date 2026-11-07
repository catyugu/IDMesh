#include <Kokkos_Core.hpp>
#include <idmesh/mpi_kokkos.hpp>
#include <idmesh/partition.hpp>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <cstdint>
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
        using Value = std::uint64_t;
        Kokkos::View<Value*> field(
            "idmesh_test_caller_owned_field", std::size_t(count) * 2);
        auto host = Kokkos::create_mirror_view(field);
        for (std::size_t i = 0; i < host.extent(0); ++i)
            host(i) = ~Value {0};
        for (Index entity : partition.owned_entities(dimension)) {
            const auto id = partition.global_ids(dimension)[entity];
            host(entity * 2) = 2 * id + dimension + 1;
            host(entity * 2 + 1) = 3 * id + dimension + 7;
        }
        Kokkos::deep_copy(field, host);

        const auto bytes_per_entity = 2 * sizeof(Value);
        const auto byte_size = field.extent(0) * sizeof(Value);
        std::vector<Value> before(host.extent(0));
        for (std::size_t i = 0; i < before.size(); ++i)
            before[i] = host(i);
        {
            idmesh::mpi::kokkos::DeviceHaloPlan abandoned(
                partition.exchange_plan(dimension), bytes_per_entity);
            abandoned.start(field.data(), byte_size, mode);
        }
        Kokkos::deep_copy(host, field);
        for (std::size_t i = 0; i < before.size(); ++i)
            if (host(i) != before[i])
                throw std::runtime_error("abandoned device halo updated ghost records");

        idmesh::mpi::kokkos::DeviceHaloPlan plan(
            partition.exchange_plan(dimension), bytes_per_entity);
        plan.start(field.data(), byte_size, mode);
        plan.wait();
        plan.start(field.data(), byte_size, mode);
        plan.wait();

        Kokkos::deep_copy(host, field);
        for (Index entity = 0; entity < count; ++entity) {
            const auto id = partition.global_ids(dimension)[entity];
            if (host(entity * 2) != 2 * id + dimension + 1
                || host(entity * 2 + 1) != 3 * id + dimension + 7)
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
        constexpr std::size_t values_per_entity = 2;
        using ProbeValue = std::uint64_t;
        idmesh::mpi::kokkos::DeviceHaloPlan probe(
            partition.exchange_plan(dimension),
            values_per_entity * sizeof(ProbeValue));
        {
            Kokkos::View<ProbeValue*> field(
                "idmesh_test_probe",
                std::size_t(partition.mesh().entity_count(dimension))
                    * values_per_entity);
            const auto bytes = field.extent(0) * sizeof(ProbeValue);
            probe.start(field.data(), bytes,
                idmesh::mpi::kokkos::DeviceTransport::host_staging);
            bool rejected = false;
            try {
                probe.start(field.data(), bytes,
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
            Kokkos::View<ProbeValue*> field(
                "idmesh_test_direct_probe",
                std::size_t(partition.mesh().entity_count(dimension))
                    * values_per_entity);
            const auto bytes = field.extent(0) * sizeof(ProbeValue);
            bool rejected = false;
            try {
                probe.start(field.data(), bytes,
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
