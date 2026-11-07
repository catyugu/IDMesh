#include "idmesh/mpi_kokkos.hpp"
#include "idmesh/partition.hpp"
#include "mesh_factory.hpp"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
    using Clock = std::chrono::steady_clock;
    using Index = idmesh::Index;

    struct Arguments {
        unsigned side = 128;
        unsigned iterations = 200;
        Index components = 1;
        Index ghost_layers = 1;
    };

    unsigned parse_unsigned(const char* text)
    {
        std::size_t consumed = 0;
        const auto value = std::stoul(text, &consumed);
        if (consumed != std::string(text).size())
            throw std::invalid_argument("invalid unsigned integer");
        return static_cast<unsigned>(value);
    }

    Arguments parse_arguments(int argc, char** argv)
    {
        Arguments args;
        for (int i = 1; i < argc; ++i) {
            const std::string token = argv[i];
            if (token == "--help") {
                std::cout << "Usage: idmesh_mpi_kokkos_benchmark "
                             "[--side N] [--iterations N] [--components N] "
                             "[--ghost-layers N]\n";
                std::exit(0);
            }
            if (++i >= argc)
                throw std::invalid_argument("missing benchmark option value");
            if (token == "--side")
                args.side = parse_unsigned(argv[i]);
            else if (token == "--iterations")
                args.iterations = parse_unsigned(argv[i]);
            else if (token == "--components")
                args.components = parse_unsigned(argv[i]);
            else if (token == "--ghost-layers")
                args.ghost_layers = parse_unsigned(argv[i]);
            else
                throw std::invalid_argument("unknown benchmark option: " + token);
        }
        if (args.side == 0 || args.iterations == 0 || args.components == 0
            || args.ghost_layers > 8)
            throw std::invalid_argument("invalid benchmark argument");
        return args;
    }

    struct Timing {
        std::vector<double> samples;

        double median() const
        {
            auto values = samples;
            std::sort(values.begin(), values.end());
            return values[values.size() / 2];
        }
    };

    double milliseconds(Clock::time_point begin, Clock::time_point end)
    {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);
    int status = 1;
    try {
        const auto args = parse_arguments(argc, argv);
        const auto context = idmesh::mpi::Context::world();

        idmesh_bench::GeneratedMesh generated;
        generated.side = args.side;
        generated.dimension = 2;
        generated.shape = idmesh_bench::Shape::mixed;
        const auto prefix = (std::filesystem::temp_directory_path()
            / ("idmesh-mpi-benchmark-" + std::to_string(context.size())))
                                .string();
        std::uint64_t global_cells = 0;
        if (context.rank() == 0) {
            const auto source = std::move(idmesh_bench::make_input(generated)).finalize(
                {.topology = idmesh::TopologyLayout::cell_facets,
                    .relations = idmesh::RelationDirection::bidirectional});
            global_cells = source.entity_count(source.dimension());
            const auto owners = idmesh::partition_graph(source, context.size());
            idmesh::write_partition_shards(
                source, owners, args.ghost_layers, prefix);
        }
        MPI_Bcast(&global_cells, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        const auto partition = idmesh::mpi::read_partition_shard(
            context, prefix + "." + std::to_string(context.rank()) + ".idm",
            {.topology = idmesh::TopologyLayout::cell_facets,
                .relations = idmesh::RelationDirection::bidirectional});
        const auto dimension = partition.mesh().dimension();
        const auto& exchange = partition.exchange_plan(dimension);

        const auto local_cells = exchange.local_entity_count();
        idmesh::kokkos::DeviceField field(local_cells, args.components);
        std::vector<double> host(
            static_cast<std::size_t>(local_cells) * args.components, -1.0);
        for (Index entity : partition.owned_entities(dimension))
            for (Index component = 0; component < args.components; ++component)
                host[static_cast<std::size_t>(entity) * args.components + component]
                    = static_cast<double>(context.rank())
                    + 0.001 * static_cast<double>(entity)
                    + 0.000001 * static_cast<double>(component);
        field.upload(host);

        idmesh::mpi::kokkos::DeviceHaloPlan halo(exchange, args.components);
        halo.start(field, idmesh::mpi::kokkos::DeviceTransport::host_staging);
        halo.wait();
        MPI_Barrier(MPI_COMM_WORLD);

        Timing start_timing, wait_timing, total_timing;
        for (unsigned iteration = 0; iteration < args.iterations; ++iteration) {
            MPI_Barrier(MPI_COMM_WORLD);
            const auto total_begin = Clock::now();
            auto begin = total_begin;
            halo.start(field, idmesh::mpi::kokkos::DeviceTransport::host_staging);
            auto end = Clock::now();
            start_timing.samples.push_back(milliseconds(begin, end));

            begin = end;
            halo.wait();
            end = Clock::now();
            wait_timing.samples.push_back(milliseconds(begin, end));
            total_timing.samples.push_back(milliseconds(total_begin, end));
        }

        double start_local = start_timing.median();
        double wait_local = wait_timing.median();
        double total_local = total_timing.median();
        double start_slowest = 0.0, wait_slowest = 0.0, total_slowest = 0.0;
        MPI_Reduce(&start_local, &start_slowest, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&wait_local, &wait_slowest, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&total_local, &total_slowest, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        std::size_t send_records = 0, receive_records = 0;
        for (int peer = 0; peer < exchange.rank_count(); ++peer) {
            send_records += exchange.send_indices(peer).size();
            receive_records += exchange.receive_indices(peer).size();
        }
        std::uint64_t send_local
            = send_records * args.components * sizeof(double);
        std::uint64_t receive_local
            = receive_records * args.components * sizeof(double);
        std::uint64_t send_total = 0, receive_total = 0;
        MPI_Reduce(&send_local, &send_total, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&receive_local, &receive_total, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

        if (context.rank() == 0) {
            std::cout << std::fixed << std::setprecision(6)
                      << "ranks,side,cells,components,ghost_layers,iterations,"
                         "host_staging_start_ms_p50_max,"
                         "host_staging_wait_ms_p50_max,"
                         "host_staging_total_ms_p50_max,"
                         "send_bytes_per_exchange,receive_bytes_per_exchange\n"
                      << context.size() << ',' << args.side << ','
                      << global_cells << ','
                      << args.components << ',' << args.ghost_layers << ','
                      << args.iterations << ',' << start_slowest << ','
                      << wait_slowest << ',' << total_slowest << ','
                      << send_total << ',' << receive_total << '\n';
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (context.rank() == 0)
            for (int peer = 0; peer < context.size(); ++peer)
                std::filesystem::remove(
                    prefix + "." + std::to_string(peer) + ".idm");
        MPI_Barrier(MPI_COMM_WORLD);
        status = 0;
    }
    catch (const std::exception& error) {
        int rank = -1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::cerr << "rank " << rank << ": " << error.what() << '\n';
    }

    Kokkos::finalize();
    if (status != 0)
        MPI_Abort(MPI_COMM_WORLD, status);
    MPI_Finalize();
    return status;
}
