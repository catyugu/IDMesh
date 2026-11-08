#include "idmesh/mpi.hpp"
#include "idmesh/partition.hpp"
#include "mesh_factory.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>

namespace {
    using Clock = std::chrono::steady_clock;
    using Index = idmesh::Index;

    struct Arguments {
        unsigned side = 96;
        unsigned iterations = 50;
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
                if (i == 1)
                    std::cout << "Usage: idmesh_mpi_benchmark [--side N] [--iterations N] "
                                 "[--components N] [--ghost-layers N]\n";
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

    double milliseconds(Clock::time_point begin, Clock::time_point end)
    {
        return std::chrono::duration<double, std::milli>(end - begin).count();
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

    void fill_field(std::vector<double>& values, int rank, Index components)
    {
        for (std::size_t cell = 0; cell < values.size() / components; ++cell)
            for (Index c = 0; c < components; ++c)
                values[cell * components + c]
                    = static_cast<double>(rank) + 0.001 * static_cast<double>(cell)
                    + 0.000001 * static_cast<double>(c);
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
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
        const auto& plan = partition.exchange_plan(dimension);

        std::vector<double> field(
            static_cast<std::size_t>(plan.local_entity_count()) * args.components);
        fill_field(field, context.rank(), args.components);

        std::size_t send_records = 0, receive_records = 0;
        for (int peer = 0; peer < plan.rank_count(); ++peer) {
            send_records += plan.send_indices(peer).size();
            receive_records += plan.receive_indices(peer).size();
        }

        const auto setup_begin = Clock::now();
        auto reusable
            = plan.make_request(std::size_t(args.components) * sizeof(double));
        const auto setup_ms = milliseconds(setup_begin, Clock::now());
        reusable.start(std::as_writable_bytes(std::span(field)));
        reusable.wait();
        MPI_Barrier(MPI_COMM_WORLD);

        Timing start_phase, wait_phase, reused;
        for (unsigned iteration = 0; iteration < args.iterations; ++iteration) {
            fill_field(field, context.rank(), args.components);
            MPI_Barrier(MPI_COMM_WORLD);
            const auto total_begin = Clock::now();
            auto begin = total_begin;
            reusable.start(std::as_writable_bytes(std::span(field)));
            auto end = Clock::now();
            start_phase.samples.push_back(milliseconds(begin, end));
            begin = end;
            reusable.wait();
            end = Clock::now();
            wait_phase.samples.push_back(milliseconds(begin, end));
            reused.samples.push_back(milliseconds(total_begin, end));
        }

        const double setup_local = setup_ms;
        double setup_max = 0.0;
        MPI_Reduce(&setup_local, &setup_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        double start_local = start_phase.median();
        double wait_local = wait_phase.median();
        double reused_local = reused.median();
        double start_max = 0.0, wait_max = 0.0, reused_max = 0.0;
        MPI_Reduce(&start_local, &start_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&wait_local, &wait_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&reused_local, &reused_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        std::uint64_t sent_local = send_records * args.components * sizeof(double);
        std::uint64_t recv_local = receive_records * args.components * sizeof(double);
        std::uint64_t sent_total = 0, recv_total = 0;
        MPI_Reduce(&sent_local, &sent_total, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&recv_local, &recv_total, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

        if (context.rank() == 0) {
            std::cout << std::fixed << std::setprecision(6)
                      << "ranks,side,cells,components,ghost_layers,iterations,"
                         "reused_start_ms_p50_max,"
                         "reused_wait_ms_p50_max,reused_total_ms_p50_max,"
                         "send_bytes_per_exchange,receive_bytes_per_exchange,"
                         "request_setup_ms_max\n"
                      << context.size() << ',' << args.side << ','
                      << global_cells << ','
                      << args.components << ',' << args.ghost_layers << ','
                      << args.iterations << ',' << start_max << ','
                      << wait_max << ',' << reused_max << ','
                      << sent_total << ',' << recv_total << ','
                      << setup_max << '\n';
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
    MPI_Finalize();
    return status;
}
