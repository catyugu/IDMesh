#include "idmesh/mpi.hpp"
#include "mesh_factory.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
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
        const auto source = std::move(idmesh_bench::make_input(generated)).finalize(
            {.topology = idmesh::TopologyLayout::cell_facets,
                .relations = idmesh::RelationDirection::bidirectional});
        const auto partition
            = idmesh::mpi::partition_replicated(context, source, args.ghost_layers);
        const auto& plan = partition.exchange_plan();

        std::vector<double> field(
            static_cast<std::size_t>(plan.local_cell_count()) * args.components);
        fill_field(field, context.rank(), args.components);

        std::vector<double> packed_send;
        std::vector<double> packed_receive;
        std::size_t send_records = 0, receive_records = 0;
        for (int peer = 0; peer < plan.rank_count(); ++peer) {
            send_records += plan.send_indices(peer).size();
            receive_records += plan.receive_indices(peer).size();
        }
        packed_send.resize(send_records * args.components);
        packed_receive.resize(receive_records * args.components);

        const auto pack = [&] {
            std::size_t output = 0;
            for (int peer = 0; peer < plan.rank_count(); ++peer)
                for (Index cell : plan.send_indices(peer))
                    for (Index c = 0; c < args.components; ++c)
                        packed_send[output++]
                            = field[static_cast<std::size_t>(cell) * args.components + c];
        };

        auto reusable = plan.make_request(args.components);
        plan.exchange(field, args.components);
        pack();
        plan.exchange_packed(packed_send.data(), packed_receive.data(),
            args.components, idmesh::mpi::DeviceMemoryKind::host);
        reusable.start(field);
        reusable.wait();
        MPI_Barrier(MPI_COMM_WORLD);

        Timing exchange, packed, one_shot, reused;
        for (unsigned iteration = 0; iteration < args.iterations; ++iteration) {
            fill_field(field, context.rank(), args.components);
            MPI_Barrier(MPI_COMM_WORLD);
            auto begin = Clock::now();
            plan.exchange(field, args.components);
            auto end = Clock::now();
            exchange.samples.push_back(milliseconds(begin, end));

            pack();
            MPI_Barrier(MPI_COMM_WORLD);
            begin = Clock::now();
            plan.exchange_packed(packed_send.data(), packed_receive.data(),
                args.components, idmesh::mpi::DeviceMemoryKind::host);
            end = Clock::now();
            packed.samples.push_back(milliseconds(begin, end));

            fill_field(field, context.rank(), args.components);
            MPI_Barrier(MPI_COMM_WORLD);
            begin = Clock::now();
            auto temporary = plan.begin_exchange(field, args.components);
            temporary.wait();
            end = Clock::now();
            one_shot.samples.push_back(milliseconds(begin, end));

            fill_field(field, context.rank(), args.components);
            MPI_Barrier(MPI_COMM_WORLD);
            begin = Clock::now();
            reusable.start(field);
            reusable.wait();
            end = Clock::now();
            reused.samples.push_back(milliseconds(begin, end));
        }

        double exchange_local = exchange.median();
        double packed_local = packed.median();
        double one_shot_local = one_shot.median();
        double reused_local = reused.median();
        double exchange_max = 0.0, packed_max = 0.0;
        double one_shot_max = 0.0, reused_max = 0.0;
        MPI_Reduce(&exchange_local, &exchange_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&packed_local, &packed_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&one_shot_local, &one_shot_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&reused_local, &reused_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        std::uint64_t sent_local = send_records * args.components * sizeof(double);
        std::uint64_t recv_local = receive_records * args.components * sizeof(double);
        std::uint64_t sent_total = 0, recv_total = 0;
        MPI_Reduce(&sent_local, &sent_total, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&recv_local, &recv_total, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

        if (context.rank() == 0) {
            std::cout << std::fixed << std::setprecision(6)
                      << "ranks,side,cells,components,ghost_layers,iterations,"
                         "exchange_ms_p50_max,packed_mpi_ms_p50_max,"
                         "oneshot_nonblocking_ms_p50_max,reused_nonblocking_ms_p50_max,"
                         "send_bytes_per_exchange,receive_bytes_per_exchange\n"
                      << context.size() << ',' << args.side << ','
                      << source.entity_count(source.dimension()) << ','
                      << args.components << ',' << args.ghost_layers << ','
                      << args.iterations << ',' << exchange_max << ',' << packed_max << ','
                      << one_shot_max << ',' << reused_max << ','
                      << sent_total << ',' << recv_total << '\n';
        }
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
