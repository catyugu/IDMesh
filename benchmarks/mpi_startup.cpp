#include "idmesh/mpi.hpp"
#include "idmesh/partition.hpp"
#include "mesh_factory.hpp"
#include "mpi_shard_profile.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <mpi.h>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#elif defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

namespace {

    using Clock = std::chrono::steady_clock;
    using Index = idmesh::Index;

    struct Arguments {
        bool prepare = false;
        std::string prefix;
        unsigned parts = 2;
        unsigned side = 24;
        Index ghost_layers = 1;
        idmesh::TopologyLayout layout = idmesh::TopologyLayout::complete;
        std::string layout_name = "complete";
    };

    unsigned parse_unsigned(const char* text)
    {
        std::size_t consumed = 0;
        const auto value = std::stoull(text, &consumed);
        if (consumed != std::string(text).size()
            || value > std::numeric_limits<unsigned>::max())
            throw std::invalid_argument("invalid unsigned integer");
        return static_cast<unsigned>(value);
    }

    Arguments parse_arguments(int argc, char** argv)
    {
        Arguments args;
        for (int i = 1; i < argc; ++i) {
            const std::string_view token = argv[i];
            if (token == "--help") {
                std::cout
                    << "Prepare shards:\n"
                       "  idmesh_mpi_startup_benchmark --prepare PREFIX --parts N "
                       "[--side N] [--ghost-layers N]\n"
                       "Read benchmark:\n"
                       "  mpiexec -n N idmesh_mpi_startup_benchmark --prefix PREFIX "
                       "--layout cell_vertices|cell_facets|complete\n";
                std::exit(0);
            }
            if (token == "--prepare") {
                if (++i >= argc)
                    throw std::invalid_argument("missing --prepare prefix");
                args.prepare = true;
                args.prefix = argv[i];
            }
            else if (token == "--prefix") {
                if (++i >= argc)
                    throw std::invalid_argument("missing --prefix value");
                args.prefix = argv[i];
            }
            else if (token == "--parts") {
                if (++i >= argc)
                    throw std::invalid_argument("missing --parts value");
                args.parts = parse_unsigned(argv[i]);
            }
            else if (token == "--side") {
                if (++i >= argc)
                    throw std::invalid_argument("missing --side value");
                args.side = parse_unsigned(argv[i]);
            }
            else if (token == "--ghost-layers") {
                if (++i >= argc)
                    throw std::invalid_argument("missing --ghost-layers value");
                args.ghost_layers = parse_unsigned(argv[i]);
            }
            else if (token == "--layout") {
                if (++i >= argc)
                    throw std::invalid_argument("missing --layout value");
                args.layout_name = argv[i];
                if (args.layout_name == "cell_vertices")
                    args.layout = idmesh::TopologyLayout::cell_vertices;
                else if (args.layout_name == "cell_facets")
                    args.layout = idmesh::TopologyLayout::cell_facets;
                else if (args.layout_name == "complete")
                    args.layout = idmesh::TopologyLayout::complete;
                else
                    throw std::invalid_argument("invalid --layout");
            }
            else {
                throw std::invalid_argument(
                    "unknown startup benchmark option: " + std::string(token));
            }
        }
        if (args.prefix.empty() || args.parts == 0 || args.side == 0
            || args.ghost_layers > 8)
            throw std::invalid_argument("invalid startup benchmark arguments");
        return args;
    }

    std::uint64_t peak_resident_bytes()
    {
#if defined(_WIN32)
        PROCESS_MEMORY_COUNTERS info {};
        if (!GetProcessMemoryInfo(
                GetCurrentProcess(), &info, static_cast<DWORD>(sizeof(info))))
            return 0;
        return static_cast<std::uint64_t>(info.PeakWorkingSetSize);
#elif defined(__unix__) || defined(__APPLE__)
        rusage usage {};
        if (getrusage(RUSAGE_SELF, &usage) != 0)
            return 0;
#if defined(__APPLE__)
        return static_cast<std::uint64_t>(usage.ru_maxrss);
#else
        return static_cast<std::uint64_t>(usage.ru_maxrss) * 1024ULL;
#endif
#else
        return 0;
#endif
    }

    double milliseconds(Clock::time_point begin, Clock::time_point end)
    {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    }

    int prepare(const Arguments& args)
    {
        idmesh_bench::GeneratedMesh generated;
        generated.side = args.side;
        generated.dimension = 3;
        generated.shape = idmesh_bench::Shape::mixed;
        const auto mesh = std::move(idmesh_bench::make_input(generated)).finalize(
            {.topology = idmesh::TopologyLayout::cell_facets,
                .relations = idmesh::RelationDirection::bidirectional});
        const auto owners = idmesh::partition_graph(mesh, static_cast<int>(args.parts));
        idmesh::write_partition_shards(
            mesh, owners, args.ghost_layers, args.prefix);
        std::cout << "prepared," << args.parts << ',' << args.side << ','
                  << mesh.entity_count(mesh.dimension()) << '\n';
        return 0;
    }

    template <class T>
    T reduce_max(T value, MPI_Datatype type)
    {
        T result {};
        MPI_Reduce(&value, &result, 1, type, MPI_MAX, 0, MPI_COMM_WORLD);
        return result;
    }

} // namespace

int main(int argc, char** argv)
{
    try {
        const auto args = parse_arguments(argc, argv);
        if (args.prepare)
            return prepare(args);

        MPI_Init(&argc, &argv);
        int status = 0;
        try {
            const auto context = idmesh::mpi::Context::world();
            if (static_cast<unsigned>(context.size()) != args.parts)
                throw std::invalid_argument(
                    "--parts must match the MPI runtime size");

            MPI_Barrier(MPI_COMM_WORLD);
            const auto peak_before = peak_resident_bytes();
            idmesh::mpi::detail::ShardReadProfile profile;
            const auto begin = Clock::now();
            const auto partition = idmesh::mpi::detail::read_partition_shard_profiled(
                context,
                args.prefix + "." + std::to_string(context.rank()) + ".idm",
                {.topology = args.layout,
                    .relations = idmesh::RelationDirection::bidirectional},
                profile);
            const auto end = Clock::now();
            const auto peak_after = peak_resident_bytes();

            const auto total_ms
                = reduce_max(milliseconds(begin, end), MPI_DOUBLE);
            const auto parse_ms = reduce_max(profile.parse_ms, MPI_DOUBLE);
            const auto mesh_ms = reduce_max(profile.local_mesh_ms, MPI_DOUBLE);
            const auto metadata_ms
                = reduce_max(profile.metadata_ms, MPI_DOUBLE);
            const auto plan_ms
                = reduce_max(profile.exchange_plan_ms, MPI_DOUBLE);
            const auto validate_ms
                = reduce_max(profile.validate_ms, MPI_DOUBLE);

            std::uint64_t peak_delta
                = peak_after > peak_before ? peak_after - peak_before : 0;
            peak_delta = reduce_max(peak_delta, MPI_UINT64_T);

            const Index dim = partition.mesh().dimension();
            const std::uint64_t cells
                = partition.mesh().entity_count(dim);
            const std::uint64_t facets
                = dim > 0 && partition.mesh().has_entities(dim - 1)
                ? partition.mesh().entity_count(dim - 1)
                : 0;
            const std::uint64_t edges
                = partition.mesh().has_entities(1)
                ? partition.mesh().entity_count(1)
                : 0;
            const auto max_cells = reduce_max(cells, MPI_UINT64_T);
            const auto max_facets = reduce_max(facets, MPI_UINT64_T);
            const auto max_edges = reduce_max(edges, MPI_UINT64_T);

            if (context.rank() == 0) {
                std::cout << std::fixed << std::setprecision(3)
                          << "ranks,layout,local_cells_max,local_facets_max,"
                             "local_edges_max,total_ms_max,parse_ms_max,"
                             "local_mesh_ms_max,metadata_ms_max,"
                             "exchange_plan_ms_max,validate_ms_max,"
                             "peak_rss_delta_bytes_max\n"
                          << context.size() << ',' << args.layout_name << ','
                          << max_cells << ',' << max_facets << ','
                          << max_edges << ',' << total_ms << ',' << parse_ms
                          << ',' << mesh_ms << ',' << metadata_ms << ','
                          << plan_ms << ',' << validate_ms << ','
                          << peak_delta << '\n';
            }
        }
        catch (const std::exception& error) {
            int rank = -1;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            std::cerr << "rank " << rank << ": " << error.what() << '\n';
            status = 1;
        }
        if (status)
            MPI_Abort(MPI_COMM_WORLD, status);
        MPI_Finalize();
        return status;
    }
    catch (const std::exception& error) {
        std::cerr << "startup benchmark error: " << error.what() << '\n';
        return 1;
    }
}
