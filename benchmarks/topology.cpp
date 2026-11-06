#include "idmesh/mesh.hpp"
#include "mesh_factory.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <psapi.h>
#include <windows.h>
#elif defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#if defined(__linux__)
#include <unistd.h>
#endif
#endif

namespace {
    using Clock = std::chrono::steady_clock;
    using Index = idmesh::Index;

    struct Arguments {
        idmesh_bench::GeneratedMesh generated;
        unsigned rounds = 20;
        unsigned trials = 5;
        std::string gmsh;
    };

    unsigned parse_unsigned(const char* value)
    {
        const std::string text(value);
        std::size_t consumed = 0;
        const auto result = std::stoull(text, &consumed);
        if (consumed != text.size() || result > std::numeric_limits<unsigned>::max())
            throw std::invalid_argument("invalid unsigned integer: " + text);
        return static_cast<unsigned>(result);
    }

    std::uint64_t parse_seed(const char* value)
    {
        const std::string text(value);
        std::size_t consumed = 0;
        const auto result = std::stoull(text, &consumed);
        if (consumed != text.size())
            throw std::invalid_argument("invalid seed");
        return result;
    }

    Arguments parse_arguments(int argc, char** argv)
    {
        Arguments args;
        unsigned positional = 0;
        for (int i = 1; i < argc; ++i) {
            const std::string_view token = argv[i];
            if (token == "--help") {
                std::cout << "Usage: idmesh_benchmark [side] [rounds] [--dim 2|3] "
                             "[--shape simplex|tensor|mixed] [--trials N] "
                             "[--scramble] [--seed N] [--jitter D] [--gmsh FILE]\n"
                             "Output: CSV, including mesh payload and process memory.\n";
                std::exit(0);
            }
            if (token == "--scramble") {
                args.generated.scramble = true;
                continue;
            }
            if (token == "--dim" || token == "--shape" || token == "--trials"
                || token == "--seed" || token == "--jitter" || token == "--gmsh") {
                if (++i >= argc)
                    throw std::invalid_argument("missing benchmark option value");
                if (token == "--dim")
                    args.generated.dimension = parse_unsigned(argv[i]);
                else if (token == "--shape")
                    args.generated.shape = idmesh_bench::parse_shape(argv[i]);
                else if (token == "--trials")
                    args.trials = parse_unsigned(argv[i]);
                else if (token == "--seed")
                    args.generated.seed = parse_seed(argv[i]);
                else if (token == "--gmsh")
                    args.gmsh = argv[i];
                else {
                    std::size_t consumed = 0;
                    args.generated.jitter = std::stod(argv[i], &consumed);
                    if (consumed != std::string_view(argv[i]).size())
                        throw std::invalid_argument("invalid jitter");
                }
            }
            else if (token.starts_with('-')) {
                throw std::invalid_argument("unknown benchmark option");
            }
            else if (positional == 0) {
                args.generated.side = parse_unsigned(argv[i]);
                ++positional;
            }
            else if (positional == 1) {
                args.rounds = parse_unsigned(argv[i]);
                ++positional;
            }
            else {
                throw std::invalid_argument("unexpected positional benchmark argument");
            }
        }
        const auto& g = args.generated;
        if ((g.dimension != 2 && g.dimension != 3) || args.trials == 0
            || args.trials > 100 || args.rounds == 0 || g.side == 0
            || g.side > (g.dimension == 2 ? 1024u : 64u)
            || !(g.jitter >= 0.0 && g.jitter <= 0.2)) {
            throw std::invalid_argument("invalid dimension, side, rounds, trials or jitter");
        }
        return args;
    }

    double milliseconds(Clock::time_point start, Clock::time_point end)
    {
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    struct Timing {
        std::vector<double> samples;

        double percentile(std::size_t numerator, std::size_t denominator) const
        {
            if (samples.empty())
                return 0.0;
            auto sorted = samples;
            std::sort(sorted.begin(), sorted.end());
            const auto rank = (sorted.size() * numerator + denominator - 1) / denominator;
            return sorted[std::max<std::size_t>(1, rank) - 1];
        }
        double median() const { return percentile(50, 100); }
        double p95() const { return percentile(95, 100); }
    };

    // RSS is process-wide and includes allocator retained pages. Peak RSS is
    // lifetime-wide, not a per-trial heap peak; zero indicates unavailable.
    std::uint64_t resident_bytes()
    {
#if defined(_WIN32)
        PROCESS_MEMORY_COUNTERS counters {};
        if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters)))
            return counters.WorkingSetSize;
#elif defined(__linux__)
        std::ifstream file("/proc/self/statm");
        std::uint64_t pages = 0, resident = 0;
        if (file >> pages >> resident)
            return resident * static_cast<std::uint64_t>(sysconf(_SC_PAGESIZE));
#endif
        return 0;
    }

    std::uint64_t peak_resident_bytes()
    {
#if defined(_WIN32)
        PROCESS_MEMORY_COUNTERS counters {};
        if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters)))
            return counters.PeakWorkingSetSize;
#elif defined(__unix__) || defined(__APPLE__)
        rusage usage {};
        if (getrusage(RUSAGE_SELF, &usage) == 0) {
#if defined(__APPLE__)
            return static_cast<std::uint64_t>(usage.ru_maxrss);
#else
            return static_cast<std::uint64_t>(usage.ru_maxrss) * 1024;
#endif
        }
#endif
        return 0;
    }

    std::size_t relation_bytes(const idmesh::Mesh& mesh)
    {
        std::size_t bytes = 0;
        for (Index from = 0; from <= mesh.dimension(); ++from) {
            for (Index to = 0; to <= mesh.dimension(); ++to) {
                if (!mesh.has_relation(from, to))
                    continue;
                const auto relation = mesh.relation(from, to);
                bytes += relation.offsets.size_bytes() + relation.indices.size_bytes()
                    + relation.signs.size_bytes() + relation.orientations.size_bytes();
            }
        }
        return bytes;
    }

    // Visible logical payload lower bound; excludes vector spare capacity,
    // allocator metadata, maps, shared_ptr control blocks and temporary input.
    std::size_t visible_payload_bytes(const idmesh::Mesh& mesh)
    {
        std::size_t bytes = relation_bytes(mesh);
        for (Index dimension = 0; dimension <= mesh.dimension(); ++dimension) {
            if (!mesh.has_entities(dimension))
                continue;
            const Index count = mesh.entity_count(dimension);
            bytes += static_cast<std::size_t>(count + 1) * sizeof(Index);
            for (Index id = 0; id < count; ++id)
                bytes += mesh.entity_vertices(dimension, id).size_bytes();
        }
        bytes += mesh.coordinates().size_bytes();
        bytes += mesh.cell_types().size_bytes();
        bytes += mesh.cell_tags().size_bytes();
        bytes += mesh.facet_tags().size_bytes();
        const auto labels = mesh.physical_labels();
        bytes += labels.size_bytes();
        for (const auto& label : labels)
            bytes += label.vertices.size() * sizeof(Index);

        const auto extra_nodes = mesh.geometry_node_count() - mesh.coordinates().size();
        bytes += static_cast<std::size_t>(extra_nodes) * sizeof(std::array<double, 3>);
        bool has_quadratic = false;
        for (Index cell = 0; cell < mesh.entity_count(mesh.dimension()); ++cell) {
            if (mesh.cell_order(cell) != 1) {
                has_quadratic = true;
                bytes += mesh.cell_geometry_nodes(cell).size_bytes();
            }
        }
        if (has_quadratic)
            bytes += (static_cast<std::size_t>(mesh.entity_count(mesh.dimension())) + 1) * sizeof(Index);
        return bytes;
    }

    struct Traversal {
        Timing time;
        std::uint64_t checksum = 0;
        std::uint64_t incidences = 0;
        bool available = false;
    };

    struct Walk {
        const char* name;
        Index from;
        Index to;
        bool oriented = false;
    };

    struct WalkResult {
        double elapsed_ms = 0;
        std::uint64_t checksum = 0;
        std::uint64_t incidences = 0;
    };

    WalkResult traverse(const idmesh::Mesh& mesh, const Walk& walk, unsigned rounds)
    {
        const auto relation = mesh.relation(walk.from, walk.to);
        std::uint64_t checksum = 0;
        const auto begin = Clock::now();
        for (unsigned trial = 0; trial < rounds; ++trial) {
            for (Index row = 0; row < relation.rows; ++row) {
                const auto indices = relation.row(row);
                const auto signs = walk.oriented ? relation.row_signs(row) : std::span<const std::int8_t> {};
                const auto orientations = walk.oriented ? relation.row_orientations(row) : std::span<const idmesh::Orientation> {};
                for (std::size_t i = 0; i < indices.size(); ++i) {
                    const std::uint64_t id = static_cast<std::uint64_t>(indices[i]) + 1;
                    if (walk.oriented) {
                        checksum += id * (static_cast<std::uint64_t>(orientations[i].rotation) + 1)
                            + (signs[i] < 0 ? 13u : 7u)
                            + (orientations[i].reversed ? 17u : 0u);
                    }
                    else {
                        checksum += id;
                    }
                }
            }
        }
        const auto end = Clock::now();
        return {milliseconds(begin, end), checksum, relation.indices.size()};
    }

    void run(const Arguments& args, idmesh::TopologyOptions options, const char* configuration)
    {
        Timing input, finalize, resident;
        std::vector<Walk> walks;
        std::vector<Traversal> samples;
        Index dimension = args.generated.dimension;
        const auto initialize_walks = [&](Index dim) {
            dimension = dim;
            walks = {
                {"cell_vertex", dim, 0},
                {"cell_facet", dim, dim - 1},
                {"facet_cell", dim - 1, dim},
                {"facet_vertex", dim - 1, 0},
                {"cell_facet_oriented", dim, dim - 1, true}};
            samples.resize(walks.size());
        };
        Index cell_count = 0, facet_count = 0, edge_count = 0;
        std::size_t relation_payload = 0, total_payload = 0;
        std::uint64_t peak = 0;
        for (unsigned trial = 0; trial < args.trials; ++trial) {
            const auto start = Clock::now();
            idmesh::Mesh mesh;
            Clock::time_point input_ready;
            if (args.gmsh.empty()) {
                auto builder = idmesh_bench::make_input(args.generated);
                input_ready = Clock::now();
                mesh = std::move(builder).finalize(options);
            }
            else {
                input_ready = start;
                mesh = idmesh::read_gmsh(args.gmsh, options);
            }
            const auto finalized = Clock::now();
            input.samples.push_back(milliseconds(start, input_ready));
            finalize.samples.push_back(milliseconds(input_ready, finalized));
            resident.samples.push_back(static_cast<double>(resident_bytes()));
            peak = std::max(peak, peak_resident_bytes());

            const Index dim = mesh.dimension();
            if (trial == 0) {
                initialize_walks(dim);
                cell_count = mesh.entity_count(dim);
                facet_count = mesh.has_entities(dim - 1) ? mesh.entity_count(dim - 1) : 0;
                edge_count = mesh.has_entities(1) ? mesh.entity_count(1) : 0;
                relation_payload = relation_bytes(mesh);
                total_payload = visible_payload_bytes(mesh);
            }
            else if (dimension != dim || cell_count != mesh.entity_count(dim)
                || relation_payload != relation_bytes(mesh)
                || total_payload != visible_payload_bytes(mesh)) {
                throw std::runtime_error("mesh payload changed across benchmark trials");
            }
            for (std::size_t j = 0; j < walks.size(); ++j) {
                auto walk = walks[j];
                if (walk.from > dim || !mesh.has_relation(walk.from, walk.to))
                    continue;
                // Vertex-to-vertex is not a materialized facet relation in 1D.
                const auto result = traverse(mesh, walk, args.rounds);
                auto& measurement = samples[j];
                if (trial > 0 && measurement.checksum != result.checksum)
                    throw std::runtime_error("topology checksum changed across trials");
                measurement.available = true;
                measurement.checksum = result.checksum;
                measurement.incidences = result.incidences;
                measurement.time.samples.push_back(result.elapsed_ms);
            }
        }
        const char* source = args.gmsh.empty() ? "generated" : "gmsh";
        for (std::size_t j = 0; j < walks.size(); ++j) {
            if (!samples[j].available)
                continue;
            const auto& walk = samples[j];
            std::cout << dimension << ',' << idmesh_bench::shape_name(args.generated.shape) << ','
                      << (args.gmsh.empty() ? args.generated.side : 0) << ','
                      << configuration << ',' << walks[j].name << ',' << cell_count << ','
                      << facet_count << ',' << edge_count << ',' << args.trials << ',' << args.rounds
                      << ',' << input.median() << ',' << finalize.median() << ',' << finalize.p95()
                      << ',' << walk.time.median() << ',' << walk.time.p95()
                      << ',' << relation_payload << ','
                      << (cell_count ? double(relation_payload) / cell_count : 0.)
                      << ',' << walk.checksum << ',' << source << ','
                      << args.generated.scramble << ',' << args.generated.seed << ','
                      << args.generated.jitter << ',' << total_payload << ','
                      << static_cast<std::uint64_t>(resident.median()) << ',' << peak << ','
                      << walk.incidences << '\n';
        }
    }
}

int main(int argc, char** argv)
{
    try {
        const auto args = parse_arguments(argc, argv);
        std::cout << std::fixed << std::setprecision(3)
                  << "dimension,shape,side,configuration,relation,cells,facets,edges,trials,rounds,"
                     "input_ms_p50,finalize_ms_p50,finalize_ms_p95,traversal_ms_p50,"
                     "traversal_ms_p95,relation_bytes,relation_bytes_per_cell,checksum,"
                     "source,scramble,seed,jitter,visible_payload_bytes,"
                     "process_rss_bytes_p50,process_peak_rss_bytes,incidences_per_round\n";
        // Gmsh input has a single combined read+finalize operation in finalize_ms.
        run(args, idmesh::TopologyOptions::cells_and_vertices(), "minimal");
        run(args, idmesh::TopologyOptions::finite_volume(), "finite_volume");
        run(args, idmesh::TopologyOptions::full(), "full");
    }
    catch (const std::exception& error) {
        std::cerr << "benchmark error: " << error.what() << '\n';
        return 1;
    }
}
