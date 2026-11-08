#include "idmesh/io.hpp"
#include "idmesh/kokkos.hpp"
#include "mesh_factory.hpp"

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#if defined(__linux__)
#include <unistd.h>
#endif

namespace {
    using Clock = std::chrono::steady_clock;

    struct Timing {
        std::vector<double> samples;
        double percentile(unsigned n) const
        {
            auto sorted = samples;
            std::sort(sorted.begin(), sorted.end());
            return sorted[(sorted.size() * n + 99) / 100 - 1];
        }
        double median() const { return percentile(50); }
        double p95() const { return percentile(95); }
    };

    struct Arguments {
        idmesh_bench::GeneratedMesh generated;
        unsigned trials = 5;
        unsigned rounds = 20;
        std::string gmsh;
        std::string configuration = "all";
    };

    unsigned parse_unsigned(const char* value)
    {
        const std::string text(value);
        std::size_t consumed = 0;
        const auto result = std::stoull(text, &consumed);
        if (consumed != text.size() || result > 10000000)
            throw std::invalid_argument("invalid integer option");
        return static_cast<unsigned>(result);
    }

    Arguments parse_arguments(int argc, char** argv)
    {
        Arguments a;
        unsigned positional = 0;
        for (int i = 1; i < argc; ++i) {
            const std::string_view option = argv[i];
            if (option == "--help") {
                std::cout << "Usage: idmesh_kokkos_benchmark [side] [rounds] [--dim 2|3] "
                             "[--shape simplex|tensor|mixed] [--trials N] "
                             "[--scramble] [--seed N] [--jitter D] [--gmsh FILE] "
                             "[--config all|minimal|finite_volume|full]\n";
                std::exit(0);
            }
            if (option == "--scramble") {
                a.generated.scramble = true;
                continue;
            }
            if (option == "--shape" || option == "--dim" || option == "--trials"
                || option == "--gmsh" || option == "--seed" || option == "--jitter"
                || option == "--config") {
                if (++i >= argc)
                    throw std::invalid_argument("missing option value");
                if (option == "--shape")
                    a.generated.shape = idmesh_bench::parse_shape(argv[i]);
                else if (option == "--dim")
                    a.generated.dimension = parse_unsigned(argv[i]);
                else if (option == "--trials")
                    a.trials = parse_unsigned(argv[i]);
                else if (option == "--gmsh")
                    a.gmsh = argv[i];
                else if (option == "--config")
                    a.configuration = argv[i];
                else if (option == "--seed")
                    a.generated.seed = std::stoull(argv[i]);
                else
                    a.generated.jitter = std::stod(argv[i]);
            }
            else if (option.starts_with('-')) {
                throw std::invalid_argument("unknown option");
            }
            else if (positional++ == 0)
                a.generated.side = parse_unsigned(argv[i]);
            else if (positional == 2)
                a.rounds = parse_unsigned(argv[i]);
            else
                throw std::invalid_argument("too many positional arguments");
        }
        if (a.configuration != "all" && a.configuration != "minimal"
            && a.configuration != "finite_volume" && a.configuration != "full")
            throw std::invalid_argument("invalid --config");
        if ((a.generated.dimension != 2 && a.generated.dimension != 3)
            || a.generated.side == 0
            || a.generated.side > (a.generated.dimension == 2 ? 1024u : 64u)
            || a.trials == 0 || a.trials > 100 || a.rounds == 0
            || !(a.generated.jitter >= 0.0 && a.generated.jitter <= 0.2))
            throw std::invalid_argument("invalid arguments");
        return a;
    }

    double elapsed(Clock::time_point begin)
    {
        return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
    }

    std::uint64_t resident_bytes()
    {
#if defined(__linux__)
        std::ifstream file("/proc/self/statm");
        std::uint64_t pages = 0, resident = 0;
        if (file >> pages >> resident)
            return resident * static_cast<std::uint64_t>(sysconf(_SC_PAGESIZE));
#endif
        return 0; // Process RSS is unavailable on this platform.
    }

    void emit(const Arguments& a, const idmesh::Mesh& mesh, const char* configuration,
        const char* phase, const Timing& values, double checked_value,
        std::uint64_t mesh_bytes, std::uint64_t relation_bytes,
        const Timing& resident, const Timing& increase)
    {
        std::cout << mesh.dimension() << ',' << idmesh_bench::shape_name(a.generated.shape)
                  << ',' << (a.gmsh.empty() ? a.generated.side : 0)
                  << ',' << configuration << ',' << phase << ','
                  << mesh.entity_count(mesh.dimension()) << ','
                  << a.trials << ',' << a.rounds << ',' << values.median()
                  << ',' << values.p95() << ',' << checked_value << ','
                  << (a.gmsh.empty() ? "generated" : "gmsh") << ','
                  << a.generated.scramble << ',' << a.generated.seed << ','
                  << mesh_bytes << ',' << relation_bytes << ','
                  << static_cast<std::uint64_t>(resident.median()) << ','
                  << static_cast<std::uint64_t>(increase.median()) << '\n';
    }

    void run_case(const Arguments& a, idmesh::BuildOptions options, const char* name)
    {
        const auto mesh = [&] {
            if (a.gmsh.empty()) {
                auto builder = idmesh_bench::make_input(a.generated);
                return std::move(builder).finalize(options);
            }
            return idmesh::read_gmsh(a.gmsh, options);
        }();
        Timing geometry_upload, relation_upload, centroid, incidence,
            geometry_checksum, resident, increase;
        std::uint64_t mesh_payload = 0, relation_payload = 0;
        const auto dim = mesh.dimension();
        const bool has_relation = mesh.has_relation(dim, dim - 1);
        const auto reference = has_relation ? mesh.relation(dim, dim - 1).targets.size() : 0;
        double checked_centroid = 0.0;
        std::uint64_t checked_geometry = 0;
        for (unsigned trial = 0; trial < a.trials; ++trial) {
            const auto before_rss = resident_bytes();
            auto start = Clock::now();
            auto device = std::make_unique<idmesh::kokkos::DeviceMesh>(mesh);
            Kokkos::fence();
            geometry_upload.samples.push_back(elapsed(start));

            std::unique_ptr<idmesh::kokkos::DeviceRelation> relation;
            if (has_relation) {
                start = Clock::now();
                relation = std::make_unique<idmesh::kokkos::DeviceRelation>(mesh, dim, dim - 1);
                Kokkos::fence();
                relation_upload.samples.push_back(elapsed(start));
            }
            // Mirror accounting excludes application-owned fields and the host
            // Mesh, which already exists before the RSS snapshot.
            const auto after_rss = resident_bytes();
            resident.samples.push_back(static_cast<double>(after_rss));
            increase.samples.push_back(
                static_cast<double>(after_rss > before_rss ? after_rss - before_rss : 0));
            if (trial == 0) {
                const std::uint64_t cells = device->cell_count();
                const std::uint64_t nodes = device->geometry_node_count();
                mesh_payload = cells * (sizeof(idmesh::CellType) + sizeof(std::uint8_t))
                    + 2 * (cells + 1) * sizeof(idmesh::Index)
                    + (device->cell_vertex_entry_count()
                          + device->high_order_geometry_entry_count()) * sizeof(idmesh::Index)
                    + 3 * nodes * sizeof(double);
                if (relation) {
                    const std::uint64_t entries = relation->entry_count();
                    relation_payload = (std::uint64_t(relation->row_count()) + 1) * sizeof(idmesh::Index)
                        + entries * (sizeof(idmesh::Index)
                            + (relation->oriented()
                                    ? sizeof(std::int8_t) + sizeof(idmesh::Orientation)
                                    : 0));
                }
            }

            start = Clock::now();
            std::array<double, 3> center {};
            for (unsigned round = 0; round < a.rounds; ++round) {
                const auto* offsets = device->cell_vertex_offsets();
                const auto* vertices = device->cell_vertices();
                const auto* x = device->geometry_x();
                const auto* y = device->geometry_y();
                const auto* z = device->geometry_z();
                for (int component = 0; component < 3; ++component) {
                    double sum = 0;
                    Kokkos::parallel_reduce("idmesh_bench_centroid", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, device->cell_count()), KOKKOS_LAMBDA(const idmesh::Index cell, double& partial) {
                                const auto begin = offsets[cell];
                                const auto end = offsets[cell + 1];
                                double value = 0;
                                for (idmesh::Index i = begin; i < end; ++i) {
                                    const auto vertex = vertices[i];
                                    value += component == 0 ? x[vertex]
                                        : component == 1 ? y[vertex] : z[vertex];
                                }
                                partial += value / double(end - begin); }, sum);
                    center[component] = sum;
                }
            }
            Kokkos::fence();
            centroid.samples.push_back(elapsed(start));
            const double scalar = center[0] + center[1] + center[2];
            if (trial && std::abs(scalar - checked_centroid) > 1e-9 * (1 + std::abs(scalar)))
                throw std::runtime_error("device centroid result changed across trials");
            checked_centroid = scalar;

            start = Clock::now();
            std::uint64_t geometry_value = 0;
            for (unsigned round = 0; round < a.rounds; ++round) {
                const auto* orders = device->cell_orders();
                const auto* vertex_offsets = device->cell_vertex_offsets();
                const auto* vertices = device->cell_vertices();
                const auto* geometry_offsets = device->high_order_geometry_offsets();
                const auto* geometry = device->high_order_geometry_nodes();
                Kokkos::parallel_reduce("idmesh_bench_geometry_ids", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, device->cell_count()), KOKKOS_LAMBDA(const idmesh::Index cell, std::uint64_t& sum) {
                            if (orders[cell] == 1) {
                                for (idmesh::Index i = vertex_offsets[cell];
                                     i < vertex_offsets[cell + 1]; ++i)
                                    sum += std::uint64_t(vertices[i]) + 1;
                            }
                            else {
                                for (idmesh::Index i = geometry_offsets[cell];
                                     i < geometry_offsets[cell + 1]; ++i)
                                    sum += std::uint64_t(geometry[i]) + 1;
                            } }, geometry_value);
            }
            Kokkos::fence();
            geometry_checksum.samples.push_back(elapsed(start));
            if (trial && geometry_value != checked_geometry)
                throw std::runtime_error("device geometry checksum changed");
            checked_geometry = geometry_value;

            if (has_relation) {
                start = Clock::now();
                std::uint64_t relation_value = 0;
                for (unsigned round = 0; round < a.rounds; ++round) {
                    const auto* targets = relation->targets();
                    const auto entries = relation->entry_count();
                    Kokkos::parallel_reduce("idmesh_bench_relation_targets", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, static_cast<idmesh::Index>(entries)), KOKKOS_LAMBDA(const idmesh::Index i, std::uint64_t& sum) { sum += std::uint64_t(targets[i]) + 1; }, relation_value);
                }
                Kokkos::fence();
                incidence.samples.push_back(elapsed(start));
                if (relation->entry_count() != reference)
                    throw std::runtime_error("device relation entry count mismatch");
            }
        }
        const auto report = [&](const char* phase, const Timing& timing, double value) {
            emit(a, mesh, name, phase, timing, value,
                mesh_payload, relation_payload, resident, increase);
        };
        report("device_geometry_upload", geometry_upload, checked_centroid);
        report("device_centroid_kernel", centroid, checked_centroid);
        report("device_geometry_checksum", geometry_checksum,
            static_cast<double>(checked_geometry));
        if (has_relation) {
            report("device_relation_upload", relation_upload,
                static_cast<double>(reference));
            report("device_incidence_kernel", incidence,
                static_cast<double>(reference));
        }
    }
}

int main(int argc, char** argv)
{
    try {
        const auto args = parse_arguments(argc, argv);
        Kokkos::initialize();
        int status = 0;
        try {
            std::cout << std::fixed << std::setprecision(3)
                      << "dimension,shape,side,configuration,phase,cells,trials,rounds,"
                         "ms_p50,ms_p95,result,source,scramble,seed,"
                         "device_mesh_payload_bytes,device_relation_payload_bytes,"
                         "rss_after_mirrors_bytes_p50,rss_delta_mirrors_bytes_p50\n";
            if (args.configuration == "all" || args.configuration == "minimal")
                run_case(args,
                    idmesh::BuildOptions {.topology = idmesh::TopologyLayout::cell_vertices, .relations = idmesh::RelationDirection::downward},
                    "minimal");
            if (args.configuration == "all" || args.configuration == "finite_volume")
                run_case(args,
                    idmesh::BuildOptions {.topology = idmesh::TopologyLayout::cell_facets},
                    "finite_volume");
            if (args.configuration == "all" || args.configuration == "full")
                run_case(args, idmesh::BuildOptions {}, "full");
        }
        catch (const std::exception& error) {
            std::cerr << "Kokkos benchmark: " << error.what() << '\n';
            status = 1;
        }
        Kokkos::finalize();
        return status;
    }
    catch (const std::exception& error) {
        std::cerr << "Kokkos benchmark initialization: " << error.what() << '\n';
        return 1;
    }
}
