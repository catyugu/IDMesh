#include "idmesh/kokkos.hpp"
#include "mesh_factory.hpp"

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

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
                             "[--scramble] [--seed N] [--jitter D] [--gmsh FILE]\n";
                std::exit(0);
            }
            if (option == "--scramble") {
                a.generated.scramble = true;
                continue;
            }
            if (option == "--shape" || option == "--dim" || option == "--trials"
                || option == "--gmsh" || option == "--seed" || option == "--jitter") {
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

    void emit(const Arguments& a, const idmesh::Mesh& mesh, const char* configuration,
        const char* phase, const Timing& values, double checked_value)
    {
        std::cout << mesh.dimension() << ',' << idmesh_bench::shape_name(a.generated.shape)
                  << ',' << (a.gmsh.empty() ? a.generated.side : 0)
                  << ',' << configuration << ',' << phase << ','
                  << mesh.entity_count(mesh.dimension()) << ','
                  << a.trials << ',' << a.rounds << ',' << values.median()
                  << ',' << values.p95() << ',' << checked_value << ','
                  << (a.gmsh.empty() ? "generated" : "gmsh") << ','
                  << a.generated.scramble << ',' << a.generated.seed << '\n';
    }

    void run_case(const Arguments& a, idmesh::TopologyOptions options, const char* name)
    {
        idmesh::Mesh mesh;
        if (a.gmsh.empty()) {
            auto builder = idmesh_bench::make_input(a.generated);
            mesh = std::move(builder).finalize(options);
        }
        else {
            mesh = idmesh::read_gmsh(a.gmsh, options);
        }
        Timing geometry_upload, relation_upload, centroid, incidence, geometry_checksum;
        const auto dim = mesh.dimension();
        const bool has_relation = mesh.has_relation(dim, dim - 1);
        const auto reference = has_relation ? mesh.relation(dim, dim - 1).indices.size() : 0;
        double checked_centroid = 0.0;
        std::uint64_t checked_geometry = 0;
        for (unsigned trial = 0; trial < a.trials; ++trial) {
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

            start = Clock::now();
            std::array<double, 3> center {};
            for (unsigned round = 0; round < a.rounds; ++round)
                center = device->centroid_sum();
            Kokkos::fence();
            centroid.samples.push_back(elapsed(start));
            const double scalar = center[0] + center[1] + center[2];
            if (trial && std::abs(scalar - checked_centroid) > 1e-9 * (1 + std::abs(scalar)))
                throw std::runtime_error("device centroid result changed across trials");
            checked_centroid = scalar;

            start = Clock::now();
            std::uint64_t geometry_value = 0;
            for (unsigned round = 0; round < a.rounds; ++round)
                geometry_value = device->geometry_connectivity_checksum();
            Kokkos::fence();
            geometry_checksum.samples.push_back(elapsed(start));
            if (trial && geometry_value != checked_geometry)
                throw std::runtime_error("device geometry checksum changed");
            checked_geometry = geometry_value;

            if (has_relation) {
                start = Clock::now();
                std::uint64_t count = 0;
                for (unsigned round = 0; round < a.rounds; ++round)
                    count = relation->incidence_count();
                Kokkos::fence();
                incidence.samples.push_back(elapsed(start));
                if (count != reference)
                    throw std::runtime_error("device incidence count mismatch");
            }
        }
        emit(a, mesh, name, "device_geometry_upload", geometry_upload, checked_centroid);
        emit(a, mesh, name, "device_centroid_kernel", centroid, checked_centroid);
        emit(a, mesh, name, "device_geometry_checksum", geometry_checksum,
            static_cast<double>(checked_geometry));
        if (has_relation) {
            emit(a, mesh, name, "device_relation_upload", relation_upload,
                static_cast<double>(reference));
            emit(a, mesh, name, "device_incidence_kernel", incidence,
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
                         "ms_p50,ms_p95,result,source,scramble,seed\n";
            run_case(args, idmesh::TopologyOptions::cells_and_vertices(), "minimal");
            run_case(args, idmesh::TopologyOptions::finite_volume(), "finite_volume");
            run_case(args, idmesh::TopologyOptions::full(), "full");
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
