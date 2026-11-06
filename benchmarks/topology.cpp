#include "idmesh/mesh.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
    using Clock = std::chrono::steady_clock;
    using Index = idmesh::Index;

    enum class Shape { simplex, tensor, mixed };

    struct Arguments {
        unsigned side = 128;
        unsigned rounds = 20;
        unsigned trials = 5;
        unsigned dimension = 2;
        Shape shape = Shape::mixed;
    };

    unsigned parse_unsigned(const char* text)
    {
        const std::string value(text);
        std::size_t consumed = 0;
        const auto number = std::stoul(value, &consumed);
        if (consumed != value.size() || number > std::numeric_limits<unsigned>::max())
            throw std::invalid_argument("invalid unsigned integer: " + value);
        return static_cast<unsigned>(number);
    }

    Arguments parse_arguments(int argc, char** argv)
    {
        Arguments args;
        unsigned positional = 0;
        for (int i = 1; i < argc; ++i) {
            const std::string_view token = argv[i];
            if (token == "--help") {
                std::cout << "Usage: idmesh_benchmark [side] [rounds] [--dim 2|3] "
                             "[--shape simplex|tensor|mixed] [--trials N]\n"
                             "Output: CSV with separate input/finalize/traversal timings.\n";
                std::exit(0);
            }
            if (token == "--dim" || token == "--shape" || token == "--trials") {
                if (++i >= argc)
                    throw std::invalid_argument("missing benchmark option value");
                if (token == "--dim")
                    args.dimension = parse_unsigned(argv[i]);
                else if (token == "--trials")
                    args.trials = parse_unsigned(argv[i]);
                else {
                    const std::string_view name = argv[i];
                    if (name == "simplex")
                        args.shape = Shape::simplex;
                    else if (name == "tensor")
                        args.shape = Shape::tensor;
                    else if (name == "mixed")
                        args.shape = Shape::mixed;
                    else
                        throw std::invalid_argument("unknown benchmark shape");
                }
            }
            else if (token.starts_with('-')) {
                throw std::invalid_argument("unknown benchmark option");
            }
            else if (positional == 0) {
                args.side = parse_unsigned(argv[i]);
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
        if ((args.dimension != 2 && args.dimension != 3) || args.trials == 0
            || args.trials > 100 || args.rounds == 0 || args.side == 0
            || args.side > (args.dimension == 2 ? 1024u : 64u)) {
            throw std::invalid_argument("invalid dimension, side, rounds or trials");
        }
        return args;
    }

    const char* shape_name(Shape shape)
    {
        switch (shape) {
        case Shape::simplex:
            return "simplex";
        case Shape::tensor:
            return "tensor";
        case Shape::mixed:
            return "mixed";
        }
        return "unknown";
    }

    idmesh::MeshBuilder make_input(const Arguments& args)
    {
        const Index side = args.side;
        idmesh::MeshBuilder builder(args.dimension, args.dimension);
        const Index plane = (side + 1) * (side + 1);
        const auto vertex = [side, plane](Index x, Index y, Index z) -> Index {
            return z * plane + y * (side + 1) + x;
        };
        const Index layers = args.dimension == 3 ? side + 1 : 1;
        for (Index z = 0; z < layers; ++z) {
            for (Index y = 0; y <= side; ++y) {
                for (Index x = 0; x <= side; ++x)
                    builder.add_vertex({double(x), double(y), double(z)});
            }
        }
        for (Index z = 0; z < (args.dimension == 3 ? side : 1); ++z) {
            for (Index y = 0; y < side; ++y) {
                for (Index x = 0; x < side; ++x) {
                    const Index v0 = vertex(x, y, z);
                    const Index v1 = vertex(x + 1, y, z);
                    const Index v2 = vertex(x + 1, y + 1, z);
                    const Index v3 = vertex(x, y + 1, z);
                    const bool split = args.shape == Shape::simplex
                        || (args.shape == Shape::mixed && (x + y) % 2 != 0);
                    if (args.dimension == 2) {
                        if (split) {
                            builder.add_cell(idmesh::CellType::triangle,
                                std::array<Index, 3> {v0, v1, v2});
                            builder.add_cell(idmesh::CellType::triangle,
                                std::array<Index, 3> {v0, v2, v3});
                        }
                        else {
                            builder.add_cell(idmesh::CellType::quadrilateral,
                                std::array<Index, 4> {v0, v1, v2, v3});
                        }
                        continue;
                    }
                    const Index v4 = vertex(x, y, z + 1);
                    const Index v5 = vertex(x + 1, y, z + 1);
                    const Index v6 = vertex(x + 1, y + 1, z + 1);
                    const Index v7 = vertex(x, y + 1, z + 1);
                    if (args.shape == Shape::simplex) {
                        // Consistent six-tetrahedron subdivision of each cube.
                        for (const auto& tet : std::array<std::array<Index, 4>, 6> {{
                                 {{v0, v1, v2, v6}}, {{v0, v2, v3, v6}},
                                 {{v0, v3, v7, v6}}, {{v0, v7, v4, v6}},
                                 {{v0, v4, v5, v6}}, {{v0, v5, v1, v6}}}}) {
                            builder.add_cell(idmesh::CellType::tetrahedron, tet);
                        }
                    }
                    else if (split) {
                        // Extrude two triangles into conforming prisms.
                        builder.add_cell(idmesh::CellType::prism,
                            std::array<Index, 6> {v0, v1, v2, v4, v5, v6});
                        builder.add_cell(idmesh::CellType::prism,
                            std::array<Index, 6> {v0, v2, v3, v4, v6, v7});
                    }
                    else {
                        builder.add_cell(idmesh::CellType::hexahedron,
                            std::array<Index, 8> {v0, v1, v2, v3, v4, v5, v6, v7});
                    }
                }
            }
        }
        return builder;
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

    struct Traversal {
        Timing time;
        std::uint64_t checksum = 0;
        bool available = false;
    };

    std::pair<double, std::uint64_t> traverse(
        const idmesh::Mesh& mesh, Index from, Index to, unsigned rounds)
    {
        const auto relation = mesh.relation(from, to);
        std::uint64_t checksum = 0;
        const auto begin = Clock::now();
        for (unsigned trial = 0; trial < rounds; ++trial) {
            for (Index row = 0; row < relation.rows; ++row) {
                for (Index value : relation.row(row))
                    checksum += static_cast<std::uint64_t>(value) + 1;
            }
        }
        const auto end = Clock::now();
        return {milliseconds(begin, end), checksum};
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

    void run(const Arguments& args, idmesh::TopologyOptions options, const char* configuration)
    {
        Timing input, finalize;
        std::array<Traversal, 3> walks;
        Index cell_count = 0, facet_count = 0, edge_count = 0;
        std::size_t bytes = 0;
        for (unsigned trial = 0; trial < args.trials; ++trial) {
            const auto start = Clock::now();
            auto builder = make_input(args);
            const auto input_ready = Clock::now();
            auto mesh = std::move(builder).finalize(options);
            const auto finalized = Clock::now();
            input.samples.push_back(milliseconds(start, input_ready));
            finalize.samples.push_back(milliseconds(input_ready, finalized));
            const Index dim = mesh.dimension();
            const std::array<std::pair<Index, Index>, 3> relations {{
                {dim, 0}, {dim, dim - 1}, {dim - 1, dim}}};
            if (trial == 0) {
                cell_count = mesh.entity_count(dim);
                facet_count = mesh.has_entities(dim - 1) ? mesh.entity_count(dim - 1) : 0;
                edge_count = mesh.has_entities(1) ? mesh.entity_count(1) : 0;
                bytes = relation_bytes(mesh);
            }
            else if (cell_count != mesh.entity_count(dim) || bytes != relation_bytes(mesh)) {
                throw std::runtime_error("topology size changed across benchmark trials");
            }
            for (std::size_t j = 0; j < walks.size(); ++j) {
                const auto [from, to] = relations[j];
                if (!mesh.has_relation(from, to))
                    continue;
                const auto [elapsed, checksum] = traverse(mesh, from, to, args.rounds);
                if (trial > 0 && walks[j].checksum != checksum)
                    throw std::runtime_error("topology checksum changed across trials");
                walks[j].available = true;
                walks[j].checksum = checksum;
                walks[j].time.samples.push_back(elapsed);
            }
        }
        for (std::size_t j = 0; j < walks.size(); ++j) {
            if (!walks[j].available)
                continue;
            constexpr std::array<std::string_view, 3> names {
                "cell_vertex", "cell_facet", "facet_cell"};
            std::cout << args.dimension << ',' << shape_name(args.shape) << ',' << args.side << ','
                      << configuration << ',' << names[j] << ',' << cell_count << ','
                      << facet_count << ',' << edge_count << ',' << args.trials << ',' << args.rounds
                      << ',' << input.median() << ',' << finalize.median() << ',' << finalize.p95()
                      << ',' << walks[j].time.median() << ',' << walks[j].time.p95()
                      << ',' << bytes << ',' << double(bytes) / double(cell_count)
                      << ',' << walks[j].checksum << '\n';
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
                     "traversal_ms_p95,relation_bytes,relation_bytes_per_cell,checksum\n";
        run(args, idmesh::TopologyOptions::cells_and_vertices(), "minimal");
        run(args, idmesh::TopologyOptions::finite_volume(), "finite_volume");
        run(args, idmesh::TopologyOptions::full(), "full");
    }
    catch (const std::exception& error) {
        std::cerr << "benchmark error: " << error.what() << '\n';
        return 1;
    }
}
