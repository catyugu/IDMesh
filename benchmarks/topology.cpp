#include "idmesh/mesh.hpp"
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using Index = idmesh::Index;

std::uint64_t consume(const idmesh::Mesh& mesh, unsigned rounds)
{
    std::uint64_t sum = 0;
    const auto cells = mesh.dimension();
    auto cv = mesh.relation(cells, 0);
    for (unsigned round = 0; round < rounds; ++round) {
        for (Index c = 0; c < mesh.entity_count(cells); ++c) {
            for (Index v : cv.row(c)) sum += v;
        }
    }
    return sum;
}

void benchmark(unsigned side, unsigned repetitions, idmesh::TopologyOptions options, const char* name)
{
    const auto start = Clock::now();
    idmesh::MeshBuilder builder(2, 2);
    for (unsigned y = 0; y <= side; ++y)
        for (unsigned x = 0; x <= side; ++x)
            builder.add_vertex({double(x), double(y), 0.0});
    const auto vertex = [side](unsigned x, unsigned y) -> Index {
        return static_cast<Index>(y * (side + 1) + x);
    };
    for (unsigned y = 0; y < side; ++y) {
        for (unsigned x = 0; x < side; ++x) {
            const std::array<Index, 4> corners {
                vertex(x,y), vertex(x+1,y), vertex(x+1,y+1), vertex(x,y+1)
            };
            if ((x + y) % 2 == 0) {
                builder.add_cell(idmesh::CellType::quadrilateral, corners);
            } else {
                const std::array<Index, 3> left {corners[0],corners[1],corners[2]};
                const std::array<Index, 3> right {corners[0],corners[2],corners[3]};
                builder.add_cell(idmesh::CellType::triangle, left);
                builder.add_cell(idmesh::CellType::triangle, right);
            }
        }
    }
    auto mesh = std::move(builder).finalize(options);
    const auto built = Clock::now();
    const std::uint64_t checksum = consume(mesh, repetitions);
    const auto done = Clock::now();
    const auto build_ms = std::chrono::duration<double, std::milli>(built - start).count();
    const auto traverse_ms = std::chrono::duration<double, std::milli>(done - built).count();
    std::size_t relation_bytes = 0;
    for (Index from = 0; from <= mesh.dimension(); ++from) {
        for (Index to = 0; to <= mesh.dimension(); ++to) {
            if (!mesh.has_relation(from, to)) continue;
            auto view = mesh.relation(from, to);
            relation_bytes += view.offsets.size_bytes() + view.indices.size_bytes()
                + view.signs.size_bytes() + view.orientations.size_bytes();
        }
    }
    std::cout << std::fixed << std::setprecision(3)
        << name << ","
        << mesh.entity_count(mesh.dimension()) << ","
        << mesh.entity_count(1) << ","
        << build_ms << "," << traverse_ms << ","
        << relation_bytes << "," << checksum << "\n";
}
}

int main(int argc, char** argv)
{
    try {
        unsigned side = 128, repetitions = 20;
        if (argc > 1) side = static_cast<unsigned>(std::stoul(argv[1]));
        if (argc > 2) repetitions = static_cast<unsigned>(std::stoul(argv[2]));
        if (side == 0 || side > 2048 || repetitions == 0)
            throw std::invalid_argument("side must be 1..2048 and repetitions must be positive");
        std::cout << "configuration,cells,edges,build_ms,traverse_ms,relation_bytes,checksum\n";
        benchmark(side, repetitions, idmesh::TopologyOptions::cells_and_vertices(), "minimal");
        benchmark(side, repetitions, idmesh::TopologyOptions::finite_volume(), "finite_volume");
        benchmark(side, repetitions, idmesh::TopologyOptions::full(), "full");
    } catch (const std::exception& error) {
        std::cerr << "benchmark error: " << error.what() << "\n";
        return 1;
    }
}
