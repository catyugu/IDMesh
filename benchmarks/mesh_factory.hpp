#pragma once

#include "idmesh/mesh.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace idmesh_bench {
    using Index = idmesh::Index;

    enum class Shape { simplex,
        tensor,
        mixed };

    struct GeneratedMesh {
        unsigned side = 128;
        unsigned dimension = 2;
        Shape shape = Shape::mixed;
        bool scramble = false;
        std::uint64_t seed = 0;
        double jitter = 0.0;
        bool cell_labels = false;
    };

    inline const char* shape_name(Shape shape)
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

    inline Shape parse_shape(std::string_view shape)
    {
        if (shape == "simplex")
            return Shape::simplex;
        if (shape == "tensor")
            return Shape::tensor;
        if (shape == "mixed")
            return Shape::mixed;
        throw std::invalid_argument("invalid --shape; use simplex, tensor or mixed");
    }

    inline idmesh::MeshBuilder make_input(const GeneratedMesh& args)
    {
        const Index side = args.side;
        const Index width = side + 1;
        const Index plane = width * width;
        const Index vertex_count = args.dimension == 3 ? plane * width : plane;
        idmesh::MeshBuilder builder(args.dimension, args.dimension);

        // Scrambling changes both local entity IDs and cell insertion order;
        // it preserves the combinatorial mesh and supports reproducible trials.
        std::vector<Index> vertex_order;
        std::vector<Index> old_to_new;
        std::mt19937_64 generator(args.seed);
        if (args.scramble) {
            vertex_order.resize(vertex_count);
            std::iota(vertex_order.begin(), vertex_order.end(), Index {0});
            std::shuffle(vertex_order.begin(), vertex_order.end(), generator);
            old_to_new.resize(vertex_count);
            for (Index new_id = 0; new_id < vertex_count; ++new_id)
                old_to_new[vertex_order[new_id]] = new_id;
        }
        std::uniform_real_distribution<double> displacement(-args.jitter, args.jitter);
        for (Index id = 0; id < vertex_count; ++id) {
            const Index old_id = args.scramble ? vertex_order[id] : id;
            const Index x = old_id % width;
            const Index y = (old_id / width) % width;
            const Index z = args.dimension == 3 ? old_id / plane : 0;
            std::array<double, 3> point {double(x), double(y), double(z)};
            if (args.jitter != 0.0 && x > 0 && y > 0 && x < side && y < side
                && (args.dimension == 2 || (z > 0 && z < side))) {
                point[0] += displacement(generator);
                point[1] += displacement(generator);
                if (args.dimension == 3)
                    point[2] += displacement(generator);
            }
            builder.add_vertex(point);
        }
        const auto vertex = [&](Index x, Index y, Index z) -> Index {
            const Index old_id = z * plane + y * width + x;
            return args.scramble ? old_to_new[old_id] : old_id;
        };

        const Index box_count = args.dimension == 3 ? side * side * side : side * side;
        std::vector<Index> box_order;
        if (args.scramble) {
            box_order.resize(box_count);
            std::iota(box_order.begin(), box_order.end(), Index {0});
            std::shuffle(box_order.begin(), box_order.end(), generator);
        }
        for (Index item = 0; item < box_count; ++item) {
            const Index box = args.scramble ? box_order[item] : item;
            const Index x = box % side;
            const Index y = (box / side) % side;
            const Index z = args.dimension == 3 ? box / (side * side) : 0;
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
                // Six conforming tetrahedra per cube.
                for (const auto& tet : std::array<std::array<Index, 4>, 6> {{{{v0, v1, v2, v6}}, {{v0, v2, v3, v6}},
                         {{v0, v3, v7, v6}}, {{v0, v7, v4, v6}},
                         {{v0, v4, v5, v6}}, {{v0, v5, v1, v6}}}}) {
                    builder.add_cell(idmesh::CellType::tetrahedron, tet, args.cell_labels ? 1 : 0);
                }
            }
            else if (split) {
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
        return builder;
    }
} // namespace idmesh_bench
