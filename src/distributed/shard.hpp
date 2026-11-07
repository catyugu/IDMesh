#pragma once

#include "idmesh/mesh.hpp"

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace idmesh::detail::shard {

    // Shard v3: DIM, VERTICES, GEOMETRY_NODES, CELLS, ENTITIES,
    // HALOS, LABELS, NAMES, END. Entity IDs in the file are global.
    inline constexpr char magic[] = "IDMESH_SHARD";
    inline constexpr int version = 3;

    struct Vertex {
        std::array<double, 3> position {};
        int owner = -1;
    };

    struct Cell {
        std::uint64_t id = 0;
        int owner = -1;
        CellType type = CellType::segment;
        std::vector<std::uint64_t> geometry;
    };

    struct EntityIdentity {
        Index dimension = 0;
        std::uint64_t id = 0;
        int owner = -1;
        std::vector<std::uint64_t> vertices;
    };

    struct HaloRecord {
        Index dimension = 0;
        int peer = -1;
        std::vector<std::uint64_t> send;
        std::vector<std::uint64_t> receive;
    };

    struct Label {
        Index dimension = 0;
        int value = 0;
        std::vector<std::uint64_t> entities;
    };

    struct Shard {
        Index dimension = 0;
        Index geometric_dimension = 0;
        std::map<std::uint64_t, Vertex> vertices;
        std::map<std::uint64_t, std::array<double, 3>> geometry_nodes;
        std::vector<Cell> cells;
        std::vector<EntityIdentity> entities;
        std::vector<HaloRecord> halos;
        std::vector<Label> labels;
        std::map<std::pair<Index, int>, std::string> label_names;
    };

    Shard read(const std::string& path, int rank, int size);

} // namespace idmesh::detail::shard
