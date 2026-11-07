#pragma once

#include "idmesh/export.hpp"
#include "idmesh/mesh.hpp"

#include <span>
#include <string>
#include <vector>

namespace idmesh {

    IDMESH_API std::vector<int> partition_graph(const Mesh& source, int parts);

    IDMESH_API void write_partition_shards(const Mesh& source,
        std::span<const int> owners, Index ghost_layers, const std::string& prefix);

} // namespace idmesh
