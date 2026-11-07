#pragma once

#include "idmesh/export.hpp"
#include "idmesh/mesh.hpp"

#include <span>
#include <string>
#include <vector>

namespace idmesh {

    // Dependency-free deterministic baseline partitioner.
    IDMESH_API std::vector<int> partition_graph(const Mesh& source, int parts);

    // 'owners[cell]' is the target rank. The ownership vector is the stable
    // integration boundary for external partitioners; it need not originate
    // from partition_graph().
    IDMESH_API void write_partition_shards(const Mesh& source,
        std::span<const int> owners, Index ghost_layers, const std::string& prefix);

} // namespace idmesh
