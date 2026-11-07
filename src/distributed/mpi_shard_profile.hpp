#pragma once

#include "idmesh/mpi.hpp"

#include <string>

namespace idmesh::mpi::detail {

    struct ShardReadProfile {
        double parse_ms = 0.0;
        double local_mesh_ms = 0.0;
        double metadata_ms = 0.0;
        double exchange_plan_ms = 0.0;
        double validate_ms = 0.0;
    };

    // Developer-only profiling entry point. This header is not installed.
    IDMESH_MPI_API PartitionedMesh read_partition_shard_profiled(
        const Context& context, const std::string& path,
        BuildOptions options, ShardReadProfile& profile);

} // namespace idmesh::mpi::detail
