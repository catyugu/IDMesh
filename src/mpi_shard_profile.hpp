#pragma once

#include "idmesh/mpi.hpp"

#include <cstdint>
#include <string>

namespace idmesh::mpi::detail {

    struct ShardReadProfile {
        double parse_ms = 0.0;
        double local_mesh_ms = 0.0;
        double owner_candidates_ms = 0.0;
        double embedded_identity_ms = 0.0;
        bool registry_used = false;
        double registry_request_ms = 0.0;
        double registry_resolve_ms = 0.0;
        double registry_reply_ms = 0.0;
        double exchange_plan_ms = 0.0;
        double validate_ms = 0.0;
        std::uint64_t registry_send_bytes = 0;
        std::uint64_t registry_receive_bytes = 0;
        std::uint64_t exchange_plan_send_bytes = 0;
        std::uint64_t exchange_plan_receive_bytes = 0;
    };

    // Developer-only profiling entry point. This header is not installed.
    IDMESH_MPI_API PartitionedMesh read_partition_shard_profiled(
        const Context& context, const std::string& path,
        BuildOptions options, ShardReadProfile& profile);

} // namespace idmesh::mpi::detail
