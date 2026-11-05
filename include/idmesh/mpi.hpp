#pragma once

#include "idmesh/mesh.hpp"
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#if defined(_WIN32)
#if defined(IDMESH_MPI_EXPORTS)
#define IDMESH_MPI_API __declspec(dllexport)
#elif defined(IDMESH_MPI_SHARED)
#define IDMESH_MPI_API __declspec(dllimport)
#else
#define IDMESH_MPI_API
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define IDMESH_MPI_API __attribute__((visibility("default")))
#else
#define IDMESH_MPI_API
#endif

namespace idmesh::mpi {

    // MPI must already be initialized by the application. No mpi.h dependency
    // appears in public IDMesh headers. World communicator is the initial scope.
    IDMESH_MPI_API int world_rank();
    IDMESH_MPI_API int world_size();

    enum class DeviceMemoryKind : std::uint8_t {
        Host,
        CUDA,
        HIP
    };
    IDMESH_MPI_API bool gpu_aware_support(DeviceMemoryKind memory);

    struct PartitionedMesh;

    class IDMESH_MPI_API ExchangePlan {
    public:
        ExchangePlan() = default;
        // values: owned cells first, then ghosts. Fixed number of interleaved
        // double components per cell. Blocking, thread-unsafe, host-buffer API.
        void exchange_cells(std::span<double> values, Index components = 1) const;
        bool empty() const noexcept;
        int peer_count() const noexcept;
        Index local_cell_count() const noexcept;
        std::span<const Index> send_indices(int peer) const;
        std::span<const Index> receive_indices(int peer) const;

        // Already-packed contiguous buffers, laid out peer-by-peer in
        // exactly send_indices()/receive_indices() order. Device pointers are
        // accepted only when the MPI implementation explicitly reports GPU
        // awareness. The caller synchronizes device kernels around MPI calls.
        void exchange_packed(const void* send, void* receive, Index components,
            DeviceMemoryKind memory) const;

    private:
        struct Data;
        std::shared_ptr<Data> data_;
        explicit ExchangePlan(std::shared_ptr<Data> data);
        friend struct PartitionedMesh;
        friend PartitionedMesh partition_replicated(const Mesh&, Index);
        friend PartitionedMesh read_partition_shard(const std::string&, TopologyOptions);
    };

    struct PartitionedMesh {
        Mesh mesh;
        Index owned_cells = 0;
        // A global ID is the dimension-specific local ID in the shared source mesh.
        // Indices of each vector correspond to the finalized local mesh.
        std::vector<std::vector<std::uint64_t>> global_ids;
        std::vector<std::vector<int>> owner_ranks;
        ExchangePlan exchange;
    };

    // Initial correct-but-replicated distribution path: all MPI ranks must hold
    // the same source mesh, with matching dimension-specific local IDs.
    // Final local topology is owned-cells-then-facet-neighbor ghosts. Source
    // faces (or edges in 2D) and facet support must be materialized.
    // Large distributed file loading and graph partitioners are separate work.
    IDMESH_MPI_API PartitionedMesh partition_replicated(
        const Mesh& source, Index ghost_layers = 1);

    // Fully distributed read: one rank-local ASCII shard per MPI process.
    // Shared entity IDs and owners are negotiated via a distributed registry.
    IDMESH_MPI_API PartitionedMesh read_partition_shard(
        const std::string& path, TopologyOptions options = TopologyOptions::full());

    // Source-side deterministic facet dual-graph partition and shard exporter.
    // The source-side preprocessing operation is not a distributed file reader.
    IDMESH_MPI_API std::vector<int> partition_graph(const Mesh& source, int parts);
    IDMESH_MPI_API void write_partition_shards(const Mesh& source,
        std::span<const int> owners, Index ghost_layers, const std::string& prefix);

} // namespace idmesh::mpi
