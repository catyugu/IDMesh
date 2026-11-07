#pragma once

#include "idmesh/export.hpp"

#include "idmesh/mesh.hpp"
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace idmesh::mpi {

    enum class DeviceMemoryKind : std::uint8_t {
        host,
        cuda,
        hip
    };

    class PartitionedMesh;
    class ExchangePlan;

    class IDMESH_MPI_API ExchangeRequest {
    public:
        ~ExchangeRequest();
        ExchangeRequest(ExchangeRequest&&) noexcept;
        ExchangeRequest& operator=(ExchangeRequest&&) noexcept;
        ExchangeRequest(const ExchangeRequest&) = delete;
        ExchangeRequest& operator=(const ExchangeRequest&) = delete;

        // Rebinds the preallocated request to a field and starts MPI_Ialltoallv.
        // The field span must remain alive until wait() completes.
        void start(std::span<double> values);
        void wait();
        bool active() const noexcept;
        Index component_count() const noexcept;

    private:
        struct Data;
        std::shared_ptr<Data> data_;
        explicit ExchangeRequest(std::shared_ptr<Data> data);
        friend class ExchangePlan;
    };

    class IDMESH_MPI_API Context {
    public:
        static Context world();
        // Pass the integer returned by MPI_Comm_c2f. The communicator is
        // duplicated internally, so the caller may release its original handle.
        static Context from_fortran_handle(std::int64_t handle);

        int rank() const noexcept;
        int size() const noexcept;

    private:
        struct Data;
        std::shared_ptr<Data> data_;
        explicit Context(std::shared_ptr<Data> data);

        friend PartitionedMesh partition_replicated(
            const Context&, const Mesh&, Index);
        friend PartitionedMesh read_partition_shard(
            const Context&, const std::string&, BuildOptions);
    };

    class IDMESH_MPI_API ExchangePlan {
    public:
        ExchangePlan() = default;

        void exchange(std::span<double> values, Index components = 1) const;
        // Packs owned values immediately, starts a nonblocking collective, and
        // unpacks ghost values in wait(). The span must remain alive until then.
        ExchangeRequest begin_exchange(
            std::span<double> values, Index components = 1) const;
        // Precomputes counts/displacements and allocates communication buffers
        // once. Reuse with start()/wait() in steady exchange loops.
        ExchangeRequest make_request(Index components = 1) const;
        bool empty() const noexcept;
        int rank_count() const noexcept;
        Index local_cell_count() const noexcept;
        std::span<const Index> send_indices(int peer) const;
        std::span<const Index> receive_indices(int peer) const;
        bool direct_memory_supported(DeviceMemoryKind memory) const;

        // Buffers are packed peer-by-peer in exactly send_indices()/
        // receive_indices() order. Non-host pointers require explicit support
        // from the MPI implementation.
        void exchange_packed(const void* send, void* receive, Index components,
            DeviceMemoryKind memory) const;

    private:
        struct Data;
        std::shared_ptr<Data> data_;
        explicit ExchangePlan(std::shared_ptr<Data> data);
        friend class PartitionedMesh;
        friend class ExchangeRequest;
        friend PartitionedMesh partition_replicated(
            const Context&, const Mesh&, Index);
        friend PartitionedMesh read_partition_shard(
            const Context&, const std::string&, BuildOptions);
    };

    class IDMESH_MPI_API PartitionedMesh {
    public:
        const Mesh& mesh() const noexcept;
        Index owned_cell_count() const noexcept;
        std::span<const std::uint64_t> global_ids(Index dimension) const;
        std::span<const int> owner_ranks(Index dimension) const;
        const ExchangePlan& exchange_plan() const noexcept;

    private:
        struct Data;
        std::shared_ptr<Data> data_;
        explicit PartitionedMesh(std::shared_ptr<Data> data);

        friend PartitionedMesh partition_replicated(
            const Context&, const Mesh&, Index);
        friend PartitionedMesh read_partition_shard(
            const Context&, const std::string&, BuildOptions);
    };

    IDMESH_MPI_API PartitionedMesh partition_replicated(
        const Context& context, const Mesh& source, Index ghost_layers = 1);

    IDMESH_MPI_API PartitionedMesh read_partition_shard(
        const Context& context, const std::string& path,
        BuildOptions options = {});

    // Source-side deterministic facet dual-graph partition and shard exporter.
    // The source-side preprocessing operation is not a distributed file reader.
    IDMESH_MPI_API std::vector<int> partition_graph(const Mesh& source, int parts);
    IDMESH_MPI_API void write_partition_shards(const Mesh& source,
        std::span<const int> owners, Index ghost_layers, const std::string& prefix);

} // namespace idmesh::mpi
