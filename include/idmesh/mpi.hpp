#pragma once

#include "idmesh/export.hpp"

#include "idmesh/mesh.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace idmesh::mpi {

    class PartitionedMesh;
    class ExchangePlan;
    namespace kokkos {
        class DeviceHaloPlan;
    }

    class IDMESH_MPI_API ExchangeRequest {
    public:
        ~ExchangeRequest();
        ExchangeRequest(ExchangeRequest&&) noexcept;
        ExchangeRequest& operator=(ExchangeRequest&&) noexcept;
        ExchangeRequest(const ExchangeRequest&) = delete;
        ExchangeRequest& operator=(const ExchangeRequest&) = delete;

        // Rebinds the preallocated request to packed fixed-width entity records.
        // The byte span must remain alive until wait() completes.
        void start(std::span<std::byte> records);
        void wait();

    private:
        struct Data;
        std::unique_ptr<Data> data_;
        explicit ExchangeRequest(std::unique_ptr<Data> data);
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

        friend PartitionedMesh read_partition_shard(
            const Context&, const std::string&, BuildOptions);
    };

    class IDMESH_MPI_API ExchangePlan {
    public:
        ExchangePlan() = default;

        // Precomputes byte counts/displacements and communication buffers for
        // fixed-width entity records. Reuse with start()/wait().
        ExchangeRequest make_request(std::size_t bytes_per_entity) const;
        bool empty() const noexcept;
        int rank_count() const noexcept;
        Index local_entity_count() const noexcept;
        std::span<const Index> send_indices(int peer) const;
        std::span<const Index> receive_indices(int peer) const;

    private:
        struct Data;
        std::shared_ptr<Data> data_;
        explicit ExchangePlan(std::shared_ptr<Data> data);
        friend class PartitionedMesh;
        friend class ExchangeRequest;
        friend class kokkos::DeviceHaloPlan;
        friend PartitionedMesh read_partition_shard(
            const Context&, const std::string&, BuildOptions);
    };

    class IDMESH_MPI_API PartitionedMesh {
    public:
        const Mesh& mesh() const noexcept;
        std::span<const Index> owned_entities(Index dimension) const;
        std::span<const std::uint64_t> global_ids(Index dimension) const;
        std::span<const int> owner_ranks(Index dimension) const;
        const ExchangePlan& exchange_plan(Index dimension) const;

    private:
        struct Data;
        std::shared_ptr<Data> data_;
        explicit PartitionedMesh(std::shared_ptr<Data> data);

        friend PartitionedMesh read_partition_shard(
            const Context&, const std::string&, BuildOptions);
    };

    IDMESH_MPI_API PartitionedMesh read_partition_shard(
        const Context& context, const std::string& path,
        BuildOptions options = {});

} // namespace idmesh::mpi
