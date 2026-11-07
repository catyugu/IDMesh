#pragma once
#include "idmesh/mpi.hpp"
#include <mpi.h>
#include <mutex>
#include <utility>
#include <vector>

namespace idmesh::mpi {
    struct Context::Data {
        MPI_Comm comm = MPI_COMM_NULL;
        int rank = 0;
        int size = 0;

        ~Data()
        {
            if (comm == MPI_COMM_NULL)
                return;
            int finalized = 0;
            MPI_Finalized(&finalized);
            if (!finalized)
                MPI_Comm_free(&comm);
        }
    };

    struct ExchangePlan::Data {
        std::vector<std::vector<Index>> send;
        std::vector<std::vector<Index>> recv;
        std::vector<int> send_records;
        std::vector<int> recv_records;
        std::vector<int> send_record_offsets;
        std::vector<int> recv_record_offsets;
        Index local_cells = 0;
        MPI_Comm comm = MPI_COMM_NULL;
        std::shared_ptr<void> context_owner;

        // Reused only by the convenience host exchange path. Packed exchange
        // remains allocation-local and does not share mutable transport state.
        mutable std::mutex host_mutex;
        mutable Index host_components = 0;
        mutable std::vector<int> host_send_counts;
        mutable std::vector<int> host_recv_counts;
        mutable std::vector<int> host_send_offsets;
        mutable std::vector<int> host_recv_offsets;
        mutable std::vector<double> host_send_buffer;
        mutable std::vector<double> host_recv_buffer;
    };

    struct ExchangeRequest::Data {
        std::shared_ptr<ExchangePlan::Data> plan;
        std::span<double> values;
        Index components = 1;
        std::vector<int> send_counts;
        std::vector<int> recv_counts;
        std::vector<int> send_offsets;
        std::vector<int> recv_offsets;
        std::vector<double> send_buffer;
        std::vector<double> recv_buffer;
        MPI_Request request = MPI_REQUEST_NULL;
        bool unpacked = false;

        ~Data();
        void unpack();
    };

    namespace detail {
        void finalize_exchange_layout(ExchangePlan::Data& data);
    }

    struct PartitionedMesh::Data {
        explicit Data(Mesh value)
            : mesh(std::move(value))
        {
        }

        Mesh mesh;
        Index owned_cells = 0;
        std::vector<std::vector<std::uint64_t>> global_ids;
        std::vector<std::vector<int>> owner_ranks;
        ExchangePlan exchange;
    };
}
