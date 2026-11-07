#pragma once
#include "idmesh/mpi.hpp"
#include <mpi.h>
#include <climits>
#include <stdexcept>
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

        void finalize_layout()
        {
            const int peers = static_cast<int>(send.size());
            send_records.resize(peers);
            recv_records.resize(peers);
            send_record_offsets.resize(peers);
            recv_record_offsets.resize(peers);
            int send_offset = 0;
            int recv_offset = 0;
            for (int peer = 0; peer < peers; ++peer) {
                if (send[peer].size() > static_cast<std::size_t>(INT_MAX)
                    || recv[peer].size() > static_cast<std::size_t>(INT_MAX))
                    throw std::length_error("halo peer record count exceeds MPI int range");
                send_records[peer] = static_cast<int>(send[peer].size());
                recv_records[peer] = static_cast<int>(recv[peer].size());
                send_record_offsets[peer] = send_offset;
                recv_record_offsets[peer] = recv_offset;
                if (send_offset > INT_MAX - send_records[peer]
                    || recv_offset > INT_MAX - recv_records[peer])
                    throw std::length_error("halo record displacement overflow");
                send_offset += send_records[peer];
                recv_offset += recv_records[peer];
            }
        }
    };

    struct ExchangeRequest::Data {
        ExchangePlan plan;
        std::span<double> values;
        Index components = 1;
        std::vector<int> send_counts;
        std::vector<int> recv_counts;
        std::vector<int> send_offsets;
        std::vector<int> recv_offsets;
        std::vector<double> send_buffer;
        std::vector<double> recv_buffer;
        MPI_Request request = MPI_REQUEST_NULL;
        bool active = false;

        ~Data();
        void unpack();
    };

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
