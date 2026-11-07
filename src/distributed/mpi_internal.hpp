#pragma once
#include "idmesh/mpi.hpp"
#include "mpi_errors.hpp"
#include <mpi.h>
#include <climits>
#include <cstddef>
#include <cstdint>
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
        std::vector<Index> send_entities;
        std::vector<Index> recv_entities;
        std::vector<int> send_entity_offsets;
        std::vector<int> recv_entity_offsets;
        std::vector<int> send_peers;
        std::vector<int> recv_peers;
        Index local_entities = 0;
        int ranks = 0;
        MPI_Comm neighbor_comm = MPI_COMM_NULL;
        std::shared_ptr<void> context_owner;

        ~Data()
        {
            if (neighbor_comm == MPI_COMM_NULL)
                return;
            int finalized = 0;
            MPI_Finalized(&finalized);
            if (!finalized)
                MPI_Comm_free(&neighbor_comm);
        }

        std::span<const Index> peer_span(
            const std::vector<Index>& entities,
            const std::vector<int>& offsets, int peer) const
        {
            const auto begin = static_cast<std::size_t>(offsets[peer]);
            const auto end = static_cast<std::size_t>(offsets[peer + 1]);
            return {entities.data() + begin, end - begin};
        }

        void initialize_layout(MPI_Comm comm,
            std::vector<std::vector<Index>> send,
            std::vector<std::vector<Index>> recv)
        {
            ranks = static_cast<int>(send.size());
            if (recv.size() != send.size())
                throw std::invalid_argument("halo send/receive rank count mismatch");
            send_entity_offsets.resize(static_cast<std::size_t>(ranks) + 1);
            recv_entity_offsets.resize(static_cast<std::size_t>(ranks) + 1);
            for (int peer = 0; peer < ranks; ++peer) {
                if (!send[peer].empty())
                    send_peers.push_back(peer);
                if (!recv[peer].empty())
                    recv_peers.push_back(peer);
                if (send_entities.size() > static_cast<std::size_t>(INT_MAX)
                    - send[peer].size()
                    || recv_entities.size() > static_cast<std::size_t>(INT_MAX)
                    - recv[peer].size())
                    throw std::length_error("halo entity layout exceeds MPI int range");
                send_entities.insert(send_entities.end(),
                    send[peer].begin(), send[peer].end());
                recv_entities.insert(recv_entities.end(),
                    recv[peer].begin(), recv[peer].end());
                send_entity_offsets[peer + 1]
                    = static_cast<int>(send_entities.size());
                recv_entity_offsets[peer + 1]
                    = static_cast<int>(recv_entities.size());
            }

            ::idmesh::mpi::detail::check_mpi(
                MPI_Dist_graph_create_adjacent(comm,
                    static_cast<int>(recv_peers.size()), recv_peers.data(),
                    MPI_UNWEIGHTED,
                    static_cast<int>(send_peers.size()), send_peers.data(),
                    MPI_UNWEIGHTED, MPI_INFO_NULL, 0, &neighbor_comm),
                "MPI_Dist_graph_create_adjacent");
            ::idmesh::mpi::detail::check_mpi(
                MPI_Comm_set_errhandler(neighbor_comm, MPI_ERRORS_RETURN),
                "MPI_Comm_set_errhandler");
        }

        void scaled_layout(std::size_t bytes_per_entity,
            std::vector<int>& send_counts, std::vector<int>& recv_counts,
            std::vector<int>& send_offsets, std::vector<int>& recv_offsets) const
        {
            if (bytes_per_entity == 0)
                throw std::invalid_argument("halo record size must be positive");
            const auto width = static_cast<std::uint64_t>(bytes_per_entity);
            send_counts.resize(send_peers.size());
            recv_counts.resize(recv_peers.size());
            send_offsets.resize(send_peers.size());
            recv_offsets.resize(recv_peers.size());

            const auto scale = [width](int value) {
                const auto scaled = static_cast<std::uint64_t>(value) * width;
                if (scaled > INT_MAX)
                    throw std::length_error(
                        "halo exchange exceeds MPI count/displacement limits");
                return static_cast<int>(scaled);
            };
            for (std::size_t i = 0; i < send_peers.size(); ++i) {
                const int peer = send_peers[i];
                send_offsets[i] = scale(send_entity_offsets[peer]);
                send_counts[i] = scale(
                    send_entity_offsets[peer + 1]
                    - send_entity_offsets[peer]);
            }
            for (std::size_t i = 0; i < recv_peers.size(); ++i) {
                const int peer = recv_peers[i];
                recv_offsets[i] = scale(recv_entity_offsets[peer]);
                recv_counts[i] = scale(
                    recv_entity_offsets[peer + 1]
                    - recv_entity_offsets[peer]);
            }
        }
    };

    struct ExchangeRequest::Data {
        ExchangePlan plan;
        std::span<std::byte> records;
        std::size_t bytes_per_entity = 0;
        std::vector<int> send_counts;
        std::vector<int> recv_counts;
        std::vector<int> send_offsets;
        std::vector<int> recv_offsets;
        std::vector<std::byte> send_buffer;
        std::vector<std::byte> recv_buffer;
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
        std::vector<std::vector<Index>> owned_entities;
        std::vector<std::vector<std::uint64_t>> global_ids;
        std::vector<std::vector<int>> owner_ranks;
        std::vector<ExchangePlan> exchanges;
    };
}
