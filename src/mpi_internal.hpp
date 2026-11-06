#pragma once
#include "idmesh/mpi.hpp"
#include <mpi.h>
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
        Index local_cells = 0;
        MPI_Comm comm = MPI_COMM_NULL;
        std::shared_ptr<void> context_owner;
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
