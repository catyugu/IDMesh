#pragma once
#include "idmesh/mpi.hpp"

namespace idmesh::mpi {
    struct ExchangePlan::Data {
        std::vector<std::vector<Index>> send;
        std::vector<std::vector<Index>> recv;
        Index local_cells = 0;
    };
}
