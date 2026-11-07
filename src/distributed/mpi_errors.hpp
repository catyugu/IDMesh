#pragma once

#include <mpi.h>

#include <stdexcept>
#include <string>

namespace idmesh::mpi::detail {
    inline void check_mpi(int result, const char* operation)
    {
        if (result == MPI_SUCCESS)
            return;

        std::string message = "IDMesh ";
        message += operation;
        message += " failed (MPI error ";
        message += std::to_string(result);
        message += ')';

        char description[MPI_MAX_ERROR_STRING] {};
        int length = 0;
        if (MPI_Error_string(result, description, &length) == MPI_SUCCESS && length > 0) {
            message += ": ";
            message.append(description, static_cast<std::size_t>(length));
        }
        throw std::runtime_error(message);
    }
} // namespace idmesh::mpi::detail
