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

    // MPI_Initialized stays true after MPI_Finalize, so both states matter.
    inline void require_active()
    {
        int initialized = 0;
        check_mpi(MPI_Initialized(&initialized), "MPI_Initialized");
        if (!initialized)
            throw std::logic_error("MPI must be initialized by the application");

        int finalized = 0;
        check_mpi(MPI_Finalized(&finalized), "MPI_Finalized");
        if (finalized)
            throw std::logic_error("MPI has already been finalized");
    }
} // namespace idmesh::mpi::detail
