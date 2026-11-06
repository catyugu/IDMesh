#pragma once

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace idmesh::detail {
    // External inputs and public API preconditions must always be validated.
    inline void require_argument(bool valid, const char* message)
    {
        if (!valid)
            throw std::invalid_argument(message);
    }

    inline void require_range(bool valid, const char* message)
    {
        if (!valid)
            throw std::out_of_range(message);
    }

    inline void require_length(bool valid, const char* message)
    {
        if (!valid)
            throw std::length_error(message);
    }

    inline void require_state(bool valid, const char* message)
    {
        if (!valid)
            throw std::logic_error(message);
    }

    inline void require_runtime(bool valid, const char* message)
    {
        if (!valid)
            throw std::runtime_error(message);
    }

    [[noreturn]] inline void assertion_failed(
        const char* expression, const char* file, int line) noexcept
    {
        std::fprintf(stderr, "IDMesh internal assertion failed: %s (%s:%d)\n",
            expression, file, line);
        std::abort();
    }
} // namespace idmesh::detail

// For bugs in internal invariants only; never for external input validation.
// Unlike assert(), this stays active in Release builds.
#define IDMESH_ASSERT(expression)                                                \
    do {                                                                         \
        if (!(expression))                                                       \
            ::idmesh::detail::assertion_failed(#expression, __FILE__, __LINE__); \
    } while (false)
