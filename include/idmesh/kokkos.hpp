#pragma once
#include "idmesh/mesh.hpp"
#include <cstdint>

#if defined(_WIN32)
#  if defined(IDMESH_KOKKOS_EXPORTS)
#    define IDMESH_KOKKOS_API __declspec(dllexport)
#  elif defined(IDMESH_KOKKOS_SHARED)
#    define IDMESH_KOKKOS_API __declspec(dllimport)
#  else
#    define IDMESH_KOKKOS_API
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define IDMESH_KOKKOS_API __attribute__((visibility("default")))
#else
#  define IDMESH_KOKKOS_API
#endif

namespace idmesh::kokkos {
// A simple device-ready topology traversal. No Kokkos types escape the SDK ABI.
IDMESH_KOKKOS_API std::uint64_t incidence_count(const Mesh& mesh, Index from, Index to);
} // namespace idmesh::kokkos
