#pragma once
#include "idmesh/mesh.hpp"
#include <cstdint>
#include <memory>

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

// Persistent mirror of one materialized topology relation in the default
// Kokkos memory space. No Kokkos header or template is exposed in this ABI.
// Construct only after Kokkos::initialize(), destroy before finalize().
class IDMESH_KOKKOS_API DeviceRelation {
public:
    DeviceRelation(const Mesh& mesh, Index from, Index to);
    Index rows() const noexcept;
    std::uint64_t incidence_count() const;
    std::uint64_t weighted_index_sum() const;
    std::int64_t boundary_coefficient_sum() const;
private:
    struct Data;
    std::shared_ptr<Data> data_;
};
} // namespace idmesh::kokkos
