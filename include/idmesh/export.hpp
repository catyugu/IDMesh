#pragma once

// Export each SDK component independently: the core may be static while the
// MPI and Kokkos adapters are shared libraries.
#if defined(_WIN32)
#define IDMESH_DETAIL_EXPORT __declspec(dllexport)
#define IDMESH_DETAIL_IMPORT __declspec(dllimport)
#elif defined(__GNUC__) || defined(__clang__)
#define IDMESH_DETAIL_EXPORT __attribute__((visibility("default")))
#define IDMESH_DETAIL_IMPORT __attribute__((visibility("default")))
#else
#define IDMESH_DETAIL_EXPORT
#define IDMESH_DETAIL_IMPORT
#endif

#if defined(IDMESH_SHARED)
#if defined(IDMESH_EXPORTS)
#define IDMESH_API IDMESH_DETAIL_EXPORT
#else
#define IDMESH_API IDMESH_DETAIL_IMPORT
#endif
#else
#define IDMESH_API
#endif

#if defined(IDMESH_MPI_SHARED)
#if defined(IDMESH_MPI_EXPORTS)
#define IDMESH_MPI_API IDMESH_DETAIL_EXPORT
#else
#define IDMESH_MPI_API IDMESH_DETAIL_IMPORT
#endif
#else
#define IDMESH_MPI_API
#endif

#if defined(IDMESH_KOKKOS_SHARED)
#if defined(IDMESH_KOKKOS_EXPORTS)
#define IDMESH_KOKKOS_API IDMESH_DETAIL_EXPORT
#else
#define IDMESH_KOKKOS_API IDMESH_DETAIL_IMPORT
#endif
#else
#define IDMESH_KOKKOS_API
#endif

#if defined(IDMESH_MPI_KOKKOS_SHARED)
#if defined(IDMESH_MPI_KOKKOS_EXPORTS)
#define IDMESH_MPI_KOKKOS_API IDMESH_DETAIL_EXPORT
#else
#define IDMESH_MPI_KOKKOS_API IDMESH_DETAIL_IMPORT
#endif
#else
#define IDMESH_MPI_KOKKOS_API
#endif
