#include "idmesh/kokkos.hpp"
#include <Kokkos_Core.hpp>
#include <stdexcept>

namespace idmesh::kokkos {
std::uint64_t incidence_count(const Mesh& mesh, Index from, Index to)
{
    const auto relation = mesh.relation(from, to);
    if (!Kokkos::is_initialized())
        throw std::logic_error("Kokkos must be initialized before using IDMesh Kokkos adapter");
    using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
    Kokkos::View<Index*, MemorySpace> offsets("idmesh_offsets", relation.rows + 1);
    auto host = Kokkos::create_mirror_view(offsets);
    if (relation.offsets.empty()) {
        for (Index row = 0; row <= relation.rows; ++row)
            host(row) = row * relation.stride;
    } else {
        for (Index row = 0; row <= relation.rows; ++row)
            host(row) = relation.offsets[row];
    }
    Kokkos::deep_copy(offsets, host);
    std::uint64_t result = 0;
    Kokkos::parallel_reduce(
        "idmesh_count_incidences",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, relation.rows),
        KOKKOS_LAMBDA(const Index row, std::uint64_t& count) {
            count += offsets(row + 1) - offsets(row);
        },
        result);
    return result;
}
} // namespace idmesh::kokkos
