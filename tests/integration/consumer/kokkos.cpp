#include <idmesh/kokkos.hpp>
#include <array>
#include <stdexcept>
#include <type_traits>
#include <utility>

static_assert(std::is_same_v<
    decltype(std::declval<const idmesh::kokkos::DeviceRelation&>().targets()),
    const idmesh::Index*>);
static_assert(std::is_same_v<
    decltype(std::declval<const idmesh::kokkos::DeviceMesh&>().geometry_x()),
    const double*>);

int main()
{
    // Kokkos runtime ownership stays with the application.
    idmesh::MeshBuilder builder(1, 1);
    builder.add_vertex({0, 0, 0});
    builder.add_vertex({1, 0, 0});
    builder.add_cell(idmesh::CellType::segment,
        std::array<idmesh::Index, 2> {0, 1});
    const auto mesh = std::move(builder).finalize();
    try {
        idmesh::kokkos::DeviceMesh device(mesh);
    }
    catch (const std::logic_error&) {
        return 0; // No Kokkos::initialize has occurred in this process.
    }
    return 1;
}
