#include "idmesh/kokkos.hpp"
#include <Kokkos_Core.hpp>
#include <array>
#include <utility>

int main(int argc, char** argv)
{
    Kokkos::initialize(argc, argv);
    int status = 1;
    {
        idmesh::MeshBuilder builder(2, 2);
        builder.add_vertex({0,0,0});
        builder.add_vertex({1,0,0});
        builder.add_vertex({0,1,0});
        std::array<idmesh::Index, 3> vertices{0,1,2};
        builder.add_cell(idmesh::CellType::triangle, vertices);
        auto mesh = std::move(builder).finalize();
        status = idmesh::kokkos::incidence_count(mesh, 2, 1) == 3 ? 0 : 1;
    }
    Kokkos::finalize();
    return status;
}
