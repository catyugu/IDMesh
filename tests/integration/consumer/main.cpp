#include <array>
#include <idmesh/mesh.hpp>
#include <utility>
int main()
{
    idmesh::MeshBuilder builder(1, 1);
    builder.add_vertex({0, 0, 0});
    builder.add_vertex({1, 0, 0});
    std::array<idmesh::Index, 2> nodes {0, 1};
    builder.add_cell(idmesh::CellType::segment, nodes);
    return std::move(builder).finalize().entity_count(1) == 1 ? 0 : 1;
}
