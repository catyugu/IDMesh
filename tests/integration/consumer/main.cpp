#include <idmesh/mesh.hpp>

#include <array>
#include <utility>

bool installed_io_roundtrip(const idmesh::Mesh& mesh);

int main()
{
    idmesh::MeshBuilder builder(1, 1);
    builder.add_vertex({0, 0, 0});
    builder.add_vertex({1, 0, 0});
    const std::array<idmesh::Index, 2> nodes {0, 1};
    builder.add_cell(idmesh::CellType::segment, nodes);
    const auto mesh = std::move(builder).finalize();
    mesh.validate();
    return mesh.entity_count(1) == 1 && installed_io_roundtrip(mesh) ? 0 : 1;
}
