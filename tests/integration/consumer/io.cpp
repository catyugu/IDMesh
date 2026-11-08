#include <idmesh/io.hpp>
#include <idmesh/data.hpp>
#include <idmesh/geometry.hpp>
#include <idmesh/partition.hpp>
#include <idmesh/vtk.hpp>

#include <filesystem>
#include <type_traits>

// Compile this translation unit against installed public headers.
using MeshReader = idmesh::Mesh (*)(const std::string&, idmesh::BuildOptions);
using MeshWriter = void (*)(const idmesh::Mesh&, const std::string&);

static_assert(std::is_same_v<decltype(&idmesh::read_gmsh), MeshReader>);
static_assert(std::is_same_v<decltype(&idmesh::read_vtk_legacy), MeshReader>);
static_assert(std::is_same_v<decltype(&idmesh::read_medit), MeshReader>);
static_assert(std::is_same_v<decltype(&idmesh::write_vtk_legacy), MeshWriter>);
static_assert(std::is_same_v<decltype(&idmesh::write_medit), MeshWriter>);

bool installed_io_roundtrip(const idmesh::Mesh& mesh)
{
    // Call actual imported symbols, rather than stopping at type checks.
    const std::filesystem::path path("idmesh-installed-consumer.vtk");
    idmesh::write_vtk_legacy(mesh, path.string());
    const auto restored = idmesh::read_vtk_legacy(path.string());
    std::filesystem::remove(path);
    return restored.dimension() == mesh.dimension()
        && restored.entity_count(0) == mesh.entity_count(0)
        && restored.entity_count(1) == mesh.entity_count(1);
}
