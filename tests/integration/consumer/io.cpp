#include <idmesh/io.hpp>

#include <type_traits>

// Compile this translation unit against installed public headers.
using MeshReader = idmesh::Mesh (*)(const std::string&, idmesh::BuildOptions);
using MeshWriter = void (*)(const idmesh::Mesh&, const std::string&);

static_assert(std::is_same_v<decltype(&idmesh::read_gmsh), MeshReader>);
static_assert(std::is_same_v<decltype(&idmesh::read_vtk_legacy), MeshReader>);
static_assert(std::is_same_v<decltype(&idmesh::read_medit), MeshReader>);
static_assert(std::is_same_v<decltype(&idmesh::write_vtk_legacy), MeshWriter>);
static_assert(std::is_same_v<decltype(&idmesh::write_medit), MeshWriter>);
