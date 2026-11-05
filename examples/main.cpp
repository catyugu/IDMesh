#include <idmesh/mesh.hpp>
#include <array>
#include <iostream>
#include <utility>
int main() { idmesh::MeshBuilder b(2,2); b.add_vertex({0,0,0}); b.add_vertex({1,0,0}); b.add_vertex({0,1,0}); std::array<idmesh::Index,3> v{0,1,2}; b.add_cell(idmesh::CellType::triangle,v); auto m=std::move(b).finalize(); std::cout << m.entity_count(1) << " edges\n"; }
