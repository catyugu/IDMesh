#include <idmesh/mesh.hpp>
#include <array>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <utility>
using idmesh::Index;
void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void verify(std::span<const Index> local,std::span<const Index> canonical,idmesh::Orientation o){
    check(local.size()==canonical.size(),"arity mismatch");
    auto n=static_cast<std::ptrdiff_t>(local.size());
    for(std::ptrdiff_t i=0;i<n;++i){
        auto j=(static_cast<std::ptrdiff_t>(o.rotation)+(o.reversed?-i:i)+2*n)%n;
        check(local[static_cast<std::size_t>(j)]==canonical[static_cast<std::size_t>(i)],"orientation mismatch");
    }
}
void orientation_test(){
    idmesh::MeshBuilder b(2,2);
    for(int i=0;i<3;++i)b.add_vertex({double(i),0,0});
    const std::array<Index,3> v{0,1,2};
    b.add_cell(idmesh::CellType::triangle,v);
    auto mesh=std::move(b).finalize();
    auto edges=mesh.relation(2,1);
    const std::array<std::array<Index,2>,3> loops{{{0,1},{1,2},{2,0}}};
    for(std::size_t i=0;i<loops.size();++i)
        verify(loops[i],mesh.entity_vertices(1,edges.row(0)[i]),edges.row_orientations(0)[i]);
    check(edges.row_signs(0)[2]==-1,"missing reversed incidence");
}
void tetra_test(){
    idmesh::MeshBuilder b(3,3);
    for(int i=0;i<4;++i)b.add_vertex({double(i),0,0});
    const std::array<Index,4> v{0,1,2,3};
    b.add_cell(idmesh::CellType::tetrahedron,v);
    auto mesh=std::move(b).finalize();
    auto faces=mesh.relation(3,2);
    const std::array<std::array<Index,3>,4> loops{{{1,2,3},{0,3,2},{0,1,3},{0,2,1}}};
    for(std::size_t i=0;i<loops.size();++i){
        auto o=faces.row_orientations(0)[i];
        verify(loops[i],mesh.entity_vertices(2,faces.row(0)[i]),o);
        check(faces.row_signs(0)[i]==o.sign(),"incorrect boundary coefficient");
    }
}
void partial_test(){
    idmesh::MeshBuilder b(3,3);
    for(int i=0;i<4;++i)b.add_vertex({double(i),0,0});
    const std::array<Index,4> v{0,1,2,3};
    b.add_cell(idmesh::CellType::tetrahedron,v);
    auto opt=idmesh::TopologyOptions::cells_and_vertices();
    opt.edges=true;
    auto mesh=std::move(b).finalize(opt);
    check(mesh.has_entities(1)&&!mesh.has_entities(2),"wrong entity dimensions");
    check(mesh.entity_count(1)==6,"unique tetra edge count");
    check(mesh.has_relation(3,1)&&mesh.relation(3,1).row(0).size()==6,"missing cell-edge connectivity");
    for(auto s:mesh.relation(3,1).row_signs(0))check(s==0,"transitive coefficient must be zero");
    mesh.validate();
}
void gmsh_test(){
    auto path=std::filesystem::temp_directory_path()/"idmesh-bad-section.msh";
    {
        std::ofstream f(path);
        f<<"$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n3\n"
             "1 0 0 0\n2 1 0 0\n3 0 1 0\n"
             "$Elements\n1\n1 2 0 1 2 3\n$EndElements\n";
    }
    bool rejected=false;
    try{(void)idmesh::read_gmsh(path.string());}catch(const std::runtime_error&){rejected=true;}
    std::filesystem::remove(path);
    check(rejected,"missing section terminator accepted");
}
int main(){
    try{orientation_test();tetra_test();partial_test();gmsh_test();std::cout<<"contracts passed\n";}
    catch(const std::exception& ex){std::cerr<<ex.what()<<'\n';return 1;}
    return 0;
}
