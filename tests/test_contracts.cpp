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
    bool rejected=false;
    try {(void)std::move(b).finalize(opt);}
    catch(const std::invalid_argument&){rejected=true;}
    check(rejected,"skip-layer topology accepted");
}
void valid_edge_face_test(){
    idmesh::MeshBuilder b(3,3);
    for(int i=0;i<4;++i)b.add_vertex({double(i),0,0});
    const std::array<Index,4> v{0,1,2,3};
    b.add_cell(idmesh::CellType::tetrahedron,v);
    auto opt=idmesh::TopologyOptions::cells_and_vertices();
    opt.edges=true;
    opt.faces=true;
    auto mesh=std::move(b).finalize(opt);
    check(mesh.has_entities(1)&&mesh.has_entities(2),"missing intermediate face layer");
    check(mesh.has_relation(3,2)&&mesh.has_relation(2,1),"missing adjacent incidence");
    check(!mesh.has_relation(3,1),"unexpected skip-layer relation");
    check(mesh.entity_count(1)==6,"wrong edge count");
    mesh.validate();
}
int main(){
    try{orientation_test();tetra_test();partial_test();valid_edge_face_test();std::cout<<"contracts passed\n";}
    catch(const std::exception& ex){std::cerr<<ex.what()<<'\n';return 1;}
    return 0;
}
