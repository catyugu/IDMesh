#include <idmesh/mesh.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
using idmesh::Index;
using Point=std::array<double,3>;
Point sub(Point a,Point b) {
    return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};
}
Point cross(Point a,Point b) {
    return {a[1]*b[2]-a[2]*b[1],
            a[2]*b[0]-a[0]*b[2],
            a[0]*b[1]-a[1]*b[0]};
}
double dot(Point a,Point b) {
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}
double norm(Point a) {return std::sqrt(dot(a,a));}
Point center(const idmesh::Mesh& mesh,Index dim,Index id) {
    auto nodes=mesh.entity_vertices(dim,id);
    Point c{};
    for(auto v:nodes) {
        const auto p=mesh.coordinates()[v];
        for(int j=0;j<3;++j)c[j]+=p[j]/nodes.size();
    }
    return c;
}
int cell_face_sign(const idmesh::Mesh& mesh,Index cell,Index facet) {
    auto relation=mesh.relation(3,2);
    const auto row=relation.row(cell);
    const auto signs=relation.row_signs(cell);
    for(std::size_t i=0;i<row.size();++i)
        if(row[i]==facet)return signs[i];
    throw std::runtime_error("facet not incident to cell");
}
Point normal(const idmesh::Mesh& mesh,Index cell,Index face) {
    const auto v=mesh.entity_vertices(2,face);
    if(v.size()<3)throw std::runtime_error("invalid facet polygon");
    auto p0=mesh.coordinates()[v[0]];
    auto p1=mesh.coordinates()[v[1]];
    auto p2=mesh.coordinates()[v[2]];
    auto n=cross(sub(p1,p0),sub(p2,p0));
    const int sign=cell_face_sign(mesh,cell,face);
    for(auto& x:n)x*=sign;
    return n;
}
void require(bool value,const char* message) {
    if(!value)throw std::runtime_error(message);
}
}
int main()
{
    try {
        idmesh::MeshBuilder builder(3,3);
        for(const auto p:std::array<Point,5>{{
            {0,0,0},{1,0,0},{0,1,0},{0,0,1},{0,0,-1}
        }})builder.add_vertex(p);
        const std::array<Index,4> a{0,1,2,3},b{0,2,1,4};
        builder.add_cell(idmesh::CellType::tetrahedron,a);
        builder.add_cell(idmesh::CellType::tetrahedron,b);
        const std::array<Index,3> boundary{1,2,3};
        builder.add_facet_tag(boundary,7);
        const auto mesh=std::move(builder).finalize(
            idmesh::TopologyOptions::finite_volume());
        require(mesh.entity_count(1)==0,"FVM must omit the edge layer");
        require(mesh.has_relation(3,2)&&mesh.has_relation(2,3),
                "FVM facet relations missing");
        const auto support=mesh.relation(2,3);
        const std::array<double,2> temperature{12,19};
        std::array<double,2> residual{};
        std::array<std::array<double,2>,2> stiffness{};
        Index internal=0,boundary_count=0;
        for(Index face=0;face<mesh.entity_count(2);++face) {
            const auto adjacency=support.row(face);
            require(adjacency.size()==1||adjacency.size()==2,"nonmanifold test facet");
            if(adjacency.size()==2) {
                const Index c0=adjacency[0],c1=adjacency[1];
                const auto n0=normal(mesh,c0,face),n1=normal(mesh,c1,face);
                for(int j=0;j<3;++j)
                    require(std::abs(n0[j]+n1[j])<1e-12,
                            "internal face normals are not opposed");
                const double area=norm(n0)/2;
                const double distance=norm(sub(center(mesh,3,c0),center(mesh,3,c1)));
                const double weight=area/distance;
                const double flux=weight*(temperature[c0]-temperature[c1]);
                residual[c0]+=flux;
                residual[c1]-=flux;
                stiffness[c0][c0]+=weight;
                stiffness[c0][c1]-=weight;
                stiffness[c1][c0]-=weight;
                stiffness[c1][c1]+=weight;
                ++internal;
            } else {
                const Index cell=adjacency[0];
                const auto n=normal(mesh,cell,face);
                require(dot(n,sub(center(mesh,2,face),center(mesh,3,cell)))>0,
                        "physical boundary facet normal is not outward");
                if(mesh.facet_tag(face)==7)
                    require(cell==0,"boundary facet label attached to wrong cell");
                ++boundary_count;
            }
        }
        require(internal==1 && boundary_count==6,"unexpected facet topology");
        require(std::abs(residual[0]+residual[1])<1e-12,
                "internal flux failed local conservation");
        for(int i=0;i<2;++i)
            require(std::abs(stiffness[i][0]+stiffness[i][1])<1e-12,
                    "assembled diffusion operator is not conservative");
        require(stiffness[0][1]<0 && stiffness[0][1]==stiffness[1][0],
                "diffusion matrix is not symmetric with negative off-diagonals");
        std::cout<<"FVM: 3D reduced topology, conservative diffusion and normals OK\n";
    }catch(const std::exception& e) {
        std::cerr<<"FVM example failure: "<<e.what()<<'\n';
        return 1;
    }
}
