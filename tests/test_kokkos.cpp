#include "idmesh/kokkos.hpp"
#include <Kokkos_Core.hpp>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <utility>

int main(int argc,char** argv)
{
    Kokkos::initialize(argc,argv);
    int status=1;
    try {
        idmesh::MeshBuilder b(3,3);
        for(int i=0;i<5;++i)b.add_vertex({double(i),0,0});
        const std::array<idmesh::Index,4> left{0,1,2,3},right{0,2,1,4};
        b.add_cell(idmesh::CellType::tetrahedron,left);
        b.add_cell(idmesh::CellType::tetrahedron,right);
        const auto mesh=std::move(b).finalize(idmesh::TopologyOptions::finite_volume());
        const auto host=mesh.relation(3,2);
        std::uint64_t count=0,checksum=0;
        std::int64_t signs=0;
        for(idmesh::Index c=0;c<host.rows;++c)
            count+=host.row(c).size();
        for(std::size_t i=0;i<host.indices.size();++i) {
            checksum+=(std::uint64_t(i)+1)*(host.indices[i]+1);
            signs+=host.signs[i];
        }
        idmesh::kokkos::DeviceRelation mirrored(mesh,3,2);
        if(mirrored.rows()!=host.rows ||
           mirrored.incidence_count()!=count ||
           mirrored.weighted_index_sum()!=checksum ||
           mirrored.boundary_coefficient_sum()!=signs ||
           mirrored.incidence_count()!=idmesh::kokkos::incidence_count(mesh,3,2))
            throw std::runtime_error("Kokkos device mirror/parallel reduction mismatch");
        status=0;
        std::cout<<"Kokkos: persistent topology mirror and parallel reductions OK\n";
    }catch(const std::exception& e){
        std::cerr<<"Kokkos test: "<<e.what()<<'\n';
    }
    Kokkos::finalize();
    return status;
}
