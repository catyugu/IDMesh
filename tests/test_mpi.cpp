#include <idmesh/mpi.hpp>
#include <mpi.h>

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
using idmesh::Index;
void require(bool condition,const char* message)
{
    if(!condition)throw std::runtime_error(message);
}
idmesh::Mesh source_mesh()
{
    idmesh::MeshBuilder builder(2,2);
    for(const auto& p:std::array<std::array<double,3>,6>{{
        {0,0,0},{1,0,0},{1,1,0},{0,1,0},{2,0,0},{2,1,0}
    }})builder.add_vertex(p);
    const std::array<Index,4> a{0,1,2,3};
    const std::array<Index,3> b{1,4,2},c{4,5,2};
    builder.add_cell(idmesh::CellType::quadrilateral,a,41);
    builder.add_cell(idmesh::CellType::triangle,b,42);
    builder.add_cell(idmesh::CellType::triangle,c,43);
    const std::array<Index,2> boundary{0,1};
    builder.add_facet_tag(boundary,71);
    return std::move(builder).finalize(idmesh::TopologyOptions::finite_volume());
}
void run(const idmesh::Mesh& source,Index depth)
{
    const auto rank=idmesh::mpi::world_rank(),size=idmesh::mpi::world_size();
    const auto part=idmesh::mpi::partition_replicated(source,depth);
    const auto d=part.mesh.dimension();
    require(part.global_ids.size()==d+1 && part.owner_ranks.size()==d+1,
            "missing distributed identity arrays");
    const Index n=part.mesh.entity_count(d);
    require(n==part.global_ids[d].size(),"local cell-global identity mismatch");
    require(part.owned_cells<=n,"owned count exceeds local cells");
    for(Index i=0;i<n;++i) {
        const auto global_id=part.global_ids[d][i];
        const int owner=static_cast<int>(global_id*size/source.entity_count(d));
        require(part.owner_ranks[d][i]==owner,"inconsistent cell ownership");
        require((i<part.owned_cells)==(owner==rank),
                "owned/ghost cell ordering incorrect");
    }
    std::vector<double> values(n*3,-1);
    for(Index i=0;i<part.owned_cells;++i)
        for(Index component=0;component<3;++component)
            values[i*3+component]=100*part.global_ids[d][i]+component;
    part.exchange.exchange_cells(values,3);
    for(Index i=0;i<n;++i)
        for(Index component=0;component<3;++component)
            require(values[i*3+component]==100*part.global_ids[d][i]+component,
                    "halo exchange corrupted ghost field");
    std::vector<double> scalar(n,-1);
    for(Index i=0;i<part.owned_cells;++i)
        scalar[i]=part.global_ids[d][i]+10;
    part.exchange.exchange_cells(scalar);
    for(Index i=0;i<n;++i)
        require(scalar[i]==part.global_ids[d][i]+10,
                "second exchange plan invocation failed");

    for(Index k=0;k<d;++k) {
        require(part.global_ids[k].size()==part.mesh.entity_count(k),
                "shared entity global ID count");
        require(part.owner_ranks[k].size()==part.mesh.entity_count(k),
                "shared entity ownership count");
        for(Index e=0;e<part.mesh.entity_count(k);++e) {
            require(part.global_ids[k][e]<source.entity_count(k),
                    "invalid shared entity global ID");
            require(part.owner_ranks[k][e]>=0 && part.owner_ranks[k][e]<size,
                    "invalid subentity owner rank");
        }
    }
    int local_owned=static_cast<int>(part.owned_cells),sum=0;
    MPI_Allreduce(&local_owned,&sum,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    require(sum==static_cast<int>(source.entity_count(d)),
            "some cells lost or multiply owned");

    // Verify that globally shared edge identities agree between partitions.
    auto globalSupport=source.relation(d-1,d);
    for(Index facet=0;facet<source.entity_count(d-1);++facet) {
        if(globalSupport.row(facet).size()!=2)continue;
        int local_present=0,all_present=0;
        for(auto id:part.global_ids[d-1])
            if(id==facet){local_present=1;break;}
        MPI_Allreduce(&local_present,&all_present,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
        require(all_present>=2,"partition interface has inconsistent shared facet ID");
    }
}
}

int main(int argc,char** argv)
{
    MPI_Init(&argc,&argv);
    try {
        auto global=source_mesh();
        run(global,1);
        run(global,2);
        if(idmesh::mpi::world_rank()==0)
            std::cout<<"MPI: partition, shared identity, 1/2 ghost layers and exchanges OK\n";
        MPI_Finalize();
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"MPI test failed on rank "<<idmesh::mpi::world_rank()
                 <<": "<<e.what()<<'\n';
        MPI_Abort(MPI_COMM_WORLD,1);
        return 1;
    }
}
