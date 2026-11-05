#include "idmesh/kokkos.hpp"
#include <Kokkos_Core.hpp>
#include <cstdint>
#include <stdexcept>

namespace idmesh::kokkos {
namespace {
using MemorySpace=Kokkos::DefaultExecutionSpace::memory_space;
using ExecutionSpace=Kokkos::DefaultExecutionSpace;
void check_runtime() {
    if(!Kokkos::is_initialized())
        throw std::logic_error("initialize Kokkos before constructing a device mirror");
}
}
struct DeviceRelation::Data {
    Index rows=0;
    Kokkos::View<Index*,MemorySpace> offsets;
    Kokkos::View<Index*,MemorySpace> indices;
    Kokkos::View<std::int8_t*,MemorySpace> signs;
};

DeviceRelation::DeviceRelation(const Mesh& mesh,Index from,Index to)
    : data_(std::make_shared<Data>())
{
    check_runtime();
    const auto r=mesh.relation(from,to);
    data_->rows=r.rows;
    data_->offsets=Kokkos::View<Index*,MemorySpace>("idmesh_offset_mirror",r.rows+1);
    data_->indices=Kokkos::View<Index*,MemorySpace>("idmesh_incidence_mirror",r.indices.size());
    data_->signs=Kokkos::View<std::int8_t*,MemorySpace>("idmesh_orientation_mirror",r.signs.size());
    auto off=Kokkos::create_mirror_view(data_->offsets);
    auto idx=Kokkos::create_mirror_view(data_->indices);
    auto sign=Kokkos::create_mirror_view(data_->signs);
    for(Index row=0;row<=r.rows;++row)
        off(row)=r.offsets.empty()?row*r.stride:r.offsets[row];
    for(std::size_t i=0;i<r.indices.size();++i) {
        idx(i)=r.indices[i];
        sign(i)=r.signs[i];
    }
    Kokkos::deep_copy(data_->offsets,off);
    Kokkos::deep_copy(data_->indices,idx);
    Kokkos::deep_copy(data_->signs,sign);
    ExecutionSpace().fence();
}
Index DeviceRelation::rows() const noexcept {
    return data_->rows;
}
std::uint64_t DeviceRelation::incidence_count() const {
    auto offsets=data_->offsets;
    std::uint64_t result=0;
    Kokkos::parallel_reduce("idmesh_batch_count",
        Kokkos::RangePolicy<ExecutionSpace>(0,data_->rows),
        KOKKOS_LAMBDA(const Index c,std::uint64_t& sum) {
            sum+=offsets(c+1)-offsets(c);
        },result);
    return result;
}
std::uint64_t DeviceRelation::weighted_index_sum() const {
    auto ids=data_->indices;
    std::uint64_t result=0;
    Kokkos::parallel_reduce("idmesh_batch_weighted_indices",
        Kokkos::RangePolicy<ExecutionSpace>(0,static_cast<Index>(ids.extent(0))),
        KOKKOS_LAMBDA(const Index i,std::uint64_t& sum) {
            sum+=(std::uint64_t(i)+1)*(std::uint64_t(ids(i))+1);
        },result);
    return result;
}
std::int64_t DeviceRelation::boundary_coefficient_sum() const {
    auto signs=data_->signs;
    std::int64_t result=0;
    Kokkos::parallel_reduce("idmesh_batch_oriented_coefficients",
        Kokkos::RangePolicy<ExecutionSpace>(0,static_cast<Index>(signs.extent(0))),
        KOKKOS_LAMBDA(const Index i,std::int64_t& sum) {
            sum+=signs(i);
        },result);
    return result;
}
std::uint64_t incidence_count(const Mesh& mesh,Index from,Index to) {
    return DeviceRelation(mesh,from,to).incidence_count();
}
} // namespace idmesh::kokkos
