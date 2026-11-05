#include "idmesh/kokkos.hpp"
#include <Kokkos_Core.hpp>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace idmesh::kokkos {
    namespace {
        using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
        using ExecutionSpace = Kokkos::DefaultExecutionSpace;
        void check_runtime()
        {
            if (!Kokkos::is_initialized())
                throw std::logic_error("initialize Kokkos before constructing a device mirror");
        }
    }
    struct DeviceRelation::Data {
        Index rows = 0;
        Kokkos::View<Index*, MemorySpace> offsets;
        Kokkos::View<Index*, MemorySpace> indices;
        Kokkos::View<std::int8_t*, MemorySpace> signs;
    };

    DeviceRelation::DeviceRelation(const Mesh& mesh, Index from, Index to)
        : data_(std::make_shared<Data>())
    {
        check_runtime();
        const auto r = mesh.relation(from, to);
        data_->rows = r.rows;
        data_->offsets = Kokkos::View<Index*, MemorySpace>("idmesh_offset_mirror", r.rows + 1);
        data_->indices = Kokkos::View<Index*, MemorySpace>("idmesh_incidence_mirror", r.indices.size());
        data_->signs = Kokkos::View<std::int8_t*, MemorySpace>("idmesh_orientation_mirror", r.signs.size());
        auto off = Kokkos::create_mirror_view(data_->offsets);
        auto idx = Kokkos::create_mirror_view(data_->indices);
        auto sign = Kokkos::create_mirror_view(data_->signs);
        for (Index row = 0; row <= r.rows; ++row)
            off(row) = r.offsets.empty() ? row * r.stride : r.offsets[row];
        for (std::size_t i = 0; i < r.indices.size(); ++i) {
            idx(i) = r.indices[i];
            sign(i) = r.signs[i];
        }
        Kokkos::deep_copy(data_->offsets, off);
        Kokkos::deep_copy(data_->indices, idx);
        Kokkos::deep_copy(data_->signs, sign);
        ExecutionSpace().fence();
    }
    Index DeviceRelation::rows() const noexcept
    {
        return data_->rows;
    }
    std::uint64_t DeviceRelation::incidence_count() const
    {
        auto offsets = data_->offsets;
        std::uint64_t result = 0;
        Kokkos::parallel_reduce("idmesh_batch_count", Kokkos::RangePolicy<ExecutionSpace>(0, data_->rows), KOKKOS_LAMBDA(const Index c, std::uint64_t& sum) { sum += offsets(c + 1) - offsets(c); }, result);
        return result;
    }
    std::uint64_t DeviceRelation::weighted_index_sum() const
    {
        auto ids = data_->indices;
        std::uint64_t result = 0;
        Kokkos::parallel_reduce("idmesh_batch_weighted_indices", Kokkos::RangePolicy<ExecutionSpace>(0, static_cast<Index>(ids.extent(0))), KOKKOS_LAMBDA(const Index i, std::uint64_t& sum) { sum += (std::uint64_t(i) + 1) * (std::uint64_t(ids(i)) + 1); }, result);
        return result;
    }
    std::int64_t DeviceRelation::boundary_coefficient_sum() const
    {
        auto signs = data_->signs;
        std::int64_t result = 0;
        Kokkos::parallel_reduce("idmesh_batch_oriented_coefficients", Kokkos::RangePolicy<ExecutionSpace>(0, static_cast<Index>(signs.extent(0))), KOKKOS_LAMBDA(const Index i, std::int64_t& sum) { sum += signs(i); }, result);
        return result;
    }

    struct DeviceMesh::Data {
        Index cells = 0;
        Index nodes = 0;
        Kokkos::View<Index*, MemorySpace> corner_offsets;
        Kokkos::View<Index*, MemorySpace> corner_indices;
        Kokkos::View<Index*, MemorySpace> geometry_offsets;
        Kokkos::View<Index*, MemorySpace> geometry_indices;
        Kokkos::View<double*, MemorySpace> x, y, z;
    };
    DeviceMesh::DeviceMesh(const Mesh& mesh)
        : data_(std::make_shared<Data>())
    {
        check_runtime();
        const auto d = mesh.dimension();
        const auto nc = mesh.entity_count(d);
        data_->cells = nc;
        data_->nodes = mesh.geometry_node_count();
        std::vector<Index> co(nc + 1), ci, go(nc + 1), gi;
        for (Index c = 0; c < nc; ++c) {
            const auto vertices = mesh.entity_vertices(d, c);
            ci.insert(ci.end(), vertices.begin(), vertices.end());
            co[c + 1] = static_cast<Index>(ci.size());
            const auto nodes = mesh.cell_geometry_nodes(c);
            gi.insert(gi.end(), nodes.begin(), nodes.end());
            go[c + 1] = static_cast<Index>(gi.size());
        }
        auto mirror_indices = [](auto& output, const auto& host,
                                  const char* label) {
            using View = std::remove_reference_t<decltype(output)>;
            output = View(std::string(label), host.size());
            auto temporary = Kokkos::create_mirror_view(output);
            for (std::size_t i = 0; i < host.size(); ++i)
                temporary(i) = host[i];
            Kokkos::deep_copy(output, temporary);
        };
        mirror_indices(data_->corner_offsets, co, "idmesh_corner_offsets");
        mirror_indices(data_->corner_indices, ci, "idmesh_corner_indices");
        mirror_indices(data_->geometry_offsets, go, "idmesh_geometry_offsets");
        mirror_indices(data_->geometry_indices, gi, "idmesh_geometry_indices");

        data_->x = Kokkos::View<double*, MemorySpace>("idmesh_x", data_->nodes);
        data_->y = Kokkos::View<double*, MemorySpace>("idmesh_y", data_->nodes);
        data_->z = Kokkos::View<double*, MemorySpace>("idmesh_z", data_->nodes);
        auto x = Kokkos::create_mirror_view(data_->x);
        auto y = Kokkos::create_mirror_view(data_->y);
        auto z = Kokkos::create_mirror_view(data_->z);
        for (Index i = 0; i < data_->nodes; ++i) {
            const auto p = mesh.geometry_node_position(i);
            x(i) = p[0];
            y(i) = p[1];
            z(i) = p[2];
        }
        Kokkos::deep_copy(data_->x, x);
        Kokkos::deep_copy(data_->y, y);
        Kokkos::deep_copy(data_->z, z);
        ExecutionSpace().fence();
    }
    Index DeviceMesh::cells() const noexcept { return data_->cells; }
    Index DeviceMesh::geometry_nodes() const noexcept { return data_->nodes; }
    std::array<double, 3> DeviceMesh::centroid_sum() const
    {
        auto offsets = data_->corner_offsets, indices = data_->corner_indices;
        auto x = data_->x, y = data_->y, z = data_->z;
        std::array<double, 3> answer {};
        for (int component = 0; component < 3; ++component) {
            double sum = 0;
            Kokkos::parallel_reduce("idmesh_batch_centroid", Kokkos::RangePolicy<ExecutionSpace>(0, data_->cells), KOKKOS_LAMBDA(const Index c, double& partial) {
                const Index begin=offsets(c),end=offsets(c+1);
                double value=0;
                for(Index i=begin;i<end;++i) {
                    const auto v=indices(i);
                    value+=component==0?x(v):(component==1?y(v):z(v));
                }
                partial+=value/(end-begin); }, sum);
            answer[component] = sum;
        }
        return answer;
    }
    std::uint64_t DeviceMesh::geometry_connectivity_checksum() const
    {
        auto ids = data_->geometry_indices;
        std::uint64_t result = 0;
        Kokkos::parallel_reduce("idmesh_batch_geometry_ids", Kokkos::RangePolicy<ExecutionSpace>(0, static_cast<Index>(ids.extent(0))), KOKKOS_LAMBDA(const Index i, std::uint64_t& sum) { sum += (std::uint64_t(i) + 1) * (std::uint64_t(ids(i)) + 1); }, result);
        return result;
    }

    std::uint64_t incidence_count(const Mesh& mesh, Index from, Index to)
    {
        return DeviceRelation(mesh, from, to).incidence_count();
    }
} // namespace idmesh::kokkos
