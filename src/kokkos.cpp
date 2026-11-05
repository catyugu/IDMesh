#include "idmesh/kokkos.hpp"
#include <Kokkos_Core.hpp>
#include <array>
#include <cmath>
#include <cstdint>
#include <span>
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

    struct DeviceCellField::Data {
        Index cells = 0;
        Index components = 0;
        Kokkos::View<double*, MemorySpace> values;
    };

    DeviceCellField::DeviceCellField(Index cells, Index components)
        : data_(std::make_shared<Data>())
    {
        check_runtime();
        if (components == 0 || static_cast<std::uint64_t>(cells) * components > invalid_index)
            throw std::invalid_argument("invalid device field shape");
        data_->cells = cells;
        data_->components = components;
        data_->values = Kokkos::View<double*, MemorySpace>(
            "idmesh_cell_field", static_cast<std::size_t>(cells) * components);
    }

    Index DeviceCellField::cells() const noexcept
    {
        return data_->cells;
    }
    Index DeviceCellField::components() const noexcept
    {
        return data_->components;
    }
    void DeviceCellField::upload(std::span<const double> values)
    {
        if (values.size() != data_->values.extent(0))
            throw std::invalid_argument("device field upload size mismatch");
        auto host = Kokkos::create_mirror_view(data_->values);
        for (std::size_t i = 0; i < values.size(); ++i)
            host(i) = values[i];
        Kokkos::deep_copy(data_->values, host);
        ExecutionSpace().fence();
    }
    std::vector<double> DeviceCellField::download() const
    {
        auto host = Kokkos::create_mirror_view(data_->values);
        Kokkos::deep_copy(host, data_->values);
        ExecutionSpace().fence();
        std::vector<double> values(host.extent(0));
        for (std::size_t i = 0; i < values.size(); ++i)
            values[i] = host(i);
        return values;
    }
    void* DeviceCellField::device_data() noexcept
    {
        return data_->values.data();
    }
    const void* DeviceCellField::device_data() const noexcept
    {
        return data_->values.data();
    }

    struct DeviceDiffusion::Data {
        Index cells = 0;
        Kokkos::View<Index*, MemorySpace> cell_offsets;
        Kokkos::View<Index*, MemorySpace> cell_facets;
        Kokkos::View<Index*, MemorySpace> facet_offsets;
        Kokkos::View<Index*, MemorySpace> facet_cells;
        Kokkos::View<double*, MemorySpace> weights;
    };

    DeviceDiffusion::DeviceDiffusion(
        const Mesh& mesh, std::span<const double> weights)
        : data_(std::make_shared<Data>())
    {
        check_runtime();
        const Index d = mesh.dimension();
        if (!mesh.has_relation(d, d - 1) || !mesh.has_relation(d - 1, d))
            throw std::invalid_argument("diffusion requires Cell-Facet/Facet-Cell");
        const auto downward = mesh.relation(d, d - 1);
        const auto support = mesh.relation(d - 1, d);
        if (weights.size() != mesh.entity_count(d - 1))
            throw std::invalid_argument("facet transmissibility count mismatch");
        for (Index f = 0; f < support.rows; ++f)
            if (support.row(f).size() > 2)
                throw std::invalid_argument("nonmanifold facet in diffusion kernel");
        for (auto w : weights)
            if (!std::isfinite(w) || w < 0)
                throw std::invalid_argument("invalid diffusion coefficient");

        data_->cells = mesh.entity_count(d);
        std::vector<Index> co(downward.rows + 1), fo(support.rows + 1);
        for (Index c = 0; c < downward.rows; ++c)
            co[c + 1] = co[c] + static_cast<Index>(downward.row(c).size());
        for (Index f = 0; f < support.rows; ++f)
            fo[f + 1] = fo[f] + static_cast<Index>(support.row(f).size());

        const auto create_index_view = [](const char* name,
                                           std::span<const Index> values) {
            Kokkos::View<Index*, MemorySpace> device(name, values.size());
            auto host = Kokkos::create_mirror_view(device);
            for (std::size_t i = 0; i < values.size(); ++i)
                host(i) = values[i];
            Kokkos::deep_copy(device, host);
            return device;
        };
        data_->cell_offsets = create_index_view("idmesh_diff_co",
            std::span<const Index>(co));
        data_->cell_facets = create_index_view("idmesh_diff_cf",
            downward.indices);
        data_->facet_offsets = create_index_view("idmesh_diff_fo",
            std::span<const Index>(fo));
        data_->facet_cells = create_index_view("idmesh_diff_fc",
            support.indices);
        data_->weights = Kokkos::View<double*, MemorySpace>(
            "idmesh_diff_weights", weights.size());
        auto host = Kokkos::create_mirror_view(data_->weights);
        for (std::size_t i = 0; i < weights.size(); ++i)
            host(i) = weights[i];
        Kokkos::deep_copy(data_->weights, host);
        ExecutionSpace().fence();
    }

    void DeviceDiffusion::apply(
        const DeviceCellField& input, DeviceCellField& output) const
    {
        if (input.cells() != data_->cells || output.cells() != data_->cells || input.components() != 1 || output.components() != 1)
            throw std::invalid_argument("diffusion requires one scalar per cell");
        const auto cell_offsets = data_->cell_offsets;
        const auto cell_facets = data_->cell_facets;
        const auto facet_offsets = data_->facet_offsets;
        const auto facet_cells = data_->facet_cells;
        const auto weight = data_->weights;
        const auto* x = static_cast<const double*>(input.device_data());
        auto* y = static_cast<double*>(output.device_data());
        Kokkos::parallel_for("idmesh_conservative_diffusion", Kokkos::RangePolicy<ExecutionSpace>(0, data_->cells), KOKKOS_LAMBDA(const Index cell) {
                double residual = 0;
                for (Index i = cell_offsets(cell);
                     i < cell_offsets(cell + 1); ++i) {
                    const Index facet = cell_facets(i);
                    const Index first = facet_offsets(facet);
                    const Index last = facet_offsets(facet + 1);
                    if (last - first != 2)
                        continue;
                    const Index a = facet_cells(first);
                    const Index b = facet_cells(first + 1);
                    const Index neighbor = a == cell ? b : a;
                    residual += weight(facet) * (x[cell] - x[neighbor]);
                }
                y[cell] = residual; });
        ExecutionSpace().fence();
    }
    Index DeviceDiffusion::cells() const noexcept
    {
        return data_->cells;
    }

    std::uint64_t incidence_count(const Mesh& mesh, Index from, Index to)
    {
        return DeviceRelation(mesh, from, to).incidence_count();
    }
} // namespace idmesh::kokkos
