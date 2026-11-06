#include "idmesh/kokkos.hpp"
#include "contracts.hpp"
#include <Kokkos_Core.hpp>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace idmesh::kokkos {
    static_assert(std::is_trivially_copyable_v<Orientation>);

    namespace {
        using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
        using ExecutionSpace = Kokkos::DefaultExecutionSpace;
        void check_runtime()
        {
            ::idmesh::detail::require_state(Kokkos::is_initialized(),
                "initialize Kokkos before constructing a device mirror");
        }
    }
    struct DeviceRelation::Data {
        Index rows = 0;
        std::size_t entries = 0;
        bool oriented = false;
        Kokkos::View<Index*, MemorySpace> offsets;
        Kokkos::View<Index*, MemorySpace> targets;
        Kokkos::View<std::int8_t*, MemorySpace> coefficients;
        Kokkos::View<Orientation*, MemorySpace> orientations;
    };

    DeviceRelation::DeviceRelation(const Mesh& mesh, Index from, Index to)
        : data_(std::make_shared<Data>())
    {
        check_runtime();
        const auto relation = mesh.relation(from, to);
        data_->rows = relation.rows;
        data_->entries = relation.targets.size();
        data_->oriented = relation.oriented();
        data_->offsets = Kokkos::View<Index*, MemorySpace>("idmesh_relation_offsets", relation.rows + 1);
        data_->targets = Kokkos::View<Index*, MemorySpace>("idmesh_relation_targets", relation.targets.size());
        data_->coefficients = Kokkos::View<std::int8_t*, MemorySpace>(
            "idmesh_relation_coefficients", relation.coefficients.size());
        data_->orientations = Kokkos::View<Orientation*, MemorySpace>(
            "idmesh_relation_orientations", relation.orientations.size());

        auto offsets = Kokkos::create_mirror_view(data_->offsets);
        auto targets = Kokkos::create_mirror_view(data_->targets);
        auto coefficients = Kokkos::create_mirror_view(data_->coefficients);
        auto orientations = Kokkos::create_mirror_view(data_->orientations);
        for (Index row = 0; row <= relation.rows; ++row)
            offsets(row) = relation.offsets.empty() ? row * relation.stride : relation.offsets[row];
        for (std::size_t i = 0; i < relation.targets.size(); ++i)
            targets(i) = relation.targets[i];
        for (std::size_t i = 0; i < relation.coefficients.size(); ++i)
            coefficients(i) = relation.coefficients[i];
        for (std::size_t i = 0; i < relation.orientations.size(); ++i)
            orientations(i) = relation.orientations[i];

        Kokkos::deep_copy(data_->offsets, offsets);
        Kokkos::deep_copy(data_->targets, targets);
        Kokkos::deep_copy(data_->coefficients, coefficients);
        Kokkos::deep_copy(data_->orientations, orientations);
        ExecutionSpace().fence();
    }

    Index DeviceRelation::row_count() const noexcept { return data_->rows; }
    std::size_t DeviceRelation::entry_count() const noexcept { return data_->entries; }
    bool DeviceRelation::oriented() const noexcept { return data_->oriented; }
    const Index* DeviceRelation::offsets() const noexcept { return data_->offsets.data(); }
    const Index* DeviceRelation::targets() const noexcept { return data_->targets.data(); }
    const std::int8_t* DeviceRelation::coefficients() const noexcept
    {
        return data_->coefficients.data();
    }
    const Orientation* DeviceRelation::orientations() const noexcept
    {
        return data_->orientations.data();
    }

    struct DeviceMesh::Data {
        Index dimension = 0;
        Index geometric_dimension = 0;
        Index cells = 0;
        Index nodes = 0;
        Kokkos::View<CellType*, MemorySpace> cell_types;
        Kokkos::View<std::uint8_t*, MemorySpace> cell_orders;
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
        data_->dimension = d;
        data_->geometric_dimension = mesh.geometric_dimension();
        data_->cells = nc;
        data_->nodes = mesh.geometry_node_count();
        data_->cell_types = Kokkos::View<CellType*, MemorySpace>("idmesh_cell_types", nc);
        data_->cell_orders = Kokkos::View<std::uint8_t*, MemorySpace>("idmesh_cell_orders", nc);
        auto types = Kokkos::create_mirror_view(data_->cell_types);
        auto orders = Kokkos::create_mirror_view(data_->cell_orders);
        for (Index c = 0; c < nc; ++c) {
            types(c) = mesh.cell_types()[c];
            orders(c) = mesh.cell_order(c);
        }
        Kokkos::deep_copy(data_->cell_types, types);
        Kokkos::deep_copy(data_->cell_orders, orders);
        std::vector<Index> co(nc + 1), ci, go(nc + 1), gi;
        for (Index c = 0; c < nc; ++c) {
            const auto vertices = mesh.entity_vertices(d, c);
            ci.insert(ci.end(), vertices.begin(), vertices.end());
            co[c + 1] = static_cast<Index>(ci.size());
            if (mesh.cell_order(c) > 1) {
                const auto nodes = mesh.cell_geometry_nodes(c);
                gi.insert(gi.end(), nodes.begin(), nodes.end());
            }
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
    Index DeviceMesh::dimension() const noexcept { return data_->dimension; }
    Index DeviceMesh::geometric_dimension() const noexcept
    {
        return data_->geometric_dimension;
    }
    Index DeviceMesh::cell_count() const noexcept { return data_->cells; }
    Index DeviceMesh::geometry_node_count() const noexcept { return data_->nodes; }
    std::size_t DeviceMesh::cell_vertex_entry_count() const noexcept
    {
        return data_->corner_indices.extent(0);
    }
    std::size_t DeviceMesh::high_order_geometry_entry_count() const noexcept
    {
        return data_->geometry_indices.extent(0);
    }
    const CellType* DeviceMesh::cell_types() const noexcept
    {
        return data_->cell_types.data();
    }
    const std::uint8_t* DeviceMesh::cell_orders() const noexcept
    {
        return data_->cell_orders.data();
    }
    const Index* DeviceMesh::cell_vertex_offsets() const noexcept
    {
        return data_->corner_offsets.data();
    }
    const Index* DeviceMesh::cell_vertices() const noexcept
    {
        return data_->corner_indices.data();
    }
    const Index* DeviceMesh::high_order_geometry_offsets() const noexcept
    {
        return data_->geometry_offsets.data();
    }
    const Index* DeviceMesh::high_order_geometry_nodes() const noexcept
    {
        return data_->geometry_indices.data();
    }
    const double* DeviceMesh::geometry_x() const noexcept { return data_->x.data(); }
    const double* DeviceMesh::geometry_y() const noexcept { return data_->y.data(); }
    const double* DeviceMesh::geometry_z() const noexcept { return data_->z.data(); }

    struct DeviceField::Data {
        Index entities = 0;
        Index components = 0;
        Kokkos::View<double*, MemorySpace> values;
    };

    DeviceField::DeviceField(Index entities, Index components)
        : data_(std::make_shared<Data>())
    {
        check_runtime();
        ::idmesh::detail::require_argument(components != 0
            && static_cast<std::uint64_t>(entities) * components <= invalid_index,
            "invalid device field shape");
        data_->entities = entities;
        data_->components = components;
        data_->values = Kokkos::View<double*, MemorySpace>(
            "idmesh_cell_field", static_cast<std::size_t>(entities) * components);
    }

    Index DeviceField::entity_count() const noexcept
    {
        return data_->entities;
    }
    Index DeviceField::component_count() const noexcept
    {
        return data_->components;
    }
    void DeviceField::upload(std::span<const double> values)
    {
        ::idmesh::detail::require_argument(values.size() == data_->values.extent(0),
            "device field upload size mismatch");
        auto host = Kokkos::create_mirror_view(data_->values);
        for (std::size_t i = 0; i < values.size(); ++i)
            host(i) = values[i];
        Kokkos::deep_copy(data_->values, host);
        ExecutionSpace().fence();
    }
    std::vector<double> DeviceField::download() const
    {
        auto host = Kokkos::create_mirror_view(data_->values);
        Kokkos::deep_copy(host, data_->values);
        ExecutionSpace().fence();
        std::vector<double> values(host.extent(0));
        for (std::size_t i = 0; i < values.size(); ++i)
            values[i] = host(i);
        return values;
    }
    std::size_t DeviceField::size() const noexcept
    {
        return data_->values.extent(0);
    }
    double* DeviceField::data() noexcept
    {
        return data_->values.data();
    }
    const double* DeviceField::data() const noexcept
    {
        return data_->values.data();
    }
} // namespace idmesh::kokkos
