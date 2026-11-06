#include "idmesh/kokkos.hpp"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <utility>

int main(int argc, char** argv)
{
    Kokkos::initialize(argc, argv);
    int status = 1;
    try {
        idmesh::MeshBuilder builder(3, 3);
        for (int i = 0; i < 5; ++i)
            builder.add_vertex({double(i), 0, 0});
        const std::array<idmesh::Index, 4> left {0, 1, 2, 3};
        const std::array<idmesh::Index, 4> right {0, 2, 1, 4};
        std::array<idmesh::Index, 6> extra {};
        for (idmesh::Index i = 0; i < extra.size(); ++i)
            extra[i] = builder.add_geometry_node({double(i + 5), 0, 0});
        builder.add_quadratic_cell(idmesh::CellType::tetrahedron, left, extra);
        builder.add_cell(idmesh::CellType::tetrahedron, right);
        const auto mesh = std::move(builder).finalize(
            idmesh::BuildOptions {.topology = idmesh::TopologyLayout::cell_facets});

        const auto host = mesh.relation(3, 2);
        idmesh::kokkos::DeviceRelation relation(mesh, 3, 2);
        if (relation.row_count() != host.rows || relation.entry_count() != host.targets.size()
            || !relation.oriented() || relation.offsets() == nullptr
            || relation.targets() == nullptr || relation.coefficients() == nullptr
            || relation.orientations() == nullptr)
            throw std::runtime_error("invalid device relation descriptor");

        const auto* targets = relation.targets();
        const auto* coefficients = relation.coefficients();
        const auto* orientations = relation.orientations();
        std::uint64_t device_checksum = 0;
        Kokkos::parallel_reduce("idmesh_test_relation_descriptor", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, static_cast<idmesh::Index>(relation.entry_count())), KOKKOS_LAMBDA(const idmesh::Index i, std::uint64_t& sum) { sum += (std::uint64_t(i) + 1) * (std::uint64_t(targets[i]) + 1)
                                                                                                                                                                                                                                             + std::uint64_t(coefficients[i] < 0 ? 3 : 7)
                                                                                                                                                                                                                                             + orientations[i].rotation; }, device_checksum);
        std::uint64_t host_checksum = 0;
        for (std::size_t i = 0; i < host.targets.size(); ++i)
            host_checksum += (std::uint64_t(i) + 1) * (std::uint64_t(host.targets[i]) + 1)
                + std::uint64_t(host.coefficients[i] < 0 ? 3 : 7)
                + host.orientations[i].rotation;
        if (device_checksum != host_checksum)
            throw std::runtime_error("device relation descriptor contents mismatch");

        idmesh::kokkos::DeviceMesh device_mesh(mesh);
        if (device_mesh.cell_count() != 2 || device_mesh.geometry_node_count() != 11
            || device_mesh.cell_vertex_entry_count() != 8
            || device_mesh.high_order_geometry_entry_count() != 10
            || device_mesh.dimension() != 3 || device_mesh.geometric_dimension() != 3
            || device_mesh.cell_types() == nullptr || device_mesh.cell_orders() == nullptr
            || device_mesh.cell_vertex_offsets() == nullptr
            || device_mesh.cell_vertices() == nullptr
            || device_mesh.high_order_geometry_offsets() == nullptr
            || device_mesh.high_order_geometry_nodes() == nullptr
            || device_mesh.geometry_x() == nullptr)
            throw std::runtime_error("invalid device mesh descriptor");

        const auto* x = device_mesh.geometry_x();
        double x_sum = 0;
        Kokkos::parallel_reduce("idmesh_test_geometry_descriptor", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, device_mesh.geometry_node_count()), KOKKOS_LAMBDA(const idmesh::Index i, double& sum) { sum += x[i]; }, x_sum);
        if (x_sum != 55.0)
            throw std::runtime_error("device geometry coordinates mismatch");

        idmesh::kokkos::DeviceField field(2, 2);
        const std::array<double, 4> values {1., 2., 3., 4.};
        field.upload(values);
        if (field.entity_count() != 2 || field.component_count() != 2 || field.size() != 4
            || field.data() == nullptr)
            throw std::runtime_error("invalid device field descriptor");
        const auto downloaded = field.download();
        if (!std::equal(downloaded.begin(), downloaded.end(), values.begin()))
            throw std::runtime_error("device field upload/download mismatch");

        status = 0;
        std::cout << "Kokkos: raw device topology, geometry and field descriptors OK\n";
    }
    catch (const std::exception& error) {
        std::cerr << "Kokkos test: " << error.what() << '\n';
    }
    Kokkos::finalize();
    return status;
}
