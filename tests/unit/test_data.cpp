#include "test.hpp"

#include <idmesh/data.hpp>
#include <idmesh/mesh.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

    using idmesh::Index;
    namespace data = idmesh::data;

    idmesh::Mesh make_mesh(
        idmesh::TopologyLayout topology
            = idmesh::TopologyLayout::complete)
    {
        idmesh::MeshBuilder builder(2, 2);
        builder.add_vertex({0, 0, 0});
        builder.add_vertex({1, 0, 0});
        builder.add_vertex({0, 1, 0});

        const std::array<Index, 3> corners {0, 1, 2};
        const std::array<std::array<double, 3>, 3> positions {{
            {{0.5, 0, 0}},
            {{0.5, 0.5, 0}},
            {{0, 0.5, 0}}}};
        std::array<Index, 3> extras {};
        for (std::size_t i = 0; i < positions.size(); ++i)
            extras[i] = builder.add_geometry_node(positions[i]);
        builder.add_quadratic_cell(
            idmesh::CellType::triangle, corners, extras);

        return std::move(builder).finalize(
            {.topology = topology,
                .relations = idmesh::RelationDirection::bidirectional});
    }

    void scalar_types()
    {
        static_assert(data::scalar_type_v<std::int8_t>
            == data::ScalarType::int8);
        static_assert(data::scalar_type_v<std::uint64_t>
            == data::ScalarType::uint64);
        static_assert(data::scalar_type_v<float>
            == data::ScalarType::float32);
        static_assert(data::scalar_type_v<double>
            == data::ScalarType::float64);

        check(data::scalar_size(data::ScalarType::int16) == 2,
            "int16 size");
        check(data::scalar_size(data::ScalarType::float64) == 8,
            "float64 size");
    }

    void dense_and_subset_data()
    {
        const auto mesh = make_mesh();

        const std::array<double, 1> temperature {312.5};
        data::MeshDataView cell_temperature {
            "temperature",
            data::entities(2),
            data::all(),
            data::make_scalar_array(std::span(temperature))};
        data::validate(mesh, cell_temperature);

        const std::array<float, 9> velocity {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9};
        data::MeshDataView vertex_velocity {
            "velocity",
            data::entities(0),
            data::all(),
            data::make_interleaved_array(
                std::span(velocity), 3)};
        data::validate(mesh, vertex_velocity);

        const std::array<Index, 2> boundary_edges {0, 2};
        const std::array<double, 6> flux {
            1, 0, 0,
            2, 0, 0};
        data::MeshDataView boundary_flux {
            "heat_flux",
            data::entities(1),
            data::select(boundary_edges),
            data::make_interleaved_array(
                std::span(flux), 3)};
        data::validate(mesh, boundary_flux);

        const std::array<double, 6> nodal_values {
            1, 2, 3, 4, 5, 6};
        data::MeshDataView geometry_field {
            "geometry_value",
            data::geometry_nodes(),
            data::all(),
            data::make_scalar_array(
                std::span(nodal_values))};
        data::validate(mesh, geometry_field);

        const std::array<double, 1> reynolds {1000.0};
        data::MeshDataView global {
            "Re",
            data::whole_mesh(),
            data::all(),
            data::make_scalar_array(std::span(reynolds))};
        data::validate(mesh, global);

        data::EntitySetView wall {
            "wall", data::entities(1),
            data::select(boundary_edges)};
        data::validate(mesh, wall);

        const std::array fields {
            cell_temperature, vertex_velocity,
            boundary_flux, geometry_field, global};
        data::StateView state {
            0.25, std::int64_t {12}, fields};
        data::validate(mesh, state);
    }

    void soa_and_strided_data()
    {
        const auto mesh = make_mesh();

        const std::array<double, 3> x {1, 2, 3};
        const std::array<double, 3> y {4, 5, 6};
        const std::array<double, 3> z {7, 8, 9};
        const std::array<std::span<const double>, 3> planes {
            std::span<const double>(x),
            std::span<const double>(y),
            std::span<const double>(z)};
        data::MeshDataView soa {
            "soa",
            data::entities(0),
            data::all(),
            data::make_component_array<double>(planes)};
        data::validate(mesh, soa);

        struct Sample {
            double x;
            std::uint32_t marker;
            double y;
        };
        static_assert(std::is_standard_layout_v<Sample>);
        const std::array<Sample, 3> values {{
            {1, 10, 2},
            {3, 11, 4},
            {5, 12, 6}}};
        const auto storage
            = std::as_bytes(std::span(values));
        const std::array<std::size_t, 2> offsets {
            offsetof(Sample, x), offsetof(Sample, y)};
        data::MeshDataView strided {
            "strided",
            data::entities(0),
            data::all(),
            data::make_strided_array<double>(
                storage, values.size(), offsets,
                sizeof(Sample))};
        data::validate(mesh, strided);
    }

    void owning_views()
    {
        data::Array array;
        array.type = data::ScalarType::float64;
        array.tuples = 2;
        array.components = 2;
        const std::array<double, 4> values {1, 2, 3, 4};
        const auto bytes = std::as_bytes(std::span(values));
        array.values.assign(bytes.begin(), bytes.end());

        const auto view = array.view();
        check(view.tuples == 2
                && view.components.size() == 2,
            "owned array view shape");
        data::validate_array(view);

        data::MeshData field;
        field.name = "pair";
        field.location = data::entities(1);
        field.selection = data::SelectionKind::indices;
        field.indices = {0, 1};
        field.values = std::move(array);
        const auto mesh = make_mesh();
        data::validate(mesh, field.view());

        data::Array malformed;
        malformed.type = data::ScalarType::float64;
        malformed.tuples = 1;
        malformed.components = 2;
        malformed.values.resize(sizeof(double));
        expect_error<std::invalid_argument>(
            [&] { (void)malformed.view(); },
            "malformed owned array accepted");
    }

    void strict_validation()
    {
        const auto mesh = make_mesh();

        const std::array<double, 1> one {1.0};
        const auto scalar
            = data::make_scalar_array(std::span(one));

        expect_error<std::invalid_argument>(
            [&] {
                data::validate(mesh,
                    data::MeshDataView {
                        "", data::entities(2),
                        data::all(), scalar});
            },
            "empty field name accepted");

        expect_error<std::invalid_argument>(
            [&] {
                data::validate(mesh,
                    data::MeshDataView {
                        "bad-dim", data::entities(3),
                        data::all(), scalar});
            },
            "invalid entity dimension accepted");

        const auto reduced = make_mesh(
            idmesh::TopologyLayout::cell_vertices);
        expect_error<std::invalid_argument>(
            [&] {
                data::validate(reduced,
                    data::MeshDataView {
                        "omitted-edge", data::entities(1),
                        data::all(), scalar});
            },
            "omitted entity layer accepted");

        const std::array<Index, 2> unsorted {2, 0};
        const std::array<double, 2> two {1, 2};
        expect_error<std::invalid_argument>(
            [&] {
                data::validate(mesh,
                    data::MeshDataView {
                        "unsorted", data::entities(1),
                        data::select(unsorted),
                        data::make_scalar_array(
                            std::span(two))});
            },
            "unsorted selection accepted");

        const std::array<Index, 2> duplicate {1, 1};
        expect_error<std::invalid_argument>(
            [&] {
                data::validate(mesh,
                    data::MeshDataView {
                        "duplicate", data::entities(1),
                        data::select(duplicate),
                        data::make_scalar_array(
                            std::span(two))});
            },
            "duplicate selection accepted");

        const std::array<Index, 1> outside {3};
        expect_error<std::invalid_argument>(
            [&] {
                data::validate(mesh,
                    data::MeshDataView {
                        "outside", data::entities(1),
                        data::select(outside), scalar});
            },
            "out-of-range selection accepted");

        const std::array<Index, 1> mesh_index {0};
        expect_error<std::invalid_argument>(
            [&] {
                data::validate(mesh,
                    data::MeshDataView {
                        "global-subset", data::whole_mesh(),
                        data::select(mesh_index), scalar});
            },
            "indexed whole-mesh data accepted");

        expect_error<std::invalid_argument>(
            [&] {
                data::MeshDataView mismatch {
                    "mismatch", data::entities(0),
                    data::all(), scalar};
                data::validate(mesh, mismatch);
            },
            "tuple mismatch accepted");

        data::ArrayView no_components;
        no_components.type = data::ScalarType::float64;
        no_components.tuples = 1;
        expect_error<std::invalid_argument>(
            [&] { data::validate_array(no_components); },
            "componentless array accepted");

        const std::array<std::byte, 8> bytes {};
        data::ArrayView truncated;
        truncated.type = data::ScalarType::float64;
        truncated.tuples = 2;
        truncated.components.push_back(
            {bytes, 0, sizeof(double)});
        expect_error<std::invalid_argument>(
            [&] { data::validate_array(truncated); },
            "truncated component storage accepted");

        data::ArrayView overlapping_stride;
        overlapping_stride.type = data::ScalarType::float64;
        overlapping_stride.tuples = 2;
        overlapping_stride.components.push_back(
            {std::span<const std::byte>(bytes),
                0, sizeof(float)});
        expect_error<std::invalid_argument>(
            [&] { data::validate_array(overlapping_stride); },
            "undersized scalar stride accepted");

        expect_error<std::invalid_argument>(
            [&] {
                data::StateView state {
                    std::numeric_limits<double>::quiet_NaN(),
                    std::nullopt, {}};
                data::validate(mesh, state);
            },
            "nonfinite state time accepted");

        expect_error<std::invalid_argument>(
            [&] {
                (void)data::make_interleaved_array(
                    std::span(two), 0);
            },
            "zero component count accepted");

        const std::array<double, 2> plane_a {1, 2};
        const std::array<double, 1> plane_b {3};
        const std::array<std::span<const double>, 2> bad_planes {
            std::span<const double>(plane_a),
            std::span<const double>(plane_b)};
        expect_error<std::invalid_argument>(
            [&] {
                (void)data::make_component_array<double>(
                    bad_planes);
            },
            "mismatched SoA planes accepted");
    }

} // namespace

int main()
{
    return idmesh::test::run({
        {"scalar types", scalar_types},
        {"dense/subset data", dense_and_subset_data},
        {"SoA/strided data", soa_and_strided_data},
        {"owning views", owning_views},
        {"strict validation", strict_validation},
    });
}
