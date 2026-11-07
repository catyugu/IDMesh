#include <idmesh/data.hpp>
#include <idmesh/mesh.hpp>
#include <idmesh/vtk.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

    namespace data = idmesh::data;
    namespace vtk = idmesh::vtk;
    using idmesh::Index;

    void check(bool condition, const char* message)
    {
        if (!condition)
            throw std::runtime_error(message);
    }

    template <class Error, class Operation>
    void expect_error(Operation&& operation, const char* message)
    {
        bool caught = false;
        try {
            operation();
        }
        catch (const Error&) {
            caught = true;
        }
        check(caught, message);
    }

    std::filesystem::path temp(const char* name)
    {
        return std::filesystem::temp_directory_path() / name;
    }

    idmesh::Mesh mesh_with_tag()
    {
        idmesh::MeshBuilder builder(2, 2);
        builder.add_vertex({0, 0, 0});
        builder.add_vertex({1, 0, 0});
        builder.add_vertex({0, 1, 0});
        const std::array<Index, 3> vertices {0, 1, 2};
        const auto cell = builder.add_cell(
            idmesh::CellType::triangle, vertices);
        builder.add_cell_label(cell, 7);
        return std::move(builder).finalize();
    }

    template <class T>
    T scalar(const data::Array& array,
        std::size_t tuple, std::size_t component = 0)
    {
        check(sizeof(T) == data::scalar_size(array.type),
            "manifest test scalar width");
        T value {};
        const auto offset =
            (tuple * array.components + component) * sizeof(T);
        check(offset + sizeof(T) <= array.values.size(),
            "manifest test scalar bounds");
        std::memcpy(
            &value, array.values.data() + offset, sizeof(T));
        return value;
    }

    const data::MeshData& find_field(
        const data::State& state, const char* name)
    {
        const auto it = std::find_if(
            state.fields.begin(), state.fields.end(),
            [&](const data::MeshData& field) {
                return field.name == name;
            });
        check(it != state.fields.end(), "series field missing");
        return *it;
    }

    void pvd_round_trip()
    {
        const auto path = temp("idmesh-manifest.pvd");
        vtk::PvdCollection input;
        input.entries = {
            {0.0, 0, "group&0", "a&0.vtu"},
            {0.0, 1, "group&0", "a&1.vtu"},
            {1.5, 0, "", "later.vtu"}};
        vtk::write_pvd(input, path.string());
        const auto output = vtk::read_pvd(path.string());
        check(output.entries.size() == input.entries.size(),
            "PVD entry count");
        for (std::size_t i = 0; i < input.entries.size(); ++i) {
            check(output.entries[i].time == input.entries[i].time
                    && output.entries[i].part == input.entries[i].part
                    && output.entries[i].group == input.entries[i].group
                    && output.entries[i].file == input.entries[i].file,
                "PVD entry round-trip");
        }
        std::filesystem::remove(path);
    }

    struct SeriesState {
        std::array<double, 3> temperature {};
        std::array<std::int32_t, 1> material {};
        std::array<data::MeshDataView, 2> fields {};
        data::StateView view;

        SeriesState(double time, std::int64_t cycle,
            double base, std::int32_t tag)
            : temperature {base, base + 1, base + 2},
              material {tag},
              view {time, cycle, fields}
        {
            fields[0] = {
                "temperature",
                data::geometry_nodes(),
                data::all(),
                data::make_scalar_array(
                    std::span(temperature))};
            fields[1] = {
                "material",
                data::entities(2),
                data::all(),
                data::make_scalar_array(
                    std::span(material))};
        }
    };

    void series_round_trip()
    {
        const auto pvd = temp("idmesh-series.pvd");
        const auto first_vtu = pvd.parent_path()
            / "idmesh-series-000000.vtu";
        const auto second_vtu = pvd.parent_path()
            / "idmesh-series-000001.vtu";

        const auto mesh = mesh_with_tag();
        SeriesState first(0.25, 10, 100.0, 3);
        SeriesState second(1.5, 11, 200.0, 4);
        const std::array<data::StateView, 2> states {
            first.view, second.view};

        vtk::write_vtu_series(
            mesh, states, pvd.string(),
            {.encoding = vtk::VtuEncoding::inline_binary});

        check(std::filesystem::exists(first_vtu)
                && std::filesystem::exists(second_vtu),
            "series VTU files not created");

        const auto manifest = vtk::read_pvd(pvd.string());
        check(manifest.entries.size() == 2
                && manifest.entries[0].file
                    == "idmesh-series-000000.vtu"
                && manifest.entries[1].file
                    == "idmesh-series-000001.vtu",
            "series relative filenames");

        const auto datasets = vtk::read_vtu_series(pvd.string());
        check(datasets.size() == 2, "series dataset count");
        check(datasets[0].state.time
                && *datasets[0].state.time == 0.25
                && datasets[1].state.time
                && *datasets[1].state.time == 1.5,
            "series times");
        check(datasets[0].state.cycle
                && *datasets[0].state.cycle == 10
                && datasets[1].state.cycle
                && *datasets[1].state.cycle == 11,
            "series cycles");

        const auto& t0 = find_field(
            datasets[0].state, "temperature");
        const auto& t1 = find_field(
            datasets[1].state, "temperature");
        check(scalar<double>(t0.values, 0) == 100.0
                && scalar<double>(t1.values, 2) == 202.0,
            "series field values");

        std::filesystem::remove(pvd);
        std::filesystem::remove(first_vtu);
        std::filesystem::remove(second_vtu);
    }

    void pvtu_round_trip()
    {
        const auto path = temp("idmesh-parallel.pvtu");
        const auto mesh = mesh_with_tag();

        std::array<double, 3> point_values {1, 2, 3};
        std::array<std::int32_t, 2> cell_values {4, 5};
        const std::array<data::MeshDataView, 2> fields {{
            {"temperature<&",
                data::geometry_nodes(), data::all(),
                data::make_scalar_array(
                    std::span(point_values))},
            {"pair",
                data::entities(2), data::all(),
                data::make_interleaved_array(
                    std::span(cell_values), 2)}}};
        const data::StateView state {
            std::nullopt, std::nullopt, fields};
        const std::array<std::string, 2> pieces {
            "rank-0.vtu", "rank&1.vtu"};

        const auto manifest = vtk::make_pvtu_manifest(
            mesh, state, pieces, 2);
        check(manifest.ghost_level == 2
                && manifest.points_type
                    == data::ScalarType::float64,
            "PVTU base schema");
        check(manifest.point_data.size() == 1,
            "PVTU PointData schema");
        check(manifest.cell_data.size() == 2,
            "PVTU CellData + physical tag schema");

        vtk::write_pvtu(manifest, path.string());
        const auto loaded = vtk::read_pvtu(path.string());
        check(loaded.ghost_level == 2
                && loaded.pieces == std::vector<std::string>(
                    pieces.begin(), pieces.end()),
            "PVTU piece round-trip");
        check(loaded.point_data.size() == 1
                && loaded.point_data[0].name == "temperature<&"
                && loaded.point_data[0].type
                    == data::ScalarType::float64
                && loaded.point_data[0].components == 1,
            "PVTU point schema round-trip");

        const auto physical = std::find_if(
            loaded.cell_data.begin(), loaded.cell_data.end(),
            [](const vtk::ArraySchema& array) {
                return array.name == "physical_tag";
            });
        check(physical != loaded.cell_data.end()
                && physical->type == data::ScalarType::int32
                && physical->components == 1,
            "PVTU physical tag schema");

        std::filesystem::remove(path);
    }

    void strict_rejections()
    {
        const auto duplicate = temp("idmesh-duplicate.pvd");
        vtk::PvdCollection collection;
        collection.entries = {
            {0.0, 0, "", "a.vtu"},
            {0.0, 0, "", "b.vtu"}};
        expect_error<std::invalid_argument>(
            [&] {
                vtk::write_pvd(collection, duplicate.string());
            },
            "duplicate PVD key accepted");
        std::filesystem::remove(duplicate);

        const auto mesh = mesh_with_tag();
        SeriesState first(1.0, 0, 0.0, 1);
        SeriesState second(1.0, 1, 1.0, 2);
        const std::array<data::StateView, 2> states {
            first.view, second.view};
        const auto series = temp("idmesh-bad-series.pvd");
        expect_error<std::invalid_argument>(
            [&] {
                vtk::write_vtu_series(
                    mesh, states, series.string());
            },
            "non-increasing series time accepted");
        std::filesystem::remove(series);

        const data::StateView timed {
            1.0, std::nullopt, {}};
        const std::array<std::string, 1> pieces {"rank-0.vtu"};
        expect_error<std::invalid_argument>(
            [&] {
                (void)vtk::make_pvtu_manifest(
                    mesh, timed, pieces);
            },
            "PVTU helper silently dropped state time");

        std::array<double, 1> global_value {1.0};
        const std::array<data::MeshDataView, 1> global_fields {{
            {"global", data::whole_mesh(), data::all(),
                data::make_scalar_array(
                    std::span(global_value))}}};
        const data::StateView global_state {
            std::nullopt, std::nullopt, global_fields};
        expect_error<std::invalid_argument>(
            [&] {
                (void)vtk::make_pvtu_manifest(
                    mesh, global_state, pieces);
            },
            "PVTU helper silently dropped whole-Mesh field");

        vtk::PvtuManifest invalid;
        invalid.points_type = data::ScalarType::int32;
        invalid.pieces = {"rank-0.vtu"};
        const auto pvtu = temp("idmesh-invalid.pvtu");
        expect_error<std::invalid_argument>(
            [&] { vtk::write_pvtu(invalid, pvtu.string()); },
            "PVTU integer point coordinates accepted");
        std::filesystem::remove(pvtu);

        const auto pvtu_series = temp("idmesh-pvtu-series.pvd");
        vtk::PvdCollection parallel;
        parallel.entries = {
            {0.0, 0, "", "step-0.pvtu"}};
        vtk::write_pvd(parallel, pvtu_series.string());
        expect_error<std::runtime_error>(
            [&] {
                (void)vtk::read_vtu_series(
                    pvtu_series.string());
            },
            "serial series reader accepted PVTU entry");
        std::filesystem::remove(pvtu_series);
    }

} // namespace

int main()
{
    pvd_round_trip();
    series_round_trip();
    pvtu_round_trip();
    strict_rejections();
    return 0;
}
