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
#include <iterator>
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

    std::filesystem::path temporary(const char* name)
    {
        return std::filesystem::temp_directory_path() / name;
    }

    idmesh::Mesh quadratic_mesh()
    {
        idmesh::MeshBuilder builder(2, 2);
        builder.add_vertex({0, 0, 0});
        builder.add_vertex({1, 0, 0});
        builder.add_vertex({0, 1, 0});
        const std::array<Index, 3> corners {0, 1, 2};
        std::array<Index, 3> extra {
            builder.add_geometry_node({0.5, 0, 0}),
            builder.add_geometry_node({0.5, 0.5, 0}),
            builder.add_geometry_node({0, 0.5, 0})};
        const auto cell = builder.add_quadratic_cell(
            idmesh::CellType::triangle, corners, extra);
        builder.add_cell_label(cell, -7);
        return std::move(builder).finalize();
    }

    template <class T>
    T value_at(const data::Array& array,
        std::size_t tuple, std::size_t component = 0)
    {
        check(sizeof(T) == data::scalar_size(array.type),
            "array scalar width");
        check(tuple < array.tuples
                && component < array.components,
            "array scalar index");
        T value {};
        const auto offset
            = (tuple * array.components + component)
            * sizeof(T);
        std::memcpy(&value,
            array.values.data() + offset, sizeof(T));
        return value;
    }

    const data::MeshData& field(
        const data::State& state, const char* name)
    {
        const auto it = std::find_if(
            state.fields.begin(), state.fields.end(),
            [&](const data::MeshData& field) {
                return field.name == name;
            });
        check(it != state.fields.end(), "missing round-trip field");
        return *it;
    }

    void check_dataset(const data::Dataset& dataset)
    {
        const auto& mesh = dataset.mesh;
        check(mesh.dimension() == 2, "round-trip dimension");
        check(mesh.entity_count(2) == 1, "round-trip cell count");
        check(mesh.geometry_node_count() == 6,
            "round-trip geometry node count");
        check(mesh.cell_order(0) == 2, "round-trip cell order");
        check(mesh.cell_types()[0] == idmesh::CellType::triangle,
            "round-trip cell type");
        const auto geometry = mesh.cell_geometry_nodes(0);
        check(geometry.size() == 6, "round-trip geometry arity");
        for (Index i = 0; i < geometry.size(); ++i)
            check(geometry[i] == i,
                "round-trip high-order node ordering");

        const auto labels = mesh.labels(2, 0);
        check(labels.size() == 1 && labels.front().value == -7,
            "round-trip physical tag");

        check(dataset.state.time
                && *dataset.state.time == 1.25,
            "round-trip time");
        check(dataset.state.cycle
                && *dataset.state.cycle == 42,
            "round-trip cycle");

        const auto& temperature = field(
            dataset.state, "temperature");
        check(temperature.location.kind
                == data::LocationKind::geometry_node
                && temperature.values.type
                    == data::ScalarType::float64
                && temperature.values.tuples == 6
                && temperature.values.components == 1,
            "temperature metadata");
        for (std::size_t i = 0; i < 6; ++i)
            check(value_at<double>(temperature.values, i)
                    == 100.0 + double(i),
                "temperature values");

        const auto& velocity = field(
            dataset.state, "velocity");
        check(velocity.location.kind
                == data::LocationKind::geometry_node
                && velocity.values.type
                    == data::ScalarType::float32
                && velocity.values.tuples == 6
                && velocity.values.components == 3,
            "velocity metadata");
        for (std::size_t i = 0; i < 6; ++i)
            for (std::size_t c = 0; c < 3; ++c)
                check(value_at<float>(
                          velocity.values, i, c)
                        == float(i * 10 + c),
                    "velocity values");

        const auto& material = field(
            dataset.state, "material");
        check(material.location.kind
                == data::LocationKind::entity
                && material.location.dimension == 2
                && material.values.type
                    == data::ScalarType::int32
                && material.values.components == 2,
            "cell field metadata");
        check(value_at<std::int32_t>(material.values, 0, 0) == 3
                && value_at<std::int32_t>(
                       material.values, 0, 1)
                    == -4,
            "cell field values");

        const auto& run = field(dataset.state, "run_id");
        check(run.location.kind == data::LocationKind::mesh
                && run.values.type
                    == data::ScalarType::uint16
                && value_at<std::uint16_t>(run.values, 0)
                    == 11,
            "mesh field");
    }

    struct SampleState {
        std::array<double, 6> temperature {
            100, 101, 102, 103, 104, 105};
        std::array<float, 18> velocity {};
        std::array<std::int32_t, 2> material {3, -4};
        std::array<std::uint16_t, 1> run {11};
        std::array<data::MeshDataView, 4> fields {};
        data::StateView state;

        SampleState()
            : state {1.25, std::int64_t {42}, fields}
        {
            for (std::size_t i = 0; i < 6; ++i)
                for (std::size_t c = 0; c < 3; ++c)
                    velocity[i * 3 + c] = float(i * 10 + c);
            fields[0] = {
                "temperature", data::geometry_nodes(), data::all(),
                data::make_scalar_array(std::span(temperature))};
            fields[1] = {
                "velocity", data::geometry_nodes(), data::all(),
                data::make_interleaved_array(
                    std::span(velocity), 3)};
            fields[2] = {
                "material", data::entities(2), data::all(),
                data::make_interleaved_array(
                    std::span(material), 2)};
            fields[3] = {
                "run_id", data::whole_mesh(), data::all(),
                data::make_scalar_array(std::span(run))};
        }
    };

    void legacy_round_trip(vtk::LegacyEncoding encoding,
        const char* name)
    {
        const auto path = temporary(name);
        const auto mesh = quadratic_mesh();
        SampleState sample;
        vtk::write_legacy(mesh, sample.state,
            path.string(), encoding);
        auto dataset = vtk::read_legacy(path.string());
        check_dataset(dataset);

        std::ifstream in(path, std::ios::binary);
        std::string header, title, mode;
        std::getline(in, header);
        std::getline(in, title);
        std::getline(in, mode);
        check(mode == (encoding == vtk::LegacyEncoding::binary
                          ? "BINARY"
                          : "ASCII"),
            "legacy encoding header");
        std::filesystem::remove(path);
    }

    void vtu_round_trip(vtk::VtuEncoding encoding,
        const char* name)
    {
        const auto path = temporary(name);
        const auto mesh = quadratic_mesh();
        SampleState sample;
        vtk::write_vtu(mesh, sample.state, path.string(),
            {.encoding = encoding});
        auto dataset = vtk::read_vtu(path.string());
        check_dataset(dataset);

        std::ifstream in(path, std::ios::binary);
        const std::string contents {
            std::istreambuf_iterator<char>(in),
            std::istreambuf_iterator<char>()};
        if (encoding == vtk::VtuEncoding::ascii)
            check(contents.find("format=\"ascii\"")
                    != std::string::npos,
                "VTU ASCII encoding");
        else if (encoding == vtk::VtuEncoding::inline_binary)
            check(contents.find("format=\"binary\"")
                    != std::string::npos,
                "VTU inline binary encoding");
        else
            check(contents.find("<AppendedData encoding=\"raw\">_")
                    != std::string::npos,
                "VTU appended encoding");
        std::filesystem::remove(path);
    }

    void vtu_uint64()
    {
        const auto path = temporary("idmesh-vtu-u64.vtu");
        idmesh::MeshBuilder builder(1, 1);
        builder.add_vertex({0, 0, 0});
        builder.add_vertex({1, 0, 0});
        const std::array<Index, 2> vertices {0, 1};
        builder.add_cell(idmesh::CellType::segment, vertices);
        const auto mesh = std::move(builder).finalize();

        std::array<std::uint64_t, 1> value {
            std::uint64_t {1} << 40};
        const std::array<data::MeshDataView, 1> fields {{
            {"large_id", data::whole_mesh(), data::all(),
                data::make_scalar_array(std::span(value))}}};
        const data::StateView state {
            std::nullopt, std::nullopt, fields};
        vtk::write_vtu(mesh, state, path.string(),
            {.encoding = vtk::VtuEncoding::inline_binary});
        const auto dataset = vtk::read_vtu(path.string());
        const auto& large = field(dataset.state, "large_id");
        check(large.values.type == data::ScalarType::uint64
                && value_at<std::uint64_t>(large.values, 0)
                    == value[0],
            "VTU UInt64 round-trip");
        std::filesystem::remove(path);
    }

    void strict_rejections()
    {
        const auto mesh = quadratic_mesh();

        std::array<std::int64_t, 1> value {9};
        const std::array<data::MeshDataView, 1> int64_fields {{
            {"wide", data::whole_mesh(), data::all(),
                data::make_scalar_array(std::span(value))}}};
        const data::StateView int64_state {
            std::nullopt, std::nullopt, int64_fields};
        const auto legacy = temporary("idmesh-vtk-wide.vtk");
        expect_error<std::invalid_argument>(
            [&] {
                vtk::write_legacy(mesh, int64_state,
                    legacy.string());
            },
            "legacy VTK accepted portable-unsupported int64 field");
        std::filesystem::remove(legacy);

        idmesh::MeshBuilder builder(2, 2);
        builder.add_vertex({0, 0, 0});
        builder.add_vertex({1, 0, 0});
        builder.add_vertex({0, 1, 0});
        const std::array<Index, 3> triangle {0, 1, 2};
        const std::array<Index, 2> edge {0, 1};
        builder.add_cell(idmesh::CellType::triangle, triangle);
        builder.add_label(1, edge, 5);
        const auto lower = std::move(builder).finalize();
        const data::StateView empty {
            std::nullopt, std::nullopt, {}};
        const auto vtu = temporary("idmesh-vtu-lower.vtu");
        expect_error<std::invalid_argument>(
            [&] {
                vtk::write_vtu(
                    lower, empty, vtu.string());
            },
            "VTU silently encoded lower-dimensional label");
        std::filesystem::remove(vtu);

        const auto compressed
            = temporary("idmesh-vtu-compressed.vtu");
        {
            std::ofstream out(compressed);
            out << "<?xml version=\"1.0\"?>\n"
                   "<VTKFile type=\"UnstructuredGrid\" "
                   "compressor=\"vtkZLibDataCompressor\">"
                   "</VTKFile>\n";
        }
        expect_error<std::runtime_error>(
            [&] {
                (void)vtk::read_vtu(
                    compressed.string());
            },
            "compressed VTU was silently accepted");
        std::filesystem::remove(compressed);
    }

} // namespace

int main()
{
    legacy_round_trip(
        vtk::LegacyEncoding::ascii,
        "idmesh-data-ascii.vtk");
    legacy_round_trip(
        vtk::LegacyEncoding::binary,
        "idmesh-data-binary.vtk");
    vtu_round_trip(
        vtk::VtuEncoding::ascii,
        "idmesh-data-ascii.vtu");
    vtu_round_trip(
        vtk::VtuEncoding::inline_binary,
        "idmesh-data-inline.vtu");
    vtu_round_trip(
        vtk::VtuEncoding::appended_raw,
        "idmesh-data-appended.vtu");
    vtu_uint64();
    strict_rejections();
    return 0;
}
