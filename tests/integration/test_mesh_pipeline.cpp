#include "test.hpp"

#include <idmesh/data.hpp>
#include <idmesh/mesh.hpp>
#include <idmesh/vtk.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <span>
#include <string>

namespace {

    template <class T>
    T scalar(const idmesh::data::Array& array, std::size_t tuple)
    {
        require(array.components == 1, "expected scalar field");
        require(array.type == idmesh::data::scalar_type_v<T>,
            "unexpected scalar type");
        require(tuple < array.tuples, "tuple out of range");
        T value {};
        std::memcpy(&value,
            array.values.data() + tuple * sizeof(T), sizeof(T));
        return value;
    }

    const idmesh::data::MeshData& field(
        const idmesh::data::State& state, std::string_view name)
    {
        const auto it = std::find_if(
            state.fields.begin(), state.fields.end(),
            [&](const auto& item) { return item.name == name; });
        require(it != state.fields.end(), "missing round-trip field");
        return *it;
    }

    void gmsh_to_vtu_pipeline()
    {
        namespace fs = std::filesystem;
        namespace data = idmesh::data;

        const auto source
            = fs::path(IDMESH_FIXTURE_DIR) / "two_triangles.msh";
        const auto output
            = fs::temp_directory_path() / "idmesh-mesh-pipeline.vtu";

        const auto mesh = idmesh::read_gmsh(source.string());
        require(mesh.dimension() == 2, "fixture dimension");
        require(mesh.entity_count(2) == 2, "fixture cell count");
        require(mesh.entity_count(0) == 4, "fixture vertex count");
        const auto first_labels = mesh.labels(2, 0);
        const auto second_labels = mesh.labels(2, 1);
        require(first_labels.size() == 1
                && first_labels.front().value == 9,
            "first physical tag");
        require(second_labels.size() == 1
                && second_labels.front().value == 10,
            "second physical tag");

        std::array<double, 4> nodal_value {};
        for (idmesh::Index node = 0; node < mesh.geometry_node_count();
             ++node) {
            const auto x = mesh.geometry_node_position(node);
            nodal_value[node] = x[0] + 2.0 * x[1];
        }
        const std::array<std::int32_t, 2> material {9, 10};
        const std::array<data::MeshDataView, 2> fields {{
            {"phi", data::geometry_nodes(), data::all(),
                data::make_scalar_array(std::span(nodal_value))},
            {"material_copy", data::entities(2), data::all(),
                data::make_scalar_array(std::span(material))},
        }};
        const data::StateView state {0.5, std::int64_t {7}, fields};

        idmesh::vtk::write_vtu(mesh, state, output.string(),
            {.encoding = idmesh::vtk::VtuEncoding::appended_raw});
        const auto loaded = idmesh::vtk::read_vtu(output.string());
        fs::remove(output);

        require(loaded.mesh.dimension() == 2, "round-trip dimension");
        require(loaded.mesh.entity_count(2) == 2,
            "round-trip cell count");
        require(loaded.state.time && *loaded.state.time == 0.5,
            "round-trip time");
        require(loaded.state.cycle && *loaded.state.cycle == 7,
            "round-trip cycle");

        const auto& phi = field(loaded.state, "phi");
        require(phi.values.tuples == 4, "nodal field tuple count");
        require(scalar<double>(phi.values, 0) == 0.0,
            "nodal field value 0");
        require(scalar<double>(phi.values, 2) == 3.0,
            "nodal field value 2");

        const auto& copied = field(loaded.state, "material_copy");
        require(copied.values.tuples == 2, "cell field tuple count");
        require(scalar<std::int32_t>(copied.values, 0) == 9
                && scalar<std::int32_t>(copied.values, 1) == 10,
            "cell field values");
    }

} // namespace

int main()
{
    return idmesh::test::run({
        {"Gmsh -> field operation -> VTU -> read-back",
            gmsh_to_vtu_pipeline},
    });
}
