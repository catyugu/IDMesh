#include <idmesh/partition.hpp>

#include <array>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
    void require(bool condition, const char* message)
    {
        if (!condition)
            throw std::runtime_error(message);
    }

    idmesh::Mesh make_mesh()
    {
        idmesh::MeshBuilder builder(2, 2);
        for (const auto& p : std::array<std::array<double, 3>, 9> {{
                 {{0, 0, 0}}, {{1, 0, 0}}, {{2, 0, 0}},
                 {{0, 1, 0}}, {{1, 1, 0}}, {{2, 1, 0}},
                 {{0, 2, 0}}, {{1, 2, 0}}, {{2, 2, 0}}}})
            builder.add_vertex(p);
        for (const auto& c : std::array<std::array<idmesh::Index, 4>, 4> {{
                 {{0, 1, 4, 3}}, {{4, 5, 8, 7}},
                 {{1, 2, 5, 4}}, {{3, 4, 7, 6}}}})
            builder.add_cell(idmesh::CellType::quadrilateral, c);
        return std::move(builder).finalize();
    }
}

int main()
{
    const auto mesh = make_mesh();
    const auto owners = idmesh::partition_graph(mesh, 2);
    require(owners.size() == mesh.entity_count(mesh.dimension()),
        "partition result size mismatch");

    std::array<int, 2> counts {};
    for (int owner : owners) {
        require(owner == 0 || owner == 1, "invalid partition owner");
        ++counts[owner];
    }
    require(counts[0] == 2 && counts[1] == 2, "partition is not balanced");

    const auto prefix = (std::filesystem::temp_directory_path()
        / "idmesh-core-partition").string();
    idmesh::write_partition_shards(mesh, owners, 1, prefix);
    for (int rank = 0; rank < 2; ++rank) {
        const auto path = prefix + "." + std::to_string(rank) + ".idm";
        {
            std::ifstream input(path);
            std::string magic;
            int version = 0;
            require(static_cast<bool>(input >> magic >> version)
                    && magic == "IDMESH_SHARD" && version == 3,
                "invalid core shard output");
        }
        std::filesystem::remove(path);
    }
    return 0;
}
