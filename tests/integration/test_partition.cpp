#include "test.hpp"

#include <idmesh/partition.hpp>

#include <array>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
    idmesh::Mesh make_mesh(idmesh::BuildOptions options = {})
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
        return std::move(builder).finalize(options);
    }
}

namespace {
void partition_shards()
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
            input.clear();
            input.seekg(0);
            const std::string contents {
                std::istreambuf_iterator<char>(input),
                std::istreambuf_iterator<char>()};
            require(contents.find("\nENTITIES ") != std::string::npos
                    && contents.find("\nHALOS ") != std::string::npos,
                "execution-ready shard metadata is missing");
        }
        std::filesystem::remove(path);
    }

    // The shard writer accepts ownership produced outside IDMesh. This is the
    // external partitioner boundary; no third-party partitioner type enters
    // the public SDK.
    const std::array<int, 4> external_owners {0, 1, 1, 0};
    const auto external_prefix = (std::filesystem::temp_directory_path()
        / "idmesh-external-partition").string();
    idmesh::write_partition_shards(
        mesh, external_owners, 0, external_prefix);
    for (int rank = 0; rank < 2; ++rank) {
        const auto path
            = external_prefix + "." + std::to_string(rank) + ".idm";
        std::ifstream input(path);
        const std::string contents {
            std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>()};
        require(contents.find("\nCELLS 2\n") != std::string::npos,
            "externally supplied ownership was not honored");
        input.close();
        std::filesystem::remove(path);
    }
}
void partial_topology_shards()
{
    const auto full = make_mesh();
    const auto owners = idmesh::partition_graph(full, 2);
    const auto prefix = (std::filesystem::temp_directory_path()
        / "idmesh-partial-topology").string();

    const auto contents = [](const std::string& filename) {
        std::ifstream input(filename);
        require(static_cast<bool>(input), "missing partition shard output");
        return std::string(std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>());
    };

    for (idmesh::Index ghost = 0; ghost <= 2; ++ghost) {
        const auto base = prefix + "-base-" + std::to_string(ghost);
        idmesh::write_partition_shards(full, owners, ghost, base);
        for (const auto layout : {
                 idmesh::TopologyLayout::cell_vertices,
                 idmesh::TopologyLayout::cell_facets,
                 idmesh::TopologyLayout::complete}) {
            for (const auto relations : {
                     idmesh::RelationDirection::downward,
                     idmesh::RelationDirection::bidirectional}) {
                const auto mesh = make_mesh({layout, relations});
                const auto candidate = prefix + "-partial";
                idmesh::write_partition_shards(
                    mesh, owners, ghost, candidate);
                for (int rank = 0; rank < 2; ++rank) {
                    const auto suffix = "." + std::to_string(rank) + ".idm";
                    require(contents(base + suffix)
                            == contents(candidate + suffix),
                        "partial topology changed shard identity or halo");
                    std::filesystem::remove(candidate + suffix);
                }
            }
        }
        for (int rank = 0; rank < 2; ++rank)
            std::filesystem::remove(
                base + "." + std::to_string(rank) + ".idm");
    }

    const std::array<int, 4> sparse_ranks {0, 1000000000, 0, 0};
    expect_error<std::invalid_argument>(
        [&] {
            idmesh::write_partition_shards(
                full, sparse_ranks, 0, prefix + "-invalid");
        },
        "sparse rank IDs attempted oversized shard allocation");
}

} // namespace

int main()
{
    return idmesh::test::run({
        {"partition shard pipeline", partition_shards},
        {"partial topology shards", partial_topology_shards},
    });
}
