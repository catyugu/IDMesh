#include <idmesh/mpi.hpp>
#include <idmesh/partition.hpp>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

    using idmesh::Index;

    void require(bool condition, const char* message)
    {
        if (!condition)
            throw std::runtime_error(message);
    }

    idmesh::Mesh source()
    {
        idmesh::MeshBuilder builder(2, 2);
        constexpr Index columns = 4, rows = 3;
        for (Index y = 0; y <= rows; ++y)
            for (Index x = 0; x <= columns; ++x)
                builder.add_vertex({double(x), double(y), 0.0});

        // Checkerboard ordering is deliberately hostile to block partitioning.
        // Graph-based region growth should exploit the actual facet connectivity.
        for (Index parity = 0; parity < 2; ++parity) {
            for (Index y = 0; y < rows; ++y) {
                for (Index x = 0; x < columns; ++x) {
                    if ((x + y) % 2 != parity)
                        continue;
                    const Index bottom = y * (columns + 1) + x;
                    const std::array<Index, 4> corners {
                        bottom, bottom + 1, bottom + columns + 2, bottom + columns + 1};
                    if (x == 0 && y == 0) {
                        const std::array<std::array<double, 3>, 5> p {{{0.5, 0, 0}, {1, 0.5, 0}, {0.5, 1, 0},
                            {0, 0.5, 0}, {0.5, 0.5, 0}}};
                        std::array<Index, 5> extra {};
                        for (std::size_t j = 0; j < p.size(); ++j)
                            extra[j] = builder.add_geometry_node(p[j]);
                        const auto cell = builder.add_quadratic_cell(
                            idmesh::CellType::quadrilateral, corners, extra);
                        builder.add_cell_label(cell, 11);
                    }
                    else {
                        const auto cell = builder.add_cell(idmesh::CellType::quadrilateral, corners);
                        builder.add_cell_label(cell, 12);
                    }
                }
            }
        }
        const std::array<Index, 2> boundary {0, 1};
        const std::array<Index, 1> vertex {0};
        builder.add_label(1, boundary, 30);
        builder.add_label(0, vertex, 31);
        builder.add_cell_label(0, 40);
        builder.set_label_name(1, 30, "bottom wall");
        return std::move(builder).finalize();
    }

    idmesh::Mesh source_3d_mixed()
    {
        idmesh::MeshBuilder builder(3, 3);
        for (const auto& p : std::array<std::array<double, 3>, 12> {{
                 {{0, 0, 0}}, {{1, 0, 0}}, {{1, 1, 0}}, {{0, 1, 0}},
                 {{0, 0, 1}}, {{1, 0, 1}}, {{1, 1, 1}}, {{0, 1, 1}},
                 {{2, 0.5, 0}}, {{2, 0.5, 1}},
                 {{2, 1.5, 0.5}}, {{3, 1.5, 0.5}}}})
            builder.add_vertex(p);

        builder.add_cell(idmesh::CellType::hexahedron,
            std::array<Index, 8> {0, 1, 2, 3, 4, 5, 6, 7});
        builder.add_cell(idmesh::CellType::prism,
            std::array<Index, 6> {1, 2, 8, 5, 6, 9});
        builder.add_cell(idmesh::CellType::pyramid,
            std::array<Index, 5> {2, 8, 9, 6, 10});
        builder.add_cell(idmesh::CellType::tetrahedron,
            std::array<Index, 4> {8, 9, 10, 11});
        return std::move(builder).finalize();
    }

    void check_entity_exchange(const idmesh::mpi::PartitionedMesh& partition,
        Index dimension, int rank)
    {
        const Index count = partition.mesh().entity_count(dimension);
        require(partition.global_ids(dimension).size() == count
                && partition.owner_ranks(dimension).size() == count,
            "distributed entity metadata mismatch");
        const auto owned = partition.owned_entities(dimension);
        std::vector<bool> is_owned(count);
        for (Index entity : owned) {
            require(entity < count && !is_owned[entity],
                "invalid or duplicate owned entity index");
            require(partition.owner_ranks(dimension)[entity] == rank,
                "owned entity has a remote owner");
            is_owned[entity] = true;
        }
        for (Index entity = 0; entity < count; ++entity)
            require(is_owned[entity]
                    == (partition.owner_ranks(dimension)[entity] == rank),
                "owned entity set disagrees with owner ranks");

        const auto& plan = partition.exchange_plan(dimension);
        require(plan.local_entity_count() == count,
            "exchange plan local entity count mismatch");
        std::vector<double> values(std::size_t(count) * 2, -1);
        for (Index entity : owned) {
            const auto global = partition.global_ids(dimension)[entity];
            values[2 * entity] = double(global * 3 + dimension + 11);
            values[2 * entity + 1] = double(global * 5 + dimension + 17);
        }
        auto request = plan.make_request(2);
        request.start(values);
        request.wait();
        for (Index entity = 0; entity < count; ++entity) {
            const auto global = partition.global_ids(dimension)[entity];
            require(values[2 * entity] == double(global * 3 + dimension + 11)
                    && values[2 * entity + 1]
                        == double(global * 5 + dimension + 17),
                "entity halo exchange mismatch");
        }
    }

    int edge_cut(const idmesh::Mesh& mesh, const std::vector<int>& partition)
    {
        const Index dim = mesh.dimension();
        const auto support = mesh.relation(dim - 1, dim);
        int cuts = 0;
        for (Index f = 0; f < mesh.entity_count(dim - 1); ++f) {
            const auto cells = support[f].targets;
            if (cells.size() == 2 && partition[cells[0]] != partition[cells[1]])
                ++cuts;
        }
        return cuts;
    }

    struct Identity {
        std::uint64_t vertex[4] {};
        std::uint64_t global = 0;
        Index dimension = 0;
        Index arity = 0;
        int owner = -1;
    };

    void test_shard(const idmesh::mpi::Context& context,
        const std::string& prefix, int rank, int size, Index ghost_depth)
    {
        std::optional<idmesh::Mesh> root;
        std::vector<int> assignments;
        if (rank == 0) {
            root.emplace(source());
            assignments = idmesh::partition_graph(*root, size);
            std::vector<int> counts(size);
            for (int owner : assignments) {
                require(owner >= 0 && owner < size, "invalid graph partition assignment");
                ++counts[owner];
            }
            const auto [minimum, maximum] = std::minmax_element(counts.begin(), counts.end());
            require(*maximum - *minimum <= 1, "graph partitions are unbalanced");

            std::vector<int> blocks(assignments.size());
            for (Index i = 0; i < assignments.size(); ++i)
                blocks[i] = static_cast<int>(
                    std::uint64_t(i) * size / assignments.size());
            require(edge_cut(*root, assignments) < edge_cut(*root, blocks),
                "graph partition did not reduce the checkerboard edge cut");
            idmesh::write_partition_shards(
                *root, assignments, ghost_depth, prefix);
            // Destroy the global source before any Rank reads its shard.
            root.reset();
        }
        MPI_Barrier(MPI_COMM_WORLD);
        const auto local_path = prefix + "." + std::to_string(rank) + ".idm";
        auto partition = idmesh::mpi::read_partition_shard(context, local_path);
        partition.mesh().validate();
        const auto dim = partition.mesh().dimension();
        require(dim == 2
                && partition.global_ids(dim).size() == partition.mesh().entity_count(dim),
            "invalid distributed mesh dimensions");
        for (Index k = 0; k <= dim; ++k)
            if (partition.mesh().has_entities(k))
                check_entity_exchange(partition, k, rank);

        // Check local canonicalization of every materialized Edge or Vertex
        // by its global-key -> global-entity-ID mapping on all MPI ranks.
        std::vector<Identity> here;
        for (Index k = 0; k < dim; ++k) {
            for (Index e = 0; e < partition.mesh().entity_count(k); ++e) {
                Identity record {};
                record.dimension = k;
                record.global = partition.global_ids(k)[e];
                record.owner = partition.owner_ranks(k)[e];
                const auto vertices = partition.mesh().entity_vertices(k, e);
                record.arity = static_cast<Index>(vertices.size());
                for (std::size_t j = 0; j < vertices.size(); ++j)
                    record.vertex[j] = partition.global_ids(0)[vertices[j]];
                std::sort(record.vertex, record.vertex + record.arity);
                here.push_back(record);
            }
        }
        const int bytes = static_cast<int>(here.size() * sizeof(Identity));
        std::vector<int> counts(size), displacements(size);
        MPI_Allgather(&bytes, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        int total = 0;
        for (int p = 0; p < size; ++p) {
            displacements[p] = total;
            total += counts[p];
        }
        require(total % sizeof(Identity) == 0, "invalid identity gather");
        std::vector<Identity> all(total / sizeof(Identity));
        MPI_Allgatherv(here.data(), bytes, MPI_BYTE,
            all.data(), counts.data(), displacements.data(), MPI_BYTE,
            MPI_COMM_WORLD);
        using Key = std::pair<Index, std::vector<std::uint64_t>>;
        std::map<Key, std::pair<std::uint64_t, int>> identities;
        std::map<std::pair<Index, std::uint64_t>, Key> reverse;
        for (const auto& record : all) {
            Key key {record.dimension, std::vector<std::uint64_t>(record.vertex, record.vertex + record.arity)};
            auto [existing, inserted] = identities.try_emplace(
                key, record.global, record.owner);
            require(inserted || existing->second == std::pair {record.global, record.owner},
                "shared entity mismatch across rank-local shards");
            const auto name = std::pair {record.dimension, record.global};
            auto [back, unique] = reverse.try_emplace(name, key);
            require(unique || back->second == key, "global entity-ID collision");
        }
        const auto labels = partition.mesh().labels();
        (void)labels;
        if (std::any_of(partition.global_ids(dim).begin(),
                partition.global_ids(dim).end(),
                [](std::uint64_t id) { return id == 0; })) {
            require(partition.mesh().cell_order(
                        static_cast<Index>(std::find(partition.global_ids(dim).begin(),
                                               partition.global_ids(dim).end(), 0)
                            - partition.global_ids(dim).begin()))
                    == 2,
                "quadratic geometry lost in rank-local shard");
        }

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            for (int p = 0; p < size; ++p)
                std::filesystem::remove(
                    prefix + "." + std::to_string(p) + ".idm");
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    void test_complete_3d(const idmesh::mpi::Context& context,
        const std::string& prefix, int rank, int size)
    {
        if (rank == 0) {
            const auto mesh = source_3d();
            const auto assignments = idmesh::partition_graph(mesh, size);
            idmesh::write_partition_shards(mesh, assignments, 1, prefix);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        const auto local_path = prefix + "." + std::to_string(rank) + ".idm";
        auto partition = idmesh::mpi::read_partition_shard(
            context, local_path,
            idmesh::BuildOptions {
                .topology = idmesh::TopologyLayout::complete,
                .relations = idmesh::RelationDirection::bidirectional});
        partition.mesh().validate();
        require(partition.mesh().dimension() == 3
                && partition.mesh().has_entities(1)
                && partition.mesh().has_entities(2),
            "complete 3D topology was not materialized");
        for (Index k = 0; k <= 3; ++k)
            check_entity_exchange(partition, k, rank);

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0)
            for (int p = 0; p < size; ++p)
                std::filesystem::remove(
                    prefix + "." + std::to_string(p) + ".idm");
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    try {
        const auto context = idmesh::mpi::Context::world();
        const auto from_handle = idmesh::mpi::Context::from_fortran_handle(
            static_cast<std::int64_t>(MPI_Comm_c2f(MPI_COMM_WORLD)));
        const int rank = context.rank(), size = context.size();
        require(rank == from_handle.rank() && size == from_handle.size(),
            "MPI communicator handle conversion changed context");
        require(size >= 2 && size <= 3, "test requires two or three MPI ranks");
        const auto base = (std::filesystem::temp_directory_path()
            / ("idmesh-parallel-" + std::to_string(size)))
                              .string();
        test_shard(context, base, rank, size, 1);
        test_shard(context, base, rank, size, 2);
        test_complete_3d(context, base + "-3d", rank, size);
        if (rank == 0)
            std::cout << "MPI shard: graph cuts, rank-local loading, all-dimension "
                      << "identity/halo, orientation and quadratic geometry OK\n";
        MPI_Finalize();
        return 0;
    }
    catch (const std::exception& error) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::cerr << "MPI shard test rank " << rank << ": " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
