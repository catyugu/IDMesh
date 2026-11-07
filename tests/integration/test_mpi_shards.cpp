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
#include <span>
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

    using EntityKey
        = std::pair<Index, std::vector<std::uint64_t>>;

    std::map<EntityKey, std::uint64_t> source_entity_ids(
        const idmesh::Mesh& mesh)
    {
        std::map<EntityKey, std::uint64_t> result;
        for (Index k = 0; k < mesh.dimension(); ++k) {
            if (!mesh.has_entities(k))
                continue;
            for (Index entity = 0;
                 entity < mesh.entity_count(k); ++entity) {
                const auto vertices
                    = mesh.entity_vertices(k, entity);
                std::vector<std::uint64_t> key(
                    vertices.begin(), vertices.end());
                std::sort(key.begin(), key.end());
                require(result.emplace(
                            EntityKey {k, std::move(key)}, entity)
                            .second,
                    "duplicate source entity signature");
            }
        }
        return result;
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
        struct Record {
            std::uint64_t first = ~std::uint64_t {0};
            std::uint64_t second = ~std::uint64_t {0};
        };
        std::vector<Record> values(count);
        for (Index entity : owned) {
            const auto global = partition.global_ids(dimension)[entity];
            values[entity].first = global * 3 + dimension + 11;
            values[entity].second = global * 5 + dimension + 17;
        }
        // Abandonment must finish transport without writing to borrowed records.
        {
            auto abandoned = plan.make_request(sizeof(Record));
            abandoned.start(std::as_writable_bytes(std::span(values)));
        }
        for (Index entity = 0; entity < count; ++entity)
            if (!is_owned[entity])
                require(values[entity].first == ~std::uint64_t {0}
                        && values[entity].second == ~std::uint64_t {0},
                    "abandoned host halo updated ghost records");

        auto request = plan.make_request(sizeof(Record));
        request.start(std::as_writable_bytes(std::span(values)));
        request.wait();
        for (Index entity = 0; entity < count; ++entity) {
            const auto global = partition.global_ids(dimension)[entity];
            require(values[entity].first == global * 3 + dimension + 11
                    && values[entity].second
                        == global * 5 + dimension + 17,
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

    struct Incidence {
        std::uint64_t upper = 0;
        std::uint64_t lower = 0;
        Index dimension = 0;
        int coefficient = 0;
        std::uint16_t rotation = 0;
        bool reversed = false;
    };

    template <class T>
    std::vector<T> gather_records(
        const std::vector<T>& here, int size)
    {
        const int bytes = static_cast<int>(here.size() * sizeof(T));
        std::vector<int> counts(size), displacements(size);
        MPI_Allgather(
            &bytes, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        int total = 0;
        for (int peer = 0; peer < size; ++peer) {
            displacements[peer] = total;
            total += counts[peer];
        }
        require(total % sizeof(T) == 0, "invalid record gather size");
        std::vector<T> all(total / sizeof(T));
        MPI_Allgatherv(here.data(), bytes, MPI_BYTE,
            all.data(), counts.data(), displacements.data(), MPI_BYTE,
            MPI_COMM_WORLD);
        return all;
    }

    void check_shared_identity(
        const idmesh::mpi::PartitionedMesh& partition, int size,
        const std::map<EntityKey, std::uint64_t>* expected = nullptr)
    {
        const Index dim = partition.mesh().dimension();
        std::vector<std::uint64_t> here;
        for (Index k = 0; k < dim; ++k) {
            if (!partition.mesh().has_entities(k))
                continue;
            for (Index entity = 0;
                 entity < partition.mesh().entity_count(k); ++entity) {
                const auto vertices
                    = partition.mesh().entity_vertices(k, entity);
                here.push_back(k);
                here.push_back(vertices.size());
                here.push_back(partition.global_ids(k)[entity]);
                here.push_back(static_cast<std::uint64_t>(
                    partition.owner_ranks(k)[entity]));
                for (Index vertex : vertices)
                    here.push_back(
                        partition.global_ids(0)[vertex]);
            }
        }

        const auto all = gather_records(here, size);
        using Key = std::pair<Index, std::vector<std::uint64_t>>;
        std::map<Key, std::pair<std::uint64_t, int>> identities;
        std::map<std::pair<Index, std::uint64_t>, Key> reverse;
        std::map<std::pair<Index, std::uint64_t>,
            std::vector<std::uint64_t>> ordered;
        std::size_t shared = 0;
        for (std::size_t position = 0; position < all.size();) {
            require(all.size() - position >= 4,
                "truncated gathered entity identity");
            const auto dimension = static_cast<Index>(all[position++]);
            const auto arity = static_cast<std::size_t>(all[position++]);
            const auto global = all[position++];
            const auto owner = static_cast<int>(all[position++]);
            require(arity > 0 && arity <= all.size() - position,
                "invalid gathered entity arity");

            std::vector<std::uint64_t> canonical(
                all.begin() + static_cast<std::ptrdiff_t>(position),
                all.begin() + static_cast<std::ptrdiff_t>(position + arity));
            position += arity;
            auto sorted = canonical;
            std::sort(sorted.begin(), sorted.end());

            Key key {dimension, std::move(sorted)};
            if (expected != nullptr) {
                const auto source = expected->find(key);
                require(source != expected->end()
                        && source->second == global,
                    "distributed global ID differs from source topology ID");
            }
            auto [existing, inserted] = identities.try_emplace(
                key, global, owner);
            require(inserted
                    || existing->second
                        == std::pair {global, owner},
                "shared entity identity/owner mismatch across ranks");
            if (!inserted)
                ++shared;

            const auto name = std::pair {dimension, global};
            auto [back, unique] = reverse.try_emplace(name, key);
            require(unique || back->second == key,
                "global entity-ID collision");

            auto [order, first]
                = ordered.try_emplace(name, canonical);
            require(first || order->second == canonical,
                "canonical entity orientation differs across ranks");
        }
        require(shared > 0, "distributed test has no shared subentities");
    }

    void check_shared_incidence(
        const idmesh::mpi::PartitionedMesh& partition, int size)
    {
        const Index dim = partition.mesh().dimension();
        std::vector<Incidence> here;
        for (Index from = 1; from <= dim; ++from) {
            if (!partition.mesh().has_entities(from)
                || !partition.mesh().has_relation(from, from - 1))
                continue;
            const auto relation
                = partition.mesh().relation(from, from - 1);
            for (Index upper = 0; upper < relation.rows; ++upper) {
                const auto row = relation[upper];
                require(row.coefficients.size() == row.targets.size()
                        && row.orientations.size() == row.targets.size(),
                    "oriented boundary metadata missing");
                for (std::size_t i = 0; i < row.targets.size(); ++i) {
                    here.push_back({
                        partition.global_ids(from)[upper],
                        partition.global_ids(from - 1)[row.targets[i]],
                        from,
                        row.coefficients[i],
                        row.orientations[i].rotation,
                        row.orientations[i].reversed});
                }
            }
        }

        const auto all = gather_records(here, size);
        using Key
            = std::tuple<Index, std::uint64_t, std::uint64_t>;
        using Value = std::tuple<int, std::uint16_t, bool>;
        std::map<Key, Value> incidences;
        std::size_t repeated = 0;
        for (const auto& record : all) {
            const Key key {
                record.dimension, record.upper, record.lower};
            const Value value {record.coefficient,
                record.rotation, record.reversed};
            auto [existing, inserted]
                = incidences.try_emplace(key, value);
            require(inserted || existing->second == value,
                "boundary sign/orientation differs across ranks");
            if (!inserted)
                ++repeated;
        }
        require(repeated > 0,
            "distributed test has no repeated shared incidences");
    }

    std::vector<std::uint64_t> test_shard(
        const idmesh::mpi::Context& context,
        const std::string& prefix, int rank, int size, Index ghost_depth)
    {
        std::optional<idmesh::Mesh> root;
        std::vector<int> assignments;
        std::map<EntityKey, std::uint64_t> expected_ids;
        if (rank == 0) {
            root.emplace(source());
            expected_ids = source_entity_ids(*root);
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

        check_shared_identity(
            partition, size, rank == 0 ? &expected_ids : nullptr);
        check_shared_incidence(partition, size);

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

        std::vector<std::uint64_t> local_cells(
            partition.global_ids(dim).begin(),
            partition.global_ids(dim).end());
        std::sort(local_cells.begin(), local_cells.end());

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            for (int p = 0; p < size; ++p)
                std::filesystem::remove(
                    prefix + "." + std::to_string(p) + ".idm");
        }
        MPI_Barrier(MPI_COMM_WORLD);
        return local_cells;
    }

    void test_complete_3d(const idmesh::mpi::Context& context,
        const std::string& prefix, int rank, int size)
    {
        std::map<EntityKey, std::uint64_t> expected_ids;
        if (rank == 0) {
            const auto mesh = source_3d_mixed();
            expected_ids = source_entity_ids(mesh);
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
        check_shared_identity(
            partition, size, rank == 0 ? &expected_ids : nullptr);
        check_shared_incidence(partition, size);

        unsigned local_types = 0;
        for (const auto type : partition.mesh().cell_types())
            local_types |= 1u << static_cast<unsigned>(type);
        unsigned global_types = 0;
        MPI_Allreduce(&local_types, &global_types, 1,
            MPI_UNSIGNED, MPI_BOR, MPI_COMM_WORLD);
        for (const auto type : {
                 idmesh::CellType::tetrahedron,
                 idmesh::CellType::hexahedron,
                 idmesh::CellType::prism,
                 idmesh::CellType::pyramid})
            require((global_types
                        & (1u << static_cast<unsigned>(type)))
                    != 0,
                "mixed 3D distributed fixture lost a cell family");

        auto facets = idmesh::mpi::read_partition_shard(
            context, local_path,
            idmesh::BuildOptions {
                .topology = idmesh::TopologyLayout::cell_facets,
                .relations = idmesh::RelationDirection::bidirectional});
        facets.mesh().validate();
        require(facets.mesh().has_entities(2)
                && !facets.mesh().has_entities(1),
            "cell_facets layout materialized the wrong topology");
        check_entity_exchange(facets, 0, rank);
        check_entity_exchange(facets, 2, rank);
        check_entity_exchange(facets, 3, rank);

        auto minimal = idmesh::mpi::read_partition_shard(
            context, local_path,
            idmesh::BuildOptions {
                .topology = idmesh::TopologyLayout::cell_vertices,
                .relations = idmesh::RelationDirection::bidirectional});
        minimal.mesh().validate();
        require(!minimal.mesh().has_entities(1)
                && !minimal.mesh().has_entities(2),
            "cell_vertices layout materialized intermediate topology");
        check_entity_exchange(minimal, 0, rank);
        check_entity_exchange(minimal, 3, rank);

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
    // MPI_Initialized must reject use both before initialization and after finalization.
    bool rejected_before_init = false;
    try {
        (void)idmesh::mpi::Context::world();
    }
    catch (const std::logic_error&) {
        rejected_before_init = true;
    }
    if (!rejected_before_init) {
        std::cerr << "MPI context constructed before MPI_Init\n";
        return 1;
    }

    MPI_Init(&argc, &argv);
    const auto handle = static_cast<std::int64_t>(MPI_Comm_c2f(MPI_COMM_WORLD));
    try {
        // All communicator owners must be destroyed before MPI_Finalize.
        {
            const auto context = idmesh::mpi::Context::world();
            const auto from_handle
                = idmesh::mpi::Context::from_fortran_handle(handle);
            const int rank = context.rank(), size = context.size();
            require(rank == from_handle.rank() && size == from_handle.size(),
                "MPI communicator handle conversion changed context");
            require(size >= 2 && size <= 3, "test requires two or three MPI ranks");
            const auto base = (std::filesystem::temp_directory_path()
                / ("idmesh-parallel-" + std::to_string(size)))
                                  .string();
            const auto depth1
                = test_shard(context, base + "-g1", rank, size, 1);
            const auto depth2
                = test_shard(context, base + "-g2", rank, size, 2);
            require(std::includes(
                        depth2.begin(), depth2.end(),
                        depth1.begin(), depth1.end()),
                "two-layer ghost cells are not a superset of one-layer ghosts");
            int local_growth = depth2.size() > depth1.size() ? 1 : 0;
            int any_growth = 0;
            MPI_Allreduce(&local_growth, &any_growth, 1,
                MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            require(any_growth != 0,
                "second ghost layer did not add any cell");
            test_complete_3d(context, base + "-3d", rank, size);
            if (rank == 0)
                std::cout << "MPI shard: graph cuts, rank-local loading, all-dimension "
                          << "identity/halo, orientation and quadratic geometry OK\n";
        }

        MPI_Finalize();
        bool world_rejected = false, handle_rejected = false;
        try {
            (void)idmesh::mpi::Context::world();
        }
        catch (const std::logic_error&) {
            world_rejected = true;
        }
        try {
            (void)idmesh::mpi::Context::from_fortran_handle(handle);
        }
        catch (const std::logic_error&) {
            handle_rejected = true;
        }
        if (!world_rejected || !handle_rejected) {
            std::cerr << "MPI context accepted a finalized MPI runtime\n";
            return 1;
        }
        return 0;
    }
    catch (const std::exception& error) {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (finalized) {
            std::cerr << "MPI shard test after MPI_Finalize: "
                      << error.what() << '\n';
            return 1;
        }
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::cerr << "MPI shard test rank " << rank << ": " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
