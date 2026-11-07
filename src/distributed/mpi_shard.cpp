#include "contracts.hpp"
#include "core/entity_signature.hpp"
#include "idmesh/mpi.hpp"
#include "mpi_errors.hpp"
#include "mpi_internal.hpp"
#include "mpi_shard_profile.hpp"
#include "shard.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <map>
#include <mpi.h>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace idmesh::mpi {
    namespace {

        using Clock = std::chrono::steady_clock;

        thread_local detail::ShardReadProfile* active_profile = nullptr;

        double milliseconds(Clock::time_point begin, Clock::time_point end)
        {
            return std::chrono::duration<double, std::milli>(
                end - begin).count();
        }

        struct ProfileBinding {
            explicit ProfileBinding(detail::ShardReadProfile& profile)
                : previous(active_profile)
            {
                active_profile = &profile;
            }

            ~ProfileBinding() { active_profile = previous; }

            detail::ShardReadProfile* previous = nullptr;
        };

        using ::idmesh::detail::shard::Cell;
        using ::idmesh::detail::shard::Label;
        using ::idmesh::detail::shard::Shard;

        using Key = std::pair<Index, ::idmesh::detail::EntitySignature<std::uint64_t>>;

        struct LocalMeshData {
            Mesh mesh;
            std::vector<std::uint64_t> cell_ids;
            std::vector<int> cell_owners;
            std::vector<std::uint64_t> local_to_global_vertex;
        };

        LocalMeshData build_local_mesh(
            const Shard& input, BuildOptions options)
        {
            const Index dim = input.dimension;
            MeshBuilder builder(
                input.dimension, input.geometric_dimension);
            std::map<std::uint64_t, Index>
                vertices, high_order, cells;

            // Ascending global corner IDs preserve canonical orientation.
            for (const auto& [id, vertex] : input.vertices)
                vertices[id] = builder.add_vertex(vertex.position);

            std::vector<std::uint64_t> cell_ids;
            std::vector<int> cell_owners;
            for (const Cell& cell : input.cells) {
                // A standard linear/quadratic cell has at most 27 geometry nodes.
                std::array<Index, 27> nodes {};
                const Index corner_count = cell_vertex_count(cell.type);
                for (Index j = 0; j < corner_count; ++j)
                    nodes[j] = vertices.at(cell.geometry[j]);
                for (std::size_t j = corner_count;
                     j < cell.geometry.size(); ++j) {
                    const auto id = cell.geometry[j];
                    auto [it, inserted]
                        = high_order.try_emplace(id, invalid_index);
                    if (inserted)
                        it->second = builder.add_geometry_node(
                            input.geometry_nodes.at(id));
                    nodes[j] = it->second;
                }
                const std::span<const Index> corners(nodes.data(), corner_count);
                const std::span<const Index> extras(
                    nodes.data() + corner_count,
                    cell.geometry.size() - corner_count);
                const Index local = extras.empty()
                    ? builder.add_cell(cell.type, corners)
                    : builder.add_quadratic_cell(
                          cell.type, corners, extras);
                ::idmesh::detail::require_runtime(
                    local == cell_ids.size(),
                    "unexpected shard builder cell ordering");
                cells[cell.id] = local;
                cell_ids.push_back(cell.id);
                cell_owners.push_back(cell.owner);
            }

            for (const Label& label : input.labels) {
                if (label.dimension == dim) {
                    ::idmesh::detail::require_argument(
                        label.entities.size() == 1
                            && cells.contains(label.entities[0]),
                        "invalid top-cell label reference");
                    builder.add_cell_label(
                        cells.at(label.entities[0]), label.value);
                }
                else {
                    std::array<Index, 4> corners {};
                    for (std::size_t j = 0; j < label.entities.size(); ++j)
                        corners[j] = vertices.at(label.entities[j]);
                    builder.add_label(label.dimension,
                        std::span(corners).first(label.entities.size()),
                        label.value);
                }
            }
            for (const auto& [key, name] : input.label_names)
                builder.set_label_name(
                    key.first, key.second, name);

            LocalMeshData result {
                std::move(builder).finalize(options),
                std::move(cell_ids),
                std::move(cell_owners),
                {}};
            result.local_to_global_vertex.resize(vertices.size());
            for (const auto& [global, local] : vertices)
                result.local_to_global_vertex[local] = global;
            return result;
        }

        struct IdentityMetadata {
            std::vector<std::vector<std::uint64_t>> global_ids;
            std::vector<std::vector<int>> owner_ranks;
        };

        IdentityMetadata load_identity_metadata(
            const Shard& input, LocalMeshData& local,
            detail::ShardReadProfile* profile)
        {
            const auto begin
                = profile ? Clock::now() : Clock::time_point {};
            const Mesh& mesh = local.mesh;
            const Index dim = mesh.dimension();

            IdentityMetadata metadata;
            metadata.global_ids.resize(dim + 1);
            metadata.owner_ranks.resize(dim + 1);
            metadata.global_ids[dim] = std::move(local.cell_ids);
            metadata.owner_ranks[dim] = std::move(local.cell_owners);

            metadata.global_ids[0] = std::move(local.local_to_global_vertex);
            metadata.owner_ranks[0].resize(mesh.entity_count(0));
            for (Index vertex = 0;
                 vertex < mesh.entity_count(0); ++vertex) {
                const auto global = metadata.global_ids[0][vertex];
                metadata.owner_ranks[0][vertex]
                    = input.vertices.at(global).owner;
            }

            std::map<Key, std::pair<std::uint64_t, int>> embedded;
            for (const auto& entity : input.entities) {
                const auto canonical
                    = ::idmesh::detail::canonicalize_entity<
                        std::uint64_t>(
                            std::span(entity.vertices).first(entity.arity));
                Key key {
                    entity.dimension, canonical.signature};
                ::idmesh::detail::require_runtime(
                    embedded.emplace(
                        std::move(key),
                        std::pair {entity.id, entity.owner}).second,
                    "duplicate shard entity identity key");
            }

            for (Index k = 1; k < dim; ++k) {
                if (!mesh.has_entities(k))
                    continue;
                const Index count = mesh.entity_count(k);
                metadata.global_ids[k].resize(count);
                metadata.owner_ranks[k].resize(count);
                for (Index entity = 0; entity < count; ++entity) {
                    const auto local_vertices
                        = mesh.entity_vertices(k, entity);
                    std::array<std::uint64_t, 4> global_vertices {};
                    ::idmesh::detail::require_runtime(
                        local_vertices.size() <= global_vertices.size(),
                        "invalid standard-cell entity arity");
                    for (std::size_t j = 0; j < local_vertices.size(); ++j)
                        global_vertices[j]
                            = metadata.global_ids[0][local_vertices[j]];
                    const auto canonical
                        = ::idmesh::detail::canonicalize_entity<
                            std::uint64_t>(std::span(global_vertices).first(
                                local_vertices.size()));
                    const Key key {
                        k, canonical.signature};
                    const auto found = embedded.find(key);
                    ::idmesh::detail::require_runtime(
                        found != embedded.end(),
                        "shard lacks materialized entity identity");
                    metadata.global_ids[k][entity]
                        = found->second.first;
                    metadata.owner_ranks[k][entity]
                        = found->second.second;
                }
            }

            if (profile)
                profile->metadata_ms
                    = milliseconds(begin, Clock::now());
            return metadata;
        }

        struct ExchangeLayout {
            std::vector<Index> owned;
            std::vector<std::vector<Index>> send;
            std::vector<std::vector<Index>> receive;
        };

        std::vector<ExchangeLayout> load_exchange_layouts(
            int rank, int size, const Shard& input,
            const Mesh& mesh, const IdentityMetadata& metadata)
        {
            const Index dim = mesh.dimension();
            std::vector<ExchangeLayout> layouts(dim + 1);
            std::vector<std::map<std::uint64_t, Index>>
                local_by_global(dim + 1);

            for (Index k = 0; k <= dim; ++k) {
                if (!mesh.has_entities(k))
                    continue;
                const Index count = mesh.entity_count(k);
                ::idmesh::detail::require_runtime(
                    metadata.global_ids[k].size() == count
                        && metadata.owner_ranks[k].size() == count,
                    "distributed entity metadata size mismatch");
                auto& layout = layouts[k];
                layout.send.resize(size);
                layout.receive.resize(size);
                for (Index entity = 0; entity < count; ++entity) {
                    const auto global = metadata.global_ids[k][entity];
                    ::idmesh::detail::require_runtime(
                        local_by_global[k]
                            .emplace(global, entity).second,
                        "duplicate local global entity ID");
                    const int owner
                        = metadata.owner_ranks[k][entity];
                    ::idmesh::detail::require_runtime(
                        owner >= 0 && owner < size,
                        "invalid distributed entity owner");
                    if (owner == rank)
                        layout.owned.push_back(entity);
                }
            }

            std::vector<std::set<std::pair<int, Index>>>
                seen_send(dim + 1);
            std::vector<std::vector<int>> receive_from(dim + 1);
            for (Index k = 0; k <= dim; ++k)
                if (mesh.has_entities(k))
                    receive_from[k].assign(
                        mesh.entity_count(k), -1);

            for (const auto& halo : input.halos) {
                if (!mesh.has_entities(halo.dimension))
                    continue;
                auto& layout = layouts[halo.dimension];
                const auto& lookup
                    = local_by_global[halo.dimension];

                for (const auto global : halo.send) {
                    const auto found = lookup.find(global);
                    ::idmesh::detail::require_runtime(
                        found != lookup.end()
                            && metadata.owner_ranks[halo.dimension]
                                   [found->second]
                                == rank
                            && seen_send[halo.dimension]
                                   .emplace(halo.peer, found->second)
                                   .second,
                        "invalid shard halo send entity");
                    layout.send[halo.peer].push_back(
                        found->second);
                }
                for (const auto global : halo.receive) {
                    const auto found = lookup.find(global);
                    ::idmesh::detail::require_runtime(
                        found != lookup.end()
                            && metadata.owner_ranks[halo.dimension]
                                   [found->second]
                                == halo.peer
                            && receive_from[halo.dimension]
                                   [found->second]
                                == -1,
                        "invalid shard halo receive entity");
                    receive_from[halo.dimension][found->second]
                        = halo.peer;
                    layout.receive[halo.peer].push_back(
                        found->second);
                }
            }

            for (Index k = 0; k <= dim; ++k) {
                if (!mesh.has_entities(k))
                    continue;
                for (Index entity = 0;
                     entity < mesh.entity_count(k); ++entity) {
                    const int owner
                        = metadata.owner_ranks[k][entity];
                    const int expected
                        = owner == rank ? -1 : owner;
                    ::idmesh::detail::require_runtime(
                        receive_from[k][entity] == expected,
                        "shard halo receive coverage mismatch");
                }
            }
            return layouts;
        }

    } // namespace

    PartitionedMesh read_partition_shard(
        const Context& context, const std::string& path,
        BuildOptions options)
    {
        const int rank = context.rank(), size = context.size();
        const MPI_Comm comm = context.data_->comm;
        auto* profile = active_profile;
        if (profile)
            *profile = {};

        const auto parse_begin
            = profile ? Clock::now() : Clock::time_point {};
        const Shard input = ::idmesh::detail::shard::read(path, rank, size);
        if (profile)
            profile->parse_ms
                = milliseconds(parse_begin, Clock::now());

        const Index dim = input.dimension;
        Index min_dim = 0, max_dim = 0;
        ::idmesh::mpi::detail::check_mpi(
            MPI_Allreduce(
                &dim, &min_dim, 1, MPI_UNSIGNED, MPI_MIN, comm),
            "MPI_Allreduce");
        ::idmesh::mpi::detail::check_mpi(
            MPI_Allreduce(
                &dim, &max_dim, 1, MPI_UNSIGNED, MPI_MAX, comm),
            "MPI_Allreduce");
        ::idmesh::detail::require_runtime(
            min_dim == max_dim,
            "shards disagree on topological dimension");

        const auto mesh_begin
            = profile ? Clock::now() : Clock::time_point {};
        auto local = build_local_mesh(input, options);
        if (profile)
            profile->local_mesh_ms
                = milliseconds(mesh_begin, Clock::now());

        auto metadata
            = load_identity_metadata(input, local, profile);

        const auto exchange_begin
            = profile ? Clock::now() : Clock::time_point {};
        auto layouts = load_exchange_layouts(
            rank, size, input, local.mesh, metadata);

        auto partition = std::make_shared<PartitionedMesh::Data>(
            std::move(local.mesh));
        partition->global_ids = std::move(metadata.global_ids);
        partition->owner_ranks = std::move(metadata.owner_ranks);
        partition->owned_entities.resize(dim + 1);
        partition->exchanges.resize(dim + 1);

        for (Index k = 0; k <= dim; ++k) {
            if (!partition->mesh.has_entities(k))
                continue;
            partition->owned_entities[k]
                = std::move(layouts[k].owned);
            auto exchange
                = std::make_shared<ExchangePlan::Data>();
            exchange->context_owner = context.data_;
            exchange->local_entities
                = partition->mesh.entity_count(k);
            exchange->initialize_layout(comm,
                std::move(layouts[k].send),
                std::move(layouts[k].receive));
            partition->exchanges[k]
                = ExchangePlan(std::move(exchange));
        }
        if (profile)
            profile->exchange_plan_ms
                = milliseconds(exchange_begin, Clock::now());

        const auto validate_begin
            = profile ? Clock::now() : Clock::time_point {};
        partition->mesh.validate();
        if (profile)
            profile->validate_ms
                = milliseconds(validate_begin, Clock::now());
        return PartitionedMesh(std::move(partition));
    }

    namespace detail {

        PartitionedMesh read_partition_shard_profiled(
            const Context& context, const std::string& path,
            BuildOptions options, ShardReadProfile& profile)
        {
            ProfileBinding binding(profile);
            return read_partition_shard(
                context, path, options);
        }

    } // namespace detail

} // namespace idmesh::mpi