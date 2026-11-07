#include "contracts.hpp"
#include "entity_signature.hpp"
#include "idmesh/mpi.hpp"
#include "mpi_errors.hpp"
#include "mpi_internal.hpp"
#include "mpi_shard_profile.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <map>
#include <mpi.h>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
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

        struct Cell {
            std::uint64_t id = 0;
            int owner = -1;
            CellType type = CellType::segment;
            std::vector<std::uint64_t> geometry;
        };

        struct EntityIdentity {
            Index dimension = 0;
            std::uint64_t id = 0;
            int owner = -1;
            std::vector<std::uint64_t> vertices;
        };

        struct HaloRecord {
            Index dimension = 0;
            int peer = -1;
            std::vector<std::uint64_t> send;
            std::vector<std::uint64_t> receive;
        };

        struct Label {
            Index dimension = 0;
            int value = 0;
            std::vector<std::uint64_t> entities;
        };

        struct Shard {
            Index dimension = 0;
            Index geometric_dimension = 0;
            std::map<std::uint64_t, std::array<double, 3>> vertices;
            std::map<std::uint64_t, int> vertex_owners;
            std::map<std::uint64_t, std::array<double, 3>> geometry_nodes;
            std::vector<Cell> cells;
            std::vector<EntityIdentity> entities;
            std::vector<HaloRecord> halos;
            std::vector<Label> labels;
            std::map<std::pair<Index, int>, std::string> label_names;
        };

        Shard parse_shard(const std::string& path, int rank, int size)
        {
            std::ifstream file(path);
            ::idmesh::detail::require_runtime(
                static_cast<bool>(file),
                "cannot open local IDMesh partition shard");

            Shard shard;
            std::string word;
            int version = 0;
            ::idmesh::detail::require_runtime(
                static_cast<bool>(file >> word >> version)
                    && word == "IDMESH_SHARD" && version == 3,
                "invalid IDMesh shard version");
            ::idmesh::detail::require_runtime(
                static_cast<bool>(file >> word >> shard.dimension
                    >> shard.geometric_dimension)
                    && word == "DIM"
                    && shard.dimension >= 1 && shard.dimension <= 3
                    && shard.geometric_dimension >= shard.dimension
                    && shard.geometric_dimension <= 3,
                "invalid shard dimensions");

            std::size_t count = 0;
            ::idmesh::detail::require_runtime(
                static_cast<bool>(file >> word >> count)
                    && word == "VERTICES" && count < invalid_index,
                "invalid shard VERTICES section");
            for (std::size_t i = 0; i < count; ++i) {
                std::uint64_t id = 0;
                int owner = -1;
                std::array<double, 3> p {};
                ::idmesh::detail::require_runtime(
                    static_cast<bool>(file >> id >> owner
                        >> p[0] >> p[1] >> p[2])
                        && owner >= 0 && owner < size,
                    "truncated or invalid shard vertex");
                ::idmesh::detail::require_argument(
                    std::isfinite(p[0]) && std::isfinite(p[1])
                        && std::isfinite(p[2]),
                    "nonfinite shard vertex");
                ::idmesh::detail::require_runtime(
                    shard.vertices.emplace(id, p).second
                        && shard.vertex_owners.emplace(id, owner).second,
                    "duplicate shard vertex");
            }

            ::idmesh::detail::require_runtime(
                static_cast<bool>(file >> word >> count)
                    && word == "GEOMETRY_NODES"
                    && count < invalid_index,
                "invalid GEOMETRY_NODES section");
            for (std::size_t i = 0; i < count; ++i) {
                std::uint64_t id = 0;
                std::array<double, 3> p {};
                ::idmesh::detail::require_runtime(
                    static_cast<bool>(file >> id
                        >> p[0] >> p[1] >> p[2]),
                    "truncated shard high-order node");
                ::idmesh::detail::require_argument(
                    std::isfinite(p[0]) && std::isfinite(p[1])
                        && std::isfinite(p[2]),
                    "nonfinite high-order node");
                ::idmesh::detail::require_runtime(
                    shard.geometry_nodes.emplace(id, p).second,
                    "duplicate shard high-order node");
            }

            ::idmesh::detail::require_runtime(
                static_cast<bool>(file >> word >> count)
                    && word == "CELLS" && count < invalid_index,
                "invalid shard CELLS section");
            std::set<std::uint64_t> seen_cells;
            for (std::size_t i = 0; i < count; ++i) {
                Cell cell;
                unsigned type = 0;
                std::size_t nodes = 0;
                ::idmesh::detail::require_runtime(
                    static_cast<bool>(file >> cell.id >> cell.owner
                        >> type >> nodes),
                    "invalid shard cell header");
                ::idmesh::detail::require_runtime(
                    cell.owner >= 0 && cell.owner < size
                        && type
                            >= static_cast<unsigned>(CellType::segment)
                        && type
                            <= static_cast<unsigned>(CellType::pyramid),
                    "invalid shard cell owner or type");
                cell.type = static_cast<CellType>(type);
                ::idmesh::detail::require_argument(
                    cell_dimension(cell.type) == shard.dimension
                        && nodes >= cell_vertex_count(cell.type)
                        && nodes <= 27,
                    "invalid cell dimension or geometry arity");
                cell.geometry.resize(nodes);
                for (auto& id : cell.geometry)
                    ::idmesh::detail::require_runtime(
                        static_cast<bool>(file >> id),
                        "truncated shard cell geometry");
                for (Index j = 0;
                     j < cell_vertex_count(cell.type); ++j)
                    ::idmesh::detail::require_argument(
                        shard.vertices.contains(cell.geometry[j]),
                        "cell references missing global corner vertex");
                for (std::size_t j = cell_vertex_count(cell.type);
                     j < nodes; ++j)
                    ::idmesh::detail::require_argument(
                        shard.geometry_nodes.contains(cell.geometry[j]),
                        "cell references missing high-order geometry node");
                ::idmesh::detail::require_runtime(
                    seen_cells.insert(cell.id).second,
                    "duplicate shard global cell ID");
                shard.cells.push_back(std::move(cell));
            }

            ::idmesh::detail::require_runtime(
                static_cast<bool>(file >> word >> count)
                    && word == "ENTITIES" && count < invalid_index,
                "invalid shard ENTITIES section");
            std::set<std::pair<Index, std::uint64_t>> seen_entities;
            for (std::size_t i = 0; i < count; ++i) {
                EntityIdentity entity;
                std::size_t arity = 0;
                ::idmesh::detail::require_runtime(
                    static_cast<bool>(file >> entity.dimension
                        >> entity.id >> entity.owner >> arity)
                        && entity.dimension > 0
                        && entity.dimension < shard.dimension
                        && entity.owner >= 0 && entity.owner < size
                        && arity >= 2 && arity < invalid_index,
                    "invalid shard entity identity header");
                entity.vertices.resize(arity);
                for (auto& vertex : entity.vertices) {
                    ::idmesh::detail::require_runtime(
                        static_cast<bool>(file >> vertex),
                        "truncated shard entity identity");
                    ::idmesh::detail::require_runtime(
                        shard.vertices.contains(vertex),
                        "entity identity references missing vertex");
                }
                ::idmesh::detail::require_runtime(
                    seen_entities.emplace(
                        entity.dimension, entity.id).second,
                    "duplicate shard entity global ID");
                shard.entities.push_back(std::move(entity));
            }

            ::idmesh::detail::require_runtime(
                static_cast<bool>(file >> word >> count)
                    && word == "HALOS" && count < invalid_index,
                "invalid shard HALOS section");
            std::set<std::pair<Index, int>> seen_halos;
            for (std::size_t i = 0; i < count; ++i) {
                HaloRecord halo;
                std::size_t send_count = 0, receive_count = 0;
                ::idmesh::detail::require_runtime(
                    static_cast<bool>(file >> halo.dimension >> halo.peer
                        >> send_count >> receive_count)
                        && halo.dimension <= shard.dimension
                        && halo.peer >= 0 && halo.peer < size
                        && halo.peer != rank
                        && send_count < invalid_index
                        && receive_count < invalid_index,
                    "invalid shard halo header");
                ::idmesh::detail::require_runtime(
                    seen_halos.emplace(
                        halo.dimension, halo.peer).second,
                    "duplicate shard halo peer record");
                halo.send.resize(send_count);
                halo.receive.resize(receive_count);
                for (auto& id : halo.send)
                    ::idmesh::detail::require_runtime(
                        static_cast<bool>(file >> id),
                        "truncated shard halo send list");
                for (auto& id : halo.receive)
                    ::idmesh::detail::require_runtime(
                        static_cast<bool>(file >> id),
                        "truncated shard halo receive list");
                shard.halos.push_back(std::move(halo));
            }

            ::idmesh::detail::require_runtime(
                static_cast<bool>(file >> word >> count)
                    && word == "LABELS",
                "missing shard LABELS section");
            for (std::size_t i = 0; i < count; ++i) {
                Label label;
                std::size_t n = 0;
                ::idmesh::detail::require_runtime(
                    static_cast<bool>(file >> label.dimension
                        >> label.value >> n)
                        && label.dimension <= shard.dimension
                        && n >= 1 && n < invalid_index,
                    "invalid shard label header");
                const bool valid_arity
                    = label.dimension == shard.dimension
                    || label.dimension == 0
                    ? n == 1
                    : label.dimension == 1 ? n == 2 : n >= 3;
                ::idmesh::detail::require_runtime(
                    valid_arity, "invalid shard label arity");
                label.entities.resize(n);
                for (auto& id : label.entities)
                    ::idmesh::detail::require_runtime(
                        static_cast<bool>(file >> id),
                        "truncated shard label");
                shard.labels.push_back(std::move(label));
            }

            ::idmesh::detail::require_runtime(
                static_cast<bool>(file >> word >> count)
                    && word == "NAMES",
                "missing shard NAMES section");
            for (std::size_t i = 0; i < count; ++i) {
                Index dimension = 0;
                int value = 0;
                std::string name;
                ::idmesh::detail::require_runtime(
                    static_cast<bool>(file >> dimension >> value
                        >> std::quoted(name))
                        && dimension <= shard.dimension,
                    "invalid shard label name");
                shard.label_names[{dimension, value}]
                    = std::move(name);
            }

            ::idmesh::detail::require_runtime(
                static_cast<bool>(file >> word) && word == "END",
                "missing shard END marker");
            ::idmesh::detail::require_runtime(
                !(file >> word),
                "trailing garbage after shard END");
            ::idmesh::detail::require_runtime(
                !shard.cells.empty(),
                "shard requires at least one local cell");

            std::stable_sort(shard.cells.begin(), shard.cells.end(),
                [rank](const Cell& a, const Cell& b) {
                    return std::tuple {a.owner != rank, a.id}
                        < std::tuple {b.owner != rank, b.id};
                });
            return shard;
        }

        struct Key {
            Index dimension = 0;
            ::idmesh::detail::EntitySignature<std::uint64_t> signature;

            bool operator<(const Key& other) const
            {
                if (dimension != other.dimension)
                    return dimension < other.dimension;
                return signature < other.signature;
            }
        };

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
            for (const auto& [id, position] : input.vertices)
                vertices[id] = builder.add_vertex(position);

            std::vector<std::uint64_t> cell_ids;
            std::vector<int> cell_owners;
            for (const Cell& cell : input.cells) {
                std::vector<Index> corners, extras;
                const Index corner_count
                    = cell_vertex_count(cell.type);
                for (Index j = 0; j < corner_count; ++j)
                    corners.push_back(
                        vertices.at(cell.geometry[j]));
                for (std::size_t j = corner_count;
                     j < cell.geometry.size(); ++j) {
                    const auto id = cell.geometry[j];
                    auto [it, inserted]
                        = high_order.try_emplace(id, invalid_index);
                    if (inserted)
                        it->second = builder.add_geometry_node(
                            input.geometry_nodes.at(id));
                    extras.push_back(it->second);
                }
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
                    std::vector<Index> corners;
                    corners.reserve(label.entities.size());
                    for (auto id : label.entities) {
                        ::idmesh::detail::require_runtime(
                            vertices.contains(id),
                            "label refers to missing shard vertex");
                        corners.push_back(vertices.at(id));
                    }
                    builder.add_label(
                        label.dimension, corners, label.value);
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
            const Shard& input, const LocalMeshData& local,
            detail::ShardReadProfile* profile)
        {
            const auto begin
                = profile ? Clock::now() : Clock::time_point {};
            const Mesh& mesh = local.mesh;
            const Index dim = mesh.dimension();

            IdentityMetadata metadata;
            metadata.global_ids.resize(dim + 1);
            metadata.owner_ranks.resize(dim + 1);
            metadata.global_ids[dim] = local.cell_ids;
            metadata.owner_ranks[dim] = local.cell_owners;

            metadata.global_ids[0] = local.local_to_global_vertex;
            metadata.owner_ranks[0].resize(mesh.entity_count(0));
            for (Index vertex = 0;
                 vertex < mesh.entity_count(0); ++vertex) {
                const auto global
                    = local.local_to_global_vertex[vertex];
                const auto owner
                    = input.vertex_owners.find(global);
                ::idmesh::detail::require_runtime(
                    owner != input.vertex_owners.end(),
                    "shard lacks vertex ownership");
                metadata.owner_ranks[0][vertex] = owner->second;
            }

            std::map<Key, std::pair<std::uint64_t, int>> embedded;
            for (const auto& entity : input.entities) {
                const auto canonical
                    = ::idmesh::detail::canonicalize_entity<
                        std::uint64_t>(entity.vertices);
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
                    std::vector<std::uint64_t> global_vertices;
                    global_vertices.reserve(local_vertices.size());
                    for (Index vertex : local_vertices)
                        global_vertices.push_back(
                            local.local_to_global_vertex[vertex]);
                    const auto canonical
                        = ::idmesh::detail::canonicalize_entity<
                            std::uint64_t>(global_vertices);
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
        const Shard input = parse_shard(path, rank, size);
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
