#include "contracts.hpp"
#include "idmesh/mpi.hpp"
#include "mpi_errors.hpp"
#include "mpi_internal.hpp"
#include "mpi_shard_profile.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <mpi.h>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace idmesh::mpi {
    namespace {

        using Clock = std::chrono::steady_clock;

        thread_local detail::ShardReadProfile* active_profile = nullptr;

        double milliseconds(Clock::time_point begin, Clock::time_point end)
        {
            return std::chrono::duration<double, std::milli>(end - begin).count();
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

        template <class T>
        struct Received {
            std::vector<T> values;
            std::vector<int> counts;
            std::vector<int> offsets;
        };

        template <class T>
        Received<T> exchange_records(MPI_Comm comm,
            const std::vector<std::vector<T>>& sends,
            std::uint64_t* sent_bytes = nullptr,
            std::uint64_t* received_bytes = nullptr)
        {
            static_assert(std::is_trivially_copyable_v<T>);
            int size = 0;
            ::idmesh::mpi::detail::check_mpi(MPI_Comm_size(comm, &size), "MPI_Comm_size");
            ::idmesh::detail::require_argument(sends.size() == static_cast<std::size_t>(size),
                "MPI record destination count mismatch");
            std::vector<int> sc(size), rc(size), sd(size), rd(size);
            for (int peer = 0; peer < size; ++peer) {
                ::idmesh::detail::require_argument(sends[peer].size() <= static_cast<std::size_t>(INT_MAX) / sizeof(T),
                    "MPI record message exceeds int count");
                sc[peer] = static_cast<int>(sends[peer].size() * sizeof(T));
            }
            ::idmesh::mpi::detail::check_mpi(MPI_Alltoall(sc.data(), 1, MPI_INT, rc.data(), 1, MPI_INT,
                                                 comm),
                "MPI_Alltoall");
            std::size_t ns = 0, nr = 0;
            for (int peer = 0; peer < size; ++peer) {
                ::idmesh::detail::require_argument(rc[peer] >= 0 && rc[peer] % sizeof(T) == 0,
                    "invalid MPI record message byte count");
                ::idmesh::detail::require_argument(ns <= static_cast<std::size_t>(INT_MAX) - sc[peer] && nr <= static_cast<std::size_t>(INT_MAX) - rc[peer],
                    "MPI_Alltoallv displacement exceeds int range");
                sd[peer] = static_cast<int>(ns);
                rd[peer] = static_cast<int>(nr);
                ns += sc[peer];
                nr += rc[peer];
            }
            if (sent_bytes)
                *sent_bytes += static_cast<std::uint64_t>(ns);
            if (received_bytes)
                *received_bytes += static_cast<std::uint64_t>(nr);
            std::vector<T> send_data(ns / sizeof(T)), recv_data(nr / sizeof(T));
            std::size_t write = 0;
            for (const auto& bucket : sends) {
                std::copy(bucket.begin(), bucket.end(), send_data.begin() + write);
                write += bucket.size();
            }
            ::idmesh::mpi::detail::check_mpi(MPI_Alltoallv(send_data.data(), sc.data(), sd.data(), MPI_BYTE,
                                                 recv_data.data(), rc.data(), rd.data(), MPI_BYTE, comm),
                "MPI_Alltoallv");
            Received<T> result;
            result.values = std::move(recv_data);
            result.counts.resize(size);
            result.offsets.resize(size);
            for (int peer = 0; peer < size; ++peer) {
                result.counts[peer] = rc[peer] / static_cast<int>(sizeof(T));
                result.offsets[peer] = rd[peer] / static_cast<int>(sizeof(T));
            }
            return result;
        }

        struct Cell {
            std::uint64_t id = 0;
            int owner = -1;
            CellType type = CellType::segment;
            std::vector<std::uint64_t> geometry;
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
            std::map<std::uint64_t, std::array<double, 3>> geometry_nodes;
            std::vector<Cell> cells;
            std::vector<Label> labels;
            std::map<std::pair<Index, int>, std::string> label_names;
        };

        Shard parse_shard(const std::string& path, int rank, int size)
        {
            std::ifstream file(path);
            ::idmesh::detail::require_runtime(static_cast<bool>(file), "cannot open local IDMesh partition shard");
            Shard shard;
            std::string word;
            int version = 0;
            ::idmesh::detail::require_runtime(static_cast<bool>(file >> word >> version) && word == "IDMESH_SHARD" && version == 2, "invalid IDMesh shard version");
            ::idmesh::detail::require_runtime(static_cast<bool>(file >> word >> shard.dimension >> shard.geometric_dimension) && word == "DIM" && shard.dimension >= 1 && shard.dimension <= 3 && shard.geometric_dimension >= shard.dimension && shard.geometric_dimension <= 3, "invalid shard dimensions");
            std::size_t count = 0;
            ::idmesh::detail::require_runtime(static_cast<bool>(file >> word >> count) && word == "VERTICES" && count < invalid_index, "invalid shard VERTICES section");
            for (std::size_t i = 0; i < count; ++i) {
                std::uint64_t id = 0;
                std::array<double, 3> p {};
                ::idmesh::detail::require_runtime(static_cast<bool>(file >> id >> p[0] >> p[1] >> p[2]),
                    "truncated shard vertex");
                ::idmesh::detail::require_argument(std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]),
                    "nonfinite shard vertex");
                ::idmesh::detail::require_runtime(shard.vertices.emplace(id, p).second, "duplicate shard vertex");
            }
            ::idmesh::detail::require_runtime(static_cast<bool>(file >> word >> count) && word == "GEOMETRY_NODES" && count < invalid_index, "invalid GEOMETRY_NODES section");
            for (std::size_t i = 0; i < count; ++i) {
                std::uint64_t id = 0;
                std::array<double, 3> p {};
                ::idmesh::detail::require_runtime(static_cast<bool>(file >> id >> p[0] >> p[1] >> p[2]),
                    "truncated shard high-order node");
                ::idmesh::detail::require_argument(std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]),
                    "nonfinite high-order node");
                ::idmesh::detail::require_runtime(shard.geometry_nodes.emplace(id, p).second,
                    "duplicate shard high-order node");
            }
            ::idmesh::detail::require_runtime(static_cast<bool>(file >> word >> count) && word == "CELLS" && count < invalid_index, "invalid shard CELLS section");
            std::set<std::uint64_t> seen;
            for (std::size_t i = 0; i < count; ++i) {
                Cell cell;
                unsigned type = 0;
                std::size_t nodes = 0;
                ::idmesh::detail::require_runtime(static_cast<bool>(file >> cell.id >> cell.owner >> type >> nodes), "invalid shard cell header");
                ::idmesh::detail::require_runtime(cell.owner >= 0 && cell.owner < size && type >= static_cast<unsigned>(CellType::segment) && type <= static_cast<unsigned>(CellType::pyramid),
                    "invalid shard cell owner or type");
                cell.type = static_cast<CellType>(type);
                ::idmesh::detail::require_argument(cell_dimension(cell.type) == shard.dimension && nodes >= cell_vertex_count(cell.type) && nodes <= 27,
                    "invalid cell dimension or geometry arity");
                cell.geometry.resize(nodes);
                for (auto& id : cell.geometry)
                    ::idmesh::detail::require_runtime(static_cast<bool>(file >> id), "truncated shard cell geometry");
                for (Index j = 0; j < cell_vertex_count(cell.type); ++j)
                    ::idmesh::detail::require_argument(shard.vertices.contains(cell.geometry[j]),
                        "cell references missing global corner vertex");
                for (std::size_t j = cell_vertex_count(cell.type); j < nodes; ++j)
                    ::idmesh::detail::require_argument(shard.geometry_nodes.contains(cell.geometry[j]),
                        "cell references missing high-order geometry node");
                ::idmesh::detail::require_runtime(seen.insert(cell.id).second, "duplicate shard global cell ID");
                shard.cells.push_back(std::move(cell));
            }
            ::idmesh::detail::require_runtime(static_cast<bool>(file >> word >> count) && word == "LABELS",
                "missing shard LABELS section");
            for (std::size_t i = 0; i < count; ++i) {
                Label label;
                std::size_t n = 0;
                ::idmesh::detail::require_runtime(static_cast<bool>(file >> label.dimension >> label.value >> n) && label.dimension <= shard.dimension && n >= 1 && n <= 4,
                    "invalid shard label header");
                label.entities.resize(n);
                for (auto& id : label.entities)
                    ::idmesh::detail::require_runtime(static_cast<bool>(file >> id), "truncated shard label");
                shard.labels.push_back(std::move(label));
            }
            ::idmesh::detail::require_runtime(static_cast<bool>(file >> word >> count) && word == "NAMES",
                "missing shard NAMES section");
            for (std::size_t i = 0; i < count; ++i) {
                Index dimension = 0;
                int value = 0;
                std::string name;
                ::idmesh::detail::require_runtime(static_cast<bool>(file >> dimension >> value >> std::quoted(name))
                        && dimension <= shard.dimension,
                    "invalid shard label name");
                shard.label_names[{dimension, value}] = std::move(name);
            }
            ::idmesh::detail::require_runtime(static_cast<bool>(file >> word) && word == "END",
                "missing shard END marker");
            ::idmesh::detail::require_runtime(!(file >> word), "trailing garbage after shard END");
            ::idmesh::detail::require_runtime(!shard.cells.empty(), "shard requires at least one local cell");
            std::stable_sort(shard.cells.begin(), shard.cells.end(),
                [rank](const Cell& a, const Cell& b) {
                    return std::tuple {a.owner != rank, a.id} < std::tuple {b.owner != rank, b.id};
                });
            return shard;
        }

        struct Key {
            std::array<std::uint64_t, 4> ids {};
            std::uint8_t dim = 0;
            std::uint8_t arity = 0;

            auto tie() const { return std::tie(dim, arity, ids); }
            bool operator<(const Key& other) const { return tie() < other.tie(); }
        };

        struct RegistryMessage {
            Key key {};
            std::uint64_t global_id = 0;
            Index local_entity = invalid_index;
            int requester = -1;
            int proposed_owner = -1;
        };
        static_assert(std::is_trivially_copyable_v<RegistryMessage>);

        std::uint64_t hash_key(const Key& key)
        {
            std::uint64_t hash = 14695981039346656037ULL;
            const auto mix = [&](std::uint64_t value) {
                for (unsigned i = 0; i < 8; ++i) {
                    hash ^= (value >> (8 * i)) & 255ULL;
                    hash *= 1099511628211ULL;
                }
            };
            mix(key.dim);
            mix(key.arity);
            for (std::size_t i = 0; i < key.arity; ++i)
                mix(key.ids[i]);
            return hash;
        }

        struct EntityRequest {
            std::uint64_t global_id = 0;
            Index local_entity = 0;
        };
        static_assert(std::is_trivially_copyable_v<EntityRequest>);

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
            MeshBuilder builder(input.dimension, input.geometric_dimension);
            std::map<std::uint64_t, Index> vertices, high_order, cells;

            // Ascending global corner IDs ensure identical reference orientation.
            for (const auto& [id, position] : input.vertices)
                vertices[id] = builder.add_vertex(position);

            std::vector<std::uint64_t> cell_ids;
            std::vector<int> cell_owners;
            for (const Cell& cell : input.cells) {
                std::vector<Index> corners, extras;
                const Index corner_count = cell_vertex_count(cell.type);
                for (Index j = 0; j < corner_count; ++j)
                    corners.push_back(vertices.at(cell.geometry[j]));
                for (std::size_t j = corner_count; j < cell.geometry.size(); ++j) {
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
                    : builder.add_quadratic_cell(cell.type, corners, extras);
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
                builder.set_label_name(key.first, key.second, name);

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

        IdentityMetadata resolve_entity_identity(MPI_Comm comm,
            int rank, int size, const Mesh& mesh,
            const std::vector<std::uint64_t>& local_to_global_vertex,
            std::vector<std::uint64_t> cell_ids,
            std::vector<int> cell_owners,
            detail::ShardReadProfile* profile)
        {
            const Index dim = mesh.dimension();
            IdentityMetadata metadata;
            metadata.global_ids.resize(dim + 1);
            metadata.owner_ranks.resize(dim + 1);
            metadata.global_ids[dim] = std::move(cell_ids);
            metadata.owner_ranks[dim] = std::move(cell_owners);

            const auto owner_begin
                = profile ? Clock::now() : Clock::time_point {};
            std::vector<std::vector<int>> candidate(dim + 1);
            for (Index k = 0; k <= dim; ++k)
                candidate[k].assign(mesh.entity_count(k), size);
            candidate[dim] = metadata.owner_ranks[dim];
            for (int from = static_cast<int>(dim); from > 0; --from) {
                for (int to = from - 1; to >= 0; --to) {
                    if (!mesh.has_relation(from, to))
                        continue;
                    const auto relation = mesh.relation(from, to);
                    for (Index entity = 0;
                         entity < mesh.entity_count(from); ++entity)
                        for (Index lower : relation[entity].targets)
                            candidate[to][lower] = std::min(
                                candidate[to][lower],
                                candidate[from][entity]);
                }
            }
            if (profile)
                profile->owner_candidates_ms
                    = milliseconds(owner_begin, Clock::now());

            const auto request_begin
                = profile ? Clock::now() : Clock::time_point {};
            std::vector<std::vector<RegistryMessage>> requests(size);
            for (Index k = 0; k < dim; ++k) {
                metadata.global_ids[k].resize(mesh.entity_count(k));
                metadata.owner_ranks[k].resize(mesh.entity_count(k));
                for (Index entity = 0;
                     entity < mesh.entity_count(k); ++entity) {
                    const auto local = mesh.entity_vertices(k, entity);
                    Key key;
                    key.dim = static_cast<std::uint8_t>(k);
                    key.arity = static_cast<std::uint8_t>(local.size());
                    ::idmesh::detail::require_argument(
                        key.arity > 0 && key.arity <= 4,
                        "invalid shared entity arity");
                    for (std::size_t j = 0; j < local.size(); ++j)
                        key.ids[j] = local_to_global_vertex[local[j]];
                    std::sort(
                        key.ids.begin(), key.ids.begin() + key.arity);
                    const int service
                        = static_cast<int>(hash_key(key) % size);
                    requests[service].push_back(
                        {key, 0, entity, rank, candidate[k][entity]});
                }
            }

            const auto received = exchange_records(comm, requests,
                profile ? &profile->registry_send_bytes : nullptr,
                profile ? &profile->registry_receive_bytes : nullptr);
            if (profile)
                profile->registry_request_ms
                    = milliseconds(request_begin, Clock::now());

            const auto resolve_begin
                = profile ? Clock::now() : Clock::time_point {};
            std::map<Key, std::vector<RegistryMessage>> groups;
            for (const auto& item : received.values)
                groups[item.key].push_back(item);

            std::vector<unsigned long long> local_unique(dim), prefix(dim);
            for (const auto& [key, members] : groups) {
                (void)members;
                if (key.dim > 0)
                    ++local_unique[key.dim];
            }
            for (Index k = 1; k < dim; ++k) {
                ::idmesh::mpi::detail::check_mpi(
                    MPI_Exscan(&local_unique[k], &prefix[k], 1,
                        MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm),
                    "MPI_Exscan");
                if (rank == 0)
                    prefix[k] = 0;
            }

            std::vector<std::vector<RegistryMessage>> replies(size);
            std::vector<std::uint64_t> next(dim);
            for (const auto& [key, members] : groups) {
                const std::uint64_t global = key.dim == 0
                    ? key.ids[0]
                    : static_cast<std::uint64_t>(prefix[key.dim])
                        + next[key.dim]++;
                int min_owner = size;
                for (const auto& item : members)
                    min_owner = std::min(min_owner,
                        item.proposed_owner < size
                            ? item.proposed_owner
                            : item.requester);
                ::idmesh::detail::require_runtime(
                    min_owner >= 0 && min_owner < size,
                    "invalid entity owner");
                for (auto item : members) {
                    item.global_id = global;
                    item.proposed_owner = min_owner;
                    replies[item.requester].push_back(item);
                }
            }
            if (profile)
                profile->registry_resolve_ms
                    = milliseconds(resolve_begin, Clock::now());

            const auto reply_begin
                = profile ? Clock::now() : Clock::time_point {};
            const auto answers = exchange_records(comm, replies,
                profile ? &profile->registry_send_bytes : nullptr,
                profile ? &profile->registry_receive_bytes : nullptr);

            std::vector<std::vector<bool>> assigned(dim);
            for (Index k = 0; k < dim; ++k)
                assigned[k].resize(metadata.global_ids[k].size());
            for (const auto& answer : answers.values) {
                const auto k = answer.key.dim;
                const auto entity = answer.local_entity;
                ::idmesh::detail::require_runtime(
                    k < dim
                        && entity < metadata.global_ids[k].size()
                        && !assigned[k][entity],
                    "duplicate or invalid distributed entity response");
                assigned[k][entity] = true;
                metadata.global_ids[k][entity] = answer.global_id;
                metadata.owner_ranks[k][entity] = answer.proposed_owner;
            }
            for (Index k = 0; k < dim; ++k)
                ::idmesh::detail::require_runtime(
                    std::all_of(assigned[k].begin(), assigned[k].end(),
                        [](bool value) { return value; }),
                    "missing distributed entity response");
            if (profile)
                profile->registry_reply_ms
                    = milliseconds(reply_begin, Clock::now());

            return metadata;
        }

        struct ExchangeLayout {
            std::vector<Index> owned;
            std::vector<std::vector<Index>> send;
            std::vector<std::vector<Index>> recv;
        };

        std::vector<ExchangeLayout> build_exchange_layouts(
            MPI_Comm comm, int rank, int size, const Mesh& mesh,
            const IdentityMetadata& metadata,
            detail::ShardReadProfile* profile)
        {
            const Index dim = mesh.dimension();
            std::vector<ExchangeLayout> layouts(dim + 1);
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
                layout.recv.resize(size);
                std::map<std::uint64_t, Index> owned_lookup;
                std::vector<std::vector<EntityRequest>>
                    ghost_requests(size);
                for (Index entity = 0; entity < count; ++entity) {
                    const int owner = metadata.owner_ranks[k][entity];
                    ::idmesh::detail::require_runtime(
                        owner >= 0 && owner < size,
                        "invalid distributed entity owner");
                    const auto global = metadata.global_ids[k][entity];
                    if (owner == rank) {
                        layout.owned.push_back(entity);
                        ::idmesh::detail::require_runtime(
                            owned_lookup.emplace(global, entity).second,
                            "duplicate owned global entity ID");
                    }
                    else {
                        ghost_requests[owner].push_back(
                            {global, entity});
                    }
                }

                const auto incoming = exchange_records(
                    comm, ghost_requests,
                    profile
                        ? &profile->exchange_plan_send_bytes
                        : nullptr,
                    profile
                        ? &profile->exchange_plan_receive_bytes
                        : nullptr);
                for (int peer = 0; peer < size; ++peer) {
                    for (const auto& request : ghost_requests[peer])
                        layout.recv[peer].push_back(
                            request.local_entity);
                    const int first = incoming.offsets[peer];
                    const int received_count = incoming.counts[peer];
                    for (int i = 0; i < received_count; ++i) {
                        const auto& request
                            = incoming.values[first + i];
                        const auto found
                            = owned_lookup.find(request.global_id);
                        ::idmesh::detail::require_runtime(
                            found != owned_lookup.end(),
                            "requested ghost owner lacks its entity");
                        layout.send[peer].push_back(found->second);
                    }
                }
            }
            return layouts;
        }

    } // namespace

    PartitionedMesh read_partition_shard(
        const Context& context, const std::string& path, BuildOptions options)
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
            profile->parse_ms = milliseconds(parse_begin, Clock::now());

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

        auto metadata = resolve_entity_identity(
            comm, rank, size, local.mesh,
            local.local_to_global_vertex,
            std::move(local.cell_ids),
            std::move(local.cell_owners), profile);

        const auto exchange_begin
            = profile ? Clock::now() : Clock::time_point {};
        auto layouts = build_exchange_layouts(
            comm, rank, size, local.mesh, metadata, profile);

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
            auto exchange = std::make_shared<ExchangePlan::Data>();
            exchange->comm = comm;
            exchange->context_owner = context.data_;
            exchange->send = std::move(layouts[k].send);
            exchange->recv = std::move(layouts[k].recv);
            exchange->local_entities
                = partition->mesh.entity_count(k);
            exchange->finalize_layout();
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
            return read_partition_shard(context, path, options);
        }

    } // namespace detail

} // namespace idmesh::mpi
