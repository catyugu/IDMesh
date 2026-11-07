#include "contracts.hpp"
#include "idmesh/mpi.hpp"
#include "mpi_errors.hpp"
#include "mpi_internal.hpp"

#include <algorithm>
#include <array>
#include <climits>
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

        struct ReceivedBytes {
            std::vector<int> counts;
            std::vector<int> displacements;
        };

        template <class T>
        struct Received {
            std::vector<T> values;
            std::vector<int> counts;
            std::vector<int> offsets;
        };

        template <class T>
        Received<T> exchange_records(MPI_Comm comm, const std::vector<std::vector<T>>& sends)
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

        struct Request {
            std::uint64_t global_cell = 0;
            Index local_ghost = 0;
        };
        static_assert(std::is_trivially_copyable_v<Request>);

    } // namespace

    PartitionedMesh read_partition_shard(
        const Context& context, const std::string& path, BuildOptions options)
    {
        const int rank = context.rank(), size = context.size();
        const MPI_Comm comm = context.data_->comm;

        const Shard input = parse_shard(path, rank, size);
        const Index dim = input.dimension;
        Index min_dim = 0, max_dim = 0;
        ::idmesh::mpi::detail::check_mpi(MPI_Allreduce(&dim, &min_dim, 1, MPI_UNSIGNED, MPI_MIN, comm), "MPI_Allreduce");
        ::idmesh::mpi::detail::check_mpi(MPI_Allreduce(&dim, &max_dim, 1, MPI_UNSIGNED, MPI_MAX, comm), "MPI_Allreduce");
        ::idmesh::detail::require_runtime(min_dim == max_dim, "shards disagree on topological dimension");

        MeshBuilder builder(input.dimension, input.geometric_dimension);
        std::map<std::uint64_t, Index> vertices, high_order, cells;
        // Ascending global corner IDs ensure identical reference orientation.
        for (const auto& [id, position] : input.vertices)
            vertices[id] = builder.add_vertex(position);

        Index owned = 0;
        std::vector<std::uint64_t> cell_ids;
        std::vector<int> cell_owners;
        for (const Cell& cell : input.cells) {
            std::vector<Index> corners, extras;
            const Index corner_count = cell_vertex_count(cell.type);
            for (Index j = 0; j < corner_count; ++j)
                corners.push_back(vertices.at(cell.geometry[j]));
            for (std::size_t j = corner_count; j < cell.geometry.size(); ++j) {
                const auto id = cell.geometry[j];
                auto [it, inserted] = high_order.try_emplace(id, invalid_index);
                if (inserted)
                    it->second = builder.add_geometry_node(input.geometry_nodes.at(id));
                extras.push_back(it->second);
            }
            const Index local = extras.empty()
                ? builder.add_cell(cell.type, corners)
                : builder.add_quadratic_cell(cell.type, corners, extras);
            ::idmesh::detail::require_runtime(local == cell_ids.size(), "unexpected shard builder cell ordering");
            cells[cell.id] = local;
            cell_ids.push_back(cell.id);
            cell_owners.push_back(cell.owner);
            if (cell.owner == rank)
                ++owned;
        }
        for (const Label& label : input.labels) {
            if (label.dimension == dim) {
                ::idmesh::detail::require_argument(label.entities.size() == 1 && cells.contains(label.entities[0]), "invalid top-cell label reference");
                builder.add_cell_label(cells.at(label.entities[0]), label.value);
            }
            else {
                std::vector<Index> corners;
                for (auto id : label.entities) {
                    ::idmesh::detail::require_runtime(vertices.contains(id), "label refers to missing shard vertex");
                    corners.push_back(vertices.at(id));
                }
                builder.add_label(label.dimension, corners, label.value);
            }
        }
        for (const auto& [key, name] : input.label_names)
            builder.set_label_name(key.first, key.second, name);

        auto partition = std::make_shared<PartitionedMesh::Data>(
            std::move(builder).finalize(options));
        partition->owned_cells = owned;
        partition->global_ids.resize(dim + 1);
        partition->owner_ranks.resize(dim + 1);
        partition->global_ids[dim] = std::move(cell_ids);
        partition->owner_ranks[dim] = std::move(cell_owners);

        std::vector<std::uint64_t> local_to_global_vertex(vertices.size());
        for (const auto& [global, local] : vertices)
            local_to_global_vertex[local] = global;

        // Propagate the owning Rank of each incident cell down the *materialized*
        // boundary graph. The distributed registry resolves incomplete halos and
        // vertices shared by multiple ranks with no common facet ghost.
        std::vector<std::vector<int>> candidate(dim + 1);
        for (Index k = 0; k <= dim; ++k)
            candidate[k].assign(partition->mesh.entity_count(k), size);
        candidate[dim] = partition->owner_ranks[dim];
        for (int from = static_cast<int>(dim); from > 0; --from) {
            for (int to = from - 1; to >= 0; --to) {
                if (!partition->mesh.has_relation(from, to))
                    continue;
                const auto relation = partition->mesh.relation(from, to);
                for (Index e = 0; e < partition->mesh.entity_count(from); ++e)
                    for (Index lower : relation[e].targets)
                        candidate[to][lower] = std::min(candidate[to][lower], candidate[from][e]);
            }
        }

        std::vector<std::vector<RegistryMessage>> requests(size);
        for (Index k = 0; k < dim; ++k) {
            partition->global_ids[k].resize(partition->mesh.entity_count(k));
            partition->owner_ranks[k].resize(partition->mesh.entity_count(k));
            for (Index e = 0; e < partition->mesh.entity_count(k); ++e) {
                const auto local = partition->mesh.entity_vertices(k, e);
                Key key;
                key.dim = static_cast<std::uint8_t>(k);
                key.arity = static_cast<std::uint8_t>(local.size());
                ::idmesh::detail::require_argument(key.arity > 0 && key.arity <= 4, "invalid shared entity arity");
                for (std::size_t j = 0; j < local.size(); ++j)
                    key.ids[j] = local_to_global_vertex[local[j]];
                std::sort(key.ids.begin(), key.ids.begin() + key.arity);
                const int service = static_cast<int>(hash_key(key) % size);
                requests[service].push_back(
                    {key, 0, e, rank, candidate[k][e]});
            }
        }

        const auto received = exchange_records(comm, requests);
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
            ::idmesh::mpi::detail::check_mpi(MPI_Exscan(&local_unique[k], &prefix[k], 1,
                                                 MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm),
                "MPI_Exscan");
            if (rank == 0)
                prefix[k] = 0;
        }

        std::vector<std::vector<RegistryMessage>> replies(size);
        std::vector<std::uint64_t> next(dim);
        for (const auto& [key, members] : groups) {
            std::uint64_t global = key.dim == 0 ? key.ids[0] : static_cast<std::uint64_t>(prefix[key.dim]) + next[key.dim]++;
            int min_owner = size;
            for (const auto& item : members) {
                min_owner = std::min(min_owner, item.proposed_owner < size ? item.proposed_owner : item.requester);
            }
            ::idmesh::detail::require_runtime(min_owner >= 0 && min_owner < size, "invalid entity owner");
            for (auto item : members) {
                item.global_id = global;
                item.proposed_owner = min_owner;
                replies[item.requester].push_back(item);
            }
        }
        const auto answers = exchange_records(comm, replies);
        std::vector<std::vector<bool>> assigned(dim);
        for (Index k = 0; k < dim; ++k)
            assigned[k].resize(partition->global_ids[k].size());
        for (const auto& answer : answers.values) {
            const auto k = answer.key.dim;
            const auto e = answer.local_entity;
            ::idmesh::detail::require_runtime(k < dim && e < partition->global_ids[k].size() && !assigned[k][e], "duplicate or invalid distributed entity response");
            assigned[k][e] = true;
            partition->global_ids[k][e] = answer.global_id;
            partition->owner_ranks[k][e] = answer.proposed_owner;
        }
        for (Index k = 0; k < dim; ++k)
            ::idmesh::detail::require_argument(std::all_of(assigned[k].begin(), assigned[k].end(),
                                                   [](bool value) { return value; }),
                "missing distributed entity response");

        std::map<std::uint64_t, Index> owned_cells;
        for (Index c = 0; c < owned; ++c)
            ::idmesh::detail::require_argument(owned_cells.emplace(partition->global_ids[dim][c], c).second,
                "duplicate owned global cell ID");
        std::vector<std::vector<Request>> ghost_requests(size);
        for (Index c = owned; c < partition->mesh.entity_count(dim); ++c) {
            const int owner = partition->owner_ranks[dim][c];
            ::idmesh::detail::require_argument(owner != rank, "ghost cell wrongly owned by local rank");
            ghost_requests[owner].push_back({partition->global_ids[dim][c], c});
        }
        const auto incoming_ghosts = exchange_records(comm, ghost_requests);
        auto exchange = std::make_shared<ExchangePlan::Data>();
        exchange->comm = comm;
        exchange->context_owner = context.data_;
        exchange->send.resize(size);
        exchange->recv.resize(size);
        exchange->local_cells = partition->mesh.entity_count(dim);
        for (int peer = 0; peer < size; ++peer) {
            for (const auto& request : ghost_requests[peer])
                exchange->recv[peer].push_back(request.local_ghost);
            const int first = incoming_ghosts.offsets[peer];
            const int count = incoming_ghosts.counts[peer];
            for (int i = 0; i < count; ++i) {
                const auto& request = incoming_ghosts.values[first + i];
                const auto found = owned_cells.find(request.global_cell);
                ::idmesh::detail::require_runtime(found != owned_cells.end(), "requested ghost owner lacks its cell");
                exchange->send[peer].push_back(found->second);
            }
        }
        detail::finalize_exchange_layout(*exchange);
        partition->exchange = ExchangePlan(std::move(exchange));
        partition->mesh.validate();
        return PartitionedMesh(std::move(partition));
    }

} // namespace idmesh::mpi
