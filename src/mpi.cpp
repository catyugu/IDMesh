#include "idmesh/mpi.hpp"
#include "mpi_internal.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <map>
#include <mpi.h>
#if defined(OPEN_MPI) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#include <numeric>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace idmesh::mpi {
    namespace {
        void require(bool ok, const char* reason)
        {
            if (!ok)
                throw std::runtime_error(reason);
        }
        void check_mpi(int result)
        {
            if (result != MPI_SUCCESS)
                throw std::runtime_error("IDMesh MPI operation failed");
        }
        void ensure_initialized()
        {
            int initialized = 0;
            check_mpi(MPI_Initialized(&initialized));
            require(initialized != 0, "MPI must be initialized by the application");
        }
        using GlobalId = std::uint64_t;
        std::vector<Index> canonical_key(std::span<const Index> vertices)
        {
            std::vector<Index> key(vertices.begin(), vertices.end());
            std::sort(key.begin(), key.end());
            return key;
        }
    } // namespace

    int world_rank()
    {
        ensure_initialized();
        int rank = 0;
        check_mpi(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
        return rank;
    }
    int world_size()
    {
        ensure_initialized();
        int size = 0;
        check_mpi(MPI_Comm_size(MPI_COMM_WORLD, &size));
        return size;
    }

    ExchangePlan::ExchangePlan(std::shared_ptr<Data> data)
        : data_(std::move(data))
    {
    }
    bool ExchangePlan::empty() const noexcept
    {
        if (!data_)
            return true;
        for (auto& r : data_->recv)
            if (!r.empty())
                return false;
        for (auto& s : data_->send)
            if (!s.empty())
                return false;
        return true;
    }
    bool gpu_aware_support(DeviceMemoryKind memory)
    {
        ensure_initialized();
        if (memory == DeviceMemoryKind::Host)
            return true;
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
        if (memory == DeviceMemoryKind::CUDA)
            return MPIX_Query_cuda_support() == 1;
#endif
#if defined(OMPI_HAVE_MPI_EXT_ROCM) && OMPI_HAVE_MPI_EXT_ROCM
        if (memory == DeviceMemoryKind::HIP)
            return MPIX_Query_rocm_support() == 1;
#endif
        return false;
    }

    int ExchangePlan::peer_count() const noexcept
    {
        return data_ ? static_cast<int>(data_->send.size()) : 0;
    }
    Index ExchangePlan::local_cell_count() const noexcept
    {
        return data_ ? data_->local_cells : 0;
    }
    std::span<const Index> ExchangePlan::send_indices(int peer) const
    {
        if (!data_ || peer < 0 || peer >= peer_count())
            throw std::out_of_range("invalid halo send peer");
        return data_->send[peer];
    }
    std::span<const Index> ExchangePlan::receive_indices(int peer) const
    {
        if (!data_ || peer < 0 || peer >= peer_count())
            throw std::out_of_range("invalid halo receive peer");
        return data_->recv[peer];
    }

    void ExchangePlan::exchange_packed(
        const void* send, void* receive, Index components,
        DeviceMemoryKind memory) const
    {
        ensure_initialized();
        require(data_ != nullptr && components > 0, "invalid packed halo exchange");
        if (!gpu_aware_support(memory))
            throw std::runtime_error("MPI is not GPU aware for the requested memory space");
        const int peers = peer_count();
        std::vector<int> sc(peers), rc(peers), sd(peers), rd(peers);
        std::size_t send_size = 0, receive_size = 0;
        for (int peer = 0; peer < peers; ++peer) {
            const auto sending = data_->send[peer].size() * components;
            const auto receiving = data_->recv[peer].size() * components;
            require(sending <= INT_MAX && receiving <= INT_MAX &&
                    send_size <= static_cast<std::size_t>(INT_MAX) - sending &&
                    receive_size <= static_cast<std::size_t>(INT_MAX) - receiving,
                "packed halo exchange exceeds MPI count/displacement limits");
            sc[peer] = static_cast<int>(sending);
            rc[peer] = static_cast<int>(receiving);
            sd[peer] = static_cast<int>(send_size);
            rd[peer] = static_cast<int>(receive_size);
            send_size += sending;
            receive_size += receiving;
        }
        check_mpi(MPI_Alltoallv(send, sc.data(), sd.data(), MPI_DOUBLE,
            receive, rc.data(), rd.data(), MPI_DOUBLE, MPI_COMM_WORLD));
    }

    void ExchangePlan::exchange_cells(std::span<double> values, Index components) const
    {
        ensure_initialized();
        require(data_ != nullptr, "empty exchange plan");
        require(components > 0 && values.size() == std::size_t(data_->local_cells) * components,
            "exchange field must have one fixed-width record per local cell");
        const auto size = static_cast<int>(data_->send.size());
        std::vector<int> sc(size), rc(size), sd(size), rd(size);
        for (int peer = 0; peer < size; ++peer) {
            require(data_->send[peer].size() <= INT_MAX / components && data_->recv[peer].size() <= INT_MAX / components,
                "exchange count exceeds MPI int range");
            sc[peer] = static_cast<int>(data_->send[peer].size() * components);
            rc[peer] = static_cast<int>(data_->recv[peer].size() * components);
            if (peer > 0) {
                require(sd[peer - 1] <= INT_MAX - sc[peer - 1] && rd[peer - 1] <= INT_MAX - rc[peer - 1],
                    "MPI exchange displacement overflow");
                sd[peer] = sd[peer - 1] + sc[peer - 1];
                rd[peer] = rd[peer - 1] + rc[peer - 1];
            }
        }
        const auto sendTotal = sd.back() + sc.back();
        const auto recvTotal = rd.back() + rc.back();
        std::vector<double> sendBuffer(sendTotal), recvBuffer(recvTotal);
        for (int peer = 0; peer < size; ++peer) {
            std::size_t output = sd[peer];
            for (auto cell : data_->send[peer]) {
                for (Index j = 0; j < components; ++j)
                    sendBuffer[output++] = values[std::size_t(cell) * components + j];
            }
        }
        check_mpi(MPI_Alltoallv(sendBuffer.data(), sc.data(), sd.data(), MPI_DOUBLE,
            recvBuffer.data(), rc.data(), rd.data(), MPI_DOUBLE,
            MPI_COMM_WORLD));
        for (int peer = 0; peer < size; ++peer) {
            std::size_t input = rd[peer];
            for (auto cell : data_->recv[peer]) {
                for (Index j = 0; j < components; ++j)
                    values[std::size_t(cell) * components + j] = recvBuffer[input++];
            }
        }
    }

    PartitionedMesh partition_replicated(const Mesh& global, Index ghost_layers)
    {
        ensure_initialized();
        const int rank = world_rank(), size = world_size();
        const Index d = global.dimension();
        const Index cells = global.entity_count(d);
        require(cells > 0, "replicated partition source has no cells");
        require(ghost_layers <= 8, "ghost depth exceeds safety limit");
        require(global.has_relation(d, d - 1) && global.has_relation(d - 1, d),
            "partition requires materialized cell-facet and facet-cell relations");

        // Detect obvious rank-dependent input mistakes before collective exchanges.
        std::uint64_t fingerprint = 14695981039346656037ull;
        const auto mix = [&](std::uint64_t value) {
            fingerprint ^= value;
            fingerprint *= 1099511628211ull;
        };
        mix(d);
        mix(cells);
        mix(global.entity_count(0));
        for (Index c = 0; c < cells; ++c) {
            mix(static_cast<int>(global.cell_types()[c]));
            for (auto v : global.entity_vertices(d, c))
                mix(v);
        }
        unsigned long long local = fingerprint, low = 0, high = 0;
        check_mpi(MPI_Allreduce(&local, &low, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, MPI_COMM_WORLD));
        check_mpi(MPI_Allreduce(&local, &high, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD));
        require(low == high, "replicated source mesh identities differ across ranks");

        const auto owner = [&](Index c) {
            return static_cast<int>(std::uint64_t(c) * size / cells);
        };
        const auto facets = global.relation(d, d - 1), support = global.relation(d - 1, d);
        std::vector<std::set<Index>> needed(size);
        for (int peer = 0; peer < size; ++peer) {
            std::vector<bool> seen(cells, false);
            std::vector<Index> frontier;
            for (Index c = 0; c < cells; ++c)
                if (owner(c) == peer) {
                    seen[c] = true;
                    frontier.push_back(c);
                }
            for (Index level = 0; level < ghost_layers; ++level) {
                std::vector<Index> next;
                for (auto c : frontier)
                    for (auto f : facets.row(c))
                        for (auto neighbor : support.row(f)) {
                            if (!seen[neighbor]) {
                                seen[neighbor] = true;
                                next.push_back(neighbor);
                                if (owner(neighbor) != peer)
                                    needed[peer].insert(neighbor);
                            }
                        }
                frontier = std::move(next);
            }
        }
        std::vector<Index> order;
        order.reserve(cells);
        for (Index c = 0; c < cells; ++c)
            if (owner(c) == rank)
                order.push_back(c);
        const Index owned_count = static_cast<Index>(order.size());
        for (auto c : needed[rank])
            order.push_back(c);

        MeshBuilder builder(d, global.geometric_dimension());
        std::map<Index, Index> local_vertices, local_extras, local_cells;
        // Enumerate local corner vertices by ascending source/global ID, not
        // cell discovery order. This order-preserving renumbering guarantees
        // canonical local edges/faces have exactly the same orientation on
        // every Rank that shares them.
        std::set<Index> used_vertices;
        for (Index global_cell : order)
            for (Index vertex : global.entity_vertices(d, global_cell))
                used_vertices.insert(vertex);
        for (Index vertex : used_vertices)
            local_vertices.emplace(vertex, builder.add_vertex(global.coordinates()[vertex]));
        for (Index global_cell : order) {
            std::vector<Index> vertices, extras;
            for (auto v : global.entity_vertices(d, global_cell))
                vertices.push_back(local_vertices.at(v));
            const auto geometry = global.cell_geometry_nodes(global_cell);
            for (std::size_t k = vertices.size(); k < geometry.size(); ++k) {
                const Index id = geometry[k];
                auto [it, inserted] = local_extras.try_emplace(id, invalid_index);
                if (inserted)
                    it->second = builder.add_geometry_node(global.geometry_node_position(id));
                extras.push_back(it->second);
            }
            const auto tag = global.cell_tags()[global_cell];
            Index local_cell = invalid_index;
            if (global.cell_order(global_cell) == 1)
                local_cell = builder.add_cell(global.cell_types()[global_cell], vertices, tag);
            else
                local_cell = builder.add_quadratic_cell(global.cell_types()[global_cell],
                    vertices, extras, tag);
            local_cells[global_cell] = local_cell;
            for (int extra_tag : global.physical_tags(d, global_cell))
                builder.add_cell_physical_tag(local_cell, extra_tag);
        }
        for (const auto& label : global.physical_labels()) {
            if (label.dimension == d)
                continue;
            std::vector<Index> vertices;
            for (auto id : label.vertices) {
                const auto found = local_vertices.find(id);
                if (found == local_vertices.end())
                    break;
                vertices.push_back(found->second);
            }
            if (vertices.size() == label.vertices.size())
                builder.add_physical_tag(label.dimension, vertices, label.tag);
            if (auto name = global.physical_name(label.dimension, label.tag); !name.empty())
                builder.set_physical_name(label.dimension, label.tag, std::move(name));
        }
        TopologyOptions opts = TopologyOptions::cells_and_vertices();
        opts.facets = global.has_entities(d - 1);
        opts.edges = d >= 2 && global.has_entities(1);
        opts.faces = d == 3 && global.has_entities(2);
        opts.support = true;
        PartitionedMesh partition;
        partition.mesh = std::move(builder).finalize(opts);
        partition.owned_cells = owned_count;
        partition.global_ids.resize(d + 1);
        partition.owner_ranks.resize(d + 1);
        partition.global_ids[d].assign(order.begin(), order.end());
        for (auto id : order)
            partition.owner_ranks[d].push_back(owner(id));

        // Stable dimension-local shared identity from canonical *global* corners.
        // Vertex and cell IDs are the original replicated source indices.
        std::vector<std::map<std::vector<Index>, Index>> lookup(d);
        for (Index k = 1; k < d; ++k)
            for (Index e = 0; e < global.entity_count(k); ++e)
                lookup[k].emplace(canonical_key(global.entity_vertices(k, e)), e);
        std::vector<Index> reverse_vertices(partition.mesh.entity_count(0));
        for (const auto& [g, l] : local_vertices)
            reverse_vertices[l] = g;

        // Each shared subentity is owned by the lowest owning incident cell.
        std::vector<std::vector<int>> global_owners(d + 1);
        for (Index k = 0; k <= d; ++k)
            global_owners[k].assign(global.entity_count(k), size);
        for (Index c = 0; c < cells; ++c)
            global_owners[d][c] = owner(c);
        for (int from = static_cast<int>(d); from > 0; --from) {
            for (int to = from - 1; to >= 0; --to) {
                if (!global.has_relation(from, to))
                    continue;
                auto rel = global.relation(from, to);
                for (Index higher = 0; higher < global.entity_count(from); ++higher)
                    for (auto lower : rel.row(higher))
                        global_owners[to][lower] = std::min(
                            global_owners[to][lower], global_owners[from][higher]);
            }
        }
        for (Index k = 0; k < d; ++k) {
            for (Index e = 0; e < partition.mesh.entity_count(k); ++e) {
                Index global_id = invalid_index;
                if (k == 0)
                    global_id = reverse_vertices[e];
                else {
                    std::vector<Index> key;
                    for (auto v : partition.mesh.entity_vertices(k, e))
                        key.push_back(reverse_vertices[v]);
                    std::sort(key.begin(), key.end());
                    auto it = lookup[k].find(key);
                    require(it != lookup[k].end(), "cross-rank shared entity identity mismatch");
                    global_id = it->second;
                }
                partition.global_ids[k].push_back(global_id);
                partition.owner_ranks[k].push_back(global_owners[k][global_id]);
            }
        }
        auto exchange = std::make_shared<ExchangePlan::Data>();
        exchange->send.resize(size);
        exchange->recv.resize(size);
        exchange->local_cells = static_cast<Index>(order.size());
        for (int peer = 0; peer < size; ++peer) {
            if (peer == rank)
                continue;
            for (Index id : needed[peer])
                if (owner(id) == rank)
                    exchange->send[peer].push_back(local_cells.at(id));
            for (Index id : needed[rank])
                if (owner(id) == peer)
                    exchange->recv[peer].push_back(local_cells.at(id));
        }
        partition.exchange = ExchangePlan(std::move(exchange));
        return partition;
    }

} // namespace idmesh::mpi
