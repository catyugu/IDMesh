#include "idmesh/mpi.hpp"
#include "contracts.hpp"
#include "mpi_errors.hpp"
#include "mpi_internal.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <limits>
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
        void ensure_initialized()
        {
            int initialized = 0;
            ::idmesh::mpi::detail::check_mpi(MPI_Initialized(&initialized), "MPI_Initialized");
            ::idmesh::detail::require_state(initialized != 0, "MPI must be initialized by the application");
        }
        using GlobalId = std::uint64_t;
        std::vector<Index> canonical_key(std::span<const Index> vertices)
        {
            std::vector<Index> key(vertices.begin(), vertices.end());
            std::sort(key.begin(), key.end());
            return key;
        }
    } // namespace

    namespace {
        void scaled_layout(const auto& data, Index components,
            std::vector<int>& send_counts, std::vector<int>& recv_counts,
            std::vector<int>& send_offsets, std::vector<int>& recv_offsets)
        {
            const int peers = static_cast<int>(data.send.size());
            send_counts.resize(peers);
            recv_counts.resize(peers);
            send_offsets.resize(peers);
            recv_offsets.resize(peers);
            for (int peer = 0; peer < peers; ++peer) {
                ::idmesh::detail::require_argument(
                    data.send_records[peer] <= INT_MAX / components
                        && data.recv_records[peer] <= INT_MAX / components
                        && data.send_record_offsets[peer] <= INT_MAX / components
                        && data.recv_record_offsets[peer] <= INT_MAX / components,
                    "halo exchange exceeds MPI count/displacement limits");
                send_counts[peer] = data.send_records[peer] * static_cast<int>(components);
                recv_counts[peer] = data.recv_records[peer] * static_cast<int>(components);
                send_offsets[peer] = data.send_record_offsets[peer] * static_cast<int>(components);
                recv_offsets[peer] = data.recv_record_offsets[peer] * static_cast<int>(components);
            }
        }

        bool query_direct_device_memory(DeviceMemoryKind memory)
        {
            if (memory == DeviceMemoryKind::host)
                return true;
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
            if (memory == DeviceMemoryKind::cuda)
                return MPIX_Query_cuda_support() == 1;
#endif
#if defined(OMPI_HAVE_MPI_EXT_ROCM) && OMPI_HAVE_MPI_EXT_ROCM
            if (memory == DeviceMemoryKind::hip)
                return MPIX_Query_rocm_support() == 1;
#endif
            return false;
        }

    } // namespace

    Context::Context(std::shared_ptr<Data> data)
        : data_(std::move(data))
    {
    }

    Context Context::world()
    {
        ensure_initialized();
        auto data = std::make_shared<Data>();
        ::idmesh::mpi::detail::check_mpi(MPI_Comm_dup(MPI_COMM_WORLD, &data->comm), "MPI_Comm_dup");
        ::idmesh::mpi::detail::check_mpi(
            MPI_Comm_set_errhandler(data->comm, MPI_ERRORS_RETURN), "MPI_Comm_set_errhandler");
        ::idmesh::mpi::detail::check_mpi(MPI_Comm_rank(data->comm, &data->rank), "MPI_Comm_rank");
        ::idmesh::mpi::detail::check_mpi(MPI_Comm_size(data->comm, &data->size), "MPI_Comm_size");
        return Context(std::move(data));
    }

    Context Context::from_fortran_handle(std::int64_t handle)
    {
        ensure_initialized();
        if (handle < std::numeric_limits<MPI_Fint>::min()
            || handle > std::numeric_limits<MPI_Fint>::max())
            throw std::out_of_range("MPI Fortran communicator handle");
        const MPI_Comm source = MPI_Comm_f2c(static_cast<MPI_Fint>(handle));
        ::idmesh::detail::require_argument(source != MPI_COMM_NULL, "MPI communicator is null");
        auto data = std::make_shared<Data>();
        ::idmesh::mpi::detail::check_mpi(MPI_Comm_dup(source, &data->comm), "MPI_Comm_dup");
        ::idmesh::mpi::detail::check_mpi(
            MPI_Comm_set_errhandler(data->comm, MPI_ERRORS_RETURN), "MPI_Comm_set_errhandler");
        ::idmesh::mpi::detail::check_mpi(MPI_Comm_rank(data->comm, &data->rank), "MPI_Comm_rank");
        ::idmesh::mpi::detail::check_mpi(MPI_Comm_size(data->comm, &data->size), "MPI_Comm_size");
        return Context(std::move(data));
    }

    int Context::rank() const noexcept { return data_->rank; }
    int Context::size() const noexcept { return data_->size; }
    ExchangeRequest::ExchangeRequest(std::shared_ptr<Data> data)
        : data_(std::move(data))
    {
    }
    ExchangeRequest::~ExchangeRequest() = default;
    ExchangeRequest::ExchangeRequest(ExchangeRequest&&) noexcept = default;
    ExchangeRequest& ExchangeRequest::operator=(ExchangeRequest&&) noexcept = default;

    void ExchangeRequest::Data::unpack()
    {
        if (!active)
            return;
        for (int peer = 0; peer < plan.rank_count(); ++peer) {
            std::size_t input = recv_offsets[peer];
            for (Index cell : plan.receive_indices(peer))
                for (Index component = 0; component < components; ++component)
                    values[static_cast<std::size_t>(cell) * components + component]
                        = recv_buffer[input++];
        }
        values = {};
        active = false;
    }

    ExchangeRequest::Data::~Data()
    {
        if (!active)
            return;
        int finalized = 0;
        if (MPI_Finalized(&finalized) != MPI_SUCCESS || finalized)
            return;
        if (request == MPI_REQUEST_NULL
            || MPI_Wait(&request, MPI_STATUS_IGNORE) == MPI_SUCCESS)
            unpack();
    }

    void ExchangeRequest::start(std::span<double> values)
    {
        ensure_initialized();
        ::idmesh::detail::require_state(data_ != nullptr, "empty exchange request");
        ::idmesh::detail::require_state(!data_->active,
            "exchange request is already active");
        ::idmesh::detail::require_argument(
            values.size()
                == std::size_t(data_->plan.local_cell_count()) * data_->components,
            "exchange field must have one fixed-width record per local cell");

        data_->values = values;
        for (int peer = 0; peer < data_->plan.rank_count(); ++peer) {
            std::size_t output = data_->send_offsets[peer];
            for (Index cell : data_->plan.send_indices(peer))
                for (Index component = 0; component < data_->components; ++component)
                    data_->send_buffer[output++]
                        = values[static_cast<std::size_t>(cell) * data_->components
                            + component];
        }

        MPI_Request request = MPI_REQUEST_NULL;
        ::idmesh::mpi::detail::check_mpi(
            MPI_Ialltoallv(data_->send_buffer.data(), data_->send_counts.data(),
                data_->send_offsets.data(), MPI_DOUBLE,
                data_->recv_buffer.data(), data_->recv_counts.data(),
                data_->recv_offsets.data(), MPI_DOUBLE, data_->plan.data_->comm,
                &request),
            "MPI_Ialltoallv");
        data_->request = request;
        data_->active = true;
    }

    void ExchangeRequest::wait()
    {
        ensure_initialized();
        ::idmesh::detail::require_state(data_ != nullptr, "empty exchange request");
        if (!data_->active)
            return;
        if (data_->request != MPI_REQUEST_NULL) {
            ::idmesh::mpi::detail::check_mpi(
                MPI_Wait(&data_->request, MPI_STATUS_IGNORE), "MPI_Wait");
        }
        data_->unpack();
    }

    bool ExchangeRequest::active() const noexcept
    {
        return data_ != nullptr && data_->active;
    }

    Index ExchangeRequest::component_count() const noexcept
    {
        return data_ ? data_->components : 0;
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
    int ExchangePlan::rank_count() const noexcept
    {
        return data_ ? static_cast<int>(data_->send.size()) : 0;
    }
    Index ExchangePlan::local_cell_count() const noexcept
    {
        return data_ ? data_->local_cells : 0;
    }
    std::span<const Index> ExchangePlan::send_indices(int peer) const
    {
        if (!data_ || peer < 0 || peer >= rank_count())
            throw std::out_of_range("invalid halo send peer");
        return data_->send[peer];
    }
    std::span<const Index> ExchangePlan::receive_indices(int peer) const
    {
        if (!data_ || peer < 0 || peer >= rank_count())
            throw std::out_of_range("invalid halo receive peer");
        return data_->recv[peer];
    }

    bool ExchangePlan::direct_memory_supported(DeviceMemoryKind memory) const
    {
        return data_ != nullptr && query_direct_device_memory(memory);
    }

    void ExchangePlan::exchange_packed(
        const void* send, void* receive, Index components,
        DeviceMemoryKind memory) const
    {
        ensure_initialized();
        ::idmesh::detail::require_argument(data_ != nullptr && components > 0, "invalid packed halo exchange");
        if (!direct_memory_supported(memory))
            throw std::runtime_error("MPI is not GPU aware for the requested memory space");
        std::vector<int> sc, rc, sd, rd;
        scaled_layout(*data_, components, sc, rc, sd, rd);
        ::idmesh::mpi::detail::check_mpi(MPI_Alltoallv(send, sc.data(), sd.data(), MPI_DOUBLE,
                                             receive, rc.data(), rd.data(), MPI_DOUBLE, data_->comm),
            "MPI_Alltoallv");
    }

    void ExchangePlan::exchange(std::span<double> values, Index components) const
    {
        ensure_initialized();
        ::idmesh::detail::require_state(data_ != nullptr, "empty exchange plan");
        ::idmesh::detail::require_argument(components > 0 && values.size() == std::size_t(data_->local_cells) * components,
            "exchange field must have one fixed-width record per local cell");
        std::lock_guard lock(data_->host_mutex);
        const auto size = static_cast<int>(data_->send.size());
        if (data_->host_components != components) {
            scaled_layout(*data_, components,
                data_->host_send_counts, data_->host_recv_counts,
                data_->host_send_offsets, data_->host_recv_offsets);
            data_->host_components = components;
        }
        const auto send_total = data_->host_send_offsets.empty()
            ? 0
            : data_->host_send_offsets.back() + data_->host_send_counts.back();
        const auto recv_total = data_->host_recv_offsets.empty()
            ? 0
            : data_->host_recv_offsets.back() + data_->host_recv_counts.back();
        data_->host_send_buffer.resize(send_total);
        data_->host_recv_buffer.resize(recv_total);

        for (int peer = 0; peer < size; ++peer) {
            std::size_t output = data_->host_send_offsets[peer];
            for (auto cell : data_->send[peer]) {
                for (Index j = 0; j < components; ++j)
                    data_->host_send_buffer[output++]
                        = values[std::size_t(cell) * components + j];
            }
        }
        ::idmesh::mpi::detail::check_mpi(
            MPI_Alltoallv(data_->host_send_buffer.data(), data_->host_send_counts.data(),
                data_->host_send_offsets.data(), MPI_DOUBLE,
                data_->host_recv_buffer.data(), data_->host_recv_counts.data(),
                data_->host_recv_offsets.data(), MPI_DOUBLE, data_->comm),
            "MPI_Alltoallv");
        for (int peer = 0; peer < size; ++peer) {
            std::size_t input = data_->host_recv_offsets[peer];
            for (auto cell : data_->recv[peer]) {
                for (Index j = 0; j < components; ++j)
                    values[std::size_t(cell) * components + j]
                        = data_->host_recv_buffer[input++];
            }
        }
    }

    ExchangeRequest ExchangePlan::make_request(Index components) const
    {
        ensure_initialized();
        ::idmesh::detail::require_state(data_ != nullptr, "empty exchange plan");
        ::idmesh::detail::require_argument(components > 0,
            "exchange component count must be positive");

        auto request = std::make_shared<ExchangeRequest::Data>();
        request->plan = *this;
        request->components = components;
        scaled_layout(*data_, components,
            request->send_counts, request->recv_counts,
            request->send_offsets, request->recv_offsets);

        const auto send_total = request->send_offsets.empty()
            ? 0
            : request->send_offsets.back() + request->send_counts.back();
        const auto recv_total = request->recv_offsets.empty()
            ? 0
            : request->recv_offsets.back() + request->recv_counts.back();
        request->send_buffer.resize(send_total);
        request->recv_buffer.resize(recv_total);
        return ExchangeRequest(std::move(request));
    }

    ExchangeRequest ExchangePlan::begin_exchange(
        std::span<double> values, Index components) const
    {
        auto request = make_request(components);
        request.start(values);
        return request;
    }

    PartitionedMesh::PartitionedMesh(std::shared_ptr<Data> data)
        : data_(std::move(data))
    {
    }

    const Mesh& PartitionedMesh::mesh() const noexcept { return data_->mesh; }
    Index PartitionedMesh::owned_cell_count() const noexcept { return data_->owned_cells; }

    std::span<const std::uint64_t> PartitionedMesh::global_ids(Index dimension) const
    {
        if (dimension >= data_->global_ids.size())
            throw std::out_of_range("global ID dimension");
        return data_->global_ids[dimension];
    }

    std::span<const int> PartitionedMesh::owner_ranks(Index dimension) const
    {
        if (dimension >= data_->owner_ranks.size())
            throw std::out_of_range("owner-rank dimension");
        return data_->owner_ranks[dimension];
    }

    const ExchangePlan& PartitionedMesh::exchange_plan() const noexcept
    {
        return data_->exchange;
    }

    PartitionedMesh partition_replicated(
        const Context& context, const Mesh& global, Index ghost_layers)
    {
        const int rank = context.rank(), size = context.size();
        const MPI_Comm comm = context.data_->comm;
        const Index d = global.dimension();
        const Index cells = global.entity_count(d);
        ::idmesh::detail::require_argument(cells > 0, "replicated partition source has no cells");
        ::idmesh::detail::require_argument(ghost_layers <= 8, "ghost depth exceeds safety limit");
        ::idmesh::detail::require_argument(global.has_relation(d, d - 1) && global.has_relation(d - 1, d),
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
        ::idmesh::mpi::detail::check_mpi(MPI_Allreduce(&local, &low, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, comm), "MPI_Allreduce");
        ::idmesh::mpi::detail::check_mpi(MPI_Allreduce(&local, &high, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm), "MPI_Allreduce");
        ::idmesh::detail::require_runtime(low == high, "replicated source mesh identities differ across ranks");

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
                    for (auto f : facets[c].targets)
                        for (auto neighbor : support[f].targets) {
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
            Index local_cell = invalid_index;
            if (global.cell_order(global_cell) == 1)
                local_cell = builder.add_cell(global.cell_types()[global_cell], vertices);
            else
                local_cell = builder.add_quadratic_cell(
                    global.cell_types()[global_cell], vertices, extras);
            local_cells[global_cell] = local_cell;
            for (const auto& label : global.labels(d, global_cell))
                builder.add_cell_label(local_cell, label.value);
        }
        for (const auto& label : global.labels()) {
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
                builder.add_label(label.dimension, vertices, label.value);
            if (auto name = global.label_name(label.dimension, label.value); !name.empty())
                builder.set_label_name(label.dimension, label.value, std::string(name));
        }
        BuildOptions opts;
        opts.relations = RelationDirection::bidirectional;
        if (d == 3 && global.has_entities(1) && global.has_entities(2))
            opts.topology = TopologyLayout::complete;
        else if (d > 1 && global.has_entities(d - 1))
            opts.topology = TopologyLayout::cell_facets;
        else
            opts.topology = TopologyLayout::cell_vertices;
        auto partition = std::make_shared<PartitionedMesh::Data>(
            std::move(builder).finalize(opts));
        partition->owned_cells = owned_count;
        partition->global_ids.resize(d + 1);
        partition->owner_ranks.resize(d + 1);
        partition->global_ids[d].assign(order.begin(), order.end());
        for (auto id : order)
            partition->owner_ranks[d].push_back(owner(id));

        // Stable dimension-local shared identity from canonical *global* corners.
        // Vertex and cell IDs are the original replicated source indices.
        std::vector<std::map<std::vector<Index>, Index>> lookup(d);
        for (Index k = 1; k < d; ++k)
            for (Index e = 0; e < global.entity_count(k); ++e)
                lookup[k].emplace(canonical_key(global.entity_vertices(k, e)), e);
        std::vector<Index> reverse_vertices(partition->mesh.entity_count(0));
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
                    for (auto lower : rel[higher].targets)
                        global_owners[to][lower] = std::min(
                            global_owners[to][lower], global_owners[from][higher]);
            }
        }
        for (Index k = 0; k < d; ++k) {
            for (Index e = 0; e < partition->mesh.entity_count(k); ++e) {
                Index global_id = invalid_index;
                if (k == 0)
                    global_id = reverse_vertices[e];
                else {
                    std::vector<Index> key;
                    for (auto v : partition->mesh.entity_vertices(k, e))
                        key.push_back(reverse_vertices[v]);
                    std::sort(key.begin(), key.end());
                    auto it = lookup[k].find(key);
                    ::idmesh::detail::require_runtime(it != lookup[k].end(), "cross-rank shared entity identity mismatch");
                    global_id = it->second;
                }
                partition->global_ids[k].push_back(global_id);
                partition->owner_ranks[k].push_back(global_owners[k][global_id]);
            }
        }
        auto exchange = std::make_shared<ExchangePlan::Data>();
        exchange->comm = comm;
        exchange->context_owner = context.data_;
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
        exchange->finalize_layout();
        partition->exchange = ExchangePlan(std::move(exchange));
        return PartitionedMesh(std::move(partition));
    }

} // namespace idmesh::mpi
