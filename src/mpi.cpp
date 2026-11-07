#include "idmesh/mpi.hpp"
#include "contracts.hpp"
#include "mpi_errors.hpp"
#include "mpi_internal.hpp"

#include <cstdint>
#include <limits>
#include <mpi.h>
#include <stdexcept>
#include <utility>

namespace idmesh::mpi {
    namespace {
        void ensure_initialized()
        {
            int initialized = 0;
            ::idmesh::mpi::detail::check_mpi(MPI_Initialized(&initialized), "MPI_Initialized");
            ::idmesh::detail::require_state(initialized != 0, "MPI must be initialized by the application");
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

    ExchangeRequest::ExchangeRequest(std::unique_ptr<Data> data)
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
            for (Index entity : plan.receive_indices(peer))
                for (Index component = 0; component < components; ++component)
                    values[static_cast<std::size_t>(entity) * components + component]
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
                == std::size_t(data_->plan.local_entity_count()) * data_->components,
            "exchange field must have one fixed-width record per local entity");

        data_->values = values;
        for (int peer = 0; peer < data_->plan.rank_count(); ++peer) {
            std::size_t output = data_->send_offsets[peer];
            for (Index entity : data_->plan.send_indices(peer))
                for (Index component = 0; component < data_->components; ++component)
                    data_->send_buffer[output++]
                        = values[static_cast<std::size_t>(entity) * data_->components
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

    ExchangePlan::ExchangePlan(std::shared_ptr<Data> data)
        : data_(std::move(data))
    {
    }

    bool ExchangePlan::empty() const noexcept
    {
        if (!data_)
            return true;
        for (const auto& r : data_->recv)
            if (!r.empty())
                return false;
        for (const auto& s : data_->send)
            if (!s.empty())
                return false;
        return true;
    }

    int ExchangePlan::rank_count() const noexcept
    {
        return data_ ? static_cast<int>(data_->send.size()) : 0;
    }

    Index ExchangePlan::local_entity_count() const noexcept
    {
        return data_ ? data_->local_entities : 0;
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

    ExchangeRequest ExchangePlan::make_request(Index components) const
    {
        ensure_initialized();
        ::idmesh::detail::require_state(data_ != nullptr, "empty exchange plan");
        ::idmesh::detail::require_argument(components > 0,
            "exchange component count must be positive");

        auto request = std::make_unique<ExchangeRequest::Data>();
        request->plan = *this;
        request->components = components;
        data_->scaled_layout(components,
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

    PartitionedMesh::PartitionedMesh(std::shared_ptr<Data> data)
        : data_(std::move(data))
    {
    }

    const Mesh& PartitionedMesh::mesh() const noexcept
    {
        return data_->mesh;
    }

    std::span<const Index> PartitionedMesh::owned_entities(Index dimension) const
    {
        if (dimension >= data_->owned_entities.size())
            throw std::out_of_range("owned-entity dimension");
        return data_->owned_entities[dimension];
    }

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

    const ExchangePlan& PartitionedMesh::exchange_plan(Index dimension) const
    {
        if (dimension >= data_->exchanges.size())
            throw std::out_of_range("exchange-plan dimension");
        if (!data_->mesh.has_entities(dimension))
            throw std::invalid_argument("entity dimension not materialized");
        return data_->exchanges[dimension];
    }

} // namespace idmesh::mpi
