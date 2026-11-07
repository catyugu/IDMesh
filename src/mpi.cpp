#include "idmesh/mpi.hpp"
#include "contracts.hpp"
#include "mpi_errors.hpp"
#include "mpi_internal.hpp"

#include <algorithm>
#include <cstddef>
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
            std::size_t input = static_cast<std::size_t>(recv_offsets[peer]);
            for (Index entity : plan.receive_indices(peer)) {
                const auto output
                    = static_cast<std::size_t>(entity) * bytes_per_entity;
                std::copy_n(recv_buffer.data() + input, bytes_per_entity,
                    records.data() + output);
                input += bytes_per_entity;
            }
        }
        records = {};
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

    void ExchangeRequest::start(std::span<std::byte> records)
    {
        ensure_initialized();
        ::idmesh::detail::require_state(data_ != nullptr, "empty exchange request");
        ::idmesh::detail::require_state(!data_->active,
            "exchange request is already active");
        ::idmesh::detail::require_argument(
            records.size()
                == std::size_t(data_->plan.local_entity_count())
                    * data_->bytes_per_entity,
            "exchange storage must contain one fixed-width record per local entity");

        data_->records = records;
        for (int peer = 0; peer < data_->plan.rank_count(); ++peer) {
            std::size_t output
                = static_cast<std::size_t>(data_->send_offsets[peer]);
            for (Index entity : data_->plan.send_indices(peer)) {
                const auto input
                    = static_cast<std::size_t>(entity)
                    * data_->bytes_per_entity;
                std::copy_n(records.data() + input,
                    data_->bytes_per_entity,
                    data_->send_buffer.data() + output);
                output += data_->bytes_per_entity;
            }
        }

        MPI_Request request = MPI_REQUEST_NULL;
        ::idmesh::mpi::detail::check_mpi(
            MPI_Ialltoallv(data_->send_buffer.data(), data_->send_counts.data(),
                data_->send_offsets.data(), MPI_BYTE,
                data_->recv_buffer.data(), data_->recv_counts.data(),
                data_->recv_offsets.data(), MPI_BYTE, data_->plan.data_->comm,
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

    ExchangeRequest ExchangePlan::make_request(
        std::size_t bytes_per_entity) const
    {
        ensure_initialized();
        ::idmesh::detail::require_state(data_ != nullptr, "empty exchange plan");
        ::idmesh::detail::require_argument(bytes_per_entity > 0,
            "exchange record size must be positive");

        auto request = std::make_unique<ExchangeRequest::Data>();
        request->plan = *this;
        request->bytes_per_entity = bytes_per_entity;
        data_->scaled_layout(bytes_per_entity,
            request->send_counts, request->recv_counts,
            request->send_offsets, request->recv_offsets);

        const auto send_total = request->send_offsets.empty()
            ? std::size_t {0}
            : static_cast<std::size_t>(request->send_offsets.back())
                + static_cast<std::size_t>(request->send_counts.back());
        const auto recv_total = request->recv_offsets.empty()
            ? std::size_t {0}
            : static_cast<std::size_t>(request->recv_offsets.back())
                + static_cast<std::size_t>(request->recv_counts.back());
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
