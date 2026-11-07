#include "idmesh/mpi_kokkos.hpp"
#include "contracts.hpp"
#include "mpi_errors.hpp"
#include "mpi_internal.hpp"

#include <Kokkos_Core.hpp>
#include <cstdint>
#include <mpi.h>
#if defined(OPEN_MPI) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace idmesh::mpi::kokkos {
    namespace {
        using Execution = Kokkos::DefaultExecutionSpace;
        using Memory = Execution::memory_space;
        using DeviceBuffer = Kokkos::View<double*, Memory>;
        using HostBuffer = typename DeviceBuffer::HostMirror;

        enum class DeviceBackend : std::uint8_t {
            host,
            cuda,
            hip
        };

        DeviceBackend current_device_backend()
        {
#if defined(KOKKOS_ENABLE_CUDA)
            return DeviceBackend::cuda;
#elif defined(KOKKOS_ENABLE_HIP)
            return DeviceBackend::hip;
#else
            return DeviceBackend::host;
#endif
        }

        bool gpu_aware_mpi_available(DeviceBackend backend)
        {
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
            if (backend == DeviceBackend::cuda)
                return MPIX_Query_cuda_support() == 1;
#endif
#if defined(OMPI_HAVE_MPI_EXT_ROCM) && OMPI_HAVE_MPI_EXT_ROCM
            if (backend == DeviceBackend::hip)
                return MPIX_Query_rocm_support() == 1;
#endif
            return false;
        }

        template <class T>
        Kokkos::View<T*, Memory> copy_to_device(
            const char* label, const std::vector<T>& values)
        {
            Kokkos::View<T*, Memory> device(std::string(label), values.size());
            auto host = Kokkos::create_mirror_view(device);
            for (std::size_t i = 0; i < values.size(); ++i)
                host(i) = values[i];
            Kokkos::deep_copy(device, host);
            return device;
        }

    } // namespace

    struct DeviceHaloPlan::Data {
        ExchangePlan plan;
        Index components = 1;
        Index entities = 0;
        Index sending = 0;
        Index receiving = 0;
        std::vector<int> send_counts;
        std::vector<int> receive_counts;
        std::vector<int> send_offsets;
        std::vector<int> receive_offsets;
        Kokkos::View<Index*, Memory> send_entities;
        Kokkos::View<Index*, Memory> receive_entities;
        DeviceBuffer send_buffer;
        DeviceBuffer receive_buffer;
        HostBuffer host_send_buffer;
        HostBuffer host_receive_buffer;
        MPI_Request request = MPI_REQUEST_NULL;
        double* field_data = nullptr;
        bool active = false;
        bool staged = false;

        ~Data()
        {
            if (!active || request == MPI_REQUEST_NULL)
                return;
            int initialized = 0;
            int finalized = 0;
            if (MPI_Initialized(&initialized) != MPI_SUCCESS || !initialized)
                return;
            if (MPI_Finalized(&finalized) != MPI_SUCCESS || finalized)
                return;
            MPI_Wait(&request, MPI_STATUS_IGNORE);
        }
    };

    DeviceHaloPlan::DeviceHaloPlan(const ExchangePlan& plan, Index components)
        : data_(std::make_unique<Data>())
    {
        ::idmesh::detail::require_state(Kokkos::is_initialized(),
            "initialize Kokkos before creating DeviceHaloPlan");
        ::idmesh::detail::require_argument(components != 0 && plan.rank_count() > 0,
            "invalid device halo component or plan");
        data_->plan = plan;
        data_->entities = plan.local_entity_count();
        data_->components = components;

        std::vector<Index> send, receive;
        for (int peer = 0; peer < plan.rank_count(); ++peer) {
            const auto sending = plan.send_indices(peer);
            const auto receiving = plan.receive_indices(peer);
            send.insert(send.end(), sending.begin(), sending.end());
            receive.insert(receive.end(), receiving.begin(), receiving.end());
        }
        if (send.size() > static_cast<std::size_t>(invalid_index) / components
            || receive.size() > static_cast<std::size_t>(invalid_index) / components)
            throw std::length_error("device halo count exceeds local index limit");

        data_->sending = static_cast<Index>(send.size());
        data_->receiving = static_cast<Index>(receive.size());
        plan.data_->scaled_layout(components,
            data_->send_counts, data_->receive_counts,
            data_->send_offsets, data_->receive_offsets);

        data_->send_entities = copy_to_device("idmesh_device_halo_send_ids", send);
        data_->receive_entities = copy_to_device("idmesh_device_halo_receive_ids", receive);
        const auto send_values = send.size() * components;
        const auto receive_values = receive.size() * components;
        data_->send_buffer = DeviceBuffer("idmesh_device_halo_send", send_values);
        data_->receive_buffer = DeviceBuffer("idmesh_device_halo_receive", receive_values);
        // Host-accessible execution spaces alias device storage; CUDA/HIP allocate
        // persistent staging buffers exactly once here.
        data_->host_send_buffer = Kokkos::create_mirror_view(data_->send_buffer);
        data_->host_receive_buffer = Kokkos::create_mirror_view(data_->receive_buffer);
        Execution().fence();
    }

    DeviceHaloPlan::~DeviceHaloPlan() = default;
    DeviceHaloPlan::DeviceHaloPlan(DeviceHaloPlan&&) noexcept = default;
    DeviceHaloPlan& DeviceHaloPlan::operator=(DeviceHaloPlan&&) noexcept = default;

    Index DeviceHaloPlan::component_count() const noexcept
    {
        return data_->components;
    }

    bool DeviceHaloPlan::direct_device_transport_available() const
    {
        const auto backend = current_device_backend();
        return backend != DeviceBackend::host
            && gpu_aware_mpi_available(backend);
    }

    void DeviceHaloPlan::start(idmesh::kokkos::DeviceField& field,
        DeviceTransport mode)
    {
        ::idmesh::detail::require_state(data_ != nullptr, "empty device halo plan");
        ::idmesh::detail::require_state(!data_->active,
            "device halo exchange is already active");
        ::idmesh::detail::require_argument(
            field.entity_count() == data_->entities
                && field.component_count() == data_->components,
            "halo field shape does not match plan");

        const auto backend = current_device_backend();
        const bool host_memory = backend == DeviceBackend::host;
        const bool direct = host_memory || gpu_aware_mpi_available(backend);
        if (mode == DeviceTransport::require_gpu_aware
            && (host_memory || !direct))
            throw std::runtime_error("no GPU-aware MPI available for device halo");
        const bool use_direct = mode != DeviceTransport::host_staging && direct;

        const auto components = data_->components;
        const auto nsend = data_->sending;
        auto send_ids = data_->send_entities;
        auto send_data = data_->send_buffer;
        auto* field_data = field.data();

        Kokkos::parallel_for("idmesh_device_halo_pack",
            Kokkos::RangePolicy<Execution>(
                0, static_cast<Index>(nsend * components)),
            KOKKOS_LAMBDA(const Index i) {
                const Index entity = send_ids(i / components);
                const Index component = i % components;
                send_data(i) = field_data[entity * components + component];
            });
        // MPI may consume the packed buffer immediately after Ialltoallv.
        Execution().fence();

        const void* send_pointer = send_data.data();
        void* receive_pointer = data_->receive_buffer.data();
        if (!use_direct) {
            Kokkos::deep_copy(data_->host_send_buffer, send_data);
            send_pointer = data_->host_send_buffer.data();
            receive_pointer = data_->host_receive_buffer.data();
        }

        MPI_Request request = MPI_REQUEST_NULL;
        ::idmesh::mpi::detail::check_mpi(
            MPI_Ialltoallv(send_pointer, data_->send_counts.data(),
                data_->send_offsets.data(), MPI_DOUBLE,
                receive_pointer, data_->receive_counts.data(),
                data_->receive_offsets.data(), MPI_DOUBLE,
                data_->plan.data_->comm, &request),
            "MPI_Ialltoallv");
        data_->request = request;
        data_->field_data = field_data;
        data_->staged = !use_direct;
        data_->active = true;
    }

    void DeviceHaloPlan::wait()
    {
        ::idmesh::detail::require_state(data_ != nullptr, "empty device halo plan");
        if (!data_->active)
            return;

        if (data_->request != MPI_REQUEST_NULL) {
            ::idmesh::mpi::detail::check_mpi(
                MPI_Wait(&data_->request, MPI_STATUS_IGNORE), "MPI_Wait");
        }

        if (data_->staged)
            Kokkos::deep_copy(data_->receive_buffer, data_->host_receive_buffer);

        const auto components = data_->components;
        const auto nrecv = data_->receiving;
        auto receive_ids = data_->receive_entities;
        auto receive_data = data_->receive_buffer;
        auto* field_data = data_->field_data;

        Kokkos::parallel_for("idmesh_device_halo_unpack",
            Kokkos::RangePolicy<Execution>(
                0, static_cast<Index>(nrecv * components)),
            KOKKOS_LAMBDA(const Index i) {
                const Index entity = receive_ids(i / components);
                const Index component = i % components;
                field_data[entity * components + component] = receive_data(i);
            });
        Execution().fence();

        data_->field_data = nullptr;
        data_->staged = false;
        data_->active = false;
    }

} // namespace idmesh::mpi::kokkos
