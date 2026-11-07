#include "idmesh/mpi_kokkos.hpp"
#include "contracts.hpp"

#include <Kokkos_Core.hpp>
#include <climits>
#include <cstdint>
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

        DeviceMemoryKind current_memory_kind()
        {
#if defined(KOKKOS_ENABLE_CUDA)
            return DeviceMemoryKind::cuda;
#elif defined(KOKKOS_ENABLE_HIP)
            return DeviceMemoryKind::hip;
#else
            return DeviceMemoryKind::host;
#endif
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

        void make_layout(const ExchangePlan& plan, Index components,
            std::vector<int>& send_counts, std::vector<int>& receive_counts,
            std::vector<int>& send_offsets, std::vector<int>& receive_offsets)
        {
            const int peers = plan.rank_count();
            send_counts.resize(peers);
            receive_counts.resize(peers);
            send_offsets.resize(peers);
            receive_offsets.resize(peers);

            int send_offset = 0;
            int receive_offset = 0;
            for (int peer = 0; peer < peers; ++peer) {
                const auto sending = plan.send_indices(peer).size();
                const auto receiving = plan.receive_indices(peer).size();
                ::idmesh::detail::require_argument(
                    sending <= static_cast<std::size_t>(INT_MAX) / components
                        && receiving <= static_cast<std::size_t>(INT_MAX) / components,
                    "device halo peer record count exceeds MPI int range");
                const int send_count = static_cast<int>(sending * components);
                const int receive_count = static_cast<int>(receiving * components);
                ::idmesh::detail::require_argument(
                    send_offset <= INT_MAX - send_count
                        && receive_offset <= INT_MAX - receive_count,
                    "device halo displacement overflow");
                send_counts[peer] = send_count;
                receive_counts[peer] = receive_count;
                send_offsets[peer] = send_offset;
                receive_offsets[peer] = receive_offset;
                send_offset += send_count;
                receive_offset += receive_count;
            }
        }
    } // namespace

    struct DeviceHaloPlan::Data {
        ExchangePlan plan;
        Index components = 1;
        Index cells = 0;
        Index sending = 0;
        Index receiving = 0;
        std::vector<int> send_counts;
        std::vector<int> receive_counts;
        std::vector<int> send_offsets;
        std::vector<int> receive_offsets;
        Kokkos::View<Index*, Memory> send_cells;
        Kokkos::View<Index*, Memory> receive_cells;
        DeviceBuffer send_buffer;
        DeviceBuffer receive_buffer;
        HostBuffer host_send_buffer;
        HostBuffer host_receive_buffer;
    };

    DeviceHaloPlan::DeviceHaloPlan(const ExchangePlan& plan, Index components)
        : data_(std::make_unique<Data>())
    {
        ::idmesh::detail::require_state(Kokkos::is_initialized(),
            "initialize Kokkos before creating DeviceHaloPlan");
        ::idmesh::detail::require_argument(components != 0 && plan.rank_count() > 0,
            "invalid device halo component or plan");
        data_->plan = plan;
        data_->cells = plan.local_cell_count();
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
        make_layout(plan, components,
            data_->send_counts, data_->receive_counts,
            data_->send_offsets, data_->receive_offsets);

        data_->send_cells = copy_to_device("idmesh_device_halo_send_ids", send);
        data_->receive_cells = copy_to_device("idmesh_device_halo_receive_ids", receive);
        const auto send_values = send.size() * components;
        const auto receive_values = receive.size() * components;
        data_->send_buffer = DeviceBuffer("idmesh_device_halo_send", send_values);
        data_->receive_buffer = DeviceBuffer("idmesh_device_halo_receive", receive_values);
        // create_mirror_view aliases host-accessible device storage and allocates
        // staging storage only when the execution memory is not host accessible.
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
        const auto memory = current_memory_kind();
        return memory != DeviceMemoryKind::host
            && data_->plan.direct_memory_supported(memory);
    }

    void DeviceHaloPlan::exchange(idmesh::kokkos::DeviceField& field,
        DeviceTransport mode) const
    {
        if (field.entity_count() != data_->cells
            || field.component_count() != data_->components)
            throw std::invalid_argument("halo field shape does not match plan");

        const auto components = data_->components;
        const auto nsend = data_->sending;
        const auto nrecv = data_->receiving;
        auto send_ids = data_->send_cells;
        auto receive_ids = data_->receive_cells;
        auto send_data = data_->send_buffer;
        auto receive_data = data_->receive_buffer;
        auto* field_data = field.data();

        Kokkos::parallel_for("idmesh_device_halo_pack",
            Kokkos::RangePolicy<Execution>(0, static_cast<Index>(nsend * components)),
            KOKKOS_LAMBDA(const Index i) {
                const Index cell = send_ids(i / components);
                const Index component = i % components;
                send_data(i) = field_data[cell * components + component];
            });
        Execution().fence();

        const auto memory = current_memory_kind();
        const bool direct = memory == DeviceMemoryKind::host
            || data_->plan.direct_memory_supported(memory);
        if (mode == DeviceTransport::require_gpu_aware
            && (memory == DeviceMemoryKind::host || !direct))
            throw std::runtime_error("no GPU-aware MPI available for device halo");
        const bool use_direct = mode != DeviceTransport::host_staging && direct;

        if (use_direct) {
            data_->plan.exchange_packed(send_data.data(), receive_data.data(),
                data_->send_counts, data_->receive_counts,
                data_->send_offsets, data_->receive_offsets, memory);
        }
        else {
            Kokkos::deep_copy(data_->host_send_buffer, send_data);
            data_->plan.exchange_packed(
                data_->host_send_buffer.data(), data_->host_receive_buffer.data(),
                data_->send_counts, data_->receive_counts,
                data_->send_offsets, data_->receive_offsets,
                DeviceMemoryKind::host);
            Kokkos::deep_copy(receive_data, data_->host_receive_buffer);
        }

        Kokkos::parallel_for("idmesh_device_halo_unpack",
            Kokkos::RangePolicy<Execution>(0, static_cast<Index>(nrecv * components)),
            KOKKOS_LAMBDA(const Index i) {
                const Index cell = receive_ids(i / components);
                const Index component = i % components;
                field_data[cell * components + component] = receive_data(i);
            });
        Execution().fence();
    }

} // namespace idmesh::mpi::kokkos
