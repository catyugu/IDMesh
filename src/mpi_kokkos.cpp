#include "idmesh/mpi_kokkos.hpp"

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace idmesh::mpi::kokkos {
namespace {
using Execution = Kokkos::DefaultExecutionSpace;
using Memory = Execution::memory_space;

DeviceMemoryKind current_memory_kind()
{
#if defined(KOKKOS_ENABLE_CUDA)
    return DeviceMemoryKind::CUDA;
#elif defined(KOKKOS_ENABLE_HIP)
    return DeviceMemoryKind::HIP;
#else
    return DeviceMemoryKind::Host;
#endif
}

template<class T>
Kokkos::View<T*, Memory> copy_to_device(
    const char* label, const std::vector<T>& values)
{
    Kokkos::View<T*, Memory> device(label, values.size());
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
    Index cells = 0;
    Index sending = 0;
    Index receiving = 0;
    Kokkos::View<Index*, Memory> send_cells;
    Kokkos::View<Index*, Memory> receive_cells;
    Kokkos::View<double*, Memory> send_buffer;
    Kokkos::View<double*, Memory> receive_buffer;
};

DeviceHaloPlan::DeviceHaloPlan(const ExchangePlan& plan, Index components)
    : data_(std::make_shared<Data>())
{
    if (!Kokkos::is_initialized())
        throw std::logic_error("initialize Kokkos before creating DeviceHaloPlan");
    if (components == 0 || plan.peer_count() <= 0)
        throw std::invalid_argument("invalid device halo component or plan");
    data_->plan = plan;
    data_->cells = plan.local_cell_count();
    data_->components = components;
    std::vector<Index> send, recv;
    for (int peer = 0; peer < plan.peer_count(); ++peer) {
        const auto s = plan.send_indices(peer);
        const auto r = plan.receive_indices(peer);
        send.insert(send.end(), s.begin(), s.end());
        recv.insert(recv.end(), r.begin(), r.end());
    }
    if (send.size() > static_cast<std::size_t>(invalid_index) / components ||
        recv.size() > static_cast<std::size_t>(invalid_index) / components)
        throw std::overflow_error("device halo count exceeds local index limit");
    data_->sending = static_cast<Index>(send.size());
    data_->receiving = static_cast<Index>(recv.size());
    data_->send_cells = copy_to_device("idmesh_device_halo_send_ids", send);
    data_->receive_cells = copy_to_device("idmesh_device_halo_receive_ids", recv);
    data_->send_buffer = Kokkos::View<double*, Memory>(
        "idmesh_device_halo_send", send.size() * components);
    data_->receive_buffer = Kokkos::View<double*, Memory>(
        "idmesh_device_halo_receive", recv.size() * components);
    Execution().fence();
}

Index DeviceHaloPlan::components() const noexcept
{
    return data_->components;
}

bool DeviceHaloPlan::gpu_aware_available() const
{
    const auto memory = current_memory_kind();
    return memory != DeviceMemoryKind::Host && gpu_aware_support(memory);
}

void DeviceHaloPlan::exchange(idmesh::kokkos::DeviceCellField& field,
    DeviceTransport mode) const
{
    if (field.cells() != data_->cells ||
        field.components() != data_->components)
        throw std::invalid_argument("halo field shape does not match plan");

    const auto components = data_->components;
    const auto nsend = data_->sending;
    const auto nrecv = data_->receiving;
    auto send_ids = data_->send_cells;
    auto recv_ids = data_->receive_cells;
    auto send_data = data_->send_buffer;
    auto recv_data = data_->receive_buffer;
    auto* field_data = static_cast<double*>(field.device_data());

    Kokkos::parallel_for("idmesh_device_halo_pack",
        Kokkos::RangePolicy<Execution>(0,
            static_cast<Index>(nsend * components)),
        KOKKOS_LAMBDA(const Index i) {
            const Index cell = send_ids(i / components);
            const Index component = i % components;
            send_data(i) = field_data[cell * components + component];
        });
    Execution().fence();

    const auto memory = current_memory_kind();
    const bool direct = memory == DeviceMemoryKind::Host ||
        gpu_aware_support(memory);
    if (mode == DeviceTransport::RequireGpuAware &&
        (memory == DeviceMemoryKind::Host || !direct))
        throw std::runtime_error("no GPU-aware MPI available for device halo");
    const bool use_direct = mode != DeviceTransport::HostStaging && direct;

    if (use_direct) {
        data_->plan.exchange_packed(send_data.data(), recv_data.data(),
            components, memory);
    } else {
        auto host_send = Kokkos::create_mirror_view(send_data);
        auto host_receive = Kokkos::create_mirror_view(recv_data);
        Kokkos::deep_copy(host_send, send_data);
        Execution().fence();
        data_->plan.exchange_packed(host_send.data(), host_receive.data(),
            components, DeviceMemoryKind::Host);
        Kokkos::deep_copy(recv_data, host_receive);
        Execution().fence();
    }

    Kokkos::parallel_for("idmesh_device_halo_unpack",
        Kokkos::RangePolicy<Execution>(0,
            static_cast<Index>(nrecv * components)),
        KOKKOS_LAMBDA(const Index i) {
            const Index cell = recv_ids(i / components);
            const Index component = i % components;
            field_data[cell * components + component] = recv_data(i);
        });
    Execution().fence();
}

} // namespace idmesh::mpi::kokkos
