#pragma once

#include "idmesh/export.hpp"
#include "idmesh/mesh.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <vector>

namespace idmesh::data {

    enum class ScalarType : std::uint8_t {
        int8,
        uint8,
        int16,
        uint16,
        int32,
        uint32,
        int64,
        uint64,
        float32,
        float64
    };

    IDMESH_API std::size_t scalar_size(ScalarType type);

    template <class T>
    struct ScalarTypeOf;

    template <>
    struct ScalarTypeOf<std::int8_t> {
        static constexpr ScalarType value = ScalarType::int8;
    };
    template <>
    struct ScalarTypeOf<std::uint8_t> {
        static constexpr ScalarType value = ScalarType::uint8;
    };
    template <>
    struct ScalarTypeOf<std::int16_t> {
        static constexpr ScalarType value = ScalarType::int16;
    };
    template <>
    struct ScalarTypeOf<std::uint16_t> {
        static constexpr ScalarType value = ScalarType::uint16;
    };
    template <>
    struct ScalarTypeOf<std::int32_t> {
        static constexpr ScalarType value = ScalarType::int32;
    };
    template <>
    struct ScalarTypeOf<std::uint32_t> {
        static constexpr ScalarType value = ScalarType::uint32;
    };
    template <>
    struct ScalarTypeOf<std::int64_t> {
        static constexpr ScalarType value = ScalarType::int64;
    };
    template <>
    struct ScalarTypeOf<std::uint64_t> {
        static constexpr ScalarType value = ScalarType::uint64;
    };
    template <>
    struct ScalarTypeOf<float> {
        static constexpr ScalarType value = ScalarType::float32;
    };
    template <>
    struct ScalarTypeOf<double> {
        static constexpr ScalarType value = ScalarType::float64;
    };

    template <class T>
    inline constexpr ScalarType scalar_type_v
        = ScalarTypeOf<std::remove_cv_t<T>>::value;

    enum class LocationKind : std::uint8_t {
        entity,
        geometry_node,
        mesh
    };

    struct DataLocation {
        LocationKind kind = LocationKind::entity;
        Index dimension = 0;
    };

    constexpr DataLocation entities(Index dimension) noexcept
    {
        return {LocationKind::entity, dimension};
    }

    constexpr DataLocation geometry_nodes() noexcept
    {
        return {LocationKind::geometry_node, 0};
    }

    constexpr DataLocation whole_mesh() noexcept
    {
        return {LocationKind::mesh, 0};
    }

    enum class SelectionKind : std::uint8_t {
        all,
        indices
    };

    struct DataSelectionView {
        SelectionKind kind = SelectionKind::all;
        std::span<const Index> indices;
    };

    constexpr DataSelectionView all() noexcept
    {
        return {SelectionKind::all, {}};
    }

    constexpr DataSelectionView select(
        std::span<const Index> indices) noexcept
    {
        return {SelectionKind::indices, indices};
    }

    // Describes one scalar component across tuples. The scalar at tuple i
    // begins at storage[offset_bytes + i * stride_bytes].
    struct ComponentView {
        std::span<const std::byte> storage;
        std::size_t offset_bytes = 0;
        std::size_t stride_bytes = 0;
    };

    // Component descriptors are owned metadata; numerical storage is not.
    // Components may share one storage span (AoS/interleaved) or refer to
    // independent spans (SoA).
    struct ArrayView {
        ScalarType type = ScalarType::float64;
        std::size_t tuples = 0;
        std::vector<ComponentView> components;
    };

    template <class T, std::size_t Extent>
    ArrayView make_scalar_array(std::span<T, Extent> values)
    {
        ArrayView result;
        result.type = scalar_type_v<T>;
        result.tuples = values.size();
        result.components.push_back(
            {std::as_bytes(values), 0, sizeof(std::remove_cv_t<T>)});
        return result;
    }

    template <class T, std::size_t Extent>
    ArrayView make_interleaved_array(
        std::span<T, Extent> values, std::uint32_t components)
    {
        if (components == 0
            || values.size() % components != 0)
            throw std::invalid_argument(
                "invalid interleaved array shape");

        using Value = std::remove_cv_t<T>;
        ArrayView result;
        result.type = scalar_type_v<Value>;
        result.tuples = values.size() / components;
        const auto storage = std::as_bytes(values);
        const auto stride
            = static_cast<std::size_t>(components) * sizeof(Value);
        result.components.reserve(components);
        for (std::uint32_t component = 0;
             component < components; ++component) {
            result.components.push_back(
                {storage,
                    static_cast<std::size_t>(component)
                        * sizeof(Value),
                    stride});
        }
        return result;
    }

    template <class T>
    ArrayView make_component_array(
        std::span<const std::span<const T>> components)
    {
        if (components.empty())
            throw std::invalid_argument(
                "component array requires at least one component");

        const std::size_t tuples = components.front().size();
        ArrayView result;
        result.type = scalar_type_v<T>;
        result.tuples = tuples;
        result.components.reserve(components.size());
        for (const auto values : components) {
            if (values.size() != tuples)
                throw std::invalid_argument(
                    "component array tuple counts differ");
            result.components.push_back(
                {std::as_bytes(values), 0,
                    sizeof(std::remove_cv_t<T>)});
        }
        return result;
    }

    template <class T, std::size_t Count>
    ArrayView make_component_array(
        const std::array<std::span<const T>, Count>& components)
    {
        return make_component_array<T>(
            std::span<const std::span<const T>>(
                components.data(), components.size()));
    }

    // Generic AoS/strided storage. Each offset names one scalar component
    // within a tuple whose next instance begins tuple_stride_bytes later.
    template <class T>
    ArrayView make_strided_array(
        std::span<const std::byte> storage, std::size_t tuples,
        std::span<const std::size_t> component_offsets,
        std::size_t tuple_stride_bytes)
    {
        if (component_offsets.empty())
            throw std::invalid_argument(
                "strided array requires at least one component");

        ArrayView result;
        result.type = scalar_type_v<T>;
        result.tuples = tuples;
        result.components.reserve(component_offsets.size());
        for (const auto offset : component_offsets)
            result.components.push_back(
                {storage, offset, tuple_stride_bytes});
        return result;
    }

    struct EntitySetView {
        std::string_view name;
        DataLocation location;
        DataSelectionView selection;
    };

    struct MeshDataView {
        std::string_view name;
        DataLocation location;
        DataSelectionView selection;
        ArrayView values;
    };

    struct StateView {
        std::optional<double> time;
        std::optional<std::int64_t> cycle;
        std::span<const MeshDataView> fields;
    };

    // Number of addressable locations before applying a selection.
    IDMESH_API std::size_t location_size(
        const Mesh& mesh, DataLocation location);

    // Validates the location and selection, returning its tuple count.
    IDMESH_API std::size_t selection_size(
        const Mesh& mesh, DataLocation location,
        DataSelectionView selection);

    IDMESH_API void validate_array(const ArrayView& array);
    IDMESH_API void validate(
        const Mesh& mesh, const EntitySetView& set);
    IDMESH_API void validate(
        const Mesh& mesh, const MeshDataView& field);
    IDMESH_API void validate(
        const Mesh& mesh, const StateView& state);

} // namespace idmesh::data
