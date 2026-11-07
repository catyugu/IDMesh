#include "idmesh/data.hpp"

#include "contracts.hpp"

#include <cmath>
#include <limits>

namespace idmesh::data {

    namespace {

        std::size_t checked_component_end(
            const ComponentView& component,
            std::size_t tuples, std::size_t bytes)
        {
            if (tuples == 0)
                return component.offset_bytes;

            ::idmesh::detail::require_argument(
                component.stride_bytes >= bytes,
                "array component stride is smaller than its scalar");
            ::idmesh::detail::require_argument(
                component.offset_bytes
                    <= std::numeric_limits<std::size_t>::max()
                        - bytes,
                "array component offset overflows");

            const auto last = tuples - 1;
            ::idmesh::detail::require_argument(
                last == 0
                    || component.stride_bytes
                        <= (std::numeric_limits<std::size_t>::max()
                               - component.offset_bytes - bytes)
                            / last,
                "array component extent overflows");
            return component.offset_bytes
                + last * component.stride_bytes + bytes;
        }

        void validate_location(
            const Mesh& mesh, DataLocation location)
        {
            switch (location.kind) {
            case LocationKind::entity:
                ::idmesh::detail::require_argument(
                    location.dimension <= mesh.dimension(),
                    "data entity dimension exceeds mesh dimension");
                ::idmesh::detail::require_argument(
                    mesh.has_entities(location.dimension),
                    "data refers to an omitted entity dimension");
                break;
            case LocationKind::geometry_node:
                ::idmesh::detail::require_argument(
                    location.dimension == 0,
                    "geometry-node data cannot carry an entity dimension");
                break;
            case LocationKind::mesh:
                ::idmesh::detail::require_argument(
                    location.dimension == 0,
                    "mesh data cannot carry an entity dimension");
                break;
            default:
                throw std::invalid_argument("invalid data location");
            }
        }

    } // namespace

    std::size_t scalar_size(ScalarType type)
    {
        switch (type) {
        case ScalarType::int8:
        case ScalarType::uint8:
            return 1;
        case ScalarType::int16:
        case ScalarType::uint16:
            return 2;
        case ScalarType::int32:
        case ScalarType::uint32:
        case ScalarType::float32:
            return 4;
        case ScalarType::int64:
        case ScalarType::uint64:
        case ScalarType::float64:
            return 8;
        }
        throw std::invalid_argument("invalid scalar type");
    }

    std::size_t location_size(
        const Mesh& mesh, DataLocation location)
    {
        validate_location(mesh, location);
        switch (location.kind) {
        case LocationKind::entity:
            return mesh.entity_count(location.dimension);
        case LocationKind::geometry_node:
            return mesh.geometry_node_count();
        case LocationKind::mesh:
            return 1;
        }
        throw std::invalid_argument("invalid data location");
    }

    std::size_t selection_size(
        const Mesh& mesh, DataLocation location,
        DataSelectionView selection)
    {
        const auto count = location_size(mesh, location);
        switch (selection.kind) {
        case SelectionKind::all:
            ::idmesh::detail::require_argument(
                selection.indices.empty(),
                "all-selection cannot carry explicit indices");
            return count;

        case SelectionKind::indices:
            ::idmesh::detail::require_argument(
                location.kind != LocationKind::mesh,
                "mesh data does not support indexed selection");
            for (std::size_t i = 0;
                 i < selection.indices.size(); ++i) {
                ::idmesh::detail::require_argument(
                    selection.indices[i] < count,
                    "data selection index out of range");
                if (i != 0)
                    ::idmesh::detail::require_argument(
                        selection.indices[i - 1]
                            < selection.indices[i],
                        "data selection indices must be strictly increasing");
            }
            return selection.indices.size();
        }
        throw std::invalid_argument("invalid data selection");
    }

    void validate_array(const ArrayView& array)
    {
        const auto bytes = scalar_size(array.type);
        ::idmesh::detail::require_argument(
            !array.components.empty(),
            "data array requires at least one component");

        for (const auto& component : array.components) {
            const auto end = checked_component_end(
                component, array.tuples, bytes);
            ::idmesh::detail::require_argument(
                end <= component.storage.size(),
                "data array component exceeds its storage");
        }
    }

    void validate(
        const Mesh& mesh, const EntitySetView& set)
    {
        ::idmesh::detail::require_argument(
            !set.name.empty(),
            "entity set requires a name");
        ::idmesh::detail::require_argument(
            set.location.kind == LocationKind::entity,
            "entity set requires an entity location");
        (void)selection_size(
            mesh, set.location, set.selection);
    }

    void validate(
        const Mesh& mesh, const MeshDataView& field)
    {
        ::idmesh::detail::require_argument(
            !field.name.empty(),
            "mesh data requires a name");
        const auto tuples = selection_size(
            mesh, field.location, field.selection);
        validate_array(field.values);
        ::idmesh::detail::require_argument(
            field.values.tuples == tuples,
            "mesh data tuple count does not match its selection");
    }

    void validate(
        const Mesh& mesh, const StateView& state)
    {
        if (state.time)
            ::idmesh::detail::require_argument(
                std::isfinite(*state.time),
                "state time must be finite");
        for (const auto& field : state.fields)
            validate(mesh, field);
    }

    ArrayView Array::view() const
    {
        const auto bytes = scalar_size(type);
        ::idmesh::detail::require_argument(
            components > 0,
            "owned data array requires components");
        ::idmesh::detail::require_argument(
            tuples <= std::numeric_limits<std::size_t>::max()
                    / components
                && tuples * components
                    <= std::numeric_limits<std::size_t>::max()
                        / bytes,
            "owned data array size overflows");
        ::idmesh::detail::require_argument(
            values.size() == tuples * components * bytes,
            "owned data array byte count mismatch");

        ArrayView result;
        result.type = type;
        result.tuples = tuples;
        const auto storage
            = std::span<const std::byte>(
                values.data(), values.size());
        const auto stride
            = static_cast<std::size_t>(components) * bytes;
        result.components.reserve(components);
        for (std::uint32_t component = 0;
             component < components; ++component) {
            result.components.push_back(
                {storage,
                    static_cast<std::size_t>(component) * bytes,
                    stride});
        }
        return result;
    }

    MeshDataView MeshData::view() const
    {
        const DataSelectionView selected
            = selection == SelectionKind::all
            ? all()
            : select(indices);
        return {
            name, location, selected, values.view()};
    }


} // namespace idmesh::data
