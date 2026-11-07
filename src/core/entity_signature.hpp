#pragma once

#include "idmesh/mesh.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace idmesh::detail {

    // Only standard reference-cell edges and faces are supported (arity 1..4).
    template <class Id>
    class EntitySignature {
    public:
        EntitySignature() = default;

        explicit EntitySignature(std::span<const Id> values)
        {
            assign(values);
        }

        void assign(std::span<const Id> values)
        {
            if (values.size() > vertices_.size())
                throw std::invalid_argument("entity signature exceeds four vertices");
            std::copy(values.begin(), values.end(), vertices_.begin());
            size_ = static_cast<std::uint8_t>(values.size());
        }

        std::span<const Id> view() const noexcept
        {
            return {vertices_.data(), size_};
        }

        std::size_t size() const noexcept { return size_; }
        bool empty() const noexcept { return size_ == 0; }
        const Id* begin() const noexcept { return vertices_.data(); }
        const Id* end() const noexcept { return vertices_.data() + size_; }
        Id operator[](std::size_t i) const noexcept { return vertices_[i]; }

        bool operator==(const EntitySignature& other) const noexcept
        {
            return std::equal(begin(), end(), other.begin(), other.end());
        }

        bool operator<(const EntitySignature& other) const noexcept
        {
            return std::lexicographical_compare(
                begin(), end(), other.begin(), other.end());
        }

    private:
        std::array<Id, 4> vertices_ {};
        std::uint8_t size_ = 0;
    };

    template <class Id>
    struct CanonicalEntity {
        EntitySignature<Id> signature;
        Orientation orientation;
    };

    template <class Id>
    CanonicalEntity<Id> canonicalize_entity(std::span<const Id> local)
    {
        CanonicalEntity<Id> result;
        if (local.size() > 4)
            throw std::invalid_argument("nonstandard entity signature arity");
        if (local.empty())
            return result;
        if (local.size() == 1) {
            result.signature.assign(local);
            return result;
        }
        if (local.size() == 2) {
            const std::array<Id, 2> ordered {
                std::min(local[0], local[1]), std::max(local[0], local[1])};
            const bool reversed = local[0] > local[1];
            result.signature.assign(ordered);
            result.orientation = {
                static_cast<std::uint16_t>(reversed ? 1 : 0), reversed};
            return result;
        }

        const auto first = static_cast<std::size_t>(
            std::min_element(local.begin(), local.end()) - local.begin());
        bool reversed = false;
        for (std::size_t i = 1; i < local.size(); ++i) {
            const Id forward = local[(first + i) % local.size()];
            const Id backward = local[(first + local.size() - i) % local.size()];
            if (forward != backward) {
                reversed = backward < forward;
                break;
            }
        }

        std::array<Id, 4> canonical {};
        for (std::size_t i = 0; i < local.size(); ++i) {
            const auto position = reversed
                ? (first + local.size() - i) % local.size()
                : (first + i) % local.size();
            canonical[i] = local[position];
        }
        result.signature.assign(std::span(canonical).first(local.size()));
        result.orientation = {static_cast<std::uint16_t>(first), reversed};
        return result;
    }

    template <class Id>
    std::size_t hash_entity_signature(
        const EntitySignature<Id>& signature) noexcept
    {
        std::size_t hash = sizeof(std::size_t) == 8
            ? std::size_t {14695981039346656037ULL}
            : std::size_t {2166136261U};
        for (Id vertex : signature) {
            hash ^= static_cast<std::size_t>(vertex);
            hash *= sizeof(std::size_t) == 8
                ? std::size_t {1099511628211ULL}
                : std::size_t {16777619U};
        }
        hash ^= signature.size();
        return hash;
    }

} // namespace idmesh::detail
