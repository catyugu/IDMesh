#pragma once

#include "idmesh/mesh.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace idmesh::detail {

    template <class Id, std::size_t Inline = 4>
    class EntitySignature {
    public:
        EntitySignature() = default;

        explicit EntitySignature(std::span<const Id> values)
        {
            assign(values);
        }

        void assign(std::span<const Id> values)
        {
            size_ = values.size();
            if (values.size() <= Inline) {
                large_.clear();
                std::copy(values.begin(), values.end(), small_.begin());
            }
            else {
                large_.assign(values.begin(), values.end());
            }
        }

        std::span<const Id> view() const noexcept
        {
            return size_ <= Inline
                ? std::span<const Id>(small_.data(), size_)
                : std::span<const Id>(large_.data(), large_.size());
        }

        std::size_t size() const noexcept { return size_; }
        bool empty() const noexcept { return size_ == 0; }
        const Id* begin() const noexcept { return view().data(); }
        const Id* end() const noexcept { return view().data() + view().size(); }
        Id operator[](std::size_t i) const noexcept { return view()[i]; }

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
        std::array<Id, Inline> small_ {};
        std::vector<Id> large_;
        std::size_t size_ = 0;
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

        // Standard-cell facets have at most four vertices. Keep their
        // canonicalization allocation-free; retain the larger-key fallback
        // for omitted-layer labels.
        std::array<Id, 4> small {};
        std::vector<Id> large(local.size() > small.size() ? local.size() : 0);
        std::span<Id> canonical = local.size() <= small.size()
            ? std::span<Id>(small).first(local.size())
            : std::span<Id>(large);
        for (std::size_t i = 0; i < local.size(); ++i) {
            const auto position = reversed
                ? (first + local.size() - i) % local.size()
                : (first + i) % local.size();
            canonical[i] = local[position];
        }
        result.signature.assign(canonical);
        result.orientation = {static_cast<std::uint16_t>(first), reversed};
        return result;
    }

    template <class Id, std::size_t Inline>
    std::size_t hash_entity_signature(
        const EntitySignature<Id, Inline>& signature) noexcept
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
