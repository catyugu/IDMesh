#include "shard.hpp"

#include "contracts.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <set>
#include <tuple>

namespace idmesh::detail::shard {

    Shard read(const std::string& path, int rank, int size)
    {
        std::ifstream file(path);
        ::idmesh::detail::require_runtime(
            static_cast<bool>(file),
            "cannot open local IDMesh partition shard");

        Shard shard;
        std::string word;
        int version = 0;
        ::idmesh::detail::require_runtime(
            static_cast<bool>(file >> word >> version)
                && word == magic && version == ::idmesh::detail::shard::version,
            "invalid IDMesh shard version");
        ::idmesh::detail::require_runtime(
            static_cast<bool>(file >> word >> shard.dimension
                >> shard.geometric_dimension)
                && word == "DIM"
                && shard.dimension >= 1 && shard.dimension <= 3
                && shard.geometric_dimension >= shard.dimension
                && shard.geometric_dimension <= 3,
            "invalid shard dimensions");

        std::size_t count = 0;
        ::idmesh::detail::require_runtime(
            static_cast<bool>(file >> word >> count)
                && word == "VERTICES" && count < invalid_index,
            "invalid shard VERTICES section");
        for (std::size_t i = 0; i < count; ++i) {
            std::uint64_t id = 0;
            int owner = -1;
            std::array<double, 3> p {};
            ::idmesh::detail::require_runtime(
                static_cast<bool>(file >> id >> owner
                    >> p[0] >> p[1] >> p[2])
                    && owner >= 0 && owner < size,
                "truncated or invalid shard vertex");
            ::idmesh::detail::require_argument(
                std::isfinite(p[0]) && std::isfinite(p[1])
                    && std::isfinite(p[2]),
                "nonfinite shard vertex");
            ::idmesh::detail::require_runtime(
                shard.vertices.emplace(id, Vertex {p, owner}).second,
                "duplicate shard vertex");
        }

        ::idmesh::detail::require_runtime(
            static_cast<bool>(file >> word >> count)
                && word == "GEOMETRY_NODES"
                && count < invalid_index,
            "invalid GEOMETRY_NODES section");
        for (std::size_t i = 0; i < count; ++i) {
            std::uint64_t id = 0;
            std::array<double, 3> p {};
            ::idmesh::detail::require_runtime(
                static_cast<bool>(file >> id
                    >> p[0] >> p[1] >> p[2]),
                "truncated shard high-order node");
            ::idmesh::detail::require_argument(
                std::isfinite(p[0]) && std::isfinite(p[1])
                    && std::isfinite(p[2]),
                "nonfinite high-order node");
            ::idmesh::detail::require_runtime(
                shard.geometry_nodes.emplace(id, p).second,
                "duplicate shard high-order node");
        }

        ::idmesh::detail::require_runtime(
            static_cast<bool>(file >> word >> count)
                && word == "CELLS" && count < invalid_index,
            "invalid shard CELLS section");
        std::set<std::uint64_t> seen_cells;
        for (std::size_t i = 0; i < count; ++i) {
            Cell cell;
            unsigned type = 0;
            std::size_t nodes = 0;
            ::idmesh::detail::require_runtime(
                static_cast<bool>(file >> cell.id >> cell.owner
                    >> type >> nodes),
                "invalid shard cell header");
            ::idmesh::detail::require_runtime(
                cell.owner >= 0 && cell.owner < size
                    && type
                        >= static_cast<unsigned>(CellType::segment)
                    && type
                        <= static_cast<unsigned>(CellType::pyramid),
                "invalid shard cell owner or type");
            cell.type = static_cast<CellType>(type);
            ::idmesh::detail::require_argument(
                cell_dimension(cell.type) == shard.dimension
                    && nodes >= cell_vertex_count(cell.type)
                    && nodes <= 27,
                "invalid cell dimension or geometry arity");
            cell.geometry.resize(nodes);
            for (auto& id : cell.geometry)
                ::idmesh::detail::require_runtime(
                    static_cast<bool>(file >> id),
                    "truncated shard cell geometry");
            for (Index j = 0;
                 j < cell_vertex_count(cell.type); ++j)
                ::idmesh::detail::require_argument(
                    shard.vertices.contains(cell.geometry[j]),
                    "cell references missing global corner vertex");
            for (std::size_t j = cell_vertex_count(cell.type);
                 j < nodes; ++j)
                ::idmesh::detail::require_argument(
                    shard.geometry_nodes.contains(cell.geometry[j]),
                    "cell references missing high-order geometry node");
            ::idmesh::detail::require_runtime(
                seen_cells.insert(cell.id).second,
                "duplicate shard global cell ID");
            shard.cells.push_back(std::move(cell));
        }

        ::idmesh::detail::require_runtime(
            static_cast<bool>(file >> word >> count)
                && word == "ENTITIES" && count < invalid_index,
            "invalid shard ENTITIES section");
        std::set<std::pair<Index, std::uint64_t>> seen_entities;
        for (std::size_t i = 0; i < count; ++i) {
            EntityIdentity entity;
            std::size_t arity = 0;
            ::idmesh::detail::require_runtime(
                static_cast<bool>(file >> entity.dimension
                    >> entity.id >> entity.owner >> arity)
                    && entity.dimension > 0
                    && entity.dimension < shard.dimension
                    && entity.owner >= 0 && entity.owner < size
                    && arity >= 2 && arity < invalid_index,
                "invalid shard entity identity header");
            entity.vertices.resize(arity);
            for (auto& vertex : entity.vertices) {
                ::idmesh::detail::require_runtime(
                    static_cast<bool>(file >> vertex),
                    "truncated shard entity identity");
                ::idmesh::detail::require_runtime(
                    shard.vertices.contains(vertex),
                    "entity identity references missing vertex");
            }
            ::idmesh::detail::require_runtime(
                seen_entities.emplace(
                    entity.dimension, entity.id).second,
                "duplicate shard entity global ID");
            shard.entities.push_back(std::move(entity));
        }

        ::idmesh::detail::require_runtime(
            static_cast<bool>(file >> word >> count)
                && word == "HALOS" && count < invalid_index,
            "invalid shard HALOS section");
        std::set<std::pair<Index, int>> seen_halos;
        for (std::size_t i = 0; i < count; ++i) {
            HaloRecord halo;
            std::size_t send_count = 0, receive_count = 0;
            ::idmesh::detail::require_runtime(
                static_cast<bool>(file >> halo.dimension >> halo.peer
                    >> send_count >> receive_count)
                    && halo.dimension <= shard.dimension
                    && halo.peer >= 0 && halo.peer < size
                    && halo.peer != rank
                    && send_count < invalid_index
                    && receive_count < invalid_index,
                "invalid shard halo header");
            ::idmesh::detail::require_runtime(
                seen_halos.emplace(
                    halo.dimension, halo.peer).second,
                "duplicate shard halo peer record");
            halo.send.resize(send_count);
            halo.receive.resize(receive_count);
            for (auto& id : halo.send)
                ::idmesh::detail::require_runtime(
                    static_cast<bool>(file >> id),
                    "truncated shard halo send list");
            for (auto& id : halo.receive)
                ::idmesh::detail::require_runtime(
                    static_cast<bool>(file >> id),
                    "truncated shard halo receive list");
            shard.halos.push_back(std::move(halo));
        }

        ::idmesh::detail::require_runtime(
            static_cast<bool>(file >> word >> count)
                && word == "LABELS",
            "missing shard LABELS section");
        for (std::size_t i = 0; i < count; ++i) {
            Label label;
            std::size_t n = 0;
            ::idmesh::detail::require_runtime(
                static_cast<bool>(file >> label.dimension
                    >> label.value >> n)
                    && label.dimension <= shard.dimension
                    && n >= 1 && n < invalid_index,
                "invalid shard label header");
            const bool valid_arity
                = label.dimension == shard.dimension
                || label.dimension == 0
                ? n == 1
                : label.dimension == 1 ? n == 2 : n >= 3;
            ::idmesh::detail::require_runtime(
                valid_arity, "invalid shard label arity");
            label.entities.resize(n);
            for (auto& id : label.entities)
                ::idmesh::detail::require_runtime(
                    static_cast<bool>(file >> id),
                    "truncated shard label");
            shard.labels.push_back(std::move(label));
        }

        ::idmesh::detail::require_runtime(
            static_cast<bool>(file >> word >> count)
                && word == "NAMES",
            "missing shard NAMES section");
        for (std::size_t i = 0; i < count; ++i) {
            Index dimension = 0;
            int value = 0;
            std::string name;
            ::idmesh::detail::require_runtime(
                static_cast<bool>(file >> dimension >> value
                    >> std::quoted(name))
                    && dimension <= shard.dimension,
                "invalid shard label name");
            shard.label_names[{dimension, value}]
                = std::move(name);
        }

        ::idmesh::detail::require_runtime(
            static_cast<bool>(file >> word) && word == "END",
            "missing shard END marker");
        ::idmesh::detail::require_runtime(
            !(file >> word),
            "trailing garbage after shard END");
        ::idmesh::detail::require_runtime(
            !shard.cells.empty(),
            "shard requires at least one local cell");

        std::stable_sort(shard.cells.begin(), shard.cells.end(),
            [rank](const Cell& a, const Cell& b) {
                return std::tuple {a.owner != rank, a.id}
                    < std::tuple {b.owner != rank, b.id};
            });
        return shard;
    }

} // namespace idmesh::detail::shard
