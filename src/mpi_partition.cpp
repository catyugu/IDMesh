#include "idmesh/mpi.hpp"
#include "contracts.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace idmesh::mpi {
    namespace {


        std::vector<std::vector<Index>> dual_graph(const Mesh& mesh)
        {
            const auto dim = mesh.dimension();
            ::idmesh::detail::require_argument(mesh.has_relation(dim - 1, dim),
                "graph partitioning requires Facet-to-Cell support");
            const auto cells = mesh.entity_count(dim);
            std::vector<std::vector<Index>> graph(cells);
            const auto support = mesh.relation(dim - 1, dim);
            for (Index f = 0; f < mesh.entity_count(dim - 1); ++f) {
                const auto adjacent = support[f].targets;
                for (std::size_t i = 0; i < adjacent.size(); ++i) {
                    for (std::size_t j = i + 1; j < adjacent.size(); ++j) {
                        if (adjacent[i] == adjacent[j])
                            continue;
                        graph[adjacent[i]].push_back(adjacent[j]);
                        graph[adjacent[j]].push_back(adjacent[i]);
                    }
                }
            }
            for (auto& neighbors : graph) {
                std::sort(neighbors.begin(), neighbors.end());
                neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
            }
            return graph;
        }

        std::vector<int> distance_from(const std::vector<std::vector<Index>>& graph,
            Index origin)
        {
            std::vector<int> distance(graph.size(), std::numeric_limits<int>::max());
            std::queue<Index> next;
            distance[origin] = 0;
            next.push(origin);
            while (!next.empty()) {
                const Index current = next.front();
                next.pop();
                for (auto neighbor : graph[current]) {
                    if (distance[neighbor] != std::numeric_limits<int>::max())
                        continue;
                    distance[neighbor] = distance[current] + 1;
                    next.push(neighbor);
                }
            }
            return distance;
        }

    } // namespace

    std::vector<int> partition_graph(const Mesh& source, int parts)
    {
        const Index count = source.entity_count(source.dimension());
        ::idmesh::detail::require_argument(parts > 0 && count >= static_cast<Index>(parts),
            "graph partitioner needs one or more owned cells per Rank");
        const auto graph = dual_graph(source);
        std::vector<Index> seeds {0};
        std::vector<int> nearest = distance_from(graph, 0);
        for (int p = 1; p < parts; ++p) {
            Index selected = invalid_index;
            int best = -1;
            for (Index cell = 0; cell < count; ++cell) {
                if (std::find(seeds.begin(), seeds.end(), cell) != seeds.end())
                    continue;
                if (selected == invalid_index || nearest[cell] > best) {
                    selected = cell;
                    best = nearest[cell];
                }
            }
            ::idmesh::detail::require_runtime(selected != invalid_index, "could not choose graph partition seed");
            seeds.push_back(selected);
            const auto distance = distance_from(graph, selected);
            for (Index cell = 0; cell < count; ++cell)
                nearest[cell] = std::min(nearest[cell], distance[cell]);
        }

        std::vector<int> owners(count, -1);
        std::vector<Index> target(parts), load(parts);
        std::vector<std::deque<Index>> frontiers(parts);
        for (int p = 0; p < parts; ++p) {
            target[p] = count / parts + (p < count % parts ? 1 : 0);
            owners[seeds[p]] = p;
            load[p] = 1;
            for (auto neighbor : graph[seeds[p]])
                frontiers[p].push_back(neighbor);
        }

        Index assigned = static_cast<Index>(parts);
        while (assigned < count) {
            bool progress = false;
            for (int p = 0; p < parts; ++p) {
                if (load[p] >= target[p])
                    continue;
                Index next = invalid_index;
                while (!frontiers[p].empty()) {
                    const Index candidate = frontiers[p].front();
                    frontiers[p].pop_front();
                    if (owners[candidate] == -1) {
                        next = candidate;
                        break;
                    }
                }
                if (next == invalid_index) {
                    // A disconnected component or exhausted frontier; choose
                    // an unassigned cell, preferring links to this partition.
                    int highest_score = -1;
                    for (Index cell = 0; cell < count; ++cell) {
                        if (owners[cell] != -1)
                            continue;
                        const int score = static_cast<int>(std::count_if(
                            graph[cell].begin(), graph[cell].end(),
                            [&](Index neighbor) { return owners[neighbor] == p; }));
                        if (next == invalid_index || score > highest_score) {
                            next = cell;
                            highest_score = score;
                        }
                    }
                }
                if (next == invalid_index)
                    continue;
                owners[next] = p;
                ++load[p];
                ++assigned;
                progress = true;
                for (auto neighbor : graph[next])
                    if (owners[neighbor] == -1)
                        frontiers[p].push_back(neighbor);
            }
            ::idmesh::detail::require_runtime(progress, "graph partitioner failed to balance cells");
        }
        return owners;
    }

    void write_partition_shards(const Mesh& source, std::span<const int> owners,
        Index ghost_layers, const std::string& prefix)
    {
        const Index d = source.dimension();
        const Index n = source.entity_count(d);
        ::idmesh::detail::require_argument(owners.size() == n && ghost_layers <= 8,
            "invalid shard owner count or ghost depth");
        ::idmesh::detail::require_argument(n > 0, "cannot export empty source mesh");
        int parts = 0;
        for (int owner : owners) {
            ::idmesh::detail::require_argument(owner >= 0 && owner < std::numeric_limits<int>::max() - 1,
                "invalid shard owner rank");
            parts = std::max(parts, owner + 1);
        }
        ::idmesh::detail::require_argument(parts >= 1, "empty shard partition");
        const auto graph = dual_graph(source);
        std::vector<std::vector<Index>> owned(parts);
        for (Index cell = 0; cell < n; ++cell)
            owned[owners[cell]].push_back(cell);

        for (int rank = 0; rank < parts; ++rank) {
            ::idmesh::detail::require_argument(!owned[rank].empty(), "every exported shard must own cells");
            std::vector<bool> seen(n);
            std::vector<Index> frontier;
            for (Index c : owned[rank]) {
                seen[c] = true;
                frontier.push_back(c);
            }
            std::set<Index> ghosts;
            for (Index depth = 0; depth < ghost_layers; ++depth) {
                std::vector<Index> next;
                for (Index cell : frontier) {
                    for (auto neighbor : graph[cell]) {
                        if (seen[neighbor])
                            continue;
                        seen[neighbor] = true;
                        next.push_back(neighbor);
                        if (owners[neighbor] != rank)
                            ghosts.insert(neighbor);
                    }
                }
                frontier = std::move(next);
            }
            std::vector<Index> local = owned[rank];
            local.insert(local.end(), ghosts.begin(), ghosts.end());

            std::set<Index> corners;
            std::set<Index> geometry;
            for (Index c : local) {
                for (Index v : source.entity_vertices(d, c))
                    corners.insert(v);
                const auto g = source.cell_geometry_nodes(c);
                for (std::size_t j = cell_vertex_count(source.cell_types()[c]);
                    j < g.size(); ++j)
                    geometry.insert(g[j]);
            }
            std::vector<const Label*> labels;
            std::map<std::pair<Index, int>, std::string> names;
            std::set<Index> local_cells(local.begin(), local.end());
            for (const auto& label : source.labels()) {
                if (label.dimension == d) {
                    if (label.entity == invalid_index || !local_cells.contains(label.entity))
                        continue;
                }
                else {
                    const auto all_corners_present = std::all_of(
                        label.vertices.begin(), label.vertices.end(),
                        [&](Index vertex) { return corners.contains(vertex); });
                    if (!all_corners_present)
                        continue;
                }
                labels.push_back(&label);
                auto name = source.label_name(label.dimension, label.value);
                if (!name.empty())
                    names[{label.dimension, label.value}] = std::string(name);
            }

            std::ofstream out(prefix + "." + std::to_string(rank) + ".idm");
            ::idmesh::detail::require_runtime(static_cast<bool>(out), "cannot create partition shard file");
            out << "IDMESH_SHARD 2\nDIM " << d << ' ' << source.geometric_dimension() << '\n';
            out << std::setprecision(17);
            out << "VERTICES " << corners.size() << '\n';
            for (Index id : corners) {
                const auto p = source.coordinates()[id];
                out << id << ' ' << p[0] << ' ' << p[1] << ' ' << p[2] << '\n';
            }
            out << "GEOMETRY_NODES " << geometry.size() << '\n';
            for (Index id : geometry) {
                const auto p = source.geometry_node_position(id);
                out << id << ' ' << p[0] << ' ' << p[1] << ' ' << p[2] << '\n';
            }
            out << "CELLS " << local.size() << '\n';
            for (Index cell : local) {
                const auto g = source.cell_geometry_nodes(cell);
                out << cell << ' ' << owners[cell] << ' '
                    << static_cast<int>(source.cell_types()[cell]) << ' ' << g.size();
                for (Index id : g)
                    out << ' ' << id;
                out << '\n';
            }
            out << "LABELS " << labels.size() << '\n';
            for (const auto* label : labels) {
                if (label->dimension == d) {
                    out << d << ' ' << label->value << " 1 " << label->entity << '\n';
                }
                else {
                    out << label->dimension << ' ' << label->value << ' ' << label->vertices.size();
                    for (Index vertex : label->vertices)
                        out << ' ' << vertex;
                    out << '\n';
                }
            }
            out << "NAMES " << names.size() << '\n';
            for (const auto& [key, name] : names)
                out << key.first << ' ' << key.second << ' ' << std::quoted(name) << '\n';
            out << "END\n";
            ::idmesh::detail::require_runtime(static_cast<bool>(out), "partition shard write failure");
        }
    }

} // namespace idmesh::mpi
