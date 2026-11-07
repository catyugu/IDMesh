#include "idmesh/mesh.hpp"
#include "contracts.hpp"
#include "entity_signature.hpp"
#include "reference_cell.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace idmesh {
    namespace {

        using Loop = detail::EntitySignature<Index>;

        struct LoopHash {
            std::size_t operator()(const Loop& key) const noexcept
            {
                return detail::hash_entity_signature(key);
            }
        };

        Loop boundary_loop(
            CellType type, Index facet, std::span<const Index> vertices)
        {
            std::array<Index, 4> local {};
            const auto local_vertices = reference_facet_vertices(type, facet);
            IDMESH_ASSERT(local_vertices.size() <= local.size());
            for (std::size_t i = 0; i < local_vertices.size(); ++i)
                local[i] = vertices[local_vertices[i]];
            return Loop(std::span<const Index>(
                local.data(), local_vertices.size()));
        }

        Loop polygon_edge(
            std::span<const Index> vertices, std::size_t edge)
        {
            const std::array<Index, 2> endpoints {
                vertices[edge], vertices[(edge + 1) % vertices.size()]};
            return Loop(endpoints);
        }

        struct RelationData {
            bool present = false;
            Index rows = 0;
            Index stride = 0;
            std::vector<Index> offsets;
            std::vector<Index> indices;
            std::vector<std::int8_t> signs;
            std::vector<Orientation> orientations;

            RelationView view() const
            {
                return {offsets, indices, signs, orientations, rows, stride};
            }
        };

        struct Incidence {
            Index entity = invalid_index;
            std::int8_t coefficient = 0;
            Orientation orientation {};
        };

        // Build-time incidences stay contiguous instead of allocating one
        // vector per entity. Stable scattering preserves the original row order.
        struct PendingIncidence {
            Index row;
            Incidence entry;
        };

        using PendingRelation = std::vector<PendingIncidence>;

        RelationData allocate_relation(std::span<const Index> counts, bool oriented)
        {
            detail::require_length(counts.size() < invalid_index, "too many relation rows");
            RelationData result;
            result.present = true;
            result.rows = static_cast<Index>(counts.size());
            std::size_t total = 0;
            for (Index count : counts) {
                total += count;
                detail::require_length(total < invalid_index, "too many relation entries");
            }
            const bool fixed = counts.empty()
                || std::all_of(counts.begin(), counts.end(),
                    [first = counts.front()](Index count) { return count == first; });
            if (fixed) {
                result.stride = counts.empty() ? 0 : counts.front();
            }
            else {
                result.offsets.resize(counts.size() + 1);
                for (std::size_t row = 0; row < counts.size(); ++row)
                    result.offsets[row + 1] = result.offsets[row] + counts[row];
            }
            result.indices.resize(total);
            if (oriented) {
                result.signs.resize(total);
                result.orientations.resize(total);
            }
            return result;
        }

        std::vector<Index> write_positions(const RelationData& relation)
        {
            std::vector<Index> positions(relation.rows);
            if (relation.offsets.empty()) {
                for (Index row = 0; row < relation.rows; ++row)
                    positions[row] = static_cast<Index>(static_cast<std::size_t>(row) * relation.stride);
            }
            else {
                std::copy(relation.offsets.begin(), relation.offsets.end() - 1, positions.begin());
            }
            return positions;
        }

        RelationData compress(const PendingRelation& entries, Index row_count, bool oriented)
        {
            std::vector<Index> counts(row_count, 0);
            for (const auto& pending : entries) {
                IDMESH_ASSERT(pending.row < row_count);
                detail::require_length(counts[pending.row] < invalid_index, "relation row overflow");
                ++counts[pending.row];
            }
            auto result = allocate_relation(counts, oriented);
            auto positions = write_positions(result);
            for (const auto& pending : entries) {
                const Index slot = positions[pending.row]++;
                result.indices[slot] = pending.entry.entity;
                if (oriented) {
                    result.signs[slot] = pending.entry.coefficient;
                    result.orientations[slot] = pending.entry.orientation;
                }
            }
            return result;
        }

        // Transpose the finalized forward arrays directly. Unlike a
        // vector-of-vectors transpose, memory use is linear and predictable.
        RelationData transpose(const RelationData& forward, Index row_count)
        {
            std::vector<Index> counts(row_count, 0);
            for (Index target : forward.indices) {
                IDMESH_ASSERT(target < row_count);
                detail::require_length(counts[target] < invalid_index, "transposed relation row overflow");
                ++counts[target];
            }
            const bool oriented = !forward.signs.empty();
            auto result = allocate_relation(counts, oriented);
            auto positions = write_positions(result);
            const auto view = forward.view();
            for (Index row = 0; row < forward.rows; ++row) {
                const auto incidence = view[row];
                for (std::size_t i = 0; i < incidence.targets.size(); ++i) {
                    const Index slot = positions[incidence.targets[i]]++;
                    result.indices[slot] = row;
                    if (oriented) {
                        result.signs[slot] = incidence.coefficients[i];
                        result.orientations[slot] = incidence.orientations[i];
                    }
                }
            }
            return result;
        }

    } // namespace

    struct Mesh::Data {
        Index dimension = 0;
        Index geometric_dimension = 0;
        // Top-dimensional Cell connectivity enters already packed from MeshBuilder.
        std::array<Index, 4> counts {};
        std::array<bool, 4> enabled {};
        std::array<std::vector<Index>, 4> vertex_offsets;
        std::array<std::vector<Index>, 4> vertex_indices;
        std::array<std::array<RelationData, 4>, 4> relations;
        std::vector<std::array<double, 3>> coordinates;
        std::vector<CellType> types;
        std::vector<Label> labels;
        std::vector<Index> unbound_label_vertices;
        std::map<std::pair<Index, int>, std::string> label_names;
        std::vector<std::array<double, 3>> extra_geometry_nodes;
        std::vector<Index> geometry_offsets;
        std::vector<Index> geometry_indices;
        std::vector<std::uint8_t> cell_orders;
    };

    struct MeshBuilder::Data {
        Index dimension_ = 0;
        Index geometric_dimension_ = 0;
        std::vector<std::array<double, 3>> coordinates_;
        std::vector<CellType> types_;
        std::vector<Index> cell_offsets_;
        std::vector<Index> cell_vertices_;
        std::vector<std::array<double, 3>> extra_geometry_nodes_;
        std::vector<Index> cell_extra_offsets_;
        std::vector<Index> cell_extra_nodes_;
        std::vector<std::uint8_t> cell_orders_;

        struct LabelAssignment {
            Index dimension;
            Index begin;
            Index end;
            int value;
        };
        std::vector<Index> label_assignment_vertices_;
        std::vector<LabelAssignment> label_assignments_;
        std::vector<std::pair<Index, int>> cell_labels_;
        std::map<std::pair<Index, int>, std::string> label_names_;
    };

    Index cell_facet_count(CellType type)
    {
        return static_cast<Index>(detail::reference_cell(type).facets.size());
    }

    std::span<const std::uint8_t> reference_facet_vertices(CellType type, Index facet)
    {
        const auto reference = detail::reference_cell(type);
        if (facet >= reference.facets.size())
            throw std::out_of_range("reference facet index");
        const auto& entry = reference.facets[facet];
        return {entry.vertices.data(), entry.size};
    }

    Index cell_dimension(CellType type)
    {
        return detail::reference_cell(type).dimension;
    }

    Index cell_vertex_count(CellType type)
    {
        return detail::reference_cell(type).vertices;
    }

    RelationRow RelationView::operator[](Index row) const
    {
        if (row >= rows)
            throw std::out_of_range("relation row");
        const auto begin = offsets.empty()
            ? static_cast<std::size_t>(row) * stride
            : offsets[row];
        const auto end = offsets.empty() ? begin + stride : offsets[row + 1];
        const auto count = end - begin;
        return {
            targets.subspan(begin, count),
            coefficients.empty() ? std::span<const std::int8_t> {}
                                 : coefficients.subspan(begin, count),
            orientations.empty() ? std::span<const Orientation> {}
                                 : orientations.subspan(begin, count)};
    }

    Mesh::Mesh(std::shared_ptr<const Data> data)
        : data_(std::move(data))
    {
    }
    Index Mesh::dimension() const
    {
        if (!data_)
            throw std::logic_error("empty mesh");
        return data_->dimension;
    }
    Index Mesh::geometric_dimension() const
    {
        if (!data_)
            throw std::logic_error("empty mesh");
        return data_->geometric_dimension;
    }
    Index Mesh::entity_count(Index dimension) const
    {
        if (!data_)
            throw std::logic_error("empty mesh");
        if (dimension > data_->dimension)
            throw std::out_of_range("entity dimension");
        return data_->counts[dimension];
    }
    bool Mesh::has_entities(Index dimension) const
    {
        return data_ && dimension <= data_->dimension && data_->enabled[dimension];
    }
    bool Mesh::has_relation(Index from, Index to) const
    {
        return data_ && from < 4 && to < 4 && data_->relations[from][to].present;
    }
    RelationView Mesh::relation(Index from, Index to) const
    {
        if (!has_relation(from, to))
            throw std::invalid_argument("relation not materialized");
        return data_->relations[from][to].view();
    }
    std::span<const Index> Mesh::entity_vertices(Index dimension, Index id) const
    {
        if (!data_)
            throw std::logic_error("empty mesh");
        if (dimension > data_->dimension || id >= data_->counts[dimension])
            throw std::out_of_range("entity index");
        const auto& offsets = data_->vertex_offsets[dimension];
        const auto& indices = data_->vertex_indices[dimension];
        return {indices.data() + offsets[id], offsets[id + 1] - offsets[id]};
    }
    std::span<const std::array<double, 3>> Mesh::coordinates() const
    {
        if (!data_)
            throw std::logic_error("empty mesh");
        return data_->coordinates;
    }
    std::span<const CellType> Mesh::cell_types() const
    {
        if (!data_)
            throw std::logic_error("empty mesh");
        return data_->types;
    }

    std::span<const Label> Mesh::labels() const
    {
        if (!data_)
            throw std::logic_error("empty mesh");
        return data_->labels;
    }

    std::span<const Label> Mesh::labels(Index dimension, Index entity) const
    {
        if (!data_)
            throw std::logic_error("empty mesh");
        if (dimension > data_->dimension || entity >= data_->counts[dimension])
            throw std::out_of_range("label entity index");
        const auto key = std::pair {dimension, entity};
        const auto first = std::lower_bound(data_->labels.begin(), data_->labels.end(), key,
            [](const Label& label, const auto& value) {
                return std::pair {label.dimension, label.entity} < value;
            });
        const auto last = std::upper_bound(first, data_->labels.end(), key,
            [](const auto& value, const Label& label) {
                return value < std::pair {label.dimension, label.entity};
            });
        const auto begin = static_cast<std::size_t>(first - data_->labels.begin());
        return std::span<const Label>(data_->labels).subspan(begin, static_cast<std::size_t>(last - first));
    }

    std::string_view Mesh::label_name(Index dimension, int value) const
    {
        if (!data_)
            throw std::logic_error("empty mesh");
        const auto found = data_->label_names.find({dimension, value});
        return found == data_->label_names.end() ? std::string_view {} : found->second;
    }

    std::uint8_t Mesh::cell_order(Index cell) const
    {
        if (!data_ || cell >= data_->cell_orders.size())
            throw std::out_of_range("cell order");
        return data_->cell_orders[cell];
    }

    std::span<const Index> Mesh::cell_geometry_nodes(Index cell) const
    {
        if (!data_ || cell >= data_->cell_orders.size())
            throw std::out_of_range("cell geometry nodes");
        // First-order geometry reuses the existing Cell-to-Vertex connectivity.
        if (data_->cell_orders[cell] == 1)
            return entity_vertices(data_->dimension, cell);
        const auto begin = data_->geometry_offsets[cell];
        const auto end = data_->geometry_offsets[cell + 1];
        return {data_->geometry_indices.data() + begin, end - begin};
    }

    std::array<double, 3> Mesh::geometry_node_position(Index node) const
    {
        if (!data_)
            throw std::logic_error("empty mesh");
        if (node < data_->coordinates.size())
            return data_->coordinates[node];
        const auto extra = static_cast<std::size_t>(node) - data_->coordinates.size();
        if (extra >= data_->extra_geometry_nodes.size())
            throw std::out_of_range("geometry node index");
        return data_->extra_geometry_nodes[extra];
    }

    Index Mesh::geometry_node_count() const
    {
        if (!data_)
            throw std::logic_error("empty mesh");
        return static_cast<Index>(data_->coordinates.size() + data_->extra_geometry_nodes.size());
    }

    MeshBuilder::MeshBuilder(Index dimension, Index geometric_dimension)
        : data_(std::make_unique<Data>())
    {
        detail::require_argument(dimension >= 1 && dimension <= 3, "topological dimension must be 1..3");
        detail::require_argument(geometric_dimension >= dimension && geometric_dimension <= 3,
            "invalid geometric dimension");
        data_->dimension_ = dimension;
        data_->geometric_dimension_ = geometric_dimension;
        data_->cell_offsets_.push_back(0);
        data_->cell_extra_offsets_.push_back(0);
    }

    MeshBuilder::~MeshBuilder() = default;
    MeshBuilder::MeshBuilder(MeshBuilder&&) noexcept = default;
    MeshBuilder& MeshBuilder::operator=(MeshBuilder&&) noexcept = default;

    Index MeshBuilder::add_vertex(std::array<double, 3> position)
    {
        for (double coordinate : position)
            detail::require_argument(std::isfinite(coordinate), "nonfinite coordinate");
        detail::require_length(data_->coordinates_.size() < invalid_index, "too many vertices");
        data_->coordinates_.push_back(position);
        return static_cast<Index>(data_->coordinates_.size() - 1);
    }

    Index MeshBuilder::add_cell(CellType type, std::span<const Index> vertices)
    {
        detail::require_argument(cell_dimension(type) == data_->dimension_, "cell dimension mismatch");
        detail::require_argument(vertices.size() == cell_vertex_count(type), "incorrect cell arity");
        std::array<Index, 8> sorted {};
        for (std::size_t i = 0; i < vertices.size(); ++i) {
            detail::require_argument(vertices[i] < data_->coordinates_.size(), "invalid vertex index");
            sorted[i] = vertices[i];
        }
        std::sort(sorted.begin(), sorted.begin() + vertices.size());
        detail::require_argument(std::adjacent_find(sorted.begin(), sorted.begin() + vertices.size())
                == sorted.begin() + vertices.size(),
            "duplicate cell vertex");
        detail::require_length(data_->types_.size() < invalid_index, "too many cells");
        detail::require_length(data_->cell_vertices_.size() < std::size_t(invalid_index) - vertices.size(),
            "cell connectivity overflow");
        data_->cell_vertices_.insert(data_->cell_vertices_.end(), vertices.begin(), vertices.end());
        data_->cell_offsets_.push_back(static_cast<Index>(data_->cell_vertices_.size()));
        data_->types_.push_back(type);
        data_->cell_orders_.push_back(1);
        data_->cell_extra_offsets_.push_back(static_cast<Index>(data_->cell_extra_nodes_.size()));
        return static_cast<Index>(data_->types_.size() - 1);
    }

    Index MeshBuilder::add_geometry_node(std::array<double, 3> position)
    {
        for (double coordinate : position)
            detail::require_argument(std::isfinite(coordinate), "nonfinite geometry node coordinate");
        detail::require_length(data_->extra_geometry_nodes_.size() < invalid_index, "too many geometry nodes");
        data_->extra_geometry_nodes_.push_back(position);
        return static_cast<Index>(data_->extra_geometry_nodes_.size() - 1);
    }

    Index MeshBuilder::add_quadratic_cell(
        CellType type, std::span<const Index> vertices,
        std::span<const Index> extra_nodes)
    {
        const auto count = vertices.size() + extra_nodes.size();
        bool valid = false;
        switch (type) {
        case CellType::segment:
            valid = count == 3;
            break;
        case CellType::triangle:
            valid = count == 6;
            break;
        case CellType::quadrilateral:
            valid = count == 8 || count == 9;
            break;
        case CellType::tetrahedron:
            valid = count == 10;
            break;
        case CellType::hexahedron:
            valid = count == 20 || count == 27;
            break;
        case CellType::prism:
            valid = count == 15 || count == 18;
            break;
        case CellType::pyramid:
            valid = count == 13 || count == 14;
            break;
        }
        detail::require_argument(valid && vertices.size() == cell_vertex_count(type),
            "unsupported quadratic geometry node count");
        for (Index index : extra_nodes)
            detail::require_argument(index < data_->extra_geometry_nodes_.size(), "invalid extra geometry node");
        detail::require_length(data_->cell_extra_nodes_.size() < std::size_t(invalid_index) - extra_nodes.size(),
            "geometry connectivity overflow");
        const Index cell = add_cell(type, vertices);
        data_->cell_orders_[cell] = 2;
        data_->cell_extra_nodes_.insert(
            data_->cell_extra_nodes_.end(), extra_nodes.begin(), extra_nodes.end());
        data_->cell_extra_offsets_.back() = static_cast<Index>(data_->cell_extra_nodes_.size());
        return cell;
    }

    void MeshBuilder::add_cell_label(Index cell, int value)
    {
        detail::require_argument(cell < data_->types_.size(), "invalid cell for label");
        data_->cell_labels_.emplace_back(cell, value);
    }

    void MeshBuilder::add_label(
        Index dimension, std::span<const Index> vertices, int value)
    {
        detail::require_argument(dimension < data_->dimension_, "use add_cell_label for top-dimensional labels");
        const bool arity = dimension == 0 ? vertices.size() == 1
            : dimension == 1              ? vertices.size() == 2
                                          : vertices.size() >= 3;
        detail::require_argument(arity, "invalid labeled entity arity");
        std::vector<Index> sorted(vertices.begin(), vertices.end());
        for (Index vertex : sorted)
            detail::require_argument(vertex < data_->coordinates_.size(),
                "invalid labeled entity vertex");
        std::sort(sorted.begin(), sorted.end());
        detail::require_argument(
            std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(),
            "duplicate labeled entity vertex");
        detail::require_length(data_->label_assignment_vertices_.size() < std::size_t(invalid_index) - vertices.size(),
            "label connectivity overflow");
        const auto begin = static_cast<Index>(data_->label_assignment_vertices_.size());
        data_->label_assignment_vertices_.insert(
            data_->label_assignment_vertices_.end(), vertices.begin(), vertices.end());
        const auto end = static_cast<Index>(data_->label_assignment_vertices_.size());
        data_->label_assignments_.push_back({dimension, begin, end, value});
    }

    void MeshBuilder::set_label_name(Index dimension, int value, std::string name)
    {
        detail::require_argument(dimension <= data_->dimension_, "label name dimension out of range");
        data_->label_names_[{dimension, value}] = std::move(name);
    }

    namespace {

        void move_base_storage(auto& source, auto& data)
        {
            data.dimension = source.dimension_;
            data.geometric_dimension = source.geometric_dimension_;
            data.coordinates = std::move(source.coordinates_);
            data.counts[source.dimension_] = static_cast<Index>(source.types_.size());
            data.vertex_offsets[source.dimension_] = std::move(source.cell_offsets_);
            data.vertex_indices[source.dimension_] = std::move(source.cell_vertices_);
            data.types = std::move(source.types_);
            data.cell_orders = std::move(source.cell_orders_);
            data.extra_geometry_nodes = std::move(source.extra_geometry_nodes_);
            data.label_names = std::move(source.label_names_);
        }

        void pack_high_order_geometry(auto& source, auto& data)
        {
            if (!std::any_of(data.cell_orders.begin(), data.cell_orders.end(),
                    [](std::uint8_t order) { return order == 2; }))
                return;

            data.geometry_offsets.reserve(
                static_cast<std::size_t>(data.counts[data.dimension]) + 1);
            data.geometry_offsets.push_back(0);
            const auto cell_vertices = [&](Index cell) -> std::span<const Index> {
                const auto& offsets = data.vertex_offsets[data.dimension];
                const auto& indices = data.vertex_indices[data.dimension];
                const auto begin = offsets[cell];
                return {indices.data() + begin, offsets[cell + 1] - begin};
            };
            for (Index c = 0; c < data.counts[data.dimension]; ++c) {
                if (data.cell_orders[c] == 2) {
                    const auto corners = cell_vertices(c);
                    data.geometry_indices.insert(
                        data.geometry_indices.end(), corners.begin(), corners.end());
                    const Index begin = source.cell_extra_offsets_[c];
                    const Index finish = source.cell_extra_offsets_[c + 1];
                    for (Index i = begin; i < finish; ++i) {
                        const auto node = static_cast<std::size_t>(source.cell_extra_nodes_[i])
                            + data.coordinates.size();
                        detail::require_length(node < invalid_index,
                            "geometry node index overflow");
                        data.geometry_indices.push_back(static_cast<Index>(node));
                    }
                    detail::require_length(data.geometry_indices.size() < invalid_index,
                        "geometry connectivity overflow");
                }
                data.geometry_offsets.push_back(
                    static_cast<Index>(data.geometry_indices.size()));
            }
            std::vector<Index>().swap(source.cell_extra_offsets_);
            std::vector<Index>().swap(source.cell_extra_nodes_);
        }

        void finalize_topology(auto& source, auto& data, BuildOptions options)
        {
            std::array<std::vector<Loop>, 4> entities;
            const auto construction_vertices =
                [&](Index dim, Index id) -> std::span<const Index> {
                if (dim == data.dimension) {
                    const auto& offsets = data.vertex_offsets[dim];
                    const auto& indices = data.vertex_indices[dim];
                    const auto begin = offsets[id];
                    return {indices.data() + begin, offsets[id + 1] - begin};
                }
                return entities[dim][id].view();
            };

            // Vertex entities have identity connectivity. Avoid constructing one
            // separately allocated singleton vector per vertex during finalize.
            std::array<std::unordered_map<Loop, Index, LoopHash>, 4> lookup;
            std::array<std::array<PendingRelation, 4>, 4> incidence;
            const auto want = [&](Index dim) {
                if (dim == 0 || dim == source.dimension_)
                    return true;
                switch (options.topology) {
                case TopologyLayout::cell_vertices:
                    return false;
                case TopologyLayout::cell_facets:
                    return dim == source.dimension_ - 1;
                case TopologyLayout::complete:
                    return true;
                }
                return false;
            };
            for (Index dim = 0; dim <= source.dimension_; ++dim)
                data.enabled[dim] = want(dim);
            if (source.dimension_ > 1 && want(source.dimension_ - 1))
                lookup[source.dimension_ - 1].reserve(data.types.size() * 2);
            auto register_entity = [&](Index dim, std::span<const Index> local) -> Incidence {
                auto canon = detail::canonicalize_entity<Index>(local);
                detail::require_length(entities[dim].size() < invalid_index, "too many topology entities");
                auto [it, inserted] = lookup[dim].try_emplace(canon.signature,
                    static_cast<Index>(entities[dim].size()));
                if (inserted)
                    entities[dim].push_back(canon.signature);
                return {it->second, static_cast<std::int8_t>(canon.orientation.sign()), canon.orientation};
            };
            auto attach = [&](Index from, Index to, Index id, Incidence entry) {
                incidence[from][to].push_back({id, entry});
            };
    
            // Cell-to-vertex is permanently available even with intermediate dimensions omitted.
            for (Index c = 0; c < data.counts[source.dimension_]; ++c) {
                const auto vertices = construction_vertices(source.dimension_, c);
                for (Index local = 0; local < vertices.size(); ++local) {
                    auto sign = source.dimension_ == 1 ? (local == 0 ? -1 : 1) : 0;
                    attach(source.dimension_, 0, c, {vertices[local], static_cast<std::int8_t>(sign), {}});
                }
                if (source.dimension_ == 1)
                    continue;
                if (want(source.dimension_ - 1)) {
                    for (Index facet = 0; facet < cell_facet_count(data.types[c]); ++facet) {
                        const auto face = boundary_loop(data.types[c], facet, vertices);
                        attach(source.dimension_, source.dimension_ - 1, c,
                            register_entity(source.dimension_ - 1, face.view()));
                    }
                }
            }
    
            if (source.dimension_ == 3 && want(1) && want(2)) {
                for (Index f = 0; f < entities[2].size(); ++f) {
                    const auto vertices = entities[2][f].view();
                    for (std::size_t edge = 0; edge < vertices.size(); ++edge) {
                        const auto endpoints = polygon_edge(vertices, edge);
                        attach(2, 1, f, register_entity(1, endpoints.view()));
                    }
                }
            }
            if (source.dimension_ >= 2 && want(1)) {
                for (Index e = 0; e < entities[1].size(); ++e) {
                    const auto& vertices = entities[1][e];
                    IDMESH_ASSERT(vertices.size() == 2);
                    attach(1, 0, e, {vertices[0], -1, {}});
                    attach(1, 0, e, {vertices[1], 1, {}});
                }
            }
    
            // Direct facet-to-vertex traversal is needed by finite-volume kernels
            // even when intermediate edges have not been materialized.
            if (source.dimension_ == 3 && want(2)) {
                for (Index f = 0; f < entities[2].size(); ++f) {
                    for (auto v : entities[2][f])
                        attach(2, 0, f, {v, 0, {}});
                }
            }
    
            // Resolve and deduplicate labels before replacing temporary entity keys
            // with borrowed spans into the finalized connectivity.
            struct PendingLabel {
                Index dimension = 0;
                Index entity = invalid_index;
                Loop unbound_vertices;
                int value = 0;
            };
            std::vector<PendingLabel> labels;
            labels.reserve(source.cell_labels_.size() + source.label_assignments_.size());
            for (const auto& [cell, value] : source.cell_labels_)
                labels.push_back({source.dimension_, cell, {}, value});
            std::vector<std::pair<Index, int>>().swap(source.cell_labels_);
            for (const auto& assignment : source.label_assignments_) {
                const Index dim = assignment.dimension;
                const std::span<const Index> assigned {
                    source.label_assignment_vertices_.data() + assignment.begin,
                    assignment.end - assignment.begin};
                const auto key = detail::canonicalize_entity<Index>(assigned);
                Index entity = invalid_index;
                if (dim == 0) {
                    entity = assigned.front();
                }
                else if (want(dim)) {
                    const auto found = lookup[dim].find(key.signature);
                    if (found != lookup[dim].end())
                        entity = found->second;
                }
                PendingLabel label {dim, entity, {}, assignment.value};
                if (entity == invalid_index)
                    label.unbound_vertices = key.signature;
                labels.push_back(label);
            }
            std::vector<Index>().swap(source.label_assignment_vertices_);
            decltype(source.label_assignments_) {}.swap(source.label_assignments_);
            for (auto& table : lookup)
                std::unordered_map<Loop, Index, LoopHash> {}.swap(table);
    
            std::sort(labels.begin(), labels.end(), [](const PendingLabel& a, const PendingLabel& b) {
                const auto a_head = std::tie(a.dimension, a.entity, a.value);
                const auto b_head = std::tie(b.dimension, b.entity, b.value);
                if (a_head != b_head)
                    return a_head < b_head;
                return std::lexicographical_compare(
                    a.unbound_vertices.begin(), a.unbound_vertices.end(),
                    b.unbound_vertices.begin(), b.unbound_vertices.end());
            });
            labels.erase(std::unique(labels.begin(), labels.end(),
                             [](const PendingLabel& a, const PendingLabel& b) {
                                 return a.dimension == b.dimension && a.entity == b.entity
                                     && a.value == b.value
                                     && a.unbound_vertices == b.unbound_vertices;
                             }),
                labels.end());
            for (Index dim = 0; dim <= source.dimension_; ++dim) {
                auto& offsets = data.vertex_offsets[dim];
                auto& indices = data.vertex_indices[dim];
                if (dim == 0) {
                    const auto count = static_cast<Index>(data.coordinates.size());
                    data.counts[0] = count;
                    offsets.resize(static_cast<std::size_t>(count) + 1);
                    indices.resize(count);
                    for (Index v = 0; v < count; ++v) {
                        offsets[v] = v;
                        indices[v] = v;
                    }
                    offsets[count] = count;
                    continue;
                }
                if (dim == source.dimension_)
                    continue;
                auto& lists = entities[dim];
                data.counts[dim] = static_cast<Index>(lists.size());
                offsets.reserve(lists.size() + 1);
                offsets.push_back(0);
                for (const auto& vertices : lists) {
                    indices.insert(indices.end(), vertices.begin(), vertices.end());
                    offsets.push_back(static_cast<Index>(indices.size()));
                }
                std::vector<Loop>().swap(lists);
            }
            for (Index from = 1; from <= source.dimension_; ++from) {
                for (Index to = 0; to < from; ++to) {
                    auto& entries = incidence[from][to];
                    if (entries.empty())
                        continue;
                    // Only consecutive dimensions form oriented boundary operators.
                    // Direct geometric connectivity omits redundant sign/permutation arrays.
                    const bool oriented = from == to + 1;
                    auto relation = compress(entries, data.counts[from], oriented);
                    PendingRelation().swap(entries);
                    if (options.relations == RelationDirection::bidirectional && from == to + 1) {
                        data.relations[to][from] = transpose(
                            relation, data.counts[to]);
                    }
                    data.relations[from][to] = std::move(relation);
                }
            }
            // Materialized entities borrow their already-packed connectivity.
            // Only labels on absent topology layers keep private vertex keys.
            std::size_t unbound_entries = 0;
            for (const auto& label : labels)
                if (label.entity == invalid_index)
                    unbound_entries += label.unbound_vertices.size();
            data.unbound_label_vertices.reserve(unbound_entries);
            data.labels.reserve(labels.size());
            for (const auto& label : labels) {
                std::span<const Index> vertices;
                if (label.entity != invalid_index) {
                    const auto& offsets = data.vertex_offsets[label.dimension];
                    const auto& indices = data.vertex_indices[label.dimension];
                    const auto begin = offsets[label.entity];
                    vertices = {indices.data() + begin, offsets[label.entity + 1] - begin};
                }
                else {
                    const auto begin = data.unbound_label_vertices.size();
                    data.unbound_label_vertices.insert(data.unbound_label_vertices.end(),
                        label.unbound_vertices.begin(), label.unbound_vertices.end());
                    vertices = {data.unbound_label_vertices.data() + begin,
                        label.unbound_vertices.size()};
                }
                data.labels.push_back({label.dimension, label.entity, vertices, label.value});
            }
    
        }

    } // namespace

    Mesh MeshBuilder::finalize(BuildOptions options) &&
    {
        auto data = std::make_shared<Mesh::Data>();
        move_base_storage(*data_, *data);
        pack_high_order_geometry(*data_, *data);
        finalize_topology(*data_, *data, options);

        Mesh mesh(std::move(data));
        mesh.validate_structure();
        return mesh;
    }

    void Mesh::validate_structure() const
    {
        if (!data_)
            throw std::logic_error("empty mesh");
        if (data_->cell_orders.size() != data_->counts[data_->dimension])
            throw std::logic_error("invalid cell geometry count");
        if (data_->geometry_offsets.empty()) {
            if (!data_->geometry_indices.empty() || std::any_of(data_->cell_orders.begin(), data_->cell_orders.end(), [](std::uint8_t order) { return order != 1; }))
                throw std::logic_error("missing high-order geometry connectivity");
        }
        else {
            if (data_->geometry_offsets.size() != data_->cell_orders.size() + 1 || data_->geometry_offsets.front() != 0 || data_->geometry_offsets.back() != data_->geometry_indices.size() || !std::is_sorted(data_->geometry_offsets.begin(), data_->geometry_offsets.end()))
                throw std::logic_error("invalid geometry CSR offsets");
            for (Index c = 0; c < data_->cell_orders.size(); ++c) {
                const auto n = data_->geometry_offsets[c + 1] - data_->geometry_offsets[c];
                if ((data_->cell_orders[c] == 1 && n != 0) || (data_->cell_orders[c] == 2 && n <= cell_vertex_count(data_->types[c])) || (data_->cell_orders[c] != 1 && data_->cell_orders[c] != 2))
                    throw std::logic_error("invalid cell geometry order");
            }
        }
        for (Index node : data_->geometry_indices)
            if (node >= geometry_node_count())
                throw std::logic_error("invalid geometry node ID");
        for (const auto& label : data_->labels) {
            if (label.dimension > data_->dimension
                || (label.entity != invalid_index && label.entity >= data_->counts[label.dimension]))
                throw std::logic_error("invalid label entity");
            if (label.entity != invalid_index) {
                const auto key = entity_vertices(label.dimension, label.entity);
                if (!std::equal(key.begin(), key.end(), label.vertices.begin(), label.vertices.end()))
                    throw std::logic_error("label connectivity mismatch");
            }
        }
        for (Index from = 1; from <= data_->dimension; ++from) {
            for (Index to = 0; to < from; ++to) {
                if (!has_relation(from, to))
                    continue;
                auto relation = this->relation(from, to);
                if (relation.rows != entity_count(from))
                    throw std::logic_error("relation row count");
                const bool oriented = from == to + 1;
                const auto payload = relation.targets.size();
                if ((oriented && (relation.coefficients.size() != payload || relation.orientations.size() != payload))
                    || (!oriented && (!relation.coefficients.empty() || !relation.orientations.empty())))
                    throw std::logic_error("relation metadata size mismatch");
                if (relation.offsets.empty()) {
                    if (static_cast<std::size_t>(relation.rows) * relation.stride != relation.targets.size())
                        throw std::logic_error("invalid fixed-stride relation");
                }
                else {
                    if (relation.offsets.size() != static_cast<std::size_t>(relation.rows) + 1
                        || relation.offsets.front() != 0 || relation.offsets.back() != relation.targets.size()
                        || !std::is_sorted(relation.offsets.begin(), relation.offsets.end()))
                        throw std::logic_error("invalid CSR offsets");
                }
                for (Index i = 0; i < relation.rows; ++i)
                    for (auto id : relation[i].targets)
                        if (id >= entity_count(to))
                            throw std::logic_error("dangling incidence");
            }
        }
    }

    void Mesh::validate() const
    {
        validate_structure();
        // Each oriented codimension-two boundary must cancel.
        for (Index k = 2; k <= data_->dimension; ++k) {
            if (!has_relation(k, k - 1) || !has_relation(k - 1, k - 2))
                continue;
            auto high = relation(k, k - 1);
            auto low = relation(k - 1, k - 2);
            for (Index entity = 0; entity < entity_count(k); ++entity) {
                std::map<Index, int> sum;
                const auto outer = high[entity];
                for (std::size_t j = 0; j < outer.targets.size(); ++j) {
                    const auto inner = low[outer.targets[j]];
                    for (std::size_t i = 0; i < inner.targets.size(); ++i)
                        sum[inner.targets[i]] += outer.coefficients[j] * inner.coefficients[i];
                }
                for (const auto& [id, coefficient] : sum) {
                    (void)id;
                    if (coefficient != 0)
                        throw std::logic_error("boundary of boundary is nonzero");
                }
            }
        }
    }

} // namespace idmesh
