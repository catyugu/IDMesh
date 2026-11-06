#include "idmesh/mesh.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace idmesh {
    namespace {

        void require(bool condition, const char* message)
        {
            if (!condition) {
                throw std::invalid_argument(message);
            }
        }

        // Standard intermediate entities have at most four corner vertices.
        // Keep construction keys inline so facet/edge canonicalization performs
        // no per-entity heap allocation.
        struct EntityKey {
            std::array<Index, 4> vertices {};
            std::uint8_t count = 0;

            std::span<const Index> view() const
            {
                return {vertices.data(), count};
            }
            std::size_t size() const { return count; }
            const Index* begin() const { return vertices.data(); }
            const Index* end() const { return vertices.data() + count; }
            Index operator[](std::size_t i) const { return vertices[i]; }
            bool operator==(const EntityKey&) const = default;
        };

        struct Canonical {
            EntityKey key;
            Orientation orientation;
        };

        EntityKey make_key(std::span<const Index> vertices)
        {
            require(!vertices.empty() && vertices.size() <= 4, "unsupported entity arity");
            EntityKey result;
            result.count = static_cast<std::uint8_t>(vertices.size());
            std::copy(vertices.begin(), vertices.end(), result.vertices.begin());
            return result;
        }

        Canonical canonicalize(std::span<const Index> local)
        {
            require(!local.empty() && local.size() <= 4, "unsupported entity arity");
            Canonical result;
            result.key.count = static_cast<std::uint8_t>(local.size());
            if (local.size() == 2) {
                result.key.vertices[0] = std::min(local[0], local[1]);
                result.key.vertices[1] = std::max(local[0], local[1]);
                const bool reversed = local[0] > local[1];
                result.orientation = {static_cast<std::uint16_t>(reversed ? 1 : 0), reversed};
                return result;
            }
            // Entity vertices are unique. The smallest vertex fixes the cyclic
            // starting point; only the two traversal directions need comparison.
            const auto first = static_cast<std::size_t>(
                std::min_element(local.begin(), local.end()) - local.begin());
            bool reversed = false;
            for (std::size_t i = 1; i < local.size(); ++i) {
                const Index forward = local[(first + i) % local.size()];
                const Index backward = local[(first + local.size() - i) % local.size()];
                if (forward != backward) {
                    reversed = backward < forward;
                    break;
                }
            }
            for (std::size_t i = 0; i < local.size(); ++i) {
                const auto position = reversed
                    ? (first + local.size() - i) % local.size()
                    : (first + i) % local.size();
                result.key.vertices[i] = local[position];
            }
            result.orientation = {static_cast<std::uint16_t>(first), reversed};
            return result;
        }

        using Loop = EntityKey;

        // Keys are compared in full: hash collisions never merge distinct entities.
        // Entity IDs remain assigned in the order of first appearance.
        struct LoopHash {
            std::size_t operator()(const Loop& key) const noexcept
            {
                std::size_t hash = sizeof(std::size_t) == 8
                    ? std::size_t {14695981039346656037ULL}
                    : std::size_t {2166136261U};
                for (Index vertex : key) {
                    hash ^= vertex;
                    hash *= sizeof(std::size_t) == 8
                        ? std::size_t {1099511628211ULL}
                        : std::size_t {16777619U};
                }
                hash ^= key.count;
                return hash;
            }
        };

        EntityKey boundary_loop(CellType type, Index facet, std::span<const Index> vertices)
        {
            EntityKey result;
            const auto local_vertices = reference_facet_vertices(type, facet);
            result.count = static_cast<std::uint8_t>(local_vertices.size());
            for (std::size_t i = 0; i < local_vertices.size(); ++i)
                result.vertices[i] = vertices[local_vertices[i]];
            return result;
        }

        EntityKey polygon_edge(std::span<const Index> vertices, std::size_t edge)
        {
            const std::array<Index, 2> endpoints {
                vertices[edge], vertices[(edge + 1) % vertices.size()]};
            return make_key(endpoints);
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

        // Build-time incidences stay contiguous instead of allocating one
        // vector per entity. Stable scattering preserves the original row order.
        struct PendingIncidence {
            Index row;
            Incidence entry;
        };

        using PendingRelation = std::vector<PendingIncidence>;

        RelationData allocate_relation(std::span<const Index> counts, bool oriented)
        {
            require(counts.size() < invalid_index, "too many relation rows");
            RelationData result;
            result.present = true;
            result.rows = static_cast<Index>(counts.size());
            std::size_t total = 0;
            for (Index count : counts) {
                total += count;
                require(total < invalid_index, "too many relation entries");
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
                require(pending.row < row_count, "invalid relation row");
                require(counts[pending.row] < invalid_index, "relation row overflow");
                ++counts[pending.row];
            }
            auto result = allocate_relation(counts, oriented);
            auto positions = write_positions(result);
            for (const auto& pending : entries) {
                const Index slot = positions[pending.row]++;
                result.indices[slot] = pending.entry.entity;
                if (oriented) {
                    result.signs[slot] = pending.entry.sign;
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
                require(target < row_count, "invalid transposed relation target");
                require(counts[target] < invalid_index, "transposed relation row overflow");
                ++counts[target];
            }
            const bool oriented = !forward.signs.empty();
            auto result = allocate_relation(counts, oriented);
            auto positions = write_positions(result);
            const auto view = forward.view();
            for (Index row = 0; row < forward.rows; ++row) {
                const auto targets = view.row(row);
                const auto signs = view.row_signs(row);
                const auto orientations = view.row_orientations(row);
                for (std::size_t i = 0; i < targets.size(); ++i) {
                    const Index slot = positions[targets[i]]++;
                    result.indices[slot] = row;
                    if (oriented) {
                        result.signs[slot] = signs[i];
                        result.orientations[slot] = orientations[i];
                    }
                }
            }
            return result;
        }

    } // namespace

    struct Mesh::Data {
        Index dimension = 0;
        Index geometric_dimension = 0;
        // Intermediate-entity construction scratch, released before publication.
        // Top-dimensional Cell connectivity enters already packed from MeshBuilder.
        std::array<std::vector<Loop>, 4> vertices;
        std::array<Index, 4> counts {};
        std::array<bool, 4> enabled {};
        std::array<std::vector<Index>, 4> vertex_offsets;
        std::array<std::vector<Index>, 4> vertex_indices;
        std::array<std::array<RelationData, 4>, 4> relations;
        std::vector<std::array<double, 3>> coordinates;
        std::vector<CellType> types;
        std::vector<int> tags;
        std::vector<int> facet_tags;
        std::vector<PhysicalLabel> physical_labels;
        std::vector<Index> unbound_label_vertices;
        std::map<std::pair<Index, int>, std::string> physical_names;
        std::vector<std::array<double, 3>> extra_geometry_nodes;
        std::vector<Index> geometry_offsets;
        std::vector<Index> geometry_indices;
        std::vector<std::uint8_t> cell_orders;
    };

    namespace {
        struct ReferenceFacet {
            std::array<std::uint8_t, 4> vertices;
            std::uint8_t size;
        };
        constexpr std::array<ReferenceFacet, 2> segment_facets {{{{{1, 0, 0, 0}}, 1}, {{{0, 0, 0, 0}}, 1}}};
        constexpr std::array<ReferenceFacet, 3> triangle_facets {{{{{0, 1, 0, 0}}, 2}, {{{1, 2, 0, 0}}, 2}, {{{2, 0, 0, 0}}, 2}}};
        constexpr std::array<ReferenceFacet, 4> quad_facets {{{{{0, 1, 0, 0}}, 2}, {{{1, 2, 0, 0}}, 2},
            {{{2, 3, 0, 0}}, 2}, {{{3, 0, 0, 0}}, 2}}};
        constexpr std::array<ReferenceFacet, 4> tetra_facets {{{{{1, 2, 3, 0}}, 3}, {{{0, 3, 2, 0}}, 3},
            {{{0, 1, 3, 0}}, 3}, {{{0, 2, 1, 0}}, 3}}};
        constexpr std::array<ReferenceFacet, 6> hex_facets {{{{{0, 3, 2, 1}}, 4}, {{{4, 5, 6, 7}}, 4},
            {{{0, 1, 5, 4}}, 4}, {{{1, 2, 6, 5}}, 4},
            {{{2, 3, 7, 6}}, 4}, {{{3, 0, 4, 7}}, 4}}};
        constexpr std::array<ReferenceFacet, 5> prism_facets {{{{{0, 2, 1, 0}}, 3}, {{{3, 4, 5, 0}}, 3},
            {{{0, 1, 4, 3}}, 4}, {{{1, 2, 5, 4}}, 4}, {{{2, 0, 3, 5}}, 4}}};
        constexpr std::array<ReferenceFacet, 5> pyramid_facets {{{{{0, 3, 2, 1}}, 4}, {{{0, 1, 4, 0}}, 3},
            {{{1, 2, 4, 0}}, 3}, {{{2, 3, 4, 0}}, 3}, {{{3, 0, 4, 0}}, 3}}};
        std::span<const ReferenceFacet> reference_facets(CellType type)
        {
            switch (type) {
            case CellType::segment:
                return segment_facets;
            case CellType::triangle:
                return triangle_facets;
            case CellType::quadrilateral:
                return quad_facets;
            case CellType::tetrahedron:
                return tetra_facets;
            case CellType::hexahedron:
                return hex_facets;
            case CellType::prism:
                return prism_facets;
            case CellType::pyramid:
                return pyramid_facets;
            }
            throw std::invalid_argument("unknown reference cell type");
        }
    } // namespace

    Index cell_facet_count(CellType type)
    {
        return static_cast<Index>(reference_facets(type).size());
    }

    std::span<const std::uint8_t> reference_facet_vertices(CellType type, Index facet)
    {
        const auto facets = reference_facets(type);
        if (facet >= facets.size())
            throw std::out_of_range("reference facet index");
        return {facets[facet].vertices.data(), facets[facet].size};
    }

    Index cell_dimension(CellType type)
    {
        switch (type) {
        case CellType::segment:
            return 1;
        case CellType::triangle:
        case CellType::quadrilateral:
            return 2;
        case CellType::tetrahedron:
        case CellType::hexahedron:
        case CellType::prism:
        case CellType::pyramid:
            return 3;
        }
        throw std::invalid_argument("unknown cell type");
    }

    Index cell_vertex_count(CellType type)
    {
        switch (type) {
        case CellType::segment:
            return 2;
        case CellType::triangle:
            return 3;
        case CellType::quadrilateral:
        case CellType::tetrahedron:
            return 4;
        case CellType::pyramid:
            return 5;
        case CellType::prism:
            return 6;
        case CellType::hexahedron:
            return 8;
        }
        throw std::invalid_argument("unknown cell type");
    }

    std::span<const Index> RelationView::row(Index index) const
    {
        if (index >= rows)
            throw std::out_of_range("relation row");
        auto begin = offsets.empty() ? static_cast<std::size_t>(index) * stride : offsets[index];
        auto end = offsets.empty() ? begin + stride : offsets[index + 1];
        return indices.subspan(begin, end - begin);
    }

    std::span<const std::int8_t> RelationView::row_signs(Index index) const
    {
        if (index >= rows)
            throw std::out_of_range("relation row");
        if (signs.empty())
            return {};
        auto begin = offsets.empty() ? static_cast<std::size_t>(index) * stride : offsets[index];
        auto end = offsets.empty() ? begin + stride : offsets[index + 1];
        return signs.subspan(begin, end - begin);
    }

    std::span<const Orientation> RelationView::row_orientations(Index index) const
    {
        if (index >= rows)
            throw std::out_of_range("relation row");
        if (orientations.empty())
            return {};
        auto begin = offsets.empty() ? static_cast<std::size_t>(index) * stride : offsets[index];
        auto end = offsets.empty() ? begin + stride : offsets[index + 1];
        return orientations.subspan(begin, end - begin);
    }

    Mesh::Mesh() = default;
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
    std::span<const int> Mesh::cell_tags() const
    {
        if (!data_)
            throw std::logic_error("empty mesh");
        return data_->tags;
    }

    std::span<const int> Mesh::facet_tags() const
    {
        if (!data_)
            throw std::logic_error("empty mesh");
        return data_->facet_tags;
    }

    int Mesh::facet_tag(Index facet) const
    {
        if (!data_)
            throw std::logic_error("empty mesh");
        if (facet >= data_->counts[data_->dimension - 1])
            throw std::out_of_range("facet index");
        return data_->facet_tags.empty() ? 0 : data_->facet_tags[facet];
    }

    std::span<const PhysicalLabel> Mesh::physical_labels() const
    {
        if (!data_)
            throw std::logic_error("empty mesh");
        return data_->physical_labels;
    }

    std::vector<int> Mesh::physical_tags(Index dim, Index entity) const
    {
        if (!data_)
            throw std::logic_error("empty mesh");
        if (dim > data_->dimension || entity >= data_->counts[dim])
            throw std::out_of_range("physical entity index");
        const auto key = std::pair {dim, entity};
        const auto& labels = data_->physical_labels;
        const auto first = std::lower_bound(labels.begin(), labels.end(), key,
            [](const PhysicalLabel& label, const auto& value) {
                return std::pair {label.dimension, label.entity} < value;
            });
        std::vector<int> result;
        for (auto it = first; it != labels.end() && std::pair {it->dimension, it->entity} == key; ++it)
            if (result.empty() || result.back() != it->tag)
                result.push_back(it->tag);
        return result;
    }

    std::string Mesh::physical_name(Index dim, int tag) const
    {
        if (!data_)
            throw std::logic_error("empty mesh");
        const auto found = data_->physical_names.find({dim, tag});
        return found == data_->physical_names.end() ? std::string {} : found->second;
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
        : dimension_(dimension)
        , geometric_dimension_(geometric_dimension)
    {
        require(dimension >= 1 && dimension <= 3, "topological dimension must be 1..3");
        require(geometric_dimension >= dimension && geometric_dimension <= 3, "invalid geometric dimension");
        cell_offsets_.push_back(0);
        cell_extra_offsets_.push_back(0);
    }

    Index MeshBuilder::add_vertex(std::array<double, 3> position)
    {
        for (double coordinate : position)
            require(std::isfinite(coordinate), "nonfinite coordinate");
        require(coordinates_.size() < invalid_index, "too many vertices");
        coordinates_.push_back(position);
        return static_cast<Index>(coordinates_.size() - 1);
    }

    Index MeshBuilder::add_cell(CellType type, std::span<const Index> vertices, int tag)
    {
        require(cell_dimension(type) == dimension_, "cell dimension mismatch");
        require(vertices.size() == cell_vertex_count(type), "incorrect cell arity");
        std::array<Index, 8> sorted {};
        for (std::size_t i = 0; i < vertices.size(); ++i) {
            require(vertices[i] < coordinates_.size(), "invalid vertex index");
            sorted[i] = vertices[i];
        }
        std::sort(sorted.begin(), sorted.begin() + vertices.size());
        require(std::adjacent_find(sorted.begin(), sorted.begin() + vertices.size())
                == sorted.begin() + vertices.size(),
            "duplicate cell vertex");
        require(types_.size() < invalid_index, "too many cells");
        require(cell_vertices_.size() + vertices.size() < invalid_index,
            "cell connectivity overflow");
        cell_vertices_.insert(cell_vertices_.end(), vertices.begin(), vertices.end());
        cell_offsets_.push_back(static_cast<Index>(cell_vertices_.size()));
        types_.push_back(type);
        tags_.push_back(tag);
        cell_orders_.push_back(1);
        cell_extra_offsets_.push_back(static_cast<Index>(cell_extra_nodes_.size()));
        return static_cast<Index>(types_.size() - 1);
    }

    Index MeshBuilder::add_geometry_node(std::array<double, 3> position)
    {
        for (double coordinate : position)
            require(std::isfinite(coordinate), "nonfinite geometry node coordinate");
        require(extra_geometry_nodes_.size() < invalid_index, "too many geometry nodes");
        extra_geometry_nodes_.push_back(position);
        return static_cast<Index>(extra_geometry_nodes_.size() - 1);
    }

    Index MeshBuilder::add_quadratic_cell(
        CellType type, std::span<const Index> vertices,
        std::span<const Index> extra_nodes, int tag)
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
        require(valid && vertices.size() == cell_vertex_count(type),
            "unsupported quadratic geometry node count");
        for (Index index : extra_nodes)
            require(index < extra_geometry_nodes_.size(), "invalid extra geometry node");
        require(cell_extra_nodes_.size() + extra_nodes.size() < invalid_index,
            "geometry connectivity overflow");
        const Index cell = add_cell(type, vertices, tag);
        cell_orders_[cell] = 2;
        cell_extra_nodes_.insert(
            cell_extra_nodes_.end(), extra_nodes.begin(), extra_nodes.end());
        cell_extra_offsets_.back() = static_cast<Index>(cell_extra_nodes_.size());
        return cell;
    }

    void MeshBuilder::add_cell_physical_tag(Index cell, int tag)
    {
        require(cell < types_.size(), "invalid cell for physical label");
        if (tag != 0)
            cell_physical_tags_.emplace_back(cell, tag);
    }

    void MeshBuilder::add_physical_tag(
        Index dimension, std::span<const Index> vertices, int tag)
    {
        require(dimension < dimension_, "use add_cell_physical_tag for cell labels");
        const bool arity = dimension == 0 ? vertices.size() == 1
            : dimension == 1              ? vertices.size() == 2
                                          : vertices.size() == 3 || vertices.size() == 4;
        require(arity, "invalid physical entity arity");
        std::array<Index, 4> sorted {};
        for (std::size_t i = 0; i < vertices.size(); ++i) {
            require(vertices[i] < coordinates_.size(), "invalid physical entity vertex");
            sorted[i] = vertices[i];
        }
        std::sort(sorted.begin(), sorted.begin() + vertices.size());
        require(std::adjacent_find(sorted.begin(), sorted.begin() + vertices.size())
                == sorted.begin() + vertices.size(),
            "duplicate physical entity vertex");
        if (tag == 0)
            return;
        require(physical_assignment_vertices_.size() + vertices.size() < invalid_index,
            "physical label connectivity overflow");
        const auto begin = static_cast<Index>(physical_assignment_vertices_.size());
        physical_assignment_vertices_.insert(
            physical_assignment_vertices_.end(), vertices.begin(), vertices.end());
        const auto finish = static_cast<Index>(physical_assignment_vertices_.size());
        physical_assignments_.push_back({dimension, begin, finish, tag});
    }

    void MeshBuilder::add_facet_tag(std::span<const Index> vertices, int tag)
    {
        add_physical_tag(dimension_ - 1, vertices, tag);
    }

    void MeshBuilder::set_physical_name(Index dimension, int tag, std::string name)
    {
        require(dimension <= dimension_, "physical name dimension out of range");
        physical_names_[{dimension, tag}] = std::move(name);
    }

    Mesh MeshBuilder::finalize(TopologyOptions options) &&
    {
        // The oriented incidence DAG must remain graded. A 3D edge layer
        // requires the intervening face layer; skipping it is invalid.
        require(dimension_ != 3 || !options.edges || options.faces || options.facets,
            "3D edges require faces: skipping an entity dimension is invalid");
        auto data = std::make_shared<Mesh::Data>();
        data->dimension = dimension_;
        data->geometric_dimension = geometric_dimension_;
        data->coordinates = std::move(coordinates_);
        data->counts[dimension_] = static_cast<Index>(types_.size());
        data->vertex_offsets[dimension_] = std::move(cell_offsets_);
        data->vertex_indices[dimension_] = std::move(cell_vertices_);
        data->types = std::move(types_);
        data->tags = std::move(tags_);
        data->cell_orders = std::move(cell_orders_);
        data->extra_geometry_nodes = std::move(extra_geometry_nodes_);
        data->physical_names = std::move(physical_names_);

        const auto construction_count = [&](Index dim) -> Index {
            return dim == dimension_ ? data->counts[dim]
                                     : static_cast<Index>(data->vertices[dim].size());
        };
        const auto construction_vertices = [&](Index dim, Index id) -> std::span<const Index> {
            if (dim == dimension_) {
                const auto& offsets = data->vertex_offsets[dim];
                const auto& indices = data->vertex_indices[dim];
                const auto begin = offsets[id];
                return {indices.data() + begin, offsets[id + 1] - begin};
            }
            return data->vertices[dim][id].view();
        };

        // No additional geometry-connectivity storage for purely linear meshes.
        // In mixed meshes, only order-two cells own full geometry-node sequences.
        if (std::any_of(data->cell_orders.begin(), data->cell_orders.end(),
                [](std::uint8_t order) { return order == 2; })) {
            data->geometry_offsets.reserve(static_cast<std::size_t>(data->counts[dimension_]) + 1);
            data->geometry_offsets.push_back(0);
            for (Index c = 0; c < data->counts[dimension_]; ++c) {
                if (data->cell_orders[c] == 2) {
                    const auto corners = construction_vertices(dimension_, c);
                    data->geometry_indices.insert(
                        data->geometry_indices.end(), corners.begin(), corners.end());
                    const Index begin = cell_extra_offsets_[c];
                    const Index finish = cell_extra_offsets_[c + 1];
                    for (Index i = begin; i < finish; ++i) {
                        const auto node = static_cast<std::size_t>(cell_extra_nodes_[i])
                            + data->coordinates.size();
                        require(node < invalid_index, "geometry node index overflow");
                        data->geometry_indices.push_back(static_cast<Index>(node));
                    }
                    require(data->geometry_indices.size() < invalid_index,
                        "geometry connectivity overflow");
                }
                data->geometry_offsets.push_back(
                    static_cast<Index>(data->geometry_indices.size()));
            }
        }
        std::vector<Index>().swap(cell_extra_offsets_);
        std::vector<Index>().swap(cell_extra_nodes_);

        // Vertex entities have identity connectivity. Avoid constructing one
        // separately allocated singleton vector per vertex during finalize.
        std::array<std::unordered_map<Loop, Index, LoopHash>, 4> lookup;
        std::array<std::array<PendingRelation, 4>, 4> incidence;
        const auto want = [&](Index dim) {
            return dim == 0 || dim == dimension_ || (dim == dimension_ - 1 && options.facets)
                || (dim == 1 && options.edges) || (dim == 2 && options.faces);
        };
        for (Index dim = 0; dim <= dimension_; ++dim)
            data->enabled[dim] = want(dim);
        if (dimension_ > 1 && want(dimension_ - 1))
            lookup[dimension_ - 1].reserve(data->types.size() * 2);
        auto register_entity = [&](Index dim, std::span<const Index> local) -> Incidence {
            auto canon = canonicalize(local);
            require(data->vertices[dim].size() < invalid_index, "too many topology entities");
            auto [it, inserted] = lookup[dim].try_emplace(canon.key,
                static_cast<Index>(data->vertices[dim].size()));
            if (inserted)
                data->vertices[dim].push_back(canon.key);
            return {it->second, static_cast<std::int8_t>(canon.orientation.sign()), canon.orientation};
        };
        auto attach = [&](Index from, Index to, Index id, Incidence entry) {
            incidence[from][to].push_back({id, entry});
        };

        // Cell-to-vertex is permanently available even with intermediate dimensions omitted.
        for (Index c = 0; c < data->counts[dimension_]; ++c) {
            const auto vertices = construction_vertices(dimension_, c);
            for (Index local = 0; local < vertices.size(); ++local) {
                auto sign = dimension_ == 1 ? (local == 0 ? -1 : 1) : 0;
                attach(dimension_, 0, c, {vertices[local], static_cast<std::int8_t>(sign), {}});
            }
            if (dimension_ == 1)
                continue;
            if (want(dimension_ - 1)) {
                for (Index facet = 0; facet < cell_facet_count(data->types[c]); ++facet) {
                    const auto face = boundary_loop(data->types[c], facet, vertices);
                    attach(dimension_, dimension_ - 1, c,
                        register_entity(dimension_ - 1, face.view()));
                }
            }
        }

        if (dimension_ == 3 && options.edges && want(2)) {
            for (Index f = 0; f < data->vertices[2].size(); ++f) {
                const auto vertices = data->vertices[2][f].view();
                for (std::size_t edge = 0; edge < vertices.size(); ++edge) {
                    const auto endpoints = polygon_edge(vertices, edge);
                    attach(2, 1, f, register_entity(1, endpoints.view()));
                }
            }
        }
        if (dimension_ == 2 && options.edges) {
            // Cell-to-edge was generated above; no extra construction is required.
        }
        if (dimension_ >= 2 && want(1)) {
            for (Index e = 0; e < data->vertices[1].size(); ++e) {
                const auto& vertices = data->vertices[1][e];
                require(vertices.size() == 2, "edge must have two vertices");
                attach(1, 0, e, {vertices[0], -1, {}});
                attach(1, 0, e, {vertices[1], 1, {}});
            }
        }

        // Direct facet-to-vertex traversal is needed by finite-volume kernels
        // even when intermediate edges have not been materialized.
        if (dimension_ == 3 && want(2)) {
            for (Index f = 0; f < data->vertices[2].size(); ++f) {
                for (auto v : data->vertices[2][f])
                    attach(2, 0, f, {v, 0, {}});
            }
        }

        // Resolve and deduplicate labels before replacing their temporary owned
        // vertex keys with borrowed spans into the compact finalized mesh.
        struct PendingPhysicalLabel {
            Index dimension;
            Index entity;
            std::vector<Index> vertices;
            int tag;
        };
        std::vector<PendingPhysicalLabel> labels;
        auto append_label = [&](Index dim, Index entity, std::span<const Index> vertices, int tag) {
            if (tag == 0)
                return;
            labels.push_back({dim, entity, std::vector<Index>(vertices.begin(), vertices.end()), tag});
        };
        for (Index cell = 0; cell < data->types.size(); ++cell)
            append_label(dimension_, cell, construction_vertices(dimension_, cell), data->tags[cell]);
        for (const auto& [cell, tag] : cell_physical_tags_)
            append_label(dimension_, cell, construction_vertices(dimension_, cell), tag);
        for (const auto& assignment : physical_assignments_) {
            const Index dim = assignment.dimension;
            const std::span<const Index> assigned {
                physical_assignment_vertices_.data() + assignment.begin,
                assignment.end - assignment.begin};
            const auto key = canonicalize(assigned);
            Index entity = invalid_index;
            if (dim == 0) {
                entity = assigned.front();
            }
            else if (want(dim)) {
                const auto found = lookup[dim].find(key.key);
                if (found != lookup[dim].end())
                    entity = found->second;
            }
            append_label(dim, entity, key.key.view(), assignment.tag);
            if (dim == dimension_ - 1 && entity != invalid_index) {
                if (data->facet_tags.empty())
                    data->facet_tags.resize(dim == 0
                            ? data->coordinates.size()
                            : data->vertices[dim].size());
                if (data->facet_tags[entity] == 0)
                    data->facet_tags[entity] = assignment.tag;
            }
        }
        std::vector<Index>().swap(physical_assignment_vertices_);
        std::vector<PhysicalAssignment>().swap(physical_assignments_);

        // Build-time sort/unique is O(L log L) rather than repeated O(L) label
        // lookups, where L can scale with the number of mesh cells.
        std::sort(labels.begin(), labels.end(), [](const PendingPhysicalLabel& a, const PendingPhysicalLabel& b) {
            return std::tie(a.dimension, a.entity, a.tag, a.vertices)
                < std::tie(b.dimension, b.entity, b.tag, b.vertices);
        });
        labels.erase(std::unique(labels.begin(), labels.end(),
                         [](const PendingPhysicalLabel& a, const PendingPhysicalLabel& b) {
                             return a.dimension == b.dimension && a.entity == b.entity
                                 && a.tag == b.tag && a.vertices == b.vertices;
                         }),
            labels.end());
        for (Index from = 1; from <= dimension_; ++from) {
            for (Index to = 0; to < from; ++to) {
                const auto& entries = incidence[from][to];
                if (entries.empty())
                    continue;
                // Only consecutive dimensions form oriented boundary operators.
                // Direct geometric connectivity omits redundant sign/permutation arrays.
                const bool oriented = from == to + 1;
                auto relation = compress(entries, construction_count(from), oriented);
                if (options.support && !(from == dimension_ && to == 0 && dimension_ > 1)) {
                    const auto targets = to == 0
                        ? data->coordinates.size()
                        : construction_count(to);
                    data->relations[to][from] = transpose(
                        relation, static_cast<Index>(targets));
                }
                data->relations[from][to] = std::move(relation);
            }
        }
        for (Index dim = 0; dim <= dimension_; ++dim) {
            auto& offsets = data->vertex_offsets[dim];
            auto& indices = data->vertex_indices[dim];
            if (dim == 0) {
                const auto count = static_cast<Index>(data->coordinates.size());
                data->counts[0] = count;
                offsets.resize(static_cast<std::size_t>(count) + 1);
                indices.resize(count);
                for (Index v = 0; v < count; ++v) {
                    offsets[v] = v;
                    indices[v] = v;
                }
                offsets[count] = count;
                continue;
            }
            if (dim == dimension_)
                continue;
            const auto& lists = data->vertices[dim];
            data->counts[dim] = static_cast<Index>(lists.size());
            offsets.reserve(lists.size() + 1);
            offsets.push_back(0);
            for (const auto& vertices : lists) {
                indices.insert(indices.end(), vertices.begin(), vertices.end());
                offsets.push_back(static_cast<Index>(indices.size()));
            }
            std::vector<Loop>().swap(data->vertices[dim]);
        }
        // Materialized entities borrow their already-packed connectivity.
        // Only labels on absent topology layers keep private vertex keys.
        std::size_t unbound_entries = 0;
        for (const auto& label : labels)
            if (label.entity == invalid_index)
                unbound_entries += label.vertices.size();
        data->unbound_label_vertices.reserve(unbound_entries);
        data->physical_labels.reserve(labels.size());
        for (const auto& label : labels) {
            std::span<const Index> vertices;
            if (label.entity != invalid_index) {
                const auto& offsets = data->vertex_offsets[label.dimension];
                const auto& indices = data->vertex_indices[label.dimension];
                const auto begin = offsets[label.entity];
                vertices = {indices.data() + begin, offsets[label.entity + 1] - begin};
            }
            else {
                const auto begin = data->unbound_label_vertices.size();
                data->unbound_label_vertices.insert(data->unbound_label_vertices.end(),
                    label.vertices.begin(), label.vertices.end());
                vertices = {data->unbound_label_vertices.data() + begin, label.vertices.size()};
            }
            data->physical_labels.push_back({label.dimension, label.entity, vertices, label.tag});
        }
        Mesh mesh(std::move(data));
        mesh.validate();
        return mesh;
    }

    void Mesh::validate() const
    {
        if (!data_)
            throw std::logic_error("empty mesh");
        if (!data_->facet_tags.empty() && data_->facet_tags.size() != data_->counts[data_->dimension - 1])
            throw std::logic_error("facet physical-tag size mismatch");
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
        for (const auto& label : data_->physical_labels) {
            if (label.dimension > data_->dimension
                || (label.entity != invalid_index && label.entity >= data_->counts[label.dimension]))
                throw std::logic_error("invalid physical label entity");
            if (label.entity != invalid_index) {
                const auto key = entity_vertices(label.dimension, label.entity);
                if (!std::equal(key.begin(), key.end(), label.vertices.begin(), label.vertices.end()))
                    throw std::logic_error("physical label connectivity mismatch");
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
                const auto payload = relation.indices.size();
                if ((oriented && (relation.signs.size() != payload || relation.orientations.size() != payload))
                    || (!oriented && (!relation.signs.empty() || !relation.orientations.empty())))
                    throw std::logic_error("relation metadata size mismatch");
                if (relation.offsets.empty()) {
                    if (static_cast<std::size_t>(relation.rows) * relation.stride != relation.indices.size())
                        throw std::logic_error("invalid fixed-stride relation");
                }
                else {
                    if (relation.offsets.size() != static_cast<std::size_t>(relation.rows) + 1
                        || relation.offsets.front() != 0 || relation.offsets.back() != relation.indices.size()
                        || !std::is_sorted(relation.offsets.begin(), relation.offsets.end()))
                        throw std::logic_error("invalid CSR offsets");
                }
                for (Index i = 0; i < relation.rows; ++i)
                    for (auto id : relation.row(i))
                        if (id >= entity_count(to))
                            throw std::logic_error("dangling incidence");
            }
        }
        // Each oriented codimension-two boundary must cancel.
        for (Index k = 2; k <= data_->dimension; ++k) {
            if (!has_relation(k, k - 1) || !has_relation(k - 1, k - 2))
                continue;
            auto high = relation(k, k - 1);
            auto low = relation(k - 1, k - 2);
            for (Index entity = 0; entity < entity_count(k); ++entity) {
                std::map<Index, int> sum;
                auto mids = high.row(entity);
                auto outer_signs = high.row_signs(entity);
                for (std::size_t j = 0; j < mids.size(); ++j) {
                    auto bottoms = low.row(mids[j]);
                    auto inner_signs = low.row_signs(mids[j]);
                    for (std::size_t i = 0; i < bottoms.size(); ++i)
                        sum[bottoms[i]] += outer_signs[j] * inner_signs[i];
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
