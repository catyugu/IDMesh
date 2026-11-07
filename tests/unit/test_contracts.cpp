#include "test.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <idmesh/mesh.hpp>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

static_assert(!std::is_default_constructible_v<idmesh::Mesh>);
static_assert(std::is_copy_constructible_v<idmesh::Mesh>);
static_assert(!std::is_copy_constructible_v<idmesh::MeshBuilder>);
static_assert(std::is_move_constructible_v<idmesh::MeshBuilder>);

namespace {
    using idmesh::Index;
    void verify_orientation(std::span<const Index> local, std::span<const Index> canonical,
        idmesh::Orientation orientation)
    {
        check(local.size() == canonical.size(), "different local and canonical arities");
        const auto n = static_cast<std::ptrdiff_t>(local.size());
        check(orientation.rotation < n, "orientation rotation out of range");
        for (std::ptrdiff_t i = 0; i < n; ++i) {
            auto pos = (static_cast<std::ptrdiff_t>(orientation.rotation)
                           + (orientation.reversed ? -i : i) + n)
                % n;
            check(local[static_cast<std::size_t>(pos)] == canonical[static_cast<std::size_t>(i)],
                "orientation permutation mismatch");
        }
    }
    void all_reference_cells()
    {
        constexpr std::array types {
            idmesh::CellType::segment, idmesh::CellType::triangle,
            idmesh::CellType::quadrilateral, idmesh::CellType::tetrahedron,
            idmesh::CellType::hexahedron, idmesh::CellType::prism,
            idmesh::CellType::pyramid};
        constexpr std::array counts {2u, 3u, 4u, 4u, 6u, 5u, 5u};
        for (std::size_t k = 0; k < types.size(); ++k) {
            auto type = types[k];
            const Index dim = idmesh::cell_dimension(type);
            check(idmesh::cell_facet_count(type) == counts[k], "reference facet count");
            idmesh::MeshBuilder builder(dim, 3);
            std::vector<Index> nodes;
            for (Index i = 0; i < idmesh::cell_vertex_count(type); ++i)
                nodes.push_back(builder.add_vertex({double(i), double(i % 3), double(i % 2)}));
            builder.add_cell(type, nodes);
            auto mesh = std::move(builder).finalize();
            mesh.validate();
            if (dim == 1) {
                auto cv = mesh.relation(1, 0);
                check(cv[0].coefficients[0] == -1 && cv[0].coefficients[1] == 1,
                    "segment endpoints orientation");
                continue;
            }
            auto relation = mesh.relation(dim, dim - 1);
            const auto facets = relation[0];
            for (std::size_t f = 0; f < facets.targets.size(); ++f) {
                std::vector<Index> local;
                for (auto v : idmesh::reference_facet_vertices(type, static_cast<Index>(f)))
                    local.push_back(nodes[v]);
                const auto o = facets.orientations[f];
                verify_orientation(local, mesh.entity_vertices(dim - 1, facets.targets[f]), o);
                check(facets.coefficients[f] == o.sign(), "facet boundary sign");
            }
            bool invalid = false;
            try {
                (void)idmesh::reference_facet_vertices(type, idmesh::cell_facet_count(type));
            }
            catch (const std::out_of_range&) {
                invalid = true;
            }
            check(invalid, "invalid reference facet query accepted");
        }
    }
    void mixed_hex_prism()
    {
        idmesh::MeshBuilder b(3, 3);
        constexpr std::array<std::array<double, 3>, 10> coords {{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
            {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1},
            {2, 0.5, 0}, {2, 0.5, 1}}};
        for (auto p : coords)
            b.add_vertex(p);
        const std::array<Index, 8> hex {0, 1, 2, 3, 4, 5, 6, 7};
        const std::array<Index, 6> prism {2, 1, 8, 6, 5, 9};
        b.add_cell(idmesh::CellType::hexahedron, hex);
        b.add_cell(idmesh::CellType::prism, prism);
        const auto mesh = std::move(b).finalize();
        auto support = mesh.relation(2, 3);
        auto downward = mesh.relation(3, 2);
        Index common = idmesh::invalid_index;
        for (Index f = 0; f < mesh.entity_count(2); ++f)
            if (support[f].targets.size() == 2) {
                check(common == idmesh::invalid_index, "more than one common face");
                common = f;
            }
        check(common != idmesh::invalid_index, "mixed common quad missing");
        int signs[2] {};
        for (Index c = 0; c < 2; ++c)
            for (std::size_t f = 0; f < downward[c].targets.size(); ++f)
                if (downward[c].targets[f] == common)
                    signs[c] = downward[c].coefficients[f];
        check(signs[0] != 0 && signs[0] == -signs[1], "mixed interface orientation");
    }
    void layer_masks()
    {
        for (auto type : {idmesh::CellType::tetrahedron, idmesh::CellType::hexahedron,
                 idmesh::CellType::prism, idmesh::CellType::pyramid}) {
            auto build = [type](idmesh::BuildOptions options) {
                idmesh::MeshBuilder b(3, 3);
                std::vector<Index> nodes;
                for (Index i = 0; i < idmesh::cell_vertex_count(type); ++i)
                    nodes.push_back(b.add_vertex({double(i), 0, 0}));
                b.add_cell(type, nodes);
                return std::move(b).finalize(options);
            };
            auto full = build(idmesh::BuildOptions {});
            check(full.has_relation(3, 2) && full.has_relation(2, 1) && full.has_relation(1, 0),
                "missing complete chain");
            check(!full.has_relation(3, 1), "unexpected skipped relation");
            for (const auto [from, to] : std::array<std::pair<Index, Index>, 2> {{{3, 0}, {2, 0}}}) {
                const auto direct = full.relation(from, to);
                check(!direct.targets.empty(), "missing direct geometric connectivity");
                check(direct.coefficients.empty() && direct.orientations.empty(),
                    "redundant metadata in geometric connectivity");
                check(!direct[0].oriented(),
                    "signless relation row should have no metadata");
            }
            check(!full.has_relation(0, 2) && !full.has_relation(0, 3),
                "geometric shortcut must stay downward-only");
            const auto boundary = full.relation(3, 2);
            check(boundary.coefficients.size() == boundary.targets.size()
                    && boundary.orientations.size() == boundary.targets.size(),
                "oriented boundary metadata missing");
            auto fv = build(idmesh::BuildOptions {.topology = idmesh::TopologyLayout::cell_facets});
            check(fv.has_entities(2) && !fv.has_entities(1), "finite volume entity mask");
            check(fv.has_relation(3, 2) && fv.has_relation(2, 0), "finite volume connections");
            fv.validate();
        }
    }

    void topology_rehash_stability()
    {
        // Enough cells to force several entity-map rehashes during construction.
        constexpr Index side = 8, row = side + 1;
        const auto vertex = [](Index x, Index y, Index z) {
            return z * row * row + y * row + x;
        };
        idmesh::MeshBuilder builder(3, 3);
        for (Index z = 0; z <= side; ++z)
            for (Index y = 0; y <= side; ++y)
                for (Index x = 0; x <= side; ++x)
                    builder.add_vertex({double(x), double(y), double(z)});

        for (Index z = 0; z < side; ++z)
            for (Index y = 0; y < side; ++y)
                for (Index x = 0; x < side; ++x) {
                    const std::array<Index, 8> corners {
                        vertex(x, y, z), vertex(x + 1, y, z),
                        vertex(x + 1, y + 1, z), vertex(x, y + 1, z),
                        vertex(x, y, z + 1), vertex(x + 1, y, z + 1),
                        vertex(x + 1, y + 1, z + 1), vertex(x, y + 1, z + 1)};
                    builder.add_cell(idmesh::CellType::hexahedron, corners);
                }

        const auto mesh = std::move(builder).finalize();
        mesh.validate();
        check(mesh.entity_count(3) == side * side * side, "cell count after rehash");
        check(mesh.entity_count(2) == 3 * side * side * (side + 1),
            "face identity after rehash");
        check(mesh.entity_count(1) == 3 * side * (side + 1) * (side + 1),
            "edge identity after rehash");
        const auto support = mesh.relation(2, 3);
        Index boundary = 0;
        for (Index face = 0; face < mesh.entity_count(2); ++face) {
            const auto count = support[face].targets.size();
            check(count == 1 || count == 2, "invalid face support after rehash");
            boundary += (count == 1);
        }
        check(boundary == 6 * side * side, "boundary face count after rehash");
    }

    void error_contracts()
    {
        expect_error<std::invalid_argument>(
            [] { idmesh::MeshBuilder invalid(0, 0); }, "invalid topological dimension");
        expect_error<std::invalid_argument>(
            [] { idmesh::MeshBuilder invalid(2, 1); }, "invalid geometric dimension");
        idmesh::MeshBuilder builder(2, 2);
        expect_error<std::invalid_argument>(
            [&] { builder.add_vertex({std::numeric_limits<double>::infinity(), 0, 0}); },
            "nonfinite vertex coordinate");
        builder.add_vertex({0, 0, 0});
        builder.add_vertex({1, 0, 0});
        builder.add_vertex({0, 1, 0});
        const std::array<Index, 3> bad {0, 1, 3};
        expect_error<std::invalid_argument>(
            [&] { builder.add_cell(idmesh::CellType::triangle, bad); },
            "invalid cell connectivity");
        const std::array<Index, 3> valid {0, 1, 2};
        builder.add_cell(idmesh::CellType::triangle, valid);
        auto mesh = std::move(builder).finalize(
            idmesh::BuildOptions {idmesh::TopologyLayout::cell_vertices,
                idmesh::RelationDirection::downward});
        expect_error<std::out_of_range>(
            [&] { (void)mesh.entity_vertices(2, 1); }, "out-of-range entity");
        expect_error<std::out_of_range>(
            [&] { (void)mesh.entity_count(3); }, "out-of-range dimension");
        expect_error<std::invalid_argument>(
            [&] { (void)mesh.relation(2, 1); }, "unmaterialized relation");
        mesh.validate();
    }

    void permutation_invariants()
    {
        constexpr std::array types {
            idmesh::CellType::segment, idmesh::CellType::triangle,
            idmesh::CellType::quadrilateral, idmesh::CellType::tetrahedron,
            idmesh::CellType::hexahedron, idmesh::CellType::prism,
            idmesh::CellType::pyramid};
        for (auto type : types) {
            const Index dimension = idmesh::cell_dimension(type);
            const Index n = idmesh::cell_vertex_count(type);
            std::vector<Index> order(n);
            std::iota(order.begin(), order.end(), Index {0});
            unsigned permutations = 0;
            do {
                idmesh::MeshBuilder builder(dimension, 3);
                for (Index i = 0; i < n; ++i)
                    builder.add_vertex({double(i), 0, 0});
                builder.add_cell(type, order);
                auto mesh = std::move(builder).finalize();
                mesh.validate();
                if (dimension > 1) {
                    const auto relation = mesh.relation(dimension, dimension - 1);
                    for (Index facet = 0; facet < idmesh::cell_facet_count(type); ++facet) {
                        std::vector<Index> local;
                        for (auto vertex : idmesh::reference_facet_vertices(type, facet))
                            local.push_back(order[vertex]);
                        verify_orientation(local,
                            mesh.entity_vertices(dimension - 1, relation[0].targets[facet]),
                            relation[0].orientations[facet]);
                    }
                }
                ++permutations;
            } while (permutations < 64 && std::next_permutation(order.begin(), order.end()));
        }
    }

}
int main()
{
    return idmesh::test::run({
        {"reference cells", all_reference_cells},
        {"mixed hex/prism", mixed_hex_prism},
        {"layer masks", layer_masks},
        {"topology rehash stability", topology_rehash_stability},
        {"permutation invariants", permutation_invariants},
        {"error contracts", error_contracts},
    });
}
