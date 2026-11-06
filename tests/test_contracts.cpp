#include <algorithm>
#include <array>
#include <cstddef>
#include <idmesh/mesh.hpp>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
    using idmesh::Index;
    void check(bool condition, const char* what)
    {
        if (!condition)
            throw std::runtime_error(what);
    }
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
                check(cv.row_signs(0)[0] == -1 && cv.row_signs(0)[1] == 1,
                    "segment endpoints orientation");
                continue;
            }
            auto relation = mesh.relation(dim, dim - 1);
            auto facets = relation.row(0);
            for (std::size_t f = 0; f < facets.size(); ++f) {
                std::vector<Index> local;
                for (auto v : idmesh::reference_facet_vertices(type, static_cast<Index>(f)))
                    local.push_back(nodes[v]);
                auto o = relation.row_orientations(0)[f];
                verify_orientation(local, mesh.entity_vertices(dim - 1, facets[f]), o);
                check(relation.row_signs(0)[f] == o.sign(), "facet boundary sign");
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
            if (support.row(f).size() == 2) {
                check(common == idmesh::invalid_index, "more than one common face");
                common = f;
            }
        check(common != idmesh::invalid_index, "mixed common quad missing");
        int signs[2] {};
        for (Index c = 0; c < 2; ++c)
            for (std::size_t f = 0; f < downward.row(c).size(); ++f)
                if (downward.row(c)[f] == common)
                    signs[c] = downward.row_signs(c)[f];
        check(signs[0] != 0 && signs[0] == -signs[1], "mixed interface orientation");
    }
    void layer_masks()
    {
        for (auto type : {idmesh::CellType::tetrahedron, idmesh::CellType::hexahedron,
                 idmesh::CellType::prism, idmesh::CellType::pyramid}) {
            auto build = [type](idmesh::TopologyOptions options) {
                idmesh::MeshBuilder b(3, 3);
                std::vector<Index> nodes;
                for (Index i = 0; i < idmesh::cell_vertex_count(type); ++i)
                    nodes.push_back(b.add_vertex({double(i), 0, 0}));
                b.add_cell(type, nodes);
                return std::move(b).finalize(options);
            };
            auto invalid = idmesh::TopologyOptions::cells_and_vertices();
            invalid.edges = true;
            bool rejected = false;
            try {
                (void)build(invalid);
            }
            catch (const std::invalid_argument&) {
                rejected = true;
            }
            check(rejected, "skipped face layer accepted");
            auto full = build(idmesh::TopologyOptions::full());
            check(full.has_relation(3, 2) && full.has_relation(2, 1) && full.has_relation(1, 0),
                "missing complete chain");
            check(!full.has_relation(3, 1), "unexpected skipped relation");
            for (const auto [from, to] : std::array<std::pair<Index, Index>, 2> {{{3, 0}, {2, 0}}}) {
                const auto direct = full.relation(from, to);
                check(!direct.indices.empty(), "missing direct geometric connectivity");
                check(direct.signs.empty() && direct.orientations.empty(),
                    "redundant metadata in geometric connectivity");
                check(direct.row_signs(0).empty() && direct.row_orientations(0).empty(),
                    "signless relation row should have no metadata");
            }
            const auto facet_support = full.relation(0, 2);
            check(facet_support.signs.empty() && facet_support.orientations.empty(),
                "redundant metadata in transposed geometric connectivity");
            const auto boundary = full.relation(3, 2);
            check(boundary.signs.size() == boundary.indices.size()
                    && boundary.orientations.size() == boundary.indices.size(),
                "oriented boundary metadata missing");
            auto fv = build(idmesh::TopologyOptions::finite_volume());
            check(fv.has_entities(2) && !fv.has_entities(1), "finite volume entity mask");
            check(fv.has_relation(3, 2) && fv.has_relation(2, 0), "finite volume connections");
            fv.validate();
        }
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
                            mesh.entity_vertices(dimension - 1, relation.row(0)[facet]),
                            relation.row_orientations(0)[facet]);
                    }
                }
                ++permutations;
            } while (permutations < 64 && std::next_permutation(order.begin(), order.end()));
        }
    }

    void default_mesh()
    {
        idmesh::Mesh mesh;
        check(!mesh.has_entities(0), "invalid default mesh");
        bool rejected = false;
        try {
            (void)mesh.coordinates();
        }
        catch (const std::logic_error&) {
            rejected = true;
        }
        check(rejected, "default mesh must reject coordinate access");
    }
}
int main()
{
    try {
        all_reference_cells();
        mixed_hex_prism();
        layer_masks();
        permutation_invariants();
        default_mesh();
        std::cout << "IDMesh contract tests passed\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Contract test failure: " << e.what() << '\n';
        return 1;
    }
}
