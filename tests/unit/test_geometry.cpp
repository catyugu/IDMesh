#include "test.hpp"

#include <array>
#include <cmath>
#include <idmesh/geometry.hpp>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
    using idmesh::Index;
    using Vec = std::array<double, 3>;
    void close(double a, double b, double tol, const char* message)
    {
        if (std::abs(a - b) > tol)
            throw std::runtime_error(message);
    }
    Vec affine(Vec p)
    {
        return {2. + 1.1 * p[0] + .3 * p[1] - .2 * p[2],
            -1. + .2 * p[0] + 1.3 * p[1] + .1 * p[2],
            3. + .15 * p[0] - .2 * p[1] + .9 * p[2]};
    }
    std::array<Vec, 3> columns {{{1.1, .2, .15}, {.3, 1.3, -.2}, {-.2, .1, .9}}};
    Vec interior(idmesh::CellType t)
    {
        switch (t) {
        case idmesh::CellType::segment:
            return {.23, 0, 0};
        case idmesh::CellType::triangle:
            return {.22, .31, 0};
        case idmesh::CellType::quadrilateral:
            return {.2, -.18, 0};
        case idmesh::CellType::tetrahedron:
            return {.2, .18, .24};
        case idmesh::CellType::hexahedron:
            return {.16, -.3, .13};
        case idmesh::CellType::prism:
            return {.2, .17, .14};
        case idmesh::CellType::pyramid:
            return {.13, -.24, .3};
        }
        throw std::runtime_error("unknown cell");
    }

    idmesh::Mesh make(idmesh::CellType type, Index count, bool curve = false)
    {
        const auto reference = idmesh::reference_geometry_nodes(type, count);
        idmesh::MeshBuilder b(idmesh::cell_dimension(type), 3);
        std::vector<Index> corner, extra;
        const auto n = idmesh::cell_vertex_count(type);
        for (Index i = 0; i < n; ++i)
            corner.push_back(b.add_vertex(affine(reference[i])));
        for (Index i = n; i < count; ++i) {
            auto p = affine(reference[i]);
            if (curve && i == n)
                p[1] += .24;
            extra.push_back(b.add_geometry_node(p));
        }
        const auto cell = extra.empty()
            ? b.add_cell(type, corner)
            : b.add_quadratic_cell(type, corner, extra);
        b.add_cell_label(cell, 5);
        return std::move(b).finalize();
    }

    void all_orders()
    {
        const std::array cases {
            std::pair {idmesh::CellType::segment, 2u},
            std::pair {idmesh::CellType::segment, 3u},
            std::pair {idmesh::CellType::triangle, 3u},
            std::pair {idmesh::CellType::triangle, 6u},
            std::pair {idmesh::CellType::quadrilateral, 4u},
            std::pair {idmesh::CellType::quadrilateral, 8u},
            std::pair {idmesh::CellType::quadrilateral, 9u},
            std::pair {idmesh::CellType::tetrahedron, 4u},
            std::pair {idmesh::CellType::tetrahedron, 10u},
            std::pair {idmesh::CellType::hexahedron, 8u},
            std::pair {idmesh::CellType::hexahedron, 20u},
            std::pair {idmesh::CellType::hexahedron, 27u},
            std::pair {idmesh::CellType::prism, 6u},
            std::pair {idmesh::CellType::prism, 15u},
            std::pair {idmesh::CellType::prism, 18u},
            std::pair {idmesh::CellType::pyramid, 5u},
            std::pair {idmesh::CellType::pyramid, 13u},
            std::pair {idmesh::CellType::pyramid, 14u}};
        for (const auto& [type, count] : cases) {
            auto mesh = make(type, count);
            const auto reference = idmesh::reference_geometry_nodes(type, count);
            const auto dim = mesh.dimension();
            for (Index i = 0; i < reference.size(); ++i) {
                if (type == idmesh::CellType::pyramid && i == 4)
                    continue;
                const auto physical = idmesh::evaluate_geometry(mesh, 0, reference[i]).point;
                const auto expected = affine(reference[i]);
                for (int k = 0; k < 3; ++k)
                    close(physical[k], expected[k], 1e-9,
                        "nodal Kronecker interpolation");
            }
            const auto r = interior(type);
            const auto evaluated = idmesh::evaluate_geometry(mesh, 0, r);
            const auto expected = affine(r);
            for (int i = 0; i < 3; ++i) {
                close(evaluated.point[i], expected[i], 1e-9, "affine value");
                for (Index j = 0; j < dim; ++j)
                    close(evaluated.jacobian[i][j], columns[j][i], 1e-9,
                        "affine Jacobian");
            }
            constexpr double h = 1e-6;
            for (Index j = 0; j < dim; ++j) {
                auto plus = r, minus = r;
                plus[j] += h;
                minus[j] -= h;
                const auto vp = idmesh::evaluate_geometry(mesh, 0, plus).point;
                const auto vm = idmesh::evaluate_geometry(mesh, 0, minus).point;
                for (int i = 0; i < 3; ++i)
                    close((vp[i] - vm[i]) / (2 * h), evaluated.jacobian[i][j], 5e-8,
                        "Jacobian finite-difference test");
            }
        }
    }
    void gmsh_node_order()
    {
        using CT = idmesh::CellType;
        struct Case {
            CT type;
            Index nodes;
            Index index;
            Vec reference;
        };
        // Independent spot checks: these indices differ from VTK ordering.
        const std::array cases {
            Case {CT::tetrahedron, 10, 8, {0, .5, .5}},
            Case {CT::tetrahedron, 10, 9, {.5, 0, .5}},
            Case {CT::hexahedron, 20, 9, {-1, 0, -1}},
            Case {CT::hexahedron, 20, 10, {-1, -1, 0}},
            Case {CT::hexahedron, 27, 20, {0, 0, -1}},
            Case {CT::hexahedron, 27, 21, {0, 1, 0}},
            Case {CT::hexahedron, 27, 23, {1, 0, 0}},
            Case {CT::prism, 15, 7, {0, .5, -1}},
            Case {CT::prism, 15, 8, {0, 0, 0}},
            Case {CT::prism, 18, 15, {.5, 0, 0}},
            Case {CT::pyramid, 13, 6, {-1, 0, 0}},
            Case {CT::pyramid, 13, 7, {-.5, -.5, .5}},
            Case {CT::pyramid, 14, 13, {0, 0, 0}}};
        for (const auto& test : cases) {
            const auto nodes = idmesh::reference_geometry_nodes(test.type, test.nodes);
            for (int i = 0; i < 3; ++i)
                close(nodes[test.index][i], test.reference[i], 1e-12,
                    "incorrect Gmsh reference-node ordering");
        }
    }
    void curvature()
    {
        const std::array cases {
            std::pair {idmesh::CellType::quadrilateral, 9u},
            std::pair {idmesh::CellType::tetrahedron, 10u},
            std::pair {idmesh::CellType::hexahedron, 20u},
            std::pair {idmesh::CellType::hexahedron, 27u},
            std::pair {idmesh::CellType::prism, 15u},
            std::pair {idmesh::CellType::prism, 18u},
            std::pair {idmesh::CellType::pyramid, 13u},
            std::pair {idmesh::CellType::pyramid, 14u}};
        constexpr double h = 1e-6;
        for (const auto& [type, nodes] : cases) {
            auto curved = make(type, nodes, true);
            auto linear = make(type, nodes);
            const auto p = interior(type);
            auto a = idmesh::evaluate_geometry(curved, 0, p);
            auto b = idmesh::evaluate_geometry(linear, 0, p);
            check(std::abs(a.point[1] - b.point[1]) > 1e-4,
                "quadratic perturbation did not curve mapping");
            for (Index j = 0; j < idmesh::cell_dimension(type); ++j) {
                auto p1 = p, p2 = p;
                p1[j] += h;
                p2[j] -= h;
                auto q1 = idmesh::evaluate_geometry(curved, 0, p1);
                auto q2 = idmesh::evaluate_geometry(curved, 0, p2);
                for (int i = 0; i < 3; ++i)
                    close((q1.point[i] - q2.point[i]) / (2 * h), a.jacobian[i][j], 5e-8,
                        "curved Jacobian finite difference");
            }
        }
    }
    void invalid()
    {
        auto mesh = make(idmesh::CellType::pyramid, 13);
        bool threw = false;
        try {
            (void)idmesh::evaluate_geometry(mesh, 0, {0, 0, 1});
        }
        catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, "pyramid singular apex should reject Jacobian");
        threw = false;
        try {
            (void)idmesh::reference_geometry_nodes(idmesh::CellType::hexahedron, 26);
        }
        catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, "unsupported node count accepted");
    }
}
int main()
{
    return idmesh::test::run({
        {"all reference cells", all_orders},
        {"Gmsh reference-node order", gmsh_node_order},
        {"curved geometry", curvature},
        {"invalid inputs", invalid},
    });
}
