#pragma once

#include "idmesh/mesh.hpp"

namespace idmesh {

    // Coordinates are in the library reference cells. The first topological_dimension
    // columns of jacobian are d(x_physical)/d(x_reference), in physical-major order.
    struct GeometryEvaluation {
        std::array<double, 3> point {};
        std::array<std::array<double, 3>, 3> jacobian {};
    };

    // Gmsh node order, including the reference corners and high-order nodes.
    // Supported counts: line 2/3, tri 3/6, quad 4/8/9, tet 4/10,
    // hex 8/20/27, prism 6/15/18, pyramid 5/13/14.
    IDMESH_API std::vector<std::array<double, 3>> reference_geometry_nodes(
        CellType type, Index nodes);

    // Evaluate the isoparametric mapping and its *analytic* derivatives.
    // Pyramid apex has a singular collapsed coordinate chart; evaluation exactly
    // at z=1 is rejected. An out-of-range reference point is also rejected.
    IDMESH_API GeometryEvaluation evaluate_geometry(
        const Mesh& mesh, Index cell, std::array<double, 3> reference);

} // namespace idmesh
