#pragma once

#include "idmesh/mesh.hpp"
#include <array>
#include <span>
#include <stdexcept>

namespace idmesh::detail {

    // vtk_order[i] = gmsh_order[vtk_from_gmsh[i]].
    // Every conversion uses this table in both directions.
    inline std::span<const std::uint8_t> vtk_from_gmsh(CellType type, Index nodes)
    {
        static constexpr std::array<std::uint8_t, 10> tet10 {
            0, 1, 2, 3, 4, 5, 6, 7, 9, 8};
        static constexpr std::array<std::uint8_t, 20> hex20 {
            0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 9, 16, 18, 19, 17, 10, 12, 14, 15};
        static constexpr std::array<std::uint8_t, 27> hex27 {
            0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 9, 16, 18, 19, 17, 10, 12, 14, 15,
            22, 23, 21, 24, 20, 25, 26};
        static constexpr std::array<std::uint8_t, 15> wedge15 {
            0, 1, 2, 3, 4, 5, 6, 9, 7, 12, 14, 13, 8, 10, 11};
        static constexpr std::array<std::uint8_t, 18> wedge18 {
            0, 1, 2, 3, 4, 5, 6, 9, 7, 12, 14, 13, 8, 10, 11, 15, 16, 17};
        static constexpr std::array<std::uint8_t, 13> pyr13 {
            0, 1, 2, 3, 4, 5, 8, 10, 6, 7, 9, 11, 12};
        static constexpr std::array<std::uint8_t, 14> pyr14 {
            0, 1, 2, 3, 4, 5, 8, 10, 6, 7, 9, 11, 12, 13};
        if (type == CellType::tetrahedron && nodes == 10)
            return tet10;
        if (type == CellType::hexahedron && nodes == 20)
            return hex20;
        if (type == CellType::hexahedron && nodes == 27)
            return hex27;
        if (type == CellType::prism && nodes == 15)
            return wedge15;
        if (type == CellType::prism && nodes == 18)
            return wedge18;
        if (type == CellType::pyramid && nodes == 13)
            return pyr13;
        if (type == CellType::pyramid && nodes == 14)
            return pyr14;
        return {};
    }

} // namespace idmesh::detail
