#pragma once

#include "idmesh/mesh.hpp"

#include <array>
#include <span>
#include <stdexcept>

namespace idmesh::detail {

    struct ReferenceFacet {
        std::array<std::uint8_t, 4> vertices {};
        std::uint8_t size = 0;
    };

    struct ReferenceCell {
        Index dimension = 0;
        Index vertices = 0;
        std::span<const ReferenceFacet> facets;
    };

    inline ReferenceCell reference_cell(CellType type)
    {
        static constexpr std::array<ReferenceFacet, 2> segment_facets {{
            {{{1, 0, 0, 0}}, 1},
            {{{0, 0, 0, 0}}, 1},
        }};
        static constexpr std::array<ReferenceFacet, 3> triangle_facets {{
            {{{0, 1, 0, 0}}, 2},
            {{{1, 2, 0, 0}}, 2},
            {{{2, 0, 0, 0}}, 2},
        }};
        static constexpr std::array<ReferenceFacet, 4> quadrilateral_facets {{
            {{{0, 1, 0, 0}}, 2},
            {{{1, 2, 0, 0}}, 2},
            {{{2, 3, 0, 0}}, 2},
            {{{3, 0, 0, 0}}, 2},
        }};
        static constexpr std::array<ReferenceFacet, 4> tetrahedron_facets {{
            {{{1, 2, 3, 0}}, 3},
            {{{0, 3, 2, 0}}, 3},
            {{{0, 1, 3, 0}}, 3},
            {{{0, 2, 1, 0}}, 3},
        }};
        static constexpr std::array<ReferenceFacet, 6> hexahedron_facets {{
            {{{0, 3, 2, 1}}, 4},
            {{{4, 5, 6, 7}}, 4},
            {{{0, 1, 5, 4}}, 4},
            {{{1, 2, 6, 5}}, 4},
            {{{2, 3, 7, 6}}, 4},
            {{{3, 0, 4, 7}}, 4},
        }};
        static constexpr std::array<ReferenceFacet, 5> prism_facets {{
            {{{0, 2, 1, 0}}, 3},
            {{{3, 4, 5, 0}}, 3},
            {{{0, 1, 4, 3}}, 4},
            {{{1, 2, 5, 4}}, 4},
            {{{2, 0, 3, 5}}, 4},
        }};
        static constexpr std::array<ReferenceFacet, 5> pyramid_facets {{
            {{{0, 3, 2, 1}}, 4},
            {{{0, 1, 4, 0}}, 3},
            {{{1, 2, 4, 0}}, 3},
            {{{2, 3, 4, 0}}, 3},
            {{{3, 0, 4, 0}}, 3},
        }};

        switch (type) {
        case CellType::segment:
            return {1, 2, segment_facets};
        case CellType::triangle:
            return {2, 3, triangle_facets};
        case CellType::quadrilateral:
            return {2, 4, quadrilateral_facets};
        case CellType::tetrahedron:
            return {3, 4, tetrahedron_facets};
        case CellType::hexahedron:
            return {3, 8, hexahedron_facets};
        case CellType::prism:
            return {3, 6, prism_facets};
        case CellType::pyramid:
            return {3, 5, pyramid_facets};
        }
        throw std::invalid_argument("unknown reference cell type");
    }

} // namespace idmesh::detail
