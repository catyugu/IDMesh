#include <array>
#include <idmesh/mesh.hpp>
#include <iostream>
#include <stdexcept>
#include <utility>

int main()
{
    try {
        idmesh::MeshBuilder builder(3, 3);
        for (const auto& xyz : std::array<std::array<double, 3>, 5> {{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, -1}}})
            builder.add_vertex(xyz);
        const std::array<idmesh::Index, 4> a {0, 1, 2, 3}, b {0, 2, 1, 4};
        builder.add_cell(idmesh::CellType::tetrahedron, a);
        builder.add_cell(idmesh::CellType::tetrahedron, b);
        const auto mesh = std::move(builder).finalize();
        const auto faces = mesh.relation(3, 2);
        const auto edges = mesh.relation(2, 1);
        const auto support = mesh.relation(2, 3);
        bool shared = false;
        for (idmesh::Index f = 0; f < mesh.entity_count(2); ++f) {
            if (support[f].targets.size() != 2)
                continue;
            shared = true;
            for (idmesh::Index cell = 0; cell < 2; ++cell) {
                const auto row = faces[cell];
                for (std::size_t j = 0; j < row.targets.size(); ++j)
                    if (row.targets[j] == f) {
                        const auto o = row.orientations[j];
                        std::cout << "FACE " << cell << ' ' << f << ' ' << o.rotation
                                  << ' ' << int(o.reversed) << ' '
                                  << int(row.coefficients[j]) << '\n';
                    }
            }
        }
        if (!shared)
            throw std::runtime_error("no shared tetra face");
        for (idmesh::Index f = 0; f < mesh.entity_count(2); ++f) {
            const auto row = edges[f];
            for (std::size_t j = 0; j < row.targets.size(); ++j)
                std::cout << "EDGE " << f << ' ' << row.targets[j] << ' '
                          << int(row.orientations[j].reversed) << '\n';
        }
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
