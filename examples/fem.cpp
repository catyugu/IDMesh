#include <idmesh/mesh.hpp>

#include <array>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
    using idmesh::Index;
    using Matrix = std::vector<std::vector<int>>;
    Matrix boundary_matrix(const idmesh::Mesh& mesh, Index from)
    {
        const Index to = from - 1;
        Matrix result(mesh.entity_count(from),
            std::vector<int>(mesh.entity_count(to)));
        const auto incidence = mesh.relation(from, to);
        for (Index row = 0; row < mesh.entity_count(from); ++row) {
            auto indices = incidence.row(row);
            auto signs = incidence.row_signs(row);
            for (std::size_t j = 0; j < indices.size(); ++j)
                result[row][indices[j]] += signs[j];
        }
        return result;
    }
    void check_product_zero(const Matrix& a, const Matrix& b)
    {
        if (a.empty() || b.empty() || a.front().size() != b.size())
            throw std::runtime_error("invalid FEEC matrix dimensions");
        for (auto& row : a)
            for (std::size_t j = 0; j < b.front().size(); ++j) {
                int value = 0;
                for (std::size_t k = 0; k < b.size(); ++k)
                    value += row[k] * b[k][j];
                if (value != 0)
                    throw std::runtime_error("discrete exterior derivative does not square to zero");
            }
    }
}
int main()
{
    try {
        idmesh::MeshBuilder builder(3, 3);
        for (const auto& p : std::array<std::array<double, 3>, 10> {{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                 {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1},
                 {2, 0.5, 0}, {2, 0.5, 1}}})
            builder.add_vertex(p);
        const std::array<Index, 8> hex {0, 1, 2, 3, 4, 5, 6, 7};
        const std::array<Index, 6> prism {2, 1, 8, 6, 5, 9};
        builder.add_cell(idmesh::CellType::hexahedron, hex);
        builder.add_cell(idmesh::CellType::prism, prism);
        const auto mesh = std::move(builder).finalize(
            idmesh::TopologyOptions::full());
        mesh.validate();
        const Matrix G = boundary_matrix(mesh, 1);
        const Matrix C = boundary_matrix(mesh, 2);
        const Matrix D = boundary_matrix(mesh, 3);
        check_product_zero(C, G);
        check_product_zero(D, C);
        const auto support = mesh.relation(2, 3);
        bool shared = false;
        for (Index f = 0; f < mesh.entity_count(2); ++f)
            if (support.row(f).size() == 2)
                shared = true;
        if (!shared)
            throw std::runtime_error("missing mixed-cell interface");
        std::cout << "FEM (FEEC): mixed hex/prism, CG=0 and DC=0 OK\n";
    }
    catch (const std::exception& e) {
        std::cerr << "FEM (FEEC) example failure: " << e.what() << '\n';
        return 1;
    }
}
