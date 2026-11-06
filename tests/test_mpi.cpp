#include <idmesh/mpi.hpp>
#include <mpi.h>

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
    using idmesh::Index;
    void require(bool condition, const char* message)
    {
        if (!condition)
            throw std::runtime_error(message);
    }
    idmesh::Mesh source_mesh()
    {
        idmesh::MeshBuilder builder(2, 2);
        for (const auto& p : std::array<std::array<double, 3>, 6> {{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {2, 0, 0}, {2, 1, 0}}})
            builder.add_vertex(p);
        const std::array<Index, 4> a {0, 1, 2, 3};
        const std::array<Index, 3> b {1, 4, 2}, c {4, 5, 2};
        const auto first = builder.add_cell(idmesh::CellType::quadrilateral, a);
        const auto second = builder.add_cell(idmesh::CellType::triangle, b);
        const auto third = builder.add_cell(idmesh::CellType::triangle, c);
        builder.add_cell_label(first, 41);
        builder.add_cell_label(second, 42);
        builder.add_cell_label(third, 43);
        const std::array<Index, 2> boundary {0, 1};
        builder.add_label(1, boundary, 71);
        return std::move(builder).finalize(idmesh::BuildOptions {.topology = idmesh::TopologyLayout::cell_facets});
    }
    void run(const idmesh::mpi::Context& context,
        const idmesh::Mesh& source, Index depth)
    {
        const auto rank = context.rank(), size = context.size();
        const auto part = idmesh::mpi::partition_replicated(context, source, depth);
        const auto d = part.mesh().dimension();
        const Index n = part.mesh().entity_count(d);
        require(n == part.global_ids(d).size()
                && n == part.owner_ranks(d).size(),
            "local cell distributed metadata mismatch");
        require(part.owned_cell_count() <= n, "owned count exceeds local cells");
        for (Index i = 0; i < n; ++i) {
            const auto global_id = part.global_ids(d)[i];
            const int owner = static_cast<int>(global_id * size / source.entity_count(d));
            require(part.owner_ranks(d)[i] == owner, "inconsistent cell ownership");
            require((i < part.owned_cell_count()) == (owner == rank),
                "owned/ghost cell ordering incorrect");
        }
        std::vector<double> values(n * 3, -1);
        for (Index i = 0; i < part.owned_cell_count(); ++i)
            for (Index component = 0; component < 3; ++component)
                values[i * 3 + component] = 100 * part.global_ids(d)[i] + component;
        part.exchange_plan().exchange(values, 3);
        for (Index i = 0; i < n; ++i)
            for (Index component = 0; component < 3; ++component)
                require(values[i * 3 + component] == 100 * part.global_ids(d)[i] + component,
                    "halo exchange corrupted ghost field");
        std::vector<double> scalar(n, -1);
        for (Index i = 0; i < part.owned_cell_count(); ++i)
            scalar[i] = part.global_ids(d)[i] + 10;
        part.exchange_plan().exchange(scalar);
        for (Index i = 0; i < n; ++i)
            require(scalar[i] == part.global_ids(d)[i] + 10,
                "second exchange plan invocation failed");

        for (Index k = 0; k < d; ++k) {
            require(part.global_ids(k).size() == part.mesh().entity_count(k),
                "shared entity global ID count");
            require(part.owner_ranks(k).size() == part.mesh().entity_count(k),
                "shared entity ownership count");
            for (Index e = 0; e < part.mesh().entity_count(k); ++e) {
                require(part.global_ids(k)[e] < source.entity_count(k),
                    "invalid shared entity global ID");
                require(part.owner_ranks(k)[e] >= 0 && part.owner_ranks(k)[e] < size,
                    "invalid subentity owner rank");
            }
        }
        int local_owned = static_cast<int>(part.owned_cell_count()), sum = 0;
        MPI_Allreduce(&local_owned, &sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        require(sum == static_cast<int>(source.entity_count(d)),
            "some cells lost or multiply owned");

        // Cross-rank orientation identity is stronger than matching facet IDs:
        // each local entity's canonical global vertex cycle must match the
        // source entity, even when ranks discovered cells in different orders.
        for (Index dimension = 1; dimension < d; ++dimension)
            for (Index entity = 0; entity < part.mesh().entity_count(dimension); ++entity) {
                const auto source_id = part.global_ids(dimension)[entity];
                const auto local_vertices = part.mesh().entity_vertices(dimension, entity);
                const auto global_vertices = source.entity_vertices(dimension, source_id);
                require(local_vertices.size() == global_vertices.size(),
                    "shared entity vertex arity mismatch");
                for (std::size_t j = 0; j < local_vertices.size(); ++j)
                    require(part.global_ids(0)[local_vertices[j]] == global_vertices[j],
                        "cross-rank canonical orientation mismatch");
            }
        const auto local_cf = part.mesh().relation(d, d - 1);
        const auto global_cf = source.relation(d, d - 1);
        for (Index local_cell = 0; local_cell < n; ++local_cell) {
            const Index global_cell = static_cast<Index>(part.global_ids(d)[local_cell]);
            const auto local_faces = local_cf[local_cell];
            const auto source_faces = global_cf[global_cell];
            for (std::size_t j = 0; j < local_faces.targets.size(); ++j) {
                const auto global_face = part.global_ids(d - 1)[local_faces.targets[j]];
                bool matched = false;
                for (std::size_t k = 0; k < source_faces.targets.size(); ++k) {
                    if (source_faces.targets[k] != global_face)
                        continue;
                    matched = true;
                    const auto a = local_faces.orientations[j];
                    const auto b = source_faces.orientations[k];
                    require(a.rotation == b.rotation && a.reversed == b.reversed
                            && local_faces.coefficients[j] == source_faces.coefficients[k],
                        "local-to-canonical facet orientation changed across ranks");
                }
                require(matched, "partition facet is absent from source cell");
            }
        }

        // Verify that globally shared edge identities agree between partitions.
        auto globalSupport = source.relation(d - 1, d);
        for (Index facet = 0; facet < source.entity_count(d - 1); ++facet) {
            if (globalSupport[facet].targets.size() != 2)
                continue;
            int local_present = 0, all_present = 0;
            for (auto id : part.global_ids(d - 1))
                if (id == facet) {
                    local_present = 1;
                    break;
                }
            MPI_Allreduce(&local_present, &all_present, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            require(all_present >= 2, "partition interface has inconsistent shared facet ID");
        }
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    try {
        {
            const auto context = idmesh::mpi::Context::world();
            const auto from_handle = idmesh::mpi::Context::from_fortran_handle(
                static_cast<std::int64_t>(MPI_Comm_c2f(MPI_COMM_WORLD)));
            require(context.rank() == from_handle.rank()
                    && context.size() == from_handle.size(),
                "MPI context handle conversion changed communicator");
            auto global = source_mesh();
            run(context, global, 1);
            run(context, global, 2);
            if (context.rank() == 0)
                std::cout << "MPI: context, partition, shared identity and exchanges OK\n";
        }
        MPI_Finalize();
        return 0;
    }
    catch (const std::exception& e) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::cerr << "MPI test failed on rank " << rank << ": " << e.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
