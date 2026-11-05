# IDMesh architecture

IDMesh is a C++20, immutable mesh topology kernel for 1D–3D mixed-cell, unstructured meshes. Its model is inspired by DMPlex's oriented incidence DAG but its storage is independent of PETSc.

## Contracts

- Entity dimensions are 0..d, where d is the topological dimension. Vertices and cells always exist. Intermediate dimensions are explicitly materialized.
- Each local entity has a dimension-specific contiguous ID. IDs are stable for the lifetime of a finalized mesh; rebuilding may renumber everything.
- Every materialized entity has a canonical vertex ordering. Edges use ascending global vertex tags; polygonal facets use the lexicographically smallest cyclic ordering, including reversals.
- Adjacent downward incidences carry a boundary coefficient (±1). Facet-to-cell orientation is derived from the corresponding downward incidence; vertex-only transitive relations have no orientation.
- Face symmetries are represented by a cyclic offset and a reflection flag, not merely a sign. Finite-element-specific DOF transformations belong above the mesh layer.
- The complete complex satisfies boundary-of-boundary = 0. A reduced storage configuration does not claim to contain boundary operators for omitted dimensions.
- Relations are materialized explicitly as CSR or fixed-stride arrays. Device iteration never performs implicit topology construction.
- Orientation maps local[i] to canonical[i] by the documented rotation/reflection rule, including two-vertex edges.
- The graded incidence DAG cannot skip a dimension: in 3D, requesting edges requires faces. Invalid layer configurations are rejected by finalize(). Direct Cell-to-Vertex and Facet-to-Vertex connectivity is retained as geometry connectivity, not a boundary operator.
- The builder is mutable; a finalized Mesh is immutable and exposes zero-copy read-only views. No dynamic mesh edits, AMR, remeshing or repartitioning are supported in the first milestone.
- Geometric coordinates are independent of entity topology. Only affine first-order geometry is in the initial milestone.
- The base CMake target and installed public headers contain no third-party API types. A separate optional Kokkos backend owns Kokkos execution/runtime dependencies.

## Standard reference cells

Segment, triangle, quadrilateral, tetrahedron, hexahedron, prism and pyramid use Gmsh first-order vertex order. Their oriented boundary facets are defined in one compiled reference table and exposed through `cell_facet_count` and `reference_facet_vertices`; the builder uses the same table. A geometrically inverted cell still has a valid combinatorial complex, while geometric Jacobian checks belong to geometry validation.

## Core storage

The standard representation stores cell types, cell-to-vertex connectivity, vertex positions and optional intermediate entities. Each requested incidence is either uniform fixed-stride or CSR and may carry a compact signed incidence and permutation code. The builder can use maps and temporary vectors; these do not survive finalization.

## Scope limits

This milestone excludes runtime topology modification, arbitrary polyhedra, high-order geometry nodes, periodic entity equivalences, nonmatching interfaces, MPI distribution and user-authored arbitrary GPU kernels. The last three are future extensions and must not be represented as complete.

## Verification

Tests must cover all seven reference cells, rotated/reversed local facet orders, mixed adjacent cells, entity omission and explicit relation selection, boundary-of-boundary identities, imported Gmsh v2/v4 ASCII meshes, malformed inputs, installed-package consumption and CPU Kokkos execution.
