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
- Optional facet physical tags attach to materialized codimension-one entities by vertex identity, and are validated against actual mesh facets. Zero denotes untagged; the dense array is omitted when no tags are supplied.
- The base CMake target and installed public headers contain no third-party API types. A separate optional Kokkos backend owns Kokkos execution/runtime dependencies.

## Standard reference cells

Segment, triangle, quadrilateral, tetrahedron, hexahedron, prism and pyramid use Gmsh first-order vertex order. Their oriented boundary facets are defined in one compiled reference table and exposed through `cell_facet_count` and `reference_facet_vertices`; the builder uses the same table. A geometrically inverted cell still has a valid combinatorial complex, while geometric Jacobian checks belong to geometry validation.

## Core storage

The standard representation stores cell types, cell-to-vertex connectivity, vertex positions and optional intermediate entities. Each requested incidence is either uniform fixed-stride or CSR and may carry a compact signed incidence and permutation code. The builder can use maps and temporary vectors; these do not survive finalization.

## Scope limits

This milestone excludes runtime topology modification, arbitrary polyhedra, high-order geometry nodes, periodic entity equivalences, nonmatching interfaces, MPI distribution and user-authored arbitrary GPU kernels. The last three are future extensions and must not be represented as complete.

## Verification

Tests must cover all seven reference cells, rotated/reversed local facet orders, mixed adjacent cells, entity omission and explicit relation selection, boundary-of-boundary identities, imported Gmsh v2/v4 ASCII meshes, malformed inputs, installed-package consumption and CPU Kokkos execution.

## Physical groups on every entity dimension

Physical labels are independent of the optional entity materialization mask.
Each `PhysicalLabel` carries a dimension, membership tag, canonical corner
vertex key and a linked local entity ID when available. If the entity dimension
was omitted (or the labeled element does not occur in the top-dimensional
complex), the link is `invalid_index` but the label survives. One entity can
belong to multiple groups. `Mesh::physical_tags(d, id)` returns all memberships
for an existing entity; `Mesh::physical_labels()` enumerates the complete
record set. Gmsh 4.1 entity multi-membership and `$PhysicalNames` are preserved.
`cell_tags()` and `facet_tag()` remain compatibility accessors for the first
physical tag only.

## Second-order geometry

The seven first-order reference topologies remain the topological entities.
`MeshBuilder::add_quadratic_cell` accepts interpolation nodes beyond corner
vertices via a separate geometry-node pool. Cell-local geometry connectivity
retains Gmsh ordering and is exposed by `cell_geometry_nodes`, `cell_order`
and `geometry_node_position`. The supported node counts are line 3, triangle 6,
quadrilateral 8/9, tetrahedron 10, hexahedron 20/27, prism 15/18 and pyramid
13/14. Shared high-order Gmsh node tags map to the same geometry-node ID;
high-order nodes do not increase the number of topological vertices. Purely
linear meshes reuse their existing cell-to-vertex arrays and allocate no
separate geometry-connectivity CSR. Geometry
interpolation is now provided by the optional geometry evaluation API; finite-element field shape functions and DOF transformations remain responsibilities of the discretization layer.

## IO capabilities and limitations

- Gmsh MSH 2.2/4.1 ASCII: first- and second-order standard cells, physical
  memberships for dimensions 0 through d, physical names, and geometry nodes.
- VTK legacy ASCII `UNSTRUCTURED_GRID`: import/export for seven linear cell
  types, optional lower-dimensional labeled elements, integer `physical_tag`
  scalar arrays. Other VTK variants and quadratic cells require separate
  ordering adapters.
- Medit ASCII `.mesh`: import/export for seven linear cell types, vertex
  references and lower-dimensional labeled elements. Binary `.meshb` and
  high-order Medit sections are outside the present implementation.
- Formats with one integer reference per element cannot express multiple
  physical memberships on a top-dimensional element. Writers reject such
  cases instead of silently dropping labels. Neither VTK nor Medit writer
  silently discards high-order geometry; both reject order-two input.

## Reference geometry maps

`idmesh/geometry.hpp` evaluates `x(xi) = sum_i N_i(xi) X_i` and
`J_{aj} = sum_i X_{ia} dN_i/dxi_j`. Shape functions use Lagrange bases
for simplex and tensor cells, quadratic serendipity on quad8/hex20,
tensor-product triangle × interval on prism18, the 15-node wedge basis,
and collapsed rational pyramid bases for pyramid5/13/14. Derivatives are
computed from the same expressions using forward-mode scalar differentiation.
The collapsed pyramid coordinate chart has a singular derivative at its apex
(z=1); evaluate_geometry rejects Jacobian evaluation at that point.

All reference node arrays use Gmsh ordering; the VTK adapter explicitly
permutes Tet10, Hex20/27, Wedge15/18 and Pyramid13. Cartesian tensor cells
use [-1,1] coordinates. Simplices use nonnegative barycentric coordinates
whose sum is at most 1. Prism combines a unit reference triangle with
z in [-1,1]; pyramid has base [-1,1]^2 at z=0 and apex at (0,0,1).

High-order VTK legacy ASCII uses native cell codes (21-29, 32) where
supported. Pyramid14 has no matching classical VTK type and is explicitly
rejected. Medit ASCII uses `EdgesP2`, `TrianglesP2`, `TetrahedraP2`
with direct node order. Medit custom P2 ordering sections and other P2
shape families are rejected, pending a standard-preserving adapter.
