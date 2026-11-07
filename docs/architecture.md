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
- Consecutive-dimension oriented boundary relations retain aligned `signs` and `orientations` arrays, including transposed support. Direct geometric Cell-to-Vertex and 3D Facet-to-Vertex relations are downward-only and store only indices: `row_signs` and `row_orientations` return empty spans. Device mirrors follow the same optional metadata contract.
- Orientation maps local[i] to canonical[i] by the documented rotation/reflection rule, including two-vertex edges.
- The graded incidence DAG cannot skip a dimension: in 3D, requesting edges requires faces. Invalid layer configurations are rejected by finalize(). Direct Cell-to-Vertex and Facet-to-Vertex connectivity is retained as geometry connectivity, not a boundary operator.
- The builder is mutable; a finalized Mesh is immutable and exposes zero-copy read-only views. Finalization performs structural validation; the explicit `Mesh::validate()` API additionally checks exhaustive algebraic invariants such as boundary-of-boundary cancellation. Dynamic mesh edits, AMR and remeshing are outside the current design.
- Geometric coordinates are independent of entity topology. First- and supported second-order geometry are stored separately from topological corner entities.
- Sparse integer labels are independent of topology materialization. Bound labels borrow entity connectivity; labels on omitted layers retain compact canonical vertex keys.
- The base CMake target and installed public headers contain no third-party API types. A separate optional Kokkos backend owns Kokkos execution/runtime dependencies.

## SDK and failure contracts

The public headers under `include/idmesh/` expose no third-party types.
Visibility definitions for the core and optional library adapters are
centralized in `export.hpp`. Implementation helpers remain private to
`src/`.

Invalid API arguments, indices, object state and capacity limits raise
typed standard C++ exceptions. Malformed mesh files and backend operations
raise runtime errors with relevant context. The internal Release-active
`IDMESH_ASSERT` is for programmer bugs, never for external data.
IDMesh does not provide an internal logging subsystem.

## Standard reference cells

Segment, triangle, quadrilateral, tetrahedron, hexahedron, prism and pyramid use Gmsh first-order vertex order. Their oriented boundary facets are defined in one compiled reference table and exposed through `cell_facet_count` and `reference_facet_vertices`; the builder uses the same table. A geometrically inverted cell still has a valid combinatorial complex, while geometric Jacobian checks belong to geometry validation.

## Core storage

The standard representation stores cell types, cell-to-vertex connectivity, vertex positions and optional intermediate entities. Each requested incidence is either uniform fixed-stride or CSR and may carry a compact signed incidence and permutation code. The builder can use maps and temporary vectors; these do not survive finalization.

## Scope limits

The current core excludes runtime topology modification, arbitrary polyhedra, periodic entity equivalences, nonmatching interfaces and user-authored arbitrary GPU kernels. MPI distribution and supported second-order geometry are implemented as described below; dynamic repartitioning and general parallel mesh ingestion remain future work.

## Verification

Tests must cover all seven reference cells, rotated/reversed local facet orders, mixed adjacent cells, entity omission and explicit relation selection, boundary-of-boundary identities, imported Gmsh v2/v4 ASCII meshes, malformed inputs, installed-package consumption and CPU Kokkos execution.

## Labels on every entity dimension

Labels are sparse integer metadata attached to entities and remain independent of
the topology materialization mask. Each `Label` carries a dimension, value,
canonical corner key and a linked local entity ID when that dimension exists.
If the entity dimension is omitted, `entity == invalid_index` and the label
retains its canonical vertex key. An entity may carry multiple values.

`Mesh::labels(d, id)` returns the zero-copy labels of a materialized entity;
`Mesh::labels()` enumerates all labels, including omitted topology layers.
`label_name(d, value)` provides optional descriptive metadata. Materialized
labels borrow packed entity connectivity; omitted-layer labels borrow a compact
internal key buffer. Copies of `Mesh` share that backing storage.

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


## MPI (optional) and executable validation

`-DIDMESH_ENABLE_MPI=ON` builds the `IDMesh::MPI` adapter without exposing
`mpi.h` from public SDK headers. The application initializes/finalizes MPI.
Production distributed meshes are loaded rank-locally from partition shards.
Every materialized entity dimension carries stable global IDs, owner ranks,
an explicit owned-entity index set, and a reusable halo `ExchangePlan`.
Lower-dimensional entities retain canonical local numbering; ownership is
therefore represented by `owned_entities(dimension)` rather than by requiring
an owned-first local range.

`exchange_plan(dimension).make_request(components)` allocates counts,
displacements and packed buffers once. Repeated `start(field)` / `wait()`
cycles perform no IDMesh-side host allocation. The same transport applies to
Vertex, Edge, Facet and Cell fields. `start` packs owner records and launches
`MPI_Ialltoallv`; `wait()` completes the collective and unpacks ghost
records, allowing independent work to overlap the communication interval.

The bundled graph partitioner and shard writer are core, dependency-free serial
preprocessing tools exposed by `idmesh/partition.hpp`. The shard writer takes
a plain Cell-owner vector; that vector is the integration boundary for external
partitioners such as METIS-family or PT-Scotch tooling, without exposing those
libraries in the SDK. The built-in partitioner's quality is not claimed
comparable to those packages. The MPI backend duplicates and owns the
communicator represented by `mpi::Context`.

`idmesh_fvm` assembles a 3D two-tetra diffusion operator using
`BuildOptions{TopologyLayout::cell_facets, RelationDirection::bidirectional}` (no Edge layer), checking conservative
internal flux and outward physical normals. `idmesh_fem` assembles G, C, D
on a mixed Hex/Prism mesh and verifies `CG=0`, `DC=0`.

The Kokkos adapter additionally offers a `DeviceRelation` owning a
persistent CSR mirror in the default Kokkos memory space. Its offsets,
targets and optional orientation metadata are exposed as typed device
pointers, so application kernels operate directly on resident data without
library-specific reduction helpers. CPU Kokkos Serial is exercised in CI;
CUDA/HIP device execution requires separate backend builds and device CI
to be certified.

## External high-order finite-element orientation validation

The `examples` CI job builds all three examples and
uses Basix's C++-backed N1E degree-two element transformations. It tests
edge reversal involutions, face rotation/reflection group relations and
the consistency of canonical DOF coefficients through both local views of
a shared tetrahedral facet. This checks the mesh's geometric permutation
contract against an independent finite-element DOF representation.
No finite-element-specific transformation matrices are stored in the mesh.

### Device-resident geometry

`idmesh::kokkos::DeviceMesh` copies geometry-node coordinates in SoA
layout, cell types/orders and cell-to-corner CSR to Kokkos's
`DefaultExecutionSpace::memory_space`. Linear cells reuse corner
connectivity. Only order-two cells occupy the separate
`high_order_geometry_*` CSR, matching the compact Host representation.
Typed raw device pointers let application kernels build their own traversal
without exposing Kokkos template types in the SDK ABI.

### Cross-rank canonical orientation invariant

Partition construction assigns local corner-vertex IDs in ascending source
global-vertex order, independently of owned/ghost cell discovery order.
Because entity canonicalization compares vertex IDs, this order-preserving
renumbering makes the canonical edge direction and facet vertex cycle
identical after mapping to global IDs on every Rank. The MPI regression test compares shared canonical entity vertex sequences and
consecutive boundary sign/permutation metadata across ranks on 2- and 3-rank
partitions, including a connected Tet/Hex/Prism/Pyramid 3D mesh. It also checks
that two Ghost layers contain the one-layer local Cell set.

## Rank-local distributed mesh reading and graph partitioning

The `IDMESH_SHARD 2` ASCII schema is:

```text
IDMESH_SHARD 2
DIM <topology_dimension> <geometry_dimension>
VERTICES <count>
<global_vertex_id> <x> <y> <z>
GEOMETRY_NODES <count>
<global_geometry_id> <x> <y> <z>
CELLS <count>
<global_cell_id> <owner_rank> <cell_type_enum> <num_geometry_nodes> <ids...>
LABELS <count>
<dimension> <value> <count> <global_corner_ids... or single global_cell_id>
NAMES <count>
<dimension> <value> "<name>"
END
```

Each Rank reads only its local shard. Top-dimensional cells are sorted
into owned cells followed by Ghost cells. Corner vertices receive local
IDs sorted by external global vertex identity, so their canonical edge
and facet orientations agree across ranks. Every materialized subentity
is registered by `(dimension, sorted global corner key)` with a
distributed hash-sharded registry. The registry assigns unique
dimension-local 64-bit global IDs and chooses the minimum owning Rank
among incident cells. Requests and answers use bounded MPI_Alltoallv,
with storage proportional to local entities plus the Rank's hash
registry share, rather than the complete global mesh.

After global identity and ownership are established, each materialized
dimension builds its halo plan from global-ID requests sent to entity owners.
The resulting `ExchangePlan` is topology-dimension agnostic and operates
without accessing the original mesh or the identity registry.

`idmesh::partition_graph` performs balanced region growing on the
Facet-adjacency cell dual graph, initializing well-separated seeds and
using BFS frontiers. `idmesh::write_partition_shards` expands a configurable
number of Ghost layers, then writes only required vertices and geometric
nodes. This is a **serial preprocessing** route; partition quality and
scalability are not claimed comparable to ParMETIS.

Limitations: shards carry explicit first- or second-order Cell
connectivity and supplied Ghost layers; the reader does not infer
missing off-rank neighbors from geometry. All distributed operations use
the communicator owned by the caller-supplied `mpi::Context`; metadata
operations are still blocking and local shard files must be consistent.
Extremely large individual messages remain subject to MPI_Alltoallv's
integer-count limits.

## GPU fields and direct MPI transport

The Kokkos backend owns `DeviceMesh`, `DeviceRelation` and
`DeviceField` storage while keeping Kokkos types out of the installed
ABI. Discretization-specific operators are intentionally outside the mesh
SDK; applications can build them from mirrored topology/geometry and device
fields. The combined MPI/Kokkos adapter packs owner records and unpacks ghost records
for fields on any materialized entity dimension using Kokkos kernels.

`DeviceHaloPlan` stores the rank-index maps in device memory and
supports `automatic`, `host_staging`, and `require_gpu_aware` transports.
When using Open MPI, capability is queried via
`MPIX_Query_cuda_support` / `MPIX_Query_rocm_support`, when present.
`DeviceHaloPlan::start` packs owned records, fences the pack dependency,
performs D2H staging when required, and launches `MPI_Ialltoallv`.
Applications may launch work that does not depend on Ghost values before
calling `wait`; MPI progress during that interval depends on the MPI runtime.
`wait` completes MPI, performs H2D staging when required, unpacks Ghost
records, and fences completion. Direct GPU-aware transport keeps the packed
buffers resident on device; host staging owns persistent host mirrors and
precomputed MPI counts/displacements. Steady-state start/wait performs no
IDMesh-side allocation.

The default CMake backend is `SERIAL`. The optional CUDA/HIP modes
compile the Kokkos adapter sources in the CUDA/HIP CMake language and
are exposed as `IDMESH_KOKKOS_DEVICE_BACKEND`. A reusable GPU workflow
tests real devices only on explicitly provisioned runners.
