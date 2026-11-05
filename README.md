# IDMesh

IDMesh is a C++20 immutable unstructured mesh library with DMPlex-inspired oriented incidence topology, mixed 1D–3D standard cells, optional entity dimensions, CSR/fixed-stride connectivity, and ASCII Gmsh import.

## Build and test

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
cmake --install build --prefix install
```

For physical groups, `cell_tags()` returns highest-dimensional Gmsh material
labels and `facet_tag(f)` returns the codimension-one physical label (or zero).
Gmsh 2.2 and 4.1 ASCII line/face boundary elements are matched to mesh facets;
Gmsh 1D point elements are matched to endpoints. When a reduced topology
configuration excludes facets, lower-dimensional physical labels are ignored.

Enable the optional Kokkos CPU adapter using `-DIDMESH_ENABLE_KOKKOS=ON`. Core public headers require no third-party dependencies.

## Install consumer

```cmake
find_package(IDMesh CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE IDMesh::IDMesh)
```

See [architecture](docs/architecture.md) and [roadmap](docs/roadmap.md). License: MIT.

## Benchmarks

`IDMESH_BUILD_BENCHMARKS` controls the standalone `idmesh_benchmark` executable (enabled for top-level development, disabled as a subproject). It is never installed.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DIDMESH_BUILD_BENCHMARKS=ON
cmake --build build --config Release
./build/idmesh_benchmark 128 20
```

CSV columns report construction time, cell-to-vertex traversal time, materialized relation bytes, and a traversal checksum for minimal, finite-volume and full configurations. This is a baseline rather than a statistically rigorous performance harness: run repeated trials in an isolated release build before drawing performance conclusions.

## Geometry order and additional formats

The topology is always defined by corner vertices. Quadratic interpolation
nodes (line3, tri6, quad8/9, tet10, hex20/27, prism15/18, pyramid13/14)
are stored separately and can be inspected with
`cell_order`, `cell_geometry_nodes` and `geometry_node_position`.
Gmsh MSH2/MSH4 ASCII import retains them, including shared geometry nodes.

Physical groups from **all** dimensions are accessible via
`physical_labels()`; labels on omitted topology layers keep their canonical
vertex keys and `entity == invalid_index`. Gmsh 4.1 permits multiple physical
groups per entity and preserves group names via `physical_name`. Legacy
`cell_tags()` and `facet_tag()` expose one primary label for compatibility.

Additional formats:
- `read_vtk_legacy` / `write_vtk_legacy`: VTK ASCII legacy UNSTRUCTURED_GRID,
  first-order cells and `physical_tag` cell attributes.
- `read_medit` / `write_medit`: Medit ASCII `.mesh` first-order cells and
  integer region references.

The `idmesh/geometry.hpp` API evaluates the first- and second-order
isoparametric map and its analytic Jacobian at a reference coordinate.
Reference node positions use Gmsh ordering, with conventional [-1,1] interval
and tensor-product coordinates and unit simplex coordinates.

VTK legacy ASCII now reads/writes 10 common quadratic variants (line3, tri6,
quad8/9, tet10, hex20/27, wedge15/18, pyramid13), with explicit Gmsh-VTK
node-order permutations. VTK's 14-node pyramid mapping is unavailable and
is rejected. Medit ASCII reads/writes native `EdgesP2`, `TrianglesP2`, and
`TetrahedraP2`; unsupported P2 shapes and custom ordering keywords are
rejected. Both formats retain one physical tag per element; multi-membership
on top cells is rejected rather than silently lost.

## Distributed and numerical examples

Build with `-DIDMESH_ENABLE_MPI=ON` and an MPI implementation to use
`IDMesh::MPI` and the MPI-free `idmesh/mpi.hpp` header. This first backend
supports replicated-input cell partitioning, global shared-entity identity,
configurable ghost layers, and blocking host cell-field halo exchanges.
MPI communication and partition construction are verified using 2 and
3 ranks. Scalable root-only/distributed-file ingestion is not yet supported.

`idmesh_fvm` validates conservative facet fluxes and physical normals on
a three-dimensional Edge-free finite-volume mesh. `idmesh_feec` constructs
the oriented G/C/D matrices on mixed Hex/Prism cells and verifies both
boundary-operator compositions vanish. The optional Kokkos adapter provides
a persistent `DeviceRelation` mirror for batched parallel reductions.
