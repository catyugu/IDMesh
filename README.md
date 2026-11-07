# IDMesh

IDMesh is a C++20 immutable unstructured-mesh library for numerical simulation.
It uses a DMPlex-inspired oriented incidence model with compact array storage,
optional topology layers, and optional MPI/Kokkos backends.

The core SDK supports:

- mixed standard cells in 1D, 2D and 3D;
- reduced topology such as Cell + Facet for finite-volume methods, or the full
  oriented complex required by compatible finite elements;
- first- and supported second-order geometry;
- Gmsh MSH 2.2/4.1 ASCII, VTK legacy ASCII and Medit ASCII I/O;
- labels independent of topology materialization;
- optional distributed-memory and heterogeneous-compute adapters.

Public core headers do not expose third-party library types.

## Build

IDMesh requires CMake 3.26+ and a C++20 compiler.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
cmake --install build --prefix install
```

The main branch CI builds and tests Linux/GCC, Windows/MSVC and macOS/Clang
SDKs and publishes the installed trees as workflow artifacts.

For developer setup, contract rules and formatting checks see
[AGENTS.md](AGENTS.md) and [development conventions](docs/development.md).

## Use from CMake

After installing IDMesh:

```cmake
find_package(IDMesh CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE IDMesh::IDMesh)
```

A minimal finite-volume-style mesh:

```cpp
#include <idmesh/mesh.hpp>

#include <array>
#include <utility>

int main()
{
    idmesh::MeshBuilder builder(3, 3);

    builder.add_vertex({0.0, 0.0, 0.0});
    builder.add_vertex({1.0, 0.0, 0.0});
    builder.add_vertex({0.0, 1.0, 0.0});
    builder.add_vertex({0.0, 0.0, 1.0});

    const std::array<idmesh::Index, 4> cell {0, 1, 2, 3};
    builder.add_cell(idmesh::CellType::tetrahedron, cell);

    const auto mesh = std::move(builder).finalize(
        idmesh::BuildOptions{idmesh::TopologyLayout::cell_facets, idmesh::RelationDirection::bidirectional});

    const auto cell_to_facet = mesh.relation(3, 2);
    return cell_to_facet.rows == 1 ? 0 : 1;
}
```

Use `BuildOptions{TopologyLayout::cell_vertices, RelationDirection::downward}` for the smallest topology,
`BuildOptions{TopologyLayout::cell_facets, RelationDirection::bidirectional}` for Cell/Facet-oriented workflows, and
`BuildOptions{}` for the complete oriented complex.

## Mesh I/O and geometry

Mesh-only I/O is declared in `idmesh/io.hpp`:

```cpp
#include <idmesh/io.hpp>

auto mesh = idmesh::read_gmsh("model.msh");
auto vtk  = idmesh::read_vtk_legacy("model.vtk");
auto medit = idmesh::read_medit("model.mesh");
```

Gmsh import preserves supported labels and second-order geometry.
For field/state-aware VTK, VTU and PVD/PVTU I/O, include `idmesh/vtk.hpp`.
`idmesh/mesh.hpp` only declares mesh topology and construction; code using
mesh-only file adapters must include `idmesh/io.hpp`.
Reference-to-physical geometry evaluation is available through
`idmesh/geometry.hpp`.

## Partition preprocessing

Serial dual-graph partitioning and rank-local shard export are part of the core SDK and require no MPI dependency:

```cpp
#include <idmesh/partition.hpp>

auto owners = idmesh::partition_graph(mesh, parts);
idmesh::write_partition_shards(mesh, owners, ghost_layers, "mesh");
```

`write_partition_shards` accepts any valid per-Cell owner-rank vector, so
third-party partitioners can be integrated without introducing their types or
libraries into the IDMesh SDK. `partition_graph` is the dependency-free
baseline implementation.

The optional MPI backend reads those shards at runtime and exposes global IDs,
owner ranks, owned-entity sets and reusable halo exchange for every materialized
topology dimension.

## Optional backends

Enable MPI:

```sh
cmake -S . -B build-mpi -DIDMESH_ENABLE_MPI=ON
```

Consumers can then link `IDMesh::MPI` and include `idmesh/mpi.hpp`.
The public header does not expose `mpi.h` types. `Context::world()` is the simple case; application subcommunicators can be passed through `Context::from_fortran_handle()`.

Enable Kokkos:

```sh
cmake -S . -B build-kokkos -DIDMESH_ENABLE_KOKKOS=ON
```

Consumers can link `IDMesh::Kokkos` and use device-resident mesh and relation mirrors exposed through typed raw device pointers without exposing Kokkos template types in the SDK ABI.

When MPI is enabled, create an `idmesh::mpi::Context` from the world communicator
or an application subcommunicator and pass it explicitly to distributed mesh operations.
When both MPI and Kokkos are enabled, `IDMesh::MPIKokkos` provides type-agnostic device-aware halo transport over caller-owned device storage, with a host-staging fallback.

## Documentation

- [Architecture](docs/architecture.md)
- [Roadmap](docs/roadmap.md)
- [Validation and known limits](docs/validation.md)
- [Development conventions](docs/development.md)
- [Local benchmark protocol](benchmarks/README.md)

IDMesh is licensed under the MIT License.
