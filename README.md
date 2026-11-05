# IDMesh

IDMesh is a C++20 immutable unstructured mesh library with DMPlex-inspired oriented incidence topology, mixed 1D–3D standard cells, optional entity dimensions, CSR/fixed-stride connectivity, and ASCII Gmsh import.

## Build and test

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
cmake --install build --prefix install
```

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
