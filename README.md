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
