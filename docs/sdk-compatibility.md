# SDK compatibility and distribution

IDMesh is currently at **0.3.0**. Its public C++20 interface consists of
installed headers under `include/idmesh/`, the CMake package `IDMesh`,
and these targets:

| Installed target | Availability | Public header |
| --- | --- | --- |
| `IDMesh::IDMesh` | Always | `mesh.hpp`, `geometry.hpp`, `data.hpp`, `io.hpp`, `vtk.hpp`, `partition.hpp` |
| `IDMesh::MPI` | `IDMESH_ENABLE_MPI=ON` | `mpi.hpp` |
| `IDMesh::Kokkos` | `IDMESH_ENABLE_KOKKOS=ON` | `kokkos.hpp` |
| `IDMesh::MPIKokkos` | Both backends enabled | `mpi_kokkos.hpp` |

An installed header may be present even when its optional backend was not
built; use the presence of the exported CMake target to detect availability.
All source files and headers under `src/` are private implementation details.
Public headers deliberately do not require third-party MPI/Kokkos includes.

## Release versions

The installed `IDMeshConfigVersion.cmake` uses CMake
`SameMinorVersion`. During the 0.x series:

- Patch releases within a minor line (for example, `0.3.0` to `0.3.1`)
  should preserve public source compatibility and the supported binary
  interface for a matching build toolchain.
- Minor releases may revise public source/API contracts and may require
  consumers to rebuild. Migration notes should accompany such changes.
- The shared-library SOVERSION follows `major.minor` (currently `0.3`).
  This separates incompatible minor lines on platforms with versioned SONAMEs.
  Windows does not use ELF SONAME rules.

Prefer a versioned package lookup in applications:

```cmake
find_package(IDMesh 0.3 CONFIG REQUIRED)
target_link_libraries(my_solver PRIVATE IDMesh::IDMesh)
```

This requests a compatible SDK in the `0.3.x` line. When targeting optional
backends, link their corresponding imported targets. A package with no MPI
or Kokkos adapter must not present those targets.

## Binary compatibility boundary

The public interface contains standard-library types, including
`std::span`, `std::string`, `std::vector`, `std::shared_ptr` and
exceptions. PImpl conceals internal storage but does **not** establish a
compiler-agnostic C ABI.

Binary compatibility of a prebuilt C++ application with an IDMesh update is
limited to the same architecture, compatible compiler/standard-library ABI,
runtime library and build configuration. In particular, Linux
`_GLIBCXX_USE_CXX11_ABI`, MSVC toolset/CRT configuration and optional
MPI/Kokkos runtimes must remain compatible. Static consumers must relink
when replacing the static library. Source compatibility does not imply a
cross-toolchain or cross-platform binary guarantee.

On ELF and macOS platforms, installed shared adapters search their own
directory for sibling IDMesh libraries using `$ORIGIN` or `@loader_path`.
On Windows, applications must make installed DLLs available on the DLL
search path or stage them beside their executable; installed CTest consumers
configure their runtime search path explicitly.

Applications own MPI and Kokkos initialization/finalization; destroy objects
backed by those runtimes before shutting them down. See
[architecture](architecture.md) for lifetime and borrowed-view contracts.

## Validation before release

The installed consumer fixture in `tests/integration/consumer/` is
intentionally pinned to the current `0.3` line. It is built outside the
source build tree with `find_package(IDMesh 0.3 CONFIG REQUIRED)`, and CTest
**executes** the compiled programs, not just the compiler/linker.

It tests the base mesh and a VTK round-trip using installed symbols; if
present, it loads and invokes the MPI, Kokkos and combined adapters without
initializing their application-managed runtimes. Backend numerical and
multi-rank correctness remains covered by the separate integration suites.

```sh
cmake --install build --config Release --prefix install
cmake -S tests/integration/consumer -B consumer-build -DCMAKE_PREFIX_PATH="$PWD/install"
cmake --build consumer-build --config Release
ctest --test-dir consumer-build -C Release --output-on-failure
```

Run these checks for each shipped build variant (static/shared, platform,
enabled adapters). The CI matrix runs them for its Linux/GCC, Windows/MSVC,
macOS/Clang, Kokkos Serial, MPI and MPI+Kokkos configurations.

These checks establish **current-release installed-source compatibility**
and executable package integrity. They do not yet prove compatibility of
historical precompiled application binaries with new shared libraries.
Such a guarantee requires retaining an archived consumer binary for each
supported ABI line and testing it against later patch releases using a
matching runtime environment.
