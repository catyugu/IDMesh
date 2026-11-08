# IDMesh developer workflow

IDMesh is a C++20 mesh SDK. Keep its immutable topology representation,
optional materialized dimensions, and public third-party-free headers intact.
Do not introduce logging infrastructure, a configuration framework, or an
additional abstraction without a current caller.

## Working loop

- Setup: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DIDMESH_BUILD_TESTS=ON`.
  MPI and Kokkos are opt-in (`IDMESH_ENABLE_MPI` /
  `IDMESH_ENABLE_KOKKOS`); Kokkos is fetched by the declared CPM version.
- Build: `cmake --build build --config Release --parallel`.
- Test: `ctest --test-dir build -C Release --output-on-failure`.
- Run: `ctest --test-dir build -C Release -R '^idmesh[.]example[.]' --output-on-failure`.
- Lint: `bash scripts/check-format.sh`, using clang-format 19.1.7.
  The formatter is a development dependency, not a runtime dependency.

Run both core and installed-package consumer tests before a release; exercise
MPI/Kokkos builds when changing adapters. Run `ctest --test-dir consumer-build
-C Release --output-on-failure` after configuring and building the installed
consumer. See `docs/sdk-compatibility.md` before changing public signatures,
exported targets, package versioning or binary layout. Hardware CUDA/HIP execution needs
the dedicated trusted GPU runner.

## Conventions

- `include/idmesh/*.hpp` is the supported SDK surface; headers under
  `src/` are implementation-private. Never expose Kokkos or MPI types through core APIs.
- `include/idmesh/export.hpp` is the sole definition of ABI export macros.
- `Index` is uint32_t and `invalid_index` is reserved; use size_t for host
  container sizes and check for narrowing/overflow before conversions.
- Reject invalid caller arguments with typed exceptions, malformed file input
  with runtime errors, invalid object state with logic errors, and invalid
  indices with out_of_range. Use the private checks in `src/contracts.hpp`.
- Use `IDMESH_ASSERT` for actual internal bugs only. It terminates in every
  build mode; never assert on data from outside the library.
- Do not print or configure logging inside the library. Examples, benchmarks,
  and CLI test drivers may decide how to display diagnostics.
- Add focused contract/integration tests for public API changes. Do not add
  compatibility aliases or test every helper in isolation.
