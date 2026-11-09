# Verification and known limits

IDMesh uses its repository CI to verify native C++20 compilation,
numerical examples, MPI multi-rank behavior and Kokkos's CPU execution
backend. Source formatting is checked locally through the pre-commit hook.

Tests are split into two CTest layers. `unit` tests are dependency-free,
in-memory checks of local contracts and algorithms. `integration` tests exercise
file-format pipelines, installed consumers and real backend runtimes, including
multi-rank MPI and Kokkos execution. Both layers use the repository's small
`tests/support/test.hpp` runner; no external C++ test framework is required.
Installed-package consumers are built and executed from a separate CMake
project against the installed SDK. They check the current package and linker
contracts; historical binary-to-new-library compatibility still needs an
archived consumer build and is not claimed by these tests.

| Area | Automated verification | Scope |
| --- | --- | --- |
| Unit layer | Linux GCC, Windows MSVC, macOS Clang | reference-cell contracts, data primitives, analytic geometry and VTK common mappings |
| Serial integration | Linux GCC, Windows MSVC, macOS Clang | Gmsh/VTK/Medit round-trips, manifests, partition shards and a fixed Gmsh -> field operation -> VTU -> read-back fixture |
| Independent applications | `examples` CI job, three CTest examples | Basix orientation, conservative FVM flux, and mixed-cell FEM/FEEC exact sequence CG=0 / DC=0 |
| Distributed MPI | OpenMPI with 2 and 3 ranks | rank-local shards, ghost-depth monotonicity, mixed Tet/Hex/Prism/Pyramid 3D topology, cross-rank identity/orientation, Vertex/Edge/Facet/Cell field exchanges, abandoned host requests and pre/post-finalize MPI guards |
| Kokkos | Serial runtime | persistent topology/geometry mirrors, matrix-free diffusion over application-owned device storage, and post-finalize construction rejection |
| MPI + Kokkos | two-rank OpenMPI + Kokkos Serial | type-agnostic all-dimension device pack/unpack over caller-owned storage, abandoned device requests, automatic and forced Host Staging, direct GPU mode rejected when unavailable |
| External FE | Basix-backed Nedelec transformation CI | rotation/reflection group relations and shared-facet consistency |
| Package consumers | CMake installed-package CTest on Linux, Windows and macOS | compatible 0.3.x package discovery, actual core mesh/VTK I/O, optional adapter symbol invocation, Windows DLL discovery and installed sibling-library RPATH resolution |


## Device-hardware acceptance

GPU hardware is not present on standard GitHub-hosted Ubuntu runners.
The `GPU hardware validation` workflow is **reusable** (`workflow_call`)
and must be invoked by a trusted caller on a provisioned accelerator runner.
It has no direct manual or automatic repository event trigger. It verifies that a real GPU
is visible before building Kokkos with CUDA/HIP and executing the device
numerical and two-rank Halo tests.

| Backend | Example runner | Compiler architecture | Kokkos architecture |
| --- | --- | --- | --- |
| CUDA | approved NVIDIA GPU runner | 75 (Tesla T4) | TURING75 |
| HIP | approved AMD GPU runner | gfx90a (MI200 series) | AMD_GFX90A |

The MPI runtime must have the relevant GPU-aware extension to enter
direct device-buffer MPI. The device Halo test enters strict direct mode
only if that runtime reports support; otherwise it verifies Host
Staging and the explicit unsupported-mode error.

**Current acceptance boundary:** CUDA and HIP hardware execution and
direct GPU-aware MPI on physical GPUs have not yet been demonstrated by
the repository's available CI runners. The reusable workflow can run
only through a trusted calling workflow with an approved GPU runner and
must not be represented as a passing hardware test until that hardware
execution completes successfully.

Benchmarks remain outside routine CTest and CI. The optional
`performance-validation.yml` workflow is limited to manual runs and the
benchmark development branch; its shared-runner results are exploratory.
See `benchmarks/README.md` for isolated-process memory measurements,
backend profiling and controlled-host acceptance criteria.

## Performance scope

`partition_graph` is a balanced deterministic region-growing heuristic
on the Cell dual graph. It is useful as a dependency-free baseline.
`write_partition_shards` is an offline serial exporter and needs the
global source mesh once. It writes stable entity identity, ownership and Halo
metadata into each execution-ready shard. **Runtime** `read_partition_shard`
therefore reads only a Rank's local owned/Ghost data, maps embedded global IDs
to local entities, and builds flat sparse-neighbor Halo plans backed by MPI
distributed-graph communicators; it performs no distributed identity registry
or Halo-discovery exchange.
Large multi-node weak scaling, ParMETIS-class partition quality, generic
distributed Gmsh ingestion, dynamic repartitioning and GPU-aware MPI
throughput are future work.

## GPU workflow trigger policy

`.github/workflows/gpu-validation.yml` exposes only `workflow_call`.
A trusted workflow may invoke it with `backend` (CUDA or HIP),
`runner_label`, `compiler_arch` and `kokkos_arch`. The GPU validation
workflow is not directly invokable from the Actions **Run workflow** UI
and does not start on push or pull request.


## Validation levels

`MeshBuilder::finalize()` performs structural checks required for publishing an
immutable mesh: relation shapes and ranges, geometry connectivity, label
references, and storage consistency. The public `Mesh::validate()` call repeats
those checks and additionally evaluates the exhaustive oriented-complex
invariant `boundary(boundary(entity)) = 0`.

Tests and diagnostic tools should call `Mesh::validate()` when they need the
full mathematical invariant check. Production construction does not repeat the
more expensive boundary-of-boundary pass on every finalized full topology.
