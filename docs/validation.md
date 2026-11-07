# Verification and known limits

IDMesh uses its repository CI to verify native C++20 compilation,
numerical examples, MPI multi-rank behavior and Kokkos's CPU execution
backend. Source formatting is checked locally through the pre-commit hook.

| Area | Automated verification | Scope |
| --- | --- | --- |
| Serial core | Linux GCC, Windows MSVC, macOS Clang | reference cells, oriented incidence, Gmsh, VTK, Medit, geometric Jacobians |
| Independent applications | `examples` CI job, three CTest examples | Basix orientation, conservative FVM flux, and mixed-cell FEM/FEEC exact sequence CG=0 / DC=0 |
| Distributed MPI | OpenMPI with 2 and 3 ranks | rank-local shards, ghost-depth monotonicity, mixed Tet/Hex/Prism/Pyramid 3D topology, cross-rank identity/orientation and Vertex/Edge/Facet/Cell field exchanges |
| Kokkos | Serial runtime | persistent topology/geometry mirrors and matrix-free diffusion over application-owned device storage |
| MPI + Kokkos | two-rank OpenMPI + Kokkos Serial | type-agnostic all-dimension device pack/unpack over caller-owned storage, automatic and forced Host Staging, direct GPU mode rejected when unavailable |
| External FE | Basix-backed Nedelec transformation CI | rotation/reflection group relations and shared-facet consistency |
| Package consumers | CMake installed-package tests | core, MPI, Kokkos and combined adapters |


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

Local-only benchmarks are intentionally excluded from all GitHub Actions
workflows and CTest registration. See `benchmarks/README.md` for isolated-process
memory measurements and reproducible performance runs.

## Performance scope

`partition_graph` is a balanced deterministic region-growing heuristic
on the Cell dual graph. It is useful as a dependency-free baseline.
`write_partition_shards` is an offline serial exporter and needs the
global source mesh once; **runtime** `read_partition_shard` only reads
a Rank's local owned/ghost data, uses a distributed identity registry, and
builds flat sparse-neighbor halo plans backed by MPI distributed-graph
communicators.
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
