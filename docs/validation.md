# Verification and known limits

IDMesh uses its repository CI to verify source formatting, native C++20
compilation, numerical examples, MPI multi-rank behavior and Kokkos's
CPU execution backend.

| Area | Automated verification | Scope |
| --- | --- | --- |
| C++ formatting | pinned clang-format 19.1.7 | all tracked .h/.hpp/.cpp |
| Serial core | Linux GCC, Windows MSVC, macOS Clang | reference cells, oriented incidence, Gmsh, VTK, Medit, geometric Jacobians |
| Independent applications | FVM and FEEC CTest executables | conservative flux and oriented CG=0 / DC=0 |
| Distributed MPI | OpenMPI with 2 and 3 ranks | replicated and rank-local shards, dual-graph cuts, 1/2 Ghost layers, shared global IDs and field exchanges |
| Kokkos | Serial runtime | persistent topology/geometry, device-owned Cell fields and matrix-free diffusion |
| MPI + Kokkos | two-rank OpenMPI + Kokkos Serial | device pack/unpack, automatic and forced Host Staging, direct GPU mode rejected when unavailable |
| External FE | Basix-backed Nedelec transformation CI | rotation/reflection group relations and shared-facet consistency |
| Package consumers | CMake installed-package tests | core, MPI, Kokkos and combined adapters |

## Device-hardware acceptance

GPU hardware is not present on standard GitHub-hosted Ubuntu runners.
The `GPU hardware validation` workflow is **manual only** and must be
run on a provisioned accelerator runner. It verifies that a real GPU
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
the repository's available CI runners. The manual workflow is ready for
an approved GPU runner, but must not be represented as a passing hardware
test until its run completes successfully.

## Performance scope

`partition_graph` is a balanced deterministic region-growing heuristic
on the Cell dual graph. It is useful as a dependency-free baseline.
`write_partition_shards` is an offline serial exporter and needs the
global source mesh once; **runtime** `read_partition_shard` only reads
a Rank's local owned/ghost data and uses a distributed identity registry.
Large multi-node weak scaling, ParMETIS-class partition quality, generic
distributed Gmsh ingestion, dynamic repartitioning and GPU-aware MPI
throughput are future work.
