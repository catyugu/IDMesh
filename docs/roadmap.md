# Roadmap and acceptance criteria

## Milestone 1: immutable serial core
- [x] Initialize the template as the real IDMesh package; remove placeholders and unused sample code.
- [x] Implement 1D–3D mixed standard reference cells and basic builder input checking.
- [x] Construct canonical shared entities, incidence orientations and boundary coefficients.
- [x] Implement optional entity dimensions and explicit CSR/fixed-stride materialization.
- [x] Expose read-only topology and geometry views.
- [x] Import first-order ASCII Gmsh 2.2/4.1 meshes with physical cell and facet labels.
- [x] Run and verify the optional CPU Kokkos adapter with Kokkos kept out of core public headers.
- [x] Test boundary-of-boundary and installation on Linux, Windows and macOS.

## Milestone 2: distributed execution
- [x] Global entity identity and deterministic cross-rank orientation matching.
- [x] Owned/ghost ranges, partitioning and halo exchange; MPI remains an optional backend.
- [x] Rank-local shard input with a distributed shared-entity registry.
- [x] Geometry mirrors and caller-owned device-aware communication with host-staging fallback.
- [x] Communicator-scoped MPI contexts without exposing `mpi.h` in the SDK.
- [x] Reusable sparse-neighbor host halo exchange with persistent request buffers.
- [ ] Multi-node weak scaling and external partitioner adapters/tooling.

## Milestone 3: advanced mesh coupling
- Arbitrary polyhedral cells, nonmatching interface maps, periodic couplings and high-order curved geometry.
- Profile-based alternative layouts and GPU topology construction.

## Non-goals
AMR, remeshing, dynamic entity mutation, FEM-specific DOF transformations, solvers and partition-aware finite-element assembly.

## Extended IO and geometry (in progress)
- [x] Retain physical group memberships and names from all Gmsh dimensions.
- [x] Represent quadratic Gmsh geometry nodes separately from topological vertices.
- [x] Add VTK legacy ASCII and Medit ASCII first-order readers and writers.
- [x] Verify the supported high-order Gmsh imports and linear-format IO round-trips on all supported platforms.
- [x] Implement analytic order-1/2 geometry interpolation and Jacobians for all standard cell families.
- [x] Add supported native VTK quadratic types and Medit P2 (line/triangle/tetrahedron).
- [ ] Add custom Medit Ordering descriptors and extended pyramid14 VTK Lagrange encoding.

## Milestone 2 incremental implementation
- [x] Rank-local MPI partition loading and deterministic global entity identity.
- [x] Dimension-generic ownership sets and reusable entity-field ExchangePlan.
- [x] Persistent Kokkos topology/geometry mirrors with typed raw device descriptors (Serial CI).
- [x] Standalone FVM conservation/normal and FEEC incidence examples.
- [x] Scalable rank-local shard input and dependency-free graph-based partitioning baseline.
- [x] GPU-backed field buffers and device-aware halo exchange with host staging fallback.
- [x] Nonblocking device-aware halo exchange with reusable staging/request state.
- [ ] Physical CUDA/HIP validation of overlap and direct GPU-aware MPI.
- [x] Add executable Basix C++-backed Nedelec DOF-transformation integration tests and CI.

## Scalable distributed input
- [x] Deterministic Cell dual-graph balanced region-growing partitioner.
- [x] Serial graph partition to rank-local shard exporter (1/2 Ghost layers).
- [x] Independently read owned/ghost Cell and quadratic geometry shards on each MPI Rank.
- [x] Distributed registry for unique shared entity IDs and ownership.
- [x] Build flat sparse-neighbor halo plans from explicit global-ID ghost requests without global mesh replication.
- [x] Keep external partitioners behind the plain Cell-owner-vector shard-writer boundary.
- [ ] Add ParMETIS/PT-Scotch adapters/tooling and multi-node weak scaling when a concrete caller requires them.
- [ ] Fully distributed parallel generation of shard files from a global parallel mesh format.

## GPU transport and numerical execution
- [x] Keep application field ownership outside the mesh SDK; host/device halo transport operates on opaque fixed-width records.
- [x] Keep topology transport in the mesh SDK; discretization-specific kernels live above it.
- [x] ExchangePlan-scoped direct device-memory capability query and safe failure on unsupported memory spaces.
- [x] Packed Kokkos device Halo plan, including automatic host staging fallback.
- [ ] Verify CUDA/HIP kernels on physical GPUs using the reusable GPU hardware workflow.
- [ ] Verify direct CUDA/ROCm-aware MPI exchange on a capable multi-GPU MPI runtime.
