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
- Global entity identity and deterministic cross-rank orientation matching.
- Owned/ghost ranges, partitioning and halo exchange; MPI remains an optional backend.
- Geometry/field mirrors and device-aware communication.

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
