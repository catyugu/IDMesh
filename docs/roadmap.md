# Roadmap and acceptance criteria

## Milestone 1: immutable serial core
- [ ] Initialize the template as the real IDMesh package; remove placeholders and unused sample code.
- [ ] Implement 1D–3D mixed standard reference cells and robust builder input checking.
- [ ] Construct canonical shared entities, incidence orientations and boundary coefficients.
- [ ] Implement optional entity dimensions and explicit CSR/fixed-stride materialization.
- [ ] Expose read-only topology and geometry views.
- [ ] Import first-order ASCII Gmsh 2.2/4.1 meshes with physical cell labels.
- [ ] Run the optional CPU Kokkos adapter with Kokkos kept out of core public headers.
- [ ] Test boundary-of-boundary and installation on Linux, Windows and macOS.

## Milestone 2: distributed execution
- Global entity identity and deterministic cross-rank orientation matching.
- Owned/ghost ranges, partitioning and halo exchange; MPI remains an optional backend.
- Geometry/field mirrors and device-aware communication.

## Milestone 3: advanced mesh coupling
- Arbitrary polyhedral cells, nonmatching interface maps, periodic couplings and high-order curved geometry.
- Profile-based alternative layouts and GPU topology construction.

## Non-goals
AMR, remeshing, dynamic entity mutation, FEM-specific DOF transformations, solvers and partition-aware finite-element assembly.
