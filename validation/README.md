# Real-engineering production-readiness validation

This directory is an **external SDK consumer**, not part of IDMesh itself.
It uses only installed public headers and `IDMesh::IDMesh`. Files under
`data/` and `results/` are local, ignored, reproducible outputs.

## Source and integrity

`meshes.json` pins upstream revisions, paths, declared byte sizes and Git
blob SHA-1 or LFS SHA-256 digests. `fetch_meshes.py` checks every byte
before making files visible and never silently accepts a changed upstream
asset. An upstream Git repository's software license does not necessarily
license embedded engineering mesh geometry; ownership/redistribution terms
for the MFEM examples are **not verified**. Do not mirror these datasets or
include them in source/release artifacts without explicit confirmation.

```sh
python3 validation/fetch_meshes.py --tier ci
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
cmake --install build --prefix install

cmake -S validation/consumer -B build-validation \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$PWD/install"
cmake --build build-validation --parallel 2

mkdir -p validation/results
./build-validation/idmesh_thermal \
  --mesh validation/data/mfem-piece.msh --problem patch \
  --validate-topology \
  --output-vtu validation/results/piece-patch.vtu \
  --report validation/results/piece-patch.json
./build-validation/idmesh_thermal \
  --mesh validation/data/mfem-piece.msh --problem heat \
  --report validation/results/piece-heat.json
```

On Windows use PowerShell paths, and choose
`build-validation/Release/idmesh_thermal.exe` with a multi-config generator.

## What is actually verified

The P1 finite-element consumer assembles the symmetric Poisson/diffusion
operator from tetrahedral connectivity and physical coordinates, with
application-owned matrix-free element storage and Jacobi-preconditioned CG.
The linear patch test imposes the affine exact solution on all boundary
vertices. The heat case imposes temperatures 0 and 1 at the low/high x
contacts and treats all other boundaries as natural zero-flux boundaries.
Both verify a recomputed free-DOF residual, discrete global reactions and
nonnegative energy; heat also checks the discrete maximum principle.
An optional VTU output round-trips the temperature field through the public
I/O API. Timings, iteration counts, mesh counts and RSS on Linux are JSON.

Note: local extrema and cross-section boundary classification are based on
the imported bounding box and are suitable for diagnostic use, not an
engineering thermal boundary-condition model for arbitrary geometries.
Linear P1 patch tests on non-degenerate tetrahedra should be exact to
solver tolerance. Cell dimension 3 / tetrahedron / first-order is required;
unsupported element families fail explicitly rather than being skipped.

The verified second mesh is the 30,431-cell induction-heating tetrahedral grid; its
linear patch test completed with a 4.09e-10 infinity error, 8.75e-11 relative
residual and 124 PCG iterations in the initial Ubuntu CI probe.
The large mechanical bracket is distributed as binary Gmsh 4.1 through Git LFS;
the stress probe converts it to supported Gmsh ASCII with a **test-only** pinned
meshio tool. This dependency is not linked to or exported by the SDK.

## Acceptance levels

- **CI**: download a pinned moderately sized real mesh; compile against the
  **installed** SDK; execute both physical and patch test cases, including
  output field round-trip.
- **Manual**: extend mesh coverage and iterate on mesh family and shape.
  This is not currently claimed to pass.
- **Stress / distributed / accelerated**: million-cell, multi-node
  weak scaling, MPI-owned field assembly, Kokkos device solve and hardware
  CUDA/HIP are *not yet* validated by this workflow. Keep results distinct
  from small-mesh CI and from synthetic microbenchmarks.
- **Operational release gate**: data acquisition reproducibility, compiler
  matrix, sanitizer runs, peak memory budgets, and a published baseline are
  pending. Do not call the SDK production ready based on this example alone.

The solver exits nonzero on failed invariants, failed convergence,
unsupported input, field-corrupting I/O or invalid numerical results. The
JSON report includes both correctness and performance observations.
