# Real engineering mesh validation

This directory contains a **standalone C++20 numerical application** that
uses only installed public IDMesh SDK headers. Its tetrahedral P1 heat
conduction solver assembles local stiffness matrices, solves with
Jacobi-preconditioned conjugate gradients, and exports fields through the
public VTU API. No application-specific solver dependency enters the SDK.

## Real meshes, reproducible acquisition

`validation/meshes.json` is the source-of-truth for external datasets.
`fetch_meshes.py` downloads only requested files and validates their
exact byte count and SHA-256 or Git blob checksum. All raw and derived data
live in ignored `validation/data/` and results in `validation/results/`.
No large mesh archives are committed to Git.

| Case | Source | Elements | In validation |
| --- | --- | ---: | --- |
| **Pantheon architectural model** | University of Tokyo ADVENTURE | **1,329,027 TET4** | Main CI: patch, heat, VTU |
| Induction-heating model | MFEM data repository | 30,431 TET4 | CI: patch |
| Mechanical bracket | meshio++ LFS example | 235,014 TET4 | CI: patch, heat, VTU |

The Pantheon source is the author's published `pantheon_b60cmc.zip`
(8,881,569 bytes, SHA-256
`d6984aa5e624b1b8660b0fe896fff975c493227cd825b8eb4e7467027593843d`).
The ADVENTURE project explicitly permits free use of sample data:
https://adventure.sys.t.u-tokyo.ac.jp/software/samples.html

Its original `.msh` is **ADVENTURE format**, not Gmsh. The
dependency-free `convert_adventure.py` reads the archived original mesh,
verifies that it contains exactly 1,329,027 tetrahedra and 273,157 vertices,
and emits native **Gmsh MSH 2.2 binary** for IDMesh to import. This is a
format conversion: no synthetic subdivision or remeshing is performed. The
original asset remains owned and distributed by its source.

MFEM geometry redistribution terms have not been independently confirmed.
Its input files are therefore not included in source or release artifacts.
The mechanical bracket is fetched from a pinned Git LFS SHA-256 and
converted via a test-only `meshio==5.3.5` environment. The converter
fills missing Gmsh physical tags with zero, so that particular case does not
validate material/physical-group preservation.

## Reproduce the million-cell case

From the repository root:

```sh
python3 validation/fetch_meshes.py --tier ci
python3 validation/convert_adventure.py \
  --archive validation/data/adventure-pantheon.zip \
  --output validation/data/adventure-pantheon.msh \
  --report validation/results/pantheon-conversion.json

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DIDMESH_BUILD_TESTS=OFF -DIDMESH_BUILD_EXAMPLES=OFF
cmake --build build --parallel 2
cmake --install build --prefix install

cmake -S validation/consumer -B build-validation \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$PWD/install"
cmake --build build-validation --parallel 2

./build-validation/idmesh_thermal \
  --mesh validation/data/adventure-pantheon.msh --problem patch \
  --max-iterations 4000 --validate-topology \
  --output-vtu validation/results/pantheon-patch.vtu \
  --report validation/results/pantheon-patch.json
./build-validation/idmesh_thermal \
  --mesh validation/data/adventure-pantheon.msh --problem heat \
  --max-iterations 6000 \
  --report validation/results/pantheon-heat.json
```

On Windows with a multi-config generator, use the appropriate
`Release/` executable and platform-specific paths.

## Measured acceptance (Ubuntu 24.04, GCC 13, Release)

The following are **single GitHub-hosted runner observations**, not
hardware-normalized speed or scaling benchmarks:

| Metric | Pantheon affine patch | Pantheon heat conduction |
| --- | ---: | ---: |
| Imported vertices | 273,157 | 273,157 |
| Imported tetrahedra | 1,329,027 | 1,329,027 |
| Materialized facets | 2,784,999 | 2,784,999 |
| Free nodal DOFs | 146,210 | 258,793 |
| Jacobi-PCG iterations | 127 | 1,393 |
| Recomputed relative residual | 8.73e-11 | 9.73e-11 |
| Max affine solution error | 1.42e-9 | — |
| Discrete reaction imbalance | 6.99e-11 | 3.71e-10 |
| Import + topology (s) | 4.34 | 4.19 |
| Assembly (s) | 0.334 | 0.331 |
| Solve (s) | 2.15 | 22.93 |
| VTU write + read (s) | 1.95 | — |
| Peak Linux process RSS (MiB) | ~786 | ~676 |

The heat case imposes temperatures 0 and 1 on the low/high x boundary
contact regions and natural zero flux elsewhere; its computed range is
`[-2.6915e-5, 1.00000037]` (maximum overshoot 2.69e-5).
An arbitrary tetrahedral P1 Galerkin matrix need not be an M-matrix, so
the discrete maximum principle cannot be required on all real meshes.
Overshoot is recorded as a diagnostic, while actual acceptance **requires**
nonnegative energy, finite solution, converged recomputed residual,
discrete flux conservation, and the exact affine patch test. The patch
also validates the exported nodal temperature's VTU round trip.

The conversion itself produces a 44,861,261-byte Gmsh binary mesh in
about 4.53 s on the observed runner, with SHA-256
`3425fff4731474e14aad46eaec1f6d23fac0f8d2e6bc2549bf33add00fcf0a49`.
Hash and conversion timing are recorded in
`pantheon-conversion.json`.

## Verification scope and remaining release gates

The application rejects invalid or degenerate tetrahedra,
nonmanifold facets, unsupported element families, non-SPD CG operations,
nonconvergence, excessive true residual, failed conservation, and
corrupted VTU field data. The workflow separately checks the expected
published million-mesh cell and node counts; numerical and timing
metrics are uploaded as JSON artifacts.

This now establishes an actual **million-tetrahedron serial numerical
end-to-end** validation lane. It does not establish a million-DOF solve:
the Pantheon heat case uses 258,793 free nodal DOFs. Multi-node MPI
numerical assembly/solve, GPU-device execution, a reproducible performance
baseline across machines, mixed-element PDE discretization, and historical
SDK/ABI compatibility remain separate production-readiness work.
