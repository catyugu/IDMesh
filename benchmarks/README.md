# Reproducible benchmark protocol

The benchmark executables are disabled in ordinary SDK builds. Configure a
dedicated Release build with `-DIDMESH_BUILD_BENCHMARKS=ON`. Set
`-DIDMESH_ENABLE_KOKKOS=ON` only when comparing the optional Kokkos
adapter. The CPU and Kokkos executables emit CSV on stdout.

## CPU measurement matrix

Run at least five trials per configuration and collect several independent
process runs for stable RSS and timing comparisons. Examples:

```sh
./idmesh_benchmark 32 10 --dim 3 --shape simplex --trials 5 > tet.csv
./idmesh_benchmark 32 10 --dim 3 --shape tensor --trials 5 > hex.csv
./idmesh_benchmark 32 10 --dim 3 --shape mixed --trials 5 > mixed.csv
./idmesh_benchmark 32 10 --dim 3 --shape mixed --scramble --seed 2026 --jitter 0.1 --trials 5 > scrambled.csv
./idmesh_benchmark 2 10 --gmsh /absolute/path/to/model.msh --trials 5 > imported.csv
```

For each family and each materialization (minimal, finite-volume, full),
compare finalization time, available adjacency traversal time, incidence
throughput, logical payload and process RSS. A zero process RSS field means
the current platform cannot report it. RSS high-water marks refer to the
entire executable lifetime. They must not be subtracted across the three
sequential materialization cases to infer per-mesh heap allocation.

The generated scrambled workload permutes vertex IDs and cell insertion
order with a fixed PRNG seed. Optional jitter perturbs interior coordinates
but preserves all cell connectivity. These are controlled synthetic stress
cases, not replacements for imported industrial meshes.

The Gmsh code path has a combined public read/finalize API; its
`finalize_ms_p50` field includes I/O, parsing and mesh construction.
Consequently, its `input_ms_p50` is zero. Do not compare Gmsh's combined
number to the synthetic builder's finalize-only number without stating
this distinction.

The first 18 CPU CSV columns remain backward compatible with the original
topology benchmark. New columns include an estimated visible payload lower
bound, whole-process RSS and incidence counts per traversal round. CPU
traversal timings cover all requested repetitions; divide by `rounds`
to obtain a per-round estimate. Recheck checksums after refactors.

## Kokkos measurements

The optional `idmesh_kokkos_benchmark` uses identical generated/Gmsh
inputs and the same three topology configurations. It measures:

- Persistent geometry mirror construction and upload.
- Persistent Cell-Facet relation mirror construction and upload, if present.
- Resident batch centroid reductions.
- Resident geometry-index checksum.
- Resident relation incidence reductions, if present.

Mirroring measurements include host packing and intermediate buffer
allocation, so they do not isolate raw host-to-device bandwidth.
Kernel timing excludes initial upload; it includes reduction dispatch and
completion. Use actual GPU hardware and a correctly configured Kokkos
device backend before calling any CUDA/HIP numbers device results.

## Environment record

Record the repository SHA, compiler/flags, CPU/GPU, memory capacity,
operating system, Kokkos backend, mesh input provenance, Gmsh version if
available, and whether execution was pinned to particular cores. Prefer
single-user isolated runs and measure identical releases when comparing
implementations. CI smoke tests only verify correctness and build
integration; CI timing numbers are not suitable performance evidence.
