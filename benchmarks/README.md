# Reproducible local performance benchmarks

IDMesh benchmarks are **developer-only** and are excluded from normal
builds and functional CI. The separate `performance-validation.yml` workflow
is opt-in (or triggered only by the dedicated `bench/performance-validation`
development branch). Its shared-runner data is explicitly exploratory;
controlled-host measurements are required for a release performance claim.

Configure an independent Release build:

```sh
cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release \
  -DIDMESH_BUILD_BENCHMARKS=ON -DIDMESH_BUILD_TESTS=OFF
cmake --build build-bench --config Release --target idmesh_benchmark
```

## Baseline matrix runner

`run_baseline.py` executes every topology configuration in a fresh process,
so peak-RSS high-water marks are comparable across `minimal`,
`finite_volume` and `full`. It writes the unmodified benchmark columns plus
`case` and `process_repeat` to `raw.csv`, together with
`metadata.json` containing the exact commands and host/runtime metadata.
It also writes `layout_summary.csv`, `traversal_summary.csv` and
`report.md`. Independent-process medians are grouped by case and
materialization level. Traversal throughput is counted in **million oriented
or unoriented incidence entries read per second**, using the original
per-process timing and relation entry count.

```sh
python3 benchmarks/run_baseline.py ./build-bench/idmesh_benchmark \
  --output benchmark-results --repeats 5

# Add larger 3D cases when the machine has sufficient memory.
python3 benchmarks/run_baseline.py ./build-bench/idmesh_benchmark \
  --output benchmark-results-large --repeats 5 --large
```

For a bounded size sweep covering 2D mixed meshes and 3D simplex, tensor and
scrambled mixed meshes, execute:

```sh
python3 benchmarks/run_baseline.py ./build-bench/benchmarks/idmesh_benchmark \
  --scale --repeats 5 --output benchmark-scale
```

This uses 11 size/shape cases, three layouts (`minimal`, `finite_volume`,
`full`) and a fresh process per layout and repeat. The reference workflow
uses three repeats to bound hosted-runner cost; use five or more for an
engineering comparison. Use CPU affinity externally when comparing revisions
on a shared workstation, and retain the raw CSV and metadata alongside the
summary. Do not confuse the exposed logical payload with physical memory.

## Workload matrix

Each command measures **one topology materialization per process**. This is
essential: process peak RSS is a lifetime high-water mark; running minimal,
finite-volume and full consecutively in one process contaminates the peak.
Use `--trials 1` for clean memory measurements, repeat independent processes
at least five times for a distribution, and retain the CSVs.

```sh
./build-bench/idmesh_benchmark 56 5 --dim 3 --shape simplex \
  --config finite_volume --trials 1 > tet-fvm.csv
./build-bench/idmesh_benchmark 48 5 --dim 3 --shape mixed \
  --scramble --seed 2026 --jitter 0.10 --config full --trials 1 > mixed-full.csv
./build-bench/idmesh_benchmark 64 5 --dim 3 --shape tensor \
  --config minimal --trials 1 > hex-minimal.csv
./build-bench/idmesh_benchmark 192 5 --dim 2 --shape mixed \
  --config full --trials 1 > two-dimensional.csv
./build-bench/idmesh_benchmark 2 5 --gmsh /path/to/industrial.msh \
  --config finite_volume --trials 1 > imported.csv
```

For direct per-configuration timing distributions, `--trials 5` can be used,
but memory snapshots may then include allocator caching from earlier trials.
The default `--config all` remains convenient for functional comparisons,
but is **not** suitable for comparing per-config peak memory. A full topology
comparison should execute each case in a fresh process.

The generated workload covers structured tet, hex, and mixed hex/prism
meshes. `--scramble` independently permutes vertex IDs and insertion order,
and `--jitter` perturbs interior coordinates without changing connectivity.
These stress cases do not represent industrial unstructured mesh geometry;
import real Gmsh 2.2/4.1 ASCII datasets separately and record their origin.

## Timing and memory semantics

`input_ms_p50` times generated builder input; `finalize_ms_p50` times
finalization. `validate_ms_p50` separately times the explicit exhaustive
`Mesh::validate()` pass after finalization. RSS and peak RSS at the finalize
phase boundary are sampled **before** this validation pass. Finalization itself
performs structural checks; the explicit call performs additional algebraic
invariant checks. For Gmsh input, the public import API combines file parsing and
finalization, so `finalize_ms_p50` covers both and `input_ms_p50` is zero.

`visible_payload_bytes` is the exposed logical array payload lower bound.
It excludes vector capacity, allocation overhead, maps, object control blocks,
temporary builder memory, and any lazy caches. It is *not* physical RSS.

Memory columns are process-wide measurements, rather than per-mesh allocator
instrumentation:

- `rss_before_bytes_p50`: RSS before input/builder construction.
- `rss_after_input_bytes_p50`: RSS after generated input preparation, before
  finalize; for Gmsh input this equals the baseline, because import and
  finalize cannot be independently timed.
- `process_rss_bytes_p50`: RSS after finalize.
- `rss_delta_bytes_p50`: nonnegative difference between post-finalize RSS and
  baseline RSS; it includes retained allocator pages and unrelated allocations.
- `peak_before_bytes_p50`, `peak_after_input_bytes_p50`,
  `peak_after_finalize_bytes_p50`: process high-water marks at phase boundaries.
- `peak_delta_bytes_p50`: increase in process peak RSS over its pre-input
  value. It captures transient build allocations even if they are later freed.
- `rss_delta_bytes_per_cell_p50`: post-finalize process RSS increase per Cell;
  this is only an indicative cost, not an exact heap accounting.
- `process_peak_rss_bytes`: high-water RSS across all benchmark trials and
  configurations in the process; **never** subtract sequential configurations.

A zero RSS value means the platform cannot supply it. The only interpretable
peak comparison is separate, fresh processes with identical environments.
Do not report timings or memory measurements from shared CI runners.

`relation_bytes` counts the visible relation offsets/indices/sign/orientation
payload. Traversal checksum and `incidences_per_round` detect functional
changes across revisions. Time the same built revision, identical inputs,
compiler flags, CPU affinity, warm-up policy and operating system before
claiming a speedup. Report medians and spread over independent processes.

## Optional Kokkos measurements

The `idmesh_kokkos_benchmark` measures persistent geometry/relation mirror
construction and upload separately from resident reductions. Device upload
includes host packing, allocation, staging and synchronization. `--config`
selects one of `minimal`, `finite_volume` and `full`; use
`--trials 1` with a fresh process when measuring host RSS. New output columns
report the **logical Kokkos View byte footprint** of `DeviceMesh` and
`DeviceRelation`, plus process RSS before/after mirror creation (RSS is
available on Linux). View bytes exclude Kokkos allocator overhead and temporary
host staging; CPU RSS does not measure GPU device allocation.

Real GPU timing and memory measurements require CUDA/HIP hardware, a matching
Kokkos build and a device memory profiler. Kokkos Serial runs in the CPU memory
space and serves only as a CPU proxy for construction and resident-kernel cost.


## Finalization decomposition

For topology-construction profiling, compare downward-only and support-enabled
configurations in separate processes:

```sh
./build-bench/idmesh_benchmark 24 1 --dim 3 --shape simplex \
  --config finite_volume_downward --trials 1
./build-bench/idmesh_benchmark 24 1 --dim 3 --shape simplex \
  --config finite_volume --trials 1
./build-bench/idmesh_benchmark 24 1 --dim 3 --shape simplex \
  --config full_downward --trials 1
./build-bench/idmesh_benchmark 24 1 --dim 3 --shape simplex \
  --config full --trials 1
```

The support-enabled minus downward-only time approximates support-transpose
cost. The FVM-to-full difference captures the additional Edge/Face
materialization work. Add `--cell-labels` to assign one physical tag to every
generated top-dimensional Cell; comparing the same command without the flag
isolates a deliberately label-heavy finalization workload.

## Exploratory hosted-runner suite

`.github/workflows/performance-validation.yml` runs a separate engineering
measurement job on explicit dispatch or the dedicated benchmark branch. It
archives topology `raw.csv`, `layout_summary.csv`,
`traversal_summary.csv`, `report.md` and `metadata.json`; the backend
artifact holds independent Kokkos mirror, MPI halo, and MPI startup CSVs,
environment details, plus `kokkos_summary.csv`, `mpi_halo_summary.csv`,
`mpi_startup_summary.csv` and `report.md`.

The hosted data documents workload scale and detects gross unexpected costs.
It does not establish stable regression thresholds: CPU topology sweep,
MPI multi-rank tests and Kokkos Serial results may be affected by other
work on the runner, allocation caching, CPU frequency and runtime scheduling.
Do not combine performance samples from different hosts into a claim.
MPI halo timings are each rank's per-iteration median reduced with
`MPI_MAX`; the slowest-rank median is not the median of global per-iteration
critical-path latency.

## MPI shard startup measurements

`idmesh_mpi_startup_benchmark` separates offline shard generation from
runtime loading so process peak RSS is not contaminated by the serial global
mesh used during preprocessing. Prepare shards once, then launch each topology
layout in a fresh MPI process:

```sh
cmake -S . -B build-mpi-bench -DCMAKE_BUILD_TYPE=Release \
  -DIDMESH_BUILD_BENCHMARKS=ON -DIDMESH_BUILD_TESTS=OFF -DIDMESH_ENABLE_MPI=ON
cmake --build build-mpi-bench --target idmesh_mpi_startup_benchmark --parallel

./build-mpi-bench/benchmarks/idmesh_mpi_startup_benchmark \
  --prepare /tmp/idmesh-startup --parts 4 --side 32 --ghost-layers 1

for layout in cell_vertices cell_facets complete; do
  mpiexec -n 4 ./build-mpi-bench/benchmarks/idmesh_mpi_startup_benchmark \
    --prefix /tmp/idmesh-startup --parts 4 --layout "$layout"
done
```

The CSV decomposes rank-local startup into shard parsing, local immutable mesh
construction, local mapping of embedded identity metadata, per-dimension
`ExchangePlan`/distributed-graph construction and final validation. There are
no runtime registry or Halo-discovery byte counters because those exchanges are
removed from the v3 reader. Phase timings are reduced with `MPI_MAX`;
`peak_rss_delta_bytes_max` is the largest per-rank process peak increase during
runtime loading.

Run layouts in independent processes and repeat the complete command on a
controlled host before making performance claims. Shared GitHub runners are
useful only for functional smoke checks, not publication-quality timing.

## MPI halo measurements

`.github/workflows/mpi-benchmark.yml` is the single manual MPI engineering
comparison workflow. It covers both the reusable host halo and MPI+Kokkos
host-staging benchmarks. Host and Kokkos baselines are separate inputs because
their historical comparison points differ; the candidate is shared. All cases
run in one MPI container, warm both revisions, and alternate execution order
across five repeats to reduce runner warm-up and frequency bias. Its output is
diagnostic rather than a CI gate; publication-quality measurements still belong
on a controlled host.

When MPI is enabled, `idmesh_mpi_benchmark` measures the reusable host
request path created by `make_request`. It reports `start` (pack +
nonblocking launch), `wait` (completion + unpack), and their combined latency,
as well as the one-time `make_request()` setup cost.
Each median is reduced with `MPI_MAX`, so the printed value represents the
slowest rank.

```sh
cmake -S . -B build-mpi-bench -DCMAKE_BUILD_TYPE=Release \
  -DIDMESH_BUILD_BENCHMARKS=ON -DIDMESH_BUILD_TESTS=OFF -DIDMESH_ENABLE_MPI=ON
cmake --build build-mpi-bench --target idmesh_mpi_benchmark --parallel

mpiexec -n 2 ./build-mpi-bench/benchmarks/idmesh_mpi_benchmark \
  --side 96 --components 1 --ghost-layers 1 --iterations 50
mpiexec -n 4 ./build-mpi-bench/benchmarks/idmesh_mpi_benchmark \
  --side 96 --components 8 --ghost-layers 1 --iterations 50
```

Run the benchmark on controlled hosts for publication-quality comparisons.

When both MPI and Kokkos are enabled, `idmesh_mpi_kokkos_benchmark` uses caller-owned Kokkos storage, forces
`DeviceTransport::host_staging` and measures nonblocking `start`, `wait`,
and their no-overlap total. The `DeviceHaloPlan` itself is constructed before
steady-state timing; `device_halo_setup_ms_max` separately measures
its one-time rank-index mirrors, MPI layout construction and persistent
host/device buffer allocation. Physical GPU runs
are required to measure useful kernel/communication overlap; Serial timings only
measure the split-path overhead.
