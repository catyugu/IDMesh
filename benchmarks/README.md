# Reproducible local performance benchmarks

IDMesh benchmarks are **developer-only**. They are not built, executed, or
timed by GitHub Actions. The automated CI checks functional SDK correctness,
while all performance and memory analysis must run locally on a controlled
machine.

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

```sh
python3 benchmarks/run_baseline.py ./build-bench/idmesh_benchmark \
  --output benchmark-results --repeats 5

# Add larger 3D cases when the machine has sufficient memory.
python3 benchmarks/run_baseline.py ./build-bench/idmesh_benchmark \
  --output benchmark-results-large --repeats 5 --large
```

The runner is a local developer convenience and is not called by GitHub
Actions. Use CPU affinity externally when comparing revisions on a shared
workstation, and retain both output files with the benchmark report.

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
finalization. `validate_ms_p50` times one additional explicit exhaustive
`Mesh::validate()` pass after finalization, which helps estimate the validation
component already included by `finalize()`. For Gmsh input, the public import API combines file parsing and
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

The `idmesh_kokkos_benchmark` executable is also local-only. It measures
persistent geometry/relation mirror construction and upload separately from
resident reductions. Device upload includes host packing, allocation,
staging and synchronization. Real GPU timings require CUDA/HIP hardware and
a matching Kokkos build; Kokkos Serial is only a CPU correctness proxy.


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
