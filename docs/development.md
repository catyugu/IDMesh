# Development conventions

## Working loop

From a fresh clone, use CMake 3.26+ and a C++20 compiler:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DIDMESH_BUILD_TESTS=ON
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
cmake --install build --config Release --prefix install
cmake -S tests/consumer -B consumer-build -DCMAKE_PREFIX_PATH="$PWD/install"
cmake --build consumer-build --config Release
```

See `AGENTS.md` for a short developer checklist. MPI and Kokkos are opt-in.

## CMake layout

The root `CMakeLists.txt` owns project options and the build entry points.
`cmake/Dependencies.cmake` acquires optional MPI/Kokkos dependencies and
configures CUDA/HIP when requested. `cmake/Targets.cmake` defines the core
and optional adapter libraries; `cmake/Install.cmake` exports the installed
SDK. The `tests/`, `examples/`, and `benchmarks/` directories each own
their executable targets and test registration.

Build options, public target aliases (`IDMesh::IDMesh`, `IDMesh::Kokkos`,
`IDMesh::MPI`, `IDMesh::MPIKokkos`), installed package files, and CTest
names remain stable. Test and example executables stage their required
runtime DLLs on Windows because they now build in subdirectories.


## MPI CI environment

The MPI-only and MPI+Kokkos build/test/install/consumer checks run sequentially
inside a single GitHub Actions job. The job uses a GHCR image containing the
Ubuntu 24.04 C++ toolchain and OpenMPI, pinned by digest in
`.github/workflows/ci.yml`. Ordinary CI never runs `apt-get` to install MPI.

The build definition lives at `.github/docker/mpi/Dockerfile`.
Editing it or `.github/workflows/mpi-image.yml` triggers a separate image
publishing workflow; it can also be started manually. After an intentional
image rebuild, copy the resulting `sha256:` digest from the publishing run
summary into the CI job's container image reference, then verify both MPI
build configurations. The publish workflow needs `packages: write`, while
the consumer job only needs `packages: read`.

The container is CI-only. Local SDK builds and installed-package consumers
use native MPI via `find_package(MPI)` and do not require Docker.



## Formatting

The checked-in `.clang-format` uses **clang-format 19.1.7**.
Provision this development tool locally, without a global installation:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install 'clang-format==19.1.7'
CLANG_FORMAT=.venv/bin/clang-format bash scripts/check-format.sh
```

On Windows, the binary resides at `.venv/Scripts/clang-format.exe`;
Git Bash can execute the shell scripts. Configure the optional staged-file
hook using `git config core.hooksPath .githooks`. Run
`scripts/format-cpp.sh` to format files. Formatting checks run through the
local pre-commit hook or the manual lint command; CI has no formatting job.

## API and failure contracts

Only `include/idmesh/` is part of the installed public API. The
`export.hpp` header owns all DLL/shared-library export definitions.
Implementation-private helpers live in `src/`.

Invalid caller arguments use `std::invalid_argument`; bad indices use
`std::out_of_range`; invalid object state uses `std::logic_error`;
size limits use `std::length_error`. Invalid file content, I/O failures
and MPI call failures use `std::runtime_error`. MPI errors include the
failed operation and an error description where supported. The private
`src/contracts.hpp` defines lightweight helpers; avoid repeating
messages at successive layers.

`IDMESH_ASSERT` is for programmer mistakes in internal invariants only.
It emits a diagnostic to stderr and terminates in both Debug and Release.
External inputs must be checked and rejected by exception. The library
does not implement logging, print routine status, or choose exit codes;
the caller controls reporting.

`Index` and topology offsets are uint32_t with `invalid_index`
reserved; MPI global IDs are uint64_t. Use std::size_t for host memory
counts and check representational limits before narrowing.

## Verification

Retain core functional/contract tests, independent installed-package
consumers, optional MPI/Kokkos paths and the existing hardware-validation
policy. Examples and benchmark executables may print diagnostics. The
benchmark remains a local-only tool rather than a CI gate.

## Technical scope

Public SDK headers cannot require Kokkos, MPI, or other third-party
headers. MPI and device backends are optional libraries and must keep
their runtime preconditions and scaling limitations documented.

## Hardware validation

`.github/workflows/gpu-validation.yml` uses only `workflow_call`.
A trusted calling workflow supplies `CUDA` or `HIP`, an approved GPU
runner label, and the matching CUDA architecture or AMD GFX target.
The runner must already have the device driver, CUDA/ROCm compiler,
CMake, and an MPI runtime available. Direct GPU-aware MPI requires a
runtime compiled with appropriate accelerator support. If unavailable,
the test still verifies the safe staging path and explicitly records
that native GPU-aware MPI was not exercised.

GitHub's standard Ubuntu runners cannot validate physical CUDA/HIP
execution. For a public repo, do not expose a persistent self-hosted
GPU machine to untrusted pull request workflows. Prefer an isolated,
ephemeral, manually approved GPU runner.
