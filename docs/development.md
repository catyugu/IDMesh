# Development conventions

## Formatting

The repository uses the checked-in `.clang-format` style with
**clang-format 19.1.7** as the reference formatter. On Windows, the
following commands work in Git Bash; on Linux/macOS, use a shell.

```sh
python -m pip install clang-format==19.1.7
git config core.hooksPath .githooks
bash scripts/format-cpp.sh
bash scripts/check-format.sh
```

The Git pre-commit hook verifies the *staged* version of changed C/C++
files and rejects a nonconforming commit. It does not rewrite partially
staged worktree files. Apply `clang-format -i path/to/file.cpp`, stage it,
and retry. Override the executable with `CLANG_FORMAT` if needed.

CI independently runs the repository-wide formatter check, so commits
made without the local hook cannot bypass formatting enforcement.
The one-time format bootstrap workflow only activates with an explicit
`[format-bootstrap]` commit message or manual dispatch.

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
