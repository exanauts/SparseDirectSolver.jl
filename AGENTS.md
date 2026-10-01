# AGENTS.md — instructions for coding agents working in this repository

This file is tool-agnostic. Claude Code reads it through `CLAUDE.md`
(`@AGENTS.md`); OpenCode, Codex-style tools and Hermes read `AGENTS.md`
directly.

## What this project is

SparseDirectSolver.jl is a portable sparse direct solver (LLᵀ/LLᴴ, LDLᵀ/LDLᴴ,
LDU) for GPUs written in Julia on KernelAbstractions.jl and GPUArrays.jl. It
replaces the closed-source NVIDIA cuDSS behind CUDSS.jl for the MadNLP
ecosystem, with the same parameter names and phases, and runs on CUDA, AMDGPU,
oneAPI, Metal and the CPU backend.

Three documents define the work. Read them in this order before touching code:

1. `PLAN.md` — the design: feature inventory, architecture (host symbolic
   analysis, three-regime multifrontal numeric phase, solve, backends), API,
   milestones, risks. Section numbers are referenced from the tasks.
2. `TASKS.md` — consecutive tasks, one per session, each with the tests that
   must pass and a Report block to fill in. **Your job is the first task whose
   status marker is `[ ]` (or `[~]` if a previous session left it unfinished).**
3. `RESEARCH.md` — background: state of the art, workload evidence, portability
   hazards. Consult when a design choice in `PLAN.md` needs its rationale.

## Session rules

* Do exactly one task from `TASKS.md`. Do not start the next one, even if time
  remains. Finish by setting the task's status marker and filling in its
  Report block (template at the top of `TASKS.md`). The owner re-evaluates
  `PLAN.md` and `TASKS.md` after every task.
* Do not edit `PLAN.md`. If the plan is wrong or impractical, do the task the
  best way you can and write the deviation and a suggested plan change into
  the Report. Do not edit other tasks in `TASKS.md` except to add a short note
  under "Open issues" of your own task that affects them.
* Do not weaken, skip or delete tests to get green. A test that cannot pass
  for a documented reason is marked `@test_broken` with a comment and listed
  in the Report.
* Keep the public names, parameter strings and phase strings of `PLAN.md`
  §1.2–§1.4 and §3.1. They mirror `../CUDSS.jl` so MadNLP can migrate
  mechanically.
* New dependencies need a one-line justification in the Report. Never `dev`
  unregistered or local checkouts into the project (there is a
  `../KernelAbstractions.jl` 0.10-dev checkout on this machine; the package
  uses the registered KernelAbstractions 0.9.x).
* The repository is `github.com/exanauts/SparseDirectSolver.jl` (remote
  `origin`, branch `main`). Commit at the end of the task with a message
  starting with the task id, e.g. `T05: symbolic pattern, ordering, etree,
  column counts`. Do not push unless asked. Do not commit `Manifest.toml`.

## Environment on this machine

* Julia 1.13 via juliaup (`/home/michel/.juliaup/bin/julia`); package compat is
  `julia = "1.10"`, so do not use syntax or stdlib features newer than 1.10.
* Linux (WSL2). One NVIDIA RTX 4080 (16 GB); CUDA.jl is functional.
* **No AMDGPU, oneAPI or Metal hardware here.** Local tests run on the
  KernelAbstractions CPU backend and on CUDA only. GitHub CI
  (`.github/workflows/ci.yml`, modelled on `../ExaPF.jl`) runs the CPU suite on
  GitHub runners and the GPU suite on the exanauts self-hosted runners labelled
  `cuda` and `amdgpu`. oneAPI and Metal extension files are written by analogy
  (task T23) and only precompile-checked.
* CUDA.jl 6.x is split into packages: `CUDACore` (`CuArray`, `CUDABackend`),
  `cuSPARSE` (`CuSparseMatrixCSR`, `CuSparseMatrixCSC`), `cuBLAS`, `cuSOLVER`.
  Use these as weak dependencies; tests load the umbrella `CUDA`.
* Reference code next to this repository:
  `../CUDSS.jl` (the API being mirrored; tests and docs are the contract),
  `../MadNLP-cudss-matching/lib/MadNLPGPU/ext/MadNLPGPUCUDAExt/cudss.jl`
  (how the main consumer uses the solver),
  `../KrylovPreconditioners.jl` (KA kernels and extension layout in the same
  ecosystem). Read them; never modify them.

## Commands

```bash
# full test suite (CPU backend + CUDA when functional)
julia --project=. -e 'using Pkg; Pkg.test()'

# CPU only / a subset of test files
SDS_TEST_GPU=0 julia --project=. -e 'using Pkg; Pkg.test()'
SDS_TEST_ONLY="test_symbolic_etree,test_options" julia --project=. -e 'using Pkg; Pkg.test()'

# instantiate / update after editing Project.toml
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# benchmarks (separate environment, needs CUDSS.jl for the cuDSS baseline)
julia --project=bench bench/cudss_baseline.jl
```

`SDS_TEST_GPU` and `SDS_TEST_ONLY` are implemented in `test/runtests.jl` (T01).

## Code conventions

* Julia style: 4-space indentation, no tabs, `snake_case` functions, `CamelCase`
  types, trailing `!` for mutating functions, docstrings on every public
  function and type, `using` only at module top level.
* Everything is generic in the element type `T` (`Float32`, `Float64`,
  `ComplexF32`, `ComplexF64`) and the index type `INT` (`Int32`, `Int64`).
  Never hard-code `Float64` or `Int` in kernels or layouts.
* Host symbolic analysis uses plain `Int` arrays and converts to `INT` once,
  with an overflow check, when moving to the device.
* KernelAbstractions kernels (see `PLAN.md` §2.4 and §2.7):
  1-D workgroups and `@localmem` reductions only; no subgroup/warp intrinsics;
  `@localmem` sizes come from `Val` parameters; no allocation and no host
  synchronization inside the numeric and solve phases; no atomics in assembly
  or extend-add (owner-pull, deterministic); atomics only where `PLAN.md`
  says so (default forward solve) and always with an atomic-free variant.
* Dense operations go through `src/dense/interface.jl` with the `impl`
  keyword (`:generic`, `:vendor`, `:ka`); never call cuBLAS/cuSOLVER directly
  from numeric code, only from the extension bindings.
* No `@allowscalar` in library code. Scalar reads of device memory happen
  only at phase boundaries (`info`, statistics) and are explicit.
* Errors use the hierarchy in `src/errors.jl`; unsupported parameters raise
  `NotSupportedError`, bad values `InvalidValueError`, never a silent fallback.
* Backend-specific code lives in `ext/`; the core must load and run with only
  the CPU backend.

## Test conventions

* Shared helpers live in `test/backends.jl`, `test/matrices.jl`,
  `test/utils.jl` (created in T01). Use them; do not define new generators or
  tolerances inside individual test files.
* Every numeric test loops over `BACKENDS` and over `ELTYPES` unless the task
  says otherwise. Seed with `Random.seed!(666)`. Tolerance is `tol(T) =
  sqrt(eps(real(T)))` on well-conditioned generators.
* Tests must not need the network. SuiteSparse downloads belong to `bench/`.
* A task is done only when its listed tests pass on both the CPU backend and
  CUDA on this machine, and the Report records the counts.

## Layout (grows with the tasks; see `PLAN.md` §4 for the target)

```text
PLAN.md  TASKS.md  RESEARCH.md  AGENTS.md
Project.toml  src/  ext/  test/  bench/  docs/
```
