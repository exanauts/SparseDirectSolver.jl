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
   must pass and a Report block to fill in. **Your job is the task you were
   given (the GitHub issue, see "GitHub workflow"); without one, it is the first
   task whose status marker is `[ ]` (or `[~]` if a previous session left it
   unfinished).**
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
  column counts`. Do not commit `Manifest.toml`. In a GitHub Actions session
  push and open the PR as described under "GitHub workflow"; in a local
  session do not push unless asked.

## GitHub workflow

Every task is tracked on GitHub and lands through a pull request that CI and a
reviewing agent must pass. The moving parts:

* **Milestones** are `PLAN.md` §5 (`M0 — …` to `M13 — …`). **Issues**: one per
  task, title `TNN — <task title>` (`External — …` for the MadNLP task), label
  `task`, milestone set. The issue only points at `TASKS.md`; the task text
  there stays the single source of truth. Tasks are strictly sequential: the
  previous task's issue must be closed before the next one starts.
* **Implementer** (`.github/workflows/claude-implement.yml`, Claude Opus 5.5,
  effort medium) starts when a task issue gets the label `claude:implement`
  (the owner, or the pipeline after the previous merge). It works on
  `ubuntu-latest` with Julia and the KA CPU backend only; CUDA is verified
  by CI on the PR. Branch `task/TNN-<slug>` from `main`; commits
  `TNN: …`; PR titled `TNN: …`, labelled `claude:pr`, milestone set, body from
  `.github/pull_request_template.md` with the exact line `Closes #<issue>`. The
  Report block is filled in before the PR is opened ("CUDA: pending CI on
  the PR" under Tests). If the task cannot be finished, the PR is opened as
  a draft and the issue labelled `needs-owner`; never a silent stop.
* **Reviewer** (`.github/workflows/claude-review.yml`, Claude Fable 5.1, effort
  high) reviews every push on a `task/**` or `claude:pr` PR against the task
  text, this file and `PLAN.md`, posts inline comments and a verdict (a
  neutral review comment, so it never blocks a merge by itself) whose first
  line is `VERDICT: APPROVE (head <sha>)` or
  `VERDICT: CHANGES_REQUESTED (head <sha>)`. On changes requested the
  implementer (same workflow, job `fix`) addresses every finding on the same
  branch, replies on threads it rejects, and pushes; at most 3 rounds, then the
  PR is labelled `needs-owner`.
* **Pipeline** (`.github/workflows/claude-pipeline.yml`) reacts to completed CI
  and review runs. A red `Run tests`/`Aqua` run on a task PR triggers a CI-fix
  round by the implementer (at most 3, comments `CI-FIX round k`). Green CI on
  the current head plus `VERDICT: APPROVE` for that head squash-merges the PR
  (which closes the issue) and starts the next task issue unless it is labelled
  `on-hold` or `external`. The repository variable `CLAUDE_AUTOPILOT=false`
  pauses merging and chaining. Only the status checks are required by the
  branch rules; approval is the verdict line, since a bot cannot approve its
  own PR.
* **Issues opened by agents**: anything found outside the task's scope (a bug
  in earlier code, a limit hit, a plan problem) becomes an issue with label
  `found-by-agent` using `.github/ISSUE_TEMPLATE/agent-finding.md`, cited in
  the Report. Do not fix it in the task's PR and do not edit `PLAN.md`; the
  owner decides what becomes a task. **Such issues hold the chain**: after a
  task PR merges, the pipeline starts the next task only when every open
  `found-by-agent` issue carries the label `triaged` or is closed; otherwise it
  comments on the next task issue and waits. Labelling the last untriaged issue
  `triaged` (or closing it) resumes the chain (`claude-triage.yml`).
* The owner re-evaluates `PLAN.md` and `TASKS.md` between tasks by pausing the
  chain (`CLAUDE_AUTOPILOT=false` or `on-hold` on the next issue), editing on
  `main`, and relabelling.

## Environment

### GitHub Actions (where the agents run)

* `ubuntu-latest`, Julia `1` from `julia-actions/setup-julia`, the project
  instantiated, `gh` authenticated as the Claude GitHub App. **CPU backend
  only**: no GPU, CUDA.jl is not installed; GPU results come from the
  self-hosted `cuda` runner through `ci.yml` on the PR (the `amdgpu` runner is
  disabled in `ci.yml` until the AMDGPU extension exists, T23).
* Reference code is cloned next to the checkout, read-only: `../CUDSS.jl` and
  `../KrylovPreconditioners.jl` (same relative paths as below). MadNLPGPU's
  cuDSS integration is not checked out; when a task needs it, read it from the
  `MadNLP/MadNLP.jl` repository on GitHub (`lib/MadNLPGPU`).

### The owner's machine

* Julia 1.13 via juliaup (`/home/michel/.juliaup/bin/julia`); package compat is
  `julia = "1.13"` (the minimum; CI tests the latest release only), so do not
  use syntax or stdlib features newer than 1.13.
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
  `../KrylovPreconditioners.jl` (KA kernels and extension layout in the same
  ecosystem). Read them; never modify them. How the main consumer uses the
  solver: MadNLPGPU's cuDSS binding in the `MadNLP/MadNLP.jl` repository
  (`lib/MadNLPGPU`).

## Commands

```bash
# full test suite (CPU backend + CUDA when functional)
julia --project=. -e 'using Pkg; Pkg.test()'

# CPU only / GPU only (as the CUDA CI jobs) / a subset of test files / all but some
SDS_TEST_GPU=0 julia --project=. -e 'using Pkg; Pkg.test()'
SDS_TEST_CPU=0 julia --project=. -e 'using Pkg; Pkg.test()'
SDS_TEST_ONLY="test_symbolic_etree,test_options" julia --project=. -e 'using Pkg; Pkg.test()'
SDS_TEST_SKIP="test_aqua" julia --project=. -e 'using Pkg; Pkg.test()'

# instantiate / update after editing Project.toml
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# benchmarks (separate environment, needs CUDSS.jl for the cuDSS baseline)
julia --project=bench bench/cudss_baseline.jl
```

`SDS_TEST_GPU` and `SDS_TEST_ONLY` are implemented in `test/runtests.jl` (T01),
`SDS_TEST_CPU` (in `test/backends.jl`) and `SDS_TEST_SKIP` were added for CI: the
CUDA jobs run with `SDS_TEST_CPU=0 SDS_TEST_SKIP=test_aqua`, since the CPU-only
jobs already cover the CPU backend and Aqua. A task's "passes on CPU and CUDA"
still means the default run (both backends) on the owner's machine.

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
.github/workflows/   ci.yml Aqua.yml (checks)  claude-implement.yml claude-review.yml claude-pipeline.yml (agents)
.github/scripts/     pr-verdict.sh (reads the reviewer's verdict)
.github/ISSUE_TEMPLATE/agent-finding.md  .github/pull_request_template.md
```
