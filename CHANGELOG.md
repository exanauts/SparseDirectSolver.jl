# Changelog

All notable changes to SparseDirectSolver.jl are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/). Task ids (`TNN`) refer to `TASKS.md`,
whose Report blocks hold the details and the measurements.

## [0.4.0] - 2026-11-13

### Breaking

- The MadNLP extension (`ext/SparseDirectSolverMadNLPExt.jl`) and the exports
  `SDSSolver` and `SDSSolverOptions` are removed, and MadNLP is no longer a
  weak dependency. The binding is MadNLP's own: `MadCoreGPU.SDSSolver` in
  MadNLP 0.12 (the default sparse solver of `rocMadNLP`, `oneMadNLP` and
  `mtlMadNLP`, selectable in `cuMadNLP`), built on `KKTSolver`. The scripts of
  `bench/e2e/` still use the removed `SDSSolver` and need SparseDirectSolver
  0.3 with MadNLP 0.11 until they are ported.

### Added

- `KKTSolver`, the front end for the KKT systems of an interior-point method:
  pipeline choice (the fused pipeline where the backend runs it, else the stock
  `DirectSolver`), the factorization parameters of `:cholesky` and `:ldl` (no
  pivoting or Bunch–Kaufman), `factorize!` for the refactorization, `solve!`
  in place, and the public `inertia` and `pipeline`. The keyword `matching`
  sets the `matching_alg` of an `L D Lᵀ` (`"algo5"`/`"algo6"` also give the
  2×2 pivot pairs, with or without pivoting), `deferred_analysis = true`
  leaves the analysis to the first `factorize!` on the values present then,
  and an `L D Lᵀ` with matching, `pivot_epsilon` or `pivot_sign` runs on the
  stock pipeline, which replaces small pivots (`fused = true` refuses them
  instead of dropping them; #193). A `SparseMatrixCSC` is read in place, as
  `csr_of_transpose` reads it: one with spare capacity in `rowval`/`nzval`
  raises `InvalidValueError`. With `"algo1"`–`"algo4"` and no pivoting the
  analysis looks for no 2×2 pivot pairs, as without matching.
- `SDSTuning` and `default_tuning` are exported from the top module.
- `matching_alg = "none"` is accepted as the cuDSS 1.0 spelling of matching
  off.

### Changed

- On the CPU backend and on Metal, Cholesky and `L D Lᵀ` without pivoting run
  through `KKTSolver` on the stock pipeline; the removed MadNLP extension
  refused them on the CPU backend.

## [0.3.0] - 2026-11-09

First registered release.

### Added

- The solver as built by tasks T01–T23 and T31 (`STATE.md` §1, milestones
  M0–M9): the CUDSS.jl handle API (`DirectSolver`, `execute!` with the cuDSS
  phase strings, `setparam!`/`getparam`, the named phase wrappers and the
  `LinearAlgebra` layer); host analysis (AMD or METIS nested dissection,
  supernodes, static factor layout); the three-regime multifrontal Cholesky,
  LDLᵀ/LDLᴴ with in-front Bunch–Kaufman pivoting and LU with in-front threshold
  pivoting; solves with multiple right-hand sides, sub-phases, iterative and
  FGMRES refinement; uniform batches; Schur complement mode; MC64 matching and
  scaling; the CUDA, AMDGPU, oneAPI and Metal extensions.
- `FusedSolver`, the fused pipeline (one dependency-counter factorization
  kernel and a partitioned-inverse solve) behind the `DirectSolver` verbs, for
  real `"SPD"` and no-pivot `"S"` matrices on GPU backends; the MadNLP
  extension's `SDSSolver` is a shell over it, or over a `DirectSolver` for
  `sds_pivoting = :bk` (T33, #153).
- Fused pipeline: blocked `L₁₁` inverses, gpu-scope fences through the
  `device_fence` hook, coalesced backward solve, and the `SDSTuning` fields
  `solve_workgroup`, `wide_host_min` and `graph_max_n` (opt-in CUDA-graph
  replay of refactorization and solve) (T34, #178).
- Fused pipeline: bitwise reproducible factorization (parent-pull extend-add,
  ordered wide-front tiles) in both modes; `deterministic_mode = 1` on
  `FusedSolver` selects the pull forward sweep; `memory_estimates` slots 13
  and 14 report the fused buffers (T35, #182).
- `solve_alg = "algo1"` on `DirectSolver`: the partitioned-inverse fused solve
  after the stock factorization (real `"SPD"`/`"HPD"`, `"S"`/`"H"` with
  `pivot_type = 'N'`, one right-hand side, CUDA/AMDGPU/oneAPI), and the
  `forward_progress(backend)` capability that gates the dependency-counter
  kernels (T36, #187).
- Tests of the MadNLP extension (`test/test_madnlp.jl`, #162, #176).
- The cuDSS 1.0 comparison on an H200 (`bench/comparison/`), with
  `solve_alg = "algo1"` rows, the 78k-bus condensed KKT system and a KKT-only
  overview (#190).

### Changed

- The `:ka` dense fallbacks launch allocation-free (static launch
  configurations); `impl = :auto` takes the vendor binding, then `:ka`, never
  `:generic`. On Metal, panel GEMM/SYRK under `:auto` move from MPS to the
  KernelAbstractions kernels (T32, #184).
- `update!` is public but no longer exported, since CUDA.jl 6.5 exports one
  too: write `SparseDirectSolver.update!` (#144, #185).
- A non-default `max_lu_nnz`, `host_nthreads` or `use_cuda_register_memory`
  warns once (stored, no effect); `schedule = "subtree+level"` raises
  `InvalidValueError` (#159, #185).
- `FusedSolver` also refuses backends without forward progress (Metal), next
  to the CPU backend (T36, #187).
- MadNLP `SDSSolver` with `sds_pivoting = :none` reports the inertia from the
  counts of positive and negative entries of `D` (T33, #153).
- The KernelAbstractions compat is `0.10.3` again (no cap at 0.10.5); the CPU
  allocation checks run in a child process per element type (#188, #189).

### Fixed

- CUDA-graph capture of ranges with cuSOLVER/cuBLAS calls: the vendor handles
  are bound to the capture stream (#174, #185).
- MadNLP `SDSSolver` can be built from a `SparseMatrixCSC` (MadNLP's CPU KKT
  systems) on the CPU backend (#175, #176).
- Flaky static-pivot refinement test on AMDGPU (#179, #178).
- A regime-B front wider than 256 under the fused solve raises
  `NotSupportedError` instead of failing an assertion (T36, #187).

## [0.2.0] and earlier

Internal versions, never released or registered; the history is in `TASKS.md`
and the git log.

[0.4.0]: https://github.com/exanauts/SparseDirectSolver.jl/releases/tag/v0.4.0
[0.3.0]: https://github.com/exanauts/SparseDirectSolver.jl/releases/tag/v0.3.0
