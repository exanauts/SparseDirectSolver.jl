# What to expect

This page is about version 0.4.0, which follows 0.3.0, the first registered
release. It says what the release covers, how fast and how accurate it is next
to cuDSS, and what it does not do yet. The other pages of the manual have the
details.

## What the release covers

* **Structures**: `"SPD"`, `"HPD"` (LLᵀ/LLᴴ), `"S"`, `"H"` (LDLᵀ/LDLᴴ with
  in-front Bunch–Kaufman 1×1/2×2 pivoting) and `"G"` (LDU with threshold
  partial pivoting inside each front).
* **Phases and parameters**: the cuDSS phase strings and parameter names through
  [`DirectSolver`](@ref) and [`execute!`](@ref), the solve sub-phases, iterative
  and FGMRES refinement, uniform batches, Schur complement mode, matching and
  scaling, and a `LinearAlgebra` layer (`cholesky`, `ldlt`, `lu`, `ldiv!`, `\`).
* **Types**: element types `Float32`, `Float64`, `ComplexF32`, `ComplexF64`;
  index types `Int32`, `Int64`.
* **Backends**: CUDA, AMDGPU, oneAPI, Metal (`Float32` only) and the
  KernelAbstractions CPU backend ([Backends](@ref)).
* **The fused pipeline** [`FusedSolver`](@ref SparseDirectSolver.FusedSolver):
  one dependency-counter factorization kernel and a partitioned-inverse solve,
  for real `"SPD"` and no-pivot `"S"` matrices with one right-hand side, on
  CUDA, AMDGPU and oneAPI. Its factorization is bitwise reproducible;
  `deterministic_mode = 1` makes its solve so too.
* **`solve_alg = "algo1"`** on [`DirectSolver`](@ref): the partitioned-inverse
  solve of the fused pipeline after the stock factorization (set before the
  analysis; real `"SPD"`/`"HPD"`, `"S"`/`"H"` with `pivot_type = 'N'`, one
  right-hand side, CUDA/AMDGPU/oneAPI). The default stays the level-batched
  sweeps.
* **[`KKTSolver`](@ref)**: the front end for interior-point KKT systems
  (refactorization, one right-hand side, inertia), which runs the fused
  pipeline where the backend allows it and the stock pipeline otherwise or
  with Bunch–Kaufman pivoting ([KKT solver](../lib/kkt.md)). MadNLP 0.12
  binds it as `SDSSolver` in its own packages ([MadNLP](@ref)).

KernelAbstractions 0.10 (≥ 0.10.3) is required, hence CUDA.jl ≥ 6.5, AMDGPU.jl ≥
2.10, oneAPI.jl ≥ 2.10 and Metal.jl ≥ 1.11, on Julia 1.13 or later.

## Speed against cuDSS

SparseDirectSolver.jl is slower than cuDSS on most matrices, and how much
depends on the matrix class. On an NVIDIA H200 against cuDSS 1.0.0 (geometric
means of the time ratio, [Performance](@ref)): on the condensed KKT systems of
MadNLP the fused pipeline refactorizes at 1.48× and solves at 1.62× the cuDSS
time, the stock Cholesky of [`DirectSolver`](@ref) at 2.44× and 2.83×
(`solve_alg = "algo1"` brings its solve to 1.41×), and LDLᵀ with the default
pivoting at 6.0× and 4.2×. On the SuiteSparse and Laplacian matrices, whose
wide fronts go through the vendor dense calls, the stock Cholesky
refactorization is 9.5× behind (the fused pipeline 1.9×). On the 78k-bus
condensed ACOPF system (n = 674 562) the fused pipeline refactorizes in
16.6 ms against 10.4 ms for cuDSS and solves in 2.34 ms against 1.63 ms. The
symbolic analysis is faster than cuDSS's on the KKT systems for the Cholesky
features (0.40×) and on par for LDLᵀ (1.01×). LU is 13.8× behind on
factorization and 15.2× on refactorization (two matrices). The numbers were
measured on one GPU architecture (sm_90); other GPUs, AMD in particular, were
timed less.

## Accuracy

* **Pivoting.** The LDLᵀ/LDLᴴ factorization uses Bunch–Kaufman inside each
  front but does not delay pivots to the parent: a pivot that fails in its
  front is perturbed to `pivot_epsilon` (static pivoting, counted in
  `"npivots"`). Iterative refinement (`ir_n_steps`, or
  `ir_mode = "fgmres"` after `using Krylov`) recovers the accuracy; badly
  scaled KKT systems also need the matching and scaling of
  `matching_alg = "algo5"` ([Symmetric indefinite systems](@ref)).
* **No pivoting.** The fused pipeline and `solve_alg = "algo1"` factor `"S"`
  matrices in the identity pivot order (`pivot_type = 'N'`; a pivot below
  `pivot_epsilon` is replaced, counted in `"npivots"`), which relies on the
  regularization of the interior-point method or on the matching and scaling
  of `matching_alg = "algo5"`; [`KKTSolver`](@ref) (and through it MadNLP's
  `SDSSolver`) reports the inertia from the signs of ``D``, a replaced pivot
  counted with the sign it was given. Plain iterative refinement
  (`ir_n_steps`, the `KKTSolver` keyword of the same name) runs on both
  pipelines and recovers the accuracy that a replaced pivot costs; the fused
  pipeline has no `ir_tol` early exit and no FGMRES refinement.
* **`"algo1"` residuals.** The partitioned-inverse solve multiplies by explicit
  inverses of the diagonal blocks of each front. On ill-conditioned blocks its
  residual is larger than that of the level sweeps: 1.1–3.3× on the condensed
  KKT systems, 35–38× on bcsstk38. Iterative refinement removes the
  difference; whether one step is enough everywhere has not been measured, so
  `"algo1"` is not the default.
* **LU** without matching fails on hard unsymmetric matrices (rajat21, TSOPF)
  in both solvers; use `matching_alg`.

## What is not there

Each item raises [`NotSupportedError`](@ref) or warns once when requested;
nothing falls back silently ([Parameters](parameters.md)).

* Non-uniform batches (matrices of different patterns in one solver): no API
  yet.
* Hybrid host memory and hybrid execution (`hybrid_memory_mode`,
  `hybrid_device_memory_limit`, `hybrid_execute_mode`): warn once, no effect.
* Mixed precision, `factor_precision` other than the input precision:
  `NotSupportedError` at the analysis.
* `schedule = "syncfree"`: `NotSupportedError` at the analysis.
* `max_lu_nnz`: stored and returned by `getparam`, not enforced (warns once),
  likewise `host_nthreads` and `use_cuda_register_memory`.
* Bunch–Kaufman pivoting under `solve_alg = "algo1"`, and every other
  combination outside its scope: `NotSupportedError`.
* The fused factorization on [`DirectSolver`](@ref) (it is reached through
  [`FusedSolver`](@ref SparseDirectSolver.FusedSolver) only) and CUDA-graph
  capture of the stock refactorization and solve. Graph replay exists in the
  fused pipeline only, opt-in through `SDSTuning.graph_max_n`.
* Delayed pivots, export and import of the nested-dissection partition tree
  (`user_nd_partition_tree` raises), more than one device (`device_count`).

## Backend caveats

* **CPU backend**: for development and testing; it is not tuned for speed and refuses the fused pipeline and `"algo1"`.
* **CUDA**: the backend measured most; the performance numbers above are from
  it.
* **AMDGPU**: tested in CI on every pull request, timed far less than CUDA.
* **oneAPI**: tested in CI on Aurora (Intel Data Center GPU Max 1550). Every
  oneMKL call synchronizes the host, and the LAPACK steps (`potrf`, `getrf`)
  run on the KernelAbstractions kernels, which are slow on wide fronts:
  roughly 4× slower than on an H200 in the refactorization and 2.4× in the
  solve, much more on large fronts (see the Performance page).
* **Metal**: `Float32` only, tested by hand on an M4 Max, no CI. No fused
  pipeline and no `"algo1"`.

## Reporting problems

Please open an issue on
[GitHub](https://github.com/exanauts/SparseDirectSolver.jl/issues) with the
backend, the element type, the structure and, if possible, the matrix. The
package is developed in agent-driven sessions (see the repository's
`AGENTS.md`); issues the agents open carry the `found-by-agent` label, and
issues from users are welcome next to them.
