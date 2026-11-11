# MadNLP

The MadNLP binding of SparseDirectSolver.jl lives in MadNLP.jl (version 0.12
and later), not in this package: the linear solver `SDSSolver` is a marker
exported by `MadCoreGPU`, implemented by its extension `MadCoreGPUSDSExt`,
which loads with SparseDirectSolver.jl. The extension is a shell over
[`KKTSolver`](@ref) ([KKT solver](../lib/kkt.md)), the MadNLP-independent front
end of this package. SparseDirectSolver.jl 0.3 shipped a MadNLP extension of
its own (for MadNLP 0.11); it was removed in 0.4.0.

## Using it

`SDSSolver` is the default sparse linear solver of MadNLP's AMD, Intel and
Apple entry points, which load SparseDirectSolver.jl themselves:

```julia
using rocMadNLP                       # AMD; oneMadNLP for Intel, mtlMadNLP for Apple
using ExaModels, AMDGPU

nlp = ExaModel(model; backend = ROCBackend())   # a JuMP model, loaded on the GPU
result = madnlp(nlp)                            # SDSSolver is the default
```

The NVIDIA entry point `cuMadNLP` defaults to cuDSS; with SparseDirectSolver.jl
loaded, `SDSSolver` is selected explicitly:

```julia
using cuMadNLP, SparseDirectSolver    # SparseDirectSolver loads MadCoreGPUSDSExt
using ExaModels, CUDA
const SDSSolver = cuMadNLP.MadCoreGPU.SDSSolver

nlp = ExaModel(model; backend = CUDABackend())
result = madnlp(nlp; linear_solver = SDSSolver)
```

The installation of the entry points (some are not registered yet) and the
choice of the KKT system and preset are described in MadNLP's documentation.
MadNLP's default preset on a GPU asks for an `L D Lᵀ` without pivoting
(`sds_factorization = :ldl`, `sds_pivoting = :none`) with a pivot bound
(`sds_pivot_epsilon`), the counterpart of cuDSS's pivot epsilon in the same
preset. The binding reads MadNLP's lower-triangular KKT matrix
(a device CSC matrix) as the CSR of its upper triangle, exactly as the cuDSS
binding does; `Float32` and `Float64` are supported (`Float32` only on Metal
and on Intel GPUs without `Float64`).

## Options

The `sds_*` options belong to MadNLP and are documented there; they are passed
as keywords of `madnlp`. Each one maps to a keyword of [`KKTSolver`](@ref):

| MadNLP option | `KKTSolver` keyword | meaning |
| :--- | :--- | :--- |
| `sds_factorization` | `factorization` | `:cholesky` (`"SPD"`) or `:ldl` (`"S"`, inertia from the signs of `D`) |
| `sds_pivoting` | `pivoting` | `:ldl` only: `:none` (pivot order of the analysis) or `:bk` (in-front Bunch–Kaufman, stock pipeline, [Symmetric indefinite systems](@ref)) |
| `sds_pivot_threshold` | `pivot_threshold` | `:bk` only: the `pivot_threshold` of the acceptance test (`nothing` keeps the package default `0.01`; cuDSS's default is `1.0`) |
| `sds_pivot_epsilon` | `pivot_epsilon` | the `pivot_epsilon` below which a pivot is replaced, on both pipelines (`nothing`: `1e-13` for `Float64`, `1e-5` for `Float32`) |
| `sds_reordering` | `reordering` | the `reordering_alg` of the analysis: `"algo4"` nested dissection (METIS), `"algo3"` AMD, `"default"` the cost-model choice ([Parameters](parameters.md)) |
| `sds_user_perm` | `user_perm` | a 1-based permutation of length `n` that replaces the reordering, for instance a cached one; every solver instance otherwise recomputes its ordering |
| `sds_tuning` | `tuning` | an [`SDSTuning`](@ref SparseDirectSolver.Fused.SDSTuning) bundle of performance settings; `nothing` selects the measured per-architecture defaults of [`default_tuning`](@ref SparseDirectSolver.Fused.default_tuning) |
| `sds_fused` | `fused` | `true` the fused pipeline, `false` the stock one, `nothing` the default of the backend |

The sizes of the primal and dual blocks, which MadNLP's KKT system knows, give
the `pivot_sign` of `KKTSolver` (`1` for the primal and `-1` for the dual
rows): the sign a replaced pivot takes in an `L D Lᵀ` without pivoting, on
both pipelines.

`KKTSolver` has two keywords for the setting of MadNLP's cuDSS preset (an
`L D Lᵀ` with matching and pivot epsilon, analyzed on the first KKT values),
whatever MadNLP's binding exposes of them:

* `matching` (default `"none"`): the `"matching_alg"` of the analysis, `:ldl`
  only, on both pipelines. With `"algo5"` and `"algo6"` the matrix is scaled
  and the 2×2 pivot pairs of the ordering come from the cycles of the scaled
  symmetric matching, with `pivoting = :none` as with `:bk`.
* `deferred_analysis` (default `false`): the constructor checks the options
  but neither analyzes nor factorizes; the first `factorize!` analyzes the
  values present then (the counterpart of the cuDSS binding's deferred
  analysis), the later ones refactorize.

`ir_n_steps` (default `nothing`, no refinement) follows every solve with that
many steps of iterative refinement inside the solver, on both pipelines, the
residual taken with the KKT values MadNLP passed (not the scaled ones of a
matching, not the replaced pivots); it is the counterpart of cuDSS's
`ir_n_steps`. MadNLP refines the solution of its KKT system itself, so it is
off by default; one step recovers most of the accuracy that static pivot
replacement costs.

Without matching, with `pivoting = :bk` the 2×2 pivot pairs of the `"S"`
analysis are chosen from the values present in the KKT matrix at the analysis
(see [Pivot pairs](@ref) under Parameters); with `:none` the pairs are off,
since a factorization without pivoting never consumes them.

## What runs underneath

[`KKTSolver`](@ref) chooses the pipeline;
[`SparseDirectSolver.pipeline`](@ref) tells which one runs.

* **Fused pipeline** (the default for `pivoting = :none` on CUDA, AMDGPU and
  oneAPI, the backends with forward progress between workgroups, with or
  without matching, `pivot_epsilon` and `pivot_sign`): the inner
  solver is a [`FusedSolver`](@ref SparseDirectSolver.FusedSolver) over the
  symbolic analysis of a regular [`DirectSolver`](@ref). One
  dependency-counter kernel per refactorization covers the whole elimination
  tree (the wide root fronts of an LDLᵀ always inside the kernel, since a
  saddle-point root mixes pivot signs; those of a Cholesky inside the kernel
  or on the vendor `potrf` path, chosen by the tuning), the inverses of the
  diagonal blocks of the fronts are computed after each factorization, and the
  solve is a partitioned-inverse sweep, one dependency-counter kernel per
  direction. This is the path that `cholesky_fused` measures on the
  [Performance](@ref) page.
* **Stock pipeline** (`pivoting = :bk`, `fused = false`, and every
  factorization on the KernelAbstractions CPU backend and on Metal): the inner
  solver is a [`DirectSolver`](@ref) with the same tuning, driven through
  `refactorize!` and `solve!` at the level-batched speed of
  [`execute!`](@ref). Cholesky, `L D Lᵀ` without pivoting and Bunch–Kaufman
  `L D Lᵀ` all run on it.

The inertia after each factorization comes from
[`SparseDirectSolver.inertia`](@ref): for `:ldl` from the signs of `D` (2×2
blocks included), for `:cholesky` `(n, 0, 0)`. It is valid when
`getparam(K, "info")` is `0`; MadNLP's binding reports a failed factorization
as an indefinite inertia, so that MadNLP regularizes and factorizes again.

## Tuning

Every performance knob of the fused pipeline lives in one immutable bundle, so
a platform's tuning is a single value; how each field was measured and the
per-architecture champions are in
[`bench/ARCH-TUNING.md`](https://github.com/exanauts/SparseDirectSolver.jl/blob/main/bench/ARCH-TUNING.md).

```julia
using SparseDirectSolver: SDSTuning
tuning = SDSTuning(; nd_nseps = 4)   # lower fill, ~2× the reordering time
madnlp(nlp; sds_tuning = tuning)
```

The bundle and its defaults are documented on the
[Fused pipeline](../lib/fused.md) page.

## End-to-end records

The end-to-end MadNLP runs on CUDA and AMD, including the 78k-bus ACOPF
system, are recorded in
[`bench/PERFORMANCE-2026-10.md`](https://github.com/exanauts/SparseDirectSolver.jl/blob/main/bench/PERFORMANCE-2026-10.md).
They were run with the removed extension (SparseDirectSolver.jl 0.3 with
MadNLP 0.11), and the scripts of
[`bench/e2e/`](https://github.com/exanauts/SparseDirectSolver.jl/tree/main/bench/e2e)
still need that combination until they are ported to MadNLP 0.12.
