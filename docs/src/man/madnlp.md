# MadNLP

Loading MadNLP.jl next to SparseDirectSolver.jl activates the MadNLP extension,
which provides the linear solver `SDSSolver` and its options `SDSSolverOptions`.
Both names are exported by SparseDirectSolver.jl but assigned by the extension,
so they are usable only once MadNLP is loaded (the same idiom as
`MadNLPGPU.CUDSSSolver`, which needs CUDA):

```julia
using MadNLP, SparseDirectSolver
using CUDA, MadNLPGPU                 # or AMDGPU: the KKT matrix must live on a GPU

result = madnlp(model; linear_solver = SDSSolver,
                kkt_system = MadNLP.SparseCondensedKKTSystem)
```

The extension is backend-agnostic: it accepts any device CSC matrix whose
`colPtr`, `rowVal` and `nzVal` are device vectors (`CuSparseMatrixCSC`,
`ROCSparseMatrixCSC`) and reads MadNLP's lower-triangular KKT matrix as the CSR
of the upper triangle (view `'U'`), exactly as `CUDSSSolver` does. `Float32` and
`Float64` are supported. The end-to-end runs on CUDA and AMD, including the
78k-bus ACOPF system, are recorded in
[`bench/PERFORMANCE-2026-10.md`](https://github.com/exanauts/SparseDirectSolver.jl/blob/main/bench/PERFORMANCE-2026-10.md)
and reproduced by the scripts of
[`bench/e2e/`](https://github.com/exanauts/SparseDirectSolver.jl/tree/main/bench/e2e).

## Options

The options are passed as keywords of `madnlp` (or in an `SDSSolverOptions`).

| option | default | meaning |
| :--- | :--- | :--- |
| `sds_factorization` | `:cholesky` | `:cholesky` factors the KKT matrix as `"SPD"` (the condensed-KKT default); `:ldl` as `"S"` without pivoting, `L D Lᵀ` in the identity pivot order, and reports the true inertia from the signs of `D`. |
| `sds_pivoting` | `:none` | `:ldl` only. `:none` runs the fused pipeline below; `:bk` routes factorization and solve through the stock in-front Bunch–Kaufman path of the handle API ([Symmetric indefinite systems](@ref)): more robust on ill-conditioned KKT systems, at the level-batched speed of [`execute!`](@ref). |
| `sds_pivot_threshold` | `nothing` | `:bk` only: the `pivot_threshold` of the in-front acceptance test (`nothing` keeps the package default `0.01`; cuDSS's default is `1.0`). |
| `sds_reordering` | `"algo4"` | the `reordering_alg` of the analysis: `"algo4"` nested dissection (METIS), `"algo3"` AMD, `"default"` the cost-model choice ([Parameters](parameters.md)). |
| `sds_user_perm` | `nothing` | a 1-based permutation of length `n` that replaces the reordering (`user_perm`), for instance a cached one; every solver instance otherwise recomputes its ordering. |
| `sds_tuning` | `nothing` | an [`SparseDirectSolver.Fused.SDSTuning`](@ref) bundle of performance knobs; `nothing` selects the measured per-architecture defaults of [`SparseDirectSolver.Fused.default_tuning`](@ref). |

With `sds_pivoting = :bk` the 2×2 pivot pairs of the `"S"` analysis are chosen
from the values present in the KKT matrix when the solver is built (see
`pivot_pairs` under [Parameters](parameters.md)); with `:none` the pairs are off, since
a factorization without pivoting never consumes them.

## What it runs

`SDSSolver` is a shell over an inner solver driven by `refactorize!`,
`solve!` and `getparam(inner, "inertia")`. With `sds_pivoting = :none` the
inner solver is a [`FusedSolver`](@ref SparseDirectSolver.FusedSolver), the fused pipeline of
`SparseDirectSolver.Fused` (`src/fused.jl`) over the symbolic analysis of a
regular [`DirectSolver`](@ref): one dependency-counter kernel per
refactorization that covers the whole elimination tree (the wide root fronts
either inside the kernel or on the vendor `potrf` path, chosen by the tuning),
per-front inverses of the diagonal blocks computed after each factorization,
and a partitioned-inverse solve in one dependency-counter sweep per direction.
This is the path that `cholesky_fused` measures on the [Performance](@ref) page;
tasks T24–T26 of the repository move the same kernels under the `execute!` of
`DirectSolver`. With `:bk` the inner solver is a `DirectSolver` with the same
tuning.

The inertia MadNLP asks for after every factorization is `(n, 0, 0)` for
`:cholesky` when the factorization succeeded; a failed factorization (a
non-positive pivot, or a sign flip inside a wide front with `:ldl`) is reported
as `(n - 2, 1, 1)`, so that MadNLP regularizes and factorizes again. With `:ldl`
the inertia is the `inertia` parameter of the inner solver: the signs of `D`
(`:none`) or, with `:bk`, the stock pivot statistics (2×2 blocks included).

## Tuning

Every performance knob of the fused pipeline lives in one immutable bundle, so
a platform's tuning is a single value; how each field was measured and the
per-architecture champions are in
[`bench/ARCH-TUNING.md`](https://github.com/exanauts/SparseDirectSolver.jl/blob/main/bench/ARCH-TUNING.md).

```julia
tuning = SparseDirectSolver.Fused.SDSTuning(; nd_nseps = 4)   # lower fill, ~2× the reordering time
madnlp(model; linear_solver = SDSSolver, sds_tuning = tuning,
       kkt_system = MadNLP.SparseCondensedKKTSystem)
```

The bundle and its defaults are documented on the
[Fused pipeline](@ref) page.
