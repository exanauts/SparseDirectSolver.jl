# KKT solver

[`KKTSolver`](@ref) is the front end for the linear systems of an
interior-point method: one real symmetric matrix with a fixed sparsity
pattern, factorized again with new values at every iteration, solved for one
right-hand side at a time, and the inertia read after each factorization. It
chooses the pipeline (the fused pipeline of
[`FusedSolver`](@ref SparseDirectSolver.FusedSolver) where the backend runs it,
else the stock [`DirectSolver`](@ref)), sets the parameters of the chosen
factorization and drives the phases. The constructor runs the analysis and the
first factorization (with `deferred_analysis = true` the first `factorize!`
does, on the values present then); afterwards the loop calls only
[`factorize!`](@ref factorize!(::KKTSolver)),
[`solve!`](@ref solve!(::KKTSolver, ::AbstractVector, ::AbstractVector)) and
[`inertia`](@ref SparseDirectSolver.inertia). MadNLP's SparseDirectSolver
binding is built on it ([MadNLP](@ref)).

```julia
using SparseDirectSolver, SparseArrays, CUDA, CUDA.CUSPARSE

# K: a symmetric quasi-definite KKT matrix on the host, its lower triangle on the GPU
Kd = CuSparseMatrixCSC(tril(K))
S = KKTSolver(Kd, 'L'; factorization = :ldl)   # analysis and first factorization
SparseDirectSolver.pipeline(S)                 # :fused on CUDA, AMDGPU and oneAPI

Kd.nzVal .= new_values                         # same pattern, the next iteration
factorize!(S)
getparam(S, "info") == 0 || error("factorization failed")
npos, nzero, nneg = SparseDirectSolver.inertia(S)
xb = CuVector(b)
solve!(S, xb)                                  # xb holds b on entry, x on return
```

On the CPU backend and on Metal the same calls run the stock pipeline;
`pivoting = :bk` selects the Bunch–Kaufman pivoting of the stock `L D Lᵀ` on
every backend ([Symmetric indefinite systems](@ref)), and `fused = false`
forces the stock pipeline. The fused `L D Lᵀ` has no matching and replaces no
pivot, so `fused = nothing` also selects the stock pipeline for an `L D Lᵀ`
with `matching` (the `"matching_alg"` of the analysis, `"algo5"` and `"algo6"`
also giving the 2×2 pivot pairs of the ordering), `pivot_epsilon` or
`pivot_sign`, and `fused = true` refuses them with a
[`NotSupportedError`](@ref). Matching and the pivot pairs are decided from the
values of the analysis: when the solver is built before the matrix holds its
first values, as an interior-point method builds it, pass
`deferred_analysis = true`.

```julia
S = KKTSolver(Kd, 'L'; factorization = :ldl, matching = "algo6", pivot_epsilon = 1e-10,
              deferred_analysis = true)        # options checked, nothing analyzed yet
Kd.nzVal .= first_values
factorize!(S)                                  # analysis (matching, pairs) and first factorization
Kd.nzVal .= new_values
factorize!(S)                                  # refactorization
```

The performance settings are an
[`SDSTuning`](@ref SparseDirectSolver.Fused.SDSTuning) passed as `tuning`
([Fused pipeline](fused.md)).

`inertia` and `pipeline` are public but not exported (MadNLP has an `inertia`
of its own): write `SparseDirectSolver.inertia` or
`using SparseDirectSolver: inertia`.

```@docs
KKTSolver
factorize!(::KKTSolver)
solve!(::KKTSolver, ::AbstractVector, ::AbstractVector)
SparseDirectSolver.inertia
SparseDirectSolver.pipeline
```
