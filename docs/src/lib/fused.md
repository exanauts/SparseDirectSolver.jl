# Fused pipeline

[`FusedSolver`](@ref SparseDirectSolver.FusedSolver) runs the fused
factorization and partitioned-inverse solve of `SparseDirectSolver.Fused`
(`src/fused.jl`) behind the verbs of [`DirectSolver`](@ref): the same
[`execute!`](@ref) phases, [`setparam!`](@ref), [`getparam`](@ref) and
[`update!`](@ref), for one right-hand side, on a GPU backend, for `"SPD"` and
no-pivot `"S"` real matrices. It is the inner solver of the MadNLP extension's
`SDSSolver` ([MadNLP](@ref)); [Performance](@ref) says when to use it.

```julia
using SparseDirectSolver, CUDA, CUDA.CUSPARSE

Ad = CuSparseMatrixCSR(tril(A))       # A: SPD, on the host
F = FusedSolver(Ad, "SPD", 'L')
analyze!(F)                           # ordering, symbolic analysis, the fused plans
factorize!(F)                         # the fused factorization
bd = CuVector(b)
x = F \ bd                            # partitioned-inverse solve
getparam(F, "info")                   # 0 on success

Ad.nzVal .*= 2                        # new values in the same buffer (the IPM loop)
refactorize!(F)
solve!(F, x, bd)
```

```@docs
SparseDirectSolver.FusedSolver
```

The performance knobs of the pipeline live in one bundle, a platform's tuning
being a single value; how each field was measured and the per-architecture
champions are in
[`bench/ARCH-TUNING.md`](https://github.com/exanauts/SparseDirectSolver.jl/blob/main/bench/ARCH-TUNING.md).
`default_tuning` gives oneAPI, Metal and any other backend that reports no CUDA
thread count per multiprocessor the AMD profile.

```@docs
SparseDirectSolver.Fused.SDSTuning
SparseDirectSolver.Fused.default_tuning
SparseDirectSolver.Fused.resolve_tuning
```
