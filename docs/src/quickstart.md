# Quick start

## On the CPU backend

A [`DirectSolver`](@ref) is created from a sparse matrix, a structure string and
the triangle that is read. A `SparseMatrixCSC` is converted to CSR arrays on the
host and factored on the KernelAbstractions CPU backend.

```@example cpu
using SparseDirectSolver, SparseArrays, LinearAlgebra, Random
Random.seed!(666)

n = 500
A = sprand(n, n, 0.01); A = A * A' + n * I    # symmetric positive definite
b = rand(n); x = similar(b)

solver = DirectSolver(tril(A), "SPD", 'L')
execute!("analysis", solver, x, b)            # ordering + symbolic factorization
execute!("factorization", solver, x, b)
execute!("solve", solver, x, b)
getparam(solver, "info"), norm(A * x - b) / norm(b)
```

The analysis results can be inspected before the numeric phases run:

```@example cpu
getparam(solver, "lu_nnz"), getparam(solver, "nsuperpanels")
```

New values with the same pattern reuse the analysis:

```@example cpu
A2 = A + 2I
update!(solver, tril(A2))
execute!("refactorization", solver, x, b)
execute!("solve", solver, x, b)
norm(A2 * x - b) / norm(b)
```

## On CUDA

With CUDA.jl loaded, the solver takes a `CuSparseMatrixCSR` (or
`CuSparseMatrixCSC`) and right-hand sides in `CuArray`s; every numeric phase
runs on the device.

```@example cuda
using SparseDirectSolver, SparseArrays, LinearAlgebra, Random
using CUDA, CUDA.CUSPARSE
Random.seed!(666)

n = 2000
A = sprand(n, n, 0.002); A = A * A' + n * I
A_gpu = CuSparseMatrixCSR(tril(A))
b_gpu = CuVector(rand(n)); x_gpu = similar(b_gpu)

solver = DirectSolver(A_gpu, "SPD", 'L')
setparam!(solver, "ir_n_steps", 1)            # one step of iterative refinement
execute!("analysis", solver, x_gpu, b_gpu)
execute!("factorization", solver, x_gpu, b_gpu)
execute!("solve", solver, x_gpu, b_gpu)
norm(CuSparseMatrixCSR(A) * x_gpu - b_gpu) / norm(b_gpu)
```

## The `LinearAlgebra` layer

`cholesky`, `ldlt` and `lu` run the analysis and the factorization at once and
return the solver, which then acts as a factorization object. Solvers created
this way perform two steps of iterative refinement per solve.

```@example cuda
F = cholesky(A_gpu; view = 'L')
x_gpu = F \ b_gpu
logdet(F) ≈ logdet(cholesky(Symmetric(Matrix(A))))
```

```@example cuda
K = [A sprand(n, 50, 0.05); sprand(50, n, 0.05) -I]    # symmetric indefinite
K = (K + K') / 2
F = ldlt(CuSparseMatrixCSR(tril(K)); view = 'L')
getparam(F, "inertia")
```

`cholesky!`, `ldlt!` and `lu!` refactor in place with new values:
`cholesky!(F, A_gpu_new)`.
