# Migrating from CUDSS.jl

The handle API keeps the names of the parameters and phases of CUDSS.jl; the
functions are renamed after Julia conventions.

| CUDSS.jl | SparseDirectSolver.jl |
| :--- | :--- |
| `CudssSolver(A, structure, view)` | [`DirectSolver`](@ref)`(A, structure, view)` |
| `cudss(phase, solver, x, b)` | [`execute!`](@ref)`(phase, solver, x, b)` |
| `cudss_set(solver, name, value)` | [`setparam!`](@ref)`(solver, name, value)` |
| `cudss_get(solver, name)` | [`getparam`](@ref)`(solver, name)` (or [`getparam!`](@ref) into a buffer) |
| `cudss_update(solver, A)` | [`update!`](@ref)`(solver, A)` |
| `CudssMatrix`, `CudssData`, `CudssConfig` | [`CSR`](@ref), [`MatrixDescriptor`](@ref), [`Options`](@ref) |
| `cholesky`, `ldlt`, `lu` on `CuSparseMatrixCSR` | the same functions, see [LinearAlgebra interface](@ref) |

Structures (`"G"`, `"S"`, `"H"`, `"SPD"`, `"HPD"`), views (`'L'`, `'U'`, `'F'`),
phase strings and parameter strings are identical, so a port is mostly a
renaming:

```julia
# CUDSS.jl
solver = CudssSolver(A_gpu, "S", 'L')
cudss_set(solver, "pivot_threshold", 0.01)
cudss("analysis", solver, x_gpu, b_gpu)
cudss("factorization", solver, x_gpu, b_gpu)
inertia = cudss_get(solver, "inertia")
cudss("solve", solver, x_gpu, b_gpu)

# SparseDirectSolver.jl
solver = DirectSolver(A_gpu, "S", 'L')
setparam!(solver, "pivot_threshold", 0.01)
execute!("analysis", solver, x_gpu, b_gpu)
execute!("factorization", solver, x_gpu, b_gpu)
inertia = getparam(solver, "inertia")
execute!("solve", solver, x_gpu, b_gpu)
```

## Differences

* cuDSS parameters that are not implemented (hybrid memory and execution
  modes, multiple devices, non-uniform batches) raise a
  [`NotSupportedError`](@ref) instead of being ignored.
* `"factorization_alg"` and `"solve_alg"` keep their spellings but select this
  package's own algorithms (see [`SparseDirectSolver.FactorizationAlg`](@ref)
  and [`SparseDirectSolver.SolveAlg`](@ref)).
* Global pivoting (`pivot_type` `'C'`/`'R'`) is not supported: pivots are chosen
  inside each front, and failed pivots are perturbed rather than delayed.
* Extensions beyond cuDSS: `pivot_sign`, `pivot_stats`, `ir_mode = "fgmres"`,
  `pivot_pairs`, `amalgamation`, `schedule` ([`EXTRA_PARAMETERS`](@ref)).
* The matrix and the vectors may live on any KernelAbstractions backend, not
  only CUDA.
