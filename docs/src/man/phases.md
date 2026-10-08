# Phases

The handle API follows cuDSS: a [`DirectSolver`](@ref) holds the matrix, the
parameters and every intermediate result, and [`execute!`](@ref) runs one phase
on it, named by the same string as in CUDSS.jl.

| phase | what it does | needs |
| :--- | :--- | :--- |
| `"reordering"` | fill-reducing ordering on the host (AMD or nested dissection, `user_perm`), matching | — |
| `"symbolic_factorization"` | elimination tree, supernodes, schedule, factor layout, device maps; allocates the numeric storage and the solve workspace | `"reordering"` |
| `"analysis"` | `"reordering"` and `"symbolic_factorization"` | — |
| `"factorization"` | numeric factorization with the current values; sets `"info"` | analysis |
| `"refactorization"` | the same, reusing the analysis and the storage | a factorization |
| `"solve"` | ``X = \operatorname{op}(A)^{-1} B``, then `ir_n_steps` steps of refinement | a factorization |
| `"solve_fwd_perm"`, `"solve_fwd"`, `"solve_diag"`, `"solve_bwd"`, `"solve_bwd_perm"`, `"solve_refinement"` | the six parts of `"solve"`, bitwise equal to it when run in sequence | a factorization |
| `"solve_fwd_schur"`, `"solve_bwd_schur"` | the solve phases of the Schur complement mode | a factorization with `schur_mode = 1` |

The named wrappers [`analyze!`](@ref), [`factorize!`](@ref),
[`refactorize!`](@ref) and [`solve!`](@ref) are the corresponding `execute!`
calls.

## Typical sequence

```julia
solver = DirectSolver(A, "S", 'L')          # A: CSR on a backend, or a backend sparse matrix
setparam!(solver, "pivot_threshold", 0.01)  # configuration before the phase that reads it
execute!("analysis", solver, X, B)          # once per sparsity pattern
for k in 1:iterations
    update!(solver, A_k)                    # new values, same pattern
    execute!(k == 1 ? "factorization" : "refactorization", solver, X, B)
    getparam(solver, "info") == 0 || error("factorization failed")
    execute!("solve", solver, X, B)
end
```

## Synchronization and errors

The phases are asynchronous on GPU backends, as in cuDSS: `execute!` returns
once the work is queued, unless `asynchronous = false`. A failed factorization
does not throw: `getparam(solver, "info")` is `0` on success and otherwise the
column of the first failed pivot (a non-positive pivot for Cholesky; an exactly
zero pivot with `pivot_epsilon = 0` for LDLᵀ and LU, which perturb the other
tiny pivots and count them in `"npivots"`).

Running a phase before the phases it depends on raises a
[`FactorizationError`](@ref); an unknown phase string raises an `ArgumentError`.
The flag `"user_host_interrupt"` (a `Threads.Atomic{Bool}`) is polled between
launch groups and raises an [`InterruptedError`](@ref) when set.

## Logging

`SDS_LOG_LEVEL=1` (or [`SparseDirectSolver.set_log_level!`](@ref)`(1)`) prints a
summary of every phase; `2` adds the residual of every refinement step. With the
default level the messages are `@debug` records, visible with
`JULIA_DEBUG=SparseDirectSolver`.

## Schur complement mode

With `schur_mode = 1` and `user_schur_indices` (one 0/1 flag per row) set before
the analysis, the flagged rows and columns are ordered last and the
factorization stops before them. `getparam(solver, "schur_matrix")` returns
``S = A_{22} - A_{21} A_{11}^{-1} A_{12}``, dense or sparse (one triangle or the
full matrix), and `"solve_fwd_schur"`, `"solve_diag"` and `"solve_bwd_schur"`
solve with the factored block around a solve with ``S`` done by the caller.
See [`execute!`](@ref) for the vector layout.

## Uniform batches

A matrix whose values hold `nbatch` sets of values for one pattern (a vector of
`nbatch · nnz` entries or an `nnz × nbatch` matrix) gives a batched solver: one
analysis, one factor per member, and per-member `"info"`, `"inertia"`,
`"npivots"` and `"pivot_stats"`. `"ubatch_index"` and `"ubatch_mask"` restrict
the factorization and the solves to some of the members. See
[`DirectSolver`](@ref).
