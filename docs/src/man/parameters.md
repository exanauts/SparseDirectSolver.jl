# Parameters

Parameters are set with [`setparam!`](@ref) and read with [`getparam`](@ref)
(or [`getparam!`](@ref) into a preallocated buffer), with the parameter strings
of CUDSS.jl. They come in three groups.

* **Configuration parameters** ([`CONFIG_PARAMETERS`](@ref), cuDSS's
  `cudssConfigParam_t`): algorithm choices and tolerances, read by the next
  phase that uses them.
* **Data parameters** ([`DATA_PARAMETERS`](@ref) and
  [`CUDSS08_DATA_PARAMETERS`](@ref), cuDSS's `cudssDataParam_t`): inputs provided
  by the user (`user_perm`, `user_schur_indices`, `user_host_interrupt`, …) and
  outputs computed by the solver (`info`, `inertia`, `diag`, `perm_row`, …).
* **Extensions** ([`EXTRA_PARAMETERS`](@ref)): parameters cuDSS does not have,
  such as `pivot_sign`, `pivot_stats`, `ir_mode` and `pivot_pairs`.

Every parameter is validated when it is set: an unknown name raises an
`ArgumentError`, a bad value an [`InvalidValueError`](@ref), and a parameter
that exists in cuDSS but is not implemented (the hybrid memory and multi-GPU
modes, for example) a [`NotSupportedError`](@ref).

```@example params
using SparseDirectSolver
opts = Options(ir_n_steps = 2, pivot_threshold = 0.1)
getparam(opts, "ir_n_steps"), getparam(opts, "reordering_alg")
```

## Configuration parameters

The fields of [`Options`](@ref) with their defaults are listed in its
docstring. The ones that matter most in practice:

| parameter | values | effect |
| :--- | :--- | :--- |
| `reordering_alg` | `"default"`, `"algo1"`–`"algo5"` | automatic, AMD (`"algo3"`), nested dissection (`"algo4"`, needs `using Metis`), natural (`"algo5"`); see [`SparseDirectSolver.ReorderingAlg`](@ref) |
| `factorization_alg` | `"default"`, `"algo1"`, `"algo2"` | automatic, small-front kernels only, vendor dense calls for large fronts |
| `matching_alg` | `"default"` (none), `"algo1"`–`"algo6"` | MC64 jobs 1–5 (`"algo6"`: automatic) before the ordering |
| `pivot_type` | `'A'`, `'N'`, `'D'`, `'L'`, `'B'` | automatic, none, diagonal, local block, Bunch–Kaufman; global pivoting (`'C'`, `'R'`) is not supported |
| `pivot_threshold` | `0.0`–`1.0`, default `0.01` | acceptance threshold of a pivot |
| `pivot_epsilon` | default `1e-13` (`Float64`), `1e-5` (`Float32`) | tiny pivots are perturbed to this size |
| `solve_mode` | `0`, `1`, `2` | solve with ``A``, ``A^T`` or ``A^H`` |
| `ir_n_steps`, `ir_tol` | integer, real | iterative refinement steps and early-exit tolerance |
| `ir_mode` | `"ir"`, `"fgmres"` | plain refinement or FGMRES (needs `using Krylov`) |
| `deterministic_mode` | `0`, `1` | `1` forbids the atomic forward solve |
| `schur_mode` | `0`, `1` | Schur complement mode |

## Data parameters

The outputs and when they become available are tabulated in the docstring of
[`getparam`](@ref). For a uniform batch the per-factorization outputs are
vectors with one entry per member.

## Caching an ordering

The ordering is usually the most expensive part of the analysis. Store the
permutation and the nested-dissection partition tree of one analysis and pass
them to a later one (another process, the same sparsity pattern), as with
cuDSS:

```julia
analyze!(solver)
perm = getparam(solver, "perm_reorder_row")
tree = getparam(solver, "nd_partition_tree")   # 2^nd_nlevels - 1 sizes, cuDSS encoding

later = DirectSolver(A, "SPD", 'L')
setparam!(later, "user_perm", perm)
setparam!(later, "user_nd_partition_tree", tree)  # checked against perm
analyze!(later)                                   # no ordering; same supernodes and lu_nnz
```

The supernode partition, `lu_nnz` and the schedule depend only on the
elimination tree of the permutation, so the second analysis reproduces the
first exactly. The tree is validated (sizes, and every column's dependencies
inside its node's ancestors) and is optional: `user_perm` alone gives the same
analysis. The 2×2 pivot pairs that `"S"`/`"H"` choose at analysis are not part
of the encoding and are not applied under a `user_perm`.

## Differences from cuDSS

* `"info"` reports the original column of the first failed pivot.
* `getparam!` replaces the set-buffer-then-get protocol of cuDSS for vector
  outputs.
* `pivot_sign` chooses the sign of the perturbation of each row, so that a
  perturbed KKT system keeps the inertia an interior-point method expects.
* `pivot_stats` returns `(npos, nneg, nzero, nperturbed, n2x2)` in one read.
* `nd_partition_tree` is available after the symbolic factorization (cuDSS:
  after the reordering) and exists for every ordering, not only nested
  dissection.
