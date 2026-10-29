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
`ArgumentError` and a bad value an [`InvalidValueError`](@ref). A parameter
that exists in cuDSS but is not implemented is handled in one of three ways
(see below).

```@example params
using SparseDirectSolver
opts = Options(ir_n_steps = 2, pivot_threshold = 0.1)
getparam(opts, "ir_n_steps"), getparam(opts, "reordering_alg")
```

## How unimplemented parameters are handled

| handling | parameters |
| :--- | :--- |
| raise [`NotSupportedError`](@ref) | `device_count` other than `1`, `device_indices`, `comm_device`, `comm_host` (when set or read); `solve_alg = "algo1"` (T24–T26, at the solve), `user_nd_partition_tree` (T30) and `schedule = "syncfree"` (at the analysis) |
| warn once and ignore | a non-default `hybrid_memory_mode`, `hybrid_device_memory_limit` or `hybrid_execute_mode` (T27); `reordering_alg` `"algo1"`/`"algo2"`, which fall back to the symmetric-pattern ordering with in-front pivoting |
| accept and ignore silently | `max_lu_nnz`, `host_nthreads`, `use_cuda_register_memory`; `schedule = "subtree+level"`, which behaves as `"auto"` |

## Configuration parameters

The fields of [`Options`](@ref) with their defaults are listed in its
docstring. The ones that matter most in practice:

| parameter | values | effect |
| :--- | :--- | :--- |
| `reordering_alg` | `"default"`, `"algo1"`–`"algo5"` | automatic (AMD or ND, whichever has the lower flops + 1e8 × supernodal schedule depth), AMD (`"algo3"`), nested dissection (`"algo4"`), natural (`"algo5"`); see [`SparseDirectSolver.ReorderingAlg`](@ref) |
| `factorization_alg` | `"default"`, `"algo1"`, `"algo2"` | automatic, small-front kernels only, vendor dense calls for large fronts |
| `matching_alg` | `"default"` (none), `"algo1"`–`"algo6"` | MC64 jobs 1–5 (`"algo6"`: automatic) before the ordering |
| `pivot_type` | `'A'`, `'N'`, `'D'`, `'L'`, `'B'` | automatic, none, diagonal (1×1 only), local block, Bunch–Kaufman; `'A'`, `'B'` and `'L'` all select Bunch–Kaufman for LDLᵀ/LDLᴴ and threshold row pivoting inside the fully-summed block for LU; global pivoting (`'C'`, `'R'`) raises `NotSupportedError` |
| `pivot_threshold` | `0.0`–`1.0`, default `0.01` | acceptance threshold of a pivot |
| `pivot_epsilon` | default `1e-13` (`Float64`), `1e-5` (`Float32`) | tiny pivots of LDLᵀ/LDLᴴ/LU are perturbed to this size; it is also the Cholesky threshold: a non-negative pivot up to `pivot_epsilon` is raised to it (issue #127), a negative or NaN one still fails |
| `pivot_pairs` | `"default"`, `"all"`, `"none"` | 2×2 pivot pairs chosen at the analysis of `"S"`/`"H"`: rows whose pivot is structurally zero, every candidate, or none; see [Pivot pairs](@ref) |
| `pivot_pair_tolerance` | real `≥ 0`, default `1e-6` | relative diagonal size below which a row is a 2×2 candidate |
| `solve_mode` | `0`, `1`, `2` | solve with ``A``, ``A^T`` or ``A^H`` |
| `solve_alg` | `"default"`, `"algo1"` | level-batched sweeps; `"algo1"` (partitioned-inverse blocks) raises `NotSupportedError` until T24–T26 |
| `ir_n_steps`, `ir_tol` | integer, real | iterative refinement steps and early-exit tolerance |
| `ir_mode` | `"ir"`, `"fgmres"` | plain refinement or FGMRES (needs `using Krylov`) |
| `deterministic_mode` | `0`, `1` | `1` forbids the atomic forward solve |
| `schur_mode` | `0`, `1` | Schur complement mode |
| `nd_nlevels`, `nd_ubfactor`, `nd_nseps`, `nd_seed` | integers (`nd_nlevels` default `10`, the others `-1` for the library default) | minimum nested-dissection levels, and METIS's imbalance factor, separators per split and seed |
| `use_superpanels` | `0`, `1` | supernode amalgamation off or on |
| `amalgamation` | `NamedTuple` of `max_width`, `zero_fraction`, `min_width` | limits of the relaxed amalgamation |
| `schedule` | `"auto"`, `"subtree+level"`, `"syncfree"` | factorization schedule (see the table above for the last two) |
| `factor_precision` | `nothing`, `Float32`, `Float64` | precision of the factors; only the input precision is implemented, another one raises `NotSupportedError` at the analysis |
| `ubatch_size`, `ubatch_index`, `ubatch_mask` | integers, 0/1 flags | uniform batches: number of members (deduced from the values by default), one member or `-1` for all, a per-member mask |

The details of each are in the [`Options`](@ref) docstring and in
[`setparam!`](@ref).

### Pivot pairs

For the symmetric indefinite structures `"S"` and `"H"`, `pivot_pairs =
"default"` orders a row whose diagonal is zero or negligible (below
`pivot_pair_tolerance` times the largest entry of its row) and whose pivot would
be structurally zero together with a 2×2 pivot partner, so that the in-front
Bunch–Kaufman pivoting can form the 2×2 block; `"all"` pairs every candidate
(about 2× `nnz(L)` on KKT systems) and `"none"` turns the pairs off. With
`matching_alg = "algo5"`/`"algo6"` the pairs come from the cycles of the
matching instead. The pairs are decided from the values present when
`"analysis"` runs, so run the analysis after the first assembly of the matrix.
See [Symmetric indefinite systems](@ref).

## Data parameters

The outputs and when they become available are tabulated in the docstring of
[`getparam`](@ref). For a uniform batch the per-factorization outputs are
vectors with one entry per member.

## Differences from cuDSS

* `"info"` reports the original column of the first failed pivot.
* `getparam!` replaces the set-buffer-then-get protocol of cuDSS for vector
  outputs.
* `pivot_sign` chooses the sign of the perturbation of each row, so that a
  perturbed KKT system keeps the inertia an interior-point method expects.
* `pivot_stats` returns `(npos, nneg, nzero, nperturbed, n2x2)` in one read.
