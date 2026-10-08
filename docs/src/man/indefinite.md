# Symmetric indefinite systems

The KKT systems of interior-point methods,

```math
K = \begin{bmatrix} H + \Sigma & J^T \\ J & -\delta I \end{bmatrix},
```

are symmetric and indefinite. They are factored with the structures `"S"`
(``L D L^T``; complex symmetric for complex element types) and `"H"`
(``L D L^H``), where ``D`` holds 1×1 and 2×2 pivots.

## Pivoting

Inside every front the factorization uses Bunch–Kaufman pivoting: a 1×1 pivot is
accepted when it is at least `pivot_threshold` (default `0.01`) times the largest
entry of its column, otherwise a 2×2 pivot is tried. Pivots are only exchanged
within the fully-summed block of a front, so a pivot that fails there is not
delayed to the parent: it is perturbed to `pivot_epsilon` and counted in
`"npivots"`. Two mechanisms keep perturbations rare on KKT matrices:

* **2×2 pivot pairs at the analysis** (`pivot_pairs`, default `"default"`): a row
  whose diagonal is structurally zero (or below `pivot_pair_tolerance`) is paired
  with a partner and the pair is kept in one supernode, so that the in-front
  pivoting can form the 2×2 block.
* **Matching** (`matching_alg = "algo5"` or `"algo6"`): the 2×2 pairs come from the
  cycles of a maximum-product matching, and the matrix is scaled symmetrically
  (the inertia is preserved).

## Perturbation sign and inertia

cuDSS perturbs a tiny pivot with the sign of the pivot. `pivot_sign` (one entry
in `(-1, 0, 1)` per row) chooses the sign instead, for example `+1` for the primal
rows and `-1` for the dual rows, so that the factored matrix ``K + E`` has the
inertia the interior-point method expects.

```julia
solver = DirectSolver(K_gpu, "S", 'L')
setparam!(solver, "pivot_sign", Int8[fill(1, nh); fill(-1, nj)])
execute!("analysis", solver, x, b)
execute!("factorization", solver, x, b)
getparam(solver, "inertia")       # (npos, nneg), as MadNLP reads it
getparam(solver, "pivot_stats")   # (npos, nneg, nzero, nperturbed, n2x2)
```

`"inertia"` is the inertia of ``D`` after perturbation; read it together with
`"npivots"`. With perturbed pivots, iterative refinement (`ir_n_steps`, or
`ir_mode = "fgmres"` after `using Krylov`) recovers the accuracy of the solve.

## General matrices

The structure `"G"` factors ``P_r P A P^T = L D U`` on the symmetric pattern of
``A + A^T``, with threshold partial pivoting by rows inside the fully-summed block
of each front (`pivot_threshold`) and static perturbation of tiny pivots.
`"perm_row"` and `"perm_col"` return the final permutations, and `solve_mode`
solves with ``A^T`` or ``A^H``. For matrices with zero or small diagonal
entries, matching (`matching_alg = "algo5"`) moves large entries to the
diagonal and scales rows and columns before the ordering.
