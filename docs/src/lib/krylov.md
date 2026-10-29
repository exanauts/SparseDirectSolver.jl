# Krylov extension

Loaded with `using Krylov`. It enables flexible-GMRES iterative refinement
(`ir_mode = "fgmres"`, the factorization as right preconditioner) by
registering `fgmres_correction`, which runs `Krylov.fgmres!`; the operators and
the refinement loop live in the core.

```@autodocs
Modules = [Base.get_extension(SparseDirectSolver, :SparseDirectSolverKrylovExt)]
```
