# MadNLP extension

Loaded with `using MadNLP`. It provides `SDSSolver`, a MadNLP linear solver
configured by the `sds_*` options of `SDSSolverOptions`; the manual page
[MadNLP](@ref) explains its use with MadNLPGPU, the options and the fused
pipeline.

```@autodocs
Modules = [Base.get_extension(SparseDirectSolver, :SparseDirectSolverMadNLPExt)]
```
