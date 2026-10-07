# Internals

Docstrings of the non-exported functions and types: the host symbolic analysis,
the numeric kernels, the solve and the dense interface. They describe the
implementation and change without notice; the design is in
[`PLAN.md`](https://github.com/exanauts/SparseDirectSolver.jl/blob/main/PLAN.md).

```@autodocs
Modules = [SparseDirectSolver]
Public = false
# methods of Base, LinearAlgebra and SparseArrays functions are on the other pages
Filter = t -> !(t isa Function) || parentmodule(t) === SparseDirectSolver
```
