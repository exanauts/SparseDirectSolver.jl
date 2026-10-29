# Metal extension

Loaded with `using Metal`. Metal has no vendor sparse matrix container, so the
matrix is a [`CSR`](@ref) that owns `MtlArray`s, and only `Float32` is
supported; panel GEMM and SYRK run on Metal Performance Shaders, and fronts at
least 64 wide are factored by the MPS Cholesky (the other dense operations stay
on the KernelAbstractions kernels).

```@autodocs
Modules = [Base.get_extension(SparseDirectSolver, :SparseDirectSolverMetalExt)]
```
