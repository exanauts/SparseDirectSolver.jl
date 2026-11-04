# Metal extension

Loaded with `using Metal`. Metal has no vendor sparse matrix container, so the
matrix is a [`CSR`](@ref) that owns `MtlArray`s, and only `Float32` is
supported; fronts at least 64 wide are factored by the MPS Cholesky, the other
dense operations run on the allocation-free KernelAbstractions kernels (Metal.jl's
`mul!`, the allocating `impl = :generic` path, still serves panel GEMM and SYRK
when selected explicitly).

```@autodocs
Modules = [Base.get_extension(SparseDirectSolver, :SparseDirectSolverMetalExt)]
```
