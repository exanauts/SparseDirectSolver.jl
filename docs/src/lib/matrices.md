# Matrices

```@docs
CSR
csr_of_transpose
to_backend
nbatch
MatrixDescriptor
SparseArrays.nnz(::SparseDirectSolver.CSR)
SparseArrays.SparseMatrixCSC(::SparseDirectSolver.CSR)
```

[`SparseDirectSolver.update!`](@ref) (not exported) also replaces the data of a `MatrixDescriptor`.
