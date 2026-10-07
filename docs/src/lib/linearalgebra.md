# LinearAlgebra interface

A [`DirectSolver`](@ref) is a `LinearAlgebra.Factorization`. The functions below
take a [`CSR`](@ref) matrix; the [CUDA extension](@ref "CUDA extension") adds
methods for `CuSparseMatrixCSR`.

```@docs
LinearAlgebra.cholesky(::SparseDirectSolver.CSR, ::LinearAlgebra.NoPivot)
LinearAlgebra.ldlt(::SparseDirectSolver.CSR)
LinearAlgebra.lu(::SparseDirectSolver.CSR)
LinearAlgebra.cholesky!(::DirectSolver, ::Any)
LinearAlgebra.ldlt!(::DirectSolver, ::Any)
LinearAlgebra.lu!(::DirectSolver, ::Any)
LinearAlgebra.ldiv!(::DirectSolver, ::AbstractVector)
Base.:\(::DirectSolver, ::AbstractVector)
LinearAlgebra.diag(::DirectSolver)
SparseArrays.nnz(::DirectSolver)
LinearAlgebra.logabsdet(::DirectSolver)
LinearAlgebra.logdet(::DirectSolver)
```
