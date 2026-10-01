module SparseDirectSolverCUDAExt

# CUDA support: CSR adapters (T02); vendor dense bindings follow in T03.

using SparseDirectSolver
using SparseDirectSolver: CSR, INDEX_ONE, INDEX_ZERO, InvalidValueError
using SparseArrays
using CUDACore
using cuSPARSE

"""
    CSR(A::CuSparseMatrixCSR)

Wrap `A` without copying: the `CSR` shares `rowPtr`, `colVal` and `nzVal`
(one-based, `transposed = false`).
"""
function SparseDirectSolver.CSR(A::CuSparseMatrixCSR)
    m, n = size(A)
    return CSR(A.rowPtr, A.colVal, A.nzVal, m, n; index = INDEX_ONE, transposed = false)
end

"""
    CSR(A::CuSparseMatrixCSC)

Wrap `A` (`m × n`) without copying as the CSR of `transpose(A)` (`n × m`,
`transposed = true`), sharing `colPtr`, `rowVal` and `nzVal`.
"""
function SparseDirectSolver.CSR(A::CuSparseMatrixCSC)
    m, n = size(A)
    return CSR(A.colPtr, A.rowVal, A.nzVal, n, m; index = INDEX_ONE, transposed = true)
end

"""
    CuSparseMatrixCSR(A::CSR)

The stored CSR matrix of `A` (the `transposed` flag is not applied) as a
`CuSparseMatrixCSR`. One-based arrays are shared; zero-based index arrays are
copied and rebased. Batches (`nbatch(A) > 1`) are rejected.
"""
function cuSPARSE.CuSparseMatrixCSR(A::CSR{T, INT, <:CuVector{INT}, <:CuArray{T}}) where {T, INT}
    nbatch(A) == 1 || throw(InvalidValueError("cannot convert a batch of $(nbatch(A)) matrices to CuSparseMatrixCSR"))
    nzval = A.nzval isa CuVector ? A.nzval : vec(A.nzval)
    rowptr, colval = A.rowptr, A.colval
    if A.index == INDEX_ZERO
        rowptr = rowptr .+ one(INT)
        colval = colval .+ one(INT)
    end
    return CuSparseMatrixCSR{T, INT}(rowptr, colval, nzval, (A.nrows, A.ncols))
end

"""
    to_backend(A::SparseMatrixCSC, ::CUDABackend; index = 'O') -> CSR

CSR arrays of `A` built on the host, then uploaded as `CuVector`s.
"""
function SparseDirectSolver.to_backend(A::SparseMatrixCSC, ::CUDABackend; index = INDEX_ONE)
    B = CSR(A; index)
    return CSR(CuVector(B.rowptr), CuVector(B.colval), CuVector(B.nzval), B.nrows, B.ncols;
               index = B.index, transposed = B.transposed)
end

end # module SparseDirectSolverCUDAExt
