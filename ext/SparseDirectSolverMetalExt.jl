module SparseDirectSolverMetalExt

using SparseDirectSolver
using SparseDirectSolver: CSR, INDEX_ONE
using SparseArrays
using Metal

"""
    to_backend(A::SparseMatrixCSC, ::MetalBackend; index = 'O') -> CSR

Build the CSR arrays of `A` on the host and upload them to Metal. Metal has no
vendor sparse matrix container, so the returned [`CSR`](@ref) directly owns
`MtlVector` index and value arrays.
"""
function SparseDirectSolver.to_backend(A::SparseMatrixCSC, ::MetalBackend; index = INDEX_ONE)
    eltype(A) === Float32 || throw(SparseDirectSolver.NotSupportedError(
        "Metal currently supports Float32 matrices; got $(eltype(A))"))
    B = CSR(A; index)
    return CSR(MtlVector(B.rowptr), MtlVector(B.colval), MtlArray(B.nzval), B.nrows, B.ncols;
               index = B.index, transposed = B.transposed)
end

"""
    max_local_bytes(::MetalBackend) -> Int

Metal's maximum static threadgroup-memory allocation per workgroup (32 KiB on
Apple GPUs). The symbolic scheduler uses this to exclude the 48 and 64 KiB
regime-A kernel classes.
"""
SparseDirectSolver.max_local_bytes(::MetalBackend) = 32768
SparseDirectSolver._allow_batched_dense(::MetalBackend) = false
SparseDirectSolver._serial_cholesky_stats(::MetalBackend) = true
SparseDirectSolver._host_reduce_cholesky_info(::MetalBackend) = true

# Metal.jl's MPS Cholesky and triangular solves work on full MtlMatrix objects,
# but panel views dispatch to host LAPACK/BLAS. Mixing MPS commands with KA
# launches also does not preserve the per-front dependency order required by
# uniform batches. Keep the solver's dense operations on one KA command stream.
function SparseDirectSolver._allow_generic_dense(op::Symbol, ::MetalBackend, ::Type{T}) where {T}
    return false
end

function SparseDirectSolver._backend_schedule_options(backend::MetalBackend, structure::SparseDirectSolver.Structure,
                                                      opts::SparseDirectSolver.Options, nbatch::Integer)
    out = SparseDirectSolver._backend_schedule_options(backend, structure, opts)
    nbatch == 1 && return out
    out === opts && (out = copy(opts))
    out.factorization_alg = SparseDirectSolver.FACTORIZATION_VENDOR
    empty!(out.subtree_budgets)
    return out
end

# The fused LDLT/LU kernels bind 32 device buffers; Apple GPUs expose 31
# indirect argument-buffer resources. Route those structures through the
# existing regime-C kernels. Cholesky's smaller fused kernels remain enabled.
function SparseDirectSolver._backend_schedule_options(::MetalBackend, structure::SparseDirectSolver.Structure,
                                                      opts::SparseDirectSolver.Options)
    structure in (SparseDirectSolver.STRUCTURE_SYMMETRIC, SparseDirectSolver.STRUCTURE_HERMITIAN,
                  SparseDirectSolver.STRUCTURE_GENERAL) || return opts
    out = copy(opts)
    out.factorization_alg = SparseDirectSolver.FACTORIZATION_VENDOR
    empty!(out.subtree_budgets)
    return out
end

end # module SparseDirectSolverMetalExt
