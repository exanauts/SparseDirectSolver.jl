module SparseDirectSolverMetalExt

using SparseDirectSolver
using SparseDirectSolver: CSR, INDEX_ONE
using SparseArrays
using Metal
using Metal.MPS
using Metal.MTL

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
SparseDirectSolver._serial_cholesky_stats(::MetalBackend, nbatch::Integer) = nbatch > 1
SparseDirectSolver._host_reduce_cholesky_info(::MetalBackend, nbatch::Integer) = nbatch > 1
SparseDirectSolver._dense_solve_front(::MetalBackend, schedule::SparseDirectSolver.Schedule, s::Integer) = false
SparseDirectSolver._solve_workgroup(::MetalBackend) = 128

const MPS_POTRF_MIN_WIDTH = 64
SparseDirectSolver._offset_potrf_enabled(::MetalBackend) = true

struct MetalPotrfPlan{M, K, S}
    panel::M
    kernel::K
    status::S   # keeps the MTLBuffer of the MPS status output alive
end

# MPS writes an `MPSMatrixDecompositionStatus` per encode; one buffer of that type is
# shared by the wide fronts of a factorization. It is never read: the pivots are
# checked afterwards by `ka_chol_check_info!`.
SparseDirectSolver._offset_potrf_status(::MetalBackend) =
    MtlArray{MPS.MPSMatrixDecompositionStatus, 0, Metal.SharedStorage}(undef)

function SparseDirectSolver._offset_potrf_plan(::MetalBackend, factor::MtlVector{Float32}, p0, f, w,
                                               status::MtlArray{MPS.MPSMatrixDecompositionStatus, 0})
    w >= MPS_POTRF_MIN_WIDTH || return nothing
    desc = MPSMatrixDescriptor(w, w, sizeof(Float32) * f, Float32)
    panel = MPSMatrix(factor, desc, (p0 - 1) * sizeof(Float32))
    kernel = MPSMatrixDecompositionCholesky(Metal.device(), false, w)
    return MetalPotrfPlan(panel, kernel, status)
end

function SparseDirectSolver._offset_potrf_info!(::MetalBackend, plan::MetalPotrfPlan, info, idx)
    fill!(view(info, idx:idx), Int32(0))
    # the KA kernels before (assembly) and after (`ka_chol_check_info!`, trsm) run on the
    # same `global_queue`, which orders them with this command buffer
    cmdbuf = MPSCommandBuffer(global_queue(Metal.device()))
    encode!(cmdbuf, plan.kernel, plan.panel, plan.panel, Base.unsafe_convert(MTLBuffer, plan.status))
    commit!(cmdbuf)
    return true
end

# Metal.jl's MPS Cholesky and triangular solves work on full MtlMatrix objects,
# but panel views dispatch to host LAPACK/BLAS. Panel GEMM/SYRK are supported
# and faster (`:generic` allocates MPS descriptors per call on Metal); TRSM stays
# on KA, POTRF too except for single-matrix fronts at least `MPS_POTRF_MIN_WIDTH`
# wide (`_offset_potrf_plan`). Uniform batches use only KA dense kernels.
function SparseDirectSolver._allow_generic_dense(op::Symbol, ::MetalBackend, ::Type{T}) where {T}
    return op === :gemm || op === :syrk
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
    if structure in (SparseDirectSolver.STRUCTURE_SPD, SparseDirectSolver.STRUCTURE_HPD)
        opts.regime_c_width == 64 && opts.regime_c_rows == 512 &&
            opts.subtree_budgets == SparseDirectSolver.DEFAULT_SUBTREE_BUDGETS &&
            opts.subtree_parallelism == SparseDirectSolver.SUBTREE_PARALLELISM &&
            opts.subtree_max_fronts == 0 && opts.memory_budget == -1 || return opts
        out = copy(opts)
        out.regime_c_width = 32
        out.regime_c_rows = 128
        return out
    end
    structure in (SparseDirectSolver.STRUCTURE_SYMMETRIC, SparseDirectSolver.STRUCTURE_HERMITIAN,
                  SparseDirectSolver.STRUCTURE_GENERAL) || return opts
    out = copy(opts)
    out.factorization_alg = SparseDirectSolver.FACTORIZATION_VENDOR
    empty!(out.subtree_budgets)
    return out
end

end # module SparseDirectSolverMetalExt
