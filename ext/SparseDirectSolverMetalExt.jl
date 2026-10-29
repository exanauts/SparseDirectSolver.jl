module SparseDirectSolverMetalExt

using SparseDirectSolver
using SparseDirectSolver: CSR, INDEX_ONE, NotSupportedError
using SparseArrays
using KernelAbstractions
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
    eltype(A) === Float32 || throw(NotSupportedError(
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
    status::S   # the MPS status output (`MPSMatrixDecompositionStatus`, an Int32 enum; 0 = success)
end

# MPS writes an `MPSMatrixDecompositionStatus` per encode into one Int32 buffer shared by the wide
# fronts of a factorization; `_mps_status_info_kernel!` folds it into `info` right after each encode,
# on the same queue, so a front reads the status of its own decomposition
SparseDirectSolver._offset_potrf_status(::MetalBackend) = Metal.zeros(Int32, 1)

function SparseDirectSolver._offset_potrf_plan(::MetalBackend, factor::MtlVector{Float32}, p0, f, w,
                                               status::MtlVector{Int32})
    w >= MPS_POTRF_MIN_WIDTH || return nothing
    desc = MPSMatrixDescriptor(w, w, sizeof(Float32) * f, Float32)
    panel = MPSMatrix(factor, desc, (p0 - 1) * sizeof(Float32))
    kernel = MPSMatrixDecompositionCholesky(Metal.device(), false, w)
    return MetalPotrfPlan(panel, kernel, status)
end

# Apple leaves the panel undefined after a failed decomposition, so the diagonal check
# alone cannot be trusted: a nonzero MPS status with no bad diagonal entry reports the
# last column `w` of the front (a failure inside it, column unknown); the status is reset
@kernel function _mps_status_info_kernel!(info, idx, status, w)
    @inbounds if @index(Global) == 1
        status[1] != Int32(0) && info[idx] == Int32(0) && (info[idx] = Int32(w))
        status[1] = Int32(0)
    end
end

function SparseDirectSolver._offset_potrf_info!(backend::MetalBackend, plan::MetalPotrfPlan, info, idx, A, dmin)
    fill!(view(info, idx:idx), Int32(0))
    # the KA kernels before (assembly) and after (the pivot and status checks, trsm) run on
    # Metal.jl's `global_queue`, the queue of this command buffer, which orders them with the
    # MPS decomposition; moving either to another queue would need explicit synchronization
    cmdbuf = MPSCommandBuffer(global_queue(Metal.device()))
    encode!(cmdbuf, plan.kernel, plan.panel, plan.panel, Base.unsafe_convert(MTLBuffer, plan.status))
    commit!(cmdbuf)
    SparseDirectSolver.ka_chol_check_info!(info, idx, A, dmin)
    _mps_status_info_kernel!(backend, 1)(info, Int(idx), plan.status, Int32(size(A, 1)); ndrange = 1)
    return true
end

# Metal.jl's MPS Cholesky and triangular solves work on full MtlMatrix objects,
# but panel views dispatch to host LAPACK/BLAS. Panel GEMM/SYRK run on MPS through
# `:generic` (`mul!` on `MtlArray` views), which allocates MPS descriptors per call
# but is faster than the KA kernels (a deviation from T31's `:auto` prefers `:ka`,
# recorded in its Report); TRSM stays on KA. POTRF stays on KA through `select_impl`;
# under `impl = :auto` with vendor calls allowed, single-matrix fronts at least
# `MPS_POTRF_MIN_WIDTH` wide take the cached MPS Cholesky (`_offset_potrf_plan`).
# Uniform batches use only KA dense kernels.
function SparseDirectSolver._allow_generic_dense(op::Symbol, ::MetalBackend, ::Type{T}) where {T}
    return op === :gemm || op === :syrk
end

# The analysis entry point (`src/solver.jl`): LDLT/LU and uniform batches run regime C only
# (`_regime_c_only`), so an explicit schedule that asks for anything else is rejected
# instead of replaced. The 3-argument method below does the replacement (test helpers).
function SparseDirectSolver._backend_schedule_options(backend::MetalBackend, structure::SparseDirectSolver.Structure,
                                                      opts::SparseDirectSolver.Options, nbatch::Integer)
    if nbatch > 1 || _regime_c_only(structure)
        what = nbatch > 1 ? "uniform batches" : "structure $(structure)"
        opts.factorization_alg in (SparseDirectSolver.FACTORIZATION_DEFAULT, SparseDirectSolver.FACTORIZATION_VENDOR) ||
            throw(NotSupportedError("factorization_alg = $(opts.factorization_alg) on Metal: $what run every " *
                                    "front through the vendor regime-C path (\"algo2\"); the fused kernels " *
                                    "bind 32 buffers and Apple GPUs expose 31"))
        isempty(opts.subtree_budgets) || opts.subtree_budgets == SparseDirectSolver.DEFAULT_SUBTREE_BUDGETS ||
            throw(NotSupportedError("explicit subtree_budgets on Metal: $what run no regime-A subtrees " *
                                    "(the fused kernels bind 32 buffers, Apple GPUs expose 31)"))
    end
    out = SparseDirectSolver._backend_schedule_options(backend, structure, opts)
    nbatch == 1 && return out
    out === opts && (out = copy(opts))
    out.factorization_alg = SparseDirectSolver.FACTORIZATION_VENDOR
    empty!(out.subtree_budgets)
    return out
end

_regime_c_only(structure) = structure in (SparseDirectSolver.STRUCTURE_SYMMETRIC, SparseDirectSolver.STRUCTURE_HERMITIAN,
                                          SparseDirectSolver.STRUCTURE_GENERAL)

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
    _regime_c_only(structure) || return opts
    out = copy(opts)
    out.factorization_alg = SparseDirectSolver.FACTORIZATION_VENDOR
    empty!(out.subtree_budgets)
    return out
end

end # module SparseDirectSolverMetalExt
