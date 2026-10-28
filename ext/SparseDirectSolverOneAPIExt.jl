module SparseDirectSolverOneAPIExt

# oneAPI (Intel GPU) support (T31), mirroring `SparseDirectSolverAMDGPUExt`: CSR
# adapters for oneMKL sparse matrices, the public API on them (T13), the vendor dense
# bindings of the dense layer (PLAN §2.6) on oneMKL, and the backend's local memory
# per workgroup for the regime-A kernels (issue #60).
#
# Every oneMKL entry point of oneAPI.jl's support library waits for its queue before
# it returns (`wait_and_throw`), and the LAPACK routines need a scratchpad: a vendor
# call is a host synchronization, LAPACK also a device allocation. `impl = :auto`
# therefore keeps LAPACK (`potrf`, `getrf`) off the vendor path
# (`_auto_vendor_dense`); the level-3 BLAS calls (`gemm`, `syrk`/`herk`, `trsm` and
# their strided batches) stay on it. The generic LinearAlgebra path is disabled
# (`_allow_generic_dense`): it reaches the same oneMKL calls, plus the allocating
# LAPACK wrappers.
#
# LAPACK status: the C shims return oneMKL's `computation_error` info on the host;
# the Julia wrappers of oneAPI.jl drop it, so `potrf`/`getrf` call the C shims
# directly (until a oneAPI.jl release returns it from `oneMKL.potrf!`/`getrf!`).
# oneMKL's GPU `potrf` (2026.1, Data Center GPU Max) names the failing column
# relative to its 32-wide panel: column 50 comes back as 18 for real types and as
# 1042 (panel 1, column 18) for complex ones; only columns 1..32 are right. The
# factor still follows LAPACK `potf2` (the failed pivot stays in `A[j, j]`), so
# `potrf!` and `potrf_info!` take the first non-positive diagonal entry as the
# column (`ka_chol_diag_info!`, `ka_chol_check_info!`); `vendor_potrf!` returns
# oneMKL's value as reported. getrf's info is right. Batched `potrf`/`getrf` report one status per batch, not per member: no
# binding (the KA kernels serve uniform batches). No `sytrf`: oneMKL rejects it on
# the Intel Data Center GPU Max with `unsupported_device`, thrown from the
# scratchpad-size shim, which does not catch it, so the process aborts.

using SparseDirectSolver
using SparseDirectSolver: CSR, DirectSolver, INDEX_ONE, INDEX_ZERO, InvalidValueError, NotSupportedError
using SparseArrays
using LinearAlgebra
using KernelAbstractions
using oneAPI
using oneAPI: oneL0, oneMKL, Support, oneStridedMatrix, oneStridedArray, sycl_queue, global_queue
using oneAPI.oneL0: ZePtr
using oneAPI.oneMKL: oneSparseMatrixCSR, oneSparseMatrixCSC

const SDS = SparseDirectSolver

"""
    CSR(A::oneSparseMatrixCSR)

Wrap `A` without copying: the `CSR` shares `rowPtr`, `colVal` and `nzVal`
(one-based, `transposed = false`).
"""
function SparseDirectSolver.CSR(A::oneSparseMatrixCSR)
    m, n = size(A)
    return CSR(A.rowPtr, A.colVal, A.nzVal, m, n; index = INDEX_ONE, transposed = false)
end

"""
    CSR(A::oneSparseMatrixCSC)

Wrap `A` (`m × n`) without copying as the CSR of `transpose(A)` (`n × m`,
`transposed = true`), sharing `colPtr`, `rowVal` and `nzVal`.
"""
function SparseDirectSolver.CSR(A::oneSparseMatrixCSC)
    m, n = size(A)
    return CSR(A.colPtr, A.rowVal, A.nzVal, n, m; index = INDEX_ONE, transposed = true)
end

"""
    oneSparseMatrixCSR(A::CSR)

The stored CSR matrix of `A` (the `transposed` flag is not applied) as a
`oneSparseMatrixCSR`. One-based arrays are shared; zero-based index arrays are
copied and rebased. Batches (`nbatch(A) > 1`) are rejected.
"""
function oneMKL.oneSparseMatrixCSR(A::CSR{T, INT, <:oneVector{INT}, <:oneArray{T}}) where {T, INT}
    nbatch(A) == 1 || throw(InvalidValueError("cannot convert a batch of $(nbatch(A)) matrices to oneSparseMatrixCSR"))
    nzval = A.nzval isa oneVector ? A.nzval : vec(A.nzval)
    rowptr, colval = A.rowptr, A.colval
    if A.index == INDEX_ZERO
        rowptr = rowptr .+ one(INT)
        colval = colval .+ one(INT)
    end
    return oneSparseMatrixCSR{T, INT}(rowptr, colval, nzval, (A.nrows, A.ncols))
end

_check_eltype(::Type{T}) where {T} =
    real(T) === Float64 && !KernelAbstractions.supports_float64(oneAPIBackend()) &&
    throw(NotSupportedError("the oneAPI device $(oneAPI.device()) has no Float64 support; got $T"))

"""
    to_backend(A::SparseMatrixCSC, ::oneAPIBackend; index = 'O') -> CSR

CSR arrays of `A` built on the host, then uploaded as `oneVector`s. Float64 and
ComplexF64 raise `NotSupportedError` on a device without Float64 support.
"""
function SparseDirectSolver.to_backend(A::SparseMatrixCSC{T}, ::oneAPIBackend; index = INDEX_ONE) where {T}
    _check_eltype(T)
    B = CSR(A; index)
    return CSR(oneVector(B.rowptr), oneVector(B.colval), oneVector(B.nzval), B.nrows, B.ncols;
               index = B.index, transposed = B.transposed)
end

"""
    max_local_bytes(::oneAPIBackend) -> Int

The shared local memory of one workgroup on the current oneAPI device
(`maxSharedLocalMemory` of its compute properties; 128 KiB on the Intel Data
Center GPU Max): the regime-A kernels may use their largest class, 64 KiB.
"""
SparseDirectSolver.max_local_bytes(::oneAPIBackend) =
    Int(oneL0.compute_properties(oneAPI.device()).maxSharedLocalMemory)

# ---------------------------------------------------------------------------
# public API (T13, PLAN §3.1): solver constructors, update!, the LinearAlgebra layer

"""
    DirectSolver(A::oneSparseMatrixCSR, structure::String, view::Char; index = 'O')
    DirectSolver(A::oneSparseMatrixCSC, structure::String, view::Char; index = 'O')

Solver on `A`'s arrays, without copies (the oneAPI twin of
`DirectSolver(::CuSparseMatrixCSR, …)`). A `oneSparseMatrixCSC` is read as the
CSR of its transpose with the view flipped, see [`DirectSolver`](@ref).
"""
SparseDirectSolver.DirectSolver(A::oneSparseMatrixCSR, structure, view; index = INDEX_ONE) =
    DirectSolver(CSR(A), structure, view; index)
SparseDirectSolver.DirectSolver(A::oneSparseMatrixCSC, structure, view; index = INDEX_ONE) =
    DirectSolver(CSR(A), structure, view; index)

"""
    update!(solver::DirectSolver, A::oneSparseMatrixCSR)
    update!(solver::DirectSolver, A::oneSparseMatrixCSC)

Point `solver` at the arrays of `A`, see [`update!`](@ref).
"""
SparseDirectSolver.update!(solver::DirectSolver, A::Union{oneSparseMatrixCSR, oneSparseMatrixCSC}) =
    SparseDirectSolver.update!(solver, CSR(A))

"""
    cholesky(A::oneSparseMatrixCSR, NoPivot(); view = 'F', check = false) -> DirectSolver
    cholesky(Symmetric(A::oneSparseMatrixCSR)) / cholesky(Hermitian(A::oneSparseMatrixCSR))

LLᵀ/LLᴴ factorization of `A` on the Intel GPU; the wrappers pass their `uplo` as
the view. See `cholesky(::CSR)`.
"""
LinearAlgebra.cholesky(A::oneSparseMatrixCSR, p::NoPivot = NoPivot(); view::Char = 'F', check::Bool = false) =
    cholesky(CSR(A), p; view, check)
LinearAlgebra.cholesky(A::Symmetric{T, <:oneSparseMatrixCSR{T}}, p::NoPivot = NoPivot();
                       check::Bool = false) where {T <: Union{Float32, Float64}} =
    cholesky(CSR(A.data), p; view = A.uplo, check)
LinearAlgebra.cholesky(A::Hermitian{T, <:oneSparseMatrixCSR{T}}, p::NoPivot = NoPivot(); check::Bool = false) where {T} =
    cholesky(CSR(A.data), p; view = A.uplo, check)

"""
    ldlt(A::oneSparseMatrixCSR; view = 'F', check = false) -> DirectSolver
    ldlt(Symmetric(A::oneSparseMatrixCSR)) / ldlt(Hermitian(A::oneSparseMatrixCSR))

LDLᵀ/LDLᴴ factorization of `A` on the Intel GPU; the wrappers pass their `uplo`
as the view. See `ldlt(::CSR)`.
"""
LinearAlgebra.ldlt(A::oneSparseMatrixCSR; view::Char = 'F', check::Bool = false) = ldlt(CSR(A); view, check)
LinearAlgebra.ldlt(A::Symmetric{T, <:oneSparseMatrixCSR{T}}; check::Bool = false) where {T <: Real} =
    ldlt(CSR(A.data); view = A.uplo, check)
LinearAlgebra.ldlt(A::Hermitian{T, <:oneSparseMatrixCSR{T}}; check::Bool = false) where {T} =
    ldlt(CSR(A.data); view = A.uplo, check)

"""
    lu(A::oneSparseMatrixCSR; check = false) -> DirectSolver

`L D U` factorization (structure `"G"`) of `A` on the Intel GPU. See `lu(::CSR)`.
"""
LinearAlgebra.lu(A::oneSparseMatrixCSR; check::Bool = false) = lu(CSR(A); check)

# ---------------------------------------------------------------------------
# dense implementation policy (see the header)

SparseDirectSolver._allow_generic_dense(op::Symbol, ::oneAPIBackend, ::Type{T}) where {T} = false
SparseDirectSolver._auto_vendor_dense(op::Symbol, ::oneAPIBackend, ::Type{T}) where {T} =
    op in (:gemm, :syrk, :herk, :trsm, :gemm_strided_batched, :trsm_strided_batched)

# ---------------------------------------------------------------------------
# vendor dense bindings (see `src/dense/vendor.jl` for the contracts)

const MklT = Union{Float32, Float64, ComplexF32, ComplexF64}
const MklC = Union{ComplexF32, ComplexF64}

_lda(A) = max(1, stride(A, 2))
_queue(A) = global_queue(oneAPI.context(A), oneAPI.device(A))
# device address of `A` for the support-library calls declared with host `Ptr` arguments
_rawptr(A::oneStridedArray{T}) where {T} = reinterpret(Ptr{T}, Base.unsafe_convert(ZePtr{T}, A))

SDS.vendor_gemm!(tA::Char, tB::Char, α, A::oneStridedMatrix{T}, B::oneStridedMatrix{T}, β,
                 C::oneStridedMatrix{T}) where {T <: MklT} =
    oneMKL.gemm!(tA, tB, T(α), A, B, T(β), C)
SDS.vendor_syrk!(uplo::Char, α, A::oneStridedMatrix{T}, β, C::oneStridedMatrix{T}) where {T <: MklT} =
    oneMKL.syrk!(uplo, 'N', T(α), A, T(β), C)
SDS.vendor_herk!(uplo::Char, α, A::oneStridedMatrix{T}, β, C::oneStridedMatrix{T}) where {T <: MklC} =
    oneMKL.herk!(uplo, 'N', real(T)(α), A, real(T)(β), C)
SDS.vendor_trsm!(side::Char, uplo::Char, trans::Char, diag::Char, α, A::oneStridedMatrix{T},
                 B::oneStridedMatrix{T}) where {T <: MklT} =
    oneMKL.trsm!(side, uplo, trans, diag, T(α), A, B)

for (elty, potrf, potrf_size, getrf, getrf_size) in
        ((:Float32, :onemklSpotrf, :onemklSpotrf_scratchpad_size, :onemklSgetrf, :onemklSgetrf_scratchpad_size),
         (:Float64, :onemklDpotrf, :onemklDpotrf_scratchpad_size, :onemklDgetrf, :onemklDgetrf_scratchpad_size),
         (:ComplexF32, :onemklCpotrf, :onemklCpotrf_scratchpad_size, :onemklCgetrf, :onemklCgetrf_scratchpad_size),
         (:ComplexF64, :onemklZpotrf, :onemklZpotrf_scratchpad_size, :onemklZgetrf, :onemklZgetrf_scratchpad_size))
    @eval begin
        # LAPACK info of oneMKL's potrf (0, the failing column as oneMKL reports it, i.e.
        # panel-relative beyond column 32 (see the header), or -1 for a SYCL error)
        function _potrf_info(uplo::Char, A::oneStridedMatrix{$elty})
            n = LinearAlgebra.checksquare(A)
            q = _queue(A)
            lwork = Support.$potrf_size(sycl_queue(q), uplo, n, _lda(A))
            work = oneVector{$elty}(undef, lwork)
            info = Support.$potrf(sycl_queue(q), uplo, n, A, _lda(A), work, lwork)
            oneAPI.unsafe_free!(work)
            return Int(info)
        end

        function _getrf_info(A::oneStridedMatrix{$elty}, ipiv::oneVector{Int64})
            m, n = size(A)
            q = _queue(A)
            lwork = Support.$getrf_size(sycl_queue(q), m, n, _lda(A))
            work = oneVector{$elty}(undef, lwork)
            info = Support.$getrf(sycl_queue(q), m, n, A, _lda(A), ipiv, work, lwork)
            oneAPI.unsafe_free!(work)
            return Int(info)
        end

    end
end

SDS.vendor_potrf!(uplo::Char, A::oneStridedMatrix{<:MklT}) = _potrf_info(uplo, A)

# the status is a host value (see the header): written into `info[idx]` on the device
function SDS.vendor_potrf_info!(uplo::Char, A::oneStridedMatrix{<:MklT}, info::oneVector{Int32}, idx::Integer)
    1 <= idx <= length(info) || throw(DimensionMismatch("info index $idx outside 1:$(length(info))"))
    fill!(view(info, idx:idx), Int32(_potrf_info(uplo, A)))
    return info
end

# oneMKL pivots are `Int64`; other index vectors go through a temporary
function _with_int64(f, ipiv::oneVector, k::Integer)
    ipiv isa oneVector{Int64} && length(ipiv) == k && return f(ipiv)
    p = oneVector{Int64}(undef, k)
    info = f(p)
    view(ipiv, 1:k) .= p
    oneAPI.unsafe_free!(p)
    return info
end

SDS.vendor_getrf!(A::oneStridedMatrix{<:MklT}, ipiv::oneVector{<:Integer}) =
    _with_int64(p -> _getrf_info(A, p), ipiv, min(size(A)...))

function SDS.vendor_gemm_strided_batched!(tA::Char, tB::Char, α, A::oneStridedArray{T, 3}, B::oneStridedArray{T, 3}, β,
                                          C::oneStridedArray{T, 3}) where {T <: MklT}
    size(A, 3) == size(B, 3) == size(C, 3) || throw(DimensionMismatch("gemm_strided_batched: batch counts differ"))
    size(C, 3) == 0 && return C
    return oneMKL.gemm_strided_batched!(tA, tB, T(α), A, B, T(β), C)
end

for (fname, elty) in ((:onemklStrsm_batch_strided, :Float32), (:onemklDtrsm_batch_strided, :Float64),
                      (:onemklCtrsm_batch_strided, :ComplexF32), (:onemklZtrsm_batch_strided, :ComplexF64))
    @eval function SDS.vendor_trsm_batched!(side::Char, uplo::Char, trans::Char, diag::Char, α,
                                            A::oneStridedArray{$elty, 3}, B::oneStridedArray{$elty, 3})
        m, n, nb = size(B)
        size(A, 1) == size(A, 2) == (side == 'L' ? m : n) && size(A, 3) == nb ||
            throw(DimensionMismatch("trsm_batched: A is $(size(A)), B is $(size(B)) (side $side)"))
        nb == 0 && return B
        alpha = Ref($elty(α))
        GC.@preserve A B alpha begin
            # the declared host `Ptr`s carry device addresses, and every pointer is passed
            # typed: oneAPI.jl's ComplexF64 binding declares them `Ptr{ComplexF32}`. The
            # `sycl_queue` call comes after the argument preparation (oneAPI.jl's Julia →
            # oneMKL ordering boundary).
            a, b = _rawptr(A), _rawptr(B)
            pα = Base.unsafe_convert(Ptr{$elty}, alpha)
            Support.$fname(sycl_queue(_queue(B)), side, uplo, trans, diag, m, n, pα, a, _lda(A),
                           stride(A, 3), b, _lda(B), stride(B, 3), nb)
        end
        return B
    end
end

end # module SparseDirectSolverOneAPIExt
