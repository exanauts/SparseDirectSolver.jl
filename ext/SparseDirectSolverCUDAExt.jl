module SparseDirectSolverCUDAExt

# CUDA support: CSR adapters (T02) and the vendor dense bindings of the dense
# layer (T03, PLAN §2.6): cuBLAS gemm/syrk/herk/trsm/strided-batched gemm/
# batched trsm, cuSOLVER potrf (also with a device info)/getrf/sytrf/potrfBatched,
# cuBLAS getrfBatched.

using SparseDirectSolver
using SparseDirectSolver: CSR, DirectSolver, INDEX_ONE, INDEX_ZERO, InvalidValueError
using SparseArrays
using LinearAlgebra
using CUDACore
using cuSPARSE
using cuBLAS
using cuSOLVER

const SDS = SparseDirectSolver

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

# ---------------------------------------------------------------------------
# public API (T13, PLAN §3.1): solver constructors, update!, the LinearAlgebra layer

"""
    DirectSolver(A::CuSparseMatrixCSR, structure::String, view::Char; index = 'O')
    DirectSolver(A::CuSparseMatrixCSC, structure::String, view::Char; index = 'O')

Solver on `A`'s arrays, without copies (≅ `CudssSolver(A, structure, view)`). A
`CuSparseMatrixCSC` is read as the CSR of its transpose with the view flipped
(MadNLP's lower-triangle CSC becomes an upper-triangle CSR), see
[`DirectSolver`](@ref).
"""
SparseDirectSolver.DirectSolver(A::CuSparseMatrixCSR, structure, view; index = INDEX_ONE) =
    DirectSolver(CSR(A), structure, view; index)
SparseDirectSolver.DirectSolver(A::CuSparseMatrixCSC, structure, view; index = INDEX_ONE) =
    DirectSolver(CSR(A), structure, view; index)

"""
    update!(solver::DirectSolver, A::CuSparseMatrixCSR)
    update!(solver::DirectSolver, A::CuSparseMatrixCSC)

Point `solver` at the arrays of `A` (≅ `cudss_update(solver, A)`), see
[`update!`](@ref).
"""
SparseDirectSolver.update!(solver::DirectSolver, A::Union{CuSparseMatrixCSR, CuSparseMatrixCSC}) =
    SparseDirectSolver.update!(solver, CSR(A))

"""
    cholesky(A::CuSparseMatrixCSR, NoPivot(); view = 'F', check = false) -> DirectSolver
    cholesky(Symmetric(A::CuSparseMatrixCSR)) / cholesky(Hermitian(A::CuSparseMatrixCSR))

LLᵀ/LLᴴ factorization of `A` on the GPU (≅ CUDSS.jl's `cholesky`); the wrappers
pass their `uplo` as the view. See `cholesky(::CSR)`.
"""
LinearAlgebra.cholesky(A::CuSparseMatrixCSR, p::NoPivot = NoPivot(); view::Char = 'F', check::Bool = false) =
    cholesky(CSR(A), p; view, check)
LinearAlgebra.cholesky(A::Symmetric{T, <:CuSparseMatrixCSR{T}}, p::NoPivot = NoPivot();
                       check::Bool = false) where {T <: Union{Float32, Float64}} =
    cholesky(CSR(A.data), p; view = A.uplo, check)
LinearAlgebra.cholesky(A::Hermitian{T, <:CuSparseMatrixCSR{T}}, p::NoPivot = NoPivot(); check::Bool = false) where {T} =
    cholesky(CSR(A.data), p; view = A.uplo, check)

"""
    ldlt(A::CuSparseMatrixCSR; view = 'F', check = false) -> DirectSolver
    ldlt(Symmetric(A::CuSparseMatrixCSR)) / ldlt(Hermitian(A::CuSparseMatrixCSR))

LDLᵀ/LDLᴴ factorization of `A` on the GPU (≅ CUDSS.jl's `ldlt`); the wrappers
pass their `uplo` as the view. See `ldlt(::CSR)`.
"""
LinearAlgebra.ldlt(A::CuSparseMatrixCSR; view::Char = 'F', check::Bool = false) = ldlt(CSR(A); view, check)
LinearAlgebra.ldlt(A::Symmetric{T, <:CuSparseMatrixCSR{T}}; check::Bool = false) where {T <: Real} =
    ldlt(CSR(A.data); view = A.uplo, check)
LinearAlgebra.ldlt(A::Hermitian{T, <:CuSparseMatrixCSR{T}}; check::Bool = false) where {T} =
    ldlt(CSR(A.data); view = A.uplo, check)

# ---------------------------------------------------------------------------
# vendor dense bindings (see `src/dense/vendor.jl` for the contracts)

const CuBlasT = Union{Float32, Float64, ComplexF32, ComplexF64}
const CuBlasC = Union{ComplexF32, ComplexF64}

# cuBLAS handles run in device pointer mode: a host scalar α/β becomes a fresh
# one-element device allocation (`CuRefValue`) on every call. The scalars the
# solver passes (0, 1, -1) come from a cached device vector per context and type
# instead, so the numeric and solve phases allocate nothing on the device.
const _SCALARS = Dict{Tuple{CuContext, DataType}, CuVector}()
const _SCALARS_LOCK = ReentrantLock()

function _scalar_cache(::Type{T}) where {T}
    key = (CUDACore.context(), T)
    return lock(_SCALARS_LOCK) do
        get!(_SCALARS, key) do
            c = CuVector{T}(T[0, 1, -1])
            CUDACore.synchronize()  # once per context and type: the constants are visible on every stream
            return c
        end
    end::CuVector{T}
end

# α as a cuBLAS scalar argument: a cached device constant, else the host value
function _blas_scalar(::Type{T}, α) where {T}
    x = T(α)
    i = iszero(x) ? 1 : isone(x) ? 2 : x == -one(T) ? 3 : 0
    return i == 0 ? x : CUDACore.CuRefArray(_scalar_cache(T), i)
end

SDS.vendor_gemm!(tA::Char, tB::Char, α, A::StridedCuMatrix{T}, B::StridedCuMatrix{T}, β,
                 C::StridedCuMatrix{T}) where {T <: CuBlasT} =
    cuBLAS.gemm!(tA, tB, _blas_scalar(T, α), A, B, _blas_scalar(T, β), C)
SDS.vendor_syrk!(uplo::Char, α, A::StridedCuMatrix{T}, β, C::StridedCuMatrix{T}) where {T <: CuBlasT} =
    cuBLAS.syrk!(uplo, 'N', _blas_scalar(T, α), A, _blas_scalar(T, β), C)
SDS.vendor_herk!(uplo::Char, α, A::StridedCuMatrix{T}, β, C::StridedCuMatrix{T}) where {T <: CuBlasC} =
    cuBLAS.herk!(uplo, 'N', _blas_scalar(real(T), α), A, _blas_scalar(real(T), β), C)
SDS.vendor_trsm!(side::Char, uplo::Char, trans::Char, diag::Char, α, A::StridedCuMatrix{T},
                 B::StridedCuMatrix{T}) where {T <: CuBlasT} =
    cuBLAS.trsm!(side, uplo, trans, diag, _blas_scalar(T, α), A, B)

SDS.vendor_potrf!(uplo::Char, A::StridedCuMatrix{<:CuBlasT}) = Int(cuSOLVER.potrf!(uplo, A)[2])

# cuSOLVER potrf writing its `devInfo` straight into `info[idx]`: unlike
# `cuSOLVER.potrf!` there is no host read of the status (numeric phase, PLAN §3.9)
for (bname, fname, elty) in ((:cusolverDnSpotrf_bufferSize, :cusolverDnSpotrf, :Float32),
                             (:cusolverDnDpotrf_bufferSize, :cusolverDnDpotrf, :Float64),
                             (:cusolverDnCpotrf_bufferSize, :cusolverDnCpotrf, :ComplexF32),
                             (:cusolverDnZpotrf_bufferSize, :cusolverDnZpotrf, :ComplexF64))
    @eval function SDS.vendor_potrf_info!(uplo::Char, A::StridedCuMatrix{$elty}, info::CuVector{Cint}, idx::Integer)
        n = LinearAlgebra.checksquare(A)
        lda = max(1, stride(A, 2))
        dh = cuSOLVER.dense_handle()
        function bufferSize()
            out = Ref{Cint}(0)
            cuSOLVER.$bname(dh, uplo, n, A, lda, out)
            return out[] * sizeof($elty)
        end
        CUDACore.with_workspace(dh.workspace_gpu, bufferSize) do buffer
            cuSOLVER.$fname(dh, uplo, n, A, lda, buffer, sizeof(buffer) ÷ sizeof($elty), pointer(info, idx))
        end
        return info
    end
end

# cuSOLVER pivots are `Cint`; other index vectors go through a temporary
function _with_cint(f, ipiv::CuVector, k::Integer)
    ipiv isa CuVector{Cint} && return f(ipiv)
    p = CuVector{Cint}(undef, k)
    info = f(p)
    view(ipiv, 1:k) .= p
    return info
end

function SDS.vendor_getrf!(A::StridedCuMatrix{<:CuBlasT}, ipiv::CuVector{<:Integer})
    return _with_cint(ipiv, min(size(A)...)) do p
        Int(cuSOLVER.getrf!(A, p)[3])
    end
end

function SDS.vendor_sytrf!(uplo::Char, A::StridedCuMatrix{<:CuBlasT}, ipiv::CuVector{<:Integer})
    return _with_cint(ipiv, LinearAlgebra.checksquare(A)) do p
        Int(cuSOLVER.sytrf!(uplo, A, p)[3])
    end
end

SDS.vendor_gemm_strided_batched!(tA::Char, tB::Char, α, A::StridedCuArray{T, 3}, B::StridedCuArray{T, 3}, β,
                                 C::StridedCuArray{T, 3}) where {T <: CuBlasT} =
    cuBLAS.gemm_strided_batched!(tA, tB, _blas_scalar(T, α), A, B, _blas_scalar(T, β), C)

function _check_square_batch(A::AbstractArray{<:Any, 3})
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("batch members are $(size(A, 1))×$(size(A, 2)), not square"))
    return size(A, 1)
end

# device vector of member pointers of a strided 3-D batch (members `stride(A, 3)` apart)
function _batch_pointers(A::StridedCuArray{T, 3}) where {T}
    base = Base.unsafe_convert(CuPtr{T}, A)
    s = stride(A, 3) * sizeof(T)
    return CuArray([base + (i - 1) * s for i in 1:size(A, 3)])
end

for (fname, fname_64, elty) in ((:cublasStrsmBatched, :cublasStrsmBatched_64, :Float32),
                                (:cublasDtrsmBatched, :cublasDtrsmBatched_64, :Float64),
                                (:cublasCtrsmBatched, :cublasCtrsmBatched_64, :ComplexF32),
                                (:cublasZtrsmBatched, :cublasZtrsmBatched_64, :ComplexF64))
    @eval function SDS.vendor_trsm_batched!(side::Char, uplo::Char, trans::Char, diag::Char, α,
                                            A::StridedCuArray{$elty, 3}, B::StridedCuArray{$elty, 3})
        m, n, nb = size(B)
        nb == 0 && return B
        lda, ldb = max(1, stride(A, 2)), max(1, stride(B, 2))
        GC.@preserve A B begin
            Aptrs = _batch_pointers(A)
            Bptrs = _batch_pointers(B)
            a = _blas_scalar($elty, α)
            if cuBLAS.version() >= v"12.0"
                cuBLAS.$fname_64(cuBLAS.handle(), side, uplo, trans, diag, m, n, a, Aptrs, lda, Bptrs, ldb, nb)
            else
                cuBLAS.$fname(cuBLAS.handle(), side, uplo, trans, diag, m, n, a, Aptrs, lda, Bptrs, ldb, nb)
            end
            CUDACore.unsafe_free!(Aptrs)
            CUDACore.unsafe_free!(Bptrs)
        end
        return B
    end
end

for (fname, elty) in ((:cusolverDnSpotrfBatched, :Float32), (:cusolverDnDpotrfBatched, :Float64),
                      (:cusolverDnCpotrfBatched, :ComplexF32), (:cusolverDnZpotrfBatched, :ComplexF64))
    @eval function SDS.vendor_potrf_batched!(uplo::Char, A::StridedCuArray{$elty, 3}, info::CuVector{Cint})
        n = _check_square_batch(A)
        nb = size(A, 3)
        nb == 0 && return info
        GC.@preserve A begin
            Aptrs = _batch_pointers(A)
            cuSOLVER.$fname(cuSOLVER.dense_handle(), uplo, n, Aptrs, max(1, stride(A, 2)), info, nb)
            CUDACore.unsafe_free!(Aptrs)
        end
        return info
    end
end

function SDS.vendor_getrf_batched!(A::StridedCuArray{T, 3}, ipiv::CuMatrix{Cint}, info::CuVector{Cint}) where {T <: CuBlasT}
    n = _check_square_batch(A)
    size(A, 3) == 0 && return info
    GC.@preserve A begin
        cuBLAS.getrf_batched!(n, _batch_pointers(A), max(1, stride(A, 2)), ipiv, info)
    end
    return info
end

end # module SparseDirectSolverCUDAExt
