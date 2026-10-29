module SparseDirectSolverCUDAExt

# CUDA support: CSR adapters (T02), the static shared-memory cap of the regime-A
# kernels (T23, issue #60) and the vendor dense bindings of the dense
# layer (T03, PLAN §2.6): cuBLAS gemm/syrk/herk/trsm/strided-batched gemm/
# batched trsm, cuSOLVER potrf (also with a device info)/getrf/potrfBatched;
# the batched trsm/potrf also on member pointers built once
# (uniform batch, T17).

using SparseDirectSolver
using LinearAlgebra
using CUDACore
using cuSPARSE
using cuBLAS
using cuSOLVER

const SDS = SparseDirectSolver

# CSR adapters and the public API (T13, PLAN §3.1) on cuSPARSE matrices: solver
# constructors, update!, the LinearAlgebra layer (see `@vendor_sparse_glue`)
SDS.@vendor_sparse_glue(cuSPARSE.CuSparseMatrixCSR, CuSparseMatrixCSC, CuVector, CuArray,
                        device = "GPU", cudss = true)

"""
    max_local_bytes(::CUDABackend) -> Int

48 KiB: `@localmem` is static shared memory, and ptxas rejects more than
48 KiB of static shared data per kernel on every CUDA device (larger amounts
need dynamic shared memory with an opt-in attribute). The regime-A kernels
therefore stop at the 48 KiB class on CUDA (issue #60).
"""
SparseDirectSolver.max_local_bytes(::CUDABackend) = 49152

"""
    max_threads_per_multiprocessor(::CUDABackend) -> Int

`CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR` of the current device.
"""
SparseDirectSolver.max_threads_per_multiprocessor(::CUDABackend) =
    Int(CUDACore.attribute(CUDACore.device(), CUDACore.DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR))

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

SDS.vendor_free!(x::CuArray) = CUDACore.unsafe_free!(x)

# cuSOLVER pivots are `Cint`; other index vectors go through a temporary
function SDS.vendor_getrf!(A::StridedCuMatrix{<:CuBlasT}, ipiv::CuVector{<:Integer})
    return SDS._with_pivot_vector(ipiv, min(size(A)...), Cint) do p
        Int(cuSOLVER.getrf!(A, p)[3])
    end
end

SDS.vendor_gemm_strided_batched!(tA::Char, tB::Char, α, A::StridedCuArray{T, 3}, B::StridedCuArray{T, 3}, β,
                                 C::StridedCuArray{T, 3}) where {T <: CuBlasT} =
    cuBLAS.gemm_strided_batched!(tA, tB, _blas_scalar(T, α), A, B, _blas_scalar(T, β), C)

# device vector of member pointers of a strided 3-D batch (members `stride(A, 3)` apart), built
# once per regime-C front of a uniform batch (T17): the numeric phase then allocates nothing
function SDS.vendor_batch_pointers(A::StridedCuArray{T, 3}) where {T <: CuBlasT}
    base = Base.unsafe_convert(CuPtr{T}, A)
    s = stride(A, 3) * sizeof(T)
    return CuArray([base + (i - 1) * s for i in 1:size(A, 3)])
end

SDS.vendor_trsm_batched!(side::Char, uplo::Char, trans::Char, diag::Char, α, A::StridedCuArray{T, 3},
                         B::StridedCuArray{T, 3}) where {T <: CuBlasT} =
    SDS._trsm_batched_via_ptrs!(side, uplo, trans, diag, α, A, B)
SDS.vendor_potrf_batched!(uplo::Char, A::StridedCuArray{T, 3}, info::CuVector{Cint}) where {T <: CuBlasT} =
    SDS._potrf_batched_via_ptrs!(uplo, A, info)

for (fname, fname_64, elty) in ((:cublasStrsmBatched, :cublasStrsmBatched_64, :Float32),
                                (:cublasDtrsmBatched, :cublasDtrsmBatched_64, :Float64),
                                (:cublasCtrsmBatched, :cublasCtrsmBatched_64, :ComplexF32),
                                (:cublasZtrsmBatched, :cublasZtrsmBatched_64, :ComplexF64))
    @eval function SDS.vendor_trsm_batched_ptrs!(side::Char, uplo::Char, trans::Char, diag::Char, α, m::Integer,
                                                 n::Integer, Aptrs::StridedCuArray{CuPtr{$elty}, 1}, lda::Integer,
                                                 Bptrs::StridedCuArray{CuPtr{$elty}, 1}, ldb::Integer, count::Integer)
        count == 0 && return Bptrs
        a = _blas_scalar($elty, α)
        if cuBLAS.version() >= v"12.0"
            cuBLAS.$fname_64(cuBLAS.handle(), side, uplo, trans, diag, m, n, a, Aptrs, max(1, lda), Bptrs, max(1, ldb),
                             count)
        else
            cuBLAS.$fname(cuBLAS.handle(), side, uplo, trans, diag, m, n, a, Aptrs, max(1, lda), Bptrs, max(1, ldb),
                          count)
        end
        return Bptrs
    end
end

for (fname, elty) in ((:cusolverDnSpotrfBatched, :Float32), (:cusolverDnDpotrfBatched, :Float64),
                      (:cusolverDnCpotrfBatched, :ComplexF32), (:cusolverDnZpotrfBatched, :ComplexF64))
    @eval function SDS.vendor_potrf_batched_ptrs!(uplo::Char, n::Integer, Aptrs::StridedCuArray{CuPtr{$elty}, 1},
                                                  lda::Integer, info::StridedCuArray{Cint, 1}, count::Integer)
        count == 0 && return info
        cuSOLVER.$fname(cuSOLVER.dense_handle(), uplo, n, Aptrs, max(1, lda), info, count)
        return info
    end
end

end # module SparseDirectSolverCUDAExt
