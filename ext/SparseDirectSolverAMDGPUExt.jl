module SparseDirectSolverAMDGPUExt

# AMDGPU (ROCm) support (T23), mirroring `SparseDirectSolverCUDAExt`: CSR adapters
# for rocSPARSE matrices, the public API on them (T13), the vendor dense bindings
# of the dense layer (PLAN §2.6) on rocBLAS/rocSOLVER, and the backend's local
# memory per workgroup for the regime-A kernels (issue #60).
#
# rocBLAS handles of AMDGPU.jl run in host pointer mode, so α/β are host `Ref`s
# (no device allocation per call, unlike cuBLAS's device pointer mode). The
# AMDGPU.jl high-level wrappers are used where they take strided views and do not
# read a status on the host; the other routines are called through the
# low-level `rocblas_*`/`rocsolver_*` bindings.

using SparseDirectSolver
using LinearAlgebra
using AMDGPU
using AMDGPU: HIP, rocBLAS, rocSOLVER, rocSPARSE
using AMDGPU.rocSPARSE: ROCSparseMatrixCSR, ROCSparseMatrixCSC

const SDS = SparseDirectSolver

# CSR adapters and the public API (T13, PLAN §3.1) on rocSPARSE matrices: solver
# constructors, update!, the LinearAlgebra layer (see `@vendor_sparse_glue`)
SDS.@vendor_sparse_glue(rocSPARSE.ROCSparseMatrixCSR, ROCSparseMatrixCSC, ROCVector, ROCArray,
                        device = "AMD GPU", platform = "ROCm")

"""
    max_local_bytes(::ROCBackend) -> Int

The local data share (LDS) of one workgroup on the current HIP device
(`hipDeviceAttributeMaxSharedMemoryPerBlock`, 64 KiB on CDNA): the regime-A
kernels may use the 64 KiB class there (issue #60).
"""
SparseDirectSolver.max_local_bytes(::ROCBackend) =
    Int(HIP.attribute(AMDGPU.device(), HIP.hipDeviceAttributeMaxSharedMemoryPerBlock))

# ---------------------------------------------------------------------------
# vendor dense bindings (see `src/dense/vendor.jl` for the contracts)

const RocBlasT = Union{Float32, Float64, ComplexF32, ComplexF64}

_lda(A) = max(1, stride(A, 2))

SDS.vendor_gemm!(tA::Char, tB::Char, α, A::StridedROCMatrix{T}, B::StridedROCMatrix{T}, β,
                 C::StridedROCMatrix{T}) where {T <: RocBlasT} =
    rocBLAS.gemm!(tA, tB, T(α), A, B, T(β), C)
SDS.vendor_syrk!(uplo::Char, α, A::StridedROCMatrix{T}, β, C::StridedROCMatrix{T}) where {T <: RocBlasT} =
    rocBLAS.syrk!(uplo, 'N', T(α), A, T(β), C)
SDS.vendor_trsm!(side::Char, uplo::Char, trans::Char, diag::Char, α, A::StridedROCMatrix{T},
                 B::StridedROCMatrix{T}) where {T <: RocBlasT} =
    rocBLAS.trsm!(side, uplo, trans, diag, T(α), A, B)

# `rocBLAS.herk!` takes contiguous `ROCMatrix`es only: the panel views go through the C binding
for (fname, elty) in ((:rocblas_cherk, :ComplexF32), (:rocblas_zherk, :ComplexF64))
    @eval function SDS.vendor_herk!(uplo::Char, α, A::StridedROCMatrix{$elty}, β, C::StridedROCMatrix{$elty})
        n = LinearAlgebra.checksquare(C)
        size(A, 1) == n || throw(DimensionMismatch("herk: C is $n×$n, A has $(size(A, 1)) rows"))
        rocBLAS.$fname(rocBLAS.handle(), uplo, 'N', n, size(A, 2), Ref(real($elty)(α)), A, _lda(A),
                       Ref(real($elty)(β)), C, _lda(C))
        return C
    end
end

SDS.vendor_potrf!(uplo::Char, A::StridedROCMatrix{<:RocBlasT}) = Int(rocSOLVER.potrf!(uplo, A)[2])

# rocSOLVER potrf writing its status straight into `info[idx]`: unlike `rocSOLVER.potrf!`
# there is no host read of the status (numeric phase, PLAN §3.9)
for (fname, elty) in ((:rocsolver_spotrf, :Float32), (:rocsolver_dpotrf, :Float64),
                      (:rocsolver_cpotrf, :ComplexF32), (:rocsolver_zpotrf, :ComplexF64))
    @eval function SDS.vendor_potrf_info!(uplo::Char, A::StridedROCMatrix{$elty}, info::ROCVector{Cint}, idx::Integer)
        n = LinearAlgebra.checksquare(A)
        1 <= idx <= length(info) || throw(DimensionMismatch("info index $idx outside 1:$(length(info))"))
        rocSOLVER.$fname(rocBLAS.handle(), uplo, n, A, _lda(A), pointer(info, idx))
        return info
    end
end

SDS.vendor_free!(x::ROCArray) = AMDGPU.unsafe_free!(x)

# rocSOLVER pivots are `Cint`; other index vectors go through a temporary
function SDS.vendor_getrf!(A::StridedROCMatrix{<:RocBlasT}, ipiv::ROCVector{<:Integer})
    return SDS._with_pivot_vector(ipiv, min(size(A)...), Cint) do p
        Int(rocSOLVER.getrf!(A, p)[3])
    end
end

# `rocBLAS.gemm_strided_batched!` takes `ROCArray{T, 3}` only; the solver passes strided batch views
for (fname, elty) in ((:rocblas_sgemm_strided_batched, :Float32), (:rocblas_dgemm_strided_batched, :Float64),
                      (:rocblas_cgemm_strided_batched, :ComplexF32), (:rocblas_zgemm_strided_batched, :ComplexF64))
    @eval function SDS.vendor_gemm_strided_batched!(tA::Char, tB::Char, α, A::StridedROCArray{$elty, 3},
                                                    B::StridedROCArray{$elty, 3}, β, C::StridedROCArray{$elty, 3})
        m = size(A, tA == 'N' ? 1 : 2)
        k = size(A, tA == 'N' ? 2 : 1)
        n = size(B, tB == 'N' ? 2 : 1)
        (m == size(C, 1) && n == size(C, 2) && k == size(B, tB == 'N' ? 1 : 2)) ||
            throw(DimensionMismatch("gemm_strided_batched: C is $(size(C, 1))×$(size(C, 2)), op(A) $m×$k"))
        nb = size(C, 3)
        size(A, 3) == size(B, 3) == nb || throw(DimensionMismatch("gemm_strided_batched: batch counts differ"))
        nb == 0 && return C
        rocBLAS.$fname(rocBLAS.handle(), tA, tB, m, n, k, Ref($elty(α)), A, _lda(A), stride(A, 3), B, _lda(B),
                       stride(B, 3), Ref($elty(β)), C, _lda(C), stride(C, 3), nb)
        return C
    end
end

# device vector of member pointers of a strided 3-D batch (members `stride(A, 3)` apart), built
# once per regime-C front of a uniform batch (T17): the numeric phase then allocates nothing
function SDS.vendor_batch_pointers(A::StridedROCArray{T, 3}) where {T <: RocBlasT}
    base = Base.unsafe_convert(Ptr{T}, A)
    s = stride(A, 3) * sizeof(T)
    return ROCArray([base + (i - 1) * s for i in 1:size(A, 3)])
end

SDS.vendor_trsm_batched!(side::Char, uplo::Char, trans::Char, diag::Char, α, A::StridedROCArray{T, 3},
                         B::StridedROCArray{T, 3}) where {T <: RocBlasT} =
    SDS._trsm_batched_via_ptrs!(side, uplo, trans, diag, α, A, B)
SDS.vendor_potrf_batched!(uplo::Char, A::StridedROCArray{T, 3}, info::StridedROCArray{Cint, 1}) where {T <: RocBlasT} =
    SDS._potrf_batched_via_ptrs!(uplo, A, info)

for (fname, elty) in ((:rocblas_strsm_batched, :Float32), (:rocblas_dtrsm_batched, :Float64),
                      (:rocblas_ctrsm_batched, :ComplexF32), (:rocblas_ztrsm_batched, :ComplexF64))
    @eval function SDS.vendor_trsm_batched_ptrs!(side::Char, uplo::Char, trans::Char, diag::Char, α, m::Integer,
                                                 n::Integer, Aptrs::StridedROCArray{Ptr{$elty}, 1}, lda::Integer,
                                                 Bptrs::StridedROCArray{Ptr{$elty}, 1}, ldb::Integer, count::Integer)
        count == 0 && return Bptrs
        rocBLAS.$fname(rocBLAS.handle(), side, uplo, trans, diag, m, n, Ref($elty(α)), Aptrs, max(1, lda), Bptrs,
                       max(1, ldb), count)
        return Bptrs
    end
end

for (fname, elty) in ((:rocsolver_spotrf_batched, :Float32), (:rocsolver_dpotrf_batched, :Float64),
                      (:rocsolver_cpotrf_batched, :ComplexF32), (:rocsolver_zpotrf_batched, :ComplexF64))
    @eval function SDS.vendor_potrf_batched_ptrs!(uplo::Char, n::Integer, Aptrs::StridedROCArray{Ptr{$elty}, 1},
                                                  lda::Integer, info::StridedROCArray{Cint, 1}, count::Integer)
        count == 0 && return info
        rocSOLVER.$fname(rocBLAS.handle(), uplo, n, Aptrs, max(1, lda), info, count)
        return info
    end
end

end # module SparseDirectSolverAMDGPUExt
