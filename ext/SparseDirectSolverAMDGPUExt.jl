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
using SparseDirectSolver: CSR, DirectSolver, INDEX_ONE, INDEX_ZERO, InvalidValueError
using SparseArrays
using LinearAlgebra
using AMDGPU
using AMDGPU: HIP, rocBLAS, rocSOLVER, rocSPARSE
using AMDGPU.rocSPARSE: ROCSparseMatrixCSR, ROCSparseMatrixCSC

const SDS = SparseDirectSolver

"""
    CSR(A::ROCSparseMatrixCSR)

Wrap `A` without copying: the `CSR` shares `rowPtr`, `colVal` and `nzVal`
(one-based, `transposed = false`).
"""
function SparseDirectSolver.CSR(A::ROCSparseMatrixCSR)
    m, n = size(A)
    return CSR(A.rowPtr, A.colVal, A.nzVal, m, n; index = INDEX_ONE, transposed = false)
end

"""
    CSR(A::ROCSparseMatrixCSC)

Wrap `A` (`m × n`) without copying as the CSR of `transpose(A)` (`n × m`,
`transposed = true`), sharing `colPtr`, `rowVal` and `nzVal`.
"""
function SparseDirectSolver.CSR(A::ROCSparseMatrixCSC)
    m, n = size(A)
    return CSR(A.colPtr, A.rowVal, A.nzVal, n, m; index = INDEX_ONE, transposed = true)
end

"""
    ROCSparseMatrixCSR(A::CSR)

The stored CSR matrix of `A` (the `transposed` flag is not applied) as a
`ROCSparseMatrixCSR`. One-based arrays are shared; zero-based index arrays are
copied and rebased. Batches (`nbatch(A) > 1`) are rejected.
"""
function rocSPARSE.ROCSparseMatrixCSR(A::CSR{T, INT, <:ROCVector{INT}, <:ROCArray{T}}) where {T, INT}
    nbatch(A) == 1 || throw(InvalidValueError("cannot convert a batch of $(nbatch(A)) matrices to ROCSparseMatrixCSR"))
    nzval = A.nzval isa ROCVector ? A.nzval : vec(A.nzval)
    rowptr, colval = A.rowptr, A.colval
    if A.index == INDEX_ZERO
        rowptr = rowptr .+ one(INT)
        colval = colval .+ one(INT)
    end
    return ROCSparseMatrixCSR{T, INT}(rowptr, colval, nzval, (A.nrows, A.ncols))
end

"""
    to_backend(A::SparseMatrixCSC, ::ROCBackend; index = 'O') -> CSR

CSR arrays of `A` built on the host, then uploaded as `ROCVector`s.
"""
function SparseDirectSolver.to_backend(A::SparseMatrixCSC, ::ROCBackend; index = INDEX_ONE)
    B = CSR(A; index)
    return CSR(ROCVector(B.rowptr), ROCVector(B.colval), ROCVector(B.nzval), B.nrows, B.ncols;
               index = B.index, transposed = B.transposed)
end

"""
    max_local_bytes(::ROCBackend) -> Int

The local data share (LDS) of one workgroup on the current HIP device
(`hipDeviceAttributeMaxSharedMemoryPerBlock`, 64 KiB on CDNA): the regime-A
kernels may use the 64 KiB class there (issue #60).
"""
SparseDirectSolver.max_local_bytes(::ROCBackend) =
    Int(HIP.attribute(AMDGPU.device(), HIP.hipDeviceAttributeMaxSharedMemoryPerBlock))

# ---------------------------------------------------------------------------
# public API (T13, PLAN §3.1): solver constructors, update!, the LinearAlgebra layer

"""
    DirectSolver(A::ROCSparseMatrixCSR, structure::String, view::Char; index = 'O')
    DirectSolver(A::ROCSparseMatrixCSC, structure::String, view::Char; index = 'O')

Solver on `A`'s arrays, without copies (the ROCm twin of
`DirectSolver(::CuSparseMatrixCSR, …)`). A `ROCSparseMatrixCSC` is read as the
CSR of its transpose with the view flipped, see [`DirectSolver`](@ref).
"""
SparseDirectSolver.DirectSolver(A::ROCSparseMatrixCSR, structure, view; index = INDEX_ONE) =
    DirectSolver(CSR(A), structure, view; index)
SparseDirectSolver.DirectSolver(A::ROCSparseMatrixCSC, structure, view; index = INDEX_ONE) =
    DirectSolver(CSR(A), structure, view; index)

"""
    update!(solver::DirectSolver, A::ROCSparseMatrixCSR)
    update!(solver::DirectSolver, A::ROCSparseMatrixCSC)

Point `solver` at the arrays of `A`, see [`update!`](@ref).
"""
SparseDirectSolver.update!(solver::DirectSolver, A::Union{ROCSparseMatrixCSR, ROCSparseMatrixCSC}) =
    SparseDirectSolver.update!(solver, CSR(A))

"""
    cholesky(A::ROCSparseMatrixCSR, NoPivot(); view = 'F', check = false) -> DirectSolver
    cholesky(Symmetric(A::ROCSparseMatrixCSR)) / cholesky(Hermitian(A::ROCSparseMatrixCSR))

LLᵀ/LLᴴ factorization of `A` on the AMD GPU; the wrappers pass their `uplo` as
the view. See `cholesky(::CSR)`.
"""
LinearAlgebra.cholesky(A::ROCSparseMatrixCSR, p::NoPivot = NoPivot(); view::Char = 'F', check::Bool = false) =
    cholesky(CSR(A), p; view, check)
LinearAlgebra.cholesky(A::Symmetric{T, <:ROCSparseMatrixCSR{T}}, p::NoPivot = NoPivot();
                       check::Bool = false) where {T <: Union{Float32, Float64}} =
    cholesky(CSR(A.data), p; view = A.uplo, check)
LinearAlgebra.cholesky(A::Hermitian{T, <:ROCSparseMatrixCSR{T}}, p::NoPivot = NoPivot(); check::Bool = false) where {T} =
    cholesky(CSR(A.data), p; view = A.uplo, check)

"""
    ldlt(A::ROCSparseMatrixCSR; view = 'F', check = false) -> DirectSolver
    ldlt(Symmetric(A::ROCSparseMatrixCSR)) / ldlt(Hermitian(A::ROCSparseMatrixCSR))

LDLᵀ/LDLᴴ factorization of `A` on the AMD GPU; the wrappers pass their `uplo`
as the view. See `ldlt(::CSR)`.
"""
LinearAlgebra.ldlt(A::ROCSparseMatrixCSR; view::Char = 'F', check::Bool = false) = ldlt(CSR(A); view, check)
LinearAlgebra.ldlt(A::Symmetric{T, <:ROCSparseMatrixCSR{T}}; check::Bool = false) where {T <: Real} =
    ldlt(CSR(A.data); view = A.uplo, check)
LinearAlgebra.ldlt(A::Hermitian{T, <:ROCSparseMatrixCSR{T}}; check::Bool = false) where {T} =
    ldlt(CSR(A.data); view = A.uplo, check)

"""
    lu(A::ROCSparseMatrixCSR; check = false) -> DirectSolver

`L D U` factorization (structure `"G"`) of `A` on the AMD GPU. See `lu(::CSR)`.
"""
LinearAlgebra.lu(A::ROCSparseMatrixCSR; check::Bool = false) = lu(CSR(A); check)

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

# rocSOLVER pivots are `Cint`; other index vectors go through a temporary
function _with_cint(f, ipiv::ROCVector, k::Integer)
    ipiv isa ROCVector{Cint} && return f(ipiv)
    p = ROCVector{Cint}(undef, k)
    info = f(p)
    view(ipiv, 1:k) .= p
    return info
end

function SDS.vendor_getrf!(A::StridedROCMatrix{<:RocBlasT}, ipiv::ROCVector{<:Integer})
    return _with_cint(ipiv, min(size(A)...)) do p
        Int(rocSOLVER.getrf!(A, p)[3])
    end
end

function SDS.vendor_sytrf!(uplo::Char, A::StridedROCMatrix{<:RocBlasT}, ipiv::ROCVector{<:Integer})
    return _with_cint(ipiv, LinearAlgebra.checksquare(A)) do p
        Int(rocSOLVER.sytrf!(uplo, A, p)[3])
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

function _check_square_batch(A::AbstractArray{<:Any, 3})
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("batch members are $(size(A, 1))×$(size(A, 2)), not square"))
    return size(A, 1)
end

# device vector of member pointers of a strided 3-D batch (members `stride(A, 3)` apart)
function _batch_pointers(A::StridedROCArray{T, 3}) where {T}
    base = Base.unsafe_convert(Ptr{T}, A)
    s = stride(A, 3) * sizeof(T)
    return ROCArray([base + (i - 1) * s for i in 1:size(A, 3)])
end

# built once per regime-C front of a uniform batch (T17): the numeric phase then allocates nothing
SDS.vendor_batch_pointers(A::StridedROCArray{T, 3}) where {T <: RocBlasT} = _batch_pointers(A)

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
    @eval function SDS.vendor_trsm_batched!(side::Char, uplo::Char, trans::Char, diag::Char, α,
                                            A::StridedROCArray{$elty, 3}, B::StridedROCArray{$elty, 3})
        m, n, nb = size(B)
        nb == 0 && return B
        GC.@preserve A B begin
            Aptrs = _batch_pointers(A)
            Bptrs = _batch_pointers(B)
            SDS.vendor_trsm_batched_ptrs!(side, uplo, trans, diag, α, m, n, Aptrs, stride(A, 2), Bptrs, stride(B, 2),
                                          nb)
            AMDGPU.unsafe_free!(Aptrs)
            AMDGPU.unsafe_free!(Bptrs)
        end
        return B
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
    @eval function SDS.vendor_potrf_batched!(uplo::Char, A::StridedROCArray{$elty, 3}, info::StridedROCArray{Cint, 1})
        n = _check_square_batch(A)
        nb = size(A, 3)
        nb == 0 && return info
        GC.@preserve A begin
            Aptrs = _batch_pointers(A)
            SDS.vendor_potrf_batched_ptrs!(uplo, n, Aptrs, stride(A, 2), info, nb)
            AMDGPU.unsafe_free!(Aptrs)
        end
        return info
    end
end

for (fname, elty) in ((:rocsolver_sgetrf_batched, :Float32), (:rocsolver_dgetrf_batched, :Float64),
                      (:rocsolver_cgetrf_batched, :ComplexF32), (:rocsolver_zgetrf_batched, :ComplexF64))
    @eval function SDS.vendor_getrf_batched!(A::StridedROCArray{$elty, 3}, ipiv::ROCMatrix{Cint},
                                             info::ROCVector{Cint})
        n = _check_square_batch(A)
        nb = size(A, 3)
        size(ipiv, 1) >= n && size(ipiv, 2) >= nb && length(info) >= nb ||
            throw(DimensionMismatch("getrf_batched: ipiv is $(size(ipiv)), info has $(length(info)) entries"))
        nb == 0 && return info
        GC.@preserve A begin
            Aptrs = _batch_pointers(A)
            rocSOLVER.$fname(rocBLAS.handle(), n, n, Aptrs, _lda(A), ipiv, size(ipiv, 1), info, nb)
            AMDGPU.unsafe_free!(Aptrs)
        end
        return info
    end
end

end # module SparseDirectSolverAMDGPUExt
