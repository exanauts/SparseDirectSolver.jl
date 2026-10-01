# Shared helpers of the KernelAbstractions fallback kernels (PLAN §2.6 item 3).
#
# Rules (PLAN §2.4, §2.7): 1-D workgroups, `@localmem` reductions with sizes
# from `Val` parameters, no subgroup intrinsics, no atomics. Every kernel takes
# matrices or 3-D strided batches; `_get`/`_set!` hide the batch index so one
# kernel serves both. On the KA CPU backend a kernel body is split at every
# `@synchronize` and only `@index` assignments are replayed per work item, so
# values derived from the local index are recomputed after each barrier.

"Default 1-D workgroup size of the single-workgroup kernels (potrf, getrf, trsm)."
const KA_WORKGROUP = 128

@inline _get(A::AbstractMatrix, i, j, b) = @inbounds A[i, j]
@inline _get(A::AbstractArray{<:Any, 3}, i, j, b) = @inbounds A[i, j, b]
@inline _set!(A::AbstractMatrix, v, i, j, b) = (@inbounds A[i, j] = v; nothing)
@inline _set!(A::AbstractArray{<:Any, 3}, v, i, j, b) = (@inbounds A[i, j, b] = v; nothing)

# Element (i, j) of op(A) for op ∈ 'N' (A), 'T' (transpose), 'C' (adjoint).
@inline function _op_get(A, ::Val{TR}, i, j, b) where {TR}
    if TR === 'N'
        return _get(A, i, j, b)
    elseif TR === 'T'
        return _get(A, j, i, b)
    else
        return conj(_get(A, j, i, b))
    end
end

# Whether (i, j) lies in the triangle `UPLO` ('L', 'U') or anywhere ('F').
@inline _in_triangle(::Val{UPLO}, i, j) where {UPLO} = UPLO === 'L' ? i >= j : UPLO === 'U' ? i <= j : true

_nbatch(A::AbstractMatrix) = 1
_nbatch(A::AbstractArray{<:Any, 3}) = size(A, 3)

_ilog2(n::Int) = (ispow2(n) || throw(InvalidValueError("workgroup size $n is not a power of two")); trailing_zeros(n))
