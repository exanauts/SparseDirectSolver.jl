# Unblocked right-looking Cholesky in one workgroup per matrix (regime C
# fallback; PLAN §2.4). The status of the factorization lives in `@localmem`
# and is copied to the device vector `info` at the end, so the launch needs no
# host synchronization.

# Element (i, k), i ≥ k, of the lower factor view of A: A[i, k] for 'L',
# conj(A[k, i]) for 'U' (A = UᴴU with U = Lᴴ).
@inline _chol_get(A, ::Val{LOWER}, i, k, b) where {LOWER} = LOWER ? _get(A, i, k, b) : conj(_get(A, k, i, b))
@inline _chol_set!(A, v, ::Val{LOWER}, i, k, b) where {LOWER} = LOWER ? _set!(A, v, i, k, b) : _set!(A, conj(v), k, i, b)

@kernel function _ka_potrf_kernel!(A, info, lower::Val, ::Val{WG}, n, offset) where {WG}
    @uniform T = eltype(A)
    st = @localmem Int32 (1,)
    li = @index(Local, Linear)
    gi = @index(Group, NTuple)
    if li == 1
        @inbounds st[1] = Int32(0)
    end
    @synchronize
    for j in 1:n
        if li == 1 && st[1] == 0
            d = real(_chol_get(A, lower, j, j, gi[2]))
            if d > 0
                _chol_set!(A, T(sqrt(d)), lower, j, j, gi[2])
            else
                @inbounds st[1] = Int32(j)
            end
        end
        @synchronize
        if st[1] == 0
            ljj = real(_chol_get(A, lower, j, j, gi[2]))
            for i in (j + li):WG:n
                _chol_set!(A, _chol_get(A, lower, i, j, gi[2]) / ljj, lower, i, j, gi[2])
            end
        end
        @synchronize
        if st[1] == 0
            r = n - j
            for q in (li - 1):WG:(r * r - 1)
                i = j + 1 + q % r
                k = j + 1 + q ÷ r
                if i >= k
                    v = _chol_get(A, lower, i, k, gi[2]) -
                        _chol_get(A, lower, i, j, gi[2]) * conj(_chol_get(A, lower, k, j, gi[2]))
                    _chol_set!(A, v, lower, i, k, gi[2])
                end
            end
        end
        @synchronize
    end
    if li == 1
        @inbounds info[offset + gi[2]] = st[1]
    end
end

"""
    ka_potrf!(uplo, A, info; workgroup = Val(128), offset = 0) -> info

KernelAbstractions fallback Cholesky of the `uplo` triangle of `A` in place
(`A = LLᴴ` for `'L'`, `UᴴU` for `'U'`), unblocked, one 1-D workgroup per
matrix. `A` is a matrix or a 3-D strided batch; `info` is a device vector of
`Int32` with one entry per matrix, set to 0 on success or to the first column
`j` (1-based) whose pivot is not positive, as LAPACK `potrf`; matrix `b` of
the batch writes `info[offset + b]`. Asynchronous: the caller reads `info` at a
phase boundary.
"""
function ka_potrf!(uplo, A::AbstractArray, info::AbstractVector{Int32}; workgroup::Val{WG} = Val(KA_WORKGROUP),
                   offset::Integer = 0) where {WG}
    ul = _uplo_char(uplo)
    _ilog2(WG)
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("potrf: matrices are $(size(A, 1))×$(size(A, 2)), not square"))
    nb = _nbatch(A)
    0 <= offset && offset + nb <= length(info) ||
        throw(DimensionMismatch("info has length $(length(info)) < offset $offset + batch count $nb"))
    nb == 0 && return info
    backend = KernelAbstractions.get_backend(A)
    kernel! = _ka_potrf_kernel!(backend, (WG, 1))
    kernel!(A, info, Val(ul == 'L'), workgroup, n, Int(offset); ndrange = (WG, nb))
    return info
end

@kernel function _ka_chol_diag_kernel!(info, A, n)
    gi = @index(Group, NTuple)
    li = @index(Local, Linear)
    if li == 1
        st = Int32(0)
        for j in 1:n
            d = _get(A, j, j, gi[2])
            if !(isfinite(real(d)) && real(d) > 0 && iszero(imag(d)))
                st = Int32(j)
                break
            end
        end
        @inbounds info[gi[2]] = st
    end
end

"""
    ka_chol_diag_info!(info, L) -> info

Pivot check of a computed Cholesky factor `L` (matrix or 3-D strided batch):
`info[b]` is set to the first `j` whose diagonal entry `L[j, j]` is not a
finite positive real, 0 if there is none. Used to validate the `info = 0` of
vendor and generic factorizations (see [`potrf!`](@ref)). Asynchronous.
"""
function ka_chol_diag_info!(info::AbstractVector{Int32}, L::AbstractArray)
    nb = _nbatch(L)
    length(info) >= nb || throw(DimensionMismatch("info has length $(length(info)) < batch count $nb"))
    nb == 0 && return info
    kernel! = _ka_chol_diag_kernel!(KernelAbstractions.get_backend(L), (1, 1))
    kernel!(info, L, size(L, 1); ndrange = (1, nb))
    return info
end

@kernel function _ka_chol_check_kernel!(info, A, n, idx)
    li = @index(Local, Linear)
    if li == 1
        @inbounds if info[idx] == 0
            st = Int32(0)
            for j in 1:n
                d = _get(A, j, j, 1)
                if !(isfinite(real(d)) && real(d) > 0 && iszero(imag(d)))
                    st = Int32(j)
                    break
                end
            end
            info[idx] = st
        end
    end
end

"""
    ka_chol_check_info!(info, idx, L) -> info

Device-side validation of the `info[idx]` a vendor or generic Cholesky left for
the computed factor `L` (a matrix): when `info[idx] == 0`, it is replaced by
the first `j` whose diagonal entry `L[j, j]` is not a finite positive real
(still 0 if there is none); a nonzero `info[idx]` is kept. Asynchronous (see
[`potrf_info!`](@ref)).
"""
function ka_chol_check_info!(info::AbstractVector{Int32}, idx::Integer, L::AbstractMatrix)
    1 <= idx <= length(info) || throw(DimensionMismatch("info index $idx outside 1:$(length(info))"))
    kernel! = _ka_chol_check_kernel!(KernelAbstractions.get_backend(L), 1)
    kernel!(info, L, size(L, 1), Int(idx); ndrange = 1)
    return info
end
