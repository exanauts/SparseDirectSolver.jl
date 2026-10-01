# Unblocked right-looking Cholesky in one workgroup per matrix (regime C
# fallback; PLAN §2.4). The status of the factorization lives in `@localmem`
# and is copied to the device vector `info` at the end, so the launch needs no
# host synchronization.

# Element (i, k), i ≥ k, of the lower factor view of A: A[i, k] for 'L',
# conj(A[k, i]) for 'U' (A = UᴴU with U = Lᴴ).
@inline _chol_get(A, ::Val{LOWER}, i, k, b) where {LOWER} = LOWER ? _get(A, i, k, b) : conj(_get(A, k, i, b))
@inline _chol_set!(A, v, ::Val{LOWER}, i, k, b) where {LOWER} = LOWER ? _set!(A, v, i, k, b) : _set!(A, conj(v), k, i, b)

@kernel function _ka_potrf_kernel!(A, info, lower::Val, ::Val{WG}, n) where {WG}
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
        @inbounds info[gi[2]] = st[1]
    end
end

"""
    ka_potrf!(uplo, A, info; workgroup = Val(128)) -> info

KernelAbstractions fallback Cholesky of the `uplo` triangle of `A` in place
(`A = LLᴴ` for `'L'`, `UᴴU` for `'U'`), unblocked, one 1-D workgroup per
matrix. `A` is a matrix or a 3-D strided batch; `info` is a device vector of
`Int32` with one entry per matrix, set to 0 on success or to the first column
`j` (1-based) whose pivot is not positive, as LAPACK `potrf`. Asynchronous:
the caller reads `info` at a phase boundary.
"""
function ka_potrf!(uplo, A::AbstractArray, info::AbstractVector{Int32}; workgroup::Val{WG} = Val(KA_WORKGROUP)) where {WG}
    ul = _uplo_char(uplo)
    _ilog2(WG)
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("potrf: matrices are $(size(A, 1))×$(size(A, 2)), not square"))
    nb = _nbatch(A)
    length(info) >= nb || throw(DimensionMismatch("info has length $(length(info)) < batch count $nb"))
    nb == 0 && return info
    backend = KernelAbstractions.get_backend(A)
    kernel! = _ka_potrf_kernel!(backend, (WG, 1))
    kernel!(A, info, Val(ul == 'L'), workgroup, n; ndrange = (WG, nb))
    return info
end
