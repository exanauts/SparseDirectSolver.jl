# Triangular solve with multiple right-hand sides, one workgroup per right-hand
# side line (a column of B for side 'L', a row for side 'R'): substitution over
# the order of A, the pivot of each step in `@localmem`, the remaining entries
# of the line updated in parallel.
#
# Both sides reduce to `M y = c` per line, with M[i, k] read from A as
#   side 'L': M = op(A)         side 'R': M = transpose(op(A))   (x op(A) = b ⇔ op(A)ᵀ xᵀ = bᵀ)
# so M[i, k] is A[i, k] or A[k, i] (SWAP), optionally conjugated (CONJ), and M is
# lower (forward substitution) or upper (backward) triangular.

@inline _trsm_m(A, ::Val{SWAP}, ::Val{CONJ}, i, k, b) where {SWAP, CONJ} =
    (v = SWAP ? _get(A, k, i, b) : _get(A, i, k, b); CONJ ? conj(v) : v)
@inline _line_get(B, ::Val{ROWLINE}, r, p, b) where {ROWLINE} = ROWLINE ? _get(B, r, p, b) : _get(B, p, r, b)
@inline _line_set!(B, v, ::Val{ROWLINE}, r, p, b) where {ROWLINE} = ROWLINE ? _set!(B, v, r, p, b) : _set!(B, v, p, r, b)

@kernel function _ka_trsm_kernel!(B, A, α, swap::Val, cj::Val, ::Val{LOWER}, ::Val{UNIT}, rowline::Val, ::Val{WG},
                                  n) where {LOWER, UNIT, WG}
    @uniform T = eltype(B)
    xj = @localmem T (1,)
    li = @index(Local, Linear)
    gi = @index(Group, NTuple)
    for p in li:WG:n
        _line_set!(B, α * _line_get(B, rowline, gi[1], p, gi[2]), rowline, gi[1], p, gi[2])
    end
    @synchronize
    for jj in 1:n
        if li == 1
            j = LOWER ? jj : n - jj + 1
            v = _line_get(B, rowline, gi[1], j, gi[2])
            if !UNIT
                v = v / _trsm_m(A, swap, cj, j, j, gi[2])
                _line_set!(B, v, rowline, gi[1], j, gi[2])
            end
            @inbounds xj[1] = v
        end
        @synchronize
        j = LOWER ? jj : n - jj + 1
        lo = LOWER ? j + 1 : 1
        hi = LOWER ? n : j - 1
        @inbounds x = xj[1]
        for p in (lo + li - 1):WG:hi
            v = _line_get(B, rowline, gi[1], p, gi[2]) - _trsm_m(A, swap, cj, p, j, gi[2]) * x
            _line_set!(B, v, rowline, gi[1], p, gi[2])
        end
        @synchronize
    end
end

# one launch with a static configuration (every flag a `Val`, see `_static_bool`)
function _ka_trsm_run!(B, A, α, swap::Val, cj::Val, lower::Val, unit::Val, rowline::Val{ROWLINE},
                       workgroup::Val{WG}) where {ROWLINE, WG}
    n = ROWLINE ? size(B, 2) : size(B, 1)
    nlines = ROWLINE ? size(B, 1) : size(B, 2)
    nb = _nbatch(B)
    (n == 0 || nlines == 0 || nb == 0) && return B
    backend = KernelAbstractions.get_backend(B)
    kernel! = _ka_trsm_kernel!(backend, (WG, 1))
    kernel!(B, A, eltype(B)(α), swap, cj, lower, unit, rowline, workgroup, n; ndrange = (WG * nlines, nb))
    return B
end

# M reads A transposed (SWAP) for side 'L' with op ≠ 'N' and for side 'R' with op = 'N'
@inline _trsm_swap(::Val{ROWLINE}, ::Val{TR}) where {ROWLINE, TR} = Val(ROWLINE ? TR === 'N' : TR !== 'N')
@inline _trsm_conj(::Val{TR}) where {TR} = Val(TR === 'C')

function _ka_trsm_launch!(side::Char, uplo::Char, trans::Char, diag::Char, α, A, B, workgroup::Val{WG}) where {WG}
    _ilog2(WG)
    left = side == 'L'
    # M is lower triangular (forward substitution) when A is lower and not transposed for side 'L', when
    # A is upper and not transposed for side 'R' (M = op(A)ᵀ), or the transposed counterparts
    lower = left ? (uplo == 'L') == (trans == 'N') : (uplo == 'U') == (trans == 'N')
    return _static_bool(!left) do vrow
        _static_trans(trans) do vtr
            _static_bool(lower) do vlower
                _static_bool(diag == 'U') do vunit
                    _ka_trsm_run!(B, A, α, _trsm_swap(vrow, vtr), _trsm_conj(vtr), vlower, vunit, vrow, workgroup)
                end
            end
        end
    end
end

"""
    ka_trsm!(side, uplo, trans, diag, α, A, B; workgroup = Val(128)) -> B

KernelAbstractions fallback for BLAS `trsm`: `B ← α op(A)⁻¹ B` (`side = 'L'`)
or `B ← α B op(A)⁻¹` (`side = 'R'`), `A` triangular (`uplo`), unit diagonal if
`diag = 'U'`, `op` from `trans ∈ ('N', 'T', 'C')`. One 1-D workgroup per
right-hand side column (row for `side = 'R'`) runs the forward/back
substitution. Asynchronous; the launch configuration is static, so a call
allocates nothing on the host.
"""
function ka_trsm!(side, uplo, trans, diag, α, A::AbstractMatrix, B::AbstractMatrix; workgroup::Val = Val(KA_WORKGROUP))
    sd, ul, tr, dg = _side_char(side), _uplo_char(uplo), _trans_char(trans), _diag_char(diag)
    _check_trsm_dims(sd, A, B)
    return _ka_trsm_launch!(sd, ul, tr, dg, α, A, B, workgroup)
end

"""
    ka_trsm_strided_batched!(side, uplo, trans, diag, α, A, B; workgroup = Val(128)) -> B

Strided batched [`ka_trsm!`](@ref) on 3-D arrays (batch = third dimension, the
second ndrange dimension of the launch).
"""
function ka_trsm_strided_batched!(side, uplo, trans, diag, α, A::AbstractArray{<:Any, 3}, B::AbstractArray{<:Any, 3};
                                  workgroup::Val = Val(KA_WORKGROUP))
    sd, ul, tr, dg = _side_char(side), _uplo_char(uplo), _trans_char(trans), _diag_char(diag)
    _check_trsm_dims(sd, A, B)
    _check_batch(B, A)
    return _ka_trsm_launch!(sd, ul, tr, dg, α, A, B, workgroup)
end
