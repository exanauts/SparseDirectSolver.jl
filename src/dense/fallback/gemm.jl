# Tiled GEMM / SYRK / HERK and their strided batched variants.
#
# One workgroup of TILE² work items (1-D) computes one TILE × TILE tile of C;
# the A and B tiles are staged in `@localmem`. The batch index is the second
# ndrange dimension. `UPLO` restricts the update to a triangle of C (syrk/herk).

@kernel function _ka_gemm_kernel!(C, A, B, α, β, tA::Val, tB::Val, uplo::Val, ::Val{HERM}, ::Val{TILE},
                                  m, n, k, ntm) where {HERM, TILE}
    @uniform T = eltype(C)
    As = @localmem T (TILE, TILE)
    Bs = @localmem T (TILE, TILE)
    acc = @private T (1,)
    li = @index(Local, Linear)
    gi = @index(Group, NTuple)
    acc[1] = zero(T)
    for kk in 0:TILE:(k - 1)
        ti = (li - 1) % TILE + 1
        tj = (li - 1) ÷ TILE + 1
        b = gi[2]
        i = ((gi[1] - 1) % ntm) * TILE + ti
        j = ((gi[1] - 1) ÷ ntm) * TILE + tj
        @inbounds As[ti, tj] = (i <= m && kk + tj <= k) ? _op_get(A, tA, i, kk + tj, b) : zero(T)
        @inbounds Bs[ti, tj] = (kk + ti <= k && j <= n) ? _op_get(B, tB, kk + ti, j, b) : zero(T)
        @synchronize
        ti = (li - 1) % TILE + 1
        tj = (li - 1) ÷ TILE + 1
        s = acc[1]
        for p in 1:TILE
            @inbounds s += As[ti, p] * Bs[p, tj]
        end
        acc[1] = s
        @synchronize
    end
    ti = (li - 1) % TILE + 1
    tj = (li - 1) ÷ TILE + 1
    b = gi[2]
    i = ((gi[1] - 1) % ntm) * TILE + ti
    j = ((gi[1] - 1) ÷ ntm) * TILE + tj
    if i <= m && j <= n && _in_triangle(uplo, i, j)
        v = iszero(β) ? α * acc[1] : α * acc[1] + β * _get(C, i, j, b)
        if HERM && i == j
            v = T(real(v))
        end
        _set!(C, v, i, j, b)
    end
end

function _ka_gemm_launch!(C, A, B, α, β, tA::Char, tB::Char, uplo::Char, herm::Bool, tile::Val{TILE}) where {TILE}
    m, n = size(C, 1), size(C, 2)
    k = size(A, tA == 'N' ? 2 : 1)
    nb = _nbatch(C)
    (m == 0 || n == 0 || nb == 0) && return C
    T = eltype(C)
    ntm = cld(m, TILE)
    ntiles = ntm * cld(n, TILE)
    backend = KernelAbstractions.get_backend(C)
    kernel! = _ka_gemm_kernel!(backend, (TILE * TILE, 1))
    kernel!(C, A, B, T(α), T(β), Val(tA), Val(tB), Val(uplo), Val(herm), tile, m, n, k, ntm;
            ndrange = (TILE * TILE * ntiles, nb))
    return C
end

_default_tile(m, n) = min(m, n) >= 64 ? Val(32) : Val(16)

"""
    ka_gemm!(C, A, B, α, β; transA = 'N', transB = 'N', tile = Val(16) or Val(32)) -> C

KernelAbstractions fallback for `C ← α op(A) op(B) + β C` (`op` from `'N'`, `'T'`,
`'C'`). Tiled, one TILE × TILE tile of `C` per 1-D workgroup of TILE² work
items; `tile` defaults to `Val(32)` when both dimensions of `C` are at least 64.
`C` is not read when `β == 0`. Asynchronous: no host synchronization.
"""
function ka_gemm!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, α, β;
                  transA = 'N', transB = 'N', tile = _default_tile(size(C)...))
    tA, tB = _trans_char(transA), _trans_char(transB)
    _check_gemm_dims(C, A, B, tA, tB)
    return _ka_gemm_launch!(C, A, B, α, β, tA, tB, 'F', false, tile)
end

"""
    ka_syrk!(C, A, α, β; uplo = 'L', conjugate = false, tile) -> C

KernelAbstractions fallback for the `uplo` triangle of `C ← α A Aᵀ + β C`
(`conjugate = true`: `α A Aᴴ + β C` with real `α`, `β` and a real diagonal, as
BLAS `herk`). The other triangle is not touched.
"""
function ka_syrk!(C::AbstractMatrix, A::AbstractMatrix, α, β; uplo = 'L', conjugate::Bool = false,
                  tile = _default_tile(size(C)...))
    ul = _uplo_char(uplo)
    _check_syrk_dims(C, A)
    return _ka_gemm_launch!(C, A, A, α, β, 'N', conjugate ? 'C' : 'T', ul, conjugate, tile)
end

"""
    ka_gemm_strided_batched!(C, A, B, α, β; transA = 'N', transB = 'N', tile) -> C

Strided batched [`ka_gemm!`](@ref): `A`, `B`, `C` are 3-D arrays (for example
[`strided_batch`](@ref) views of a flat buffer) whose third dimension is the
batch; one launch, the batch index is the second ndrange dimension.
"""
function ka_gemm_strided_batched!(C::AbstractArray{<:Any, 3}, A::AbstractArray{<:Any, 3}, B::AbstractArray{<:Any, 3},
                                  α, β; transA = 'N', transB = 'N', tile = _default_tile(size(C, 1), size(C, 2)))
    tA, tB = _trans_char(transA), _trans_char(transB)
    _check_gemm_dims(C, A, B, tA, tB)
    _check_batch(C, A, B)
    return _ka_gemm_launch!(C, A, B, α, β, tA, tB, 'F', false, tile)
end
