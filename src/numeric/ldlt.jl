# Device LDLᵀ/LDLᴴ (PLAN §2.4, §3.3): the in-front factorization of the CPU
# reference (`src/reference/ldlt.jl`, T14) as KA kernels, with the same pivot
# sequence: Bunch–Kaufman 1×1/2×2 pivots inside the fully-summed block,
# `pivot_threshold` acceptance against the whole remaining front column, static
# perturbation `±ε` with the `pivot_sign` policy, and the contribution block
# updated once per front with `F₂₂ ← F₂₂ − (L₂₁ D) L₂₁ᴴ`.
#
# * Regime A: `subtree_ldlt_kernel!`, one workgroup per subtree, the serial
#   stack of packed fronts and contribution blocks in `@localmem` (as the
#   Cholesky kernel of `src/numeric/subtree.jl`).
# * Regimes B and C: `front_ldlt_kernel!`, one workgroup per front of a launch
#   group, fused assembly (zero, scatter A, owner-pull extend-add), then the
#   pivoted factorization of the panel (regime B: `F₁₁` staged in `@localmem`,
#   the rows below it in global memory; regime C: all in global memory) and the
#   update of the front's packed contribution block on the update stack.
#
# Per pivot step (a loop over the `w` columns of the front with a uniform trip
# count; a 2×2 pivot leaves the last iterations idle): the workgroup chooses the
# pivot (the reference's `_choose_pivot`, its column maxima as workgroup
# reductions with the reference's tie-breaking, in one to three passes; work
# item 1 decides and stores D, perturbing a tiny 1×1 pivot, and counts the
# statistics), swaps rows/columns, and applies the rank-1/rank-2 update to the
# remaining fully-summed columns while reducing the next column (pass 1 of the
# next step). The pivot columns stay unscaled (`L D`) until the end of the
# front, when `_lt_finalize!` divides them by D exactly as the reference does
# after each step; then the contribution block is updated (regimes B/C: tiled
# through local memory for blocks of `_LT_TILED_M` rows or more). Four barriers
# per step when the first pass settles the pivot.
# Only the lower triangle of the front is stored; the upper entries the
# algorithm reads are `conj` (Hermitian) or plain (complex symmetric) mirrors.
# Pivot decisions, control words and the per-front counts live in `@localmem`;
# the counts go to `numeric.stats` and `reduce_stats!` sums them. No atomics:
# every destination has one writer per segment, so the result is deterministic.

"Workgroup size of the LDLᵀ/LDLᴴ front kernel (regimes B and C)."
const LDLT_WORKGROUP = 128

# pivot search of the device kernels (`_ldlt_pivot_type` of the reference)
const _LT_PIVOT_NONE = 0
const _LT_PIVOT_DIAGONAL = 1
const _LT_PIVOT_BK = 2

# control slots of the LDLᵀ kernels, after the regime-A slots of `subtree.jl`
# (the front kernel uses `_ST_NODE`, `_ST_F`, `_ST_W` and `_ST_LF` = panel offset)
const _LT_C0 = _ST_CTL + 1     # first factor column super_ptr[v]
const _LT_K = _ST_CTL + 2      # current local column
const _LT_STEP = _ST_CTL + 3   # size of the pivot taken at _LT_K (1 or 2; 0 before the first)
const _LT_C = _ST_CTL + 4      # chosen 1×1 pivot column (2×2: k)
const _LT_R = _ST_CTL + 5      # 2×2 partner column (0: 1×1 pivot)
const _LT_STAT = _ST_CTL + 5   # _LT_STAT + q: statistic q ∈ 1:5 (STAT_NPOS … STAT_N2X2) of the front
const _LT_PHASE = _ST_CTL + 11 # pivot search state: 0 chosen, 2 Bunch–Kaufman pass 2 due, 3 fallback scan due
const _LT_RBK = _ST_CTL + 12   # Bunch–Kaufman candidate row r of pass 1 (0: none)
const _LT_NTILE = _ST_CTL + 13 # tiles of the F₂₂ update of the regime-B/C kernel (0: no contribution block)
const _LT_KEND = _ST_CTL + 14  # last column a pivot step may start at (w; the block's last column in regime C)
const _LT_CTL = _ST_CTL + 14

# `pv` slots: 1–4 the pivot (d, or the inverse 2×2 block e11, e12, e21, e22), 5–6 the new diagonal
# entries of the pivot block (d, or a and c), 7–8 λ and the column maximum of pass 1
const _LT_NPV = 8

# lanes of the stage-1 combine of the cooperative pivot search (`NL` partial results → `_LT_S1` → 1)
const _LT_S1 = 16

# resolved pivoting parameters passed to the kernels (isbits); `H`: Hermitian (or real symmetric) vs complex
# symmetric (a type parameter, so the kernels need no extra `Val` argument: launches with more than 32
# arguments are not specialized and allocate on the KA CPU backend)
struct _LDLTDevice{R, H}
    ptype::Int      # _LT_PIVOT_NONE, _LT_PIVOT_DIAGONAL or _LT_PIVOT_BK
    u::R            # pivot_threshold
    eps::R          # pivot_epsilon (multiplied by max |aᵢⱼ| = real(aux[1]) when `scaled`)
    scaled::Bool    # pivot_epsilon_alg = "algo1"
end

# Hermitian (or real symmetric) vs complex symmetric
_ldlt_herm(S::Symbolic, ::Type{T}) where {T} = !(T <: Complex) || S.structure == STRUCTURE_HERMITIAN

# `H` must be a compile-time constant (`Val`): with a runtime flag the parameter type, and every call that
# takes it, would be resolved at run time
function _ldlt_device_params(S::Symbolic, ::Type{T}, opts::Options, ::Val{H} = Val(_ldlt_herm(S, T))) where {T, H}
    _is_ldlt_structure(S.structure) ||
        throw(InvalidValueError("LDLᵀ/LDLᴴ needs structure \"S\" or \"H\", got \"$(convert(String, S.structure))\""))
    R = real(T)
    p = _ldlt_pivot_type(opts.pivot_type)
    ptype = p == PIVOT_NONE ? _LT_PIVOT_NONE : p == PIVOT_DIAGONAL ? _LT_PIVOT_DIAGONAL : _LT_PIVOT_BK
    return _LDLTDevice{R, H}(ptype, R(opts.pivot_threshold), R(resolved_pivot_epsilon(opts, R)),
                             opts.pivot_epsilon_alg == PIVOT_EPSILON_SCALED)
end

# effective perturbation ε (as `_ldlt_params`: ε · max |aᵢⱼ| when scaled and the matrix is not zero)
@inline function _lt_eps(p::_LDLTDevice, aux)
    p.scaled || return p.eps
    @inbounds a = real(aux[1])
    return a > 0 ? p.eps * a : p.eps
end

# ---------------------------------------------------------------------------
# lower triangle of a front: packed `f×f` in local memory (regime A) or the
# `f×w` column-major panel in global memory (regimes B/C, columns ≤ w only)

struct _PackedFront{A}
    a::A
    off::Int
    f::Int
end

struct _PanelFront{A}
    a::A
    off::Int
    f::Int
end

@inline _fpos(F::_PackedFront, i, j) = F.off + _packed(i, j, F.f)
@inline _fpos(F::_PanelFront, i, j) = F.off + (j - 1) * F.f + i
@inline _fget(F, i, j) = @inbounds F.a[_fpos(F, i, j)]
@inline function _fset!(F, i, j, v)
    @inbounds F.a[_fpos(F, i, j)] = v
    return nothing
end
@inline _cj(x, ::Val{true}) = conj(x)
@inline _cj(x, ::Val{false}) = x
# |F[i, j]| of the symmetric/Hermitian front, from its lower triangle
@inline _fabs(F, i, j) = abs(i >= j ? _fget(F, i, j) : _fget(F, j, i))
# F[i, j] from the lower triangle (upper entries: conj for Hermitian, plain for complex symmetric)
@inline _fsym(F, i, j, h::Val) = i >= j ? _fget(F, i, j) : _cj(_fget(F, j, i), h)

# regime B: the fully-summed block F₁₁ (rows and columns `1:w`) packed `W×W` lower triangle in local
# memory `l`, the rows `w+1:f` in the `f×w` panel in global memory
struct _SplitFront{L, A}
    l::L
    a::A
    off::Int
    f::Int
    w::Int
    W::Int
end

@inline _fget(F::_SplitFront, i, j) = @inbounds i <= F.w ? F.l[_packed(i, j, F.W)] : F.a[F.off + (j - 1) * F.f + i]
@inline function _fset!(F::_SplitFront, i, j, v)
    @inbounds if i <= F.w
        F.l[_packed(i, j, F.W)] = v
    else
        F.a[F.off + (j - 1) * F.f + i] = v
    end
    return nothing
end

# the front of the current node from the control words: offset `_ST_LF` (1-based) into `fa`; `Val(W)`
# with `fa = (L11, factor)`: F₁₁ staged in the local `L11` (width class `W`)
@inline _lt_front(fa, ctl, ::Val{true}) = @inbounds _PackedFront(fa, Int(ctl[_ST_LF]) - 1, Int(ctl[_ST_F]))
@inline _lt_front(fa, ctl, ::Val{false}) = @inbounds _PanelFront(fa, Int(ctl[_ST_LF]) - 1, Int(ctl[_ST_F]))
@inline _lt_front(fa::Tuple, ctl, ::Val{W}) where {W} =
    @inbounds _SplitFront(fa[1], fa[2], Int(ctl[_ST_LF]) - 1, Int(ctl[_ST_F]), Int(ctl[_ST_W]), W)

# the front as stored, for the interchanges (the regime-C panel kernel moves stored values; see `_LazyFront`)
@inline _lt_raw_front(fa, ctl, pk::Val) = _lt_front(fa, ctl, pk)

# the buffers and the front mode of the regime-B/C kernel: `W = 0` panel in global memory, else F₁₁ staged
@inline _lt_fa(L11, factor, ::Val{0}) = factor
@inline _lt_fa(L11, factor, ::Val{W}) where {W} = (L11, factor)
@inline _lt_pk(::Val{0}) = Val(false)
@inline _lt_pk(w::Val) = w

# copy F₁₁ between the panel and local memory (`load`), or write it back with a unit diagonal
@inline function _lt_stage!(L11, factor, ctl, li, ::Val{W}, ::Val{WG}, load::Bool) where {W, WG}
    @inbounds if W > 0
        f = Int(ctl[_ST_F])
        w = Int(ctl[_ST_W])
        p0 = Int(ctl[_ST_LF]) - 1
        for q in (li - 1):WG:(w * w - 1)
            j = q ÷ w + 1
            i = q - (j - 1) * w + 1
            i >= j || continue
            if load
                L11[_packed(i, j, W)] = factor[p0 + (j - 1) * f + i]
            else
                factor[p0 + (j - 1) * f + i] = i == j ? one(eltype(factor)) : L11[_packed(i, j, W)]
            end
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# cooperative pivot search (the reference's `_choose_pivot` on the lower triangle, pivot for pivot)
#
# The column maxima of the search are reductions: every lane (work item `li ≤ NL`) reduces a strided
# share of the rows (or, in the fallback scan, of the candidate columns) into its slots of `red`/`redi`,
# `_lt_stage1!` folds them into `_LT_S1` slots and work item 1 into one (`_lt_final!`), then decides.
# `max` of absolute values is exact in any order (a NaN propagates as in the serial loop), and an
# arg-max keeps the first maximum (the smallest index on a tie, index 0 = none), so every decision
# equals the serial one. Slots: `red[1:NL]`, `red[NL .+ (1:NL)]`, `red[2NL .+ (1:NL)]` (values
# A, B, C), `redi[1:NL]`, `redi[NL .+ (1:NL)]` (indices of A and B). Per pivot step:
#
# * pass 1 (always; computed in the scale phase of the previous step, or before the first step):
#   A/idx = λ and r (largest `|F[i, k]|` of the block rows `k+1:w`, first maximum), B = the column
#   maximum of column k over rows `k+1:f` (the threshold test of a 1×1 pivot at k);
# * pass 2 (Bunch–Kaufman when `|a_kk| < αλ`): A = σ (column r over the block rows `k:w` but r),
#   B = column r over rows `k:f` but r and k, C = column k over rows `k:f` but k and r;
# * pass 3 (the fallback when the choice fails the threshold): A/idx = the acceptable 1×1 pivot with
#   the largest `|a_jj|` (`_best_1x1`, one lane per candidate column), B/idx = the largest `|a_jj|`
#   (pivot type 'D').

# `F[i, c]` bound of the threshold test: `d ≥ u maxᵢ |F[i, c]|` over rows `k:f` but c, with an early
# exit (`u ≥ 0` is finite: `u · max = max(u · |F[i, c]|)`, and a NaN fails both forms)
@inline function _lt_threshold_ok(F, c, k, f, u, d)
    m = zero(d)
    for i in k:f
        i == c && continue
        m = max(m, _fabs(F, i, c))
        d >= u * m || return false
    end
    return d >= u * m
end

@inline _lt_det2(F, k, r, h::Val) = _fget(F, k, k) * _fget(F, r, r) - _fsym(F, k, r, h) * _fget(F, r, k)

@inline function _lt_nonsingular_2x2(F, k, r, ε, h::Val)
    det = _lt_det2(F, k, r, h)
    return !iszero(det) && isfinite(det) && max(_fabs(F, k, k), _fabs(F, r, k), _fabs(F, r, r)) >= ε
end

# the 2×2 pivot on (k, r) with the column maxima m1 (column k without rows k, r) and m2 (column r
# without rows r, k) is acceptable (the reference's `_accept_2x2`)
@inline function _lt_accept_2x2(F, k, r, m1, m2, u, ε, h::Val)
    _lt_nonsingular_2x2(F, k, r, ε, h) || return false
    det = _lt_det2(F, k, r, h)
    e11 = abs(_fget(F, r, r) / det)
    e12 = abs(_fsym(F, k, r, h) / det)
    e21 = abs(_fget(F, r, k) / det)
    e22 = abs(_fget(F, k, k) / det)
    return u * (e11 * m1 + e21 * m2) <= 1 && u * (e12 * m1 + e22 * m2) <= 1
end

# fold arg-max slot b into slot a (first maximum, index 0 = none)
@inline function _lt_argmax!(red, redi, a, b)
    @inbounds begin
        ib = redi[b]
        if ib != 0
            ia = redi[a]
            if ia == 0 || red[b] > red[a] || (red[b] == red[a] && ib < ia)
                red[a] = red[b]
                redi[a] = ib
            end
        end
    end
    return nothing
end

# fold lane b into lane a for the partial results of pass `P`
@inline function _lt_combine!(red, redi, a, b, ::Val{NL}, ::Val{P}) where {NL, P}
    @inbounds if P == 1
        _lt_argmax!(red, redi, a, b)
        red[NL + a] = max(red[NL + a], red[NL + b])
    elseif P == 2
        red[a] = max(red[a], red[b])
        red[NL + a] = max(red[NL + a], red[NL + b])
        red[2 * NL + a] = max(red[2 * NL + a], red[2 * NL + b])
    else
        _lt_argmax!(red, redi, a, b)
        _lt_argmax!(red, redi, NL + a, NL + b)
    end
    return nothing
end

# lanes of pass `P` at column k that can hold a partial result (the others hold the neutral one): one per row
# below k (pass 1, with k the column the step advances to), per row from k (pass 2), per column from k (pass 3)
@inline _lt_nused(ctl, k, ::Val{P}) where {P} =
    @inbounds P == 3 ? Int(ctl[_ST_W]) - k + 1 : Int(ctl[_ST_F]) - k + 1

# stage 1: lanes 1:_LT_S1 fold lanes _LT_S1+1:min(NL, nu) (nothing when NL ≤ _LT_S1)
@inline function _lt_stage1!(red, redi, ctl, li, nl::Val{NL}, pass::Val{P}) where {NL, P}
    @inbounds if NL > _LT_S1 && li <= _LT_S1
        k = Int(ctl[_LT_K]) + (P == 1 ? Int(ctl[_LT_STEP]) : 0)
        for b in (li + _LT_S1):_LT_S1:min(NL, _lt_nused(ctl, k, pass))
            _lt_combine!(red, redi, li, b, nl, pass)
        end
    end
    return nothing
end

# work item 1: fold lanes 2:min(NL, _LT_S1, nu) into lane 1 (k: the column of the step)
@inline function _lt_final!(red, redi, ctl, k, nl::Val{NL}, pass::Val) where {NL}
    for b in 2:min(NL, _LT_S1, _lt_nused(ctl, k, pass))
        _lt_combine!(red, redi, 1, b, nl, pass)
    end
    return nothing
end

# pass 1 partials of the next column k = K + STEP (before the step that advances to it)
@inline function _lt_pass1!(fa, ctl, red, redi, li, ::Val{NL}, p::_LDLTDevice{R}, pk::Val) where {NL, R}
    @inbounds if li <= NL
        k = Int(ctl[_LT_K]) + Int(ctl[_LT_STEP])
        w = Int(ctl[_ST_W])
        λ = zero(R)
        r = 0
        cm = zero(R)
        if k <= Int(ctl[_LT_KEND]) && p.ptype != _LT_PIVOT_NONE
            F = _lt_front(fa, ctl, pk)
            for i in (k + li):NL:Int(ctl[_ST_F])
                a = _fabs(F, i, k)
                cm = max(cm, a)
                if i <= w && a > λ
                    λ = a
                    r = i
                end
            end
        end
        red[li] = λ
        redi[li] = r % eltype(redi)
        red[NL + li] = cm
    end
    return nothing
end

# pass 2 partials (column r of the Bunch–Kaufman candidate, column k without row r)
@inline function _lt_pass2!(fa, ctl, red, redi, li, ::Val{NL}, ::_LDLTDevice{R}, pk::Val) where {NL, R}
    @inbounds if li <= NL
        k = Int(ctl[_LT_K])
        r = Int(ctl[_LT_RBK])
        w = Int(ctl[_ST_W])
        σ = zero(R)
        mr = zero(R)
        mk = zero(R)
        F = _lt_front(fa, ctl, pk)
        for i in (k + li - 1):NL:Int(ctl[_ST_F])
            if i != r
                a = _fabs(F, i, r)
                i <= w && (σ = max(σ, a))
                i != k && (mr = max(mr, a))
                i != k && (mk = max(mk, _fabs(F, i, k)))
            end
        end
        red[li] = σ
        red[NL + li] = mr
        red[2 * NL + li] = mk
    end
    return nothing
end

# pass 3 partials: one lane per candidate column j of the block k:w
@inline function _lt_pass3!(fa, ctl, red, redi, li, ::Val{NL}, aux, p::_LDLTDevice{R}, pk::Val) where {NL, R}
    @inbounds if li <= NL
        k = Int(ctl[_LT_K])
        w = Int(ctl[_ST_W])
        f = Int(ctl[_ST_F])
        ε = _lt_eps(p, aux)
        u = p.u
        F = _lt_front(fa, ctl, pk)
        bv = zero(R)
        bi = 0
        gv = zero(R)
        gi = 0
        for j in (k + li - 1):NL:w
            a = _fabs(F, j, j)
            if p.ptype == _LT_PIVOT_DIAGONAL && !isnan(a) && (gi == 0 || a > gv)
                gv = a
                gi = j
            end
            a >= ε || continue                           # not acceptable (or NaN)
            bi != 0 && !(a > bv) && continue             # cannot replace this lane's best
            _lt_threshold_ok(F, j, k, f, u, a) || continue
            bv = a
            bi = j
        end
        IT = eltype(redi)
        red[li] = bv
        redi[li] = bi % IT
        red[NL + li] = gv
        redi[NL + li] = gi % IT
    end
    return nothing
end

# work item 1: take the pivot (c, r) at column k (`r = 0`: 1×1 at column c; else 2×2 on k and r) from the
# values before the interchange: D (perturbing a tiny 1×1 pivot), the pivot kinds, the statistics, `pv`
# (the pivot, its inverse block and the new diagonal entries that `_lt_swap!` writes) and the step size
@inline function _lt_select!(fa, ctl, pv, d, pivot_kind, piv, psign, perm, aux, p::_LDLTDevice, c, r, pk::Val,
                             h::Val{H}) where {H}
    @inbounds begin
        n = length(perm)
        IT = eltype(ctl)
        T = eltype(pv)
        k = Int(ctl[_LT_K])
        c0 = Int(ctl[_LT_C0])
        g = c0 + k - 1
        F = _lt_front(fa, ctl, pk)
        ctl[_LT_C] = c % IT
        ctl[_LT_R] = r % IT
        ctl[_LT_PHASE] = zero(IT)
        if r == 0
            ε = _lt_eps(p, aux)
            x = _fget(F, c, c)
            dk = H ? T(real(x)) : x
            kind = PIVOT_KIND_1X1
            if !(abs(dk) >= ε)                        # tiny (or NaN): perturb
                iszero(dk) && (ctl[_LT_STAT + STAT_NZERO] += one(IT))
                dk = _perturbation_sign(dk, Int(psign[perm[piv[c0 + c - 1]]]), H) * ε
                kind = PIVOT_KIND_PERTURBED
                ctl[_LT_STAT + STAT_NPERTURBED] += one(IT)
            end
            d[g] = dk
            d[n + g] = zero(T)
            pivot_kind[g] = kind
            if H
                if real(dk) > 0
                    ctl[_LT_STAT + STAT_NPOS] += one(IT)
                elseif real(dk) < 0
                    ctl[_LT_STAT + STAT_NNEG] += one(IT)
                end
            end
            pv[1] = dk
            pv[5] = dk
            ctl[_LT_STEP] = one(IT)
        else
            # after the interchange of k + 1 and r: F[k + 1, k] = F[r, k], F[k + 1, k + 1] = F[r, r]
            a = _fget(F, k, k)
            b = _fget(F, r, k)
            c2 = _fget(F, r, r)
            if H
                a = T(real(a))
                c2 = T(real(c2))
            end
            up = _cj(b, h)
            det = a * c2 - up * b
            pv[1] = c2 / det
            pv[2] = -up / det
            pv[3] = -b / det
            pv[4] = a / det
            pv[5] = a
            pv[6] = c2
            d[g] = a
            d[g + 1] = c2
            d[n + g] = b
            d[n + g + 1] = zero(T)
            pivot_kind[g] = PIVOT_KIND_2X2_FIRST
            pivot_kind[g + 1] = PIVOT_KIND_2X2_SECOND
            ctl[_LT_STAT + STAT_N2X2] += one(IT)
            if H
                if real(det) < 0
                    ctl[_LT_STAT + STAT_NPOS] += one(IT)
                    ctl[_LT_STAT + STAT_NNEG] += one(IT)
                elseif real(a) > 0
                    ctl[_LT_STAT + STAT_NPOS] += IT(2)
                else
                    ctl[_LT_STAT + STAT_NNEG] += IT(2)
                end
            end
            ctl[_LT_STEP] = IT(2)
        end
    end
    return nothing
end

# work item 1 after pass 1: advance to column k, then take the pivot at k when the search ends here,
# else request pass 2 or the fallback scan
@inline function _lt_decide1!(fa, ctl, pv, red, redi, d, pivot_kind, piv, psign, perm, aux,
                              p::_LDLTDevice{R}, nl::Val{NL}, pk::Val, h::Val) where {R, NL}
    @inbounds begin
        IT = eltype(ctl)
        k = Int(ctl[_LT_K]) + Int(ctl[_LT_STEP])
        ctl[_LT_K] = k % IT
        ctl[_LT_STEP] = zero(IT)
        ctl[_LT_PHASE] = zero(IT)
        k <= Int(ctl[_LT_KEND]) || return nothing
        if p.ptype == _LT_PIVOT_NONE
            _lt_select!(fa, ctl, pv, d, pivot_kind, piv, psign, perm, aux, p, k, 0, pk, h)
            return nothing
        end
        _lt_final!(red, redi, ctl, k, nl, Val(1))
        λ = red[1]
        r = Int(redi[1])
        cm = red[NL + 1]
        pv[7] = λ
        pv[8] = cm
        ctl[_LT_RBK] = r % IT
        ctl[_LT_C] = k % IT                              # the Bunch–Kaufman choice so far: 1×1 at k
        ctl[_LT_R] = zero(IT)
        F = _lt_front(fa, ctl, pk)
        ε = _lt_eps(p, aux)
        akk = _fabs(F, k, k)
        if p.ptype == _LT_PIVOT_BK && !(r == 0 || akk >= R(BUNCH_KAUFMAN_ALPHA) * λ)
            ctl[_LT_PHASE] = IT(2)
        elseif akk >= ε && akk >= p.u * cm
            _lt_select!(fa, ctl, pv, d, pivot_kind, piv, psign, perm, aux, p, k, 0, pk, h)
        else
            ctl[_LT_PHASE] = IT(3)
        end
    end
    return nothing
end

# work item 1 after pass 2: the rest of the Bunch–Kaufman choice and its threshold test
@inline function _lt_decide2!(fa, ctl, pv, red, redi, d, pivot_kind, piv, psign, perm, aux,
                              p::_LDLTDevice{R}, nl::Val{NL}, pk::Val, h::Val) where {R, NL}
    @inbounds begin
        IT = eltype(ctl)
        _lt_final!(red, redi, ctl, Int(ctl[_LT_K]), nl, Val(2))
        σ = red[1]
        mr = red[NL + 1]
        mk = red[2 * NL + 1]
        λ = real(pv[7])
        cm = real(pv[8])
        k = Int(ctl[_LT_K])
        r = Int(ctl[_LT_RBK])
        F = _lt_front(fa, ctl, pk)
        ε = _lt_eps(p, aux)
        u = p.u
        α = R(BUNCH_KAUFMAN_ALPHA)
        akk = _fabs(F, k, k)
        c, r2 = k, 0
        if akk * σ >= α * λ^2
            ok = akk >= ε && akk >= u * cm
        elseif (arr = _fabs(F, r, r); arr >= α * σ)
            c = r
            ok = arr >= ε && arr >= u * max(mr, _fabs(F, k, r))
        else
            r2 = r
            ok = _lt_accept_2x2(F, k, r, mk, mr, u, ε, h)
        end
        if ok
            _lt_select!(fa, ctl, pv, d, pivot_kind, piv, psign, perm, aux, p, c, r2, pk, h)
        else
            ctl[_LT_C] = c % IT
            ctl[_LT_R] = r2 % IT
            ctl[_LT_PHASE] = IT(3)
        end
    end
    return nothing
end

# work item 1 after pass 3: the acceptable 1×1 pivot with the largest |a_jj|, else the Bunch–Kaufman
# choice when it is not tiny (or, for 'D', the largest diagonal), else column k perturbed
@inline function _lt_decide3!(fa, ctl, pv, red, redi, d, pivot_kind, piv, psign, perm, aux,
                              p::_LDLTDevice, nl::Val{NL}, pk::Val, h::Val) where {NL}
    @inbounds begin
        _lt_final!(red, redi, ctl, Int(ctl[_LT_K]), nl, Val(3))
        k = Int(ctl[_LT_K])
        best = Int(redi[1])
        c, r = best, 0
        if best == 0
            F = _lt_front(fa, ctl, pk)
            ε = _lt_eps(p, aux)
            if p.ptype == _LT_PIVOT_DIAGONAL
                big = isnan(_fabs(F, k, k)) ? k : Int(redi[NL + 1])
                c = _fabs(F, big, big) >= ε ? big : k
            else
                c, r = Int(ctl[_LT_C]), Int(ctl[_LT_R])
                if r != 0 && !_lt_nonsingular_2x2(F, k, r, ε, h)
                    r = 0
                end
            end
        end
        _lt_select!(fa, ctl, pv, d, pivot_kind, piv, psign, perm, aux, p, c, r, pk, h)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# phases of one pivot step (work item `li` of `WG`; `PK`: packed local front)

# reset the per-front control words (work item 1)
@inline function _lt_reset!(ctl, c0)
    @inbounds begin
        IT = eltype(ctl)
        ctl[_LT_C0] = c0 % IT
        ctl[_LT_K] = one(IT)
        ctl[_LT_STEP] = zero(IT)
        ctl[_LT_C] = zero(IT)
        ctl[_LT_R] = zero(IT)
        ctl[_LT_PHASE] = zero(IT)
        ctl[_LT_RBK] = zero(IT)
        ctl[_LT_KEND] = ctl[_ST_W]
        for q in 1:5
            ctl[_LT_STAT + q] = zero(IT)
        end
    end
    return nothing
end

# identity local pivot order of the front's columns
@inline function _lt_init_piv!(piv, ctl, li, ::Val{WG}) where {WG}
    @inbounds begin
        c0 = Int(ctl[_LT_C0])
        for j in li:WG:Int(ctl[_ST_W])
            piv[c0 + j - 1] = Int32(c0 + j - 1)
        end
    end
    return nothing
end

# symmetric interchange of the columns/rows p < q of the front (lower triangle), as `_swap_front!`, then
# the new diagonal entries of the pivot block from `pv` (and the zero below a 2×2 block's diagonal)
@inline function _lt_swap!(fa, ctl, pv, piv, li, ::Val{WG}, pk::Val, h::Val) where {WG}
    @inbounds begin
        k = Int(ctl[_LT_K])
        if k <= Int(ctl[_LT_KEND])
            r = Int(ctl[_LT_R])
            p = r == 0 ? k : k + 1
            q = r == 0 ? Int(ctl[_LT_C]) : r
            F = _lt_raw_front(fa, ctl, pk)
            for i in li:WG:Int(ctl[_ST_F])
                if p != q
                    if i < p
                        x = _fget(F, p, i)
                        _fset!(F, p, i, _fget(F, q, i))
                        _fset!(F, q, i, x)
                    elseif i == p
                        x = _fget(F, p, p)
                        _fset!(F, p, p, _fget(F, q, q))
                        _fset!(F, q, q, x)
                    elseif i < q
                        x = _fget(F, i, p)
                        _fset!(F, i, p, _cj(_fget(F, q, i), h))
                        _fset!(F, q, i, _cj(x, h))
                    elseif i == q
                        _fset!(F, q, p, _cj(_fget(F, q, p), h))
                    else
                        x = _fget(F, i, p)
                        _fset!(F, i, p, _fget(F, i, q))
                        _fset!(F, i, q, x)
                    end
                end
                if i == k
                    _fset!(F, k, k, pv[5])
                    r == 0 || _fset!(F, k + 1, k, zero(eltype(pv)))   # the 2×2 block lives in D, L's is I
                elseif r != 0 && i == k + 1
                    _fset!(F, k + 1, k + 1, pv[6])
                end
            end
            if li == 1 && p != q
                c0 = Int(ctl[_LT_C0])
                x = piv[c0 + p - 1]
                piv[c0 + p - 1] = piv[c0 + q - 1]
                piv[c0 + q - 1] = x
            end
        end
    end
    return nothing
end

# update of the remaining fully-summed columns with the unscaled pivot column(s), as the reference:
# F[i, j] -= l_i F[k, j] (1×1, `l_i = F[i, k] / d`) or l1_i F[k, j] + l2_i F[k+1, j] (2×2). The pivot
# columns stay unscaled (W = L D) until `_lt_finalize!`, so the step needs no scale phase. The trailing
# triangle of the block is shared over the workgroup; each work item owns rows `w+1:f` and forms their
# multipliers once. With `TRACK`, the pass-1 partials of the next column are reduced from the values
# written (each lane's entries of that column come in increasing row order, as in `_lt_pass1!`).
@inline function _lt_update!(fa, ctl, pv, red, redi, li, ::Val{WG}, ::Val{TRACK}, p::_LDLTDevice{R}, pk::Val,
                             h::Val) where {WG, TRACK, R}
    @inbounds begin
        k = Int(ctl[_LT_K])
        w = Int(ctl[_ST_W])
        λ = zero(R)
        r = 0
        cm = zero(R)
        if k <= Int(ctl[_LT_KEND])
            F = _lt_front(fa, ctl, pk)
            f = Int(ctl[_ST_F])
            step = Int(ctl[_LT_STEP])
            kk = k + step - 1
            j0 = kk + 1                                   # the next column
            nt = w - kk
            for q in (li - 1):WG:(nt * nt - 1)
                i = kk + 1 + q % nt
                j = kk + 1 + q ÷ nt
                i >= j || continue
                if step == 1
                    l = _fget(F, i, k) / pv[1]
                    v = _fget(F, i, j) - l * _cj(_fget(F, j, k), h)
                else
                    x1 = _fget(F, i, k)
                    x2 = _fget(F, i, k + 1)
                    l1 = x1 * pv[1] + x2 * pv[3]
                    l2 = x1 * pv[2] + x2 * pv[4]
                    v = _fget(F, i, j) - (l1 * _cj(_fget(F, j, k), h) + l2 * _cj(_fget(F, j, k + 1), h))
                end
                _fset!(F, i, j, v)
                if TRACK && j == j0 && i > j0
                    a = abs(v)
                    cm = max(cm, a)
                    if a > λ
                        λ = a
                        r = i
                    end
                end
            end
            m = f - w
            if m >= WG                                    # one row per work item, its multipliers formed once
                for i in (w + li):WG:f
                    if step == 1
                        l = _fget(F, i, k) / pv[1]
                        for j in (kk + 1):w
                            v = _fget(F, i, j) - l * _cj(_fget(F, j, k), h)
                            _fset!(F, i, j, v)
                            TRACK && j == j0 && (cm = max(cm, abs(v)))
                        end
                    else
                        x1 = _fget(F, i, k)
                        x2 = _fget(F, i, k + 1)
                        l1 = x1 * pv[1] + x2 * pv[3]
                        l2 = x1 * pv[2] + x2 * pv[4]
                        for j in (kk + 1):w
                            v = _fget(F, i, j) - (l1 * _cj(_fget(F, j, k), h) + l2 * _cj(_fget(F, j, k + 1), h))
                            _fset!(F, i, j, v)
                            TRACK && j == j0 && (cm = max(cm, abs(v)))
                        end
                    end
                end
            else                                          # few rows: one entry per work item
                for q in (li - 1):WG:(m * nt - 1)
                    i = w + 1 + q % m
                    j = kk + 1 + q ÷ m
                    if step == 1
                        l = _fget(F, i, k) / pv[1]
                        v = _fget(F, i, j) - l * _cj(_fget(F, j, k), h)
                    else
                        x1 = _fget(F, i, k)
                        x2 = _fget(F, i, k + 1)
                        l1 = x1 * pv[1] + x2 * pv[3]
                        l2 = x1 * pv[2] + x2 * pv[4]
                        v = _fget(F, i, j) - (l1 * _cj(_fget(F, j, k), h) + l2 * _cj(_fget(F, j, k + 1), h))
                    end
                    _fset!(F, i, j, v)
                    TRACK && j == j0 && (cm = max(cm, abs(v)))
                end
            end
        end
        if TRACK
            red[li] = λ
            redi[li] = r % eltype(redi)
            red[WG + li] = cm
        end
    end
    return nothing
end

# scale the pivot columns into L, row by row (rows `2:f`, columns `1:min(i - 1, w)`): `l = x / d` (1×1) or
# `(x1 e11 + x2 e21, x1 e12 + x2 e22)` with the inverse 2×2 block recomputed from D exactly as
# `_lt_select!` formed it; the zero below a 2×2 block's diagonal stays
@inline function _lt_finalize!(fa, ctl, d, pivot_kind, li, ::Val{WG}, pk::Val, h::Val) where {WG}
    @inbounds begin
        n = length(pivot_kind)
        F = _lt_front(fa, ctl, pk)
        f = Int(ctl[_ST_F])
        w = Int(ctl[_ST_W])
        c0 = Int(ctl[_LT_C0])
        for i in (1 + li):WG:f
            jmax = min(i - 1, w)
            j = 1
            while j <= jmax
                g = c0 + j - 1
                if pivot_kind[g] == PIVOT_KIND_2X2_FIRST
                    if j + 1 <= jmax
                        a, b, c2 = d[g], d[n + g], d[g + 1]
                        up = _cj(b, h)
                        det = a * c2 - up * b
                        x1 = _fget(F, i, j)
                        x2 = _fget(F, i, j + 1)
                        _fset!(F, i, j, x1 * (c2 / det) + x2 * (-b / det))
                        _fset!(F, i, j + 1, x1 * (-up / det) + x2 * (a / det))
                    end
                    j += 2
                else
                    _fset!(F, i, j, _fget(F, i, j) / d[g])
                    j += 1
                end
            end
        end
    end
    return nothing
end

# F₂₂ ← F₂₂ − (L₂₁ D) L₂₁ᴴ on the packed m×m block `ca[coff .+ (1:m(m+1)/2)]` (`coff < 0`: no block)
@inline function _lt_cb_update!(fa, ctl, ca, coff, d, pivot_kind, li, ::Val{WG}, pk::Val,
                                h::Val{H}) where {WG, H}
    @inbounds begin
        n = length(pivot_kind)
        T = eltype(fa)
        f = Int(ctl[_ST_F])
        w = Int(ctl[_ST_W])
        m = f - w
        if coff >= 0 && m > 0
            F = _lt_front(fa, ctl, pk)
            c0 = Int(ctl[_LT_C0])
            for q in (li - 1):WG:(m * m - 1)
                jj = q ÷ m + 1
                ii = q - (jj - 1) * m + 1
                ii >= jj || continue
                i = w + ii
                j = w + jj
                acc = zero(T)
                k = 1
                while k <= w
                    g = c0 + k - 1
                    if pivot_kind[g] == PIVOT_KIND_2X2_FIRST
                        a, b, c = d[g], d[n + g], d[g + 1]
                        up = _cj(b, h)
                        x1 = _fget(F, i, k)
                        x2 = _fget(F, i, k + 1)
                        acc += (x1 * a + x2 * b) * _cj(_fget(F, j, k), h) + (x1 * up + x2 * c) * _cj(_fget(F, j, k + 1), h)
                        k += 2
                    else
                        acc += _fget(F, i, k) * d[g] * _cj(_fget(F, j, k), h)
                        k += 1
                    end
                end
                pos = coff + _packed(ii, jj, m)
                ca[pos] = (H && ii == jj) ? T(real(ca[pos]) - real(acc)) : ca[pos] - acc
            end
        end
    end
    return nothing
end

# tiled F₂₂ ← F₂₂ − (L₂₁ D) L₂₁ᴴ of the regime-B/C kernel (`L₂₁` in the panel, after `_lt_finalize!`): the
# packed m×m block is cut into `_LT_TB × _LT_TB` tiles (lower ones only); per tile and chunk of `_LT_TB`
# columns, `W₂₁ = L₂₁ D` (1×1 and 2×2 blocks, as `_lt_cb_update!`) of the tile's rows and `L₂₁` of its columns
# are staged in the local buffer `tb` (`2 _LT_TB²` entries, then the `_LT_TB²` accumulators), and every work
# item accumulates `_LT_TB² / WG` entries over the chunk in column order
const _LT_TB = 16
const _LT_TBUF = 3 * _LT_TB * _LT_TB

# widest regime-B width class that keeps its panel in global memory (staging F₁₁ costs more than it saves on
# narrow fronts: two more phases per front)
const _LT_GLOBAL_MAX_W = 16

# smallest contribution block taking the tiled update (smaller ones: `_lt_cb_update!`, one entry per work item)
const _LT_TILED_M = 64

@inline _lt_cb_ntiles(f, w, coff) = coff < 0 || f - w < _LT_TILED_M ? 0 : (nt = cld(f - w, _LT_TB); nt * (nt + 1) ÷ 2)
@inline _lt_cb_nchunks(ctl) = @inbounds cld(Int(ctl[_ST_W]), _LT_TB)

# tile `tt` (column-major over the lower tiles) → (I, J), I ≥ J
@inline function _lt_cb_tile(ctl, tt)
    nt = @inbounds cld(Int(ctl[_ST_F]) - Int(ctl[_ST_W]), _LT_TB)
    J = 1
    t = tt
    while t > nt - J + 1
        t -= nt - J + 1
        J += 1
    end
    return (J + t - 1, J)
end

# stage chunk `kc` of tile `tt`: tb[1:TB²] = W₂₁ rows of block I, tb[TB² .+ (1:TB²)] = L₂₁ rows of block J
@inline function _lt_cb_load!(tb, factor, ctl, d, pivot_kind, tt, kc, li, ::Val{WG}, h::Val) where {WG}
    @inbounds begin
        n = length(pivot_kind)
        T = eltype(tb)
        F = _lt_front(factor, ctl, Val(false))
        w = Int(ctl[_ST_W])
        m = Int(ctl[_ST_F]) - w
        c0 = Int(ctl[_LT_C0])
        I, J = _lt_cb_tile(ctl, tt)
        TB2 = _LT_TB * _LT_TB
        for e in (li - 1):WG:(2 * TB2 - 1)
            isw = e < TB2
            e2 = isw ? e : e - TB2
            ri = e2 % _LT_TB
            k = (kc - 1) * _LT_TB + e2 ÷ _LT_TB + 1
            i = ((isw ? I : J) - 1) * _LT_TB + ri + 1
            v = zero(T)
            if i <= m && k <= w
                x = _fget(F, w + i, k)
                if !isw
                    v = x
                else
                    g = c0 + k - 1
                    kd = pivot_kind[g]
                    if kd == PIVOT_KIND_2X2_FIRST
                        v = x * d[g] + _fget(F, w + i, k + 1) * d[n + g]
                    elseif kd == PIVOT_KIND_2X2_SECOND
                        v = _fget(F, w + i, k - 1) * _cj(d[n + g - 1], h) + x * d[g]
                    else
                        v = x * d[g]
                    end
                end
            end
            tb[e + 1] = v
        end
    end
    return nothing
end

# accumulate the staged chunk into the tile accumulators tb[2TB² .+ (1:TB²)]
@inline function _lt_cb_acc!(tb, ctl, kc, li, ::Val{WG}, h::Val) where {WG}
    @inbounds begin
        TB2 = _LT_TB * _LT_TB
        nk = min(_LT_TB, Int(ctl[_ST_W]) - (kc - 1) * _LT_TB)
        for e in (li - 1):WG:(TB2 - 1)
            r = e % _LT_TB
            c = e ÷ _LT_TB
            acc = kc == 1 ? zero(eltype(tb)) : tb[2 * TB2 + e + 1]
            for q in 0:(nk - 1)
                acc += tb[r + q * _LT_TB + 1] * _cj(tb[TB2 + c + q * _LT_TB + 1], h)
            end
            tb[2 * TB2 + e + 1] = acc
        end
    end
    return nothing
end

# subtract the tile from the packed block (real diagonal for Hermitian)
@inline function _lt_cb_store!(tb, ctl, ca, coff, tt, li, ::Val{WG}, ::Val{H}) where {WG, H}
    @inbounds begin
        T = eltype(tb)
        m = Int(ctl[_ST_F]) - Int(ctl[_ST_W])
        I, J = _lt_cb_tile(ctl, tt)
        TB2 = _LT_TB * _LT_TB
        for e in (li - 1):WG:(TB2 - 1)
            i = (I - 1) * _LT_TB + e % _LT_TB + 1
            j = (J - 1) * _LT_TB + e ÷ _LT_TB + 1
            (i <= m && j <= m && i >= j) || continue
            acc = tb[2 * TB2 + e + 1]
            pos = coff + _packed(i, j, m)
            ca[pos] = (H && i == j) ? T(real(ca[pos]) - real(acc)) : ca[pos] - acc
        end
    end
    return nothing
end

# work item 1: per-front statistics and status
@inline function _lt_stats!(stats, info, ctl, s)
    @inbounds begin
        base = (Int(s) - 1) * FRONT_STATS_FIELDS
        for q in 1:5
            stats[base + q] = Int64(ctl[_LT_STAT + q])
        end
        stats[base + STAT_INFO] = 0
        info[s] = Int32(0)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# regimes B/C: one workgroup per front, panel in global memory

@inline function _lt_front_setup!(ctl, s, super_ptr, front_ptr, front_nrows, front_ncols, cb_ptr)
    @inbounds begin
        IT = eltype(ctl)
        ctl[_ST_NODE] = s % IT
        ctl[_ST_F] = front_nrows[s] % IT
        ctl[_ST_W] = front_ncols[s] % IT
        ctl[_ST_LF] = front_ptr[s] % IT
        _lt_reset!(ctl, super_ptr[s])
        ctl[_LT_NTILE] = _lt_cb_ntiles(Int(front_nrows[s]), Int(front_ncols[s]), Int(cb_ptr[s]) - 1) % IT
    end
    return nothing
end

# unit diagonal of the panel, statistics and status
@inline function _lt_front_finish!(factor, stats, info, ctl, li, ::Val{WG}) where {WG}
    @inbounds begin
        F = _lt_front(factor, ctl, Val(false))
        for j in li:WG:Int(ctl[_ST_W])
            _fset!(F, j, j, one(eltype(factor)))
        end
        li == 1 && _lt_stats!(stats, info, ctl, ctl[_ST_NODE])
    end
    return nothing
end

"""
    front_ldlt_kernel!(backend, WG)(factor, stack, info, stats, d, piv, pivot_kind, psign, perm, aux, nzval, amap,
                                    amap_ptr, amap_src, nodes, bm, super_ptr, front_ptr, front_nrows,
                                    front_ncols, cb_ptr, child_ptr, child_list, relind_ptr, relind, maxchild, prm,
                                    Val(W), Val(WG); ndrange = WG * count * bm.nact)

LDLᵀ/LDLᴴ kernel of regimes B and C: workgroup `G` takes front
`s = nodes[bm.first + g - 1]` of batch member `k` (`g`, `k` from the
[`BatchMap`](@ref) `bm`; the per-member arrays are the member's), zeroes and assembles it (A through the `amap`, then
the `maxchild` children's packed contribution blocks in `child_list` order),
factors its `w` fully-summed columns in place with in-block
Bunch–Kaufman pivoting (`W = 0`, regime C: in the panel; regime B, width
class `W ≥ w`: `F₁₁` staged as a packed `W×W` lower triangle in `@localmem`,
the rows below it in the panel; the pivot search is a workgroup reduction), threshold acceptance and perturbation (`prm`, a
`_LDLTDevice{R, HERM}`; `HERM`: Hermitian or real symmetric, else complex symmetric),
writes D, the local pivot order `piv`, the pivot kinds and the front's
statistics, updates its packed contribution block on the update stack
(`cb_ptr[s] > 0`) with `F₂₂ − (L₂₁ D) L₂₁ᴴ` and leaves a unit-lower panel.
"""
@kernel function front_ldlt_kernel!(factor, stack, info, stats, d, piv, pivot_kind, psign, perm, aux, nzval, amap,
                                    amap_ptr, amap_src, nodes, bm, super_ptr, front_ptr, front_nrows, front_ncols,
                                    cb_ptr, child_ptr, child_list, relind_ptr, relind, maxchild,
                                    prm::_LDLTDevice{R, HERM}, ::Val{W}, ::Val{WG}) where {R, HERM, W, WG}
    @uniform TT = eltype(factor)
    @uniform IT = eltype(front_ptr)
    li = @index(Local, Linear)
    G = @index(Group, Linear)
    ctl = @localmem IT (_LT_CTL,)
    pv = @localmem TT (_LT_NPV,)
    red = @localmem R (3 * WG,)
    redi = @localmem IT (2 * WG,)
    L11 = @localmem TT (max(W * (W + 1) ÷ 2, _LT_TBUF),)          # F₁₁, then the tiles of the F₂₂ update
    if li == 1
        @inbounds s = nodes[bm.first + _bm_node(bm, G) - 1]
        _lt_front_setup!(ctl, s, super_ptr, member_panels(front_ptr, _bm_gmember(bm, G), bm.nbatch), front_nrows,
                         front_ncols, cb_ptr)
    end
    @synchronize
    @inbounds s = nodes[bm.first + _bm_node(bm, G) - 1]
    k = _bm_gmember(bm, G)
    nb = bm.nbatch
    _zero_front!(factor, _mview(stack, k, nb), s, li, member_panels(front_ptr, k, nb), front_nrows, front_ncols,
                 cb_ptr, Val(WG))
    _lt_init_piv!(_mview(piv, k, nb), ctl, li, Val(WG))
    @synchronize
    @inbounds s = nodes[bm.first + _bm_node(bm, G) - 1]
    k = _bm_gmember(bm, G)
    _scatter_front!(factor, _mview(nzval, k, bm.nbatch), amap, amap_ptr, amap_src, s,
                    _member_shift(front_ptr, s, k, bm.nbatch), li, Val(WG))
    @synchronize
    for kc in 1:maxchild
        @inbounds s = nodes[bm.first + _bm_node(bm, G) - 1]
        k = _bm_gmember(bm, G)
        _extend_add_child!(factor, _mview(stack, k, bm.nbatch), s, kc, li, member_panels(front_ptr, k, bm.nbatch),
                           front_nrows, front_ncols, cb_ptr, child_ptr, child_list, relind_ptr, relind, Val(WG))
        @synchronize
    end
    _lt_stage!(L11, factor, ctl, li, Val(W), Val(WG), true)
    @synchronize
    _lt_pass1!(_lt_fa(L11, factor, Val(W)), ctl, red, redi, li, Val(WG), prm, _lt_pk(Val(W)))
    @synchronize
    for it in 1:ctl[_ST_W]
        if WG > _LT_S1
            _lt_stage1!(red, redi, ctl, li, Val(WG), Val(1))
            @synchronize
        end
        if li == 1
            k = _bm_gmember(bm, G)
            nb = bm.nbatch
            _lt_decide1!(_lt_fa(L11, factor, Val(W)), ctl, pv, red, redi, _mview(d, k, nb), _mview(pivot_kind, k, nb), _mview(piv, k, nb),
                         psign, perm, _mview(aux, k, nb), prm, Val(WG), _lt_pk(Val(W)), Val(HERM))
        end
        @synchronize
        if ctl[_LT_PHASE] == 2
            _lt_pass2!(_lt_fa(L11, factor, Val(W)), ctl, red, redi, li, Val(WG), prm, _lt_pk(Val(W)))
            @synchronize
            if WG > _LT_S1
                _lt_stage1!(red, redi, ctl, li, Val(WG), Val(2))
                @synchronize
            end
            if li == 1
                k = _bm_gmember(bm, G)
                nb = bm.nbatch
                _lt_decide2!(_lt_fa(L11, factor, Val(W)), ctl, pv, red, redi, _mview(d, k, nb), _mview(pivot_kind, k, nb),
                             _mview(piv, k, nb), psign, perm, _mview(aux, k, nb), prm, Val(WG), _lt_pk(Val(W)), Val(HERM))
            end
            @synchronize
        end
        if ctl[_LT_PHASE] == 3
            _lt_pass3!(_lt_fa(L11, factor, Val(W)), ctl, red, redi, li, Val(WG), _mview(aux, _bm_gmember(bm, G), bm.nbatch), prm,
                       _lt_pk(Val(W)))
            @synchronize
            if WG > _LT_S1
                _lt_stage1!(red, redi, ctl, li, Val(WG), Val(3))
                @synchronize
            end
            if li == 1
                k = _bm_gmember(bm, G)
                nb = bm.nbatch
                _lt_decide3!(_lt_fa(L11, factor, Val(W)), ctl, pv, red, redi, _mview(d, k, nb), _mview(pivot_kind, k, nb),
                             _mview(piv, k, nb), psign, perm, _mview(aux, k, nb), prm, Val(WG), _lt_pk(Val(W)), Val(HERM))
            end
            @synchronize
        end
        _lt_swap!(_lt_fa(L11, factor, Val(W)), ctl, pv, _mview(piv, _bm_gmember(bm, G), bm.nbatch), li, Val(WG), _lt_pk(Val(W)), Val(HERM))
        @synchronize
        _lt_update!(_lt_fa(L11, factor, Val(W)), ctl, pv, red, redi, li, Val(WG), Val(true), prm, _lt_pk(Val(W)),
                    Val(HERM))
        @synchronize
    end
    @inbounds s = nodes[bm.first + _bm_node(bm, G) - 1]
    k = _bm_gmember(bm, G)
    nb = bm.nbatch
    _lt_finalize!(_lt_fa(L11, factor, Val(W)), ctl, _mview(d, k, nb), _mview(pivot_kind, k, nb), li, Val(WG),
                  _lt_pk(Val(W)), Val(HERM))
    @synchronize
    _lt_stage!(L11, factor, ctl, li, Val(W), Val(WG), false)
    @synchronize
    for tt in 1:ctl[_LT_NTILE]
        for kc in 1:_lt_cb_nchunks(ctl)
            k = _bm_gmember(bm, G)
            _lt_cb_load!(L11, factor, ctl, _mview(d, k, bm.nbatch), _mview(pivot_kind, k, bm.nbatch), tt, kc, li,
                         Val(WG), Val(HERM))
            @synchronize
            _lt_cb_acc!(L11, ctl, kc, li, Val(WG), Val(HERM))
            @synchronize
        end
        @inbounds s = nodes[bm.first + _bm_node(bm, G) - 1]
        @inbounds coff = Int(cb_ptr[s]) - 1
        _lt_cb_store!(L11, ctl, _mview(stack, _bm_gmember(bm, G), bm.nbatch), coff, tt, li, Val(WG), Val(HERM))
    end
    @inbounds s = nodes[bm.first + _bm_node(bm, G) - 1]
    @inbounds coff = ctl[_LT_NTILE] == 0 ? Int(cb_ptr[s]) - 1 : -1
    k = _bm_gmember(bm, G)
    nb = bm.nbatch
    _lt_cb_update!(factor, ctl, _mview(stack, k, nb), coff, _mview(d, k, nb), _mview(pivot_kind, k, nb), li, Val(WG),
                   Val(false), Val(HERM))
    _lt_front_finish!(factor, _mview(stats, k, nb), _iview(info, k, nb), ctl, li, Val(WG))
end

# ---------------------------------------------------------------------------
# regime A: one workgroup per subtree, fronts in local memory

# offset (0-based) of the trailing packed m×m triangle of the local packed front
@inline _lt_local_cb_offset(ctl) =
    @inbounds Int(ctl[_ST_LF]) - 1 + Int(ctl[_ST_W]) * (2 * Int(ctl[_ST_F]) - Int(ctl[_ST_W]) + 1) ÷ 2

# write the unit-lower panel, statistics and status; the subtree root writes its contribution block
@inline function _lt_subtree_write!(factor, stack, info, stats, buf, ctl, front_ptr, cb_ptr, li, ::Val{WG}) where {WG}
    @inbounds begin
        v = ctl[_ST_NODE]
        f = Int(ctl[_ST_F])
        w = Int(ctl[_ST_W])
        lf = Int(ctl[_ST_LF]) - 1
        p0 = Int(front_ptr[v])
        T = eltype(factor)
        for q in (li - 1):WG:(f * w - 1)
            j = q ÷ f + 1
            i = q - (j - 1) * f + 1
            factor[p0 + q] = i > j ? buf[lf + _packed(i, j, f)] : i == j ? one(T) : zero(T)
        end
        li == 1 && _lt_stats!(stats, info, ctl, v)
        c0 = cb_ptr[v]
        if c0 > 0
            m = f - w
            src = _lt_local_cb_offset(ctl) + 1
            for q in (li - 1):WG:(m * (m + 1) ÷ 2 - 1)
                stack[c0 + q] = buf[src + q]
            end
        end
    end
    return nothing
end

"""
    subtree_ldlt_kernel!(backend, WG)(factor, stack, info, stats, d, piv, pivot_kind, psign, perm, aux, nzval, amap,
                                      amap_ptr, amap_src, trees, bm, subtree_ptr, subtree_nodes, super_ptr,
                                      front_ptr, front_nrows, front_ncols, cb_ptr, local_front, local_cb, child_ptr,
                                      child_list, relind_ptr, relind, prm, Val(NE), Val(WG);
                                      ndrange = WG * count * bm.nact)

LDLᵀ/LDLᴴ kernel of regime A: workgroup `G` takes subtree
`t = trees[bm.first + g - 1]` of batch member `k` and processes its supernodes in order with the
serial stack of packed fronts and contribution blocks in a `@localmem` buffer
of `NE` entries (as [`subtree_cholesky_kernel!`](@ref)): zero, scatter A,
extend-add the children, factor the front's first `w` columns with the pivoted
LDLᵀ/LDLᴴ of [`front_ldlt_kernel!`](@ref), update the trailing contribution
block, write the unit-lower panel, D, `piv`, the pivot kinds and the
statistics, then move the block down or (subtree root) write it to the update
stack.
"""
@kernel function subtree_ldlt_kernel!(factor, stack, info, stats, d, piv, pivot_kind, psign, perm, aux, nzval, amap,
                                      amap_ptr, amap_src, trees, bm, subtree_ptr, subtree_nodes, super_ptr,
                                      front_ptr, front_nrows, front_ncols, cb_ptr, local_front, local_cb, child_ptr,
                                      child_list, relind_ptr, relind, prm::_LDLTDevice{R, HERM}, ::Val{NE},
                                      ::Val{WG}) where {R, HERM, NE, WG}
    @uniform TT = eltype(factor)
    @uniform IT = eltype(subtree_nodes)
    li = @index(Local, Linear)
    G = @index(Group, Linear)
    buf = @localmem TT (NE,)
    ctl = @localmem IT (_LT_CTL,)
    pv = @localmem TT (_LT_NPV,)
    red = @localmem R (3 * _LT_S1,)
    redi = @localmem IT (2 * _LT_S1,)
    if li == 1
        @inbounds t = trees[bm.first + _bm_node(bm, G) - 1]
        @inbounds ctl[_ST_FIRST] = subtree_ptr[t]
        @inbounds ctl[_ST_COUNT] = subtree_ptr[t + 1] - subtree_ptr[t]
    end
    @synchronize
    for k in 1:ctl[_ST_COUNT]
        if li == 1
            _subtree_setup!(ctl, k, subtree_nodes, front_nrows, front_ncols, local_front, local_cb, child_ptr)
            @inbounds _lt_reset!(ctl, super_ptr[ctl[_ST_NODE]])
        end
        @synchronize
        _subtree_zero!(buf, ctl, li, Val(WG))
        _lt_init_piv!(_mview(piv, _bm_gmember(bm, G), bm.nbatch), ctl, li, Val(WG))
        @synchronize
        _subtree_scatter!(buf, ctl, _mview(nzval, _bm_gmember(bm, G), bm.nbatch), amap, amap_ptr, amap_src, front_ptr,
                          li, Val(WG))
        @synchronize
        for kc in 1:ctl[_ST_NCHILD]
            _subtree_extend_add!(buf, ctl, kc, child_ptr, child_list, front_nrows, front_ncols, local_cb, relind_ptr,
                                 relind, li, Val(WG))
            @synchronize
        end
        _lt_pass1!(buf, ctl, red, redi, li, Val(_LT_S1), prm, Val(true))
        @synchronize
        for it in 1:ctl[_ST_W]
            if li == 1
                mb = _bm_gmember(bm, G)
                nb = bm.nbatch
                _lt_decide1!(buf, ctl, pv, red, redi, _mview(d, mb, nb), _mview(pivot_kind, mb, nb),
                             _mview(piv, mb, nb), psign, perm, _mview(aux, mb, nb), prm, Val(_LT_S1), Val(true),
                             Val(HERM))
            end
            @synchronize
            if ctl[_LT_PHASE] == 2
                _lt_pass2!(buf, ctl, red, redi, li, Val(_LT_S1), prm, Val(true))
                @synchronize
                if li == 1
                    mb = _bm_gmember(bm, G)
                    nb = bm.nbatch
                    _lt_decide2!(buf, ctl, pv, red, redi, _mview(d, mb, nb), _mview(pivot_kind, mb, nb),
                                 _mview(piv, mb, nb), psign, perm, _mview(aux, mb, nb), prm, Val(_LT_S1), Val(true),
                                 Val(HERM))
                end
                @synchronize
            end
            if ctl[_LT_PHASE] == 3
                _lt_pass3!(buf, ctl, red, redi, li, Val(_LT_S1), _mview(aux, _bm_gmember(bm, G), bm.nbatch), prm,
                           Val(true))
                @synchronize
                if li == 1
                    mb = _bm_gmember(bm, G)
                    nb = bm.nbatch
                    _lt_decide3!(buf, ctl, pv, red, redi, _mview(d, mb, nb), _mview(pivot_kind, mb, nb),
                                 _mview(piv, mb, nb), psign, perm, _mview(aux, mb, nb), prm, Val(_LT_S1), Val(true),
                                 Val(HERM))
                end
                @synchronize
            end
            _lt_swap!(buf, ctl, pv, _mview(piv, _bm_gmember(bm, G), bm.nbatch), li, Val(WG), Val(true), Val(HERM))
            @synchronize
            _lt_update!(buf, ctl, pv, red, redi, li, Val(WG), Val(false), prm, Val(true), Val(HERM))
            @synchronize
            _lt_pass1!(buf, ctl, red, redi, li, Val(_LT_S1), prm, Val(true))
            @synchronize
        end
        mb = _bm_gmember(bm, G)
        _lt_finalize!(buf, ctl, _mview(d, mb, bm.nbatch), _mview(pivot_kind, mb, bm.nbatch), li, Val(WG), Val(true),
                      Val(HERM))
        @synchronize
        mb = _bm_gmember(bm, G)
        _lt_cb_update!(buf, ctl, buf, _lt_local_cb_offset(ctl), _mview(d, mb, bm.nbatch),
                       _mview(pivot_kind, mb, bm.nbatch), li, Val(WG), Val(true), Val(HERM))
        @synchronize
        mb = _bm_gmember(bm, G)
        nb = bm.nbatch
        _lt_subtree_write!(factor, _mview(stack, mb, nb), _iview(info, mb, nb), _mview(stats, mb, nb), buf, ctl,
                           member_panels(front_ptr, mb, nb), cb_ptr, li, Val(WG))
        @synchronize
        for r in 1:ctl[_ST_ROUNDS]
            _subtree_move!(buf, ctl, r, li, Val(WG))
            @synchronize
        end
    end
end

# ---------------------------------------------------------------------------
# reductions

@kernel function _abs_max_kernel!(aux, nzval, nnz, bm, ::Val{WG}, ::Val{LOG2WG}) where {WG, LOG2WG}
    @uniform RT = real(eltype(aux))
    li = @index(Local, Linear)
    G = @index(Group, Linear)
    best = @localmem RT (WG,)
    @inbounds begin
        nz = _mview(nzval, _bm_gmember(bm, G), bm.nbatch)
        m = zero(RT)
        for p in li:WG:nnz
            m = max(m, RT(abs(nz[p])))
        end
        best[li] = m
    end
    @synchronize
    for lev in 1:LOG2WG
        @inbounds begin
            h = WG >> lev
            if li <= h
                best[li] = max(best[li], best[li + h])
            end
        end
        @synchronize
    end
    if li == 1
        @inbounds aux[_bm_gmember(bm, G)] = eltype(aux)(best[1])
    end
end

"""
    abs_max!(aux, nzval[, bm]) -> aux

`aux[k] = max |nzval of member k|` for the active members `k` of the
[`BatchMap`](@ref) `bm` (default: a single matrix, `aux[1]`), one workgroup per
member, `@localmem` tree reduction; the scale of `pivot_epsilon_alg = "algo1"`.
Asynchronous.
"""
function abs_max!(aux::AbstractVector, nzval::AbstractVector, bm::BatchMap = single_batch())
    WG = STATS_WORKGROUP
    _abs_max_kernel!(KernelAbstractions.get_backend(aux), WG)(aux, nzval, length(nzval) ÷ bm.nbatch, bm, Val(WG),
                                                             Val(_ilog2(WG)); ndrange = WG * bm.nact)
    return aux
end

@kernel function _reduce_stats_kernel!(totals, stats, ns, bm, ::Val{WG}, ::Val{NF},
                                       ::Val{LOG2WG}) where {WG, NF, LOG2WG}
    li = @index(Local, Linear)
    G = @index(Group, Linear)
    acc = @localmem Int64 (WG * NF,)
    @inbounds begin
        st = _mview(stats, _bm_gmember(bm, G), bm.nbatch)
        for q in 1:NF
            a = Int64(0)
            for s in li:WG:ns
                x = st[(s - 1) * NF + q]
                a += q == NF ? Int64(x != 0) : x
            end
            acc[(q - 1) * WG + li] = a
        end
    end
    @synchronize
    for lev in 1:LOG2WG
        @inbounds begin
            h = WG >> lev
            if li <= h
                for q in 1:NF
                    acc[(q - 1) * WG + li] += acc[(q - 1) * WG + li + h]
                end
            end
        end
        @synchronize
    end
    if li <= NF
        @inbounds _mview(totals, _bm_gmember(bm, G), bm.nbatch)[li] = acc[(li - 1) * WG + 1]
    end
end

"""
    reduce_stats!(numeric, symbolic) -> numeric

Sum the per-front statistics `numeric.stats` into `numeric.totals`
(`npos, nneg, nzero, nperturbed, n2x2`, and the number of fronts with a failed
pivot) of every active batch member: one workgroup per member, `@localmem`
tree reduction, no atomics. Asynchronous; [`pivot_totals`](@ref) reads the
result.
"""
function reduce_stats!(N::Numeric, S::Symbolic)
    WG = STATS_WORKGROUP
    bm = batch_map(N)
    _reduce_stats_kernel!(KernelAbstractions.get_backend(N.stats), WG)(N.totals, N.stats, nsupernodes(S), bm,
                                                                       Val(WG), Val(FRONT_STATS_FIELDS),
                                                                       Val(_ilog2(WG)); ndrange = WG * bm.nact)
    return N
end

"""
    pivot_totals(numeric, k = 1) -> (npos, nneg, nzero, nperturbed, n2x2)

The statistics of batch member `k` reduced on the device by
[`reduce_stats!`](@ref) (one copy of `numeric.totals` to the host: a
synchronization, done at the phase boundary by `getparam`).
"""
pivot_totals(N::Numeric, k::Integer = 1) = pivot_totals(Array(N.totals), N.nbatch, k)

function pivot_totals(totals::Vector{Int64}, nb::Integer, k::Integer)
    1 <= k <= nb || throw(InvalidValueError("batch member $k outside 1:$nb"))
    t = view(totals, ((k - 1) * FRONT_STATS_FIELDS + 1):(k * FRONT_STATS_FIELDS))
    return (npos = t[STAT_NPOS], nneg = t[STAT_NNEG], nzero = t[STAT_NZERO], nperturbed = t[STAT_NPERTURBED],
            n2x2 = t[STAT_N2X2])
end

# ---------------------------------------------------------------------------
# driver

function _launch_front_ldlt!(N::Numeric, S::Symbolic, nzval, first, count, maxchild, prm, ::Val{W},
                             ::Val{WG}) where {W, WG}
    bm = batch_map(N; first)
    kernel! = front_ldlt_kernel!(KernelAbstractions.get_backend(N.factor), WG)
    kernel!(N.factor, N.stack, N.info, N.stats, N.d, N.piv, N.pivot_kind, N.psign, S.perm, N.aux, nzval, S.amap,
            S.amap_ptr, S.amap_src, S.group_nodes, bm, S.super_ptr, S.front_ptr, S.front_nrows, S.front_ncols,
            S.cb_ptr, S.child_ptr, S.child_list, S.relind_ptr, S.relind, Int(maxchild), prm, Val(W), Val(WG);
            ndrange = WG * count * bm.nact)
    return nothing
end

function _launch_subtrees_ldlt!(N::Numeric{T}, S::Symbolic, nzval, first, count, prm, ::Val{LB},
                                ::Val{WG}) where {T, LB, WG}
    bm = batch_map(N; first)
    kernel! = subtree_ldlt_kernel!(KernelAbstractions.get_backend(N.factor), WG)
    kernel!(N.factor, N.stack, N.info, N.stats, N.d, N.piv, N.pivot_kind, N.psign, S.perm, N.aux, nzval, S.amap,
            S.amap_ptr, S.amap_src, S.group_nodes, bm, S.subtree_ptr, S.subtree_nodes, S.super_ptr,
            S.front_ptr, S.front_nrows, S.front_ncols, S.cb_ptr, S.local_front, S.local_cb, S.child_ptr, S.child_list,
            S.relind_ptr, S.relind, prm, Val((LB - SUBTREE_LOCAL_RESERVE_LDLT) ÷ sizeof(T)), Val(WG);
            ndrange = WG * count * bm.nact)
    return nothing
end

function _factorize_ldlt_groups!(N::Numeric, S::Symbolic, nzval::AbstractVector, prm, flag, gimpl::Symbol,
                                 nbv::Val, herm::Val)
    plan = N.plan
    for k in eachindex(plan.sub_first)
        _poll_interrupt(flag)
        a, b = plan.sub_first[k], plan.sub_last[k]
        _with_local_bytes(plan.sub_local[k]) do lb
            _launch_subtrees_ldlt!(N, S, nzval, a, b - a + 1, prm, lb, Val(SUBTREE_WORKGROUP))
        end
    end
    for k in eachindex(plan.group_first)
        _poll_interrupt(flag)
        a, b = plan.group_first[k], plan.group_last[k]
        W = plan.group_width[k]
        if W == 0                                   # regime C: blocked pivot steps and GEMMs (src/numeric/ldlt_c.jl)
            _factorize_ldlt_c_group!(N, S, nzval, a, b, plan.group_maxchild[k], prm, gimpl, nbv, herm)
        elseif W <= _LT_GLOBAL_MAX_W                # narrow regime-B bins: the panel in global memory
            _launch_front_ldlt!(N, S, nzval, a, b - a + 1, plan.group_maxchild[k], prm, Val(0), Val(LDLT_WORKGROUP))
        else                                        # regime B: F₁₁ in local memory
            _with_width_class(W) do w
                _launch_front_ldlt!(N, S, nzval, a, b - a + 1, plan.group_maxchild[k], prm, w, Val(LDLT_WORKGROUP))
            end
        end
    end
    return nothing
end

function _factorize_ldlt_herm!(N::Numeric{T}, S::Symbolic, nzval, opts, prm, gimpl, nbv::Val, ::Val{H}) where {T, H}
    if H
        _factorize_ldlt_groups!(N, S, nzval, prm, opts.user_host_interrupt, gimpl, nbv, Val(true))
    else
        _factorize_ldlt_groups!(N, S, nzval, _ldlt_device_params(S, T, opts, Val(false)), opts.user_host_interrupt,
                                gimpl, nbv, Val(false))
    end
    return nothing
end

"""
    factorize_ldlt!(numeric, symbolic, nzval; opts = Options()) -> 0

Multifrontal `P A Pᵀ = L D Lᴴ` (structure `"H"`, or `"S"` with real `T`) or
`L D Lᵀ` (complex symmetric `"S"`) on the device, with the in-front pivoting
and the static perturbation of the reference [`ref_ldlt!`](@ref) (same pivot
choice, same storage): `opts.pivot_type`, `pivot_threshold`, `pivot_epsilon`,
`pivot_epsilon_alg` and `pivot_sign` (copied to `numeric.psign`; a length
other than `n` raises [`InvalidValueError`](@ref)). Regime-A groups are one
[`subtree_ldlt_kernel!`](@ref) launch per budget class; every regime-B and
regime-C launch group is one [`front_ldlt_kernel!`](@ref) launch (fused
assembly and factorization, one workgroup per front, `F₁₁` in local memory for
regime-B width classes; no vendor calls, see the T15 report); then [`reduce_stats!`](@ref). `opts.user_host_interrupt` is polled
before every launch group ([`InterruptedError`](@ref)). With `pivot_epsilon_alg = "algo1"`
[`abs_max!`](@ref) computes the scale first. Fills `numeric.factor`
(unit-lower panels), `d`, `piv`, `pivot_kind`, `stats` and `totals`. The
factorization always completes (`info = 0`); the phase allocates nothing on
the device, never synchronizes with the host, and is deterministic.
"""
function factorize_ldlt!(N::Numeric{T}, S::Symbolic, nzval::AbstractVector; impl::Symbol = :auto,
                         opts::Options = Options(), nb::Integer = LDLT_C_NB) where {T}
    _check_numeric(N, S, nzval)
    1 <= nb <= LDLT_C_NB || throw(InvalidValueError("nb = $nb: the regime-C block size must be in 1:$LDLT_C_NB"))
    gimpl = select_impl(:gemm, N.factor, impl === :auto && !S.schedule.vendor_c ? :ka : impl)
    herm = _ldlt_herm(S, T)
    prm = _ldlt_device_params(S, T, opts, Val(true))           # validates the options
    ps = opts.pivot_sign
    if ps === nothing
        fill!(N.psign, Int8(0))
    else
        length(ps) == S.n ||
            throw(InvalidValueError("pivot_sign has $(length(ps)) entries, the matrix has $(S.n) rows"))
        copyto!(N.psign, ps)
    end
    prm.scaled && abs_max!(N.aux, nzval, batch_map(N))
    if nb == LDLT_C_NB
        if herm
            _factorize_ldlt_herm!(N, S, nzval, opts, prm, gimpl, Val(LDLT_C_NB), Val(true))
        else
            _factorize_ldlt_herm!(N, S, nzval, opts, prm, gimpl, Val(LDLT_C_NB), Val(false))
        end
    else                                                       # other block sizes: tests only (dynamic dispatch)
        _factorize_ldlt_herm!(N, S, nzval, opts, prm, gimpl, Val(Int(nb)), Val(herm))
    end
    reduce_stats!(N, S)
    return 0
end
