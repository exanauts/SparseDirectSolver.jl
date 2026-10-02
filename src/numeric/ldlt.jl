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
#   pivoted factorization of the panel in global memory and the update of the
#   front's packed contribution block on the update stack.
#
# Per pivot step (a loop over the `w` columns of the front with a uniform trip
# count; a 2×2 pivot leaves the last iterations idle): work item 1 chooses the
# pivot (serial search, the reference's `_choose_pivot`), the workgroup swaps
# rows/columns, work item 1 stores D (perturbing a tiny 1×1 pivot) and counts
# the statistics, the workgroup applies the rank-1/rank-2 update to the
# remaining fully-summed columns and then scales the pivot columns into L.
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
const _LT_CTL = _ST_CTL + 10

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

# the front of the current node from the control words: offset `_ST_LF` (1-based) into `fa`
@inline _lt_front(fa, ctl, ::Val{true}) = @inbounds _PackedFront(fa, Int(ctl[_ST_LF]) - 1, Int(ctl[_ST_F]))
@inline _lt_front(fa, ctl, ::Val{false}) = @inbounds _PanelFront(fa, Int(ctl[_ST_LF]) - 1, Int(ctl[_ST_F]))

# ---------------------------------------------------------------------------
# pivot choice (work item 1; the reference's `_colmax`, `_accept_1x1`, `_accept_2x2`,
# `_best_1x1`, `_choose_pivot` on the lower triangle)

@inline function _lt_colmax(F, c, k, f, skip)
    m = zero(real(eltype(F.a)))
    for i in k:f
        (i == c || i == skip) && continue
        m = max(m, _fabs(F, i, c))
    end
    return m
end

@inline function _lt_accept_1x1(F, c, k, f, u, ε)
    d = _fabs(F, c, c)
    return d >= ε && d >= u * _lt_colmax(F, c, k, f, 0)
end

@inline _lt_det2(F, k, r, h::Val) = _fget(F, k, k) * _fget(F, r, r) - _fsym(F, k, r, h) * _fget(F, r, k)

@inline function _lt_nonsingular_2x2(F, k, r, ε, h::Val)
    det = _lt_det2(F, k, r, h)
    return !iszero(det) && isfinite(det) && max(_fabs(F, k, k), _fabs(F, r, k), _fabs(F, r, r)) >= ε
end

@inline function _lt_accept_2x2(F, k, r, f, u, ε, h::Val)
    _lt_nonsingular_2x2(F, k, r, ε, h) || return false
    det = _lt_det2(F, k, r, h)
    e11 = abs(_fget(F, r, r) / det)
    e12 = abs(_fsym(F, k, r, h) / det)
    e21 = abs(_fget(F, r, k) / det)
    e22 = abs(_fget(F, k, k) / det)
    m1 = _lt_colmax(F, k, k, f, r)
    m2 = _lt_colmax(F, r, k, f, k)
    return u * (e11 * m1 + e21 * m2) <= 1 && u * (e12 * m1 + e22 * m2) <= 1
end

@inline function _lt_best_1x1(F, k, w, f, u, ε)
    best = 0
    bv = zero(real(eltype(F.a)))
    for j in k:w
        _lt_accept_1x1(F, j, k, f, u, ε) || continue
        a = _fabs(F, j, j)
        if best == 0 || a > bv
            best = j
            bv = a
        end
    end
    return best
end

# `(c, 0)`: 1×1 pivot at column c; `(k, r)`: 2×2 pivot on columns k and r (as `_choose_pivot`)
@inline function _lt_choose(F, k, w, f, p::_LDLTDevice{R}, ε, h::Val) where {R}
    p.ptype == _LT_PIVOT_NONE && return (k, 0)
    u = p.u
    if p.ptype == _LT_PIVOT_DIAGONAL
        _lt_accept_1x1(F, k, k, f, u, ε) && return (k, 0)
        best = _lt_best_1x1(F, k, w, f, u, ε)
        best != 0 && return (best, 0)
        big = k
        for j in (k + 1):w
            _fabs(F, j, j) > _fabs(F, big, big) && (big = j)
        end
        return (_fabs(F, big, big) >= ε ? big : k, 0)
    end
    α = R(BUNCH_KAUFMAN_ALPHA)
    λ = zero(R)
    r = 0
    for i in (k + 1):w
        a = _fabs(F, i, k)
        if a > λ
            λ = a
            r = i
        end
    end
    akk = _fabs(F, k, k)
    c, r2 = k, 0
    if !(r == 0 || akk >= α * λ)
        σ = zero(R)                                  # largest off-diagonal of column r in the block
        for i in k:w
            i != r && (σ = max(σ, _fabs(F, i, r)))
        end
        if akk * σ >= α * λ^2
            # 1×1 at k
        elseif _fabs(F, r, r) >= α * σ
            c = r
        else
            r2 = r
        end
    end
    (r2 == 0 ? _lt_accept_1x1(F, c, k, f, u, ε) : _lt_accept_2x2(F, k, r2, f, u, ε, h)) && return (c, r2)
    best = _lt_best_1x1(F, k, w, f, u, ε)
    best != 0 && return (best, 0)
    r2 != 0 && _lt_nonsingular_2x2(F, k, r2, ε, h) && return (c, r2)
    return (c, 0)
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

# work item 1: advance to the next column and choose its pivot
@inline function _lt_choose!(fa, ctl, aux, p::_LDLTDevice, pk::Val, h::Val)
    @inbounds begin
        IT = eltype(ctl)
        k = Int(ctl[_LT_K]) + Int(ctl[_LT_STEP])
        ctl[_LT_K] = k % IT
        ctl[_LT_STEP] = zero(IT)
        w = Int(ctl[_ST_W])
        if k <= w
            F = _lt_front(fa, ctl, pk)
            c, r = _lt_choose(F, k, w, Int(ctl[_ST_F]), p, _lt_eps(p, aux), h)
            ctl[_LT_C] = c % IT
            ctl[_LT_R] = r % IT
        end
    end
    return nothing
end

# symmetric interchange of the columns/rows p < q of the front (lower triangle), as `_swap_front!`
@inline function _lt_swap!(fa, ctl, piv, li, ::Val{WG}, pk::Val, h::Val) where {WG}
    @inbounds begin
        k = Int(ctl[_LT_K])
        if k <= Int(ctl[_ST_W])
            r = Int(ctl[_LT_R])
            p = r == 0 ? k : k + 1
            q = r == 0 ? Int(ctl[_LT_C]) : r
            if p != q
                F = _lt_front(fa, ctl, pk)
                for i in li:WG:Int(ctl[_ST_F])
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
                if li == 1
                    c0 = Int(ctl[_LT_C0])
                    x = piv[c0 + p - 1]
                    piv[c0 + p - 1] = piv[c0 + q - 1]
                    piv[c0 + q - 1] = x
                end
            end
        end
    end
    return nothing
end

# work item 1: store the pivot block in D (perturbing a tiny 1×1 pivot), the pivot kinds and the
# statistics; `pv` receives d (1×1) or the entries (e11, e12, e21, e22) of the inverse 2×2 block
@inline function _lt_pivot!(fa, ctl, pv, d, pivot_kind, piv, psign, perm, aux, p::_LDLTDevice, pk::Val,
                            h::Val{H}) where {H}
    @inbounds begin
        n = length(perm)
        IT = eltype(ctl)
        T = eltype(fa)
        k = Int(ctl[_LT_K])
        if k <= Int(ctl[_ST_W])
            F = _lt_front(fa, ctl, pk)
            g = Int(ctl[_LT_C0]) + k - 1
            if ctl[_LT_R] == 0
                ε = _lt_eps(p, aux)
                x = _fget(F, k, k)
                dk = H ? T(real(x)) : x
                kind = PIVOT_KIND_1X1
                if !(abs(dk) >= ε)                        # tiny (or NaN): perturb
                    iszero(dk) && (ctl[_LT_STAT + STAT_NZERO] += one(IT))
                    dk = _perturbation_sign(dk, Int(psign[perm[piv[g]]]), H) * ε
                    kind = PIVOT_KIND_PERTURBED
                    ctl[_LT_STAT + STAT_NPERTURBED] += one(IT)
                end
                _fset!(F, k, k, dk)
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
                ctl[_LT_STEP] = one(IT)
            else
                a = _fget(F, k, k)
                b = _fget(F, k + 1, k)
                c2 = _fget(F, k + 1, k + 1)
                if H
                    a = T(real(a))
                    c2 = T(real(c2))
                end
                up = _cj(b, h)
                _fset!(F, k, k, a)
                _fset!(F, k + 1, k + 1, c2)
                _fset!(F, k + 1, k, zero(T))              # the 2×2 block lives in D, L's block is the identity
                det = a * c2 - up * b
                pv[1] = c2 / det
                pv[2] = -up / det
                pv[3] = -b / det
                pv[4] = a / det
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
    end
    return nothing
end

# update of the remaining fully-summed columns (lower part, rows to f) with the unscaled pivot columns:
# F[i, j] -= l_i F[k, j] (1×1) or l1_i F[k, j] + l2_i F[k+1, j] (2×2), as the reference
@inline function _lt_update!(fa, ctl, pv, li, ::Val{WG}, pk::Val, h::Val) where {WG}
    @inbounds begin
        k = Int(ctl[_LT_K])
        w = Int(ctl[_ST_W])
        if k <= w
            F = _lt_front(fa, ctl, pk)
            step = Int(ctl[_LT_STEP])
            kk = k + step - 1
            nr = Int(ctl[_ST_F]) - kk
            nc = w - kk
            for q in (li - 1):WG:(nr * nc - 1)
                i = kk + 1 + q % nr
                j = kk + 1 + q ÷ nr
                i >= j || continue
                if step == 1
                    l = _fget(F, i, k) / pv[1]
                    _fset!(F, i, j, _fget(F, i, j) - l * _cj(_fget(F, j, k), h))
                else
                    x1 = _fget(F, i, k)
                    x2 = _fget(F, i, k + 1)
                    l1 = x1 * pv[1] + x2 * pv[3]
                    l2 = x1 * pv[2] + x2 * pv[4]
                    _fset!(F, i, j, _fget(F, i, j) - (l1 * _cj(_fget(F, j, k), h) + l2 * _cj(_fget(F, j, k + 1), h)))
                end
            end
        end
    end
    return nothing
end

# scale the pivot columns below the pivot block into L
@inline function _lt_scale!(fa, ctl, pv, li, ::Val{WG}, pk::Val) where {WG}
    @inbounds begin
        k = Int(ctl[_LT_K])
        if k <= Int(ctl[_ST_W])
            F = _lt_front(fa, ctl, pk)
            step = Int(ctl[_LT_STEP])
            for i in (k + step + li - 1):WG:Int(ctl[_ST_F])
                if step == 1
                    _fset!(F, i, k, _fget(F, i, k) / pv[1])
                else
                    x1 = _fget(F, i, k)
                    x2 = _fget(F, i, k + 1)
                    _fset!(F, i, k, x1 * pv[1] + x2 * pv[3])
                    _fset!(F, i, k + 1, x1 * pv[2] + x2 * pv[4])
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

@inline function _lt_front_setup!(ctl, s, super_ptr, front_ptr, front_nrows, front_ncols)
    @inbounds begin
        IT = eltype(ctl)
        ctl[_ST_NODE] = s % IT
        ctl[_ST_F] = front_nrows[s] % IT
        ctl[_ST_W] = front_ncols[s] % IT
        ctl[_ST_LF] = front_ptr[s] % IT
        _lt_reset!(ctl, super_ptr[s])
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
                                    amap_ptr, amap_src, nodes, first, super_ptr, front_ptr, front_nrows,
                                    front_ncols, cb_ptr, child_ptr, child_list, relind_ptr, relind, maxchild, prm,
                                    Val(WG); ndrange = WG * count)

LDLᵀ/LDLᴴ kernel of regimes B and C: workgroup `g` takes front
`s = nodes[first + g - 1]`, zeroes and assembles it (A through the `amap`, then
the `maxchild` children's packed contribution blocks in `child_list` order),
factors its `w` fully-summed columns in place in the panel with in-block
Bunch–Kaufman pivoting, threshold acceptance and perturbation (`prm`, a
`_LDLTDevice{R, HERM}`; `HERM`: Hermitian or real symmetric, else complex symmetric),
writes D, the local pivot order `piv`, the pivot kinds and the front's
statistics, updates its packed contribution block on the update stack
(`cb_ptr[s] > 0`) with `F₂₂ − (L₂₁ D) L₂₁ᴴ` and leaves a unit-lower panel.
"""
@kernel function front_ldlt_kernel!(factor, stack, info, stats, d, piv, pivot_kind, psign, perm, aux, nzval, amap,
                                    amap_ptr, amap_src, nodes, first, super_ptr, front_ptr, front_nrows, front_ncols,
                                    cb_ptr, child_ptr, child_list, relind_ptr, relind, maxchild,
                                    prm::_LDLTDevice{R, HERM}, ::Val{WG}) where {R, HERM, WG}
    @uniform TT = eltype(factor)
    @uniform IT = eltype(front_ptr)
    li = @index(Local, Linear)
    g = @index(Group, Linear)
    ctl = @localmem IT (_LT_CTL,)
    pv = @localmem TT (4,)
    if li == 1
        @inbounds s = nodes[first + g - 1]
        _lt_front_setup!(ctl, s, super_ptr, front_ptr, front_nrows, front_ncols)
    end
    @synchronize
    @inbounds s = nodes[first + g - 1]
    _zero_front!(factor, stack, s, li, front_ptr, front_nrows, front_ncols, cb_ptr, Val(WG))
    _lt_init_piv!(piv, ctl, li, Val(WG))
    @synchronize
    @inbounds s = nodes[first + g - 1]
    _scatter_front!(factor, nzval, amap, amap_ptr, amap_src, s, li, Val(WG))
    @synchronize
    for k in 1:maxchild
        @inbounds s = nodes[first + g - 1]
        _extend_add_child!(factor, stack, s, k, li, front_ptr, front_nrows, front_ncols, cb_ptr, child_ptr,
                           child_list, relind_ptr, relind, Val(WG))
        @synchronize
    end
    for it in 1:ctl[_ST_W]
        if li == 1
            _lt_choose!(factor, ctl, aux, prm, Val(false), Val(HERM))
        end
        @synchronize
        _lt_swap!(factor, ctl, piv, li, Val(WG), Val(false), Val(HERM))
        @synchronize
        if li == 1
            _lt_pivot!(factor, ctl, pv, d, pivot_kind, piv, psign, perm, aux, prm, Val(false), Val(HERM))
        end
        @synchronize
        _lt_update!(factor, ctl, pv, li, Val(WG), Val(false), Val(HERM))
        @synchronize
        _lt_scale!(factor, ctl, pv, li, Val(WG), Val(false))
        @synchronize
    end
    @inbounds s = nodes[first + g - 1]
    @inbounds coff = Int(cb_ptr[s]) - 1
    _lt_cb_update!(factor, ctl, stack, coff, d, pivot_kind, li, Val(WG), Val(false), Val(HERM))
    _lt_front_finish!(factor, stats, info, ctl, li, Val(WG))
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
                                      amap_ptr, amap_src, trees, first, subtree_ptr, subtree_nodes, super_ptr,
                                      front_ptr, front_nrows, front_ncols, cb_ptr, local_front, local_cb, child_ptr,
                                      child_list, relind_ptr, relind, prm, Val(NE), Val(WG); ndrange = WG * count)

LDLᵀ/LDLᴴ kernel of regime A: workgroup `g` takes subtree
`t = trees[first + g - 1]` and processes its supernodes in order with the
serial stack of packed fronts and contribution blocks in a `@localmem` buffer
of `NE` entries (as [`subtree_cholesky_kernel!`](@ref)): zero, scatter A,
extend-add the children, factor the front's first `w` columns with the pivoted
LDLᵀ/LDLᴴ of [`front_ldlt_kernel!`](@ref), update the trailing contribution
block, write the unit-lower panel, D, `piv`, the pivot kinds and the
statistics, then move the block down or (subtree root) write it to the update
stack.
"""
@kernel function subtree_ldlt_kernel!(factor, stack, info, stats, d, piv, pivot_kind, psign, perm, aux, nzval, amap,
                                      amap_ptr, amap_src, trees, first, subtree_ptr, subtree_nodes, super_ptr,
                                      front_ptr, front_nrows, front_ncols, cb_ptr, local_front, local_cb, child_ptr,
                                      child_list, relind_ptr, relind, prm::_LDLTDevice{R, HERM}, ::Val{NE},
                                      ::Val{WG}) where {R, HERM, NE, WG}
    @uniform TT = eltype(factor)
    @uniform IT = eltype(subtree_nodes)
    li = @index(Local, Linear)
    g = @index(Group, Linear)
    buf = @localmem TT (NE,)
    ctl = @localmem IT (_LT_CTL,)
    pv = @localmem TT (4,)
    if li == 1
        @inbounds t = trees[first + g - 1]
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
        _lt_init_piv!(piv, ctl, li, Val(WG))
        @synchronize
        _subtree_scatter!(buf, ctl, nzval, amap, amap_ptr, amap_src, front_ptr, li, Val(WG))
        @synchronize
        for kc in 1:ctl[_ST_NCHILD]
            _subtree_extend_add!(buf, ctl, kc, child_ptr, child_list, front_nrows, front_ncols, local_cb, relind_ptr,
                                 relind, li, Val(WG))
            @synchronize
        end
        for it in 1:ctl[_ST_W]
            if li == 1
                _lt_choose!(buf, ctl, aux, prm, Val(true), Val(HERM))
            end
            @synchronize
            _lt_swap!(buf, ctl, piv, li, Val(WG), Val(true), Val(HERM))
            @synchronize
            if li == 1
                _lt_pivot!(buf, ctl, pv, d, pivot_kind, piv, psign, perm, aux, prm, Val(true), Val(HERM))
            end
            @synchronize
            _lt_update!(buf, ctl, pv, li, Val(WG), Val(true), Val(HERM))
            @synchronize
            _lt_scale!(buf, ctl, pv, li, Val(WG), Val(true))
            @synchronize
        end
        _lt_cb_update!(buf, ctl, buf, _lt_local_cb_offset(ctl), d, pivot_kind, li, Val(WG), Val(true), Val(HERM))
        @synchronize
        _lt_subtree_write!(factor, stack, info, stats, buf, ctl, front_ptr, cb_ptr, li, Val(WG))
        @synchronize
        for r in 1:ctl[_ST_ROUNDS]
            _subtree_move!(buf, ctl, r, li, Val(WG))
            @synchronize
        end
    end
end

# ---------------------------------------------------------------------------
# reductions

@kernel function _abs_max_kernel!(aux, nzval, nnz, ::Val{WG}, ::Val{LOG2WG}) where {WG, LOG2WG}
    @uniform RT = real(eltype(aux))
    li = @index(Local, Linear)
    best = @localmem RT (WG,)
    @inbounds begin
        m = zero(RT)
        for p in li:WG:nnz
            m = max(m, RT(abs(nzval[p])))
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
        @inbounds aux[1] = eltype(aux)(best[1])
    end
end

"""
    abs_max!(aux, nzval) -> aux

`aux[1] = max |nzval[p]|` (one workgroup, `@localmem` tree reduction; the
scale of `pivot_epsilon_alg = "algo1"`). Asynchronous.
"""
function abs_max!(aux::AbstractVector, nzval::AbstractVector)
    WG = STATS_WORKGROUP
    _abs_max_kernel!(KernelAbstractions.get_backend(aux), WG)(aux, nzval, length(nzval), Val(WG), Val(_ilog2(WG));
                                                             ndrange = WG)
    return aux
end

@kernel function _reduce_stats_kernel!(totals, stats, ns, ::Val{WG}, ::Val{NF}, ::Val{LOG2WG}) where {WG, NF, LOG2WG}
    li = @index(Local, Linear)
    acc = @localmem Int64 (WG * NF,)
    @inbounds for q in 1:NF
        a = Int64(0)
        for s in li:WG:ns
            x = stats[(s - 1) * NF + q]
            a += q == NF ? Int64(x != 0) : x
        end
        acc[(q - 1) * WG + li] = a
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
        @inbounds totals[li] = acc[(li - 1) * WG + 1]
    end
end

"""
    reduce_stats!(numeric, symbolic) -> numeric

Sum the per-front statistics `numeric.stats` into `numeric.totals`
(`npos, nneg, nzero, nperturbed, n2x2`, and the number of fronts with a failed
pivot): one workgroup, `@localmem` tree reduction, no atomics. Asynchronous;
[`pivot_totals`](@ref) reads the result.
"""
function reduce_stats!(N::Numeric, S::Symbolic)
    WG = STATS_WORKGROUP
    _reduce_stats_kernel!(KernelAbstractions.get_backend(N.stats), WG)(N.totals, N.stats, nsupernodes(S), Val(WG),
                                                                       Val(FRONT_STATS_FIELDS), Val(_ilog2(WG));
                                                                       ndrange = WG)
    return N
end

"""
    pivot_totals(numeric) -> (npos, nneg, nzero, nperturbed, n2x2)

The statistics reduced on the device by [`reduce_stats!`](@ref) (one copy of
`numeric.totals` to the host: a synchronization, done at the phase boundary
by `getparam`).
"""
function pivot_totals(N::Numeric)
    t = Array(N.totals)
    return (npos = t[STAT_NPOS], nneg = t[STAT_NNEG], nzero = t[STAT_NZERO], nperturbed = t[STAT_NPERTURBED],
            n2x2 = t[STAT_N2X2])
end

# ---------------------------------------------------------------------------
# driver

function _launch_front_ldlt!(N::Numeric, S::Symbolic, nzval, first, count, maxchild, prm, ::Val{WG}) where {WG}
    kernel! = front_ldlt_kernel!(KernelAbstractions.get_backend(N.factor), WG)
    kernel!(N.factor, N.stack, N.info, N.stats, N.d, N.piv, N.pivot_kind, N.psign, S.perm, N.aux, nzval, S.amap,
            S.amap_ptr, S.amap_src, S.group_nodes, Int(first), S.super_ptr, S.front_ptr, S.front_nrows, S.front_ncols,
            S.cb_ptr, S.child_ptr, S.child_list, S.relind_ptr, S.relind, Int(maxchild), prm, Val(WG);
            ndrange = WG * count)
    return nothing
end

function _launch_subtrees_ldlt!(N::Numeric{T}, S::Symbolic, nzval, first, count, prm, ::Val{LB},
                                ::Val{WG}) where {T, LB, WG}
    kernel! = subtree_ldlt_kernel!(KernelAbstractions.get_backend(N.factor), WG)
    kernel!(N.factor, N.stack, N.info, N.stats, N.d, N.piv, N.pivot_kind, N.psign, S.perm, N.aux, nzval, S.amap,
            S.amap_ptr, S.amap_src, S.group_nodes, Int(first), S.subtree_ptr, S.subtree_nodes, S.super_ptr,
            S.front_ptr, S.front_nrows, S.front_ncols, S.cb_ptr, S.local_front, S.local_cb, S.child_ptr, S.child_list,
            S.relind_ptr, S.relind, prm, Val((LB - SUBTREE_LOCAL_RESERVE) ÷ sizeof(T)), Val(WG);
            ndrange = WG * count)
    return nothing
end

function _factorize_ldlt_groups!(N::Numeric, S::Symbolic, nzval::AbstractVector, prm, flag)
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
        _launch_front_ldlt!(N, S, nzval, a, b - a + 1, plan.group_maxchild[k], prm, Val(LDLT_WORKGROUP))
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
assembly and factorization, one workgroup per front; no vendor calls, see the
T15 report); then [`reduce_stats!`](@ref). `opts.user_host_interrupt` is polled
before every launch group ([`InterruptedError`](@ref)). With `pivot_epsilon_alg = "algo1"`
[`abs_max!`](@ref) computes the scale first. Fills `numeric.factor`
(unit-lower panels), `d`, `piv`, `pivot_kind`, `stats` and `totals`. The
factorization always completes (`info = 0`); the phase allocates nothing on
the device, never synchronizes with the host, and is deterministic.
"""
function factorize_ldlt!(N::Numeric{T}, S::Symbolic, nzval::AbstractVector; opts::Options = Options()) where {T}
    _check_numeric(N, S, nzval)
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
    prm.scaled && abs_max!(N.aux, nzval)
    if herm
        _factorize_ldlt_groups!(N, S, nzval, prm, opts.user_host_interrupt)
    else
        _factorize_ldlt_groups!(N, S, nzval, _ldlt_device_params(S, T, opts, Val(false)), opts.user_host_interrupt)
    end
    reduce_stats!(N, S)
    return 0
end
