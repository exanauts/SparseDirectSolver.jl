# Regime C of the device LDLᵀ/LDLᴴ (PLAN §2.4, §3.3; experiment 2 of PERFORMANCE.md, issue #75): the
# in-front pivoting of the reference on the fully-summed columns, blocked so that most of the flops go to
# vendor (`:vendor`) or KA (`:ka`) GEMMs through `src/dense/interface.jl`.
#
# The `w` pivot steps of a front run in static blocks of `nb` columns (block b starts at column
# `k0 = (b - 1) nb + 1`). Per block, one `panel_ldlt_kernel!` launch (one workgroup) runs the pivot steps
# of `front_ldlt_kernel!` (the cooperative search, interchanges, D, statistics) on a lazily updated panel:
# the pivots of the block are kept as columns of `Lb` (multipliers) and `Wb` (unscaled pivot columns,
# `W = L D`) in the workspace, and an entry of a column that is not a pivot column yet is read as its
# stored value minus the block's pending pivots, `F[i, j] − Σₜ Lb[i, t] Wb[j, t]ᴴ` (2×2 pivots as one
# term, in pivot order: the arithmetic of the reference's right-looking steps). A pivot column is
# materialized once, when it is taken, and stored unscaled. After the block, one GEMM applies its pivots
# to the trailing fully-summed columns (`F[k0':f, k0':w] −= Lb Wbᴴ`, `k0' = k0 + nb`) and one accumulates
# them into the contribution block (`C −= Lb₂₁ Wb₂₁ᴴ`, `m×m` workspace). `Lb`/`Wb` have `nb + 1` slots: a
# 2×2 pivot that starts at the block's last column takes the first column of the next block too, which
# is then saved before the GEMM (that updates it as a trailing column) and restored by the next block's
# kernel, which starts one column later. Unused slots are zero. A last kernel scales the pivot columns
# into L (`_lt_finalize!`), clears the upper triangle of F₁₁ (the GEMMs write it) and sets the unit
# diagonal; `pack_add!` adds the contribution block to the update stack. Shapes and launches depend on
# the analysis only: no host synchronization.

"Workgroup size of the regime-C LDLᵀ/LDLᴴ panel kernel."
const LDLT_C_WORKGROUP = 256

# workspace offsets (0-based) of Lb, Wb and the saved column of a front
@inline _ltc_offsets(f, m, nb, cb::Bool) = (lo = (cb ? m * m : 0); (lo, lo + f * (nb + 1), lo + 2 * f * (nb + 1)))

# control words of the panel kernel, after those of the LDLᵀ kernels
const _LTC_K0 = _LT_CTL + 1    # first column of the block
const _LTC_T0 = _LT_CTL + 2    # first slot of a pivot of this block (2 when it starts one column late)
const _LTC_LO = _LT_CTL + 3    # workspace offset (1-based) of Lb
const _LTC_WO = _LT_CTL + 4    # workspace offset (1-based) of Wb
const _LTC_SO = _LT_CTL + 5    # workspace offset (1-based) of the saved column
const _LTC_IDLE = _LT_CTL + 6  # 1: the front has fewer blocks (nothing to do in this launch)
const _LTC_CTL = _LT_CTL + 6

# workspace offset (0-based) of the slice of the front at position q of a chunk `nodes[qa:…]`: the slices of
# the fronts before it (`ldlt_c_work_len`, as `ldlt_c_chunk_end` lays them out)
@inline function _ltc_slice(nodes, qa, q, front_nrows, front_ncols, cb_ptr)
    base = 0
    @inbounds for q2 in qa:(q - 1)
        s2 = nodes[q2]
        base += ldlt_c_work_len(front_nrows[s2], front_ncols[s2], cb_ptr[s2] > 0)
    end
    return base
end

# front mode of the panel kernel (`Val(_Lazy{H})`)
struct _Lazy{H} end

# the panel as seen by the pivot search: columns `≥ kcur` with the block's pending pivots (slots t0:t1)
# subtracted; `H`: conj (Hermitian) or not (complex symmetric)
struct _LazyFront{A, B, K, H}
    a::A
    off::Int
    f::Int
    lb::B
    lo::Int
    wo::Int
    kind::K
    g0::Int     # pivot_kind index of slot t: g0 + t
    t0::Int
    t1::Int
    kcur::Int
end

@inline function _fget(F::_LazyFront{A, B, K, H}, i, j) where {A, B, K, H}
    @inbounds begin
        x = F.a[F.off + (j - 1) * F.f + i]
        j < F.kcur && return x
        h = Val(H)
        f = F.f
        t = F.t0
        while t <= F.t1
            if F.kind[F.g0 + t] == PIVOT_KIND_2X2_FIRST
                x -= (F.lb[F.lo + (t - 1) * f + i] * _cj(F.lb[F.wo + (t - 1) * f + j], h) +
                      F.lb[F.lo + t * f + i] * _cj(F.lb[F.wo + t * f + j], h))
                t += 2
            else
                x -= F.lb[F.lo + (t - 1) * f + i] * _cj(F.lb[F.wo + (t - 1) * f + j], h)
                t += 1
            end
        end
        return x
    end
end

@inline function _fset!(F::_LazyFront, i, j, v)
    @inbounds F.a[F.off + (j - 1) * F.f + i] = v
    return nothing
end

# the lazy front from the control words: pending slots up to the last pivot taken (column K + STEP - 1),
# or (`cur = false`) without the pivot of the current step
@inline function _ltc_front(fa, ctl, ::Val{H}, cur::Bool = true) where {H}
    @inbounds begin
        factor, lw, kind = fa
        k0 = Int(ctl[_LTC_K0])
        kc = Int(ctl[_LT_K]) + (cur ? Int(ctl[_LT_STEP]) : 0)
        return _LazyFront{typeof(factor), typeof(lw), typeof(kind), H}(factor, Int(ctl[_ST_LF]) - 1, Int(ctl[_ST_F]),
                                                                       lw, Int(ctl[_LTC_LO]) - 1, Int(ctl[_LTC_WO]) - 1,
                                                                       kind, Int(ctl[_LT_C0]) + k0 - 2,
                                                                       Int(ctl[_LTC_T0]), kc - k0, kc)
    end
end

@inline _lt_front(fa::Tuple{A, B, K}, ctl, ::Val{_Lazy{H}}) where {A, B, K, H} = _ltc_front(fa, ctl, Val(H))
@inline _lt_raw_front(fa::Tuple{A, B, K}, ctl, ::Val{_Lazy{H}}) where {A, B, K, H} =
    @inbounds _PanelFront(fa[1], Int(ctl[_ST_LF]) - 1, Int(ctl[_ST_F]))

# block setup (work item 1): the front at position q of the chunk, its control words, the block's columns,
# the workspace offsets of its slice
@inline function _ltc_setup!(ctl, nodes, qa, q, b, nb, super_ptr, front_ptr, front_nrows, front_ncols, cb_ptr,
                             pivot_kind)
    @inbounds begin
        IT = eltype(ctl)
        s = Int(nodes[q])
        f = Int(front_nrows[s])
        w = Int(front_ncols[s])
        ctl[_ST_NODE] = s % IT
        ctl[_ST_F] = f % IT
        ctl[_ST_W] = w % IT
        ctl[_ST_LF] = front_ptr[s] % IT
        _lt_reset!(ctl, super_ptr[s])
        c0 = Int(super_ptr[s])
        k0 = (b - 1) * nb + 1
        idle = k0 > w
        late = !idle && k0 > 1 && pivot_kind[c0 + k0 - 2] == PIVOT_KIND_2X2_FIRST   # k0 ended the last block
        ctl[_LTC_IDLE] = idle % IT
        ctl[_LT_K] = (late ? k0 + 1 : k0) % IT
        ctl[_LT_KEND] = (idle ? 0 : min(k0 + nb - 1, w)) % IT
        ctl[_LTC_K0] = k0 % IT
        ctl[_LTC_T0] = (late ? 2 : 1) % IT
        lo, wo, so = _ltc_offsets(f, f - w, nb, cb_ptr[s] > 0) .+ _ltc_slice(nodes, qa, q, front_nrows, front_ncols,
                                                                           cb_ptr)
        ctl[_LTC_LO] = (lo + 1) % IT
        ctl[_LTC_WO] = (wo + 1) % IT
        ctl[_LTC_SO] = (so + 1) % IT
    end
    return nothing
end

# zero Lb and Wb, restore the column saved by the last block, identity pivot order (first block)
@inline function _ltc_prepare!(factor, lw, piv, ctl, b, nb, li, ::Val{WG}) where {WG}
    @inbounds if ctl[_LTC_IDLE] == 0
        T = eltype(lw)
        f = Int(ctl[_ST_F])
        lo = Int(ctl[_LTC_LO]) - 1
        for q in li:WG:(2 * f * (nb + 1))
            lw[lo + q] = zero(T)
        end
        k0 = Int(ctl[_LTC_K0])
        if Int(ctl[_LTC_T0]) == 2
            so = Int(ctl[_LTC_SO]) - 1
            p0 = Int(ctl[_ST_LF]) - 1
            for i in (k0 - 1 + li):WG:f
                factor[p0 + (k0 - 1) * f + i] = lw[so + i]
            end
        end
        b == 1 && _lt_init_piv!(piv, ctl, li, Val(WG))
    end
    return nothing
end

# pass 1 of `_lt_pass1!` on the lazy panel, keeping the materialized column (rows k+1:f) after the saved
# column in the workspace for `_ltc_commit!`
@inline function _ltc_pass1!(fa, lw, ctl, red, redi, li, ::Val{NL}, p::_LDLTDevice{R}, h::Val) where {NL, R}
    @inbounds if li <= NL
        k = Int(ctl[_LT_K]) + Int(ctl[_LT_STEP])
        w = Int(ctl[_ST_W])
        f = Int(ctl[_ST_F])
        λ = zero(R)
        r = 0
        cm = zero(R)
        if k <= Int(ctl[_LT_KEND])                     # also for 'N': the commit reads the column
            F = _ltc_front(fa, ctl, h)
            co = Int(ctl[_LTC_SO]) - 1 + f
            for i in (k + li):NL:f
                x = _fget(F, i, k)
                lw[co + i] = x
                a = abs(x)
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

# swap rows p and q of Lb and Wb along with the interchange of the step (`_lt_swap!` moves the stored panel)
@inline function _ltc_swap_rows!(lw, ctl, nb, li, ::Val{WG}) where {WG}
    @inbounds begin
        k = Int(ctl[_LT_K])
        if k <= Int(ctl[_LT_KEND])
            r = Int(ctl[_LT_R])
            p = r == 0 ? k : k + 1
            q = r == 0 ? Int(ctl[_LT_C]) : r
            if p != q
                f = Int(ctl[_ST_F])
                for e in (li - 1):WG:(2 * (nb + 1) - 1)
                    base = Int(ctl[_LTC_LO]) - 1 + e * f     # Lb and Wb are adjacent: 2(nb + 1) columns
                    x = lw[base + p]
                    lw[base + p] = lw[base + q]
                    lw[base + q] = x
                end
            end
        end
    end
    return nothing
end

# materialize the pivot column(s) of the step below the pivot block: the stored panel gets the unscaled
# values, Wb the same, Lb the multipliers (as `_lt_update!` forms them)
@inline function _ltc_commit!(fa, lw, ctl, pv, li, ::Val{WG}, h::Val{H}) where {WG, H}
    @inbounds begin
        k = Int(ctl[_LT_K])
        if k <= Int(ctl[_LT_KEND])
            F = _ltc_front(fa, ctl, h, false)
            f = Int(ctl[_ST_F])
            step = Int(ctl[_LT_STEP])
            t = k - Int(ctl[_LTC_K0]) + 1
            lo = Int(ctl[_LTC_LO]) - 1
            wo = Int(ctl[_LTC_WO]) - 1
            # pass 1 left column k materialized when the pivot is the 1×1 one at k (no interchange)
            kept = step == 1 && Int(ctl[_LT_C]) == k
            co = Int(ctl[_LTC_SO]) - 1 + f
            for i in (k + step - 1 + li):WG:f
                if step == 1
                    x = kept ? lw[co + i] : _fget(F, i, k)
                    _fset!(F, i, k, x)
                    lw[wo + (t - 1) * f + i] = x
                    lw[lo + (t - 1) * f + i] = x / pv[1]
                else
                    x1 = _fget(F, i, k)
                    x2 = _fget(F, i, k + 1)
                    _fset!(F, i, k, x1)
                    _fset!(F, i, k + 1, x2)
                    lw[wo + (t - 1) * f + i] = x1
                    lw[wo + t * f + i] = x2
                    lw[lo + (t - 1) * f + i] = x1 * pv[1] + x2 * pv[3]
                    lw[lo + t * f + i] = x1 * pv[2] + x2 * pv[4]
                end
            end
        end
    end
    return nothing
end

# after the block: save the first column of the next block when a 2×2 pivot took it; statistics
@inline function _ltc_block_end!(factor, lw, stats, ctl, pivot_kind, b, li, ::Val{WG}) where {WG}
    @inbounds if ctl[_LTC_IDLE] == 0
        kend = Int(ctl[_LT_KEND])
        f = Int(ctl[_ST_F])
        c0 = Int(ctl[_LT_C0])
        if kend < Int(ctl[_ST_W]) && pivot_kind[c0 + kend - 1] == PIVOT_KIND_2X2_FIRST
            so = Int(ctl[_LTC_SO]) - 1
            p0 = Int(ctl[_ST_LF]) - 1
            for i in (kend + li):WG:f
                lw[so + i] = factor[p0 + kend * f + i]
            end
        end
        if li == 1
            base = (Int(ctl[_ST_NODE]) - 1) * FRONT_STATS_FIELDS
            for q in 1:5
                x = Int64(ctl[_LT_STAT + q])
                stats[base + q] = b == 1 ? x : stats[base + q] + x
            end
            stats[base + STAT_INFO] = 0
        end
    end
    return nothing
end

"""
    panel_ldlt_kernel!(backend, WG)(factor, lw, d, piv, pivot_kind, psign, perm, aux, stats, nodes, qa, b, nb,
                                    super_ptr, front_ptr, k, nbatch, front_nrows, front_ncols, cb_ptr, prm, Val(WG);
                                    ndrange = WG * count)

Block `b` (columns `(b - 1) nb + 1 : min(b nb, w)`) of the regime-C LDLᵀ/LDLᴴ
pivot steps of the `count` fronts `nodes[qa:(qa + count - 1)]`, one workgroup
each (panel of batch member `k` of `nbatch` at `member_panels(front_ptr, k, nbatch)[s]`, workspace slice in `lw` after those of the
fronts before it: `Lb`, `Wb`, the saved column, the pass-1 column): the
reference's pivot sequence on the lazily updated panel (see the top of
`src/numeric/ldlt_c.jl`), D, `piv`, the pivot kinds and the front's statistics
(added to those of the earlier blocks). Fronts with fewer blocks do nothing.
"""
@kernel function panel_ldlt_kernel!(factor, lw, d, piv, pivot_kind, psign, perm, aux, stats, nodes, qa, b, nb,
                                    super_ptr, front_ptr, mk, nbatch, front_nrows, front_ncols, cb_ptr,
                                    prm::_LDLTDevice{R, HERM}, ::Val{WG}) where {R, HERM, WG}
    @uniform TT = eltype(factor)
    @uniform IT = eltype(front_ncols)
    li = @index(Local, Linear)
    G = @index(Group, Linear)
    ctl = @localmem IT (_LTC_CTL,)
    pv = @localmem TT (_LT_NPV,)
    red = @localmem R (3 * WG,)
    redi = @localmem IT (2 * WG,)
    if li == 1
        _ltc_setup!(ctl, nodes, qa, qa + G - 1, b, nb, super_ptr, member_panels(front_ptr, mk, nbatch), front_nrows,
                    front_ncols, cb_ptr, pivot_kind)
    end
    @synchronize
    _ltc_prepare!(factor, lw, piv, ctl, b, nb, li, Val(WG))
    @synchronize
    _ltc_pass1!((factor, lw, pivot_kind), lw, ctl, red, redi, li, Val(WG), prm, Val(HERM))
    @synchronize
    for it in 1:nb
        _lt_stage1!(red, redi, ctl, li, Val(WG), Val(1))
        @synchronize
        if li == 1
            _lt_decide1!((factor, lw, pivot_kind), ctl, pv, red, redi, d, pivot_kind, piv, psign, perm, aux, prm,
                         Val(WG), Val(_Lazy{HERM}), Val(HERM))
        end
        @synchronize
        if ctl[_LT_PHASE] == 2
            _lt_pass2!((factor, lw, pivot_kind), ctl, red, redi, li, Val(WG), prm, Val(_Lazy{HERM}))
            @synchronize
            _lt_stage1!(red, redi, ctl, li, Val(WG), Val(2))
            @synchronize
            if li == 1
                _lt_decide2!((factor, lw, pivot_kind), ctl, pv, red, redi, d, pivot_kind, piv, psign, perm, aux, prm,
                             Val(WG), Val(_Lazy{HERM}), Val(HERM))
            end
            @synchronize
        end
        if ctl[_LT_PHASE] == 3
            _lt_pass3!((factor, lw, pivot_kind), ctl, red, redi, li, Val(WG), aux, prm, Val(_Lazy{HERM}))
            @synchronize
            _lt_stage1!(red, redi, ctl, li, Val(WG), Val(3))
            @synchronize
            if li == 1
                _lt_decide3!((factor, lw, pivot_kind), ctl, pv, red, redi, d, pivot_kind, piv, psign, perm, aux, prm,
                             Val(WG), Val(_Lazy{HERM}), Val(HERM))
            end
            @synchronize
        end
        _lt_swap!((factor, lw, pivot_kind), ctl, pv, piv, li, Val(WG), Val(_Lazy{HERM}), Val(HERM))
        _ltc_swap_rows!(lw, ctl, nb, li, Val(WG))
        @synchronize
        _ltc_commit!((factor, lw, pivot_kind), lw, ctl, pv, li, Val(WG), Val(HERM))
        @synchronize
        _ltc_pass1!((factor, lw, pivot_kind), lw, ctl, red, redi, li, Val(WG), prm, Val(HERM))
        @synchronize
    end
    _ltc_block_end!(factor, lw, stats, ctl, pivot_kind, b, li, Val(WG))
end

"""
    finish_ldlt_kernel!(backend, WG)(factor, d, pivot_kind, info, nodes, qa, super_ptr, front_ptr, k, nbatch,
                                     front_nrows, front_ncols, Val(HERM), Val(WG); ndrange = WG * count)

After the blocks of the regime-C fronts `nodes[qa:(qa + count - 1)]`, one
workgroup each: scale the pivot columns into L (`_lt_finalize!`), clear the
upper triangle of `F₁₁`, set the unit diagonal and the front's status.
"""
@kernel function finish_ldlt_kernel!(factor, d, pivot_kind, info, nodes, qa, super_ptr, front_ptr, mk, nbatch,
                                     front_nrows, front_ncols, ::Val{HERM}, ::Val{WG}) where {HERM, WG}
    @uniform IT = eltype(front_ncols)
    li = @index(Local, Linear)
    G = @index(Group, Linear)
    ctl = @localmem IT (_LT_CTL,)
    if li == 1
        @inbounds begin
            s = nodes[qa + G - 1]
            ctl[_ST_NODE] = s % IT
            ctl[_ST_F] = front_nrows[s] % IT
            ctl[_ST_W] = front_ncols[s] % IT
            ctl[_ST_LF] = member_panels(front_ptr, mk, nbatch)[s] % IT
            ctl[_LT_C0] = super_ptr[s] % IT
        end
    end
    @synchronize
    _lt_finalize!(factor, ctl, d, pivot_kind, li, Val(WG), Val(false), Val(HERM))
    @inbounds begin
        f = Int(ctl[_ST_F])
        w = Int(ctl[_ST_W])
        p = Int(ctl[_ST_LF]) - 1
        for q in (li - 1):WG:(w * w - 1)
            j = q ÷ w + 1
            i = q - (j - 1) * w + 1
            i <= j && (factor[p + (j - 1) * f + i] = i == j ? one(eltype(factor)) : zero(eltype(factor)))
        end
        li == 1 && (info[ctl[_ST_NODE]] = Int32(0))
    end
end

# regime C of the fronts `nodes[qa:qe]` of batch member `k`, concurrently: per block one panel launch, then the
# GEMMs of every front on its trailing columns and its contribution block; the last scaling; the contribution
# blocks onto the update stack
function _factor_chunk_ldlt_c!(N::Numeric{T}, S::Symbolic, qa::Int, qe::Int, k::Int, prm, gimpl::Symbol,
                               ::Val{NB}, ::Val{HERM}) where {T, NB, HERM}
    L, sc = S.layout, S.schedule
    nodes = sc.group_nodes
    nb = N.nbatch
    count = qe - qa + 1
    lw = view(N.work, 1:L.work_len)
    tB = HERM ? 'C' : 'T'
    d, piv, kind = _mview(N.d, k, nb), _mview(N.piv, k, nb), _mview(N.pivot_kind, k, nb)
    backend = KernelAbstractions.get_backend(N.factor)
    panel! = panel_ldlt_kernel!(backend, LDLT_C_WORKGROUP)
    nblocks = 0
    for q in qa:qe
        nblocks = max(nblocks, cld(sc.width[nodes[q]], NB))
    end
    for b in 1:nblocks
        panel!(N.factor, lw, d, piv, kind, N.psign, S.perm, _mview(N.aux, k, nb), _mview(N.stats, k, nb),
               S.group_nodes, qa, b, NB, S.super_ptr, S.front_ptr, k, nb, S.front_nrows, S.front_ncols, S.cb_ptr, prm,
               Val(LDLT_C_WORKGROUP); ndrange = LDLT_C_WORKGROUP * count)
        base = 0
        for q in qa:qe
            s = nodes[q]
            f, w = sc.rows[s], sc.width[s]
            m = f - w
            cb = m > 0 && L.cb_ptr[s] > 0
            if b <= cld(w, NB)
                p0 = panel_offset(L.panel_ptr, s, k, nb)
                P = reshape(view(N.factor, p0:(p0 + f * w - 1)), f, w)
                lo, wo, _ = _ltc_offsets(f, m, NB, cb)
                Lb = reshape(view(lw, (base + lo + 1):(base + lo + f * (NB + 1))), f, NB + 1)
                Wb = reshape(view(lw, (base + wo + 1):(base + wo + f * (NB + 1))), f, NB + 1)
                k1 = b * NB + 1                           # first column of the next block
                if k1 <= w
                    _gemm_impl!(gimpl, 'N', tB, -one(T), view(Lb, k1:f, :), view(Wb, k1:w, :), one(T),
                                view(P, k1:f, k1:w))
                end
                if cb
                    C = reshape(view(lw, (base + 1):(base + m * m)), m, m)
                    _gemm_impl!(gimpl, 'N', tB, -one(T), view(Lb, (w + 1):f, :), view(Wb, (w + 1):f, :),
                                b == 1 ? zero(T) : one(T), C)
                end
            end
            base += ldlt_c_work_len(f, w, cb)
        end
    end
    finish_ldlt_kernel!(backend, LDLT_C_WORKGROUP)(N.factor, d, kind, _iview(N.info, k, nb), S.group_nodes, qa,
                                                   S.super_ptr, S.front_ptr, k, nb, S.front_nrows, S.front_ncols,
                                                   Val(HERM),
                                                   Val(LDLT_C_WORKGROUP); ndrange = LDLT_C_WORKGROUP * count)
    base = 0
    for q in qa:qe
        s = nodes[q]
        f, w = sc.rows[s], sc.width[s]
        m = f - w
        cb = m > 0 && L.cb_ptr[s] > 0
        cb && pack_add!(_mview(N.stack, k, nb), L.cb_ptr[s], view(lw, (base + 1):(base + m * m)), m)
        base += ldlt_c_work_len(f, w, cb)
    end
    return nothing
end

# a regime-C launch group of the LDLᵀ/LDLᴴ phase: assembly launches, then its fronts in chunks
# (`ldlt_c_chunk_end`) for every active member
function _factorize_ldlt_c_group!(N::Numeric, S::Symbolic, nzval, a::Int, b::Int, maxchild::Int, prm, gimpl::Symbol,
                                  nbv::Val, herm::Val)
    zero_fronts!(N, S, a, b - a + 1)
    scatter_A!(N, S, nzval, a, b - a + 1)
    extend_add!(N, S, a, b - a + 1, maxchild)
    plan = N.plan
    q = a
    while q <= b
        e = ldlt_c_chunk_end(S.schedule, S.layout.cb_len, q, b, S.layout.work_len)
        if N.nbatch == 1
            _factor_chunk_ldlt_c!(N, S, q, e, 1, prm, gimpl, nbv, herm)
        else
            for j in 1:plan.nact[]
                _factor_chunk_ldlt_c!(N, S, q, e, Int(plan.members_host[j]), prm, gimpl, nbv, herm)
            end
        end
        q = e + 1
    end
    return nothing
end
