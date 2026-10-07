# Device LU (PLAN §2.2, §2.4, §3.3, M7): the in-front `L D U` factorization of
# the CPU reference (`src/reference/lu.jl`) as KA kernels, with the same pivot
# sequence: row interchanges inside the fully-summed rows of each front
# (threshold partial pivoting against the whole remaining front column),
# static perturbation `±ε` of tiny pivots, and the contribution block updated
# once per front with `F₂₂ ← F₂₂ − (L₂₁ D) U₁₂`.
#
# A front is kept as two lower-triangular structures with the layout of the
# Cholesky factor: `L` (the lower triangle of the front, diagonal included: the
# panel in `factor`, the contribution block in `stack`) and `Uᵀ` (the strict
# upper triangle, transposed: `ufactor`, `ustack`; its diagonal stays zero
# until the panel gets its unit diagonal). Entry `F[i, j]` is `L[i, j]` for
# `i ≥ j` and `Uᵀ[j, i]` otherwise (`_lu_get`), so the assembly, the extend-add,
# the update-stack layout, the regime-A local-memory layout and the solve
# sweeps of the symmetric structures are reused on both structures.
#
# * Regime A: `subtree_lu_kernel!`, one workgroup per subtree, the serial stack
#   of the two packed structures in two `@localmem` buffers (the schedule
#   budgets `2 sizeof(T)` bytes per packed entry, `schedule_elsize`).
# * Regimes B and C: `front_lu_kernel!`, one workgroup per front of a launch
#   group, fused assembly, the pivoted factorization of the two panels in
#   global memory and the update of the front's two packed contribution blocks.
#   Regime C uses this KA kernel too (no vendor `getrf`: see the T19 report).
#
# Per pivot step `k` (a loop over the `w` columns, uniform trip count): work item
# 1 chooses the pivot row, the workgroup swaps rows `k` and `r` (every column of
# the front), work item 1 stores D (perturbing a tiny pivot) and counts, the
# workgroup applies the rank-1 update to the rest of the fully-summed columns
# and rows, then scales column `k` into L and row `k` into U. No atomics:
# every destination has one writer per segment, so the result is deterministic.

"Workgroup size of the LU front kernel (regimes B and C)."
const LU_WORKGROUP = 128

# resolved pivoting parameters passed to the kernels (isbits)
struct _LUDevice{R}
    pivot::Bool     # in-block threshold partial pivoting
    u::R            # pivot_threshold
    eps::R          # pivot_epsilon (multiplied by max |aᵢⱼ| = real(aux[1]) when `scaled`)
    scaled::Bool    # pivot_epsilon_alg = "algo1"
end

function _lu_device_params(S::Symbolic, ::Type{T}, opts::Options) where {T}
    S.structure == STRUCTURE_GENERAL ||
        throw(InvalidValueError("LU needs structure \"G\", got \"$(convert(String, S.structure))\""))
    R = real(T)
    return _LUDevice{R}(lu_pivoting(opts.pivot_type), R(opts.pivot_threshold), R(resolved_pivot_epsilon(opts, R)),
                        opts.pivot_epsilon_alg == PIVOT_EPSILON_SCALED)
end

@inline function _lu_eps(p::_LUDevice, aux)
    p.scaled || return p.eps
    @inbounds a = real(aux[1])
    return a > 0 ? p.eps * a : p.eps
end

# F[i, j] of the front from its two structures (`Fl`: L, `Fu`: Uᵀ; `_PackedFront` or `_PanelFront`)
@inline _lu_get(Fl, Fu, i, j) = i >= j ? _fget(Fl, i, j) : _fget(Fu, j, i)
@inline _lu_set!(Fl, Fu, i, j, v) = i >= j ? _fset!(Fl, i, j, v) : _fset!(Fu, j, i, v)

# ---------------------------------------------------------------------------
# phases of one pivot step k (work item `li` of `WG`; `pk`: packed local fronts)

# work item 1: the pivot row of column k (`_lu_choose_row` of the reference)
@inline function _lu_choose!(fa, ua, ctl, k, p::_LUDevice, pk::Val)
    @inbounds begin
        IT = eltype(ctl)
        r = k
        if p.pivot
            Fl, Fu = _lt_front(fa, ctl, pk), _lt_front(ua, ctl, pk)
            f = Int(ctl[_ST_F])
            γ = zero(real(eltype(fa)))
            for i in k:f
                γ = max(γ, abs(_fget(Fl, i, k)))
            end
            best = abs(_fget(Fl, k, k))
            if !(best >= p.u * γ)
                for i in (k + 1):Int(ctl[_ST_W])
                    a = abs(_fget(Fl, i, k))
                    if a > best
                        r = i
                        best = a
                    end
                end
            end
        end
        ctl[_LT_C] = r % IT
    end
    return nothing
end

# interchange of the rows k and r ≤ w of the front (every column), and of their pivot order entries
@inline function _lu_swap!(fa, ua, ctl, piv, k, li, ::Val{WG}, pk::Val) where {WG}
    @inbounds begin
        r = Int(ctl[_LT_C])
        if r != k
            Fl, Fu = _lt_front(fa, ctl, pk), _lt_front(ua, ctl, pk)
            for j in li:WG:Int(ctl[_ST_F])
                x = _lu_get(Fl, Fu, k, j)
                _lu_set!(Fl, Fu, k, j, _lu_get(Fl, Fu, r, j))
                _lu_set!(Fl, Fu, r, j, x)
            end
            if li == 1
                c0 = Int(ctl[_LT_C0])
                x = piv[c0 + k - 1]
                piv[c0 + k - 1] = piv[c0 + r - 1]
                piv[c0 + r - 1] = x
            end
        end
    end
    return nothing
end

# work item 1: store the pivot in D (perturbing a tiny one), its kind and the statistics; `pv[1] = d`
@inline function _lu_pivot!(fa, ctl, pv, d, pivot_kind, aux, k, p::_LUDevice, pk::Val)
    @inbounds begin
        IT = eltype(ctl)
        T = eltype(fa)
        Fl = _lt_front(fa, ctl, pk)
        g = Int(ctl[_LT_C0]) + k - 1
        ε = _lu_eps(p, aux)
        dk = _fget(Fl, k, k)
        kind = PIVOT_KIND_1X1
        if !(abs(dk) >= ε) || iszero(dk)                  # tiny (or NaN, or exactly zero): perturb
            iszero(dk) && (ctl[_LT_STAT + STAT_NZERO] += one(IT))
            dk = _perturbation_sign(dk, 0, !(T <: Complex)) * ε
            kind = PIVOT_KIND_PERTURBED
            ctl[_LT_STAT + STAT_NPERTURBED] += one(IT)
            iszero(dk) && ctl[_ST_STATUS] == 0 && (ctl[_ST_STATUS] = k % IT)   # ε = 0: the front fails here
        end
        _fset!(Fl, k, k, dk)
        d[g] = dk
        pivot_kind[g] = kind
        pv[1] = dk
    end
    return nothing
end

# rank-1 update with the unscaled pivot column and row: F[i, j] -= F[i, k] / d F[k, j] for i, j > k with
# j ≤ w (columns k+1:w, rows to f) or i ≤ w (rows k+1:w, columns w+1:f)
@inline function _lu_update!(fa, ua, ctl, pv, k, li, ::Val{WG}, pk::Val) where {WG}
    @inbounds begin
        Fl, Fu = _lt_front(fa, ctl, pk), _lt_front(ua, ctl, pk)
        f = Int(ctl[_ST_F])
        w = Int(ctl[_ST_W])
        nr = f - k
        nc = w - k
        n1 = nr * nc
        d = pv[1]
        for q in (li - 1):WG:(n1 + nc * (f - w) - 1)
            if q < n1
                i = k + 1 + q % nr
                j = k + 1 + q ÷ nr
            else
                q2 = q - n1
                i = k + 1 + q2 % nc
                j = w + 1 + q2 ÷ nc
            end
            _lu_set!(Fl, Fu, i, j, _lu_get(Fl, Fu, i, j) - _lu_get(Fl, Fu, i, k) / d * _lu_get(Fl, Fu, k, j))
        end
    end
    return nothing
end

# scale column k below the pivot into L and row k right of the pivot into U
@inline function _lu_scale!(fa, ua, ctl, pv, k, li, ::Val{WG}, pk::Val) where {WG}
    @inbounds begin
        Fl, Fu = _lt_front(fa, ctl, pk), _lt_front(ua, ctl, pk)
        nr = Int(ctl[_ST_F]) - k
        d = pv[1]
        for q in (li - 1):WG:(2 * nr - 1)
            i = k + 1 + q % nr
            if q < nr
                _fset!(Fl, i, k, _fget(Fl, i, k) / d)       # L[i, k]
            else
                _fset!(Fu, i, k, _fget(Fu, i, k) / d)       # U[k, i]
            end
        end
    end
    return nothing
end

# F₂₂ ← F₂₂ − (L₂₁ D) U₁₂ on the packed m×m blocks at `coff` of `ca` (lower triangle, diagonal included) and
# `cu` (strict upper triangle, transposed); `coff < 0`: no block
@inline function _lu_cb_update!(fa, ua, ctl, ca, cu, coff, d, li, ::Val{WG}, pk::Val) where {WG}
    @inbounds begin
        T = eltype(fa)
        f = Int(ctl[_ST_F])
        w = Int(ctl[_ST_W])
        m = f - w
        if coff >= 0 && m > 0
            Fl, Fu = _lt_front(fa, ctl, pk), _lt_front(ua, ctl, pk)
            c0 = Int(ctl[_LT_C0])
            for q in (li - 1):WG:(m * m - 1)
                jj = q ÷ m + 1
                ii = q - (jj - 1) * m + 1
                acc = zero(T)
                for k in 1:w
                    acc += _fget(Fl, w + ii, k) * d[c0 + k - 1] * _fget(Fu, w + jj, k)
                end
                if ii >= jj
                    pos = coff + _packed(ii, jj, m)
                    ca[pos] -= acc
                else
                    pos = coff + _packed(jj, ii, m)
                    cu[pos] -= acc
                end
            end
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# regimes B/C: one workgroup per front, panels in global memory

# scatter the values of A into the zeroed panels of front `s`: positive `amap` offsets into `factor` (L),
# negative ones into `ufactor` (Uᵀ); runs of one destination are summed by one work item in `nzval` order
@inline function _scatter_front_lu!(factor, ufactor, nzval, amap, amap_ptr, amap_src, s, shift, li,
                                    ::Val{WG}) where {WG}
    @inbounds begin
        T = eltype(factor)
        a = amap_ptr[s]
        b = amap_ptr[s + 1] - 1
        for k in (a + li - 1):WG:b
            dest = amap[amap_src[k]]
            if k == a || amap[amap_src[k - 1]] != dest
                v = zero(T)
                kk = k
                while kk <= b
                    p = amap_src[kk]
                    amap[p] == dest || break
                    v += T(nzval[p])
                    kk += 1
                end
                if dest > 0
                    factor[dest + shift] = v
                else
                    ufactor[-dest + shift] = v
                end
            end
        end
    end
    return nothing
end

# unit diagonals of both panels, statistics and status
@inline function _lu_front_finish!(factor, ufactor, stats, info, ctl, li, ::Val{WG}) where {WG}
    @inbounds begin
        Fl, Fu = _lt_front(factor, ctl, Val(false)), _lt_front(ufactor, ctl, Val(false))
        for j in li:WG:Int(ctl[_ST_W])
            _fset!(Fl, j, j, one(eltype(factor)))
            _fset!(Fu, j, j, one(eltype(factor)))
        end
        li == 1 && _lt_stats!(stats, info, ctl, ctl[_ST_NODE])
    end
    return nothing
end

"""
    front_lu_kernel!(backend, WG)(factor, ufactor, stack, ustack, info, stats, d, piv, pivot_kind, aux, nzval, amap,
                                  amap_ptr, amap_src, nodes, bm, super_ptr, front_ptr, front_nrows, front_ncols,
                                  cb_ptr, child_ptr, child_list, relind_ptr, relind, maxchild, prm, Val(WG);
                                  ndrange = WG * count * bm.nact)

LU kernel of regimes B and C: workgroup `G` takes front
`s = nodes[bm.first + g - 1]` of batch member `k` (`g`, `k` from the
[`BatchMap`](@ref) `bm`; the per-member arrays are the member's), zeroes and
assembles its `L` and `Uᵀ` structures (A through the `amap`, then the
`maxchild` children's two packed contribution blocks in `child_list` order),
factors its `w` fully-summed columns in place in the two panels with in-block
row pivoting, threshold test and perturbation (`prm`, an `_LUDevice`), writes
D, the local row order `piv`, the pivot kinds and the front's statistics,
updates its two packed contribution blocks (`cb_ptr[s] > 0`) with
`F₂₂ − (L₂₁ D) U₁₂` and leaves unit-lower panels of `L` and `Uᵀ`.
"""
@kernel function front_lu_kernel!(factor, ufactor, stack, ustack, info, stats, d, piv, pivot_kind, aux, nzval, amap,
                                  amap_ptr, amap_src, nodes, bm, super_ptr, front_ptr, front_nrows, front_ncols,
                                  cb_ptr, child_ptr, child_list, relind_ptr, relind, maxchild, prm::_LUDevice,
                                  ::Val{WG}) where {WG}
    @uniform TT = eltype(factor)
    @uniform IT = eltype(front_ptr)
    li = @index(Local, Linear)
    G = @index(Group, Linear)
    ctl = @localmem IT (_LT_CTL,)
    pv = @localmem TT (1,)
    if li == 1
        @inbounds s = nodes[bm.first + _bm_node(bm, G) - 1]
        _lt_front_setup!(ctl, s, super_ptr, member_panels(front_ptr, _bm_gmember(bm, G), bm.nbatch), front_nrows,
                         front_ncols, cb_ptr)
    end
    @synchronize
    @inbounds s = nodes[bm.first + _bm_node(bm, G) - 1]
    k = _bm_gmember(bm, G)
    nb = bm.nbatch
    P = member_panels(front_ptr, k, nb)
    _zero_front!(factor, _mview(stack, k, nb), s, li, P, front_nrows, front_ncols, cb_ptr, Val(WG))
    _zero_front!(ufactor, _mview(ustack, k, nb), s, li, P, front_nrows, front_ncols, cb_ptr, Val(WG))
    _lt_init_piv!(_mview(piv, k, nb), ctl, li, Val(WG))
    @synchronize
    @inbounds s = nodes[bm.first + _bm_node(bm, G) - 1]
    k = _bm_gmember(bm, G)
    _scatter_front_lu!(factor, ufactor, _mview(nzval, k, bm.nbatch), amap, amap_ptr, amap_src, s,
                       _member_shift(front_ptr, s, k, bm.nbatch), li, Val(WG))
    @synchronize
    for kc in 1:maxchild
        @inbounds s = nodes[bm.first + _bm_node(bm, G) - 1]
        k = _bm_gmember(bm, G)
        nb = bm.nbatch
        P = member_panels(front_ptr, k, nb)
        _extend_add_child!(factor, _mview(stack, k, nb), s, kc, li, P, front_nrows, front_ncols, cb_ptr, child_ptr,
                           child_list, relind_ptr, relind, Val(WG))
        _extend_add_child!(ufactor, _mview(ustack, k, nb), s, kc, li, P, front_nrows, front_ncols, cb_ptr, child_ptr,
                           child_list, relind_ptr, relind, Val(WG))
        @synchronize
    end
    for it in 1:ctl[_ST_W]
        if li == 1
            _lu_choose!(factor, ufactor, ctl, it, prm, Val(false))
        end
        @synchronize
        _lu_swap!(factor, ufactor, ctl, _mview(piv, _bm_gmember(bm, G), bm.nbatch), it, li, Val(WG), Val(false))
        @synchronize
        if li == 1
            k = _bm_gmember(bm, G)
            nb = bm.nbatch
            _lu_pivot!(factor, ctl, pv, _mview(d, k, nb), _mview(pivot_kind, k, nb), _mview(aux, k, nb), it, prm,
                       Val(false))
        end
        @synchronize
        _lu_update!(factor, ufactor, ctl, pv, it, li, Val(WG), Val(false))
        @synchronize
        _lu_scale!(factor, ufactor, ctl, pv, it, li, Val(WG), Val(false))
        @synchronize
    end
    @inbounds s = nodes[bm.first + _bm_node(bm, G) - 1]
    @inbounds coff = Int(cb_ptr[s]) - 1
    k = _bm_gmember(bm, G)
    nb = bm.nbatch
    _lu_cb_update!(factor, ufactor, ctl, _mview(stack, k, nb), _mview(ustack, k, nb), coff, _mview(d, k, nb), li,
                   Val(WG), Val(false))
    _lu_front_finish!(factor, ufactor, _mview(stats, k, nb), _iview(info, k, nb), ctl, li, Val(WG))
end

# ---------------------------------------------------------------------------
# regime A: one workgroup per subtree, the two structures in local memory

# scatter A into the zeroed packed structures of the current node (`buf`: L, `ubuf`: Uᵀ)
@inline function _subtree_scatter_lu!(buf, ubuf, ctl, nzval, amap, amap_ptr, amap_src, front_ptr, li,
                                      ::Val{WG}) where {WG}
    @inbounds begin
        T = eltype(buf)
        v = ctl[_ST_NODE]
        f = ctl[_ST_F]
        lf = ctl[_ST_LF]
        p0 = front_ptr[v]
        a = amap_ptr[v]
        b = amap_ptr[v + 1] - 1
        for k in (a + li - 1):WG:b
            dest = amap[amap_src[k]]
            if k == a || amap[amap_src[k - 1]] != dest
                x = zero(T)
                kk = k
                while kk <= b
                    p = amap_src[kk]
                    amap[p] == dest || break
                    x += T(nzval[p])
                    kk += 1
                end
                loc = abs(dest) - p0                     # 0-based position in the f×w panel
                j = loc ÷ f + 1
                i = loc - (j - 1) * f + 1
                if dest > 0
                    buf[lf + _packed(i, j, f) - 1] = x
                else
                    ubuf[lf + _packed(i, j, f) - 1] = x
                end
            end
        end
    end
    return nothing
end

# write the unit-lower panels of L and Uᵀ, statistics and status; the subtree root writes its two
# contribution blocks to the update stacks
@inline function _lu_subtree_write!(factor, ufactor, stack, ustack, info, stats, buf, ubuf, ctl, front_ptr, cb_ptr, li,
                                    ::Val{WG}) where {WG}
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
            if i > j
                factor[p0 + q] = buf[lf + _packed(i, j, f)]
                ufactor[p0 + q] = ubuf[lf + _packed(i, j, f)]
            else
                x = i == j ? one(T) : zero(T)
                factor[p0 + q] = x
                ufactor[p0 + q] = x
            end
        end
        li == 1 && _lt_stats!(stats, info, ctl, v)
        c0 = cb_ptr[v]
        if c0 > 0
            m = f - w
            src = _lt_local_cb_offset(ctl) + 1
            for q in (li - 1):WG:(m * (m + 1) ÷ 2 - 1)
                stack[c0 + q] = buf[src + q]
                ustack[c0 + q] = ubuf[src + q]
            end
        end
    end
    return nothing
end

"""
    subtree_lu_kernel!(backend, WG)(factor, ufactor, stack, ustack, info, stats, d, piv, pivot_kind, aux, nzval, amap,
                                    amap_ptr, amap_src, trees, bm, subtree_ptr, subtree_nodes, super_ptr, front_ptr,
                                    front_nrows, front_ncols, cb_ptr, local_front, local_cb, child_ptr, child_list,
                                    relind_ptr, relind, prm, Val(NE), Val(WG); ndrange = WG * count * bm.nact)

LU kernel of regime A: workgroup `G` takes subtree `t = trees[bm.first + g - 1]`
of batch member `k` and processes its supernodes in order with the serial
stacks of the packed `L` and `Uᵀ` structures in two `@localmem` buffers of
`NE` entries each (the layout of [`subtree_local_layout`](@ref), shared by
both): zero, scatter A, extend-add the children, factor the front's first `w`
columns with the pivoted `L D U` of [`front_lu_kernel!`](@ref), update the
trailing contribution blocks, write the unit-lower panels, D, `piv`, the
pivot kinds and the statistics, then move both blocks down or (subtree root)
write them to the update stacks.
"""
@kernel function subtree_lu_kernel!(factor, ufactor, stack, ustack, info, stats, d, piv, pivot_kind, aux, nzval, amap,
                                    amap_ptr, amap_src, trees, bm, subtree_ptr, subtree_nodes, super_ptr, front_ptr,
                                    front_nrows, front_ncols, cb_ptr, local_front, local_cb, child_ptr, child_list,
                                    relind_ptr, relind, prm::_LUDevice, ::Val{NE}, ::Val{WG}) where {NE, WG}
    @uniform TT = eltype(factor)
    @uniform IT = eltype(subtree_nodes)
    li = @index(Local, Linear)
    G = @index(Group, Linear)
    buf = @localmem TT (NE,)
    ubuf = @localmem TT (NE,)
    ctl = @localmem IT (_LT_CTL,)
    pv = @localmem TT (1,)
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
        _subtree_zero!(ubuf, ctl, li, Val(WG))
        _lt_init_piv!(_mview(piv, _bm_gmember(bm, G), bm.nbatch), ctl, li, Val(WG))
        @synchronize
        _subtree_scatter_lu!(buf, ubuf, ctl, _mview(nzval, _bm_gmember(bm, G), bm.nbatch), amap, amap_ptr, amap_src,
                             front_ptr, li, Val(WG))
        @synchronize
        for kc in 1:ctl[_ST_NCHILD]
            _subtree_extend_add!(buf, ctl, kc, child_ptr, child_list, front_nrows, front_ncols, local_cb, relind_ptr,
                                 relind, li, Val(WG))
            _subtree_extend_add!(ubuf, ctl, kc, child_ptr, child_list, front_nrows, front_ncols, local_cb, relind_ptr,
                                 relind, li, Val(WG))
            @synchronize
        end
        for it in 1:ctl[_ST_W]
            if li == 1
                _lu_choose!(buf, ubuf, ctl, it, prm, Val(true))
            end
            @synchronize
            _lu_swap!(buf, ubuf, ctl, _mview(piv, _bm_gmember(bm, G), bm.nbatch), it, li, Val(WG), Val(true))
            @synchronize
            if li == 1
                mb = _bm_gmember(bm, G)
                nb = bm.nbatch
                _lu_pivot!(buf, ctl, pv, _mview(d, mb, nb), _mview(pivot_kind, mb, nb), _mview(aux, mb, nb), it, prm,
                           Val(true))
            end
            @synchronize
            _lu_update!(buf, ubuf, ctl, pv, it, li, Val(WG), Val(true))
            @synchronize
            _lu_scale!(buf, ubuf, ctl, pv, it, li, Val(WG), Val(true))
            @synchronize
        end
        mb = _bm_gmember(bm, G)
        off = _lt_local_cb_offset(ctl)
        _lu_cb_update!(buf, ubuf, ctl, buf, ubuf, off, _mview(d, mb, bm.nbatch), li, Val(WG), Val(true))
        @synchronize
        mb = _bm_gmember(bm, G)
        nb = bm.nbatch
        _lu_subtree_write!(factor, ufactor, _mview(stack, mb, nb), _mview(ustack, mb, nb), _iview(info, mb, nb),
                           _mview(stats, mb, nb), buf, ubuf, ctl, member_panels(front_ptr, mb, nb), cb_ptr, li,
                           Val(WG))
        @synchronize
        for r in 1:ctl[_ST_ROUNDS]
            _subtree_move!(buf, ctl, r, li, Val(WG))
            _subtree_move!(ubuf, ctl, r, li, Val(WG))
            @synchronize
        end
    end
end

# ---------------------------------------------------------------------------
# driver

function _launch_front_lu!(N::Numeric, S::Symbolic, nzval, first, count, maxchild, prm, ::Val{WG}) where {WG}
    bm = batch_map(N; first)
    kernel! = front_lu_kernel!(KernelAbstractions.get_backend(N.factor), WG)
    kernel!(N.factor, N.ufactor, N.stack, N.ustack, N.info, N.stats, N.d, N.piv, N.pivot_kind, N.aux, nzval, S.amap,
            S.amap_ptr, S.amap_src, S.group_nodes, bm, S.super_ptr, S.front_ptr, S.front_nrows, S.front_ncols,
            S.cb_ptr, S.child_ptr, S.child_list, S.relind_ptr, S.relind, Int(maxchild), prm, Val(WG);
            ndrange = WG * count * bm.nact)
    return nothing
end

function _launch_subtrees_lu!(N::Numeric{T}, S::Symbolic, nzval, first, count, prm, ::Val{LB},
                              ::Val{WG}) where {T, LB, WG}
    bm = batch_map(N; first)
    kernel! = subtree_lu_kernel!(KernelAbstractions.get_backend(N.factor), WG)
    kernel!(N.factor, N.ufactor, N.stack, N.ustack, N.info, N.stats, N.d, N.piv, N.pivot_kind, N.aux, nzval, S.amap,
            S.amap_ptr, S.amap_src, S.group_nodes, bm, S.subtree_ptr, S.subtree_nodes, S.super_ptr, S.front_ptr,
            S.front_nrows, S.front_ncols, S.cb_ptr, S.local_front, S.local_cb, S.child_ptr, S.child_list,
            S.relind_ptr, S.relind, prm, Val((LB - SUBTREE_LOCAL_RESERVE) ÷ (2 * sizeof(T))), Val(WG);
            ndrange = WG * count * bm.nact)
    return nothing
end

"""
    factorize_lu!(numeric, symbolic, nzval; opts = Options()) -> info::Int

Multifrontal `P_r P A Pᵀ = L D U` of a general matrix (structure `"G"`) on the
device, with the in-front row pivoting and the static perturbation of the
reference [`ref_lu!`](@ref) (same pivot choice, same storage):
`opts.pivot_type` ([`lu_pivoting`](@ref)), `pivot_threshold`, `pivot_epsilon`
and `pivot_epsilon_alg`. Regime-A groups are one [`subtree_lu_kernel!`](@ref)
launch per budget class; every regime-B and regime-C launch group is one
[`front_lu_kernel!`](@ref) launch (fused assembly and factorization, one
workgroup per front; no vendor calls); then [`reduce_stats!`](@ref).
`opts.user_host_interrupt` is polled before every launch group
([`InterruptedError`](@ref)). With `pivot_epsilon_alg = "algo1"`
[`abs_max!`](@ref) computes the scale first. Fills `numeric.factor` (`L`),
`numeric.ufactor` (`Uᵀ`), `d`, `piv` (local row order), `pivot_kind`, `stats`
and `totals`. Returns `info` as [`factorize_ldlt!`](@ref): an exactly zero
pivot stays zero with an effective `ε = 0` and the factorization fails there
(`info` = the original column, as [`ref_lu!`](@ref)); with `ε > 0` it always
completes (`info = 0`). The phase allocates nothing on the device, is
deterministic, and synchronizes with the host only to read `info` when the
factorization can fail (`pivot_epsilon = 0`, or `pivot_epsilon_alg = "algo1"`).
Uniform batch as [`factorize!`](@ref).
"""
function factorize_lu!(N::Numeric{T}, S::Symbolic, nzval::AbstractVector; opts::Options = Options()) where {T}
    _check_numeric(N, S, nzval)
    prm = _lu_device_params(S, T, opts)
    prm.scaled && abs_max!(N.aux, nzval, batch_map(N))
    plan = N.plan
    flag = opts.user_host_interrupt
    for k in eachindex(plan.sub_first)
        _poll_interrupt(flag)
        a, b = plan.sub_first[k], plan.sub_last[k]
        _with_local_bytes(plan.sub_local[k]) do lb
            _launch_subtrees_lu!(N, S, nzval, a, b - a + 1, prm, lb, Val(SUBTREE_WORKGROUP))
        end
    end
    for k in eachindex(plan.group_first)
        _poll_interrupt(flag)
        a, b = plan.group_first[k], plan.group_last[k]
        _launch_front_lu!(N, S, nzval, a, b - a + 1, plan.group_maxchild[k], prm, Val(LU_WORKGROUP))
    end
    assemble_schur!(N, S, nzval)
    reduce_stats!(N, S)
    return _numeric_info!(N, S, iszero(prm.eps) || prm.scaled)
end
