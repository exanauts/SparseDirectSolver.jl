# Regime A (PLAN §2.2, §2.3 step 5, §2.4): one fused kernel launch per budget
# class, one 1-D workgroup per leaf subtree. The workgroup walks the subtree's
# supernodes in processing order (`subtree_nodes`, a postorder) and keeps the
# serial multifrontal stack in `@localmem` (layout: `subtree_local_layout`):
#
# 1. zero the packed `f×f` lower triangle of the front (at `local_front[v]`,
#    right above the children's contribution blocks);
# 2. scatter A into it through `amap_ptr`/`amap_src`/`amap` (same run summation
#    as the global assembly kernel);
# 3. extend-add the children's packed contribution blocks (local memory, at
#    `local_cb[c]`) in `child_list` order, one barrier per child;
# 4. right-looking unblocked Cholesky of the first `w` columns of the whole front
#    (two barriers per column), which leaves L11, L21 and the Schur complement
#    F22 − L21 L21ᴴ in the trailing packed `m×m` triangle;
# 5. write the `f×w` panel (zero strict upper triangle) and the status to global
#    memory; the subtree root writes its contribution block (the trailing
#    triangle, already in the packed stack format) to the update stack;
# 6. any other node moves its contribution block down to `local_cb[v]` (in rounds
#    of at most the move distance, so overlapping source and destination are safe).
#
# Per-node values (node, sizes, offsets, trip counts, status) live in a small
# `@localmem` control array written by work item 1 before a barrier, so every
# loop that contains a barrier has a trip count that is uniform over the
# workgroup and readable on the KA CPU backend (which evaluates loop headers
# outside its work-item loops). No atomics: every destination has one writer per
# segment and a fixed order, so the result is deterministic.

"Workgroup size of the fused regime-A kernel."
const SUBTREE_WORKGROUP = 128

# slots of the control array of the regime-A kernel
const _ST_FIRST = 1      # subtree_ptr[t]
const _ST_COUNT = 2      # nodes in the subtree
const _ST_NODE = 3       # current supernode v
const _ST_F = 4          # rows f
const _ST_W = 5          # columns w
const _ST_LF = 6         # local offset of the packed front
const _ST_NCHILD = 7     # number of children
const _ST_ROUNDS = 8     # rounds of the contribution-block move
const _ST_STATUS = 9     # first non-positive pivot (local column), 0 = none
const _ST_LC = 10        # local destination of the contribution block (0: subtree root)
const _ST_CTL = 10

# work item 1: load the descriptors of the `k`-th node of the subtree
@inline function _subtree_setup!(ctl, k, subtree_nodes, front_nrows, front_ncols, local_front, local_cb,
                                 child_ptr)
    @inbounds begin
        IT = eltype(ctl)
        v = subtree_nodes[ctl[_ST_FIRST] + k - 1]
        f = front_nrows[v]
        w = front_ncols[v]
        m = f - w
        lf = local_front[v]
        lc = local_cb[v]
        ctl[_ST_NODE] = v % IT
        ctl[_ST_F] = f % IT
        ctl[_ST_W] = w % IT
        ctl[_ST_LF] = lf % IT
        ctl[_ST_NCHILD] = (child_ptr[v + 1] - child_ptr[v]) % IT
        ctl[_ST_LC] = lc % IT
        rounds = 0
        if lc > 0 && m > 0
            d = lf + w * (2 * f - w + 1) ÷ 2 - lc        # move distance (> 0)
            c = m * (m + 1) ÷ 2
            rounds = (c + d - 1) ÷ d
        end
        ctl[_ST_ROUNDS] = rounds % IT
        ctl[_ST_STATUS] = zero(IT)
    end
    return nothing
end

@inline function _subtree_zero!(buf, ctl, li, ::Val{WG}) where {WG}
    @inbounds begin
        lf = ctl[_ST_LF]
        f = ctl[_ST_F]
        z = zero(eltype(buf))
        for q in (lf + li - 1):WG:(lf + f * (f + 1) ÷ 2 - 1)
            buf[q] = z
        end
    end
    return nothing
end

# scatter A into the zeroed packed front (the run of duplicates of a destination is summed by one work item)
@inline function _subtree_scatter!(buf, ctl, nzval, amap, amap_ptr, amap_src, front_ptr, li, ::Val{WG}) where {WG}
    @inbounds begin
        T = eltype(buf)
        v = ctl[_ST_NODE]
        f = ctl[_ST_F]
        lf = ctl[_ST_LF]
        p0 = front_ptr[v]
        a = amap_ptr[v]
        b = amap_ptr[v + 1] - 1
        for k in (a + li - 1):WG:b
            dest = abs(amap[amap_src[k]])
            if k == a || abs(amap[amap_src[k - 1]]) != dest
                x = zero(T)
                kk = k
                while kk <= b
                    p = amap_src[kk]
                    off = amap[p]
                    abs(off) == dest || break
                    x += _amap_value(T, nzval[p], off)
                    kk += 1
                end
                loc = dest - p0                          # 0-based position in the f×w panel
                j = loc ÷ f + 1
                i = loc - (j - 1) * f + 1
                buf[lf + _packed(i, j, f) - 1] = x
            end
        end
    end
    return nothing
end

# add the packed contribution block of the `kc`-th child (local memory) to the packed front
@inline function _subtree_extend_add!(buf, ctl, kc, child_ptr, child_list, front_nrows, front_ncols, local_cb,
                                      relind_ptr, relind, li, ::Val{WG}) where {WG}
    @inbounds begin
        v = ctl[_ST_NODE]
        f = ctl[_ST_F]
        lf = ctl[_ST_LF]
        c = child_list[child_ptr[v] + kc - 1]
        mc = front_nrows[c] - front_ncols[c]
        lc = local_cb[c]
        r0 = relind_ptr[c] - 1
        for q in (li - 1):WG:(mc * mc - 1)
            jj = q ÷ mc + 1
            ii = q - (jj - 1) * mc + 1
            if ii >= jj
                ri = relind[r0 + ii]
                rj = relind[r0 + jj]
                buf[lf + _packed(ri, rj, f) - 1] += buf[lc + _packed(ii, jj, mc) - 1]
            end
        end
    end
    return nothing
end

# column j, first half: check the pivot and scale the column below the diagonal by it
# (every work item recomputes `d = sqrt(ajj)` from local memory: cheaper than sharing
# it through a barrier, and it keeps the second half free of divisions)
@inline function _subtree_chol_update!(buf, ctl, piv, j, li, ::Val{WG}) where {WG}
    @inbounds begin
        if ctl[_ST_STATUS] == 0
            f = ctl[_ST_F]
            lf = ctl[_ST_LF] - 1
            ajj = real(buf[lf + _packed(j, j, f)])
            if ajj > 0
                rd = inv(sqrt(ajj))                        # strictly below the diagonal only:
                for i in (j + li):WG:f                     # the diagonal is replaced in the second
                    buf[lf + _packed(i, j, f)] *= rd       # half, so no work item reads a value
                end                                        # another one is overwriting
            elseif li == 1
                ctl[_ST_STATUS] = j % eltype(ctl)          # not positive (or NaN): stop, as LAPACK potrf
            end
        end
    end
    return nothing
end

# column j, second half: rank-1 update of the trailing front with the scaled column
# (multiply-add only: the former divide-in-the-update doubled the flop cost in
# f64 divisions, the bulk of the fused regime-A kernel's time)
@inline function _subtree_chol_scale!(buf, ctl, piv, j, li, ::Val{WG}) where {WG}
    @inbounds begin
        if ctl[_ST_STATUS] == 0
            f = ctl[_ST_F]
            lf = ctl[_ST_LF] - 1
            li == 1 && (buf[lf + _packed(j, j, f)] = sqrt(real(buf[lf + _packed(j, j, f)])))
            r = f - j
            for q in (li - 1):WG:(r * (r + 1) ÷ 2 - 1)
                a, b = _tri_decode(q, r)
                i = j + a
                k = j + b
                buf[lf + _packed(i, k, f)] -= buf[lf + _packed(i, j, f)] * conj(buf[lf + _packed(k, j, f)])
            end
        end
    end
    return nothing
end

# write the panel and the status; the subtree root writes its contribution block to the update stack
@inline function _subtree_write!(factor, stack, info, buf, ctl, front_ptr, cb_ptr, li, ::Val{WG}) where {WG}
    @inbounds begin
        v = ctl[_ST_NODE]
        f = ctl[_ST_F]
        w = ctl[_ST_W]
        lf = ctl[_ST_LF] - 1
        p0 = front_ptr[v]
        z = zero(eltype(factor))
        for q in (li - 1):WG:(f * w - 1)
            j = q ÷ f + 1
            i = q - (j - 1) * f + 1
            factor[p0 + q] = i >= j ? buf[lf + _packed(i, j, f)] : z
        end
        li == 1 && (info[v] = ctl[_ST_STATUS] % Int32)
        c0 = cb_ptr[v]
        if c0 > 0
            m = f - w
            src = lf + w * (2 * f - w + 1) ÷ 2 + 1      # the trailing packed m×m triangle
            for q in (li - 1):WG:(m * (m + 1) ÷ 2 - 1)
                stack[c0 + q] = buf[src + q]
            end
        end
    end
    return nothing
end

# round r of the move of the contribution block from the tail of the front down to `local_cb[v]`
@inline function _subtree_move!(buf, ctl, r, li, ::Val{WG}) where {WG}
    @inbounds begin
        f = ctl[_ST_F]
        w = ctl[_ST_W]
        m = f - w
        lc = ctl[_ST_LC]
        src = ctl[_ST_LF] + w * (2 * f - w + 1) ÷ 2
        d = src - lc
        a = (r - 1) * d
        b = min(r * d, m * (m + 1) ÷ 2) - 1
        for q in (a + li - 1):WG:b
            buf[lc + q] = buf[src + q]
        end
    end
    return nothing
end

"""
    subtree_cholesky_kernel!(backend, WG)(factor, stack, info, nzval, amap, amap_ptr, amap_src, trees, bm,
                                          subtree_ptr, subtree_nodes, front_ptr, front_nrows, front_ncols, cb_ptr,
                                          local_front, local_cb, child_ptr, child_list, relind_ptr, relind,
                                          Val(NE), Val(WG); ndrange = WG * count * bm.nact)

Fused regime-A kernel: workgroup `G` takes subtree `t = trees[bm.first + g - 1]`
of batch member `k` (`g`, `k` from the [`BatchMap`](@ref) `bm`)
and processes its supernodes `subtree_nodes[subtree_ptr[t]:(subtree_ptr[t+1]-1)]`
in order with the serial stack of packed fronts and contribution blocks in a
`@localmem` buffer of `NE` entries (offsets `local_front`, `local_cb`): zero,
scatter A, extend-add the children, factor the front's first `w` columns
(Cholesky of F11, L21, Schur complement), write the panel and `info[v]`, then
move the contribution block down, or, for the subtree root (`cb_ptr > 0`),
write it to the update stack (packed lower triangle).
"""
@kernel function subtree_cholesky_kernel!(factor, stack, info, nzval, amap, amap_ptr, amap_src, trees, bm,
                                          subtree_ptr, subtree_nodes, front_ptr, front_nrows, front_ncols, cb_ptr,
                                          local_front, local_cb, child_ptr, child_list, relind_ptr, relind,
                                          ::Val{NE}, ::Val{WG}) where {NE, WG}
    @uniform TT = eltype(factor)
    @uniform RT = real(eltype(factor))
    @uniform IT = eltype(subtree_nodes)
    li = @index(Local, Linear)
    G = @index(Group, Linear)
    buf = @localmem TT (NE,)
    ctl = @localmem IT (_ST_CTL,)
    piv = @localmem RT (1,)
    if li == 1
        @inbounds t = trees[bm.first + _bm_node(bm, G) - 1]
        @inbounds ctl[_ST_FIRST] = subtree_ptr[t]
        @inbounds ctl[_ST_COUNT] = subtree_ptr[t + 1] - subtree_ptr[t]
    end
    @synchronize
    for k in 1:ctl[_ST_COUNT]
        if li == 1
            _subtree_setup!(ctl, k, subtree_nodes, front_nrows, front_ncols, local_front, local_cb, child_ptr)
        end
        @synchronize
        _subtree_zero!(buf, ctl, li, Val(WG))
        @synchronize
        _subtree_scatter!(buf, ctl, _mview(nzval, _bm_gmember(bm, G), bm.nbatch), amap, amap_ptr, amap_src, front_ptr,
                          li, Val(WG))
        @synchronize
        for kc in 1:ctl[_ST_NCHILD]
            _subtree_extend_add!(buf, ctl, kc, child_ptr, child_list, front_nrows, front_ncols, local_cb, relind_ptr,
                                 relind, li, Val(WG))
            @synchronize
        end
        for j in 1:ctl[_ST_W]
            _subtree_chol_update!(buf, ctl, piv, j, li, Val(WG))
            @synchronize
            _subtree_chol_scale!(buf, ctl, piv, j, li, Val(WG))
            @synchronize
        end
        mb = _bm_gmember(bm, G)
        _subtree_write!(factor, _mview(stack, mb, bm.nbatch), _iview(info, mb, bm.nbatch), buf, ctl,
                        member_panels(front_ptr, mb, bm.nbatch), cb_ptr, li, Val(WG))
        @synchronize
        for r in 1:ctl[_ST_ROUNDS]
            _subtree_move!(buf, ctl, r, li, Val(WG))
            @synchronize
        end
    end
end

function _launch_subtrees!(N::Numeric{T}, S::Symbolic, nzval, first, count, ::Val{LB}, ::Val{WG}) where {T, LB, WG}
    _ilog2(WG)
    bm = batch_map(N; first)
    kernel! = subtree_cholesky_kernel!(KernelAbstractions.get_backend(N.factor), WG)
    kernel!(N.factor, N.stack, N.info, nzval, S.amap, S.amap_ptr, S.amap_src, S.group_nodes, bm,
            S.subtree_ptr, S.subtree_nodes, S.front_ptr, S.front_nrows, S.front_ncols, S.cb_ptr, S.local_front,
            S.local_cb, S.child_ptr, S.child_list, S.relind_ptr, S.relind,
            Val((LB - SUBTREE_LOCAL_RESERVE) ÷ sizeof(T)), Val(WG); ndrange = WG * count * bm.nact)
    return nothing
end

# the local-memory size class as a compile-time constant (explicit branches: no dynamic dispatch, no allocation;
# every branch is compiled with the first launch, hence the short list of sizes; branch 5 is the error)
@inline function _with_local_bytes(fn, nbytes::Int)
    length(SUBTREE_LOCAL_SIZES) == 4 || error("update _with_local_bytes")
    Base.Cartesian.@nif 5 d -> (nbytes == SUBTREE_LOCAL_SIZES[d]) d -> fn(Val(SUBTREE_LOCAL_SIZES[d])) d -> throw(
        InvalidValueError("no regime-A kernel with $nbytes bytes of local memory; sizes: $SUBTREE_LOCAL_SIZES"))
end

"""
    factorize_subtrees!(numeric, symbolic, nzval, first, count, local_bytes) -> numeric

Regime-A launch: assemble and factor the `count` subtrees
`symbolic.group_nodes[first:(first + count - 1)]` (subtree ids of one budget
class) of every active batch member with one [`subtree_cholesky_kernel!`](@ref)
launch, one workgroup per (subtree, member), `local_bytes` ([`subtree_local_bytes`](@ref) of the class) of local
memory per workgroup. Panels and statuses of every supernode of the subtrees
are written, and each subtree root's contribution block goes to the update
stack. Asynchronous.
"""
function factorize_subtrees!(N::Numeric, S::Symbolic, nzval::AbstractVector, first::Integer, count::Integer,
                             local_bytes::Integer)
    count > 0 || return N
    _with_local_bytes(Int(local_bytes)) do lb
        _launch_subtrees!(N, S, nzval, first, count, lb, Val(SUBTREE_WORKGROUP))
    end
    return N
end
