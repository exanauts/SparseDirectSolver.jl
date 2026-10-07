# Assembly kernels of the numeric phase (PLAN §2.4, §3.4): zero the fronts of a
# step, scatter the values of A into their panels, and extend-add the children's
# contribution blocks. Every kernel runs one 1-D workgroup per front of the step
# (`nodes[first:(first + count - 1)]`, the step's range of `group_nodes`) and
# writes only into that front (owner-pull): no atomics, and every destination
# receives its contributions in a fixed order, so the result is bitwise
# reproducible. Contribution blocks on the update stack are packed lower
# triangles (`_packed`, `m(m+1)/2` entries). On the KA CPU backend, values
# derived from the group index are recomputed in every segment between barriers
# (see `src/dense/fallback/common.jl`). Uniform batch: one workgroup per (front,
# active member), the kernels see the member's panels (`MemberPanels` in place
# of `front_ptr`), update stack and values (`src/numeric/batch.jl`).

"Workgroup size of the assembly kernels."
const ASSEMBLY_WORKGROUP = 256

# zero the panel and the update-stack contribution block of front `s` (work item `li` of `WG`)
@inline function _zero_front!(factor, stack, s, li, front_ptr, front_nrows, front_ncols, cb_ptr, ::Val{WG}) where {WG}
    @inbounds begin
        z = zero(eltype(factor))
        p0 = front_ptr[s]
        for q in (p0 + li - 1):WG:(p0 + front_nrows[s] * front_ncols[s] - 1)
            factor[q] = z
        end
        c = cb_ptr[s]
        if c > 0
            m = Int(front_nrows[s]) - Int(front_ncols[s])  # packed sizes in Int: m(m+1)/2 overflows Int32 first
            for q in (c + li - 1):WG:(c + m * (m + 1) ÷ 2 - 1)
                stack[q] = z
            end
        end
    end
    return nothing
end

@kernel function _zero_fronts_kernel!(factor, stack, nodes, bm, front_ptr, front_nrows, front_ncols, cb_ptr,
                                      ::Val{WG}) where {WG}
    li = @index(Local, Linear)
    G = @index(Group, Linear)
    @inbounds s = nodes[bm.first + _bm_node(bm, G) - 1]
    k = _bm_gmember(bm, G)
    _zero_front!(factor, _mview(stack, k, bm.nbatch), s, li, member_panels(front_ptr, k, bm.nbatch), front_nrows,
                 front_ncols, cb_ptr, Val(WG))
end

"""
    zero_fronts!(numeric, symbolic, first, count) -> numeric

Zero the panels and the update-stack contribution blocks of the `count` fronts
`symbolic.group_nodes[first:(first + count - 1)]` (one workgroup per front).
Asynchronous.
"""
function zero_fronts!(N::Numeric, S::Symbolic, first::Integer, count::Integer)
    count > 0 || return N
    WG = ASSEMBLY_WORKGROUP
    bm = batch_map(N; first)
    kernel! = _zero_fronts_kernel!(KernelAbstractions.get_backend(N.factor), WG)
    kernel!(N.factor, N.stack, S.group_nodes, bm, S.front_ptr, S.front_nrows, S.front_ncols, S.cb_ptr,
            Val(WG); ndrange = WG * count * bm.nact)
    return N
end

# value of nzval entry `x` as added at amap offset `off` (negative: conjugate)
@inline _amap_value(::Type{T}, x, off) where {T} = off < 0 ? T(conj(x)) : T(x)

# assemble the values of A into the zeroed panel of front `s` (work item `li` of `WG`); `shift` moves the
# `amap` offsets (single-matrix layout) to the batch member's panel
@inline function _scatter_front!(factor, nzval, amap, amap_ptr, amap_src, s, shift, li, ::Val{WG}) where {WG}
    @inbounds begin
        T = eltype(factor)
        a = amap_ptr[s]
        b = amap_ptr[s + 1] - 1
        # the entries of a front are sorted by destination: the work item holding
        # the first entry of a run of duplicates sums the run in nzval order
        for k in (a + li - 1):WG:b
            dest = abs(amap[amap_src[k]])
            if k == a || abs(amap[amap_src[k - 1]]) != dest
                v = zero(T)
                kk = k
                while kk <= b
                    p = amap_src[kk]
                    off = amap[p]
                    abs(off) == dest || break
                    v += _amap_value(T, nzval[p], off)
                    kk += 1
                end
                factor[dest + shift] = v
            end
        end
    end
    return nothing
end

@kernel function _scatter_A_kernel!(factor, nzval, amap, amap_ptr, amap_src, nodes, bm, front_ptr,
                                    ::Val{WG}) where {WG}
    li = @index(Local, Linear)
    G = @index(Group, Linear)
    @inbounds s = nodes[bm.first + _bm_node(bm, G) - 1]
    k = _bm_gmember(bm, G)
    _scatter_front!(factor, _mview(nzval, k, bm.nbatch), amap, amap_ptr, amap_src, s,
                    _member_shift(front_ptr, s, k, bm.nbatch), li, Val(WG))
end

# shift of the factor offsets of front `s` from the single-matrix layout to batch member `k`
@inline _member_shift(front_ptr, s, k, nb) = @inbounds Int(member_panels(front_ptr, k, nb)[s]) - Int(front_ptr[s])

"""
    scatter_A!(numeric, symbolic, nzval, first, count) -> numeric

Assemble the stored values `nzval` (on the backend of `numeric`) into the
zeroed panels of the fronts `symbolic.group_nodes[first:(first + count - 1)]`
through `amap_ptr`/`amap_src`/`amap` (one workgroup per front, one work item
per nonzero; duplicated entries are summed by one work item in `nzval` order,
negative offsets add the conjugate). Asynchronous.
"""
function scatter_A!(N::Numeric, S::Symbolic, nzval::AbstractVector, first::Integer, count::Integer)
    count > 0 || return N
    WG = ASSEMBLY_WORKGROUP
    bm = batch_map(N; first)
    kernel! = _scatter_A_kernel!(KernelAbstractions.get_backend(N.factor), WG)
    kernel!(N.factor, nzval, S.amap, S.amap_ptr, S.amap_src, S.group_nodes, bm, S.front_ptr, Val(WG);
            ndrange = WG * count * bm.nact)
    return N
end

# add the packed contribution block of the `k`-th child of front `s` (if any) to
# the panel and the packed contribution block of `s` (work item `li` of `WG`)
@inline function _extend_add_child!(factor, stack, s, k, li, front_ptr, front_nrows, front_ncols, cb_ptr,
                                    child_ptr, child_list, relind_ptr, relind, ::Val{WG}) where {WG}
    @inbounds begin
        kc = child_ptr[s] + k - 1
        if kc < child_ptr[s + 1]
            c = child_list[kc]
            cb = cb_ptr[c]
            if cb > 0
                mc = Int(front_nrows[c]) - Int(front_ncols[c])   # packed offsets in Int (`_packed` of a block that
                fp = Int(front_nrows[s])                         # fits INT can overflow INT for m > 46340)
                wp = Int(front_ncols[s])
                mp = fp - wp
                pp = front_ptr[s]
                cp = cb_ptr[s]
                r0 = relind_ptr[c] - 1
                for q in (li - 1):WG:(mc * mc - 1)
                    jj = q ÷ mc + 1
                    ii = q - (jj - 1) * mc + 1
                    if ii >= jj                         # lower triangle of the block
                        ri = relind[r0 + ii]
                        rj = relind[r0 + jj]
                        v = stack[cb + _packed(ii, jj, mc) - 1]
                        if rj <= wp
                            d = pp + (rj - 1) * fp + ri - 1
                            factor[d] += v
                        else
                            d = cp + _packed(ri - wp, rj - wp, mp) - 1
                            stack[d] += v
                        end
                    end
                end
            end
        end
    end
    return nothing
end

@kernel function _extend_add_kernel!(factor, stack, nodes, bm, front_ptr, front_nrows, front_ncols, cb_ptr,
                                     child_ptr, child_list, relind_ptr, relind, maxchild, ::Val{WG}) where {WG}
    li = @index(Local, Linear)
    G = @index(Group, Linear)
    # children one after the other (fixed order, a barrier between them): rows
    # of one child go to distinct destinations, rows of two children may not
    for k in 1:maxchild
        @inbounds s = nodes[bm.first + _bm_node(bm, G) - 1]
        mb = _bm_gmember(bm, G)
        _extend_add_child!(factor, _mview(stack, mb, bm.nbatch), s, k, li, member_panels(front_ptr, mb, bm.nbatch),
                           front_nrows, front_ncols, cb_ptr, child_ptr, child_list, relind_ptr, relind, Val(WG))
        @synchronize
    end
end

"""
    extend_add!(numeric, symbolic, first, count, maxchild) -> numeric

Owner-pull extend-add: the workgroup of each front `s` of
`symbolic.group_nodes[first:(first + count - 1)]` adds its children's packed
contribution blocks (update stack, `cb_ptr`) to its panel and its own packed
contribution block through `relind`, children in `child_list` order
(`maxchild` ≥ the largest child count of these fronts). No atomics;
deterministic. Asynchronous.
"""
function extend_add!(N::Numeric, S::Symbolic, first::Integer, count::Integer, maxchild::Integer)
    (count > 0 && maxchild > 0) || return N
    WG = ASSEMBLY_WORKGROUP
    bm = batch_map(N; first)
    kernel! = _extend_add_kernel!(KernelAbstractions.get_backend(N.factor), WG)
    kernel!(N.factor, N.stack, S.group_nodes, bm, S.front_ptr, S.front_nrows, S.front_ncols, S.cb_ptr,
            S.child_ptr, S.child_list, S.relind_ptr, S.relind, Int(maxchild), Val(WG); ndrange = WG * count * bm.nact)
    return N
end

@kernel function _pack_add_kernel!(stack, c0, work, m)
    q = @index(Global, Linear)
    @inbounds if q <= m * m
        j = (q - 1) ÷ m + 1
        i = q - (j - 1) * m
        if i >= j
            stack[c0 + _packed(i, j, m) - 1] += work[q]
        end
    end
end

# work item q of block b: the `m×m` block b of `work` (member `members[j0 + b + 1]`) into its member's stack
@kernel function _pack_add_batch_kernel!(stack, c0, work, m, members, j0, nb)
    q0 = @index(Global, Linear)
    b = (q0 - 1) ÷ (m * m)
    q = q0 - b * m * m
    @inbounds begin
        j = (q - 1) ÷ m + 1
        i = q - (j - 1) * m
        if i >= j
            k = Int(members[j0 + b + 1])
            v = work[b * m * m + q]
            i == j && (v = eltype(stack)(real(v)))       # gemm with F21ᴴ: keep the herk real diagonal
            _mview(stack, k, nb)[c0 + _packed(i, j, m) - 1] += v
        end
    end
end

"""
    pack_add!(stack, c0, work, m) -> stack
    pack_add!(stack, c0, work, m, members, j0, count, nbatch) -> stack

Add the lower triangle of the column-major `m×m` matrix `work[1:m^2]` to the
packed contribution block `stack[c0:(c0 + m(m+1)/2 - 1)]` (the regime-C step
after the vendor `syrk`/`herk` into the workspace). Batched form: the `count`
blocks `work[(b-1)m² + 1 : b m²]` go to the stacks of batch members
`members[j0 + b]` (device vector) of an `nbatch`-member update stack. One
launch, one work item per entry. Asynchronous.
"""
function pack_add!(stack::AbstractVector, c0::Integer, work::AbstractVector, m::Integer)
    m > 0 || return stack
    kernel! = _pack_add_kernel!(KernelAbstractions.get_backend(stack), ASSEMBLY_WORKGROUP)
    kernel!(stack, Int(c0), work, Int(m); ndrange = m * m)
    return stack
end

function pack_add!(stack::AbstractVector, c0::Integer, work::AbstractVector, m::Integer, members::AbstractVector{Int32},
                   j0::Integer, count::Integer, nbatch::Integer)
    m > 0 && count > 0 || return stack
    kernel! = _pack_add_batch_kernel!(KernelAbstractions.get_backend(stack), ASSEMBLY_WORKGROUP)
    kernel!(stack, Int(c0), work, Int(m), members, Int(j0), Int(nbatch); ndrange = m * m * count)
    return stack
end
