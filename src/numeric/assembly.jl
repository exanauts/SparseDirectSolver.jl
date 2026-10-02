# Assembly kernels of the numeric phase (PLAN §2.4, §3.4): zero the fronts of a
# step, scatter the values of A into their panels, and extend-add the children's
# contribution blocks. Every kernel runs one 1-D workgroup per front of the step
# (`nodes[first:(first + count - 1)]`, the step's range of `group_nodes`) and
# writes only into that front (owner-pull): no atomics, and every destination
# receives its contributions in a fixed order, so the result is bitwise
# reproducible. On the KA CPU backend, values derived from the group index are
# recomputed in every segment between barriers (see `src/dense/fallback/common.jl`).

"Workgroup size of the assembly kernels."
const ASSEMBLY_WORKGROUP = 256

# zero the panel and the update-stack contribution block of front `s` (work item `li` of `WG`)
@inline function _zero_front!(factor, stack, s, li, front_ptr, front_nrows, front_ncols, cb_ptr, ::Val{WG}) where {WG}
    @inbounds begin
        z = zero(eltype(factor))
        for q in (front_ptr[s] + li - 1):WG:(front_ptr[s + 1] - 1)
            factor[q] = z
        end
        c = cb_ptr[s]
        if c > 0
            m = front_nrows[s] - front_ncols[s]
            for q in (c + li - 1):WG:(c + m * m - 1)
                stack[q] = z
            end
        end
    end
    return nothing
end

@kernel function _zero_fronts_kernel!(factor, stack, nodes, first, front_ptr, front_nrows, front_ncols, cb_ptr,
                                      ::Val{WG}) where {WG}
    li = @index(Local, Linear)
    g = @index(Group, Linear)
    @inbounds s = nodes[first + g - 1]
    _zero_front!(factor, stack, s, li, front_ptr, front_nrows, front_ncols, cb_ptr, Val(WG))
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
    kernel! = _zero_fronts_kernel!(KernelAbstractions.get_backend(N.factor), WG)
    kernel!(N.factor, N.stack, S.group_nodes, Int(first), S.front_ptr, S.front_nrows, S.front_ncols, S.cb_ptr,
            Val(WG); ndrange = WG * count)
    return N
end

# value of nzval entry `x` as added at amap offset `off` (negative: conjugate)
@inline _amap_value(::Type{T}, x, off) where {T} = off < 0 ? T(conj(x)) : T(x)

# assemble the values of A into the zeroed panel of front `s` (work item `li` of `WG`)
@inline function _scatter_front!(factor, nzval, amap, amap_ptr, amap_src, s, li, ::Val{WG}) where {WG}
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
                factor[dest] = v
            end
        end
    end
    return nothing
end

@kernel function _scatter_A_kernel!(factor, nzval, amap, amap_ptr, amap_src, nodes, first, ::Val{WG}) where {WG}
    li = @index(Local, Linear)
    g = @index(Group, Linear)
    @inbounds s = nodes[first + g - 1]
    _scatter_front!(factor, nzval, amap, amap_ptr, amap_src, s, li, Val(WG))
end

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
    kernel! = _scatter_A_kernel!(KernelAbstractions.get_backend(N.factor), WG)
    kernel!(N.factor, nzval, S.amap, S.amap_ptr, S.amap_src, S.group_nodes, Int(first), Val(WG); ndrange = WG * count)
    return N
end

# add the lower triangle of the contribution block of the `k`-th child of front `s`
# (if any) to the panel and contribution block of `s` (work item `li` of `WG`)
@inline function _extend_add_child!(factor, stack, s, k, li, front_ptr, front_nrows, front_ncols, cb_ptr,
                                    child_ptr, child_list, relind_ptr, relind, ::Val{WG}) where {WG}
    @inbounds begin
        kc = child_ptr[s] + k - 1
        if kc < child_ptr[s + 1]
            c = child_list[kc]
            cb = cb_ptr[c]
            if cb > 0
                mc = front_nrows[c] - front_ncols[c]
                fp = front_nrows[s]
                wp = front_ncols[s]
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
                        v = stack[cb + q]
                        if rj <= wp
                            d = pp + (rj - 1) * fp + ri - 1
                            factor[d] += v
                        else
                            d = cp + (rj - wp - 1) * mp + ri - wp - 1
                            stack[d] += v
                        end
                    end
                end
            end
        end
    end
    return nothing
end

@kernel function _extend_add_kernel!(factor, stack, nodes, first, front_ptr, front_nrows, front_ncols, cb_ptr,
                                     child_ptr, child_list, relind_ptr, relind, maxchild, ::Val{WG}) where {WG}
    li = @index(Local, Linear)
    g = @index(Group, Linear)
    # children one after the other (fixed order, a barrier between them): rows
    # of one child go to distinct destinations, rows of two children may not
    for k in 1:maxchild
        @inbounds s = nodes[first + g - 1]
        _extend_add_child!(factor, stack, s, k, li, front_ptr, front_nrows, front_ncols, cb_ptr, child_ptr,
                           child_list, relind_ptr, relind, Val(WG))
        @synchronize
    end
end

"""
    extend_add!(numeric, symbolic, first, count, maxchild) -> numeric

Owner-pull extend-add: the workgroup of each front `s` of
`symbolic.group_nodes[first:(first + count - 1)]` adds the lower triangles of
its children's contribution blocks (update stack, `cb_ptr`) to its panel and
its own contribution block through `relind`, children in `child_list` order
(`maxchild` ≥ the largest child count of these fronts). No atomics;
deterministic. Asynchronous.
"""
function extend_add!(N::Numeric, S::Symbolic, first::Integer, count::Integer, maxchild::Integer)
    (count > 0 && maxchild > 0) || return N
    WG = ASSEMBLY_WORKGROUP
    kernel! = _extend_add_kernel!(KernelAbstractions.get_backend(N.factor), WG)
    kernel!(N.factor, N.stack, S.group_nodes, Int(first), S.front_ptr, S.front_nrows, S.front_ncols, S.cb_ptr,
            S.child_ptr, S.child_list, S.relind_ptr, S.relind, Int(maxchild), Val(WG); ndrange = WG * count)
    return N
end
