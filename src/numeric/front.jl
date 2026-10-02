# Regime B (PLAN §2.2, §2.4): one fused kernel per (step, bin) launch group, one
# 1-D workgroup per front. The workgroup assembles its front (zero, scatter A,
# owner-pull extend-add of the children, as the assembly kernels), loads the
# lower triangle of F11 into `@localmem` (packed, `W(W+1)/2` entries for the
# width class `W`), factors it unblocked (right-looking, two barriers per
# column), writes L11 back, solves F21 ← F21 L11⁻ᴴ one panel row per work item
# (row tiles of `WG` rows streamed from the panel, L11 read from local memory),
# and updates its own contribution block F22 ← F22 − F21 F21ᴴ (lower triangle)
# on the update stack. Every front writes only its own panel and contribution
# block: no atomics, deterministic. The status of the front (`potrf` info: the
# first non-positive pivot, local column) goes to `info[s]`.

"Width classes with a fused regime-B kernel (`@localmem` holds the packed `W×W` lower triangle)."
const REGIME_B_WIDTHS = (8, 16, 32, 64)

"Largest front width with a fused regime-B kernel; wider bins (raised `regime_c_width`) take the regime-C path."
const REGIME_B_MAX_WIDTH = 64

"Workgroup size of the fused regime-B kernel."
const FRONT_WORKGROUP = 128

# position of (i, j), i ≥ j, in the packed column-major lower triangle of a W×W matrix
@inline _packed(i, j, W) = (j - 1) * (2 * W - j + 2) ÷ 2 + i - j + 1

# load the lower triangle of F11 of front `s` into local memory, reset the status
@inline function _front_load!(L11, st, factor, s, li, front_ptr, front_nrows, front_ncols, ::Val{W},
                              ::Val{WG}) where {W, WG}
    @inbounds begin
        f = front_nrows[s]
        w = front_ncols[s]
        p0 = front_ptr[s]
        for q in (li - 1):WG:(w * w - 1)
            j = q ÷ w + 1
            i = q - (j - 1) * w + 1
            i >= j && (L11[_packed(i, j, W)] = factor[p0 + (j - 1) * f + i - 1])
        end
        li == 1 && (st[1] = Int32(0))
    end
    return nothing
end

# column j of the unblocked Cholesky, first half: check the pivot and apply the
# rank-1 update of the trailing block with the scaled column (column j itself is
# not written, so every work item reads the same unscaled values)
@inline function _front_chol_update!(L11, st, piv, j, s, li, front_ncols, ::Val{W}, ::Val{WG}) where {W, WG}
    @inbounds begin
        w = front_ncols[s]
        if j <= w && st[1] == 0
            ajj = real(L11[_packed(j, j, W)])
            if ajj > 0
                d = sqrt(ajj)
                li == 1 && (piv[1] = d)
                r = w - j
                for q in (li - 1):WG:(r * r - 1)
                    k = j + 1 + q ÷ r
                    i = j + 1 + q % r
                    if i >= k
                        lij = L11[_packed(i, j, W)] / d
                        lkj = L11[_packed(k, j, W)] / d
                        L11[_packed(i, k, W)] -= lij * conj(lkj)
                    end
                end
            elseif li == 1
                st[1] = Int32(j)                     # not positive (or NaN): stop, as LAPACK potrf
            end
        end
    end
    return nothing
end

# column j, second half: scale the column by the pivot stored by the first half
@inline function _front_chol_scale!(L11, st, piv, j, s, li, front_ncols, ::Val{W}, ::Val{WG}) where {W, WG}
    @inbounds begin
        w = front_ncols[s]
        if j <= w && st[1] == 0
            d = piv[1]
            for i in (j + li):WG:w
                L11[_packed(i, j, W)] /= d
            end
            li == 1 && (L11[_packed(j, j, W)] = d)
        end
    end
    return nothing
end

# write L11 back to the panel, solve the rows of F21 (one row per work item), store the status
@inline function _front_trsm!(factor, info, L11, st, s, li, front_ptr, front_nrows, front_ncols, ::Val{W},
                              ::Val{WG}) where {W, WG}
    @inbounds begin
        f = front_nrows[s]
        w = front_ncols[s]
        p0 = front_ptr[s]
        for q in (li - 1):WG:(w * w - 1)
            j = q ÷ w + 1
            i = q - (j - 1) * w + 1
            i >= j && (factor[p0 + (j - 1) * f + i - 1] = L11[_packed(i, j, W)])
        end
        if st[1] == 0
            for i in (w + li):WG:f
                for k in 1:w
                    x = factor[p0 + (k - 1) * f + i - 1]
                    for j in 1:(k - 1)
                        x -= factor[p0 + (j - 1) * f + i - 1] * conj(L11[_packed(k, j, W)])
                    end
                    factor[p0 + (k - 1) * f + i - 1] = x / real(L11[_packed(k, k, W)])
                end
            end
        end
        li == 1 && (info[s] = st[1])
    end
    return nothing
end

# F22 ← F22 − F21 F21ᴴ on the lower triangle of the contribution block (real diagonal, as herk)
@inline function _front_syrk!(factor, stack, st, s, li, front_ptr, front_nrows, front_ncols, cb_ptr,
                              ::Val{WG}) where {WG}
    @inbounds begin
        T = eltype(factor)
        c0 = cb_ptr[s]
        if st[1] == 0 && c0 > 0
            f = front_nrows[s]
            w = front_ncols[s]
            m = f - w
            p0 = front_ptr[s] + w - 1                # F21[i, k] = factor[p0 + (k - 1) * f + i]
            for q in (li - 1):WG:(m * m - 1)
                jj = q ÷ m + 1
                ii = q - (jj - 1) * m + 1
                if ii >= jj
                    acc = zero(T)
                    for k in 1:w
                        acc += factor[p0 + (k - 1) * f + ii] * conj(factor[p0 + (k - 1) * f + jj])
                    end
                    d = c0 + q
                    stack[d] = ii == jj ? T(real(stack[d]) - real(acc)) : stack[d] - acc
                end
            end
        end
    end
    return nothing
end

"""
    front_cholesky_kernel!(backend, WG)(factor, stack, info, nzval, amap, amap_ptr, amap_src, nodes, first,
                                        front_ptr, front_nrows, front_ncols, cb_ptr, child_ptr, child_list,
                                        relind_ptr, relind, maxchild, Val(ASM), Val(W), Val(NL), Val(WG);
                                        ndrange = WG * count)

Fused regime-B kernel: workgroup `g` takes front `s = nodes[first + g - 1]`
(width `w ≤ W`), and, when `ASM`, zeroes and assembles it (A through the
`amap`, then the `maxchild` children's contribution blocks in `child_list`
order), then factors `F11 = L11 L11ᴴ` in `@localmem` (`NL = W(W+1)/2`
entries), solves `F21 ← F21 L11⁻ᴴ`, updates its contribution block on the
update stack (`cb_ptr[s] > 0`) and sets `info[s]` (0, or the local column of
the first non-positive pivot; the front is then left partially factored).
Without `ASM` the panels and contribution blocks must already be assembled.
"""
@kernel function front_cholesky_kernel!(factor, stack, info, nzval, amap, amap_ptr, amap_src, nodes, first,
                                        front_ptr, front_nrows, front_ncols, cb_ptr, child_ptr, child_list,
                                        relind_ptr, relind, maxchild, ::Val{ASM}, ::Val{W}, ::Val{NL},
                                        ::Val{WG}) where {ASM, W, NL, WG}
    @uniform TT = eltype(factor)
    @uniform RT = real(eltype(factor))
    li = @index(Local, Linear)
    g = @index(Group, Linear)
    L11 = @localmem TT (NL,)
    st = @localmem Int32 (1,)
    piv = @localmem RT (1,)
    if ASM
        @inbounds s = nodes[first + g - 1]
        _zero_front!(factor, stack, s, li, front_ptr, front_nrows, front_ncols, cb_ptr, Val(WG))
    end
    @synchronize
    if ASM
        @inbounds s = nodes[first + g - 1]
        _scatter_front!(factor, nzval, amap, amap_ptr, amap_src, s, li, Val(WG))
    end
    @synchronize
    for k in 1:maxchild
        @inbounds s = nodes[first + g - 1]
        _extend_add_child!(factor, stack, s, k, li, front_ptr, front_nrows, front_ncols, cb_ptr, child_ptr,
                           child_list, relind_ptr, relind, Val(WG))
        @synchronize
    end
    @inbounds s = nodes[first + g - 1]
    _front_load!(L11, st, factor, s, li, front_ptr, front_nrows, front_ncols, Val(W), Val(WG))
    @synchronize
    for j in 1:W
        @inbounds s = nodes[first + g - 1]
        _front_chol_update!(L11, st, piv, j, s, li, front_ncols, Val(W), Val(WG))
        @synchronize
        @inbounds s = nodes[first + g - 1]
        _front_chol_scale!(L11, st, piv, j, s, li, front_ncols, Val(W), Val(WG))
        @synchronize
    end
    @inbounds s = nodes[first + g - 1]
    _front_trsm!(factor, info, L11, st, s, li, front_ptr, front_nrows, front_ncols, Val(W), Val(WG))
    @synchronize
    @inbounds s = nodes[first + g - 1]
    _front_syrk!(factor, stack, st, s, li, front_ptr, front_nrows, front_ncols, cb_ptr, Val(WG))
end

function _launch_front_cholesky!(factor, stack, info, nzval, amap, amap_ptr, amap_src, nodes, first, count,
                                 front_ptr, front_nrows, front_ncols, cb_ptr, child_ptr, child_list, relind_ptr,
                                 relind, maxchild, asm::Val, ::Val{W}, ::Val{WG}) where {W, WG}
    _ilog2(WG)
    kernel! = front_cholesky_kernel!(KernelAbstractions.get_backend(factor), WG)
    kernel!(factor, stack, info, nzval, amap, amap_ptr, amap_src, nodes, Int(first), front_ptr, front_nrows,
            front_ncols, cb_ptr, child_ptr, child_list, relind_ptr, relind, Int(maxchild), asm, Val(W),
            Val(W * (W + 1) ÷ 2), Val(WG); ndrange = WG * count)
    return nothing
end

# the width class W as a compile-time constant (explicit branches: no dynamic dispatch, no allocation)
@inline function _with_width_class(fn, W::Int)
    W == 8 && return fn(Val(8))
    W == 16 && return fn(Val(16))
    W == 32 && return fn(Val(32))
    W == 64 && return fn(Val(64))
    throw(InvalidValueError("no fused regime-B kernel for width class $W; expected one of $REGIME_B_WIDTHS"))
end

"""
    factorize_fronts_b!(numeric, symbolic, nzval, first, count, maxchild, W) -> numeric

Regime-B launch: assemble and factor the `count` fronts
`symbolic.group_nodes[first:(first + count - 1)]` (widths `≤ W`,
`W ∈ $(REGIME_B_WIDTHS)`, at most `maxchild` children each) with one
[`front_cholesky_kernel!`](@ref) launch, one workgroup per front; `info[s]`
receives each front's status. Asynchronous.
"""
function factorize_fronts_b!(N::Numeric, S::Symbolic, nzval::AbstractVector, first::Integer, count::Integer,
                             maxchild::Integer, W::Integer)
    count > 0 || return N
    _with_width_class(Int(W)) do w
        _launch_front_cholesky!(N.factor, N.stack, N.info, nzval, S.amap, S.amap_ptr, S.amap_src, S.group_nodes,
                                first, count, S.front_ptr, S.front_nrows, S.front_ncols, S.cb_ptr, S.child_ptr,
                                S.child_list, S.relind_ptr, S.relind, maxchild, Val(true), w,
                                Val(FRONT_WORKGROUP))
    end
    return N
end

"""
    front_cholesky!(factor, stack, info, nodes, first, count, front_ptr, front_nrows, front_ncols, cb_ptr;
                    width = 64, workgroup = Val(FRONT_WORKGROUP)) -> info

Dense part of the regime-B kernel on already assembled fronts (benchmarks and
tests): for each front `s = nodes[first + k - 1]`, `k = 1:count`, with panel
`factor[front_ptr[s]:(front_ptr[s + 1] - 1)]` (`f×w`, `f = front_nrows[s]`,
`w = front_ncols[s] ≤ width`, `width ∈ $(REGIME_B_WIDTHS)`) and contribution
block `stack[cb_ptr[s]:(cb_ptr[s] + m^2 - 1)]` (`m = f - w`, skipped when
`cb_ptr[s] == 0`): `F11 = L11 L11ᴴ`, `F21 ← F21 L11⁻ᴴ`,
`F22 ← F22 − F21 F21ᴴ` (lower triangle), `info[s]` = status. Asynchronous.
"""
function front_cholesky!(factor::AbstractVector, stack::AbstractVector, info::AbstractVector{Int32},
                         nodes::AbstractVector, first::Integer, count::Integer, front_ptr::AbstractVector,
                         front_nrows::AbstractVector, front_ncols::AbstractVector, cb_ptr::AbstractVector;
                         width::Integer = REGIME_B_MAX_WIDTH, workgroup::Val = Val(FRONT_WORKGROUP))
    count > 0 || return info
    _with_width_class(Int(width)) do w
        _launch_front_cholesky!(factor, stack, info, factor, nodes, nodes, nodes, nodes, first, count, front_ptr,
                                front_nrows, front_ncols, cb_ptr, nodes, nodes, nodes, nodes, 0, Val(false), w,
                                workgroup)
    end
    return info
end
