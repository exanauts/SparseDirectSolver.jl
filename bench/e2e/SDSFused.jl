# Prototype kernels of PR #107 assembled as a backend-portable module, so the
# MadNLP wrapper can run the fused factorization and the algo1/fused solve on
# CUDA and AMDGPU alike. Single-solver-instance (module-global plan state).
module SDSFused

using LinearAlgebra, SparseArrays, Statistics, Printf
using SparseDirectSolver
const SDS = SparseDirectSolver
const KA = SDS.KernelAbstractions
const Atomix = SDS.Atomix
using .KA: @kernel, @index, @localmem, @synchronize, @Const, @uniform

const BACKEND = Ref{Any}(nothing)
DVEC(v) = (out = KA.allocate(BACKEND[], eltype(v), size(v)); copyto!(out, v); out)
DVEC(::Type{T}, v) where {T} = (out = KA.allocate(BACKEND[], T, length(v)); copyto!(out, T.(v)); out)



const WG = 32
const MW = 256  # max front width on the inverse path
const WGF = 128 # default workgroup size; the fused kernel takes it per plan (fp.wgf)
const CHWG = 1024 # workgroup size of the chain kernels
const NSTRIP = 4  # strip parallelism of the chain kernels

# ---------------------------------------------------------------------------
# inversion: thread j of the front's workgroup computes column j of inv(L11)
# (non-unit lower triangular), written col-major at inv_ptr[v]
@kernel function invert_l11_kernel!(inv11, @Const(factor), @Const(inv_ptr), @Const(list),
                                    @Const(front_ptr), @Const(front_nrows), @Const(front_ncols), ::Val{WGV}) where {WGV}
    li = @index(Local, Linear)
    g = @index(Group, Linear)
    @inbounds begin
        v = list[g]
        f = Int(front_nrows[v])
        w = Int(front_ncols[v])
        p0 = Int(front_ptr[v]) - 1
        ip = Int(inv_ptr[v]) - 1
        for j in li:WGV:w
            cb = ip + (j - 1) * w
            for k in 1:(j - 1)
                inv11[cb + k] = zero(eltype(inv11))
            end
            for k in j:w
                acc = k == j ? one(eltype(inv11)) : zero(eltype(inv11))
                for m in j:(k - 1)
                    acc -= factor[p0 + (m - 1) * f + k] * inv11[cb + m]
                end
                inv11[cb + k] = acc / factor[p0 + (k - 1) * f + k]
            end
        end
    end
end

# ---------------------------------------------------------------------------
# forward: z = inv(L11) y per front, then the atomic L21 update (stock scatter)
@kernel function fwd1_kernel!(Y, @Const(factor), @Const(inv11), @Const(inv_ptr), @Const(list), first, count,
                              @Const(subtree_ptr), @Const(subtree_nodes), @Const(super_ptr), @Const(rowptr),
                              @Const(rowval), @Const(front_ptr), @Const(front_nrows), @Const(front_ncols),
                              ::Val{SUB}, ::Val{WGV}) where {SUB, WGV}
    li = @index(Local, Linear)
    g = @index(Group, Linear)
    t = @localmem Float64 (MW,)
    @inbounds begin
        e = list[first + g - 1]
        a0 = SUB ? Int(subtree_ptr[e]) : 0
        cnt = SUB ? Int(subtree_ptr[e + 1]) - a0 : 1
        limit = SUB ? Int(super_ptr[subtree_nodes[a0 + cnt - 1] + 1]) - 1 : 0
        for k in 1:cnt
            v = SUB ? subtree_nodes[a0 + k - 1] : e
            f = Int(front_nrows[v])
            w = Int(front_ncols[v])
            p0 = Int(front_ptr[v]) - 1
            rp = Int(rowptr[v]) - 1
            c0 = Int(super_ptr[v]) - 1
            ip = Int(inv_ptr[v]) - 1
            w > MW && continue
            # t = inv(L11) * Y[c0+1 : c0+w]
            for kk in li:WGV:w
                acc = zero(Float64)
                for j in 1:w
                    acc += inv11[ip + (j - 1) * w + kk] * Y[c0 + j]
                end
                t[kk] = acc
            end
            @synchronize
            for kk in li:WGV:w
                Y[c0 + kk] = t[kk]
            end
            # L21 update from t (Y rows above `limit` belong to other workgroups: atomic)
            for i in (w + li):WGV:f
                acc = zero(Float64)
                for j in 1:w
                    acc += factor[p0 + (j - 1) * f + i] * t[j]
                end
                row = Int(rowval[rp + i])
                if row > limit
                    Atomix.@atomic Y[row] += -acc
                else
                    Y[row] -= acc
                end
            end
            @synchronize
        end
    end
end

# ---------------------------------------------------------------------------
# backward: gather from finished ancestors, then x = inv(L11)ᵀ z per front
@kernel function bwd1_kernel!(Y, @Const(factor), @Const(inv11), @Const(inv_ptr), @Const(list), first, count,
                              @Const(subtree_ptr), @Const(subtree_nodes), @Const(super_ptr), @Const(rowptr),
                              @Const(rowval), @Const(front_ptr), @Const(front_nrows), @Const(front_ncols),
                              ::Val{SUB}, ::Val{WGV}) where {SUB, WGV}
    li = @index(Local, Linear)
    g = @index(Group, Linear)
    t = @localmem Float64 (MW,)
    @inbounds begin
        e = list[first + g - 1]
        a0 = SUB ? Int(subtree_ptr[e]) : 0
        cnt = SUB ? Int(subtree_ptr[e + 1]) - a0 : 1
        for k in 1:cnt
            v = SUB ? subtree_nodes[a0 + cnt - k] : e      # reverse processing order
            f = Int(front_nrows[v])
            w = Int(front_ncols[v])
            p0 = Int(front_ptr[v]) - 1
            rp = Int(rowptr[v]) - 1
            c0 = Int(super_ptr[v]) - 1
            ip = Int(inv_ptr[v]) - 1
            w > MW && continue
            # t = Y[c0+k] - L21ᵀ x(rows below)
            for kk in li:WGV:w
                acc = zero(Float64)
                col = p0 + (kk - 1) * f
                for i in (w + 1):f
                    acc += factor[col + i] * Y[rowval[rp + i]]
                end
                t[kk] = Y[c0 + kk] - acc
            end
            @synchronize
            # x = inv(L11)ᵀ t  (inv col-major: invᵀ[k, j] = inv11[(k-1)w + j], contiguous in j)
            for kk in li:WGV:w
                acc = zero(Float64)
                cb = ip + (kk - 1) * w
                for j in 1:w
                    acc += inv11[cb + j] * t[j]
                end
                Y[c0 + kk] = acc
            end
            @synchronize
        end
    end
end

# ---------------------------------------------------------------------------
function build_inverse(s)
    S = s.symbolic
    sc = S.schedule
    reg = Array(sc.regime)
    width = sc.width
    ns = length(reg)
    inv_ptr_h = zeros(Int64, ns + 1)
    p = 1
    for v in 1:ns
        inv_ptr_h[v] = p
        if width[v] <= MW
            p += width[v]^2
        else
            @assert reg[v] == SDS.REGIME_C "front $v: width $(width[v]) > $MW but not regime C"
        end
    end
    inv_ptr_h[ns + 1] = p
    inv11 = KA.zeros(BACKEND[], Float64, p - 1)
    inv_ptr = DVEC(inv_ptr_h)
    ab = DVEC(Int32.(findall(v -> width[v] <= MW, 1:ns)))
    backend = KA.get_backend(inv11)
    invert! = () -> invert_l11_kernel!(backend, WG)(inv11, s.numeric.factor, inv_ptr, ab, S.front_ptr,
                                                   S.front_nrows, S.front_ncols, Val(WG);
                                                   ndrange = WG * length(ab))
    return inv11, inv_ptr, invert!, sizeof(Float64) * (p - 1)
end

function algo1_solve!(x, s, b, inv11, inv_ptr)
    S, N, ws = s.symbolic, s.numeric, s.workspace
    bm = SDS.batch_map(N; nrhs = 1)
    SDS.permute_rhs!(ws.Y, b, S.perm; transposed = false, bm)
    plan = ws.plan
    nodes = S.schedule.group_nodes
    widths = S.schedule.width
    p = SDS._solve_impls(N, S, :auto)
    backend = KA.get_backend(ws.Y)
    Yv = vec(ws.Y)
    args(first, count) = (Yv, N.factor, inv11, inv_ptr, S.group_nodes, first, count, S.subtree_ptr,
                          S.subtree_nodes, S.super_ptr, S.rowptr, S.rowval, S.front_ptr, S.front_nrows,
                          S.front_ncols)
    for k in eachindex(plan.kind)
        a, b_ = plan.first[k], plan.last[k]
        if plan.kind[k] == SDS.SOLVE_DENSE
            any(widths[nodes[q]] <= MW for q in a:b_) &&
                fwd1_kernel!(backend, WG)(args(a, b_ - a + 1)..., Val(false), Val(WG); ndrange = WG * (b_ - a + 1))
            for q in a:b_
                widths[nodes[q]] > MW && SDS._fwd_dense!(ws, S, N, nodes[q], 1, false, p, SDS._SV_CHOLESKY)
            end
        else
            sub = plan.kind[k] == SDS.SOLVE_SUBTREES
            fwd1_kernel!(backend, WG)(args(a, b_ - a + 1)..., Val(sub), Val(WG); ndrange = WG * (b_ - a + 1))
        end
    end
    for k in reverse(eachindex(plan.kind))
        a, b_ = plan.first[k], plan.last[k]
        if plan.kind[k] == SDS.SOLVE_DENSE
            for q in b_:-1:a
                widths[nodes[q]] > MW && SDS._bwd_dense!(ws, S, N, nodes[q], 1, p, SDS._SV_CHOLESKY)
            end
            any(widths[nodes[q]] <= MW for q in a:b_) &&
                bwd1_kernel!(backend, WG)(args(a, b_ - a + 1)..., Val(false), Val(WG); ndrange = WG * (b_ - a + 1))
        else
            sub = plan.kind[k] == SDS.SOLVE_SUBTREES
            bwd1_kernel!(backend, WG)(args(a, b_ - a + 1)..., Val(sub), Val(WG); ndrange = WG * (b_ - a + 1))
        end
    end
    SDS.unpermute_solution!(x, ws.Y, S.perm; transposed = false, bm)
    return x
end



# ---------------------------------------------------------------------------
# fused sweeps (algo2): one launch per direction over every front of width
# <= MW below the first wide front, etree dependency counters instead of
# level-batched launches. CUDA-only (threadfence, resident-grid spinning).
@kernel function fwd2_kernel!(Y, @Const(factor), @Const(inv11), @Const(inv_ptr),
                              @Const(flist), @Const(slist), nsub, @Const(subtree_ptr), @Const(subtree_nodes),
                              @Const(snparent), arrived, @Const(nchild), @Const(super_ptr),
                              @Const(rowptr), @Const(rowval), @Const(front_ptr), @Const(front_nrows),
                              @Const(front_ncols), ::Val{WGV}) where {WGV}
    li = @index(Local, Linear)
    g = @index(Group, Linear)
    yb = @localmem Float64 (MW,)
    t = @localmem Float64 (MW,)
    @inbounds if g <= nsub
        e = Int(slist[g])                           # regime-A subtree: serial walk, then signal
        a0 = Int(subtree_ptr[e])
        cnt = Int(subtree_ptr[e + 1]) - a0
        root = Int(subtree_nodes[a0 + cnt - 1])
        limit = Int(super_ptr[root + 1]) - 1
        for k in 1:cnt
            v = Int(subtree_nodes[a0 + k - 1])
            f = Int(front_nrows[v])
            w = Int(front_ncols[v])
            p0 = Int(front_ptr[v]) - 1
            rp = Int(rowptr[v]) - 1
            c0 = Int(super_ptr[v]) - 1
            ip = Int(inv_ptr[v]) - 1
            for kk in li:WGV:w
                acc = zero(Float64)
                for j in 1:w
                    acc += inv11[ip + (j - 1) * w + kk] * Y[c0 + j]
                end
                t[kk] = acc
            end
            @synchronize
            for kk in li:WGV:w
                Y[c0 + kk] = t[kk]
            end
            for i in (w + li):WGV:f
                acc = zero(Float64)
                for j in 1:w
                    acc += factor[p0 + (j - 1) * f + i] * t[j]
                end
                row = Int(rowval[rp + i])
                if row > limit
                    Atomix.@atomic Y[row] += -acc
                else
                    Y[row] -= acc
                end
            end
            @synchronize
        end
        if li == 1
            Threads.atomic_fence()
            pv = Int(snparent[root])
            pv > 0 && (Atomix.@atomic arrived[pv] += Int32(1))
        end
    else begin
        v = Int(flist[g - nsub])
        f = Int(front_nrows[v])
        w = Int(front_ncols[v])
        p0 = Int(front_ptr[v]) - 1
        rp = Int(rowptr[v]) - 1
        c0 = Int(super_ptr[v]) - 1
        ip = Int(inv_ptr[v]) - 1
        if li == 1
            tgt = nchild[v]
            pause = Int32(4)
            while (Atomix.@atomic arrived[v] += Int32(0)) < tgt
                z = 0.0
                for _ in 1:pause
                    z += 1.0
                end
                z < 0 && (yb[1] = z)        # keep the backoff loop alive
                pause = min(pause << 1, Int32(128))
            end
        end
        @synchronize
        Threads.atomic_fence()
        for kk in li:WGV:w                          # coherent read of the front's columns
            yb[kk] = Atomix.@atomic Y[c0 + kk] += 0.0
        end
        @synchronize
        for kk in li:WGV:w                          # t = inv(L11) * y
            acc = zero(Float64)
            for j in 1:w
                acc += inv11[ip + (j - 1) * w + kk] * yb[j]
            end
            t[kk] = acc
        end
        @synchronize
        for kk in li:WGV:w
            Y[c0 + kk] = t[kk]
        end
        for i in (w + li):WGV:f                     # L21 update, atomic scatter
            acc = zero(Float64)
            for j in 1:w
                acc += factor[p0 + (j - 1) * f + i] * t[j]
            end
            Atomix.@atomic Y[Int(rowval[rp + i])] += -acc
        end
        @synchronize
        if li == 1
            Threads.atomic_fence()
            pv = Int(snparent[v])
            pv > 0 && (Atomix.@atomic arrived[pv] += Int32(1))
        end
    end end
end

@kernel function bwd2_kernel!(Y, @Const(factor), @Const(inv11), @Const(inv_ptr),
                              @Const(flist), @Const(slist), nsub, @Const(subtree_ptr), @Const(subtree_nodes),
                              @Const(snparent), done, @Const(super_ptr), @Const(rowptr),
                              @Const(rowval), @Const(front_ptr), @Const(front_nrows), @Const(front_ncols),
                              ::Val{WGV}) where {WGV}
    li = @index(Local, Linear)
    g = @index(Group, Linear)
    xb = @localmem Float64 (WGV,)
    t = @localmem Float64 (MW,)
    t2 = @localmem Float64 (MW,)
    @inbounds if g > length(flist)
        e = Int(slist[g - length(flist)])          # regime-A subtree: wait for the root's parent, walk down
        a0 = Int(subtree_ptr[e])
        cnt = Int(subtree_ptr[e + 1]) - a0
        root = Int(subtree_nodes[a0 + cnt - 1])
        if li == 1
            pv = Int(snparent[root])
            if pv > 0
                pause = Int32(4)
                while (Atomix.@atomic done[pv] += Int32(0)) == Int32(0)
                    z = 0.0
                    for _ in 1:pause
                        z += 1.0
                    end
                    z < 0 && (t[1] = z)
                    pause = min(pause << 1, Int32(128))
                end
            end
        end
        @synchronize
        Threads.atomic_fence()
        for k in cnt:-1:1
            v = Int(subtree_nodes[a0 + k - 1])
            f = Int(front_nrows[v])
            w = Int(front_ncols[v])
            p0 = Int(front_ptr[v]) - 1
            rp = Int(rowptr[v]) - 1
            c0 = Int(super_ptr[v]) - 1
            ip = Int(inv_ptr[v]) - 1
            for kk in li:WGV:w
                acc = zero(Float64)
                col = p0 + (kk - 1) * f
                for i in (w + 1):f
                    acc += factor[col + i] * Y[Int(rowval[rp + i])]
                end
                t2[kk] = Y[c0 + kk] - acc
            end
            @synchronize
            for kk in li:WGV:w
                acc = zero(Float64)
                cb = ip + (kk - 1) * w
                for j in 1:w
                    acc += inv11[cb + j] * t2[j]
                end
                Y[c0 + kk] = acc
            end
            @synchronize
        end
    else begin
        v = Int(flist[length(flist) - g + 1])     # reverse topo order: parents get smaller block ids
        f = Int(front_nrows[v])
        w = Int(front_ncols[v])
        p0 = Int(front_ptr[v]) - 1
        rp = Int(rowptr[v]) - 1
        c0 = Int(super_ptr[v]) - 1
        ip = Int(inv_ptr[v]) - 1
        if li == 1
            pv = Int(snparent[v])
            if pv > 0
                pause = Int32(4)
                while (Atomix.@atomic done[pv] += Int32(0)) == Int32(0)
                    z = 0.0
                    for _ in 1:pause
                        z += 1.0
                    end
                    z < 0 && (t[1] = z)
                    pause = min(pause << 1, Int32(128))
                end
            end
        end
        @synchronize
        Threads.atomic_fence()
        for kk in li:WGV:w
            t[kk] = zero(Float64)
        end
        @synchronize
        i0 = w + 1
        while i0 <= f                               # chunked coherent gather of ancestor solutions
            ii = i0 + li - 1
            if ii <= f
                xb[li] = Atomix.@atomic Y[Int(rowval[rp + ii])] += 0.0
            end
            @synchronize
            m = min(WGV, f - i0 + 1)
            for kk in li:WGV:w
                acc = t[kk]
                col = p0 + (kk - 1) * f + i0 - 1
                for c in 1:m
                    acc += factor[col + c] * xb[c]
                end
                t[kk] = acc
            end
            @synchronize
            i0 += WGV
        end
        for kk in li:WGV:w                          # z = y - L21' x, then x = inv(L11)' z
            t2[kk] = Y[c0 + kk] - t[kk]
        end
        @synchronize
        for kk in li:WGV:w
            acc = zero(Float64)
            cb = ip + (kk - 1) * w
            for j in 1:w
                acc += inv11[cb + j] * t2[j]
            end
            Y[c0 + kk] = acc
        end
        @synchronize
        if li == 1
            Threads.atomic_fence()
            Atomix.@atomic done[v] += Int32(1)
        end
    end end
end


# chain kernels: ONE workgroup walks the (near-serial) tall-front chain at the
# top of the tree in plan order. Launched after the fused bottom tier, so every
# input is already in Y (stream order): no spins, no fences, no atomics — the
# single block owns every row it touches; plain loads/stores + block barriers.
@kernel function chain_fwd_kernel!(Y, @Const(factor), @Const(inv11), @Const(inv_ptr), @Const(clist),
                                   @Const(super_ptr), @Const(rowptr), @Const(rowval), @Const(front_ptr),
                                   @Const(front_nrows), @Const(front_ncols), ::Val{WGV}) where {WGV}
    li = @index(Local, Linear)
    yb = @localmem Float64 (MW,)
    t = @localmem Float64 (MW,)
    tp = @localmem Float64 (MW * NSTRIP,)
    @inbounds for q in 1:length(clist)
        v = Int(clist[q])
        f = Int(front_nrows[v])
        w = Int(front_ncols[v])
        p0 = Int(front_ptr[v]) - 1
        rp = Int(rowptr[v]) - 1
        c0 = Int(super_ptr[v]) - 1
        ip = Int(inv_ptr[v]) - 1
        for kk in li:WGV:w
            yb[kk] = Y[c0 + kk]
        end
        @synchronize
        for kk in li:WGV:w
            acc = zero(Float64)
            for j in 1:w
                acc += inv11[ip + (j - 1) * w + kk] * yb[j]
            end
            t[kk] = acc
        end
        @synchronize
        for kk in li:WGV:w
            Y[c0 + kk] = t[kk]
        end
        for i in (w + li):WGV:f
            acc = zero(Float64)
            for j in 1:w
                acc += factor[p0 + (j - 1) * f + i] * t[j]
            end
            Y[Int(rowval[rp + i])] -= acc
        end
        @synchronize
    end
end

@kernel function chain_bwd_kernel!(Y, @Const(factor), @Const(inv11), @Const(inv_ptr), @Const(clist),
                                   @Const(super_ptr), @Const(rowptr), @Const(rowval), @Const(front_ptr),
                                   @Const(front_nrows), @Const(front_ncols), ::Val{WGV}) where {WGV}
    li = @index(Local, Linear)
    t = @localmem Float64 (MW,)
    t2 = @localmem Float64 (MW,)
    tp = @localmem Float64 (MW * NSTRIP,)
    @inbounds for q in length(clist):-1:1
        v = Int(clist[q])
        f = Int(front_nrows[v])
        w = Int(front_ncols[v])
        p0 = Int(front_ptr[v]) - 1
        rp = Int(rowptr[v]) - 1
        c0 = Int(super_ptr[v]) - 1
        ip = Int(inv_ptr[v]) - 1
        kk = (li - 1) % w + 1                      # strip-parallel gather: t = L21' x
        st = (li - 1) ÷ w + 1
        if st <= NSTRIP
            acc = zero(Float64)
            col = p0 + (kk - 1) * f
            for i in (w + st):NSTRIP:f
                acc += factor[col + i] * Y[Int(rowval[rp + i])]
            end
            tp[(st - 1) * MW + kk] = acc
        end
        @synchronize
        for k2 in li:WGV:w
            acc = Y[c0 + k2]
            for st2 in 1:NSTRIP
                acc -= tp[(st2 - 1) * MW + k2]
            end
            t2[k2] = acc
        end
        @synchronize
        kk2 = (li - 1) % w + 1                     # strip-parallel GEMV: x = inv(L11)' t2
        st2b = (li - 1) ÷ w + 1
        if st2b <= NSTRIP
            acc = zero(Float64)
            cb = ip + (kk2 - 1) * w
            for j in st2b:NSTRIP:w
                acc += inv11[cb + j] * t2[j]
            end
            tp[(st2b - 1) * MW + kk2] = acc
        end
        @synchronize
        for k2 in li:WGV:w
            acc = zero(Float64)
            for st3 in 1:NSTRIP
                acc += tp[(st3 - 1) * MW + k2]
            end
            t[k2] = acc
        end
        @synchronize
        for k2 in li:WGV:w
            Y[c0 + k2] = t[k2]
        end
        @synchronize
    end
end

# fused segment: the maximal run of plan entries from `kstart` in which every
# front has width <= wcap (regime-A subtree entries break a run)
function build_fused(s; kstart::Int = 1, wcap::Int = MW, merged::Bool = (kstart == 1))
    S = s.symbolic
    ws = s.workspace
    plan = ws.plan
    nodes = S.schedule.group_nodes
    widths = S.schedule.width
    narrow(k) = plan.kind[k] != SDS.SOLVE_SUBTREES &&
                all(widths[nodes[q]] <= wcap for q in plan.first[k]:plan.last[k])
    i1 = kstart
    while i1 <= length(plan.kind) && !narrow(i1)
        i1 += 1
    end
    i1 > length(plan.kind) && return nothing
    i2 = i1
    while i2 + 1 <= length(plan.kind) && narrow(i2 + 1)
        i2 += 1
    end
    flist_h = Int32[nodes[q] for k in i1:i2 for q in plan.first[k]:plan.last[k]]
    ns = length(widths)
    in_range = falses(ns)
    for v in flist_h
        in_range[v] = true
    end
    child_ptr_h = Array(S.child_ptr)
    child_list_h = Array(S.child_list)
    nchild_h = Int32[child_ptr_h[v + 1] - child_ptr_h[v] for v in 1:ns]
    stp_h = Array(S.subtree_ptr)
    stn_h = Array(S.subtree_nodes)
    live_root = falses(ns)                   # roots of subtrees merged into this launch (the merged tier only)
    if merged
        for e in 1:(length(stp_h) - 1)
            live_root[stn_h[stp_h[e + 1] - 1]] = true
        end
    end
    arrived_base_h = zeros(Int32, ns)
    for v in 1:ns
        in_range[v] || continue
        nc = child_ptr_h[v + 1] - child_ptr_h[v]
        cnt = 0
        for k in 1:nc
            c = child_list_h[child_ptr_h[v] + k - 1]
            (in_range[c] || live_root[c]) || (cnt += 1)
        end
        arrived_base_h[v] = Int32(cnt)
    end
    done_base_h = Int32[in_range[v] ? 0 : 1 for v in 1:ns]
    @printf "fused: entries %d:%d of %d, %d fronts\n" i1 i2 length(plan.kind) length(flist_h)
    return (; i1, i2, flist = DVEC(flist_h), arrived = KA.zeros(BACKEND[], Int32, ns),
            arrived_base = DVEC(arrived_base_h), done = KA.zeros(BACKEND[], Int32, ns),
            done_base = DVEC(done_base_h), nchild = DVEC(nchild_h), nfused = length(flist_h))
end

function run_fused!(fz, wg, Yv, s, backend)
    S, N = s.symbolic, s.numeric
    if wg > 512
        chain_fwd_kernel!(backend, CHWG)(Yv, N.factor, INV11[], INVPTR[], fz.flist, S.super_ptr, S.rowptr,
                                         S.rowval, S.front_ptr, S.front_nrows, S.front_ncols, Val(CHWG);
                                         ndrange = CHWG)
        return nothing
    end
    copyto!(fz.arrived, fz.arrived_base)
    if wg == 64
        fwd2_kernel!(backend, 64)(Yv, N.factor, INV11[], INVPTR[], fz.flist, SSUB[], NSUB[], S.subtree_ptr,
                                  S.subtree_nodes, S.snparent, fz.arrived, fz.nchild,
                                  S.super_ptr, S.rowptr, S.rowval, S.front_ptr, S.front_nrows, S.front_ncols,
                                  Val(64); ndrange = 64 * (fz.nfused + NSUB[]))
    else
        error("unused")
    end
end

function run_fused_bwd!(fz, wg, Yv, s, backend)
    S, N = s.symbolic, s.numeric
    if wg > 512
        chain_bwd_kernel!(backend, CHWG)(Yv, N.factor, INV11[], INVPTR[], fz.flist, S.super_ptr, S.rowptr,
                                         S.rowval, S.front_ptr, S.front_nrows, S.front_ncols, Val(CHWG);
                                         ndrange = CHWG)
        return nothing
    end
    copyto!(fz.done, fz.done_base)
    if wg == 64
        bwd2_kernel!(backend, 64)(Yv, N.factor, INV11[], INVPTR[], fz.flist, SSUB[], NSUB[], S.subtree_ptr,
                                  S.subtree_nodes, S.snparent, fz.done,
                                  S.super_ptr, S.rowptr, S.rowval, S.front_ptr, S.front_nrows, S.front_ncols,
                                  Val(64); ndrange = 64 * (fz.nfused + NSUB[]))
    else
        error("unused")
    end
end

const INV11 = Ref{Any}()
const INVPTR = Ref{Any}()
const SSUB = Ref{Any}()      # subtree ids sorted by size, longest first
const NSUB = Ref{Int}(0)

function algo2_solve!(x, s, b, inv11, inv_ptr, fz; debug::Bool = false)
    S, N, ws = s.symbolic, s.numeric, s.workspace
    bm = SDS.batch_map(N; nrhs = 1)
    SDS.permute_rhs!(ws.Y, b, S.perm; transposed = false, bm)
    plan = ws.plan
    nodes = S.schedule.group_nodes
    widths = S.schedule.width
    p = SDS._solve_impls(N, S, :auto)
    backend = KA.get_backend(ws.Y)
    Yv = vec(ws.Y)
    args(first, count) = (Yv, N.factor, inv11, inv_ptr, S.group_nodes, first, count, S.subtree_ptr,
                          S.subtree_nodes, S.super_ptr, S.rowptr, S.rowval, S.front_ptr, S.front_nrows,
                          S.front_ncols)
    fwd_entry(k) = begin
        a, b_ = plan.first[k], plan.last[k]
        if plan.kind[k] == SDS.SOLVE_SUBTREES
        elseif plan.kind[k] == SDS.SOLVE_DENSE
            any(widths[nodes[q]] <= MW for q in a:b_) &&
                fwd1_kernel!(backend, WG)(args(a, b_ - a + 1)..., Val(false), Val(WG); ndrange = WG * (b_ - a + 1))
            for q in a:b_
                widths[nodes[q]] > MW && SDS._fwd_dense!(ws, S, N, nodes[q], 1, false, p, SDS._SV_CHOLESKY)
            end
        else
            fwd1_kernel!(backend, WG)(args(a, b_ - a + 1)..., Val(plan.kind[k] == SDS.SOLVE_SUBTREES), Val(WG);
                                      ndrange = WG * (b_ - a + 1))
        end
    end
    bwd_entry(k) = begin
        a, b_ = plan.first[k], plan.last[k]
        if plan.kind[k] == SDS.SOLVE_SUBTREES
        elseif plan.kind[k] == SDS.SOLVE_DENSE
            for q in b_:-1:a
                widths[nodes[q]] > MW && SDS._bwd_dense!(ws, S, N, nodes[q], 1, p, SDS._SV_CHOLESKY)
            end
            any(widths[nodes[q]] <= MW for q in a:b_) &&
                bwd1_kernel!(backend, WG)(args(a, b_ - a + 1)..., Val(false), Val(WG); ndrange = WG * (b_ - a + 1))
        else
            bwd1_kernel!(backend, WG)(args(a, b_ - a + 1)..., Val(plan.kind[k] == SDS.SOLVE_SUBTREES), Val(WG);
                                      ndrange = WG * (b_ - a + 1))
        end
    end
    INV11[] = inv11; INVPTR[] = inv_ptr
    segs, covered = fz
    k = 1
    for (fzk, wg) in segs
        while k < fzk.i1
            fwd_entry(k); k += 1
        end
        run_fused!(fzk, wg, Yv, s, backend)
        k = fzk.i2 + 1
    end
    while k <= length(plan.kind)
        fwd_entry(k); k += 1
    end
    debug && (KA.synchronize(BACKEND[]); println("  algo2: forward done"); flush(stdout))
    if SDS._is_ldlt_structure(S.structure)      # LDL: apply D between the unit-triangular sweeps
        Yv ./= view(N.d, 1:length(Yv))
    end
    k = length(plan.kind)
    for (fzk, wg) in Iterators.reverse(segs)
        while k > fzk.i2
            bwd_entry(k); k -= 1
        end
        run_fused_bwd!(fzk, wg, Yv, s, backend)
        k = fzk.i1 - 1
    end
    while k >= 1
        bwd_entry(k); k -= 1
    end
    debug && (KA.synchronize(BACKEND[]); println("  algo2: backward done"); flush(stdout))
    SDS.unpermute_solution!(x, ws.Y, S.perm; transposed = false, bm)
    return x
end






const SYRK_IN = 64
const TILE = 32
const NL64 = 2080                 # packed 64x64 lower triangle

# ---------------------------------------------------------------------------
# no-pivot LDLᵀ mode of the fused kernel (real element types): the factor stores
# the UNIT lower panels and D goes to `d` (global factor-column indexing), the
# stock LDL layout, so the stock solve and the fused solve (one extra D pass
# between the sweeps) both work on it. A zero pivot sets info[v] to the local
# column and the front's consumers skip, exactly the Cholesky failure path;
# NEGATIVE pivots factor through (inertia for MadNLP). The mode is a compile-time
# `Val{LDL}` kernel parameter: the Cholesky instantiation is unchanged machine code.
#
# Panel convention while a front is in flight: the packed `sh` triangle keeps the
# pivot d_j ON THE DIAGONAL (the unit is implied), so the in-panel TRSM both
# reconstructs the unscaled column (`* sh[j,j]`) and divides by the pivot
# (`/ sh[kk,kk]`) with the same reads as Cholesky; write-out emits the unit
# diagonal. The TRSM role reloads the panel from the factor (diag = 1) and
# restores d into the diagonal slots from `d`; the TILE roles stage the 32
# pivots of each k-block in sh[2049:2080] (free: NL64 - 2*1024) and fold them
# into the Ai tile only.

# regime A: a dedicated no-pivot LDL variant of SDS.subtree_cholesky_kernel! (the stock
# kernel stays untouched; its CPU-launch allocation budget is load-bearing in the tests).
# Identical skeleton and assembly helpers; only the two column phases and the extension
# slots differ: the whole trailing triangle (CB included) takes the d-folded rank-1, so
# no separate finalize/CB pass exists, and d/piv/pivot_kind are filled as the stock LDL
# path (identity pivot order; a zero pivot sets the front status, as Cholesky).
const _SP_C0 = SDS._ST_CTL + 1
const _SP_CTL = _SP_C0

@inline function _subtree_ldl_update!(buf, ctl, pv, j, li, d, piv, pivot_kind, ::Val{WG}) where {WG}
    @inbounds begin
        T = eltype(buf)
        IT = eltype(ctl)
        f = ctl[SDS._ST_F]
        lf = ctl[SDS._ST_LF] - 1
        if ctl[SDS._ST_STATUS] == 0
            dk = real(buf[lf + SDS._packed(j, j, f)])
            if !iszero(dk)
                if li == 1
                    g = Int(ctl[_SP_C0]) + j - 1
                    d[g] = dk
                    piv[g] = Int32(g)
                    pivot_kind[g] = SDS.PIVOT_KIND_1X1
                    pv[1] = dk
                end
                rd = inv(dk)
                for i in (j + li):WG:f
                    buf[lf + SDS._packed(i, j, f)] *= rd
                end
            elseif li == 1
                ctl[SDS._ST_STATUS] = j % IT
            end
        end
    end
    return nothing
end

@inline function _subtree_ldl_scale!(buf, ctl, pv, j, li, ::Val{WG}) where {WG}
    @inbounds begin
        if ctl[SDS._ST_STATUS] == 0
            f = ctl[SDS._ST_F]
            lf = ctl[SDS._ST_LF] - 1
            dk = pv[1]
            li == 1 && (buf[lf + SDS._packed(j, j, f)] = one(eltype(buf)))
            r = f - j
            for q in (li - 1):WG:(r * (r + 1) ÷ 2 - 1)
                a, b = SDS._tri_decode(q, r)
                i = j + a
                k = j + b
                buf[lf + SDS._packed(i, k, f)] -= buf[lf + SDS._packed(i, j, f)] * dk *
                                                  buf[lf + SDS._packed(k, j, f)]
            end
        end
    end
    return nothing
end

@kernel function subtree_ldl_kernel!(factor, stack, info, d, piv, pivot_kind, @Const(nzval), @Const(amap),
                                     @Const(amap_ptr), @Const(amap_src), @Const(trees), bm, @Const(subtree_ptr),
                                     @Const(subtree_nodes), @Const(super_ptr), @Const(front_ptr),
                                     @Const(front_nrows), @Const(front_ncols), @Const(cb_ptr),
                                     @Const(local_front), @Const(local_cb), @Const(child_ptr),
                                     @Const(child_list), @Const(relind_ptr), @Const(relind),
                                     ::Val{NE}, ::Val{WG}) where {NE, WG}
    @uniform TT = eltype(factor)
    @uniform IT = eltype(subtree_nodes)
    li = @index(Local, Linear)
    G = @index(Group, Linear)
    buf = @localmem TT (NE,)
    ctl = @localmem IT (_SP_CTL,)
    pv = @localmem TT (1,)
    if li == 1
        @inbounds t = trees[bm.first + SDS._bm_node(bm, G) - 1]
        @inbounds ctl[SDS._ST_FIRST] = subtree_ptr[t]
        @inbounds ctl[SDS._ST_COUNT] = subtree_ptr[t + 1] - subtree_ptr[t]
    end
    @synchronize
    for k in 1:ctl[SDS._ST_COUNT]
        if li == 1
            SDS._subtree_setup!(ctl, k, subtree_nodes, front_nrows, front_ncols, local_front, local_cb,
                                child_ptr)
            @inbounds ctl[_SP_C0] = super_ptr[ctl[SDS._ST_NODE]] % IT
        end
        @synchronize
        SDS._subtree_zero!(buf, ctl, li, Val(WG))
        @synchronize
        SDS._subtree_scatter!(buf, ctl, SDS._mview(nzval, SDS._bm_gmember(bm, G), bm.nbatch), amap, amap_ptr,
                              amap_src, front_ptr, li, Val(WG))
        @synchronize
        for kc in 1:ctl[SDS._ST_NCHILD]
            SDS._subtree_extend_add!(buf, ctl, kc, child_ptr, child_list, front_nrows, front_ncols, local_cb,
                                     relind_ptr, relind, li, Val(WG))
            @synchronize
        end
        for j in 1:ctl[SDS._ST_W]
            _subtree_ldl_update!(buf, ctl, pv, j, li, SDS._mview(d, SDS._bm_gmember(bm, G), bm.nbatch),
                                 SDS._mview(piv, SDS._bm_gmember(bm, G), bm.nbatch),
                                 SDS._mview(pivot_kind, SDS._bm_gmember(bm, G), bm.nbatch), Val(WG))
            @synchronize
            _subtree_ldl_scale!(buf, ctl, pv, j, li, Val(WG))
            @synchronize
        end
        mb = SDS._bm_gmember(bm, G)
        SDS._subtree_write!(factor, SDS._mview(stack, mb, bm.nbatch), SDS._iview(info, mb, bm.nbatch), buf, ctl,
                            SDS.member_panels(front_ptr, mb, bm.nbatch), cb_ptr, li, Val(WG))
        @synchronize
        for r in 1:ctl[SDS._ST_ROUNDS]
            SDS._subtree_move!(buf, ctl, r, li, Val(WG))
            @synchronize
        end
    end
end

@inline function _front_ldl_update!(sh, st, pv, j, v, li, front_ncols, d, g0, ::Val{WG}) where {WG}
    @inbounds begin
        w = Int(front_ncols[v])
        if j <= w && st[1] == Int32(0)
            dk = real(sh[SDS._packed(j, j, 64)])
            if !iszero(dk)
                if li == 1
                    d[g0 + j] = dk
                    pv[1] = dk
                end
                rd = inv(dk)
                for i in (j + li):WG:w
                    sh[SDS._packed(i, j, 64)] *= rd
                end
            elseif li == 1
                st[1] = Int32(j)
            end
        end
    end
    return nothing
end

@inline function _front_ldl_scale!(sh, st, pv, j, v, li, front_ncols, ::Val{WG}) where {WG}
    @inbounds begin
        w = Int(front_ncols[v])
        if j <= w && st[1] == Int32(0)
            dk = pv[1]
            r = w - j
            for q in (li - 1):WG:(r * (r + 1) ÷ 2 - 1)
                a, b = SDS._tri_decode(q, r)
                i = j + a
                k = j + b
                sh[SDS._packed(i, k, 64)] -= sh[SDS._packed(i, j, 64)] * dk * sh[SDS._packed(k, j, 64)]
            end
        end
    end
    return nothing
end

# F22 ← F22 − (L21 D) L21ᵀ on the private CB; d_k read from the sh diagonal (still live)
@inline function _front_ldl_syrk!(factor, pcb, sh, st, v, li, front_ptr, front_nrows, front_ncols, cb2,
                                  ::Val{WG}) where {WG}
    @inbounds begin
        TT = eltype(factor)
        c0 = Int(cb2[v])
        if st[1] == Int32(0) && c0 > 0
            f = Int(front_nrows[v])
            w = Int(front_ncols[v])
            m = f - w
            p0 = Int(front_ptr[v]) + w - 1
            for q in (li - 1):WG:(m * (m + 1) ÷ 2 - 1)
                ii, jj = SDS._tri_decode(q, m)
                acc = zero(TT)
                for k in 1:w
                    acc += factor[p0 + (k - 1) * f + ii] * real(sh[SDS._packed(k, k, 64)]) *
                           factor[p0 + (k - 1) * f + jj]
                end
                pcb[c0 + SDS._packed(ii, jj, m) - 1] -= acc
            end
        end
    end
    return nothing
end

const ROLE_PANEL_B = Int8(1)
const ROLE_PANEL_C = Int8(2)
const ROLE_TRSM = Int8(3)
const ROLE_TILE = Int8(4)
const ROLE_EA = Int8(5)
const ROLE_WPANEL = Int8(6)       # wide fronts in-kernel: blocked 64-panel Cholesky
const ROLE_WTRSM = Int8(7)        #   (rb = panel k | rb = chunk, rc = k | rc = k*1024 + tjj)
const ROLE_WTILE = Int8(8)
const ROLE_WSIG = Int8(9)

# spin with backoff on an Int32 cell until pred(value) holds; thread 1 only
@inline function _spin_until_eq!(arr, idx, target, scratch, scap)
    pause = min(Int32(4), scap)
    while (Atomix.@atomic arr[idx] += Int32(0)) != target
        z = 0.0
        for _ in 1:pause
            z += 1.0
        end
        z < 0 && (scratch[1] = z)
        pause = min(pause << 1, scap)
    end
    return nothing
end

@kernel function fused_fact_kernel!(factor, stack, pcb, info, @Const(nzval), @Const(amap), @Const(amap_ptr),
                                    @Const(amap_src), @Const(cb2), @Const(wt), @Const(role), @Const(ra), @Const(rb), @Const(rc),
                                    @Const(nchild), arrived, ea_left, panel_done, trsm_left, tiles_left,
                                    cb_done, @Const(snparent), @Const(front_ptr), @Const(front_nrows),
                                    @Const(front_ncols), @Const(cb_ptr), @Const(child_ptr),
                                    @Const(child_list), @Const(relind_ptr), @Const(relind), base,
                                    @Const(wbase), wupd, wpdone2, wtrsml, wcbl, scap, d, @Const(super_ptr),
                                    ::Val{LDL}, ::Val{WG}) where {LDL, WG}
    @uniform TT = eltype(factor)
    @uniform RT = real(eltype(factor))
    li = @index(Local, Linear)
    G = Int(base) + @index(Group, Linear)
    sh = @localmem TT (NL64,)
    st = @localmem Int32 (1,)
    piv = @localmem RT (1,)
    fscr = @localmem Float64 (1,)
    @inbounds begin
        r = role[G]
        if r == ROLE_EA
            v = Int(ra[G])                          # parent
            c = Int(rb[G])                          # child
            q0 = (Int(rc[G]) - 1) * 16384
            if li == 1
                _spin_until_eq!(cb_done, c, Int32(1), fscr, scap)
            end
            @synchronize
            Threads.atomic_fence()
            cpriv = Int(cb2[c])                  # child CB: private (mega) or stock (subtree root)
            cb = cpriv > 0 ? cpriv : Int(cb_ptr[c])
            mc = Int(front_nrows[c]) - Int(front_ncols[c])
            fp = Int(front_nrows[v])
            wp = Int(front_ncols[v])
            mp = fp - wp
            pp = Int(front_ptr[v])
            cp = Int(cb2[v])                     # parent CB is always private
            r0 = Int(relind_ptr[c]) - 1
            qe = min(q0 + 16384, mc * mc) - 1
            for q in (q0 + li - 1):WG:qe
                jj = q ÷ mc + 1
                ii = q - (jj - 1) * mc + 1
                if ii >= jj
                    ri = Int(relind[r0 + ii])
                    rj = Int(relind[r0 + jj])
                    x = cpriv > 0 ? pcb[cb + SDS._packed(ii, jj, mc) - 1] :
                                    stack[cb + SDS._packed(ii, jj, mc) - 1]
                    if rj <= wp
                        Atomix.@atomic factor[pp + (rj - 1) * fp + ri - 1] += x
                    else
                        Atomix.@atomic pcb[cp + SDS._packed(ri - wp, rj - wp, mp) - 1] += x
                    end
                end
            end
            @synchronize
            Threads.atomic_fence()      # ALL threads: each wave drains its own stores (gfx9)
            @synchronize
            if li == 1
                Atomix.@atomic ea_left[v] += Int32(-1)
            end
        elseif r == ROLE_PANEL_B || r == ROLE_PANEL_C
            v = Int(ra[G])
            if li == 1
                if r == ROLE_PANEL_B
                    _spin_until_eq!(arrived, v, nchild[v], fscr, scap)
                else
                    _spin_until_eq!(ea_left, v, Int32(0), fscr, scap)
                end
            end
            @synchronize
            Threads.atomic_fence()
            if r == ROLE_PANEL_B
                fz = Int(front_nrows[v])
                wz = Int(front_ncols[v])
                pz = Int(front_ptr[v])
                for q in (li - 1):WG:(fz * wz - 1)   # zero the panel; private CBs are pre-zeroed
                    factor[pz + q] = zero(TT)
                end
                @synchronize
                SDS._scatter_front!(factor, nzval, amap, amap_ptr, amap_src, v,
                                    SDS._member_shift(front_ptr, v, 1, 1), li, Val(WG))
                @synchronize
                nc = Int(child_ptr[v + 1]) - Int(child_ptr[v])
                for kc in 1:nc
                    c = Int(child_list[child_ptr[v] + kc - 1])
                    cpr = Int(cb2[c])
                    cbc = cpr > 0 ? cpr : Int(cb_ptr[c])
                    if cbc > 0
                        mc = Int(front_nrows[c]) - Int(front_ncols[c])
                        r0 = Int(relind_ptr[c]) - 1
                        mp = fz - wz
                        cp2 = Int(cb2[v])
                        for q in (li - 1):WG:(mc * mc - 1)
                            jj = q ÷ mc + 1
                            ii = q - (jj - 1) * mc + 1
                            if ii >= jj
                                ri = Int(relind[r0 + ii])
                                rj = Int(relind[r0 + jj])
                                x = cpr > 0 ? pcb[cbc + SDS._packed(ii, jj, mc) - 1] :
                                              stack[cbc + SDS._packed(ii, jj, mc) - 1]
                                if rj <= wz
                                    factor[pz + (rj - 1) * fz + ri - 1] += x
                                else
                                    pcb[cp2 + SDS._packed(ri - wz, rj - wz, mp) - 1] += x
                                end
                            end
                        end
                    end
                    @synchronize
                end
            end
            SDS._front_load!(sh, st, factor, v, li, front_ptr, front_nrows, front_ncols, Val(64), Val(WG))
            @synchronize
            w = Int(front_ncols[v])
            g0 = Int(super_ptr[v]) - 1
            for j in 1:w
                if LDL
                    _front_ldl_update!(sh, st, piv, j, v, li, front_ncols, d, g0, Val(WG))
                else
                    SDS._front_chol_update!(sh, st, piv, j, v, li, front_ncols, Val(64), Val(WG))
                end
                @synchronize
                if LDL
                    _front_ldl_scale!(sh, st, piv, j, v, li, front_ncols, Val(WG))
                else
                    SDS._front_chol_scale!(sh, st, piv, j, v, li, front_ncols, Val(64), Val(WG))
                end
                @synchronize
            end
            f = Int(front_nrows[v])
            p0 = Int(front_ptr[v])
            for q in (li - 1):WG:(w * w - 1)
                j = q ÷ w + 1
                i = q - (j - 1) * w + 1
                i >= j && (factor[p0 + (j - 1) * f + i - 1] = (LDL && i == j) ? one(TT) :
                                                              sh[SDS._packed(i, j, 64)])
            end
            m = f - w
            if st[1] == Int32(0) && m <= SYRK_IN
                for i in (w + li):WG:f
                    for kk in 1:w
                        x = factor[p0 + (kk - 1) * f + i - 1]
                        for j in 1:(kk - 1)
                            lij = factor[p0 + (j - 1) * f + i - 1]
                            x -= LDL ? lij * (real(sh[SDS._packed(j, j, 64)]) * sh[SDS._packed(kk, j, 64)]) :
                                       lij * sh[SDS._packed(kk, j, 64)]
                        end
                        factor[p0 + (kk - 1) * f + i - 1] = x / sh[SDS._packed(kk, kk, 64)]
                    end
                end
            end
            li == 1 && (info[v] = st[1])
            @synchronize
            if m <= SYRK_IN
                if LDL
                    _front_ldl_syrk!(factor, pcb, sh, st, v, li, front_ptr, front_nrows, front_ncols, cb2,
                                     Val(WG))
                else
                    SDS._front_syrk!(factor, pcb, st, v, li, front_ptr, front_nrows, front_ncols, cb2,
                                     Val(WG))
                end
                @synchronize
                Threads.atomic_fence()      # ALL threads: each wave drains its own stores (gfx9)
                @synchronize
                if li == 1
                    Atomix.@atomic cb_done[v] += Int32(1)
                    pv = Int(snparent[v])
                    pv > 0 && (Atomix.@atomic arrived[pv] += Int32(1))
                end
            else
                Threads.atomic_fence()      # ALL threads (gfx9)
                @synchronize
                if li == 1
                    Atomix.@atomic panel_done[v] += Int32(1)
                end
            end
        elseif r == ROLE_TRSM
            v = Int(ra[G])
            if li == 1
                _spin_until_eq!(panel_done, v, Int32(1), fscr, scap)
            end
            @synchronize
            Threads.atomic_fence()
            f = Int(front_nrows[v])
            w = Int(front_ncols[v])
            p0 = Int(front_ptr[v])
            for q in li:WG:(w * (w + 1) ÷ 2)
                j = 1
                acc = w
                qq = q
                while qq > acc
                    qq -= acc
                    acc -= 1
                    j += 1
                end
                i = j + qq - 1
                # LDL: the factor diagonal is unit; restore d in the same loop (a second
                # loop would race this one on the diagonal slots)
                sh[SDS._packed(i, j, 64)] = (LDL && i == j) ? TT(d[Int(super_ptr[v]) - 1 + j]) :
                                            factor[p0 + (j - 1) * f + i - 1]
            end
            @synchronize
            if info[v] == Int32(0)
                i = w + (Int(rb[G]) - 1) * WG + li
                if i <= f
                    for kk in 1:w
                        x = factor[p0 + (kk - 1) * f + i - 1]
                        for j in 1:(kk - 1)
                            lij = factor[p0 + (j - 1) * f + i - 1]
                            x -= LDL ? lij * (real(sh[SDS._packed(j, j, 64)]) * sh[SDS._packed(kk, j, 64)]) :
                                       lij * sh[SDS._packed(kk, j, 64)]
                        end
                        factor[p0 + (kk - 1) * f + i - 1] = x / sh[SDS._packed(kk, kk, 64)]
                    end
                end
            end
            @synchronize
            Threads.atomic_fence()      # ALL threads: each wave drains its own stores (gfx9)
            @synchronize
            if li == 1
                left = (Atomix.@atomic trsm_left[v] += Int32(-1))
                if left == Int32(0) && tiles_left[v] == Int32(0)   # new-value semantics: 0 = last finisher
                    Atomix.@atomic cb_done[v] += Int32(1)
                    pv = Int(snparent[v])
                    pv > 0 && (Atomix.@atomic arrived[pv] += Int32(1))
                end
            end
        elseif r == ROLE_TILE
            v = Int(ra[G])
            if li == 1
                _spin_until_eq!(trsm_left, v, Int32(0), fscr, scap)
            end
            @synchronize
            Threads.atomic_fence()
            f = Int(front_nrows[v])
            w = Int(front_ncols[v])
            m = f - w
            p0 = Int(front_ptr[v]) + w - 1
            c0 = Int(cb2[v])
            i0 = (Int(rb[G]) - 1) * TILE
            j0 = (Int(rc[G]) - 1) * TILE
            ok = info[v] == Int32(0) && c0 > 0
            acc1 = zero(TT); acc2 = zero(TT); acc3 = zero(TT); acc4 = zero(TT)
            acc5 = zero(TT); acc6 = zero(TT); acc7 = zero(TT); acc8 = zero(TT)
            kk0 = 0
            while kk0 < w                           # k in two halves: Ai/Aj share sh (2 x 1024)
                kb = min(32, w - kk0)
                if LDL                              # stage the block's pivots; fold into Ai only
                    if ok
                        g0 = Int(super_ptr[v]) - 1
                        for c in li:WG:kb
                            sh[2048 + c] = TT(d[g0 + kk0 + c])
                        end
                    end
                    @synchronize
                end
                if ok
                    for q in (li - 1):WG:(TILE * kb - 1)
                        rr = q % TILE + 1
                        cc = q ÷ TILE + 1
                        row = i0 + rr
                        sh[rr + (cc - 1) * TILE] = row <= m ?
                            (LDL ? factor[p0 + (kk0 + cc - 1) * f + row] * sh[2048 + cc] :
                                   factor[p0 + (kk0 + cc - 1) * f + row]) : zero(TT)
                        row2 = j0 + rr
                        sh[1024 + rr + (cc - 1) * TILE] =
                            row2 <= m ? factor[p0 + (kk0 + cc - 1) * f + row2] : zero(TT)
                    end
                end
                @synchronize
                if ok
                    for (slot, q) in enumerate((li - 1):WG:(TILE * TILE - 1))
                        rr = q % TILE + 1
                        cc = q ÷ TILE + 1
                        a = zero(TT)
                        for kk in 1:kb
                            a += sh[rr + (kk - 1) * TILE] * sh[1024 + cc + (kk - 1) * TILE]
                        end
                        slot == 1 && (acc1 += a); slot == 2 && (acc2 += a)
                        slot == 3 && (acc3 += a); slot == 4 && (acc4 += a)
                        slot == 5 && (acc5 += a); slot == 6 && (acc6 += a)
                        slot == 7 && (acc7 += a); slot == 8 && (acc8 += a)
                    end
                end
                @synchronize
                kk0 += 32
            end
            if ok
                pw = Int(wt[v])
                if pw == 0
                    for (slot, q) in enumerate((li - 1):WG:(TILE * TILE - 1))
                        rr = q % TILE + 1
                        cc = q ÷ TILE + 1
                        ii = i0 + rr
                        jj = j0 + cc
                        if ii <= m && jj <= m && ii >= jj
                            a = slot == 1 ? acc1 : slot == 2 ? acc2 : slot == 3 ? acc3 : slot == 4 ? acc4 :
                                slot == 5 ? acc5 : slot == 6 ? acc6 : slot == 7 ? acc7 : acc8
                            pcb[c0 + SDS._packed(ii, jj, m) - 1] -= a
                        end
                    end
                else                                 # write the finished CB entry through into the parent
                    fpw = Int(front_nrows[pw])
                    wpw = Int(front_ncols[pw])
                    mpw = fpw - wpw
                    ppw = Int(front_ptr[pw])
                    cpw = Int(cb2[pw])
                    r0w = Int(relind_ptr[v]) - 1
                    for (slot, q) in enumerate((li - 1):WG:(TILE * TILE - 1))
                        rr = q % TILE + 1
                        cc = q ÷ TILE + 1
                        ii = i0 + rr
                        jj = j0 + cc
                        if ii <= m && jj <= m && ii >= jj
                            a = slot == 1 ? acc1 : slot == 2 ? acc2 : slot == 3 ? acc3 : slot == 4 ? acc4 :
                                slot == 5 ? acc5 : slot == 6 ? acc6 : slot == 7 ? acc7 : acc8
                            x = pcb[c0 + SDS._packed(ii, jj, m) - 1] - a   # children's extend-adds + own syrk
                            ri = Int(relind[r0w + ii])
                            rj = Int(relind[r0w + jj])
                            if rj <= wpw
                                factor[ppw + (rj - 1) * fpw + ri - 1] += x
                            else
                                pcb[cpw + SDS._packed(ri - wpw, rj - wpw, mpw) - 1] += x
                            end
                        end
                    end
                end
            end
            @synchronize
            Threads.atomic_fence()      # ALL threads: each wave drains its own stores (gfx9)
            @synchronize
            if li == 1
                left = (Atomix.@atomic tiles_left[v] += Int32(-1))
                if left == Int32(0)                  # Atomix += returns the NEW value: 0 = last finisher
                    Atomix.@atomic cb_done[v] += Int32(1)
                    pv = Int(snparent[v])
                    pv > 0 && (Atomix.@atomic arrived[pv] += Int32(1))
                    Int(wt[v]) > 0 && (Atomix.@atomic ea_left[Int(wt[v])] += Int32(-1))
                end
            end
        elseif r == ROLE_WPANEL
            v = Int(ra[G]); k = Int(rb[G])
            sl = Int(wbase[v]) + k
            if li == 1
                if k == 1
                    _spin_until_eq!(ea_left, v, Int32(0), fscr, scap)
                else
                    _spin_until_eq!(wupd, sl, Int32(0), fscr, scap)
                end
            end
            @synchronize
            Threads.atomic_fence()
            f = Int(front_nrows[v]); w = Int(front_ncols[v])
            c0 = (k - 1) * 64
            kb = min(64, w - c0)
            p0 = Int(front_ptr[v])
            li == 1 && (st[1] = Int32(0))
            @synchronize
            for q in (li - 1):WG:(kb * kb - 1)
                jj = q ÷ kb + 1
                ii = q - (jj - 1) * kb + 1
                ii >= jj && (sh[SDS._packed(ii, jj, 64)] = factor[p0 + (c0 + jj - 1) * f + c0 + ii - 1])
            end
            @synchronize
            for j in 1:kb
                if li == 1
                    dj = real(sh[SDS._packed(j, j, 64)])
                    if LDL
                        if st[1] == Int32(0) && !iszero(dj)
                            piv[1] = dj                      # d stays on the diagonal
                            d[Int(super_ptr[v]) - 1 + c0 + j] = dj
                        elseif st[1] == Int32(0)
                            st[1] = Int32(c0 + j)
                            piv[1] = one(RT)
                        end
                    else
                        if st[1] == Int32(0) && dj > zero(RT)
                            piv[1] = sqrt(dj)
                            sh[SDS._packed(j, j, 64)] = piv[1]
                        elseif st[1] == Int32(0)
                            st[1] = Int32(c0 + j)
                            piv[1] = one(RT)
                        end
                    end
                end
                @synchronize
                pvt = piv[1]
                if st[1] == Int32(0)
                    rpv = inv(pvt)
                    for i in (j + li):WG:kb
                        if LDL
                            sh[SDS._packed(i, j, 64)] *= rpv
                        else
                            sh[SDS._packed(i, j, 64)] /= pvt
                        end
                    end
                end
                @synchronize
                if st[1] == Int32(0)
                    for q in (li - 1):WG:(kb * kb - 1)
                        jj = q ÷ kb + 1
                        ii = q - (jj - 1) * kb + 1
                        if ii >= jj && jj > j
                            sh[SDS._packed(ii, jj, 64)] -= LDL ?
                                sh[SDS._packed(ii, j, 64)] * pvt * sh[SDS._packed(jj, j, 64)] :
                                sh[SDS._packed(ii, j, 64)] * sh[SDS._packed(jj, j, 64)]
                        end
                    end
                end
                @synchronize
            end
            for q in (li - 1):WG:(kb * kb - 1)
                jj = q ÷ kb + 1
                ii = q - (jj - 1) * kb + 1
                ii >= jj && (factor[p0 + (c0 + jj - 1) * f + c0 + ii - 1] =
                                 (LDL && ii == jj) ? one(TT) : sh[SDS._packed(ii, jj, 64)])
            end
            li == 1 && (k == 1 || st[1] != Int32(0)) && (info[v] = st[1])
            @synchronize
            Threads.atomic_fence()      # ALL threads: each wave drains its own stores (gfx9)
            @synchronize
            if li == 1
                Atomix.@atomic wpdone2[sl] += Int32(1)
            end
        elseif r == ROLE_WTRSM
            v = Int(ra[G]); ch = Int(rb[G]); k = Int(rc[G])
            sl = Int(wbase[v]) + k
            if li == 1
                _spin_until_eq!(wpdone2, sl, Int32(1), fscr, scap)
            end
            @synchronize
            Threads.atomic_fence()
            f = Int(front_nrows[v]); w = Int(front_ncols[v])
            c0 = (k - 1) * 64
            kb = min(64, w - c0)
            p0 = Int(front_ptr[v])
            for q in li:WG:(kb * (kb + 1) ÷ 2)
                j = 1
                acc = kb
                qq = q
                while qq > acc
                    qq -= acc
                    acc -= 1
                    j += 1
                end
                i = j + qq - 1
                # LDL: the factor diagonal is unit; restore d in the same loop (a second
                # loop would race this one on the diagonal slots)
                sh[SDS._packed(i, j, 64)] = (LDL && i == j) ? TT(d[Int(super_ptr[v]) - 1 + c0 + j]) :
                                            factor[p0 + (c0 + j - 1) * f + c0 + i - 1]
            end
            @synchronize
            if info[v] == Int32(0)
                i = c0 + kb + (ch - 1) * WG + li
                if i <= f
                    for kk in 1:kb
                        x = factor[p0 + (c0 + kk - 1) * f + i - 1]
                        for j in 1:(kk - 1)
                            lij = factor[p0 + (c0 + j - 1) * f + i - 1]
                            x -= LDL ? lij * (real(sh[SDS._packed(j, j, 64)]) * sh[SDS._packed(kk, j, 64)]) :
                                       lij * sh[SDS._packed(kk, j, 64)]
                        end
                        factor[p0 + (c0 + kk - 1) * f + i - 1] = x / sh[SDS._packed(kk, kk, 64)]
                    end
                end
            end
            @synchronize
            Threads.atomic_fence()      # ALL threads: each wave drains its own stores (gfx9)
            @synchronize
            if li == 1
                Atomix.@atomic wtrsml[sl] += Int32(-1)
            end
        elseif r == ROLE_WTILE
            v = Int(ra[G]); tii = Int(rb[G])
            k = Int(rc[G]) ÷ 1024; tjj = Int(rc[G]) % 1024
            sl = Int(wbase[v]) + k
            if li == 1
                _spin_until_eq!(wtrsml, sl, Int32(0), fscr, scap)
            end
            @synchronize
            Threads.atomic_fence()
            f = Int(front_nrows[v]); w = Int(front_ncols[v])
            m = f - w
            c0 = (k - 1) * 64
            kb = min(64, w - c0)
            p0 = Int(front_ptr[v])
            rb0 = c0 + kb
            i0 = rb0 + (tii - 1) * TILE
            j0 = rb0 + (tjj - 1) * TILE
            ok = info[v] == Int32(0)
            acc1 = zero(TT); acc2 = zero(TT); acc3 = zero(TT); acc4 = zero(TT)
            acc5 = zero(TT); acc6 = zero(TT); acc7 = zero(TT); acc8 = zero(TT)
            kk0 = 0
            while kk0 < kb
                kbk = min(32, kb - kk0)
                if LDL                              # stage the block's pivots; fold into Ai only
                    if ok
                        g0 = Int(super_ptr[v]) - 1 + c0
                        for c in li:WG:kbk
                            sh[2048 + c] = TT(d[g0 + kk0 + c])
                        end
                    end
                    @synchronize
                end
                if ok
                    for q in (li - 1):WG:(TILE * kbk - 1)
                        rr = q % TILE + 1
                        cc = q ÷ TILE + 1
                        row = i0 + rr
                        sh[rr + (cc - 1) * TILE] = row <= f ?
                            (LDL ? factor[p0 + (c0 + kk0 + cc - 1) * f + row - 1] * sh[2048 + cc] :
                                   factor[p0 + (c0 + kk0 + cc - 1) * f + row - 1]) : zero(TT)
                        row2 = j0 + rr
                        sh[1024 + rr + (cc - 1) * TILE] =
                            row2 <= f ? factor[p0 + (c0 + kk0 + cc - 1) * f + row2 - 1] : zero(TT)
                    end
                end
                @synchronize
                if ok
                    for (slot, q) in enumerate((li - 1):WG:(TILE * TILE - 1))
                        rr = q % TILE + 1
                        cc = q ÷ TILE + 1
                        a = zero(TT)
                        for kk in 1:kbk
                            a += sh[rr + (kk - 1) * TILE] * sh[1024 + cc + (kk - 1) * TILE]
                        end
                        slot == 1 && (acc1 += a); slot == 2 && (acc2 += a)
                        slot == 3 && (acc3 += a); slot == 4 && (acc4 += a)
                        slot == 5 && (acc5 += a); slot == 6 && (acc6 += a)
                        slot == 7 && (acc7 += a); slot == 8 && (acc8 += a)
                    end
                end
                @synchronize
                kk0 += 32
            end
            if ok
                c0cb = Int(cb2[v])
                for (slot, q) in enumerate((li - 1):WG:(TILE * TILE - 1))
                    rr = q % TILE + 1
                    cc = q ÷ TILE + 1
                    ii = i0 + rr
                    jj = j0 + cc
                    if ii <= f && ii >= jj
                        a = slot == 1 ? acc1 : slot == 2 ? acc2 : slot == 3 ? acc3 : slot == 4 ? acc4 :
                            slot == 5 ? acc5 : slot == 6 ? acc6 : slot == 7 ? acc7 : acc8
                        if jj <= w
                            Atomix.@atomic factor[p0 + (jj - 1) * f + ii - 1] -= a
                        elseif c0cb > 0
                            Atomix.@atomic pcb[c0cb + SDS._packed(ii - w, jj - w, m) - 1] -= a
                        end
                    end
                end
            end
            @synchronize
            Threads.atomic_fence()      # ALL threads: each wave drains its own stores (gfx9)
            @synchronize
            if li == 1
                if j0 < w
                    Atomix.@atomic wupd[Int(wbase[v]) + j0 ÷ 64 + 1] += Int32(-1)
                end
                if j0 + TILE > w
                    Atomix.@atomic wcbl[v] += Int32(-1)
                end
            end
        elseif r == ROLE_WSIG
            v = Int(ra[G]); K = Int(rb[G])
            sl = Int(wbase[v]) + K
            if li == 1
                _spin_until_eq!(wpdone2, sl, Int32(1), fscr, scap)
                _spin_until_eq!(wtrsml, sl, Int32(0), fscr, scap)
                _spin_until_eq!(wcbl, v, Int32(0), fscr, scap)
                Threads.atomic_fence()
                Atomix.@atomic cb_done[v] += Int32(1)
                pv = Int(snparent[v])
                pv > 0 && (Atomix.@atomic arrived[pv] += Int32(1))
            end
        end
    end
end

# ---------------------------------------------------------------------------
# host-wide LDL path: the Cholesky vendor sequence (potrf/trsm/herk) with the
# diagonal blocks factored by a small kernel (no vendor no-pivot LDL exists),
# the SAME vendor TRSM with the unit-triangular flag, broadcast column scalings
# for D, and the CB through the SAME vendor GEMM against a D-scaled scratch.

# one workgroup: no-pivot LDL of the kb×kb diagonal block at (c0b+1, c0b+1) of the
# f×w panel at p0; unit-L written back (diag = 1), D to d[g0 + c0b + j]
@kernel function wide_ldl_panel_kernel!(factor, d, info, v, p0, f, c0b, kb, g0, ::Val{WG}) where {WG}
    li = @index(Local, Linear)
    @uniform TT = eltype(factor)
    sh = @localmem TT (NL64,)
    st = @localmem Int32 (1,)
    pv = @localmem TT (1,)
    @inbounds begin
        for q in li:WG:(kb * (kb + 1) ÷ 2)
            j = 1
            acc = kb
            qq = q
            while qq > acc
                qq -= acc
                acc -= 1
                j += 1
            end
            i = j + qq - 1
            sh[SDS._packed(i, j, 64)] = factor[p0 + (c0b + j - 1) * f + c0b + i - 1]
        end
        li == 1 && (st[1] = Int32(0))
    end
    @synchronize
    for j in 1:kb
        @inbounds begin
            if li == 1
                dk = real(sh[SDS._packed(j, j, 64)])
                if st[1] == Int32(0) && !iszero(dk)
                    pv[1] = dk
                    d[g0 + c0b + j] = dk
                elseif st[1] == Int32(0)
                    st[1] = Int32(c0b + j)
                    pv[1] = one(TT)
                end
            end
        end
        @synchronize
        @inbounds begin
            pvt = pv[1]
            if st[1] == Int32(0)
                rpv = inv(pvt)
                for i in (j + li):WG:kb
                    sh[SDS._packed(i, j, 64)] *= rpv
                end
            end
        end
        @synchronize
        @inbounds begin
            pvt = pv[1]
            if st[1] == Int32(0)
                for q in (li - 1):WG:(kb * kb - 1)
                    jj = q ÷ kb + 1
                    ii = q - (jj - 1) * kb + 1
                    if ii >= jj && jj > j
                        sh[SDS._packed(ii, jj, 64)] -= sh[SDS._packed(ii, j, 64)] * pvt * sh[SDS._packed(jj, j, 64)]
                    end
                end
            end
        end
        @synchronize
    end
    @inbounds begin
        for q in (li - 1):WG:(kb * kb - 1)
            jj = q ÷ kb + 1
            ii = q - (jj - 1) * kb + 1
            ii >= jj && (factor[p0 + (c0b + jj - 1) * f + c0b + ii - 1] =
                             ii == jj ? one(TT) : sh[SDS._packed(ii, jj, 64)])
        end
        li == 1 && st[1] != Int32(0) && info[v] == Int32(0) && (info[v] = st[1])
    end
end

@kernel function zero_upper_kernel!(factor, p0, f, w)
    q = @index(Global, Linear)
    @inbounds begin
        j = (q - 1) ÷ w + 1
        i = q - (j - 1) * w
        i < j && (factor[p0 + (j - 1) * f + i - 1] = zero(eltype(factor)))
    end
end

# the host-wide front for LDL: blocked right-looking over 64-column panels, then one
# D-scaled GEMM for the contribution block (the Cholesky path's trsm/herk shape)
function _factor_panel_c_ldl!(N, S, fp, v, p, backend)
    T = eltype(N.factor)
    f = Int(S.schedule.rows[v])
    w = Int(S.schedule.width[v])
    p0 = Int(S.layout.panel_ptr[v])
    c0 = fp.cb2_h[v]
    g0 = fp.sptr_h[v] - 1
    m = f - w
    gi = SDS.select_impl(:gemm, N.factor, :auto)
    P = reshape(view(N.factor, p0:(p0 + f * w - 1)), f, w)
    for k in 1:cld(w, 64)
        c0b = (k - 1) * 64
        kb = min(64, w - c0b)
        wide_ldl_panel_kernel!(backend, 128)(N.factor, N.d, N.info, v, p0, f, c0b, kb, g0, Val(128);
                                             ndrange = 128)
        r1 = c0b + kb + 1
        r1 <= f || continue
        Lblk = view(P, (c0b + 1):(c0b + kb), (c0b + 1):(c0b + kb))
        R = view(P, r1:f, (c0b + 1):(c0b + kb))                     # A L1⁻ᵀ = W = L D
        SDS._trsm_impl!(p.trsm, 'R', 'L', 'C', 'U', one(T), Lblk, R)
        dvb = transpose(view(N.d, (g0 + c0b + 1):(g0 + c0b + kb)))
        if r1 <= w                                                  # trailing panel columns
            Wscr = reshape(view(fp.ldlw, 1:((f - r1 + 1) * kb)), f - r1 + 1, kb)
            Wscr .= R                                               # keep W; R becomes final L
            R ./= dvb
            Lc = view(P, r1:w, (c0b + 1):(c0b + kb))
            Ctr = view(P, r1:f, r1:w)
            SDS._gemm_impl!(gi, 'N', 'T', -one(T), Wscr, Lc, one(T), Ctr)
        else
            R ./= dvb
        end
    end
    zero_upper_kernel!(backend, 128)(N.factor, p0, f, w; ndrange = w * w)
    (m > 0 && c0 > 0) || return nothing
    L21 = view(P, (w + 1):f, 1:w)
    W21 = reshape(view(fp.ldlw, 1:(m * w)), m, w)
    W21 .= L21 .* transpose(view(N.d, (g0 + 1):(g0 + w)))
    C = reshape(view(N.work, 1:(m * m)), m, m)
    SDS._gemm_impl!(gi, 'N', 'T', -one(T), W21, L21, zero(T), C)
    SDS.pack_add!(fp.pcb, c0, N.work, m)
    return nothing
end

@kernel function panel_zero_kernel!(factor, @Const(list), @Const(front_ptr), @Const(front_nrows),
                                    @Const(front_ncols), ::Val{WG}) where {WG}
    li = @index(Local, Linear)
    G = @index(Group, Linear)
    @inbounds begin
        v = Int(list[G])
        f = Int(front_nrows[v])
        w = Int(front_ncols[v])
        p0 = Int(front_ptr[v])
        for q in (li - 1):WG:(f * w - 1)
            factor[p0 + q] = zero(eltype(factor))
        end
    end
end

@kernel function waiter_kernel!(ea_left, v, fscr_unused)
    li = @index(Local, Linear)
    if li == 1
        pause = min(Int32(4), scap)
        @inbounds while (Atomix.@atomic ea_left[v] += Int32(0)) != Int32(0)
            z = 0.0
            for _ in 1:pause
                z += 1.0
            end
            z < 0 && (ea_left[v] = Int32(0))
            pause = min(pause << 1, scap)
        end
    end
end

@kernel function signal_kernel!(cb_done, arrived, @Const(snparent), v)
    li = @index(Local, Linear)
    @inbounds if li == 1
        Threads.atomic_fence()
        Atomix.@atomic cb_done[v] += Int32(1)
        pv = Int(snparent[v])
        pv > 0 && (Atomix.@atomic arrived[pv] += Int32(1))
    end
end

# subtree launcher with a custom (sorted) list, local memory sized by eltype
function launch_subtrees_sorted!(N, S, nzval, list, lb, backend, ldl::Bool = false)
    SDS._with_local_bytes(Int(lb)) do LBv
        _sorted_subtree_launch(N, S, nzval, list, Val(ldl), LBv, Val(SDS.SUBTREE_WORKGROUP), backend)
    end
end
function _sorted_subtree_launch(N::SDS.Numeric{T}, S, nzval, list, ::Val{LDL}, ::Val{LB}, ::Val{WG},
                                backend) where {T, LDL, LB, WG}
    bm = SDS.batch_map(N; first = 1)
    if LDL
        # the schedule sized the "S" budget classes with the LDLT reserve: match it
        k! = subtree_ldl_kernel!(backend, WG)
        k!(N.factor, N.stack, N.info, N.d, N.piv, N.pivot_kind, nzval, S.amap, S.amap_ptr, S.amap_src,
           list, bm, S.subtree_ptr, S.subtree_nodes, S.super_ptr, S.front_ptr, S.front_nrows, S.front_ncols,
           S.cb_ptr, S.local_front, S.local_cb, S.child_ptr, S.child_list, S.relind_ptr, S.relind,
           Val((LB - SDS.SUBTREE_LOCAL_RESERVE_LDLT) ÷ sizeof(T)), Val(WG);
           ndrange = WG * length(list) * bm.nact)
    else
        k! = SDS.subtree_cholesky_kernel!(backend, WG)
        k!(N.factor, N.stack, N.info, nzval, S.amap, S.amap_ptr, S.amap_src, list, bm, S.subtree_ptr,
           S.subtree_nodes, S.front_ptr, S.front_nrows, S.front_ncols, S.cb_ptr, S.local_front, S.local_cb,
           S.child_ptr, S.child_list, S.relind_ptr, S.relind,
           Val((LB - SDS.SUBTREE_LOCAL_RESERVE) ÷ sizeof(T)), Val(WG); ndrange = WG * length(list) * bm.nact)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# host: the topologically ordered role list + counter bases + stream-2 program
function build_fused_plan(s; ldl::Bool = false, inkernel_wides::Union{Nothing, Bool} = nothing,
                          merge_segments::Bool = false, wgf::Int = WGF, scap::Integer = 128)
    S = s.symbolic
    plan = s.numeric.plan
    ldl && (eltype(s.numeric.factor) <: Real || error("fused LDL mode: real element types only"))
    # LDL defaults to in-kernel wides: the host-wide LDL path (_factor_panel_c_ldl!) is
    # correct but needs ~5 launches per 64-column block where Cholesky's potrf is one
    # vendor call -- measured SLOWER on the GV100 (55.5 vs 38.6 ms tuned). Pass
    # inkernel_wides = false explicitly to use it.
    inkernel_wides = inkernel_wides === nothing ? ldl : inkernel_wides
    nodes = S.schedule.group_nodes
    rows_h = S.schedule.rows
    width_h = S.schedule.width
    cbp = Array(S.cb_ptr)
    chp = Array(S.child_ptr)
    chl = Array(S.child_list)
    stp = Array(S.subtree_ptr)
    stn = Array(S.subtree_nodes)
    ns = length(width_h)

    covered = falses(ns)                            # fronts factored by the subtree launches
    for q in eachindex(stn)
        covered[stn[q]] = true
    end

    role = Int8[]; ra = Int32[]; rb = Int32[]; rc = Int32[]
    wides = Int[]
    ea_left = zeros(Int32, ns)
    trsm_left = zeros(Int32, ns)
    tiles_left = zeros(Int32, ns)
    arrived_base = zeros(Int32, ns)
    cb_done_base = zeros(Int32, ns)
    for v in 1:ns
        covered[v] && (cb_done_base[v] = 1)
    end

    # proposal B: a child writes its CB through into the parent when the parent is a
    # pre-assembled C-group front and the child is its ONLY CB child (any kind)
    cset = Set{Int}()
    for k in eachindex(plan.group_first)
        plan.group_width[k] > 0 && continue
        for q in plan.group_first[k]:plan.group_last[k]
            push!(cset, Int(nodes[q]))
        end
    end
    wt_h = zeros(Int32, ns)
    nwt = 0; ewt = 0
    for v in 1:ns
        (covered[v] || cbp[v] == 0) && continue
        width_h[v] <= 64 || continue             # wide children have no tiles to settle the debt
        m = rows_h[v] - width_h[v]
        m > SYRK_IN || continue
        pv = Int(S.partition.snparent[v])
        (pv > 0 && pv in cset) || continue
        count(c -> cbp[c] > 0, chl[chp[pv]:(chp[pv + 1] - 1)]) == 1 || continue
        wt_h[v] = Int32(pv)
        nwt += 1; ewt += m * (m + 1) ÷ 2
    end
    println("write-through: ", nwt, " fronts, ", round(ewt / 1e6; digits = 2), "M CB entries"); flush(stdout)

    emit_ea(v) = for kc in chp[v]:(chp[v + 1] - 1)
        c = chl[kc]
        cbp[c] > 0 || continue
        if wt_h[c] == v
            ea_left[v] += 1                      # settled by the child's last write-through tile
            continue
        end
        mc = rows_h[c] - width_h[c]
        for ch in 1:cld(mc * mc, 16384)
            push!(role, ROLE_EA); push!(ra, Int32(v)); push!(rb, Int32(c)); push!(rc, Int32(ch))
            ea_left[v] += 1
        end
    end
    emit_split(v) = begin
        m = rows_h[v] - width_h[v]
        m > SYRK_IN || return nothing
        for ch in 1:cld(m, wgf)
            push!(role, ROLE_TRSM); push!(ra, Int32(v)); push!(rb, Int32(ch)); push!(rc, Int32(0))
            trsm_left[v] += 1
        end
        if cbp[v] > 0
            T = cld(m, TILE)
            for tjj in 1:T, tii in tjj:T
                push!(role, ROLE_TILE); push!(ra, Int32(v)); push!(rb, Int32(tii)); push!(rc, Int32(tjj))
                tiles_left[v] += 1
            end
        end
        return nothing
    end

    # in-kernel wides: per-(front, panel) counter slots; host arithmetic MIRRORS the
    # kernel's decrement conditions exactly (j0 < w -> wupd[kj]; j0 + TILE > w -> wcb[v])
    wbase_h = zeros(Int32, ns)
    wupd_b = Int32[]; wtl_b = Int32[]
    wcb_b = zeros(Int32, ns)
    nwslots = 0
    emit_wide(v) = begin
        f = rows_h[v]; w = width_h[v]
        K = cld(w, 64)
        wbase_h[v] = Int32(nwslots)
        append!(wupd_b, zeros(Int32, K)); append!(wtl_b, zeros(Int32, K))
        for k in 1:K
            push!(role, ROLE_WPANEL); push!(ra, Int32(v)); push!(rb, Int32(k)); push!(rc, Int32(0))
            c0 = (k - 1) * 64
            kb = min(64, w - c0)
            nbelow = f - (c0 + kb)
            nch = cld(nbelow, wgf)
            for ch in 1:nch
                push!(role, ROLE_WTRSM); push!(ra, Int32(v)); push!(rb, Int32(ch)); push!(rc, Int32(k))
            end
            wtl_b[nwslots + k] = Int32(nch)
            nt = cld(nbelow, TILE)
            for tjj in 1:nt, tii in tjj:nt
                push!(role, ROLE_WTILE); push!(ra, Int32(v)); push!(rb, Int32(tii))
                push!(rc, Int32(k * 1024 + tjj))
                j0 = c0 + kb + (tjj - 1) * TILE
                j0 < w && (wupd_b[nwslots + j0 ÷ 64 + 1] += Int32(1))
                j0 + TILE > w && (wcb_b[v] += Int32(1))
            end
        end
        push!(role, ROLE_WSIG); push!(ra, Int32(v)); push!(rb, Int32(K)); push!(rc, Int32(0))
        nwslots += K
        return nothing
    end

    segments = NTuple{3, Any}[]                  # (base, count, wides-at-boundary)
    seg_base = 0
    for k in eachindex(plan.group_first)
        a, b = plan.group_first[k], plan.group_last[k]
        if plan.group_width[k] > 0
            for q in a:b
                v = nodes[q]
                push!(role, ROLE_PANEL_B); push!(ra, Int32(v)); push!(rb, Int32(0)); push!(rc, Int32(0))
            end
            for q in a:b
                emit_split(nodes[q])
            end
        else
            gwides = Int[]
            for q in a:b
                emit_ea(nodes[q])
            end
            for q in a:b
                v = nodes[q]
                if width_h[v] <= 64
                    push!(role, ROLE_PANEL_C); push!(ra, Int32(v)); push!(rb, Int32(0)); push!(rc, Int32(0))
                else
                    push!(gwides, v)
                end
            end
            for q in a:b
                width_h[nodes[q]] <= 64 && emit_split(nodes[q])
            end
            if !isempty(gwides)
                if inkernel_wides
                    foreach(emit_wide, gwides)
                    gwides = Int[]
                end
                if !(inkernel_wides && merge_segments)   # merged: one launch, no split
                    push!(segments, (seg_base, length(role) - seg_base, gwides))
                    seg_base = length(role)
                end
                append!(wides, gwides)
            end
        end
    end
    push!(segments, (seg_base, length(role) - seg_base, Int[]))

    cb2_h = zeros(Int64, ns)
    off = 1
    for v in 1:ns
        (covered[v] || cbp[v] == 0) && continue
        m = rows_h[v] - width_h[v]
        cb2_h[v] = off
        off += m * (m + 1) ÷ 2
    end
    czero = Int32[]                              # C-group fronts: panel zeroed by a prelaunch kernel
    for k in eachindex(plan.group_first)
        plan.group_width[k] > 0 && continue
        append!(czero, Int32.(nodes[plan.group_first[k]:plan.group_last[k]]))
    end
    nchild = Int32[chp[v + 1] - chp[v] for v in 1:ns]
    for v in 1:ns
        cnt = 0
        for kc in chp[v]:(chp[v + 1] - 1)
            covered[chl[kc]] && (cnt += 1)
        end
        arrived_base[v] = Int32(cnt)
    end

    stp2 = stp
    ssub = Vector{Any}()
    for k in eachindex(plan.sub_first)
        ids = [nodes[q] for q in plan.sub_first[k]:plan.sub_last[k]]
        sort!(ids; by = e -> stp2[e + 1] - stp2[e], rev = true)
        push!(ssub, DVEC(ids))
    end

    println("segments: ", length(segments)); flush(stdout)
    return (; role = DVEC(role), ra = DVEC(ra), rb = DVEC(rb), rc = DVEC(rc),
            nblocks = length(role), wides, ssub, segments,
            cb2 = DVEC(Int32.(cb2_h)), cb2_h, pcb = KA.zeros(BACKEND[], Float64, off - 1),
            czero = DVEC(czero), wt = DVEC(wt_h),
            nchild = DVEC(nchild),
            arrived_base = DVEC(arrived_base), ea_base = DVEC(ea_left),
            trsm_base = DVEC(trsm_left), tiles_base = DVEC(tiles_left),
            cb_base = DVEC(cb_done_base),
            arrived = KA.zeros(BACKEND[], Int32, ns), ea = KA.zeros(BACKEND[], Int32, ns),
            trsm = KA.zeros(BACKEND[], Int32, ns), tiles = KA.zeros(BACKEND[], Int32, ns),
            pdone = KA.zeros(BACKEND[], Int32, ns), cbd = KA.zeros(BACKEND[], Int32, ns),
            wbase = DVEC(wbase_h), wupd_b = DVEC(wupd_b), wtl_b = DVEC(wtl_b),
            wcb_b = DVEC(wcb_b), wupd = KA.zeros(BACKEND[], Int32, max(nwslots, 1)),
            wtl = KA.zeros(BACKEND[], Int32, max(nwslots, 1)), wpd = KA.zeros(BACKEND[], Int32, max(nwslots, 1)),
            wcb = KA.zeros(BACKEND[], Int32, ns), wgf, scap = Int32(scap), ldl,
            sptr_h = Array(S.super_ptr),
            ldlw = KA.zeros(BACKEND[], Float64,
                            ldl ? maximum(vcat([1], [max(rows_h[v] * 64,
                                                         (rows_h[v] - width_h[v]) * width_h[v])
                                                for v in wides])) : 1),
            s2 = nothing)
end

function refact_fused!(s, nzval, fp; nstreams::Int = 1)
    N, S = s.numeric, s.symbolic
    plan = N.plan
    backend = KA.get_backend(N.factor)
    p = SDS._front_impls(N, S, :auto)
    copyto!(fp.arrived, fp.arrived_base)
    copyto!(fp.ea, fp.ea_base)
    copyto!(fp.trsm, fp.trsm_base)
    copyto!(fp.tiles, fp.tiles_base)
    copyto!(fp.cbd, fp.cb_base)
    fill!(fp.pdone, Int32(0))
    length(fp.wupd_b) > 0 && copyto!(fp.wupd, fp.wupd_b)
    length(fp.wtl_b) > 0 && copyto!(fp.wtl, fp.wtl_b)
    fill!(fp.wpd, Int32(0))
    copyto!(fp.wcb, fp.wcb_b)
    for k in eachindex(plan.sub_first)
        launch_subtrees_sorted!(N, S, nzval, fp.ssub[k], plan.sub_local[k], backend, fp.ldl)
    end
    nodes = S.schedule.group_nodes
    fill!(fp.pcb, 0.0)
    isempty(fp.czero) ||
        panel_zero_kernel!(backend, WGF)(N.factor, fp.czero, S.front_ptr, S.front_nrows, S.front_ncols,
                                         Val(WGF); ndrange = WGF * length(fp.czero))
    for k in eachindex(plan.group_first)               # scatter A into every C-group front (panel only)
        plan.group_width[k] > 0 && continue
        a, b = plan.group_first[k], plan.group_last[k]
        SDS.scatter_A!(N, S, nzval, a, b - a + 1)
    end
    for (base, count, gwides) in fp.segments
        count > 0 &&
            fused_fact_kernel!(backend, fp.wgf)(N.factor, N.stack, fp.pcb, N.info, nzval, S.amap, S.amap_ptr,
                                             S.amap_src, fp.cb2, fp.wt, fp.role, fp.ra, fp.rb, fp.rc, fp.nchild,
                                             fp.arrived, fp.ea, fp.pdone, fp.trsm, fp.tiles, fp.cbd,
                                             S.snparent, S.front_ptr, S.front_nrows, S.front_ncols,
                                             S.cb_ptr, S.child_ptr, S.child_list, S.relind_ptr, S.relind,
                                             Int32(base), fp.wbase, fp.wupd, fp.wpd, fp.wtl, fp.wcb,
                                             fp.scap, N.d, S.super_ptr, Val(fp.ldl), Val(fp.wgf);
                                             ndrange = fp.wgf * count)
        if true
            for v in gwides
                if fp.ldl
                    _factor_panel_c_ldl!(N, S, fp, v, p, backend)
                else
                    SDS._factor_panel_c!(N.factor, fp.pcb, N.work, N.info, v, S.layout.panel_ptr[v],
                                         S.schedule.rows[v], S.schedule.width[v], fp.cb2_h[v], p)
                end
                signal_kernel!(backend, 32)(fp.cbd, fp.arrived, S.snparent, v; ndrange = 32)
            end
        elseif false
            nothing
        end
    end
    SDS.assemble_schur!(N, S, nzval)
    fp.ldl || SDS.cholesky_stats!(N, S)        # LDL: inertia comes from the signs of d
    return SDS._numeric_info!(N, S, true)
end


end # module
