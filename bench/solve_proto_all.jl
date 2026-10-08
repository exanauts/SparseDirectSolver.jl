# Prototype (PERFORMANCE.md experiments 5/6, issues #82/#25): partitioned-inverse
# solve with fused dependency-counter sweeps, CUDA only, SPD, 1 RHS, nbatch = 1.
#
# After factorization, invert every front's L11 (w <= MW) into a side buffer;
# each front's TRSV becomes one GEMV. The sweeps then run as:
#   fwd:  ONE fused kernel (regime-A subtrees as leading blocks + every front of
#         width <= 128 below the first wide front, etree child-counters with
#         spin/threadfence signaling, atomic L21 scatter)
#       + ONE single-workgroup "chain" kernel walking the near-serial tall
#         fronts (w <= 256) at the top with plain block barriers
#       + the stock vendor dense path for the remaining wide roots.
#   bwd:  the same in reverse (chain waits dispatch parents-first: fronts MUST
#         get reverse-topological block ids or the kernel deadlocks).
# Measured on a Quadro GV100, pglib_opf_case78484_epigrids condensed KKT
# (n = 674562): stock sweeps 18.0 ms -> 6.3 ms; cuDSS 3.8 ms. See the PR body
# for the full matrix table (wide-front matrices regress; pick per schedule).
# Constraints: CHWG >= NSTRIP * MW (strip indexing); NSTRIP/MW compile-time.
#
#   CUDA_VISIBLE_DEVICES=1 julia +1.13 --project=bench bench/solve_proto_78k.jl

using SparseDirectSolver, SparseArrays, LinearAlgebra, Random, Statistics, Printf
using CUDA, CUDA.CUSPARSE
using Metis
const SDS = SparseDirectSolver
const KA = SDS.KernelAbstractions
const Atomix = SDS.Atomix
using .KA: @kernel, @index, @localmem, @synchronize, @Const, @uniform

include(joinpath(@__DIR__, "matrices.jl"))
using .BenchMatrices: read_mtx

const WG = 32
const MW = 256  # max front width on the inverse path
const WGF = 128 # workgroup size of the fused kernels
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
    inv11 = CUDA.zeros(Float64, p - 1)
    inv_ptr = CuArray(inv_ptr_h)
    ab = CuArray(Int32.(findall(v -> width[v] <= MW, 1:ns)))
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
            CUDA.threadfence()
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
        CUDA.threadfence()
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
            CUDA.threadfence()
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
        CUDA.threadfence()
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
        CUDA.threadfence()
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
            CUDA.threadfence()
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
function build_fused(s; kstart::Int = 1, wcap::Int = MW)
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
    live_root = falses(ns)                   # roots of subtrees merged into this launch (wcap covers tier 1 only)
    if wcap <= 128
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
    return (; i1, i2, flist = CuArray(flist_h), arrived = CUDA.zeros(Int32, ns),
            arrived_base = CuArray(arrived_base_h), done = CUDA.zeros(Int32, ns),
            done_base = CuArray(done_base_h), nchild = CuArray(nchild_h), nfused = length(flist_h))
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
    debug && (CUDA.synchronize(); println("  algo2: forward done"); flush(stdout))
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
    debug && (CUDA.synchronize(); println("  algo2: backward done"); flush(stdout))
    SDS.unpermute_solution!(x, ws.Y, S.perm; transposed = false, bm)
    return x
end

short_name(nm) = begin
    m = match(r"(gpu_)?_?([A-Za-z0-9_!]+)", split(nm, '(')[1])
    m === nothing ? nm : m.captures[2]
end

function byname_profile(f, label)
    f(); f(); CUDA.synchronize()
    r = CUDA.@profile trace = true f()
    d = r.device
    agg = Dict{String, Tuple{Int, Float64}}()
    for i in eachindex(d.id)
        d.grid[i] === missing && continue
        knm = short_name(d.name[i])
        c, t = get(agg, knm, (0, 0.0))
        agg[knm] = (c + 1, t + (d.stop[i] - d.start[i]) * 1e3)
    end
    println("== ", label, " ==")
    for (knm, (c, t)) in sort(collect(agg); by = x -> -x[2][2])
        @printf "%-38s %5d kernels  %8.3f ms\n" knm c t
    end
    flush(stdout)
end

# ---------------------------------------------------------------------------
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
using .BenchMatrices: bench_matrices

function bench_one(M; cw = 128)
    A = SparseMatrixCSC{Float64, Int}(M.A)
    n = size(A, 1)
    Random.seed!(666)
    bh = rand(n)
    Lh = tril(A)
    println("==== ", M.name, "  n = ", n, " ===="); flush(stdout)

println("---- regime_c_width = ", cw, " ----")
bd = CuArray(bh); xd = similar(bd)
Ad = CuSparseMatrixCSR(Lh)
s = DirectSolver(Ad, "SPD", 'L')
s.options.regime_c_width = cw
ex(ph) = SDS.execute!(ph, s, xd, bd; asynchronous = false)
print("analysis… "); @time ex("analysis")
ex("factorization")
ex("solve")                      # allocates ws, warms stock path
x_stock = Array(xd)
println("stock relres      ", norm(bh - A * x_stock) / norm(bh))

inv11, inv_ptr, invert!, bytes = build_inverse(s)
@printf "inverse buffer     %.1f MB\n" bytes / 1e6
invert!(); CUDA.synchronize()
tinv = median([(CUDA.@elapsed (invert!(); CUDA.synchronize())) for _ in 1:5])
@printf "invert pass        %.2f ms (per refactorization)\n" tinv * 1e3

algo1_solve!(xd, s, bd, inv11, inv_ptr); CUDA.synchronize()
x1 = Array(xd)
println("algo1 relres      ", norm(bh - A * x1) / norm(bh))
println("‖x_algo1 − x_stock‖/‖x‖ = ", norm(x1 - x_stock) / norm(x_stock))

for _ in 1:3
    algo1_solve!(xd, s, bd, inv11, inv_ptr)
end
CUDA.synchronize()
t1 = [(@elapsed (algo1_solve!(xd, s, bd, inv11, inv_ptr); CUDA.synchronize())) for _ in 1:20]
t0 = [(@elapsed ex("solve")) for _ in 1:20]
@printf "stock solve        %.2f ms (median of 20)\n" median(t0) * 1e3
@printf "algo1 solve        %.2f ms (median of 20)\n" median(t1) * 1e3
plan0 = s.workspace.plan
nodes0 = s.symbolic.schedule.group_nodes
stp = Array(s.symbolic.subtree_ptr)
sub_ids = Int32[]
for k in eachindex(plan0.kind)
    plan0.kind[k] == SDS.SOLVE_SUBTREES || continue
    append!(sub_ids, Int32.(nodes0[plan0.first[k]:plan0.last[k]]))
end
sort!(sub_ids; by = e -> stp[e + 1] - stp[e], rev = true)
SSUB[] = CuArray(sub_ids); NSUB[] = length(sub_ids)
fz1 = build_fused(s; kstart = 1, wcap = 128)
if fz1 === nothing
    println("(no fused range; skipping)")
    GC.gc(); CUDA.reclaim()
    return nothing
end
fz2 = build_fused(s; kstart = fz1.i2 + 1, wcap = 256)
segs = [(fz1, 64)]
fz2 === nothing || push!(segs, (fz2, 1024))
fz = (segs, true)
algo2_solve!(xd, s, bd, inv11, inv_ptr, fz; debug = true); CUDA.synchronize()
x2 = Array(xd)
println("algo2 relres      ", norm(bh - A * x2) / norm(bh))
println("‖x_algo2 − x_stock‖/‖x‖ = ", norm(x2 - x_stock) / norm(x_stock))
for _ in 1:3
    algo2_solve!(xd, s, bd, inv11, inv_ptr, fz)
end
CUDA.synchronize()
t2v = [(@elapsed (algo2_solve!(xd, s, bd, inv11, inv_ptr, fz); CUDA.synchronize())) for _ in 1:20]
@printf "algo2 solve        %.2f ms (median of 20)\n" median(t2v) * 1e3

GC.gc(); CUDA.reclaim()
return nothing
    GC.gc(); CUDA.reclaim()
    return nothing
end

for M in bench_matrices()
    "SPD" in M.structures || continue
    try
        bench_one(M)
    catch err
        println("ERROR on ", M.name, ": ", sprint(showerror, err)[1:min(end, 300)])
        flush(stdout)
        GC.gc(); CUDA.reclaim()
    end
end
println("ALL DONE")
