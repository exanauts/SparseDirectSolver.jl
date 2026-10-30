# The fused pipeline (PR #107 kernels + the October 2026 campaign), assembled
# as a backend-portable submodule: the fused factorization (Cholesky and
# no-pivot LDL^T) and the partitioned-inverse/fused solve, on CUDA and AMDGPU
# alike. The plans are built on the solver's backend and carry their own device
# state (no module globals). Consumed by the MadNLP package extension
# (ext/SparseDirectSolverMadNLPExt.jl); the tuning bundle `SDSTuning` /
# `default_tuning` lives here because it parameterizes these kernels, not the
# MadNLP glue.
module Fused

using LinearAlgebra
const SDS = Base.parentmodule(@__MODULE__)    # the enclosing SparseDirectSolver module
const KA = SDS.KernelAbstractions
const Atomix = SDS.Atomix
using .KA: @kernel, @index, @localmem, @synchronize, @Const, @uniform

"""
    SDSTuning

Every performance knob of the fused pipeline in one immutable bundle, so a
platform's tuning is a single value (see bench/ARCH-TUNING.md for how each
field was measured and the per-architecture champions). All fields are
runtime parameters — no environment variables, no recompilation.
"""
Base.@kwdef struct SDSTuning
    wides::Symbol = :auto                     # :auto | :host | :merged
    subtree_budgets::Vector{Int} = [16384]    # regime-A local-memory class(es)
    subtree_parallelism::Int = 16384
    regime_c_width::Int = 32
    regime_c_rows::Int = 64
    amalgamation::NamedTuple = (max_width = 48, zero_fraction = 0.25, min_width = 8)
    workgroup::Int = 128                      # fused-kernel workgroup size
    spin_cap::Int = 128                       # dependency-counter backoff cap
    # METIS knobs. nseps = 1 by default: first-touch reordering latency wins (owner call,
    # 2026-10-12); nseps = 4 is the 78k champion (-6% refactorization via lower fill, PR #124)
    # at ~2x the reordering time -- opt in through `sds_tuning` when factor time dominates.
    nd_nseps::Int = 1
    nd_seed::Int = 3
end

"""
    default_tuning(backend) -> SDSTuning

The measured per-architecture defaults — the single place they live.
2048-threads/SM NVIDIA parts (sm_70/80/90): host wides, 16K budgets.
1536-thread parts (sm_86/89, Blackwell workstation): merged wides, 8K budgets.
AMD (gfx906-measured): merged, 8K, workgroup 256, spin cap 16. A backend
without a CUDA resident-thread count (`max_threads_per_multiprocessor` returns
0: AMDGPU, oneAPI, Metal, the CPU backend, anything but CUDA) gets this AMD
profile.
"""
function default_tuning(backend)
    # NVIDIA parts report their resident-thread count through the CUDA extension
    threads = SDS.max_threads_per_multiprocessor(backend)
    if threads > 0
        return threads < 2048 ? SDSTuning(; wides = :merged, subtree_budgets = [8192]) :
                                SDSTuning(; wides = :host)
    end
    return SDSTuning(; wides = :merged, subtree_budgets = [8192], workgroup = 256, spin_cap = 16)
end

"""
    resolve_tuning(t::SDSTuning, backend) -> SDSTuning

`t` with `wides = :auto` replaced by the wide-front strategy of
`default_tuning(backend)`; any other `t` is returned as is.
"""
function resolve_tuning(t::SDSTuning, backend)
    t.wides === :auto || return t
    return SDSTuning(default_tuning(backend).wides, t.subtree_budgets, t.subtree_parallelism,
                     t.regime_c_width, t.regime_c_rows, t.amalgamation, t.workgroup, t.spin_cap,
                     t.nd_nseps, t.nd_seed)
end

# device copy of a host integer vector (empty vectors included)
_dvec(backend, v) = SDS._device_vec(backend, eltype(v), v)

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
    @uniform TT = eltype(Y)
    li = @index(Local, Linear)
    g = @index(Group, Linear)
    t = @localmem TT (MW,)
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
                acc = zero(TT)
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
                acc = zero(TT)
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
    @uniform TT = eltype(Y)
    li = @index(Local, Linear)
    g = @index(Group, Linear)
    t = @localmem TT (MW,)
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
                acc = zero(TT)
                col = p0 + (kk - 1) * f
                for i in (w + 1):f
                    acc += factor[col + i] * Y[rowval[rp + i]]
                end
                t[kk] = Y[c0 + kk] - acc
            end
            @synchronize
            # x = inv(L11)ᵀ t  (inv col-major: invᵀ[k, j] = inv11[(k-1)w + j], contiguous in j)
            for kk in li:WGV:w
                acc = zero(TT)
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
# per-front inverses of the L11 blocks (width <= MW) for the partitioned-inverse solve:
# returns the device buffer, its per-front offsets, a closure that (re)computes the
# inverses from the current factor, and the buffer size in bytes
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
    backend = KA.get_backend(s.numeric.factor)
    T = eltype(s.numeric.factor)
    inv11 = KA.zeros(backend, T, p - 1)
    inv_ptr = _dvec(backend, inv_ptr_h)
    ab = _dvec(backend, Int32.(findall(v -> width[v] <= MW, 1:ns)))
    invert! = () -> invert_l11_kernel!(backend, WG)(inv11, s.numeric.factor, inv_ptr, ab, S.front_ptr,
                                                   S.front_nrows, S.front_ncols, Val(WG);
                                                   ndrange = WG * length(ab))
    return inv11, inv_ptr, invert!, sizeof(T) * (p - 1)
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
    @uniform TT = eltype(Y)
    li = @index(Local, Linear)
    g = @index(Group, Linear)
    yb = @localmem TT (MW,)
    t = @localmem TT (MW,)
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
                acc = zero(TT)
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
                acc = zero(TT)
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
                z = Int32(0)
                for _ in 1:pause
                    z += Int32(1)
                end
                z < Int32(0) && (yb[1] = TT(z))        # keep the backoff loop alive
                pause = min(pause << 1, Int32(128))
            end
        end
        @synchronize
        Threads.atomic_fence()
        for kk in li:WGV:w                          # coherent read of the front's columns
            yb[kk] = Atomix.@atomic Y[c0 + kk] += zero(TT)
        end
        @synchronize
        for kk in li:WGV:w                          # t = inv(L11) * y
            acc = zero(TT)
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
            acc = zero(TT)
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
    @uniform TT = eltype(Y)
    li = @index(Local, Linear)
    g = @index(Group, Linear)
    xb = @localmem TT (WGV,)
    t = @localmem TT (MW,)
    t2 = @localmem TT (MW,)
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
                    z = Int32(0)
                    for _ in 1:pause
                        z += Int32(1)
                    end
                    z < Int32(0) && (t[1] = TT(z))
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
                acc = zero(TT)
                col = p0 + (kk - 1) * f
                for i in (w + 1):f
                    acc += factor[col + i] * Y[Int(rowval[rp + i])]
                end
                t2[kk] = Y[c0 + kk] - acc
            end
            @synchronize
            for kk in li:WGV:w
                acc = zero(TT)
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
                    z = Int32(0)
                    for _ in 1:pause
                        z += Int32(1)
                    end
                    z < Int32(0) && (t[1] = TT(z))
                    pause = min(pause << 1, Int32(128))
                end
            end
        end
        @synchronize
        Threads.atomic_fence()
        for kk in li:WGV:w
            t[kk] = zero(TT)
        end
        @synchronize
        i0 = w + 1
        while i0 <= f                               # chunked coherent gather of ancestor solutions
            ii = i0 + li - 1
            if ii <= f
                xb[li] = Atomix.@atomic Y[Int(rowval[rp + ii])] += zero(TT)
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
            acc = zero(TT)
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
    @uniform TT = eltype(Y)
    li = @index(Local, Linear)
    yb = @localmem TT (MW,)
    t = @localmem TT (MW,)
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
            acc = zero(TT)
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
            acc = zero(TT)
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
    @uniform TT = eltype(Y)
    li = @index(Local, Linear)
    t = @localmem TT (MW,)
    t2 = @localmem TT (MW,)
    tp = @localmem TT (MW * NSTRIP,)
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
            acc = zero(TT)
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
            acc = zero(TT)
            cb = ip + (kk2 - 1) * w
            for j in st2b:NSTRIP:w
                acc += inv11[cb + j] * t2[j]
            end
            tp[(st2b - 1) * MW + kk2] = acc
        end
        @synchronize
        for k2 in li:WGV:w
            acc = zero(TT)
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
    # the regime-A subtrees the merged launch walks itself, longest first
    sub_ids = Int32[]
    if merged
        for k in eachindex(plan.kind)
            plan.kind[k] == SDS.SOLVE_SUBTREES || continue
            append!(sub_ids, Int32.(nodes[plan.first[k]:plan.last[k]]))
        end
        sort!(sub_ids; by = e -> stp_h[e + 1] - stp_h[e], rev = true)
    end
    SDS._log(SDS.LOG_DEBUG, () -> "fused solve: plan entries $i1:$i2 of $(length(plan.kind)), " *
                                  "$(length(flist_h)) fronts, $(length(sub_ids)) subtrees")
    backend = KA.get_backend(s.numeric.factor)
    return (; i1, i2, flist = _dvec(backend, flist_h), arrived = KA.zeros(backend, Int32, ns),
            arrived_base = _dvec(backend, arrived_base_h), done = KA.zeros(backend, Int32, ns),
            done_base = _dvec(backend, done_base_h), nchild = _dvec(backend, nchild_h),
            nfused = length(flist_h), slist = _dvec(backend, sub_ids), nsub = length(sub_ids))
end

# one fused segment, forward sweep: `wg > 512` runs the single-workgroup chain kernel,
# `wg == 64` the dependency-counter sweep (the workgroup size the sweep is tuned for)
function run_fused!(fz, wg, Yv, s, inv11, inv_ptr, backend)
    S, N = s.symbolic, s.numeric
    if wg > 512
        chain_fwd_kernel!(backend, CHWG)(Yv, N.factor, inv11, inv_ptr, fz.flist, S.super_ptr, S.rowptr,
                                         S.rowval, S.front_ptr, S.front_nrows, S.front_ncols, Val(CHWG);
                                         ndrange = CHWG)
        return nothing
    end
    wg == 64 || throw(SDS.InvalidValueError("fused solve: workgroup size $wg; expected 64 or > 512 (chain)"))
    copyto!(fz.arrived, fz.arrived_base)
    fwd2_kernel!(backend, 64)(Yv, N.factor, inv11, inv_ptr, fz.flist, fz.slist, fz.nsub, S.subtree_ptr,
                              S.subtree_nodes, S.snparent, fz.arrived, fz.nchild,
                              S.super_ptr, S.rowptr, S.rowval, S.front_ptr, S.front_nrows, S.front_ncols,
                              Val(64); ndrange = 64 * (fz.nfused + fz.nsub))
    return nothing
end

# the backward counterpart of `run_fused!`
function run_fused_bwd!(fz, wg, Yv, s, inv11, inv_ptr, backend)
    S, N = s.symbolic, s.numeric
    if wg > 512
        chain_bwd_kernel!(backend, CHWG)(Yv, N.factor, inv11, inv_ptr, fz.flist, S.super_ptr, S.rowptr,
                                         S.rowval, S.front_ptr, S.front_nrows, S.front_ncols, Val(CHWG);
                                         ndrange = CHWG)
        return nothing
    end
    wg == 64 || throw(SDS.InvalidValueError("fused solve: workgroup size $wg; expected 64 or > 512 (chain)"))
    copyto!(fz.done, fz.done_base)
    bwd2_kernel!(backend, 64)(Yv, N.factor, inv11, inv_ptr, fz.flist, fz.slist, fz.nsub, S.subtree_ptr,
                              S.subtree_nodes, S.snparent, fz.done,
                              S.super_ptr, S.rowptr, S.rowval, S.front_ptr, S.front_nrows, S.front_ncols,
                              Val(64); ndrange = 64 * (fz.nfused + fz.nsub))
    return nothing
end

# the fused solve: `fz = (segments, covered)` with `segments` a vector of
# `(build_fused(...), wg)` pairs in plan order; plan entries outside the segments
# run through the level-batched kernels (`fwd1_kernel!`/`bwd1_kernel!`, vendor
# dense calls for fronts wider than MW)
function algo2_solve!(x, s, b, inv11, inv_ptr, fz)
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
    segs, _ = fz
    k = 1
    for (fzk, wg) in segs
        while k < fzk.i1
            fwd_entry(k); k += 1
        end
        run_fused!(fzk, wg, Yv, s, inv11, inv_ptr, backend)
        k = fzk.i2 + 1
    end
    while k <= length(plan.kind)
        fwd_entry(k); k += 1
    end
    if SDS._is_ldlt_structure(S.structure)      # LDL: apply D between the unit-triangular sweeps
        Yv ./= view(N.d, 1:length(Yv))
    end
    k = length(plan.kind)
    for (fzk, wg) in Iterators.reverse(segs)
        while k > fzk.i2
            bwd_entry(k); k -= 1
        end
        run_fused_bwd!(fzk, wg, Yv, s, inv11, inv_ptr, backend)
        k = fzk.i1 - 1
    end
    while k >= 1
        bwd_entry(k); k -= 1
    end
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
            if isfinite(dk) && !iszero(dk)   # NaN/Inf must fail like potrf, not "factor"
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
    li == 1 && SDS._subtree_begin!(ctl, trees, bm, subtree_ptr, G)
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
                            SDS.member_panels(front_ptr, mb, bm.nbatch), cb_ptr, li, Val(WG), Val(false))
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
            if isfinite(dk) && !iszero(dk)   # NaN/Inf must fail like potrf, not "factor"
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

# spin with backoff on an Int32 cell until it equals `target`; thread 1 only
@inline function _spin_until_eq!(arr, idx, target, scratch, scap)
    pause = min(Int32(4), scap)
    while (Atomix.@atomic arr[idx] += Int32(0)) != target
        z = Int32(0)                    # integer backoff: no Float64 in kernels (Metal)
        for _ in 1:pause
            z += Int32(1)
        end
        z < Int32(0) && (scratch[1] = z)
        pause = min(pause << 1, scap)
    end
    return nothing
end

@kernel function fused_fact_kernel!(factor, stack, pcb, info, @Const(nzval), @Const(amap), @Const(amap_ptr),
                                    @Const(amap_src), @Const(cb2), @Const(wt), @Const(role), @Const(ra), @Const(rb), @Const(rc),
                                    @Const(nchild), arrived, ea_left, panel_done, trsm_left, tiles_left,
                                    @Const(tiles_base),
                                    cb_done, @Const(snparent), @Const(front_ptr), @Const(front_nrows),
                                    @Const(front_ncols), @Const(cb_ptr), @Const(child_ptr),
                                    @Const(child_list), @Const(relind_ptr), @Const(relind), base,
                                    @Const(wbase), wupd, wpdone2, wtrsml, wcbl, scap, eac, epsc, d, @Const(super_ptr),
                                    ::Val{LDL}, ::Val{WG}) where {LDL, WG}
    @uniform TT = eltype(factor)
    @uniform RT = real(eltype(factor))
    li = @index(Local, Linear)
    G = Int(base) + @index(Group, Linear)
    sh = @localmem TT (NL64,)
    st = @localmem Int32 (1,)
    piv = @localmem RT (1,)
    fscr = @localmem Int32 (1,)   # spin-backoff scratch (integer: no Float64 in kernels)
    @inbounds begin
        r = role[G]
        if r == ROLE_EA
            v = Int(ra[G])                          # parent
            c = Int(rb[G])                          # child
            q0 = (Int(rc[G]) - 1) * Int(eac)
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
            qe = min(q0 + Int(eac), mc * mc) - 1
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
                    SDS._front_chol_update!(sh, st, j, v, li, front_ncols, epsc, Val(64), Val(WG))
                end
                @synchronize
                if LDL
                    _front_ldl_scale!(sh, st, piv, j, v, li, front_ncols, Val(WG))
                else
                    SDS._front_chol_scale!(sh, st, j, v, li, front_ncols, epsc, Val(64), Val(WG))
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
                # tiles_base, not the live tiles_left: the live counter can reach 0 while
                # this (stalled) last TRSM is between its decrement and this read, and the
                # last tile signals too -- a double signal overshoots cb_done/arrived and
                # the != spin loops of the consumers never terminate. The static "front
                # has no tiles" predicate is what this check means.
                if left == Int32(0) && tiles_base[v] == Int32(0)   # new-value semantics: 0 = last finisher
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
                        if st[1] == Int32(0) && isfinite(dj) && !iszero(dj)
                            piv[1] = dj                      # d stays on the diagonal
                            d[Int(super_ptr[v]) - 1 + c0 + j] = dj
                        elseif st[1] == Int32(0)
                            st[1] = Int32(c0 + j)
                            piv[1] = one(RT)
                        end
                    else
                        # same pivot-epsilon semantics as the narrow kernels (#127):
                        # [0, epsc] perturbs to epsc, negative/NaN fails the front
                        if st[1] == Int32(0) && (dj > epsc || (epsc > zero(RT) && dj >= zero(RT)))
                            piv[1] = sqrt(max(dj, epsc))
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

# potrf-derived LDL of a positive-definite wide front: A = C Cᵀ = L D Lᵀ with
# L = C diag(C)⁻¹ and d = diag(C)², and F21_chol = L21 D^½ makes the Cholesky
# herk CB bit-identical to the LDL one — so the host path is the VERBATIM
# Cholesky vendor sequence plus three micro-kernels. Valid only where the
# front is positive definite (always, for an SPD matrix factored as "S"); a
# saddle-point front fails potrf, so FusedSolver never selects it (#141): only an
# explicit `build_fused_plan(...; wides_potrf = true)` does.
@kernel function chol2ldl_d_kernel!(factor, d, p0, f, g0)
    j = @index(Global, Linear)
    @inbounds begin
        c = real(factor[p0 + (j - 1) * f + j - 1])
        d[g0 + j] = c * c
    end
end
@kernel function chol2ldl_scale_kernel!(factor, p0, f, w)      # strict lower: /= diag of the column
    q = @index(Global, Linear)
    @inbounds begin
        j = (Int(q) - 1) ÷ f + 1
        i = Int(q) - (j - 1) * f
        i > j && (factor[p0 + (j - 1) * f + i - 1] /= real(factor[p0 + (j - 1) * f + j - 1]))
    end
end
@kernel function chol2ldl_diag_kernel!(factor, p0, f)
    j = @index(Global, Linear)
    @inbounds factor[p0 + (j - 1) * f + j - 1] = one(eltype(factor))
end

function _factor_panel_c_ldl_potrf!(N, S, fp, v, p, backend)
    f = Int(S.schedule.rows[v])
    w = Int(S.schedule.width[v])
    p0 = Int(S.layout.panel_ptr[v])
    g0 = fp.sptr_h[v] - 1
    SDS._factor_panel_c!(N.factor, fp.pcb, N.work, N.info, v, p0, f, w, fp.cb2_h[v], p)
    chol2ldl_d_kernel!(backend, 128)(N.factor, N.d, p0, f, g0; ndrange = w)
    chol2ldl_scale_kernel!(backend, 128)(N.factor, p0, f, w; ndrange = f * w)
    chol2ldl_diag_kernel!(backend, 128)(N.factor, p0, f; ndrange = w)
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
function launch_subtrees_sorted!(N, S, nzval, list, lb, backend, ldl::Bool = false, epsc::Real = 0.0)
    SDS._with_local_bytes(Int(lb)) do LBv
        _sorted_subtree_launch(N, S, nzval, list, Val(ldl), LBv, Val(SDS.SUBTREE_WORKGROUP), backend, epsc)
    end
end
function _sorted_subtree_launch(N::SDS.Numeric{T}, S, nzval, list, ::Val{LDL}, ::Val{LB}, ::Val{WG},
                                backend, epsc::Real = 0.0) where {T, LDL, LB, WG}
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
           S.child_ptr, S.child_list, S.relind_ptr, S.relind, real(T)(epsc),
           Val((LB - SDS.SUBTREE_LOCAL_RESERVE) ÷ sizeof(T)), Val(WG); ndrange = WG * length(list) * bm.nact)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# host: the topologically ordered role list + counter bases + stream-2 program
function build_fused_plan(s; ldl::Bool = false, inkernel_wides::Union{Nothing, Bool} = nothing,
                          wides_potrf::Bool = false, merge_segments::Bool = false,
                          wgf::Int = WGF, scap::Integer = 128, ea_chunk::Integer = 16384)
    S = s.symbolic
    plan = s.numeric.plan
    backend = KA.get_backend(s.numeric.factor)
    T = eltype(s.numeric.factor)
    ldl && (T <: Real || throw(SDS.NotSupportedError("fused LDL mode: real element types only")))
    # the tile roles keep cld(TILE*TILE, wgf) partial sums in exactly 8 registers per
    # thread; a smaller workgroup needs more than 8 and would silently drop/reuse slots
    (wgf >= 128 && wgf % 32 == 0) ||
        throw(SDS.InvalidValueError("fused plan: workgroup size $wgf; expected a multiple of 32, >= 128"))
    # LDL host-wide fronts exist only as the potrf-derived path (wides_potrf = true);
    # otherwise LDL wides are factored in-kernel
    inkernel_wides = inkernel_wides === nothing ? (ldl && !wides_potrf) : inkernel_wides
    ldl && !inkernel_wides && !wides_potrf &&
        throw(SDS.InvalidValueError("fused LDL plan: host-wide fronts need wides_potrf = true " *
                                    "(or inkernel_wides = true)"))
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
    SDS._log(SDS.LOG_DEBUG, () -> "fused plan: write-through $nwt fronts, $(round(ewt / 1e6; digits = 2))M CB entries")

    emit_ea(v) = for kc in chp[v]:(chp[v + 1] - 1)
        c = chl[kc]
        cbp[c] > 0 || continue
        if wt_h[c] == v
            ea_left[v] += 1                      # settled by the child's last write-through tile
            continue
        end
        mc = rows_h[c] - width_h[c]
        for ch in 1:cld(mc * mc, ea_chunk)
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
            # ntl, not T: this closure assigns in function scope, and T is the element type
            ntl = cld(m, TILE)
            for tjj in 1:ntl, tii in tjj:ntl
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
            # the WTILE role packs rc = k * 1024 + tjj, so tjj must fit in 10 bits;
            # beyond that the tile would silently decode as a different panel
            nt <= 1023 ||
                throw(SDS.NotSupportedError("fused plan: wide front with $nbelow rows below " *
                                            "the panel; the WTILE encoding holds cld(rows, $TILE) <= 1023"))
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
        push!(ssub, _dvec(backend, ids))
    end

    SDS._log(SDS.LOG_DEBUG, () -> "fused plan: $(length(role)) roles, $(length(segments)) segments, " *
                                  "$(length(wides)) host wide fronts")
    return (; role = _dvec(backend, role), ra = _dvec(backend, ra), rb = _dvec(backend, rb),
            rc = _dvec(backend, rc), wides, ssub, segments,
            cb2 = _dvec(backend, Int32.(cb2_h)), cb2_h, pcb = KA.zeros(backend, T, off - 1),
            czero = _dvec(backend, czero), wt = _dvec(backend, wt_h),
            nchild = _dvec(backend, nchild),
            arrived_base = _dvec(backend, arrived_base), ea_base = _dvec(backend, ea_left),
            trsm_base = _dvec(backend, trsm_left), tiles_base = _dvec(backend, tiles_left),
            cb_base = _dvec(backend, cb_done_base),
            arrived = KA.zeros(backend, Int32, ns), ea = KA.zeros(backend, Int32, ns),
            trsm = KA.zeros(backend, Int32, ns), tiles = KA.zeros(backend, Int32, ns),
            pdone = KA.zeros(backend, Int32, ns), cbd = KA.zeros(backend, Int32, ns),
            wbase = _dvec(backend, wbase_h), wupd_b = _dvec(backend, wupd_b), wtl_b = _dvec(backend, wtl_b),
            wcb_b = _dvec(backend, wcb_b), wupd = KA.zeros(backend, Int32, max(nwslots, 1)),
            wtl = KA.zeros(backend, Int32, max(nwslots, 1)), wpd = KA.zeros(backend, Int32, max(nwslots, 1)),
            wcb = KA.zeros(backend, Int32, ns), wgf, scap = Int32(scap), eac = Int32(ea_chunk), ldl,
            # captured at plan build: a later setparam!("pivot_epsilon") changes the stock
            # path but not an already-built fused plan (rebuild the plan to change it)
            chol_eps = ldl ? 0.0 : Float64(SDS.resolved_pivot_epsilon(s.options, real(T))),
            ldl_potrf = ldl && wides_potrf,
            sptr_h = Array(S.super_ptr))
end

function refact_fused!(s, nzval, fp)
    N, S = s.numeric, s.symbolic
    plan = N.plan
    backend = KA.get_backend(N.factor)
    p = SDS._front_impls(N, S, :auto)
    # per-front statuses start clean every refactorization. Each front's factorization
    # stores its status into info[v] (the subtree kernels, the narrow-panel kernels at
    # their first panel step, the host-wide fronts through the potrf status), and the
    # fused kernel's trsm/update groups gate on info[v] == 0; the reset is the guard
    # that a status left by one failed refactorization never survives into the next,
    # whichever writer a front goes through.
    # (Before the subtree launches: kernels on this stream would otherwise be clobbered.)
    fill!(view(N.info, 1:SDS.nsupernodes(S)), Int32(0))
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
        launch_subtrees_sorted!(N, S, nzval, fp.ssub[k], plan.sub_local[k], backend, fp.ldl, fp.chol_eps)
    end
    nodes = S.schedule.group_nodes
    fill!(fp.pcb, zero(eltype(fp.pcb)))
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
                                             fp.arrived, fp.ea, fp.pdone, fp.trsm, fp.tiles, fp.tiles_base, fp.cbd,
                                             S.snparent, S.front_ptr, S.front_nrows, S.front_ncols,
                                             S.cb_ptr, S.child_ptr, S.child_list, S.relind_ptr, S.relind,
                                             Int32(base), fp.wbase, fp.wupd, fp.wpd, fp.wtl, fp.wcb,
                                             fp.scap, fp.eac, real(eltype(N.factor))(fp.chol_eps), N.d,
                                             S.super_ptr, Val(fp.ldl), Val(fp.wgf);
                                             ndrange = fp.wgf * count)
        for v in gwides                              # host-wide fronts at the segment boundary
            if fp.ldl && fp.ldl_potrf
                _factor_panel_c_ldl_potrf!(N, S, fp, v, p, backend)
            else
                SDS._factor_panel_c!(N.factor, fp.pcb, N.work, N.info, v, S.layout.panel_ptr[v],
                                     S.schedule.rows[v], S.schedule.width[v], fp.cb2_h[v], p, fp.chol_eps)
            end
            signal_kernel!(backend, 32)(fp.cbd, fp.arrived, S.snparent, v; ndrange = 32)
        end
    end
    SDS.assemble_schur!(N, S, nzval)
    # the stats kernel ALSO reduces the per-front info into info[ns + 1], which
    # _numeric_info! reads -- LDL needs that reduction too: without it a zero/NaN
    # pivot sets a front's info while the reduced slot keeps reporting the stale
    # success of an earlier factorization. In LDL mode N.stats' positive-pivot
    # fields are Cholesky-interpretation garbage; inertia comes from the signs of
    # d (the MadNLP extension), never from these stats.
    SDS.cholesky_stats!(N, S)
    return SDS._numeric_info!(N, S, true)
end

# ---------------------------------------------------------------------------
# FusedSolver: the fused pipeline behind the handle-layer verbs of PLAN §3.1 (T33).
# The recipe below is the one the MadNLP extension ran inline before (inner
# DirectSolver: tuning, analysis, factorization, a solve; then the three plans), so
# the kernels, plans and launch sequence are those of `SDSSolver`.

# the knobs of an `SDSTuning` that live in the options of a DirectSolver, and the
# native nested dissection with the tuning's METIS knobs (a `user_perm` set later wins)
function _apply_tuning!(s, t::SDSTuning)
    o = s.options
    o.amalgamation = t.amalgamation
    o.regime_c_width = t.regime_c_width
    o.regime_c_rows = t.regime_c_rows          # the fused kernels own both phases: no coupling conflict
    o.subtree_parallelism = t.subtree_parallelism
    o.subtree_budgets = copy(t.subtree_budgets)
    SDS.setparam!(s, "reordering_alg", "algo4")
    SDS.setparam!(s, "nd_nseps", t.nd_nseps)
    SDS.setparam!(s, "nd_seed", t.nd_seed)
    return s
end

# the parameters `setparam!` accepts on a FusedSolver (read by the next analysis)
const FUSED_SETTABLE = ("user_perm", "reordering_alg", "nd_nseps", "nd_seed", "pivot_epsilon")
# the analysis data parameters `getparam` forwards to the inner DirectSolver
const FUSED_FORWARDED = ("perm_reorder_row", "perm_reorder_col", "perm_row", "perm_col", "lu_nnz", "flops",
                         "nsuperpanels", "memory_estimates")

"""
    FusedSolver(A, structure, view; tuning = nothing, index = 'O')
    FusedSolver(rowptr, colval, nzval, structure, view; tuning = nothing, index = 'O')

The fused pipeline of `SparseDirectSolver.Fused` (dependency-counter
factorization mega-kernel and partitioned-inverse solve, the `SDSSolver` path
of the MadNLP extension) as an `AbstractDirectSolver`, driven by the same verbs
as [`DirectSolver`](@ref SparseDirectSolver.DirectSolver): [`execute!`](@ref SparseDirectSolver.execute!), [`analyze!`](@ref SparseDirectSolver.analyze!),
[`factorize!`](@ref SparseDirectSolver.factorize!), [`refactorize!`](@ref SparseDirectSolver.refactorize!), [`solve!`](@ref SparseDirectSolver.solve!),
[`setparam!`](@ref SparseDirectSolver.setparam!), [`getparam`](@ref SparseDirectSolver.getparam), [`update!`](@ref SparseDirectSolver.update!), `ldiv!` and `\\`
for one right-hand side. `A` is anything [`DirectSolver`](@ref SparseDirectSolver.DirectSolver) accepts with
these arguments (a [`CSR`](@ref SparseDirectSolver.CSR), `CuSparseMatrixCSR`/`CuSparseMatrixCSC`,
`ROCSparseMatrixCSR`, …) on a GPU backend; the values alias the caller's
`nzval`, so writing new values into it and calling `refactorize!` is the
interior-point loop (`update!` with the same pattern is forwarded).

`structure` is `"SPD"` (fused Cholesky) or `"S"` (fused LDLᵀ without pivoting:
`pivot_type = 'N'`, `pivot_pairs = "none"`). `tuning` is an
[`SDSTuning`](@ref SparseDirectSolver.Fused.SDSTuning), `nothing` meaning
[`default_tuning`](@ref SparseDirectSolver.Fused.default_tuning)`(backend)`,
resolved by [`resolve_tuning`](@ref SparseDirectSolver.Fused.resolve_tuning);
the constructor applies it to the inner `DirectSolver` (amalgamation, regime-C
width and rows, subtree parallelism and budgets, `reordering_alg = "algo4"`,
`nd_nseps`, `nd_seed`). The inner solver is the field `inner`.

Phases (`execute!(phase, F, X, B; asynchronous = true)`):

* `"reordering"`, `"symbolic_factorization"`, `"analysis"`: the inner
  solver's phase; the symbolic part then factorizes and solves once with the
  inner solver (stock kernels, it sizes the solve workspace) and builds the
  fused factorization plan, the per-front inverses and the fused solve
  segments. `X` and `B` are not used.
* `"factorization"`, `"refactorization"` (needs a previous
  `"factorization"`): the fused factorization of the current values, then the
  inverses of the diagonal blocks; sets `"info"` (`0`, or the original column
  of the first failed pivot: a non-positive Cholesky pivot, or for `"S"` a zero
  or non-finite pivot). A failed factorization does not throw. `"S"` factors
  its wide fronts in the fused kernel under either `wides` strategy (a front's
  pivots may take both signs).
* `"solve"`: `X = A⁻¹ B` with the partitioned-inverse solve, for a vector (or
  an `n × 1` matrix) on the solver's backend; `X === B` is allowed.

`setparam!` accepts `"user_perm"`, `"reordering_alg"`, `"nd_nseps"`,
`"nd_seed"` and `"pivot_epsilon"` (the Cholesky pivot epsilon; the fused
LDLᵀ factors with epsilon `0`), all read by the next analysis. `getparam`
returns `"info"`, `"inertia"` (`(npos, nneg)`; `"SPD"`: the positive pivots and
`0`; `"S"`: the counts of positive and negative entries of `D`), `"npivots"`
(`0` for `"S"`: the fused LDLᵀ perturbs no pivot), the analysis data
`"perm_reorder_row"`, `"perm_reorder_col"`, `"perm_row"`, `"perm_col"`,
`"lu_nnz"`, `"flops"`, `"nsuperpanels"`, `"memory_estimates"` of the inner
solver, and the configuration parameters.

Everything the pipeline does not do raises [`NotSupportedError`](@ref SparseDirectSolver.NotSupportedError), at
construction or in the phase, never a silent switch to the stock path: the
KernelAbstractions `CPU()` backend (the dependency-counter kernels spin on
resident workgroups, which the CPU backend runs one after another), complex
element types, the structures `"HPD"`, `"H"` and `"G"`, a uniform batch
(`nbatch > 1`), `nrhs > 1`, a transposed solve (`solve_mode ≠ 0`), the solve
sub-phases (`"solve_fwd_perm"` … `"solve_refinement"`, the Schur phases),
Schur mode, `matching_alg ≠ "default"`, `ir_n_steps > 0`,
`deterministic_mode = 1` (the concurrent extend-add sums in arrival order),
a `factor_precision` other than the working precision, `"S"` with
`pivot_type ≠ 'N'`, a matrix without a front of width ≤ 256 for the fused
solve, and every `setparam!` name not listed above. The options of the inner
solver are checked again at the analysis. [`DirectSolver`](@ref SparseDirectSolver.DirectSolver) remains the
general solver.
"""
mutable struct FusedSolver{T, INT, S <: SDS.DirectSolver{T, INT}} <: SDS.AbstractDirectSolver{T, INT}
    inner::S                 # the DirectSolver: analysis, numeric storage, solve workspace
    tuning::SDSTuning        # resolved (`wides` is :host or :merged)
    ldl::Bool                # "S": no-pivot LDLᵀ
    nzval::Any               # the values the fused factorization reads (`_factor_values(inner)`)
    fp::Any                  # factorization plan (`build_fused_plan`)
    inv11::Any               # per-front L11 inverses (`build_inverse`)
    inv_ptr::Any
    invert!::Any
    fz::Any                  # fused solve segments (`build_fused`)
    info::Int
    stage::Int
end

function FusedSolver(A, structure, view; tuning::Union{Nothing, SDSTuning} = nothing, kwargs...)
    return _fused_solver(SDS.DirectSolver(A, structure, view; kwargs...), tuning)
end

function FusedSolver(rowptr::AbstractVector{<:Integer}, colval::AbstractVector{<:Integer}, nzval::AbstractVecOrMat,
                     structure, view; tuning::Union{Nothing, SDSTuning} = nothing, kwargs...)
    return _fused_solver(SDS.DirectSolver(rowptr, colval, nzval, structure, view; kwargs...), tuning)
end

# the tuning a solver on `backend` runs with (`nothing`: the architecture default); shared
# with the `:bk` path of the MadNLP extension
_resolved_tuning(tuning, backend) = resolve_tuning(something(tuning, default_tuning(backend)), backend)

function _fused_solver(s::SDS.DirectSolver{T, INT}, tuning::Union{Nothing, SDSTuning}) where {T, INT}
    backend = s.backend
    backend isa KA.CPU &&
        throw(SDS.NotSupportedError("FusedSolver: the KernelAbstractions CPU backend is not supported (the " *
                                    "dependency-counter kernels spin on resident workgroups); use DirectSolver"))
    T <: Real ||
        throw(SDS.NotSupportedError("FusedSolver: complex element type $T is not supported (the fused kernels " *
                                    "have no conjugation); use DirectSolver"))
    st = s.structure
    st == SDS.STRUCTURE_SPD || st == SDS.STRUCTURE_SYMMETRIC ||
        throw(SDS.NotSupportedError("FusedSolver: structure \"$(convert(String, st))\" is not supported; " *
                                    "expected \"SPD\" or \"S\" (use DirectSolver)"))
    s.nbatch == 1 ||
        throw(SDS.NotSupportedError("FusedSolver: a uniform batch ($(s.nbatch) members) is not supported"))
    t = _resolved_tuning(tuning, backend)
    ldl = st == SDS.STRUCTURE_SYMMETRIC
    if ldl
        SDS.setparam!(s, "pivot_type", 'N')
        # no-pivot LDL never consumes 2x2 pivot pairs, and "S" analysis pays dearly for
        # detecting them on saddle systems (augmented 78k KKT, n = 2.09M: analysis 316 s
        # with the default pair detection, 64 s without -- measured on a GV100)
        SDS.setparam!(s, "pivot_pairs", "none")
    end
    _apply_tuning!(s, t)
    return FusedSolver{T, INT, typeof(s)}(s, t, ldl, nothing, nothing, nothing, nothing, nothing, nothing, 0,
                                          SDS.STAGE_NONE)
end

Base.size(F::FusedSolver) = size(F.inner)
Base.size(F::FusedSolver, d::Integer) = size(F.inner, d)

function Base.show(io::IO, F::FusedSolver{T, INT}) where {T, INT}
    s = F.inner
    print(io, "FusedSolver{", T, ", ", INT, "}(n = ", size(F, 1), ", nnz = ", SDS.nnz(s.A), ", structure = \"",
          convert(String, s.structure), "\", backend = ", nameof(typeof(s.backend)), ", wides = :",
          F.tuning.wides, ", ", SDS._STAGE_NAMES[F.stage + 1])
    F.stage == SDS.STAGE_FACTORIZED && print(io, ", info = ", F.info)
    print(io, ")")
    return nothing
end

# the options a FusedSolver cannot honour: set only by mutating `F.inner.options` directly
# (setparam! rejects them), checked again at every analysis
function _check_fused_options(F::FusedSolver{T}) where {T}
    o = F.inner.options
    unsupported(what) = throw(SDS.NotSupportedError("FusedSolver: $what is not supported; use DirectSolver"))
    o.matching_alg == SDS.MATCHING_NONE || unsupported("matching_alg = \"$(convert(String, o.matching_alg))\"")
    o.schur_mode == 0 || unsupported("schur_mode = 1")
    o.ir_n_steps == 0 || unsupported("ir_n_steps = $(o.ir_n_steps) (iterative refinement)")
    o.deterministic_mode == 0 || unsupported("deterministic_mode = 1")
    o.factor_precision === nothing || o.factor_precision === real(T) ||
        unsupported("factor_precision = $(o.factor_precision) for $T input")
    o.ubatch_size == 0 || o.ubatch_size == 1 || unsupported("ubatch_size = $(o.ubatch_size)")
    o.solve_alg == SDS.SOLVE_DEFAULT || unsupported("solve_alg = \"$(convert(String, o.solve_alg))\"")
    F.ldl && o.pivot_type != SDS.PIVOT_NONE &&
        unsupported("structure \"S\" with pivot_type = '$(convert(Char, o.pivot_type))'")
    o.user_schur_indices === nothing || unsupported("user_schur_indices")
    return nothing
end

# drop the plans of an earlier analysis (a new analysis rebuilds them)
function _clear_plans!(F::FusedSolver)
    F.nzval = F.fp = F.inv11 = F.inv_ptr = F.invert! = F.fz = nothing
    F.info = 0
    return F
end

# the symbolic part of the analysis: inner symbolic phase, one stock factorization and
# solve (the solve workspace), then the three fused plans
function _fused_symbolic!(F::FusedSolver{T}) where {T}
    s = F.inner
    SDS.execute!("symbolic_factorization", s, nothing, nothing; asynchronous = false)
    n = size(s, 1)
    x = KA.zeros(s.backend, T, n)
    b = KA.zeros(s.backend, T, n)
    SDS.execute!("factorization", s, x, b; asynchronous = false)
    SDS.execute!("solve", s, x, b; asynchronous = false)   # allocates the solve workspace
    t = F.tuning
    merged = t.wides === :merged
    # LDL wide fronts are always factored in-kernel (the no-pivot LDLᵀ of the fused
    # kernel), whatever the tuning: the host path for them would be the potrf-derived
    # LDL (wides_potrf), which is valid only while a front's pivots share a sign. The
    # root separators of a saddle-point matrix under nested dissection hold primal and
    # dual rows, so potrf fails on a matrix the no-pivot LDLᵀ factors (#141). `wides =
    # :host` keeps its meaning for Cholesky; for LDL it only keeps the segment split.
    inkernel = merged || F.ldl
    fp = build_fused_plan(s; ldl = F.ldl, inkernel_wides = inkernel, merge_segments = merged,
                          wides_potrf = false, wgf = t.workgroup, scap = t.spin_cap)
    inv11, inv_ptr, invertf, _ = build_inverse(s)
    # the solve plan: one merged dependency-counter sweep over the whole tree
    fz1 = build_fused(s; kstart = 1, wcap = 256)
    fz1 === nothing && throw(SDS.NotSupportedError("fused solve: no plan entry with fronts of width <= 256"))
    F.nzval = SDS._factor_values(s)
    F.fp = fp
    F.inv11, F.inv_ptr, F.invert! = inv11, inv_ptr, invertf
    F.fz = ([(fz1, 64)], true)
    F.info = 0
    F.stage = SDS.STAGE_ANALYZED
    return F
end

# the fused factorization behind a function barrier (the plans are untyped fields)
function _fused_refactor!(s, nzval, fp, invertf)
    info = Int(refact_fused!(s, nzval, fp))
    invertf()
    return info
end

function _fused_factorize!(F::FusedSolver, p::SDS.Phase)
    name = convert(String, p)
    F.stage >= SDS.STAGE_ANALYZED || throw(SDS._phase_error(name, "needs \"analysis\""))
    p == SDS.PHASE_REFACTORIZATION && F.stage < SDS.STAGE_FACTORIZED &&
        throw(SDS._phase_error(name, "needs a previous \"factorization\""))
    F.nzval = SDS._factor_values(F.inner)
    F.info = _fused_refactor!(F.inner, F.nzval, F.fp, F.invert!)
    F.stage = SDS.STAGE_FACTORIZED
    return F
end

# a right-hand side of the fused solve: a vector, or an n × 1 matrix, on the solver's backend
function _fused_rhs(F::FusedSolver{T}, X, name) where {T}
    if X isa AbstractMatrix
        size(X, 2) == 1 ||
            throw(SDS.NotSupportedError("FusedSolver: $(size(X, 2)) right-hand sides; the fused solve takes one"))
        X = vec(X)
    end
    X isa AbstractVector ||
        throw(SDS.NotSupportedError("FusedSolver: $name must be a vector (or an n × 1 matrix), got a " *
                                    "$(typeof(X)); batches and descriptors are not supported"))
    eltype(X) == T || throw(SDS.InvalidValueError("$name ($(eltype(X))) must have the solver's element type $T"))
    length(X) == size(F, 1) ||
        throw(SDS.InvalidValueError("$name has $(length(X)) entries, the matrix has $(size(F, 1)) rows"))
    SDS._check_rhs_backend(F.inner, X, name)
    return X
end

function _fused_solve!(F::FusedSolver, X, B)
    F.stage >= SDS.STAGE_FACTORIZED || throw(SDS._phase_error("solve", "needs a \"factorization\""))
    o = F.inner.options
    o.solve_mode == 0 ||
        throw(SDS.NotSupportedError("FusedSolver: solve_mode = $(o.solve_mode) (transposed solve) is not supported"))
    o.ir_n_steps == 0 ||
        throw(SDS.NotSupportedError("FusedSolver: ir_n_steps = $(o.ir_n_steps) (iterative refinement) is not supported"))
    x = _fused_rhs(F, X, "X")
    b = _fused_rhs(F, B, "B")
    algo2_solve!(x, F.inner, b, F.inv11, F.inv_ptr, F.fz)
    return F
end

function SDS.execute!(phase::AbstractString, F::FusedSolver, X, B; asynchronous::Bool = true)
    p = SDS._parse_phase(phase)
    if p == SDS.PHASE_REORDERING || p == SDS.PHASE_ANALYSIS
        _check_fused_options(F)
        _clear_plans!(F)
        F.stage = SDS.STAGE_NONE
        SDS.execute!("reordering", F.inner, nothing, nothing; asynchronous = false)
        F.stage = SDS.STAGE_REORDERED
        p == SDS.PHASE_ANALYSIS && _fused_symbolic!(F)
    elseif p == SDS.PHASE_SYMBOLIC_FACTORIZATION
        F.stage >= SDS.STAGE_REORDERED || throw(SDS._phase_error(phase, "needs \"reordering\""))
        _check_fused_options(F)
        _clear_plans!(F)
        F.stage = SDS.STAGE_REORDERED
        _fused_symbolic!(F)
    elseif p == SDS.PHASE_FACTORIZATION || p == SDS.PHASE_REFACTORIZATION
        _fused_factorize!(F, p)
    elseif p == SDS.PHASE_SOLVE
        _fused_solve!(F, X, B)
    else
        throw(SDS.NotSupportedError("FusedSolver: phase \"$phase\" is not supported; the fused solve runs as " *
                                    "the one phase \"solve\""))
    end
    asynchronous || KA.synchronize(F.inner.backend)
    return nothing
end

function SDS.analyze!(F::FusedSolver, X = nothing, B = nothing; asynchronous::Bool = true)
    SDS.execute!("analysis", F, X, B; asynchronous)
    return F
end

function SDS.factorize!(F::FusedSolver, X = nothing, B = nothing; asynchronous::Bool = true)
    SDS.execute!("factorization", F, X, B; asynchronous)
    return F
end

function SDS.refactorize!(F::FusedSolver, X = nothing, B = nothing; asynchronous::Bool = true)
    SDS.execute!("refactorization", F, X, B; asynchronous)
    return F
end

function SDS.solve!(F::FusedSolver, X, B; asynchronous::Bool = true)
    SDS.execute!("solve", F, X, B; asynchronous)
    return X
end

function SDS.setparam!(F::FusedSolver, name::AbstractString, value)
    SDS.parameter_spec(name)                   # unknown names: the ArgumentError of DirectSolver
    name in FUSED_SETTABLE ||
        throw(SDS.NotSupportedError("FusedSolver: parameter \"$name\" cannot be set; the fused pipeline takes " *
                                    join(("\"$p\"" for p in FUSED_SETTABLE), ", ") * " (use DirectSolver)"))
    SDS.setparam!(F.inner, name, value)
    return nothing
end

function SDS.getparam(F::FusedSolver{T, INT}, name::AbstractString) where {T, INT}
    spec = SDS.parameter_spec(name)
    name == "info" && return F.info
    if name == "inertia" || name == "npivots"
        F.stage >= SDS.STAGE_FACTORIZED ||
            throw(SDS.FactorizationError(0, "the data parameter \"$name\" needs the \"factorization\" phase"))
        F.ldl || return SDS.getparam(F.inner, name)   # Cholesky: the stats of the fused factorization
        name == "npivots" && return zero(INT)          # the fused LDLᵀ perturbs no pivot
        d = view(F.inner.numeric.d, 1:size(F, 1))
        return (INT(count(>(zero(T)), d)), INT(count(<(zero(T)), d)))
    end
    name in FUSED_FORWARDED && return SDS.getparam(F.inner, name)
    spec.status === :output || spec.status === :solver ||
        return SDS.getparam(F.inner.options, name)
    throw(SDS.NotSupportedError("FusedSolver: the data parameter \"$name\" is not provided (use DirectSolver)"))
end

function SDS.update!(F::FusedSolver, A)
    SDS.update!(F.inner, A)
    F.nzval = SDS._factor_values(F.inner)
    return F
end

function SDS.update!(F::FusedSolver, rowptr::AbstractVector{<:Integer}, colval::AbstractVector{<:Integer},
                     nzval::AbstractVecOrMat)
    SDS.update!(F.inner, rowptr, colval, nzval)
    F.nzval = SDS._factor_values(F.inner)
    return F
end

end # module
