# The fused pipeline (PR #107 kernels + the October 2026 campaign), assembled
# as a backend-portable submodule: the fused factorization (Cholesky and
# no-pivot LDL^T) and the partitioned-inverse/fused solve, on CUDA and AMDGPU
# alike. The plans are built on the solver's backend and carry their own device
# state (no module globals). Consumed by `FusedSolver` and the `KKTSolver`
# front end (src/kkt.jl); the tuning bundle `SDSTuning` / `default_tuning` lives
# here because it parameterizes these kernels, not the front end.
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

The T34 fields: `solve_workgroup` is the workgroup size of the fused solve
sweeps (a multiple of 32 in 32:256); `wide_host_min` puts the wide fronts
wider than it on the host vendor path in a `:merged` Cholesky plan (the
per-front wide strategy; `typemax(Int)`, the default, keeps every wide front
in-kernel; ignored for `"S"`); `graph_max_n` replays the refactorization and
the solve sweeps from a CUDA graph when `n <= graph_max_n`, vendor dense calls
in the captured range included (the host-wide fronts, the solve fronts wider
than 256; #174); `0`, the default, never captures (no measured gain, see
bench/ARCH-TUNING.md).
"""
Base.@kwdef struct SDSTuning
    wides::Symbol = :auto                     # :auto | :host | :merged (Cholesky; an LDLᵀ
                                              # factors wides in-kernel, :host keeps the split)
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
    # T34 knobs
    solve_workgroup::Int = 64                 # fused-solve (fwd2/bwd2) workgroup size: 32..256, multiple of 32
    wide_host_min::Int = typemax(Int)         # :merged Cholesky: fronts wider than this on the host vendor path
    graph_max_n::Int = 0                      # CUDA-graph replay of refactorization and solve when n <= this
end

"""
    default_tuning(backend) -> SDSTuning

The measured per-architecture defaults — the single place they live.
2048-threads/SM NVIDIA parts (sm_70/80/90): host wides, 16K budgets, fused-solve
workgroup 128 (T34, measured on an H200; sm_70/80 not re-measured).
1536-thread parts (sm_86/89, Blackwell workstation): merged wides, 8K budgets,
solve workgroup 64 (T34 did not measure them).
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
                                SDSTuning(; wides = :host, solve_workgroup = 128)
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
    w = default_tuning(backend).wides
    return SDSTuning((f === :wides ? w : getfield(t, f) for f in fieldnames(SDSTuning))...)
end

# device copy of a host integer vector (empty vectors included)
_dvec(backend, v) = SDS._device_vec(backend, eltype(v), v)

const WG = 32
const MW = 256  # max front width on the inverse path
const WGF = 128 # default workgroup size; the fused kernel takes it per plan (fp.wgf)
const CHWG = 1024 # workgroup size of the chain kernels
const BWR = 8     # work items per column in the gather of bwd2_kernel! (coalesced rows)
const NSTRIP = 4  # strip parallelism of the chain kernels

# ---------------------------------------------------------------------------
# inversion: thread j of the front's workgroup computes column j of inv(L11)
# (non-unit lower triangular), written col-major at inv_ptr[v]. The reference the T34
# gate compares against (bench); the pipeline runs invert_blocked_kernel! instead
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
#
# deterministic_mode = 1 (`DET`, T35): no atomics. Every front writes its update to its own buffer in `U`
# (the rows `w+1:f` of its gather list, the stock deterministic layout of PLAN §2.5) and first pulls
# its children's buffers in `child_list` order (`_fused_pull!`); a front wider than MW only pulls (its
# dense update follows, `SDS._fwd_dense!` with `det = true`)
@kernel function fwd1_kernel!(Y, @Const(factor), @Const(inv11), @Const(inv_ptr), @Const(list), first, count,
                              @Const(subtree_ptr), @Const(subtree_nodes), @Const(super_ptr), @Const(rowptr),
                              @Const(rowval), @Const(front_ptr), @Const(front_nrows), @Const(front_ncols),
                              U, @Const(child_ptr), @Const(child_list), @Const(relind_ptr), @Const(relind),
                              ::Val{SUB}, ::Val{WGV}, ::Val{DET}) where {SUB, WGV, DET}
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
            if DET
                _fused_pull!(Y, U, v, w, f, rp, c0, child_ptr, child_list, rowptr, front_nrows, front_ncols,
                             relind_ptr, relind, li, Val(WGV))
            end
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
                if DET
                    U[rp + i] -= acc
                else
                    row = Int(rowval[rp + i])
                    if row > limit
                        Atomix.@atomic Y[row] += -acc
                    else
                        Y[row] -= acc
                    end
                end
            end
            @synchronize
        end
    end
end

# deterministic forward (T35): zero the update buffer of front v (rows w+1:f of its gather list at rp),
# then add the children's buffers in child_list order, into the front's columns of `Y` (or of its local
# copy `yb`, when given) and into its own buffer (the rows of one child are distinct: no conflicts). Ends
# on a barrier.
@inline function _fused_pull!(Y, U, v, w, f, rp, c0, child_ptr, child_list, rowptr, front_nrows, front_ncols,
                              relind_ptr, relind, li, ::Val{WGV}, yb = nothing) where {WGV}
    @inbounds begin
        z = zero(eltype(U))
        for i in (w + li):WGV:f
            U[rp + i] = z
        end
        @synchronize
        k0 = Int(child_ptr[v])
        for kc in k0:(Int(child_ptr[v + 1]) - 1)
            c = Int(child_list[kc])
            wc = Int(front_ncols[c])
            mc = Int(front_nrows[c]) - wc
            src = Int(rowptr[c]) + wc - 1
            r0 = Int(relind_ptr[c]) - 1
            for k in li:WGV:mc
                t = Int(relind[r0 + k])
                x = U[src + k]
                if t > w
                    U[rp + t] += x
                elseif yb === nothing
                    Y[c0 + t] += x
                else
                    yb[t] += x
                end
            end
            @synchronize
        end
    end
    return nothing
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
# blocked inversion (T34): one workgroup per (front, 32-column block J) computes the block
# column J of inv(L11) by block forward substitution, X_IJ = -inv(L_II) Σ_{K=J}^{I-1} L_IK X_KJ,
# with the 32 x 32 diagonal-block inverses formed in local memory (thread c: column c, the
# arithmetic of invert_l11_kernel!, so a front of width <= 32 gets the same bits). The
# per-thread serial chain is O(32²) per block instead of O(w²) per column: for the fronts
# of width 65..256 it replaces invert_l11_kernel!'s latency-bound loops (#155).
const IB = 32
const INVWG = 128

@kernel function invert_blocked_kernel!(inv11, @Const(factor), @Const(inv_ptr), @Const(pv), @Const(pj),
                                        @Const(front_ptr), @Const(front_nrows), @Const(front_ncols),
                                        ::Val{WGV}) where {WGV}
    @uniform TT = eltype(inv11)
    li = @index(Local, Linear)
    g = @index(Group, Linear)
    Bs = @localmem TT (IB * IB,)        # inverse of the diagonal block L_II (col-major)
    Ls = @localmem TT (IB * IB,)        # staged block of L
    Xs = @localmem TT (IB * IB,)        # staged block X_KJ
    Ts = @localmem TT (IB * IB,)        # Σ_K L_IK X_KJ
    @inbounds begin
        v = Int(pv[g])
        J = Int(pj[g])                  # 0-based block column
        f = Int(front_nrows[v])
        w = Int(front_ncols[v])
        p0 = Int(front_ptr[v]) - 1
        ip = Int(inv_ptr[v]) - 1
        nb = cld(w, IB)
        c0 = J * IB
        bj = min(IB, w - c0)
        if c0 > 0                       # rows above the diagonal block are zero
            for q in (li - 1):WGV:(c0 * bj - 1)
                c = q ÷ c0 + 1
                k = q - (c - 1) * c0 + 1
                inv11[ip + (c0 + c - 1) * w + k] = zero(TT)
            end
        end
        for I in J:(nb - 1)
            r0 = I * IB
            bi = min(IB, w - r0)
            for q in (li - 1):WGV:(IB * IB - 1)
                Ts[q + 1] = zero(TT)
            end
            for K in J:(I - 1)          # full blocks: K < I <= nb - 1
                k0 = K * IB
                @synchronize
                for q in (li - 1):WGV:(IB * IB - 1)
                    rr = q % IB + 1
                    cc = q ÷ IB + 1
                    Ls[q + 1] = rr <= bi ? factor[p0 + (k0 + cc - 1) * f + r0 + rr] : zero(TT)
                    Xs[q + 1] = cc <= bj ? inv11[ip + (c0 + cc - 1) * w + k0 + rr] : zero(TT)
                end
                @synchronize
                for q in (li - 1):WGV:(IB * IB - 1)
                    rr = q % IB + 1
                    cc = q ÷ IB + 1
                    a = zero(TT)
                    for m in 1:IB
                        a += Ls[rr + (m - 1) * IB] * Xs[m + (cc - 1) * IB]
                    end
                    Ts[q + 1] += a
                end
            end
            @synchronize
            for q in (li - 1):WGV:(IB * IB - 1)
                rr = q % IB + 1
                cc = q ÷ IB + 1
                Ls[q + 1] = (rr <= bi && cc <= rr) ? factor[p0 + (r0 + cc - 1) * f + r0 + rr] : zero(TT)
            end
            @synchronize
            for c in li:WGV:bi
                cb = (c - 1) * IB
                for k in 1:(c - 1)
                    Bs[cb + k] = zero(TT)
                end
                for k in c:bi
                    acc = k == c ? one(TT) : zero(TT)
                    for m in c:(k - 1)
                        acc -= Ls[k + (m - 1) * IB] * Bs[cb + m]
                    end
                    Bs[cb + k] = acc / Ls[k + (k - 1) * IB]
                end
            end
            @synchronize
            for q in (li - 1):WGV:(IB * IB - 1)
                rr = q % IB + 1
                cc = q ÷ IB + 1
                if rr <= bi && cc <= bj
                    x = zero(TT)
                    if I == J
                        x = Bs[q + 1]
                    else
                        a = zero(TT)
                        for m in 1:rr
                            a += Bs[rr + (m - 1) * IB] * Ts[m + (cc - 1) * IB]
                        end
                        x = -a
                    end
                    inv11[ip + (c0 + cc - 1) * w + r0 + rr] = x
                end
            end
            @synchronize                # X_IJ is read back by the next block row
        end
    end
end

# ---------------------------------------------------------------------------
# per-front inverses of the L11 blocks (width <= MW) for the partitioned-inverse solve:
# returns the device buffer, its per-front offsets, a closure that (re)computes the
# inverses from the current factor (`invert_blocked_kernel!`), and the device bytes of every
# buffer it allocates (inverses, offsets, the (front, block column) lists). Every front wider
# than MW must be solved by the dense calls of a SOLVE_DENSE plan entry (the regime-C path):
# `fwd1_kernel!`/`bwd1_kernel!` skip such fronts, so a wider front in a regime-B entry (a raised
# `regime_c_width`) raises NotSupportedError
function build_inverse(s)
    S = s.symbolic
    sc = S.schedule
    width = sc.width
    ns = length(width)
    plan = s.workspace.plan
    nodes = sc.group_nodes
    for k in eachindex(plan.kind)
        plan.kind[k] == SDS.SOLVE_FRONTS || continue
        for q in plan.first[k]:plan.last[k]
            v = nodes[q]
            width[v] <= MW ||
                throw(SDS.NotSupportedError("fused solve: front $v of width $(width[v]) > $MW is not on the regime-C " *
                                            "path (regime_c_width = $(sc.regime_c_width)); the partitioned-inverse " *
                                            "solve needs every front wider than $MW in regime C"))
        end
    end
    inv_ptr_h = zeros(Int64, ns + 1)
    p = 1
    for v in 1:ns
        inv_ptr_h[v] = p
        width[v] <= MW && (p += width[v]^2)
    end
    inv_ptr_h[ns + 1] = p
    backend = KA.get_backend(s.numeric.factor)
    T = eltype(s.numeric.factor)
    inv11 = KA.zeros(backend, T, p - 1)
    inv_ptr = _dvec(backend, inv_ptr_h)
    # (front, block column) pairs, the longest block columns first
    pairs = Tuple{Int32, Int32}[]
    for v in 1:ns
        width[v] <= MW || continue
        for J in 0:(cld(width[v], IB) - 1)
            push!(pairs, (Int32(v), Int32(J)))
        end
    end
    sort!(pairs; by = q -> cld(width[q[1]], IB) - q[2], rev = true)
    pv = _dvec(backend, first.(pairs))
    pj = _dvec(backend, last.(pairs))
    np = length(pairs)
    invert! = () -> (np > 0 && invert_blocked_kernel!(backend, INVWG)(inv11, s.numeric.factor, inv_ptr, pv, pj,
                                                                       S.front_ptr, S.front_nrows, S.front_ncols,
                                                                       Val(INVWG); ndrange = INVWG * np); nothing)
    bytes = Int64(sizeof(T)) * (p - 1) + Int64(sizeof(Int64)) * (ns + 1) + Int64(2 * sizeof(Int32)) * np
    return inv11, inv_ptr, invert!, bytes
end

# ---------------------------------------------------------------------------
# fused sweeps (algo2): one launch per direction over every front of width
# <= MW below the first wide front, etree dependency counters instead of
# level-batched launches. CUDA-only (threadfence, resident-grid spinning).
@kernel function fwd2_kernel!(Y, @Const(factor), @Const(inv11), @Const(inv_ptr),
                              @Const(flist), @Const(slist), nsub, @Const(subtree_ptr), @Const(subtree_nodes),
                              @Const(snparent), arrived, @Const(nchild), @Const(super_ptr),
                              @Const(rowptr), @Const(rowval), @Const(front_ptr), @Const(front_nrows),
                              @Const(front_ncols), U, @Const(child_ptr), @Const(child_list), @Const(relind_ptr),
                              @Const(relind), ::Val{WGV}, ::Val{DET}) where {WGV, DET}
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
            if DET                                  # the children walked before v, by this workgroup
                _fused_pull!(Y, U, v, w, f, rp, c0, child_ptr, child_list, rowptr, front_nrows, front_ncols,
                             relind_ptr, relind, li, Val(WGV))
            end
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
                if DET
                    U[rp + i] -= acc
                else
                    row = Int(rowval[rp + i])
                    if row > limit
                        Atomix.@atomic Y[row] += -acc
                    else
                        Y[row] -= acc
                    end
                end
            end
            @synchronize
        end
        if DET
            SDS.device_fence(Y)                     # ALL threads: the root's buffer is read by its parent
            @synchronize
        end
        if li == 1
            SDS.device_fence(Y)
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
        SDS.device_fence(Y)
        if DET                                      # only this front writes its columns of Y
            for kk in li:WGV:w
                yb[kk] = Y[c0 + kk]
            end
            @synchronize
            _fused_pull!(Y, U, v, w, f, rp, c0, child_ptr, child_list, rowptr, front_nrows, front_ncols,
                         relind_ptr, relind, li, Val(WGV), yb)
        else
            for kk in li:WGV:w                      # coherent read of the front's columns
                yb[kk] = Atomix.@atomic Y[c0 + kk] += zero(TT)
            end
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
        for i in (w + li):WGV:f                     # L21 update: atomic scatter, or the own buffer (DET)
            acc = zero(TT)
            for j in 1:w
                acc += factor[p0 + (j - 1) * f + i] * t[j]
            end
            if DET
                U[rp + i] -= acc
            else
                Atomix.@atomic Y[Int(rowval[rp + i])] += -acc
            end
        end
        if DET
            SDS.device_fence(Y)                     # ALL threads: the buffer is read by the parent
        end
        @synchronize
        if li == 1
            SDS.device_fence(Y)
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
        SDS.device_fence(Y)
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
        SDS.device_fence(Y)
        for kk in li:WGV:w
            t[kk] = zero(TT)
        end
        @synchronize
        i0 = w + 1
        lane = (li - 1) % BWR + 1                   # row lane of the work item
        slot = (li - 1) ÷ BWR + 1                   # its column within a batch of WGV ÷ BWR columns
        while i0 <= f                               # chunked coherent gather of ancestor solutions
            ii = i0 + li - 1
            if ii <= f
                xb[li] = Atomix.@atomic Y[Int(rowval[rp + ii])] += zero(TT)
            end
            @synchronize
            m = min(WGV, f - i0 + 1)
            # t += L21(chunk, :)ᵀ xb: BWR work items share a column and read consecutive rows
            # (coalesced, T34), their partial sums are reduced through t2 (free until below)
            kb0 = 0
            while kb0 < w
                kk = kb0 + slot
                acc = zero(TT)
                if kk <= w
                    col = p0 + (kk - 1) * f + i0 - 1
                    for c in lane:BWR:m
                        acc += factor[col + c] * xb[c]
                    end
                end
                t2[li] = acc
                @synchronize
                if li <= WGV ÷ BWR && kb0 + li <= w
                    a = t[kb0 + li]
                    for r in 1:BWR
                        a += t2[(li - 1) * BWR + r]
                    end
                    t[kb0 + li] = a
                end
                @synchronize
                kb0 += WGV ÷ BWR
            end
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
            SDS.device_fence(Y)
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

# the dependency-counter sweeps take a workgroup of 32..512 work items, a multiple of 32
# (`SDSTuning.solve_workgroup`; 64 is the size they were written for)
# (at most MW: bwd2_kernel! reduces one partial sum per work item through a MW buffer)
# the accepted `solve_workgroup` values: a multiple of 32 in 32:MW (32:256), checked for every
# tuning when a `FusedSolver` is built; the chain kernel (`wg > 512` in `run_fused!`) is bench-only
# and returns before this check
_check_solve_workgroup(wg) = (32 <= wg <= MW && wg % 32 == 0) ||
    throw(SDS.InvalidValueError("fused solve: workgroup size $wg; expected a multiple of 32 in 32:$MW"))

# one fused segment, forward sweep: `wg > 512` runs the single-workgroup chain kernel,
# otherwise the dependency-counter sweep with workgroups of `wg` work items (`det`: the deterministic
# variant, update buffers in `ws.U` pulled by the parents, T35)
function run_fused!(fz, wg, Yv, s, inv11, inv_ptr, backend, det::Bool = false)
    S, N = s.symbolic, s.numeric
    if wg > 512
        chain_fwd_kernel!(backend, CHWG)(Yv, N.factor, inv11, inv_ptr, fz.flist, S.super_ptr, S.rowptr,
                                         S.rowval, S.front_ptr, S.front_nrows, S.front_ncols, Val(CHWG);
                                         ndrange = CHWG)
        return nothing
    end
    _check_solve_workgroup(wg)
    copyto!(fz.arrived, fz.arrived_base)
    Uv = vec(s.workspace.U)
    k = fwd2_kernel!(backend, wg)
    args = (Yv, N.factor, inv11, inv_ptr, fz.flist, fz.slist, fz.nsub, S.subtree_ptr, S.subtree_nodes, S.snparent,
            fz.arrived, fz.nchild, S.super_ptr, S.rowptr, S.rowval, S.front_ptr, S.front_nrows, S.front_ncols, Uv,
            S.child_ptr, S.child_list, S.relind_ptr, S.relind)
    if det
        k(args..., Val(wg), Val(true); ndrange = wg * (fz.nfused + fz.nsub))
    else
        k(args..., Val(wg), Val(false); ndrange = wg * (fz.nfused + fz.nsub))
    end
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
    _check_solve_workgroup(wg)
    copyto!(fz.done, fz.done_base)
    bwd2_kernel!(backend, wg)(Yv, N.factor, inv11, inv_ptr, fz.flist, fz.slist, fz.nsub, S.subtree_ptr,
                              S.subtree_nodes, S.snparent, fz.done,
                              S.super_ptr, S.rowptr, S.rowval, S.front_ptr, S.front_nrows, S.front_ncols,
                              Val(wg); ndrange = wg * (fz.nfused + fz.nsub))
    return nothing
end

# the fused solve: `fz = (segments, covered)` with `segments` a vector of
# `(build_fused(...), wg)` pairs in plan order; plan entries outside the segments
# run through the level-batched kernels (`fwd1_kernel!`/`bwd1_kernel!`, vendor
# dense calls for fronts wider than MW). `det`: the deterministic forward sweep of
# deterministic_mode = 1 (no atomics; the update buffers `ws.U` of PLAN §2.5, pulled by the parents)
function algo2_solve!(x, s, b, inv11, inv_ptr, fz, det::Bool = false)
    S, N, ws = s.symbolic, s.numeric, s.workspace
    bm = SDS.batch_map(N; nrhs = 1)
    SDS.permute_rhs!(ws.Y, b, S.perm; transposed = false, bm)
    algo2_sweeps!(s, inv11, inv_ptr, fz, det)
    SDS.unpermute_solution!(x, ws.Y, S.perm; transposed = false, bm)
    return x
end

# the sweeps of `algo2_solve!` on the solve workspace `Y` (the permuted right-hand side in,
# the permuted solution out): the part a CUDA graph replays (T34). The three parts are the
# solve sub-phases of `solve_alg = "algo1"` ("solve_fwd", "solve_diag", "solve_bwd", T36)
function algo2_sweeps!(s, inv11, inv_ptr, fz, det::Bool = false)
    algo2_forward!(s, inv11, inv_ptr, fz, det)
    algo2_diagonal!(s)
    algo2_backward!(s, inv11, inv_ptr, fz)
    return nothing
end

# the forward sweep of the fused solve: the segments by the dependency-counter kernel, the
# plan entries outside them by `fwd1_kernel!` and the vendor dense calls (fronts wider than MW)
function algo2_forward!(s, inv11, inv_ptr, fz, det::Bool = false)
    S, N, ws = s.symbolic, s.numeric, s.workspace
    plan = ws.plan
    nodes = S.schedule.group_nodes
    widths = S.schedule.width
    p = SDS._solve_impls(N, S, :auto)
    backend = KA.get_backend(ws.Y)
    Yv = vec(ws.Y)
    Uv = vec(ws.U)
    segs, _ = fz
    args(first, count) = (Yv, N.factor, inv11, inv_ptr, S.group_nodes, first, count, S.subtree_ptr,
                          S.subtree_nodes, S.super_ptr, S.rowptr, S.rowval, S.front_ptr, S.front_nrows,
                          S.front_ncols)
    fargs(first, count) = (args(first, count)..., Uv, S.child_ptr, S.child_list, S.relind_ptr, S.relind)
    fwd1!(a, b_) = det ?
        fwd1_kernel!(backend, WG)(fargs(a, b_ - a + 1)..., Val(false), Val(WG), Val(true); ndrange = WG * (b_ - a + 1)) :
        fwd1_kernel!(backend, WG)(fargs(a, b_ - a + 1)..., Val(false), Val(WG), Val(false); ndrange = WG * (b_ - a + 1))
    # SOLVE_SUBTREES entries are run by run_fused!/run_fused_bwd!, so the entries launched here never pass
    # SUB = true to fwd1_kernel!/bwd1_kernel!.
    fwd_entry(k) = begin
        a, b_ = plan.first[k], plan.last[k]
        if plan.kind[k] == SDS.SOLVE_SUBTREES
        elseif plan.kind[k] == SDS.SOLVE_DENSE
            # deterministic: the launch also pulls the children of the fronts wider than MW
            (det || any(widths[nodes[q]] <= MW for q in a:b_)) && fwd1!(a, b_)
            for q in a:b_
                widths[nodes[q]] > MW && SDS._fwd_dense!(ws, S, N, nodes[q], 1, det, p, SDS._SV_CHOLESKY)
            end
        else
            fwd1!(a, b_)
        end
    end
    k = 1
    for (fzk, wg) in segs
        while k < fzk.i1
            fwd_entry(k); k += 1
        end
        run_fused!(fzk, wg, Yv, s, inv11, inv_ptr, backend, det)
        k = fzk.i2 + 1
    end
    while k <= length(plan.kind)
        fwd_entry(k); k += 1
    end
    return nothing
end

# the diagonal step between the unit-triangular sweeps of LDLᵀ (1×1 pivots only: D⁻¹ on the first
# column of `Y`); nothing for Cholesky
function algo2_diagonal!(s)
    S, N, ws = s.symbolic, s.numeric, s.workspace
    if SDS._is_ldlt_structure(S.structure)
        n = S.n
        Yv = vec(ws.Y)
        if length(Yv) == n
            Yv ./= view(N.d, 1:n)
        else                                     # a workspace sized for more right-hand sides
            view(Yv, 1:n) ./= view(N.d, 1:n)
        end
    end
    return nothing
end

# the backward sweep of the fused solve (the forward plan in reverse)
function algo2_backward!(s, inv11, inv_ptr, fz)
    S, N, ws = s.symbolic, s.numeric, s.workspace
    plan = ws.plan
    nodes = S.schedule.group_nodes
    widths = S.schedule.width
    p = SDS._solve_impls(N, S, :auto)
    backend = KA.get_backend(ws.Y)
    Yv = vec(ws.Y)
    segs, _ = fz
    args(first, count) = (Yv, N.factor, inv11, inv_ptr, S.group_nodes, first, count, S.subtree_ptr,
                          S.subtree_nodes, S.super_ptr, S.rowptr, S.rowval, S.front_ptr, S.front_nrows,
                          S.front_ncols)
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
            bwd1_kernel!(backend, WG)(args(a, b_ - a + 1)..., Val(false), Val(WG); ndrange = WG * (b_ - a + 1))
        end
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
    return nothing
end

# ---------------------------------------------------------------------------
# The partitioned-inverse fused solve as one object (T36): the L11 inverses of an analysis
# (`build_inverse`), its dependency-counter solve segment (`build_fused`) and the optional
# CUDA-graph replay of the sweeps. The one solve driver of the fused solve, shared by
# `DirectSolver` (`solve_alg = "algo1"`, after the stock factorization: the factor in the stock
# layout is all the solve reads) and `FusedSolver` (after the fused factorization).
mutable struct FusedSolve{VT, VI, IF, FZ}
    inv11::VT                # per-front L11 inverses (`build_inverse`)
    inv_ptr::VI
    invert!::IF              # recomputes them from the current factor (one launch)
    fz::FZ                   # `([(segment, workgroup)], true)`, the segments of `algo2_solve!`
    bytes::Int64             # device bytes: inverses, offsets, block lists, segment counters
    graph::Bool              # replay the sweeps from a captured graph (FusedSolver, `graph_max_n`)
    gsolve::Any              # the captured sweeps (`nothing`: not yet; `false`: not capturable)
    gdet::Bool               # the deterministic_mode the graph was captured with
    current::Bool            # the inverses are those of the current factor (DirectSolver's bookkeeping)
end

# the fused solve of the analysis of `s` (a DirectSolver, or anything with its `symbolic`,
# `numeric` and `workspace`): inverse storage and the dependency-counter segments: the merged one
# from the first plan entry (workgroups of `workgroup` work items) and, with `all_segments`, one
# more per later run of plan entries whose fronts are all of width <= MW (between the fronts wider
# than MW near the root, which stay on the vendor dense sweeps; workgroups of `upper_workgroup`:
# these runs hold few, wide fronts). Without `all_segments` (FusedSolver) the entries after the
# first segment run on the level-batched inverse kernels. Raises NotSupportedError when no plan
# entry has fronts of width <= MW, or a front wider than MW is not on the regime-C path
function fused_solve_plan(s; workgroup::Int, graph::Bool = false, all_segments::Bool = false,
                          upper_workgroup::Int = MW)
    _check_solve_workgroup(workgroup)
    _check_solve_workgroup(upper_workgroup)
    inv11, inv_ptr, invertf, bytes = build_inverse(s)
    fz1 = build_fused(s; kstart = 1, wcap = MW)
    fz1 === nothing && throw(SDS.NotSupportedError("fused solve: no plan entry with fronts of width <= $MW"))
    segs = [(fz1, workgroup)]
    while all_segments
        fzk = build_fused(s; kstart = last(segs)[1].i2 + 1, wcap = MW, merged = false)
        fzk === nothing && break
        push!(segs, (fzk, upper_workgroup))
    end
    for (fzk, _) in segs
        bytes += sum(k -> _device_bytes(fzk[k]), keys(fzk))
    end
    return FusedSolve(inv11, inv_ptr, invertf, (segs, true), bytes, graph, nothing, false, false)
end

# the three sweeps on the permuted right-hand side in `s.workspace.Y`; with `fs.graph` the first
# call per `det` runs eagerly (it compiles the kernels) and captures the graph, later calls replay it
function fused_sweeps!(fs::FusedSolve, s, det::Bool)
    if fs.graph
        det == fs.gdet || (fs.gsolve = nothing)    # a graph is captured for one forward-sweep variant
        if fs.gsolve === nothing
            algo2_sweeps!(s, fs.inv11, fs.inv_ptr, fs.fz, det)
            inv11, inv_ptr, fz = fs.inv11, fs.inv_ptr, fs.fz
            backend = KA.get_backend(s.workspace.Y)
            g = det ? SDS.capture_graph(backend, () -> algo2_sweeps!(s, inv11, inv_ptr, fz, true)) :
                      SDS.capture_graph(backend, () -> algo2_sweeps!(s, inv11, inv_ptr, fz, false))
            fs.gsolve = something(g, false)
            fs.gdet = det
            return nothing
        end
        if fs.gsolve !== false
            fs.gsolve()
            return nothing
        end
    end
    algo2_sweeps!(s, fs.inv11, fs.inv_ptr, fs.fz, det)
    return nothing
end

"""
    fused_solve!(fs, s, X, B, det; transposed = false, pin, scale_in, pout, scale_out) -> X

The partitioned-inverse fused solve `X = A⁻¹ B` for one right-hand side with the factor of `s`
(a `DirectSolver`): permute `B` into the solve workspace (`pin`, `scale_in`: the permutation and
scaling of the solve phases, `solve_io`; default the analysis permutation without scaling), the
fused sweeps of `fs` (`det`: the pull forward sweep of `deterministic_mode = 1`), and the inverse
permutation into `X`. Asynchronous; the solve driver of both `DirectSolver` with
`solve_alg = "algo1"` and `FusedSolver`.
"""
function fused_solve!(fs::FusedSolve, s, X, B, det::Bool; transposed::Bool = false, pin = s.symbolic.perm,
                      scale_in = nothing, pout = s.symbolic.perm, scale_out = nothing)
    ws = s.workspace
    bm = SDS.batch_map(s.numeric; nrhs = 1)
    SDS.permute_rhs!(ws.Y, B, pin; transposed, bm, scale = scale_in)
    fused_sweeps!(fs, s, det)
    SDS.unpermute_solution!(X, ws.Y, pout; transposed, bm, scale = scale_out)
    return X
end

# the hooks of the handle layer (src/solver.jl, src/solve/refinement.jl): the fused solve of an
# analysis with solve_alg = "algo1", its inverses, its sweeps on (ws, S, N) and the sub-phases
SDS._fused_invert!(fs::FusedSolve) = (fs.invert!(); fs.current = true; nothing)
SDS._fused_stale!(fs::FusedSolve) = (fs.current = false; nothing)
SDS._fused_current(fs::FusedSolve) = fs.current
SDS._fused_bytes(fs::FusedSolve) = fs.bytes
_sv(ws, S, N) = (; symbolic = S, numeric = N, workspace = ws)
SDS._fused_sweeps!(fs::FusedSolve, ws, S, N, det::Bool) = fused_sweeps!(fs, _sv(ws, S, N), det)
SDS._fused_forward!(fs::FusedSolve, ws, S, N, det::Bool) = algo2_forward!(_sv(ws, S, N), fs.inv11, fs.inv_ptr, fs.fz, det)
SDS._fused_diagonal!(fs::FusedSolve, ws, S, N) = algo2_diagonal!(_sv(ws, S, N))
SDS._fused_backward!(fs::FusedSolve, ws, S, N) = algo2_backward!(_sv(ws, S, N), fs.inv11, fs.inv_ptr, fs.fz)
SDS._fused_solve_plan(s::SDS.DirectSolver) =
    fused_solve_plan(s; workgroup = default_tuning(s.backend).solve_workgroup, all_segments = true)

const SYRK_IN = 64
const TILE = 32
const NL64 = 2080                 # packed 64x64 lower triangle

# ---------------------------------------------------------------------------
# no-pivot LDLᵀ mode of the fused kernel (real element types): the factor stores
# the UNIT lower panels and D goes to `d` (global factor-column indexing), the
# stock LDL layout, so the stock solve and the fused solve (one extra D pass
# between the sweeps) both work on it. A pivot with |d| < pivot_epsilon is replaced
# by ±pivot_epsilon (`_ldl_static_pivot`, #194); a pivot that stays zero (epsilon 0)
# or is not finite sets info[v] to the local column and the front's consumers skip,
# exactly the Cholesky failure path;
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

# the static pivot of the fused LDLᵀ (#194), the rule of the stock kernel (`_lt_select!`) with
# `pivot_type = 'N'`: a pivot with |d| < eps, or d == 0, is replaced by sign·eps, where sign is the
# requested `psg[g]` (`pivot_sign` in factor-column order) when nonzero, else the sign of d (+ for 0).
# Returns (pivot, replaced). A pure function of (d, eps, psg[g]): the factorization stays bitwise
# deterministic. eps = 0 returns a zero pivot as ±0 (the caller fails the front, as before #194);
# NaN and Inf are not "tiny" and come back unchanged, so the caller's finiteness test fails them
# (the stock kernel replaces a NaN pivot and accepts an Inf one).
@inline function _ldl_static_pivot(dk::R, eps::R, psg, g) where {R}
    if abs(dk) < eps || iszero(dk)
        s = @inbounds psg[g]
        return (s != 0 ? R(s) : (dk < zero(R) ? -one(R) : one(R))) * eps, true
    end
    return dk, false
end

@inline function _subtree_ldl_update!(buf, ctl, pv, j, li, d, piv, pivot_kind, eps, psg, ::Val{WG}) where {WG}
    @inbounds begin
        T = eltype(buf)
        IT = eltype(ctl)
        f = ctl[SDS._ST_F]
        lf = ctl[SDS._ST_LF] - 1
        if ctl[SDS._ST_STATUS] == 0
            g = Int(ctl[_SP_C0]) + j - 1
            dk, rep = _ldl_static_pivot(real(buf[lf + SDS._packed(j, j, f)]), eps, psg, g)
            if isfinite(dk) && !iszero(dk)   # NaN/Inf must fail like potrf, not "factor"
                if li == 1
                    d[g] = dk
                    piv[g] = Int32(g)
                    pivot_kind[g] = rep ? SDS.PIVOT_KIND_PERTURBED : SDS.PIVOT_KIND_1X1
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
                                     @Const(child_list), @Const(relind_ptr), @Const(relind), eps, @Const(psg),
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
                                 SDS._mview(pivot_kind, SDS._bm_gmember(bm, G), bm.nbatch), eps, psg, Val(WG))
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

# the pivot of column j (replaced by `_ldl_static_pivot`) goes to d, pivot_kind and pv[1]; every thread
# derives it from the sh diagonal itself, and `_front_ldl_scale!` writes a replaced pivot back into the
# diagonal slot (read by the in-panel TRSM and the SYRK) after the barrier, when no thread reads it here
@inline function _front_ldl_update!(sh, st, pv, j, v, li, front_ncols, d, g0, eps, psg, pivot_kind,
                                    ::Val{WG}) where {WG}
    @inbounds begin
        w = Int(front_ncols[v])
        if j <= w && st[1] == Int32(0)
            dk, rep = _ldl_static_pivot(real(sh[SDS._packed(j, j, 64)]), eps, psg, g0 + j)
            if isfinite(dk) && !iszero(dk)   # NaN/Inf must fail like potrf, not "factor"
                if li == 1
                    d[g0 + j] = dk
                    pivot_kind[g0 + j] = rep ? SDS.PIVOT_KIND_PERTURBED : SDS.PIVOT_KIND_1X1
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
            li == 1 && (sh[SDS._packed(j, j, 64)] = dk)   # the (replaced) pivot; the loop below reads i, k > j only
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
const ROLE_EA = Int8(5)          # extend-add, parent pull: (parent, block of its columns), see build_fused_plan
const ROLE_WPANEL = Int8(6)       # wide fronts in-kernel: blocked 64-panel Cholesky
const ROLE_WTRSM = Int8(7)        #   (rb = panel k | rb = chunk, rc = k | rc = k*1024 + tjj)
const ROLE_WTILE = Int8(8)
const ROLE_WSIG = Int8(9)

# ordered wide tiles: wait until every 32 x 32 cell of front v (cld(f, 32) cells per side, packed lower
# triangle from wcbase[v] + 1) that the tile with first row i0 + 1 and first column j0 + 1 touches has
# seen `target` panels; thread 1 only
@inline function _wcell_wait!(wcell, wcbase, v, f, i0, j0, target, scratch, scap)
    nc = cld(f, TILE)
    b = Int(wcbase[v])
    for J in (j0 ÷ TILE + 1):(min(j0 + TILE, f) - 1) ÷ TILE + 1
        for I in max(J, i0 ÷ TILE + 1):(min(i0 + TILE, f) - 1) ÷ TILE + 1
            _spin_until_eq!(wcell, b + SDS._packed(I, J, nc), target, scratch, scap)
        end
    end
    return nothing
end

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
                                    @Const(wbase), wupd, wpdone2, wtrsml, wcbl, scap, epsc, d, @Const(super_ptr),
                                    @Const(pl_c), @Const(pl_q0), @Const(pl_q1), wcell, @Const(wcbase),
                                    @Const(psg), pivot_kind, ::Val{LDL}, ::Val{WG}) where {LDL, WG}
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
            # extend-add, parent pull (T35): one block of the parent's columns; the children in child_list
            # order, each over the packed range of its columns that land in the block. Plain adds: the
            # blocks of one parent are disjoint, the entries of one child too (relind is injective), and the
            # children follow each other after a barrier, so every entry sums its children in a fixed order
            v = Int(ra[G])
            e0 = Int(rb[G])
            ne = Int(rc[G])
            fp = Int(front_nrows[v])
            wp = Int(front_ncols[v])
            mp = fp - wp
            pp = Int(front_ptr[v])
            cp = Int(cb2[v])
            for e in e0:(e0 + ne - 1)
                c = Int(pl_c[e])
                if li == 1
                    _spin_until_eq!(cb_done, c, Int32(1), fscr, scap)
                end
                @synchronize
                SDS.device_fence(factor)
                cpriv = Int(cb2[c])
                cb = cpriv > 0 ? cpriv : Int(cb_ptr[c])
                mc = Int(front_nrows[c]) - Int(front_ncols[c])
                r0 = Int(relind_ptr[c]) - 1
                for q in (Int(pl_q0[e]) + li - 1):WG:(Int(pl_q1[e]) - 1)
                    ii, jj = SDS._tri_decode(q, mc)
                    ri = Int(relind[r0 + ii])
                    rj = Int(relind[r0 + jj])
                    x = cpriv > 0 ? pcb[cb + q] : stack[cb + q]
                    if rj <= wp
                        factor[pp + (rj - 1) * fp + ri - 1] += x
                    else
                        pcb[cp + SDS._packed(ri - wp, rj - wp, mp) - 1] += x
                    end
                end
                @synchronize
            end
            SDS.device_fence(factor)      # ALL threads (gfx9)
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
            SDS.device_fence(factor)
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
                    _front_ldl_update!(sh, st, piv, j, v, li, front_ncols, d, g0, epsc, psg, pivot_kind, Val(WG))
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
                SDS.device_fence(factor)      # ALL threads: each wave drains its own stores (gfx9)
                @synchronize
                if li == 1
                    Atomix.@atomic cb_done[v] += Int32(1)
                    pv = Int(snparent[v])
                    pv > 0 && (Atomix.@atomic arrived[pv] += Int32(1))
                end
            else
                SDS.device_fence(factor)      # ALL threads (gfx9)
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
            SDS.device_fence(factor)
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
            SDS.device_fence(factor)      # ALL threads: each wave drains its own stores (gfx9)
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
            SDS.device_fence(factor)
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
            SDS.device_fence(factor)      # ALL threads: each wave drains its own stores (gfx9)
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
            SDS.device_fence(factor)
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
                        gj = Int(super_ptr[v]) - 1 + c0 + j
                        dj, rep = _ldl_static_pivot(dj, epsc, psg, gj)
                        if st[1] == Int32(0) && isfinite(dj) && !iszero(dj)
                            piv[1] = dj                      # d stays on the diagonal
                            sh[SDS._packed(j, j, 64)] = dj
                            d[gj] = dj
                            pivot_kind[gj] = rep ? SDS.PIVOT_KIND_PERTURBED : SDS.PIVOT_KIND_1X1
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
            SDS.device_fence(factor)      # ALL threads: each wave drains its own stores (gfx9)
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
            SDS.device_fence(factor)
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
            SDS.device_fence(factor)      # ALL threads: each wave drains its own stores (gfx9)
            @synchronize
            if li == 1
                Atomix.@atomic wtrsml[sl] += Int32(-1)
            end
        elseif r == ROLE_WTILE
            v = Int(ra[G]); tii = Int(rb[G])
            k = Int(rc[G]) ÷ 1024; tjj = Int(rc[G]) % 1024
            sl = Int(wbase[v]) + k
            f = Int(front_nrows[v]); w = Int(front_ncols[v])
            m = f - w
            c0 = (k - 1) * 64
            kb = min(64, w - c0)
            p0 = Int(front_ptr[v])
            rb0 = c0 + kb
            i0 = rb0 + (tii - 1) * TILE
            j0 = rb0 + (tjj - 1) * TILE
            if li == 1
                _spin_until_eq!(wtrsml, sl, Int32(0), fscr, scap)
                if k > 1
                    # ordered panels: every 32 x 32 cell of the front (absolute rows and columns) this
                    # tile touches holds the updates of panels 1..k-1 (the cell's counter). Below the
                    # last panel the tile is one cell; the last panel's tiles may straddle four
                    _wcell_wait!(wcell, wcbase, v, f, i0, j0, Int32(k - 1), fscr, scap)
                end
            end
            @synchronize
            SDS.device_fence(factor)
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
                        if jj <= w                   # the panels in order (the cell counters): plain
                            factor[p0 + (jj - 1) * f + ii - 1] -= a
                        elseif c0cb > 0
                            pcb[c0cb + SDS._packed(ii - w, jj - w, m) - 1] -= a
                        end
                    end
                end
            end
            @synchronize
            SDS.device_fence(factor)      # ALL threads: each wave drains its own stores (gfx9)
            @synchronize
            if li == 1
                # ordered panels: this panel is applied to the tile's cell (a tile below the last
                # panel is exactly one cell; nothing waits on the last panel's)
                64 * k < w &&
                    (Atomix.@atomic wcell[Int(wcbase[v]) + SDS._packed(i0 ÷ TILE + 1, j0 ÷ TILE + 1, cld(f, TILE))] += Int32(1))
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
                SDS.device_fence(factor)
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

# the C-group fronts of the fused plan: zero the panel, then scatter A into it, one
# workgroup per front and one launch for every C group (T34; before: panel_zero_kernel!
# plus one scatter_A! launch per C group)
@kernel function czero_scatter_kernel!(factor, @Const(nzval), @Const(amap), @Const(amap_ptr), @Const(amap_src),
                                       @Const(list), @Const(front_ptr), @Const(front_nrows), @Const(front_ncols),
                                       ::Val{WG}) where {WG}
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
    @synchronize
    @inbounds begin
        v2 = Int(list[G])
        SDS._scatter_front!(factor, nzval, amap, amap_ptr, amap_src, v2, SDS._member_shift(front_ptr, v2, 1, 1),
                            li, Val(WG))
    end
end

@kernel function signal_kernel!(cb_done, arrived, @Const(snparent), v)
    li = @index(Local, Linear)
    @inbounds if li == 1
        SDS.device_fence(cb_done)
        Atomix.@atomic cb_done[v] += Int32(1)
        pv = Int(snparent[v])
        pv > 0 && (Atomix.@atomic arrived[pv] += Int32(1))
    end
end

# subtree launcher with a custom (sorted) list, local memory sized by eltype
# (`psg`: the LDL plan's `pivot_sign` in factor-column order, `epsc` the LDL pivot epsilon; see `build_fused_plan`)
function launch_subtrees_sorted!(N, S, nzval, list, lb, backend, ldl::Bool = false, epsc::Real = 0.0, psg = nothing)
    SDS._with_local_bytes(Int(lb)) do LBv
        _sorted_subtree_launch(N, S, nzval, list, Val(ldl), LBv, Val(SDS.SUBTREE_WORKGROUP), backend, epsc, psg)
    end
end
function _sorted_subtree_launch(N::SDS.Numeric{T}, S, nzval, list, ::Val{LDL}, ::Val{LB}, ::Val{WG},
                                backend, epsc::Real = 0.0, psg = nothing) where {T, LDL, LB, WG}
    bm = SDS.batch_map(N; first = 1)
    if LDL
        # the schedule sized the "S" budget classes with the LDLT reserve: match it
        k! = subtree_ldl_kernel!(backend, WG)
        k!(N.factor, N.stack, N.info, N.d, N.piv, N.pivot_kind, nzval, S.amap, S.amap_ptr, S.amap_src,
           list, bm, S.subtree_ptr, S.subtree_nodes, S.super_ptr, S.front_ptr, S.front_nrows, S.front_ncols,
           S.cb_ptr, S.local_front, S.local_cb, S.child_ptr, S.child_list, S.relind_ptr, S.relind,
           real(T)(epsc), psg, Val((LB - SDS.SUBTREE_LOCAL_RESERVE_LDLT) ÷ sizeof(T)), Val(WG);
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
                          wgf::Int = WGF, scap::Integer = 128, ea_chunk::Integer = 16384,
                          host_min::Integer = typemax(Int))
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
    ldl && host_min < typemax(Int) && !wides_potrf &&
        throw(SDS.InvalidValueError("fused LDL plan: host-wide fronts (host_min = $host_min) need wides_potrf = true"))
    ldl && !inkernel_wides && !wides_potrf &&
        throw(SDS.InvalidValueError("fused LDL plan: host-wide fronts need wides_potrf = true " *
                                    "(or inkernel_wides = true)"))
    # the pivot epsilon (Cholesky: the narrow kernels'; LDL: static pivot replacement, #194) and, for LDL,
    # `pivot_sign` in factor-column order (the stock kernel reads psign[perm[g]] under static pivoting)
    pivot_eps = Float64(SDS.resolved_pivot_epsilon(s.options, real(T)))
    ldl && wides_potrf && pivot_eps > 0 &&
        throw(SDS.NotSupportedError("fused LDL plan: wides_potrf cannot replace pivots; set pivot_epsilon = 0"))
    ps = s.options.pivot_sign
    if ldl && ps !== nothing
        length(ps) == S.n ||
            throw(SDS.InvalidValueError("pivot_sign has $(length(ps)) entries, the matrix has $(S.n) rows"))
        psg = _dvec(backend, Int8.(ps[Array(S.perm)]))
    else
        psg = KA.zeros(backend, Int8, ldl ? max(S.n, 1) : 1)   # Cholesky: never read
    end
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

    # the extend-add of a C-group parent (T35: deterministic, the parent pull; it replaced the atomic
    # push of every child's CB chunks into the parent, which summed in arrival order): per parent a
    # ROLE_EA for each block of its columns (about ea_chunk / 2 child entries), with the list (child,
    # packed range of the child's CB) of its children in child_list order. relind is increasing, so the
    # child columns that land in a block of parent columns are contiguous, and so are their packed entries.
    pl_c = Int32[]; pl_q0 = Int32[]; pl_q1 = Int32[]
    relp_h = Array(S.relind_ptr)
    rel_h = Array(S.relind)
    emit_pull(v, cs) = begin
        fpv = rows_h[v]
        cost = zeros(Int, fpv)
        for c in cs
            mc = rows_h[c] - width_h[c]
            r0 = Int(relp_h[c]) - 1
            for jj in 1:mc
                cost[rel_h[r0 + jj]] += mc - jj + 1
            end
        end
        target = max(1, Int(ea_chunk) ÷ 2)
        Ja = 1
        while Ja <= fpv
            Jb = Ja
            acc = cost[Ja]
            while Jb < fpv && acc + cost[Jb + 1] <= target
                Jb += 1
                acc += cost[Jb]
            end
            e0 = length(pl_c) + 1
            for c in cs
                mc = rows_h[c] - width_h[c]
                r0 = Int(relp_h[c])
                rr = view(rel_h, r0:(r0 + mc - 1))
                a = searchsortedfirst(rr, Ja)
                b = searchsortedlast(rr, Jb)
                a <= b || continue
                q1 = SDS._packed(mc, b, mc)
                q1 <= typemax(Int32) ||
                    throw(SDS.NotSupportedError("fused plan: a contribution block of $mc rows exceeds the Int32 " *
                                                "packed range of the extend-add"))
                push!(pl_c, Int32(c)); push!(pl_q0, Int32(SDS._packed(a, a, mc) - 1)); push!(pl_q1, Int32(q1))
            end
            ne = length(pl_c) - e0 + 1
            if ne > 0
                push!(role, ROLE_EA); push!(ra, Int32(v)); push!(rb, Int32(e0)); push!(rc, Int32(ne))
                ea_left[v] += 1
            end
            Ja = Jb + 1
        end
        return nothing
    end
    emit_ea(v) = begin
        cs = Int[]
        for kc in chp[v]:(chp[v + 1] - 1)
            c = chl[kc]
            cbp[c] > 0 || continue
            if wt_h[c] == v
                ea_left[v] += 1                  # settled by the child's last write-through tile
                continue
            end
            push!(cs, c)
        end
        isempty(cs) || emit_pull(v, cs)
        return nothing
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
    # ordered wide tiles (T35): one Int32 "panels applied" counter per 32 x 32 cell of the lower
    # triangle of each in-kernel wide front; the tiles of panel k wait for panel k - 1 on their cell
    # and subtract with plain stores (before: atomic subtractions of every panel in arrival order)
    wcbase_h = zeros(Int32, ns)
    nwcells = 0
    emit_wide(v) = begin
        f = rows_h[v]; w = width_h[v]
        K = cld(w, 64)
        wbase_h[v] = Int32(nwslots)
        wcbase_h[v] = Int32(nwcells)
        nwcells += cld(f, TILE) * (cld(f, TILE) + 1) ÷ 2
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

    segments = Tuple{Int, Int, Vector{Int}}[]   # (base, count, wides-at-boundary)
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
                    # per-front strategy (T34): a front wider than `host_min` stays on the host
                    # vendor path at a segment boundary, the others are factored in-kernel
                    foreach(emit_wide, filter(v -> width_h[v] <= host_min, gwides))
                    gwides = filter(v -> width_h[v] > host_min, gwides)
                end
                if !(inkernel_wides && merge_segments) || !isempty(gwides)   # merged: one launch, no split
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
    ssub = map(eachindex(plan.sub_first)) do k    # typed: the launches dispatch statically
        ids = [nodes[q] for q in plan.sub_first[k]:plan.sub_last[k]]
        sort!(ids; by = e -> stp2[e + 1] - stp2[e], rev = true)
        _dvec(backend, ids)
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
            wcb = KA.zeros(backend, Int32, ns), wgf, scap = Int32(scap), ldl,
            # the kernel object and its Val parameters, typed in the plan so that the launch
            # in refact_fused_launches! is statically dispatched (T34: no boxing of its arguments)
            kfact = fused_fact_kernel!(backend, wgf), ldlv = Val(ldl), wgv = Val(wgf),
            # the extend-add lists and the wide-tile cell counters (T35)
            pl_c = _dvec(backend, pl_c), pl_q0 = _dvec(backend, pl_q0),
            pl_q1 = _dvec(backend, pl_q1), wcell = KA.zeros(backend, Int32, max(nwcells, 1)),
            wcbase = _dvec(backend, wcbase_h), nwcells,
            # captured at plan build: a later setparam!("pivot_epsilon") or ("pivot_sign") changes the
            # stock path but not an already-built fused plan (rebuild the plan to change it). Cholesky:
            # the pivot epsilon of the narrow kernels; LDL: the static-pivot epsilon (#194)
            pivot_eps, psg,
            ldl_potrf = ldl && wides_potrf,
            sptr_h = Array(S.super_ptr))
end

# the fused factorization of `nzval`; returns the info (one host read after the launches of
# `refact_fused_launches!`). The inverses of the solve are the caller's (`build_inverse`).
function refact_fused!(s, nzval, fp)
    refact_fused_launches!(s, nzval, fp)
    return SDS._numeric_info!(s.numeric, s.symbolic, true)
end

# the device work of `refact_fused!`: no host synchronization, the part a CUDA graph
# replays when no front takes the host vendor path (T34)
function refact_fused_launches!(s, nzval, fp)
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
    fp.nwcells > 0 && fill!(fp.wcell, Int32(0))
    for k in eachindex(plan.sub_first)
        launch_subtrees_sorted!(N, S, nzval, fp.ssub[k], plan.sub_local[k], backend, fp.ldl, fp.pivot_eps, fp.psg)
    end
    fill!(fp.pcb, zero(eltype(fp.pcb)))
    isempty(fp.czero) ||                               # zero + scatter A into every C-group front (panel only)
        czero_scatter_kernel!(backend, WGF)(N.factor, nzval, S.amap, S.amap_ptr, S.amap_src, fp.czero, S.front_ptr,
                                            S.front_nrows, S.front_ncols, Val(WGF); ndrange = WGF * length(fp.czero))
    for (base, count, gwides) in fp.segments
        count > 0 &&
            fp.kfact(N.factor, N.stack, fp.pcb, N.info, nzval, S.amap, S.amap_ptr,
                                             S.amap_src, fp.cb2, fp.wt, fp.role, fp.ra, fp.rb, fp.rc, fp.nchild,
                                             fp.arrived, fp.ea, fp.pdone, fp.trsm, fp.tiles, fp.tiles_base, fp.cbd,
                                             S.snparent, S.front_ptr, S.front_nrows, S.front_ncols,
                                             S.cb_ptr, S.child_ptr, S.child_list, S.relind_ptr, S.relind,
                                             Int32(base), fp.wbase, fp.wupd, fp.wpd, fp.wtl, fp.wcb,
                                             fp.scap, real(eltype(N.factor))(fp.pivot_eps), N.d,
                                             S.super_ptr, fp.pl_c, fp.pl_q0, fp.pl_q1, fp.wcell,
                                             fp.wcbase, fp.psg, N.pivot_kind, fp.ldlv, fp.wgv;
                                             ndrange = fp.wgf * count)
        for v in gwides                              # host-wide fronts at the segment boundary
            if fp.ldl && fp.ldl_potrf
                _factor_panel_c_ldl_potrf!(N, S, fp, v, p, backend)
            else
                SDS._factor_panel_c!(N.factor, fp.pcb, N.work, N.info, v, S.layout.panel_ptr[v],
                                     S.schedule.rows[v], S.schedule.width[v], fp.cb2_h[v], p, fp.pivot_eps)
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
    # d (`getparam(F, "inertia")` of a FusedSolver), never from these stats.
    SDS.cholesky_stats!(N, S)
    return nothing
end

# ---------------------------------------------------------------------------
# FusedSolver: the fused pipeline behind the handle-layer verbs of PLAN §3.1 (T33).
# The recipe below: an inner DirectSolver (tuning, analysis, factorization, a
# solve), then the three plans. `KKTSolver` (src/kkt.jl) builds its fused pipeline
# through it.

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

# the parameters `setparam!` accepts on a FusedSolver (read by the next analysis; deterministic_mode by the
# next solve)
const FUSED_SETTABLE = ("user_perm", "reordering_alg", "nd_nseps", "nd_seed", "pivot_epsilon", "pivot_sign",
                        "matching_alg", "deterministic_mode")

# the analysis data parameters `getparam` forwards to the inner DirectSolver
const FUSED_FORWARDED = ("perm_reorder_row", "perm_reorder_col", "perm_row", "perm_col", "lu_nnz", "flops",
                         "nsuperpanels", "memory_estimates", "perm_matching", "scale_row", "scale_col")

"""
    FusedSolver(A, structure, view; tuning = nothing, index = 'O')
    FusedSolver(rowptr, colval, nzval, structure, view; tuning = nothing, index = 'O')

The fused pipeline of `SparseDirectSolver.Fused` (dependency-counter
factorization mega-kernel and partitioned-inverse solve, the fused pipeline of
[`KKTSolver`](@ref SparseDirectSolver.KKTSolver)) as an `AbstractDirectSolver`, driven by the same verbs
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
  pivots may take both signs), so the in-kernel wide-front limit applies to it
  on every part: a front with more than `1023 · 32` rows below a panel raises
  `NotSupportedError` at analysis.
* `"solve"`: `X = A⁻¹ B` with the partitioned-inverse solve, for a vector (or
  an `n × 1` matrix) on the solver's backend; `X === B` is allowed. The solve
  is the one of `DirectSolver`'s `solve_alg = "algo1"` (one driver, T36).

`setparam!` accepts `"user_perm"`, `"reordering_alg"`, `"nd_nseps"`,
`"nd_seed"` and `"pivot_epsilon"` (the Cholesky pivot epsilon; the fused
LDLᵀ factors with epsilon `0`), all read by the next analysis, and
`"deterministic_mode"`, read at every solve. The factorization is
deterministic under either mode: the children of a parent add in
`child_list` order and the panels of an in-kernel wide front apply in order,
with plain adds, so factor and `D` are bitwise reproducible for the same
values. `deterministic_mode = 0` (the default) scatters the forward-solve
updates with atomic adds, so the solution can differ in the last bits from
run to run; `deterministic_mode = 1` runs the forward sweep without atomics
(every front writes its update to its own buffer, the parent pulls its
children's buffers in order), and the solution is bitwise reproducible too.
`getparam`
returns `"info"`, `"inertia"` (`(npos, nneg)`; `"SPD"`: the positive pivots and
`0`; `"S"`: the counts of positive and negative entries of `D`), `"npivots"`
(`0` for `"S"`: the fused LDLᵀ perturbs no pivot), the analysis data
`"perm_reorder_row"`, `"perm_reorder_col"`, `"perm_row"`, `"perm_col"`,
`"lu_nnz"`, `"flops"`, `"nsuperpanels"` of the inner solver,
`"memory_estimates"` (the inner solver's, plus after the analysis the fused
plans' device buffers: slot 13 the private contribution blocks of the fused
factorization, slot 14 every other buffer of the plans, both added to slots
1, 2 and 5), and the configuration parameters.

Everything the pipeline does not do raises [`NotSupportedError`](@ref SparseDirectSolver.NotSupportedError), at
construction or in the phase, never a silent switch to the stock path: the
KernelAbstractions `CPU()` backend (the dependency-counter kernels spin on
resident workgroups, which the CPU backend runs one after another) and any
other backend without inter-workgroup forward progress (Metal), complex
element types, the structures `"HPD"`, `"H"` and `"G"`, a uniform batch
(`nbatch > 1`), `nrhs > 1`, a transposed solve (`solve_mode ≠ 0`), the solve
sub-phases (`"solve_fwd_perm"` … `"solve_refinement"`, the Schur phases),
Schur mode, `matching_alg ≠ "default"`, `ir_n_steps > 0`,
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
    fs::Any                  # the fused solve (`FusedSolve`: inverses, segments, solve graph), shared
                             # with DirectSolver's solve_alg = "algo1" (T36)
    info::Int
    stage::Int
    # CUDA-graph replay (T34, `SDSTuning.graph_max_n`): the captured refactorization
    # (`nothing`: not captured yet; `false`: not capturable), and the values it was captured with
    # (a new `nzval` buffer captures again); the solve graph lives in `fs`
    gfact::Any
    gnz::Any
end

# the fields of the fused solve before T36 (`F.inv11`, `F.inv_ptr`, `F.invert!`, `F.fz`, `F.gsolve`,
# `F.gdet`; bench/e2e/fused_probe.jl and the tests read them) now live in `F.fs`
const _FS_FIELDS = (:inv11, :inv_ptr, :invert!, :fz, :gsolve, :gdet)

function Base.getproperty(F::FusedSolver, name::Symbol)
    if name in _FS_FIELDS
        fs = getfield(F, :fs)
        fs === nothing && return name === :gdet ? false : nothing
        return getfield(fs, name)
    end
    return getfield(F, name)
end

Base.propertynames(F::FusedSolver, private::Bool = false) = (fieldnames(FusedSolver)..., _FS_FIELDS...)

function FusedSolver(A, structure, view; tuning::Union{Nothing, SDSTuning} = nothing, kwargs...)
    return _fused_solver(SDS.DirectSolver(A, structure, view; kwargs...), tuning)
end

function FusedSolver(rowptr::AbstractVector{<:Integer}, colval::AbstractVector{<:Integer}, nzval::AbstractVecOrMat,
                     structure, view; tuning::Union{Nothing, SDSTuning} = nothing, kwargs...)
    return _fused_solver(SDS.DirectSolver(rowptr, colval, nzval, structure, view; kwargs...), tuning)
end

# the tuning a solver on `backend` runs with (`nothing`: the architecture default); shared
# with the stock pipeline of `KKTSolver`
_resolved_tuning(tuning, backend) = resolve_tuning(something(tuning, default_tuning(backend)), backend)

function _fused_solver(s::SDS.DirectSolver{T, INT}, tuning::Union{Nothing, SDSTuning}) where {T, INT}
    backend = s.backend
    backend isa KA.CPU &&
        throw(SDS.NotSupportedError("FusedSolver: the KernelAbstractions CPU backend is not supported (the " *
                                    "dependency-counter kernels spin on resident workgroups); use DirectSolver"))
    SDS.forward_progress(backend) ||
        throw(SDS.NotSupportedError("FusedSolver: the backend $(nameof(typeof(backend))) guarantees no " *
                                    "inter-workgroup forward progress (PLAN §2.7); use DirectSolver"))
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
    _check_solve_workgroup(t.solve_workgroup)
    ldl = st == SDS.STRUCTURE_SYMMETRIC
    if ldl
        SDS.setparam!(s, "pivot_type", 'N')
        # no-pivot LDL never consumes 2x2 pivot pairs, and "S" analysis pays dearly for
        # detecting them on saddle systems (augmented 78k KKT, n = 2.09M: analysis 316 s
        # with the default pair detection, 64 s without -- measured on a GV100). The
        # constructor default; with matching the pairs of its cycles shape the ordering,
        # and every analysis re-applies the rule (`_fused_pairs!`)
        SDS.setparam!(s, "pivot_pairs", "none")
    end
    _apply_tuning!(s, t)
    return FusedSolver{T, INT, typeof(s)}(s, t, ldl, nothing, nothing, nothing, 0, SDS.STAGE_NONE, nothing, nothing)
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
    F.ldl || o.matching_alg == SDS.MATCHING_NONE ||
        unsupported("structure \"SPD\" with matching_alg = \"$(convert(String, o.matching_alg))\"")
    o.schur_mode == 0 || unsupported("schur_mode = 1")
    o.ir_n_steps == 0 || unsupported("ir_n_steps = $(o.ir_n_steps) (iterative refinement)")
    o.factor_precision === nothing || o.factor_precision === real(T) ||
        unsupported("factor_precision = $(o.factor_precision) for $T input")
    o.ubatch_size == 0 || o.ubatch_size == 1 || unsupported("ubatch_size = $(o.ubatch_size)")
    o.solve_alg == SDS.SOLVE_DEFAULT || unsupported("solve_alg = \"$(convert(String, o.solve_alg))\"")
    F.ldl && o.pivot_type != SDS.PIVOT_NONE &&
        unsupported("structure \"S\" with pivot_type = '$(convert(Char, o.pivot_type))'")
    F.ldl && o.pivot_epsilon_alg != SDS.PIVOT_EPSILON_DEFAULT &&
        unsupported("structure \"S\" with pivot_epsilon_alg = \"$(convert(String, o.pivot_epsilon_alg))\"")
    o.user_schur_indices === nothing || unsupported("user_schur_indices")
    return nothing
end

# the 2×2 pivot pairs of an "S" analysis: none without matching (the no-pivot LDLᵀ consumes no pair, and
# detecting them is expensive on saddle systems), the pairs of the matching's cycles with it (they shape the
# ordering, as on the stock pipeline of `KKTSolver`); evaluated at every analysis
function _fused_pairs!(F::FusedSolver)
    F.ldl || return F
    SDS.setparam!(F.inner, "pivot_pairs", F.inner.options.matching_alg == SDS.MATCHING_NONE ? "none" : "default")
    return F
end

# drop the plans of an earlier analysis (a new analysis rebuilds them)
function _clear_plans!(F::FusedSolver)
    F.nzval = F.fp = F.fs = nothing
    F.gfact = F.gnz = nothing
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
    # `wide_host_min`: the per-front wide strategy of a merged Cholesky plan (LDL keeps every
    # wide front in-kernel, see above)
    host_min = merged && !F.ldl ? t.wide_host_min : typemax(Int)
    fp = build_fused_plan(s; ldl = F.ldl, inkernel_wides = inkernel, merge_segments = merged,
                          wides_potrf = false, wgf = t.workgroup, scap = t.spin_cap, host_min)
    # the solve plan (the one of solve_alg = "algo1"): the inverses and one merged dependency-counter
    # sweep over the whole tree
    fs = fused_solve_plan(s; workgroup = t.solve_workgroup, graph = _graph_solve_ok(F))
    F.nzval = SDS._factor_values(s)
    F.fp = fp
    F.fs = fs
    F.info = 0
    F.stage = SDS.STAGE_ANALYZED
    return F
end

# the device work of the fused factorization and the inverses behind a function barrier (the
# plans are untyped fields)
function _fused_launches!(s, nzval, fp, invertf)
    refact_fused_launches!(s, nzval, fp)
    invertf()
    return nothing
end

# CUDA-graph replay (`SDSTuning.graph_max_n`, the backend's `capture_graph`): for n up to the
# tuning's bound. The captured ranges may contain vendor dense calls (the host-wide fronts'
# potrf/trsm/syrk in the factorization, the fronts wider than MW in the solve): the CUDA
# extension binds the cuSOLVER and cuBLAS handles to the capture stream before the capture
# begins, which keeps cuSOLVER 12.3.5 from invalidating it (#174)
_graph_size_ok(F::FusedSolver) = size(F, 1) <= F.tuning.graph_max_n
_graph_fact_ok(F::FusedSolver) = _graph_size_ok(F)
_graph_solve_ok(F::FusedSolver) = _graph_size_ok(F)

# the captured graph of `f` on the solver's backend, `false` when the backend has none or the
# capture fails (the phase then launches eagerly, as without the tuning)
_capture(F::FusedSolver, f) = something(SDS.capture_graph(F.inner.backend, f), false)

function _fused_refactor!(F::FusedSolver)
    s = F.inner
    # the first factorization runs eagerly (it compiles the kernels a capture records)
    if F.stage >= SDS.STAGE_FACTORIZED && _graph_fact_ok(F)
        if F.gfact === nothing || F.gnz !== F.nzval
            nz, fp, invf = F.nzval, F.fp, F.fs.invert!
            F.gfact = _capture(F, () -> _fused_launches!(s, nz, fp, invf))
            F.gnz = F.nzval
        end
        if F.gfact !== false
            F.gfact()
            return Int(SDS._numeric_info!(s.numeric, s.symbolic, true))
        end
    end
    _fused_launches!(s, F.nzval, F.fp, F.fs.invert!)
    return Int(SDS._numeric_info!(s.numeric, s.symbolic, true))
end

function _fused_factorize!(F::FusedSolver, p::SDS.Phase)
    name = convert(String, p)
    F.stage >= SDS.STAGE_ANALYZED || throw(SDS._phase_error(name, "needs \"analysis\""))
    p == SDS.PHASE_REFACTORIZATION && F.stage < SDS.STAGE_FACTORIZED &&
        throw(SDS._phase_error(name, "needs a previous \"factorization\""))
    F.nzval = SDS._factor_values(F.inner)
    F.info = _fused_refactor!(F)
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
    det = o.deterministic_mode == 1               # read at every solve, as DirectSolver does
    F.fs.graph = _graph_solve_ok(F)               # the tuning may change between solves
    # the solve driver of solve_alg = "algo1" (T36), with the matching's permutations and scalings (no extra
    # launch: they are applied inside the two permutation kernels)
    pin, sin, pout, sout = SDS.solve_io(F.inner.symbolic, SDS._solve_scaling(F.inner), false)
    fused_solve!(F.fs, F.inner, x, b, det; pin, scale_in = sin, pout, scale_out = sout)
    return F
end

function SDS.execute!(phase::AbstractString, F::FusedSolver, X, B; asynchronous::Bool = true)
    p = SDS._parse_phase(phase)
    if p == SDS.PHASE_REORDERING || p == SDS.PHASE_ANALYSIS
        _check_fused_options(F)
        _fused_pairs!(F)
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
    name == "pivot_sign" && !F.ldl &&
        throw(SDS.NotSupportedError("FusedSolver: \"pivot_sign\" applies to the LDLᵀ of structure \"S\" only"))
    if name == "matching_alg" && !F.ldl
        SDS.setparam!(F.inner, name, value)        # parses the value (bad strings: InvalidValueError)
        if F.inner.options.matching_alg != SDS.MATCHING_NONE
            F.inner.options.matching_alg = SDS.MATCHING_NONE
            throw(SDS.NotSupportedError("FusedSolver: matching_alg = $(repr(value)) applies to structure \"S\" " *
                                        "only (the fused Cholesky has no matching)"))
        end
        return nothing
    end
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
        # the replaced static pivots (#194), counted on the device at the phase boundary like the inertia
        name == "npivots" &&
            return INT(count(==(SDS.PIVOT_KIND_PERTURBED), view(F.inner.numeric.pivot_kind, 1:size(F, 1))))
        d = view(F.inner.numeric.d, 1:size(F, 1))
        return (INT(count(>(zero(T)), d)), INT(count(<(zero(T)), d)))
    end
    name == "memory_estimates" && return _fused_memory_estimates(F)
    name in FUSED_FORWARDED && return SDS.getparam(F.inner, name)
    spec.status === :output || spec.status === :solver ||
        return SDS.getparam(F.inner.options, name)
    throw(SDS.NotSupportedError("FusedSolver: the data parameter \"$name\" is not provided (use DirectSolver)"))
end

# device bytes of the device arrays among `xs` (host arrays and other values count 0; a host vector of
# device vectors, the plan's typed subtree lists, counts its elements)
_device_bytes(x::Array) = eltype(x) <: AbstractArray ? sum(_device_bytes, x; init = Int64(0)) : Int64(0)
_device_bytes(x::AbstractArray) = Int64(length(x)) * sizeof(eltype(x))
_device_bytes(x) = Int64(0)

# "memory_estimates" of a FusedSolver: the inner solver's, plus (after the analysis) slot 13, the private
# contribution blocks of the fused factorization (`fp.pcb`, #109), and slot 14, every other device buffer of
# the fused plans (role list and counters, the extend-add lists and wide-tile cell counters, the L11 inverses, the
# solve segments' counters); both are added to the device totals in slots 1, 2 and 5
function _fused_memory_estimates(F::FusedSolver)
    est = SDS.getparam(F.inner, "memory_estimates")
    F.fp === nothing && return est
    fp = F.fp
    pcb = _device_bytes(fp.pcb)
    rest = sum(k -> k === :pcb ? Int64(0) : _device_bytes(fp[k]), keys(fp)) + F.fs.bytes
    est[13] = pcb
    est[14] = rest
    for k in (1, 2, 5)
        est[k] += pcb + rest
    end
    return est
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
