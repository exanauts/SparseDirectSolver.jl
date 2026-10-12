module GpuRefine

# Device node-separator refinement: a round-based, data-parallel approximation
# of METIS's FM_2WayNodeRefine1Sided (MetisExact.jl:1911), so the uncoarsening
# walk of a bisection can stay on the device instead of round-tripping to the
# host FM at every level. This is the prerequisite for level-batched
# sub-bisections (the Batched.jl dead-end failed on exactly these host trips).
#
# Semantics relative to the serial FM:
#   * a PASS moves separator vertices to ONE side (`to`), alternating sides
#     across passes, exactly like the 1-sided FM;
#   * within a pass, moves are selected in parallel: every separator vertex
#     with gain = vwgt[v] - Σ_{u∈N(v), where[u]==other} vwgt[u] ≥ 0 moves,
#     subject to a balance budget (badmaxpwgt, METIS's cap). The serial code's
#     priority order, incremental rinfo and negative-gain hill-climb with
#     best-prefix rollback are replaced by whole-pass accept/revert: the pass
#     is kept only if it reduces the separator weight (ties: improves balance).
#   * separator validity is preserved for ANY subset of movers: all movers go
#     to the same side, and every other-side neighbor of a mover is pulled
#     into the separator in the same pass, so no side-to-side edge can appear.
#
# Known approximation costs (accepted; the quality gate is end nnz_L):
#   * concurrent movers overcount shared pulled neighbors in their gains;
#   * no negative-gain hill-climbing;
#   * under a binding balance budget the atomic claim makes the selected
#     subset nondeterministic (the budget rarely binds in the ND chain).

using SparseDirectSolver.KernelAbstractions
const KA = KernelAbstractions
using SparseDirectSolver.Atomix

export refine_rounds!, project_where!

@kernel function _project_kernel!(fwhere, @Const(cwhere), @Const(cmap))
    v = @index(Global)
    @inbounds fwhere[v] = cwhere[cmap[v] + 1]
end

# gain of moving separator vertex v to `to` (other = 1 - to); non-separator
# vertices get typemin so the selection predicate is a single compare
@kernel function _gains_kernel!(gain, @Const(xadj), @Const(adjncy), @Const(vwgt),
                                @Const(where_), other::Int32)
    v = @index(Global)
    @inbounds begin
        if where_[v] == Int32(2)
            g = Int32(vwgt[v])
            for p in (xadj[v] + 1):xadj[v + 1]
                u = adjncy[p] + 1
                where_[u] == other && (g -= vwgt[u])
            end
            gain[v] = g
        else
            gain[v] = typemin(Int32)
        end
    end
end

# candidate filter: gain ≥ 0 AND locally maximal among separator neighbors
# (ties by hashed id). Adjacent separator vertices therefore never move in
# the same pass, which removes the dominant gain-overcount (shared pulled
# neighbors of adjacent movers) and is the standard conflict rule of
# round-based refinement (Jet / mt-KaHyPar style).
@kernel function _filter_kernel!(cand, @Const(gain), @Const(xadj), @Const(adjncy),
                                 @Const(where_), seed::UInt32, thr::Int32)
    v = @index(Global)
    @inbounds begin
        c = Int32(0)
        g = gain[v]
        if g >= thr
            hv = (UInt32(v) * 0x9e3779b9) ⊻ seed
            c = Int32(1)
            for p in (xadj[v] + 1):xadj[v + 1]
                u = adjncy[p] + 1
                where_[u] == Int32(2) || continue
                gu = gain[u]
                gu < thr && continue
                hu = (UInt32(u) * 0x9e3779b9) ⊻ seed
                if gu > g || (gu == g && hu > hv)
                    c = Int32(0)
                    break
                end
            end
        end
        cand[v] = c
    end
end

# select movers among candidates under the balance budget. budget[1] starts
# at badmaxpwgt - pwgts[to]; an atomic claim keeps pwgts[to] under cap.
@kernel function _select_kernel!(mask, budget, nmov, @Const(cand), @Const(vwgt))
    v = @index(Global)
    @inbounds begin
        m = Int32(0)
        if cand[v] == Int32(1)
            left = (Atomix.@atomic budget[1] += Int32(-vwgt[v]))
            if left >= Int32(0)
                m = Int32(1)
                Atomix.@atomic nmov[1] += Int32(1)
            end
        end
        mask[v] = m
    end
end

@kernel function _apply_kernel!(where_, @Const(mask), to::Int32)
    v = @index(Global)
    @inbounds mask[v] == Int32(1) && (where_[v] = to)
end

# pull: any `other`-side vertex adjacent to a mover enters the separator
@kernel function _pull_kernel!(where_, @Const(xadj), @Const(adjncy), @Const(mask),
                               other::Int32)
    v = @index(Global)
    @inbounds if where_[v] == other
        for p in (xadj[v] + 1):xadj[v + 1]
            if mask[adjncy[p] + 1] == Int32(1)
                where_[v] = Int32(2)
                break
            end
        end
    end
end

@kernel function _pwgts_kernel!(pwgts, @Const(where_), @Const(vwgt))
    v = @index(Global)
    @inbounds Atomix.@atomic pwgts[where_[v] + 1] += Int32(vwgt[v])
end

function _pwgts!(backend, dpw, dwhere, dvw, n)
    fill!(dpw, Int32(0))
    _pwgts_kernel!(backend)(dpw, dwhere, dvw; ndrange = n)
    KA.synchronize(backend)
    return Int.(Array(dpw))
end

"""
    project_where!(backend, fwhere, cwhere, cmap, n)

`fwhere[v] = cwhere[cmap[v]+1]` on the device (METIS Project2WayNodePartition
without the host parameter rebuild).
"""
function project_where!(backend, fwhere, cwhere, cmap, n::Integer)
    n > 0 && _project_kernel!(backend)(fwhere, cwhere, cmap; ndrange = Int(n))
    KA.synchronize(backend)
    return fwhere
end

"""
    refine_rounds!(backend, dx, da, dv, dwhere, n, ubfactors; niter = 10)
        -> (pwgts, npasses)

Round-based 1-sided node refinement of `dwhere` (0/1/2 Int32, device) against
the device CSR (`dx`, `da` 0-based ids, `dv` weights). Runs up to `2niter`
alternating passes; each pass is accepted only if the separator weight drops
(ties: balance improves), reverted otherwise; stops after an even/odd pair
without improvement, like the serial FM's termination.
"""
function refine_rounds!(backend, dx, da, dv, dwhere, n::Integer, ubfactors::Real;
                        niter::Int = 10, patience::Int = 4)
    n == 0 && return (Int[0, 0, 0], 0)
    gain = KA.allocate(backend, Int32, n)
    cand = KA.allocate(backend, Int32, n)
    mask = KA.allocate(backend, Int32, n)
    best = KA.allocate(backend, Int32, n)
    budget = KA.allocate(backend, Int32, 1)
    dpw = KA.allocate(backend, Int32, 3)
    pw = _pwgts!(backend, dpw, dwhere, dv, n)
    badmaxpwgt = trunc(Int, 0.5 * ubfactors * (pw[1] + pw[2] + pw[3]))
    to = pw[1] < pw[2] ? 1 : 0
    copyto!(best, dwhere)                          # global best, restored at exit
    best_pw = pw
    since_best = 0
    npasses = 0
    nmov = KA.allocate(backend, Int32, 1)
    nmov_h = Int32[0]
    for pass in 0:(2 * niter - 1)
        other, to = to, (to + 1) % 2
        # a PASS is a wavefront sweep at fixed direction, like the serial
        # 1-sided FM where pulled vertices join the queue mid-pass: repeat
        # micro-rounds until no admissible mover remains
        for micro in 0:199
            _gains_kernel!(backend)(gain, dx, da, dv, dwhere, Int32(other); ndrange = Int(n))
            _filter_kernel!(backend)(cand, gain, dx, da, dwhere,
                                     UInt32(2654435761) + UInt32(64 * pass + micro),
                                     Int32(0); ndrange = Int(n))
            pwt = _pwgts!(backend, dpw, dwhere, dv, n)
            copyto!(budget, Int32[max(badmaxpwgt - pwt[to + 1], 0)])
            fill!(nmov, Int32(0))
            _select_kernel!(backend)(mask, budget, nmov, cand, dv; ndrange = Int(n))
            KA.synchronize(backend)
            copyto!(nmov_h, nmov)
            nmov_h[1] == Int32(0) && break
            _apply_kernel!(backend)(dwhere, mask, Int32(to); ndrange = Int(n))
            _pull_kernel!(backend)(dwhere, dx, da, mask, Int32(other); ndrange = Int(n))
            KA.synchronize(backend)
        end
        pw = _pwgts!(backend, dpw, dwhere, dv, n)
        npasses += 1
        # hill-climb with a global-best record, like the serial FM's rollback:
        # keep walking through non-improving passes (bounded by `patience` and
        # the serial code's 1.10x escape hatch), remember the best state seen
        if pw[3] < best_pw[3] ||
           (pw[3] == best_pw[3] && abs(pw[1] - pw[2]) < abs(best_pw[1] - best_pw[2]))
            copyto!(best, dwhere)
            best_pw = pw
            since_best = 0
        else
            since_best += 1
            (since_best > patience || pw[3] > 1.10 * best_pw[3]) && break
        end
    end
    copyto!(dwhere, best)
    return (best_pw, npasses)
end

"""
    balance_rounds!(backend, dx, da, dv, dwhere, n, ubfactors) -> pwgts

Parallel FM_2WayNodeBalance: rounds of moves toward the LIGHT side. Positive
gains move freely; when none remain and the heavy side still exceeds METIS's
cap (0.5·ubfactors·(pw1+pw2)), the maximum-gain class moves even at negative
gain. Stops when the sides cross, the cap is satisfied with no free moves, or
nothing is admissible.
"""
function balance_rounds!(backend, dx, da, dv, dwhere, n::Integer, ubfactors::Real;
                         maxrounds::Int = 64)
    gain = KA.allocate(backend, Int32, n)
    cand = KA.allocate(backend, Int32, n)
    mask = KA.allocate(backend, Int32, n)
    budget = KA.allocate(backend, Int32, 1)
    nmov = KA.allocate(backend, Int32, 1)
    nmov_h = Int32[0]
    dpw = KA.allocate(backend, Int32, 3)
    pw = _pwgts!(backend, dpw, dwhere, dv, n)
    badmax = trunc(Int, 0.5 * ubfactors * (pw[1] + pw[2]))
    max(pw[1], pw[2]) < badmax && return pw
    to = pw[1] < pw[2] ? 0 : 1
    other = 1 - to
    for r in 1:maxrounds
        pw[to + 1] >= pw[other + 1] && break
        _gains_kernel!(backend)(gain, dx, da, dv, dwhere, Int32(other); ndrange = Int(n))
        KA.synchronize(backend)
        gmax = maximum(gain)
        gmax == typemin(Int32) && break            # no separator vertices
        thr = Int32(0)
        if gmax < Int32(0)
            pw[other + 1] <= badmax && break       # balanced enough, only bad moves left
            thr = gmax                             # admit the best (negative) class only
        end
        _filter_kernel!(backend)(cand, gain, dx, da, dwhere,
                                 UInt32(0x85ebca6b) + UInt32(r), thr; ndrange = Int(n))
        b = max(badmax - pw[to + 1], 0)
        b == 0 && break
        copyto!(budget, Int32[b])
        fill!(nmov, Int32(0))
        _select_kernel!(backend)(mask, budget, nmov, cand, dv; ndrange = Int(n))
        KA.synchronize(backend)
        copyto!(nmov_h, nmov)
        nmov_h[1] == Int32(0) && break
        _apply_kernel!(backend)(dwhere, mask, Int32(to); ndrange = Int(n))
        _pull_kernel!(backend)(dwhere, dx, da, mask, Int32(other); ndrange = Int(n))
        pw = _pwgts!(backend, dpw, dwhere, dv, n)
    end
    return pw
end

end # module
