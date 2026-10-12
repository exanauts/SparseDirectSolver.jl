module GpuND

# M2: device kernels for the NodeND pipeline, written on KernelAbstractions
# against the exact host port (MetisExact.jl) as the stage oracle.
#
# Parallelization contract, stage by stage:
#   matching     — hashed-handshake HEM: heaviest-edge preference, maxvwgt
#                  respected, per-round hash tie-breaking. Deterministic given
#                  (graph, seed, rounds) but NOT the sequential SHEM order, so
#                  it is validated by invariants + end quality, not bitwise.
#   contraction  — given a matching it is order-free: validated ROW-SET
#                  identical (sorted (id, wgt) rows) against the host
#                  CreateCoarseGraph fed the same match array.
#   edge order   — GPU coarse rows are sorted by coarse id (canonical),
#                  where METIS's are hash-insertion-ordered; downstream GPU
#                  stages define their own deterministic order.
#
# All indices 0-based in device arrays, matching the host port.

using SparseDirectSolver.KernelAbstractions
const KA = KernelAbstractions

export gpu_match, gpu_contract

# ---------------------------------------------------------------------------
# matching: propose / resolve rounds

@kernel function propose_kernel!(prop, @Const(xadj), @Const(adjncy), @Const(adjwgt),
                                 @Const(vwgt), @Const(match), @Const(maxvwgt), salt::UInt64)
    gi = @index(Global)
    v0 = gi - 1                          # 0-based vertex
    v = Int32(v0)
    if match[v + 1] == Int32(-2)            # -2 = unmatched (device sentinel)
        best = Int32(-1)
        bw = Int32(-1)
        bh = UInt64(0)
        mv = maxvwgt[v + 1]                  # per-vertex cap (batched: per subgraph)
        if mv > Int32(0) && vwgt[v + 1] < mv
            for p in (xadj[v + 1] + 1):xadj[v + 2]
                u = adjncy[p]
                w = adjwgt[p]
                if u != v && match[u + 1] == Int32(-2) && vwgt[v + 1] + vwgt[u + 1] <= mv
                    hu = (UInt64(u) * 0x9e3779b97f4a7c15 + salt) >> 1
                    if w > bw || (w == bw && (best == Int32(-1) || hu > bh))
                        best = u
                        bw = w
                        bh = hu
                    end
                end
            end
        end
        prop[v + 1] = best
    else
        prop[v + 1] = Int32(-1)
    end
end

@kernel function resolve_kernel!(match, @Const(prop))
    gi = @index(Global)
    v0 = gi - 1
    v = Int32(v0)
    u = prop[v + 1]
    if u != Int32(-1) && prop[u + 1] == v && match[v + 1] == Int32(-2)
        # mutual proposal; both sides see it symmetrically, no atomics needed
        match[v + 1] = u
    end
end

"""
    gpu_match(backend, xadj, adjncy, adjwgt, vwgt, maxvwgt; rounds = 8, seed = 1) -> match

Heavy-edge matching by hashed mutual proposals. `match[v+1]` is the partner
(0-based) or `v` for a singleton. Input arrays are device vectors (Int32,
0-based ids); the result is a device vector.
"""
function gpu_match(backend, xadj, adjncy, adjwgt, vwgt, maxvwgt;
                   rounds::Int = 8, seed::Integer = 1)
    n = length(xadj) - 1
    if maxvwgt isa Integer                   # scalar cap -> per-vertex vector
        mv = KA.allocate(backend, Int32, n)
        fill!(mv, Int32(maxvwgt))
        maxvwgt = mv
    end
    match = KA.allocate(backend, Int32, n)
    fill!(match, Int32(-2))
    prop = KA.allocate(backend, Int32, n)
    pk = propose_kernel!(backend)
    rk = resolve_kernel!(backend)
    for r in 1:rounds
        salt = UInt64(seed) * 0xbf58476d1ce4e5b9 + UInt64(r) * 0x94d049bb133111eb
        pk(prop, xadj, adjncy, adjwgt, vwgt, match, maxvwgt, salt; ndrange = n)
        rk(match, prop; ndrange = n)
    end
    # 2-hop matching for hub-dense graphs (METIS Match_2HopAny semantics):
    # unmatched low-degree vertices pair through a common neighbour. Parallel
    # form: per-neighbour atomic-min "two smallest unmatched" registers, then
    # the same mutual-proposal resolve as the handshake. Gated like METIS at
    # >10% unmatched.
    nun = count_unmatched(backend, match)
    if nun > n ÷ 10
        minu = KA.allocate(backend, Int32, n)
        min2u = KA.allocate(backend, Int32, n)
        for (rnds, maxdeg) in ((2, 2), (2, 3), (3, typemax(Int32)))
            for _ in 1:rnds
                fill!(minu, typemax(Int32))
                fill!(min2u, typemax(Int32))
                minu_kernel!(backend)(minu, xadj, adjncy, match, maxvwgt, Int32(maxdeg); ndrange = n)
                min2_kernel!(backend)(min2u, minu, xadj, adjncy, match, maxvwgt, Int32(maxdeg); ndrange = n)
                twohop_propose_kernel!(backend)(prop, minu, min2u, xadj, adjncy, match, maxvwgt,
                                                Int32(maxdeg); ndrange = n)
                resolve_kernel!(backend)(match, prop; ndrange = n)
            end
            nun = count_unmatched(backend, match)
            nun > n ÷ 10 || break
        end
    end
    # leftovers become singletons [KERNEL]
    singles_kernel!(backend)(match; ndrange = n)
    KA.synchronize(backend)
    return match
end

function count_unmatched(backend, match)
    f = KA.allocate(backend, Int32, length(match))
    unmatched_flag_kernel!(backend)(f, match; ndrange = length(match))
    return Int(sum(f))                       # the reduction synchronizes
end

@kernel function unmatched_flag_kernel!(f, @Const(match))
    gi = @index(Global)
    f[gi] = match[gi] == Int32(-2) ? Int32(1) : Int32(0)
end

@kernel function minu_kernel!(minu, @Const(xadj), @Const(adjncy), @Const(match), @Const(maxvwgt), maxdeg::Int32)
    gi = @index(Global)
    v0 = gi - 1
    v = Int32(v0)
    if match[v + 1] == Int32(-2) && maxvwgt[v + 1] > Int32(0) && xadj[v + 2] - xadj[v + 1] < maxdeg
        for p in (xadj[v + 1] + 1):xadj[v + 2]
            c = adjncy[p]
            KA.@atomic minu[c + 1] min v
        end
    end
end

@kernel function min2_kernel!(min2u, @Const(minu), @Const(xadj), @Const(adjncy),
                              @Const(match), @Const(maxvwgt), maxdeg::Int32)
    gi = @index(Global)
    v0 = gi - 1
    v = Int32(v0)
    if match[v + 1] == Int32(-2) && maxvwgt[v + 1] > Int32(0) && xadj[v + 2] - xadj[v + 1] < maxdeg
        for p in (xadj[v + 1] + 1):xadj[v + 2]
            c = adjncy[p]
            if minu[c + 1] != v
                KA.@atomic min2u[c + 1] min v
            end
        end
    end
end

@kernel function twohop_propose_kernel!(prop, @Const(minu), @Const(min2u), @Const(xadj),
                                        @Const(adjncy), @Const(match), @Const(maxvwgt), maxdeg::Int32)
    gi = @index(Global)
    v0 = gi - 1
    v = Int32(v0)
    best = Int32(-1)
    if match[v + 1] == Int32(-2) && maxvwgt[v + 1] > Int32(0) && xadj[v + 2] - xadj[v + 1] < maxdeg
        for p in (xadj[v + 1] + 1):xadj[v + 2]
            c = adjncy[p]
            u = minu[c + 1] == v ? min2u[c + 1] : minu[c + 1]
            if u != typemax(Int32) && u != v && match[u + 1] == Int32(-2)
                (best == Int32(-1) || u < best) && (best = u)
            end
        end
    end
    prop[v + 1] = best
end

@kernel function singles_kernel!(match)
    gi = @index(Global)
    v0 = gi - 1
    if match[v0 + 1] == Int32(-2)
        match[v0 + 1] = Int32(v0)
    end
end

# ---------------------------------------------------------------------------
# contraction: coarse ids by scan over pair representatives, then
# pack (coarse row, coarse col) into UInt64 keys, sort, reduce duplicates.

@kernel function repflag_kernel!(flag, @Const(match))
    gi = @index(Global)
    v0 = gi - 1
    flag[v0 + 1] = Int32(v0) <= match[v0 + 1] ? Int32(1) : Int32(0)
end

@kernel function cmap_kernel!(cmap, @Const(match), @Const(repid))
    gi = @index(Global)
    v0 = gi - 1
    u = match[v0 + 1]
    if Int32(v0) <= u
        cmap[v0 + 1] = repid[v0 + 1]
    else
        cmap[v0 + 1] = repid[u + 1]
    end
end

@kernel function packkeys_kernel!(keys, vals, @Const(xadj), @Const(adjncy),
                                  @Const(adjwgt), @Const(cmap))
    gi = @index(Global)
    v0 = gi - 1
    cv = cmap[v0 + 1]
    for p in (xadj[v0 + 1] + 1):xadj[v0 + 2]
        cu = cmap[adjncy[p] + 1]
        if cu == cv
            keys[p] = typemax(UInt64)            # self edge: sorts to the end
            vals[p] = Int32(0)
        else
            keys[p] = (UInt64(cv) << 32) | UInt64(cu)
            vals[p] = adjwgt[p]
        end
    end
end

@kernel function boundary_kernel!(isnew, @Const(keys))
    i = @index(Global)
    k = keys[i]
    if k == typemax(UInt64)
        isnew[i] = Int32(0)
    elseif i == 1
        isnew[i] = Int32(1)
    else
        isnew[i] = keys[i - 1] == k ? Int32(0) : Int32(1)
    end
end

@kernel function cvwgt_kernel!(cvwgt, @Const(vwgt), @Const(match), @Const(cmap))
    gi = @index(Global)
    v0 = gi - 1
    u = match[v0 + 1]
    if Int32(v0) <= u
        w = vwgt[v0 + 1]
        u != Int32(v0) && (w += vwgt[u + 1])
        cvwgt[cmap[v0 + 1] + 1] = w
    end
end

@kernel function scatter_edges_kernel!(cadjncy, cadjwgt, rowcount, @Const(keys),
                                       @Const(segsum), @Const(isnew), @Const(pos))
    i = @index(Global)
    if isnew[i] == Int32(1)
        j = pos[i]                               # output slot (1-based)
        k = keys[i]
        cadjncy[j] = Int32(k & 0xffffffff)
        cadjwgt[j] = segsum[i]
        KA.@atomic rowcount[Int(k >> 32) + 2] += Int32(1)
    end
end

"""
    gpu_contract(backend, xadj, adjncy, adjwgt, vwgt, match)
        -> (cn, cxadj, cadjncy, cadjwgt, cvwgt, cmap)

Contract a matching. Coarse rows come out sorted by coarse id. v2: fully
device-resident — sortperm + cumsum through GPUArrays, segmented reduction by
Int32 atomics (deterministic: integer adds commute), flags/scatters as
kernels. The representative scan is a cumsum on device too.
"""
function gpu_contract(backend, xadj, adjncy, adjwgt, vwgt, match)
    n = length(xadj) - 1
    m = length(adjncy)
    flag = KA.allocate(backend, Int32, n)
    repflag_kernel!(backend)(flag, match; ndrange = n)
    KA.synchronize(backend)
    incl = cumsum(flag)                      # inclusive scan (GPUArrays / Base)
    cn = Int(Array(view(incl, n:n))[1])
    repid = KA.allocate(backend, Int32, n)
    exclusive_kernel!(backend)(repid, incl, flag; ndrange = n)
    cmap = KA.allocate(backend, Int32, n)
    cmap_kernel!(backend)(cmap, match, repid; ndrange = n)
    keys = KA.allocate(backend, UInt64, m)
    vals = KA.allocate(backend, Int32, m)
    packkeys_kernel!(backend)(keys, vals, xadj, adjncy, adjwgt, cmap; ndrange = n)
    KA.synchronize(backend)
    p = sortperm(keys)
    keys = keys[p]
    vals = vals[p]
    isnew = KA.allocate(backend, Int32, m)
    boundary_kernel!(backend)(isnew, keys; ndrange = m)
    KA.synchronize(backend)
    segid = cumsum(isnew)                    # segment index per element (1-based)
    nseg = Int(Array(view(segid, m:m))[1])
    cadjncy = KA.allocate(backend, Int32, nseg)
    cadjwgt = KA.allocate(backend, Int32, nseg)
    fill!(cadjwgt, Int32(0))
    rowcount = KA.allocate(backend, Int32, Int(cn) + 1)
    fill!(rowcount, Int32(0))
    segfill_kernel!(backend)(cadjncy, cadjwgt, rowcount, keys, vals, isnew, segid; ndrange = m)
    KA.synchronize(backend)
    cxadj = cumsum(rowcount)                 # rowcount[1] = 0 stays 0
    cvwgt = KA.allocate(backend, Int32, Int(cn))
    cvwgt_kernel!(backend)(cvwgt, vwgt, match, cmap; ndrange = n)
    KA.synchronize(backend)
    return Int(cn), cxadj, cadjncy, cadjwgt, cvwgt, cmap
end

@kernel function exclusive_kernel!(excl, @Const(incl), @Const(flag))
    gi = @index(Global)
    excl[gi] = incl[gi] - flag[gi]
end

@kernel function segfill_kernel!(cadjncy, cadjwgt, rowcount, @Const(keys), @Const(vals),
                                 @Const(isnew), @Const(segid))
    gi = @index(Global)
    k = keys[gi]
    if k != typemax(UInt64)
        sid = segid[gi]
        KA.@atomic cadjwgt[sid] += vals[gi]
        if isnew[gi] == Int32(1)
            cadjncy[sid] = Int32(k & 0xffffffff)
            KA.@atomic rowcount[Int(k >> 32) + 2] += Int32(1)
        end
    end
end

end # module
