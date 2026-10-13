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

Contract a matching. Coarse rows come out sorted by coarse id; fully
device-resident, the representative scan and segment scans are device cumsums,
segmented reduction by Int32 atomics (deterministic: integer adds commute).

Two grouping paths, element-identical output:

* packed (the common case, `cn < 2^20` and `m < 2^24`): one radix `sort!` of
  `(crow << 44) | (ccol << 24) | edge index` — the free low bits carry the
  edge index, so no `sortperm` and no gather; the weight is read back through
  the embedded index. `sort!` of a plain `UInt64` vector is ~3x faster than
  `sortperm` on the same keys (GV100, 7.3M edges: 50ms vs 167ms), which makes
  the whole contraction ~1.8x faster.
* fallback (larger graphs): the original `sortperm` over `(crow << 32) | ccol`
  keys with a value gather.
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
    local isnew, segid, nseg, cadjncy, cadjwgt, rowcount
    if cn < (1 << 20) && m < (1 << 24)
        packed = KA.allocate(backend, UInt64, m)
        packed_kernel!(backend)(packed, xadj, adjncy, cmap; ndrange = n)
        KA.synchronize(backend)
        sort!(packed)                        # plain radix sort, no permutation
        isnew = KA.allocate(backend, Int32, m)
        boundary_packed_kernel!(backend)(isnew, packed; ndrange = m)
        KA.synchronize(backend)
        segid = cumsum(isnew)                # segment index per element (1-based)
        nseg = Int(Array(view(segid, m:m))[1])
        cadjncy = KA.allocate(backend, Int32, nseg)
        cadjwgt = KA.allocate(backend, Int32, nseg)
        fill!(cadjwgt, Int32(0))
        rowcount = KA.allocate(backend, Int32, Int(cn) + 1)
        fill!(rowcount, Int32(0))
        segfill_packed_kernel!(backend)(cadjncy, cadjwgt, rowcount, packed, adjwgt,
                                        isnew, segid; ndrange = m)
    else
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
        segid = cumsum(isnew)
        nseg = Int(Array(view(segid, m:m))[1])
        cadjncy = KA.allocate(backend, Int32, nseg)
        cadjwgt = KA.allocate(backend, Int32, nseg)
        fill!(cadjwgt, Int32(0))
        rowcount = KA.allocate(backend, Int32, Int(cn) + 1)
        fill!(rowcount, Int32(0))
        segfill_kernel!(backend)(cadjncy, cadjwgt, rowcount, keys, vals, isnew, segid; ndrange = m)
    end
    KA.synchronize(backend)
    cxadj = cumsum(rowcount)                 # rowcount[1] = 0 stays 0
    cvwgt = KA.allocate(backend, Int32, Int(cn))
    cvwgt_kernel!(backend)(cvwgt, vwgt, match, cmap; ndrange = n)
    KA.synchronize(backend)
    return Int(cn), cxadj, cadjncy, cadjwgt, cvwgt, cmap
end

# packed-path kernels: the key is (crow << 44) | (ccol << 24) | edge index,
# self edges sort to the end as typemax
@kernel function packed_kernel!(packed, @Const(xadj), @Const(adjncy), @Const(cmap))
    gi = @index(Global)
    v0 = gi - 1
    cv = cmap[v0 + 1]
    for p in (xadj[v0 + 1] + 1):xadj[v0 + 2]
        cu = cmap[adjncy[p] + 1]
        if cu == cv
            packed[p] = typemax(UInt64)
        else
            packed[p] = (UInt64(cv) << 44) | (UInt64(cu) << 24) | UInt64(p - 1)
        end
    end
end

@kernel function boundary_packed_kernel!(isnew, @Const(packed))
    i = @index(Global)
    k = packed[i]
    if k == typemax(UInt64)
        isnew[i] = Int32(0)
    elseif i == 1
        isnew[i] = Int32(1)
    else
        isnew[i] = (packed[i - 1] >> 24) == (k >> 24) ? Int32(0) : Int32(1)
    end
end

@kernel function segfill_packed_kernel!(cadjncy, cadjwgt, rowcount, @Const(packed),
                                        @Const(adjwgt), @Const(isnew), @Const(segid))
    i = @index(Global)
    k = packed[i]
    if k != typemax(UInt64)
        sid = segid[i]
        idx = Int(k & 0xffffff)              # original edge position (0-based)
        KA.@atomic cadjwgt[sid] += adjwgt[idx + 1]
        if isnew[i] == Int32(1)
            cadjncy[sid] = Int32((k >> 24) & 0xfffff)
            KA.@atomic rowcount[Int(k >> 44) + 2] += Int32(1)
        end
    end
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
