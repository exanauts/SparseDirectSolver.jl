module GpuBatch

# Level-batched sub-bisections: the ~80% of NodeND time that is thousands of
# small host bisections, re-expressed so a whole recursion level of blocks is
# ONE block-diagonal device graph, coarsened in single kernel passes. Blocks
# are disconnected components of the batch, so GpuND's matching/contraction
# apply unchanged: matching proposals follow edges and cannot cross blocks,
# and contraction preserves the block structure by construction. Per-block
# control (CoarsenTo, freezing a finished block) rides on the per-vertex
# maxvwgt vector gpu_match already takes (cap 0 = no matches = identity).
#
# The 2026-10-09 Batched.jl attempt is the documented dead-end this replaces:
# it re-sliced every block to a host GraphT at every level (12x regression).
# Here the batch stays device-resident across levels; the host sees only
# per-block counters (nb Int32s per level).

using SparseDirectSolver.KernelAbstractions
const KA = KernelAbstractions
using SparseDirectSolver.Atomix
using ..MetisExact, ..GpuND
const ME = MetisExact

export build_batch, coarsen_batch!, batch_invariants

struct Batch
    n::Int                    # total vertices
    nb::Int                   # blocks
    dx; da; dw; dv            # device CSR, GpuND conventions (0-based ids)
    bid                       # device Int32, 1-based block id per vertex
    nv_b::Vector{Int}         # host per-block vertex counts
    tvw_b::Vector{Int}        # host per-block total vertex weight
end

"""
    build_batch(backend, graphs::Vector{ME.GraphT}) -> Batch

Concatenate independent subgraphs into one block-diagonal device CSR
(vertex ids shifted; no edges are added, so blocks stay disconnected).
"""
function build_batch(backend, graphs::Vector{ME.GraphT})
    nb = length(graphs)
    n = sum(g.nvtxs for g in graphs)
    ne = sum(g.nedges for g in graphs)
    xadj = Vector{Int32}(undef, n + 1)
    adj = Vector{Int32}(undef, ne)
    adjw = Vector{Int32}(undef, ne)
    vw = Vector{Int32}(undef, n)
    bid = Vector{Int32}(undef, n)
    nv_b = Vector{Int}(undef, nb)
    tvw_b = zeros(Int, nb)
    xadj[1] = 0
    off = 0
    voff = 0
    for (b, g) in enumerate(graphs)
        nv_b[b] = g.nvtxs
        for i in 1:g.nvtxs
            vw[voff + i] = Int32(g.vwgt[i])
            tvw_b[b] += g.vwgt[i]
            bid[voff + i] = Int32(b)
            xadj[voff + i + 1] = Int32(off + g.xadj[i + 1])
        end
        for p in 1:g.nedges
            adj[off + p] = Int32(g.adjncy[p] + voff)
            adjw[off + p] = Int32(g.adjwgt[p])
        end
        off += g.nedges
        voff += g.nvtxs
    end
    dev(x) = (y = KA.allocate(backend, Int32, length(x)); copyto!(y, x); y)
    Batch(n, nb, dev(xadj), dev(adj), dev(adjw), dev(vw), dev(bid), nv_b, tvw_b)
end

@kernel function _mv_kernel!(mv, @Const(bid), @Const(caps))
    v = @index(Global)
    @inbounds mv[v] = caps[bid[v]]
end

# every fine vertex of a coarse vertex carries the same block id
@kernel function _bidmap_kernel!(bidc, @Const(bid), @Const(cmap))
    v = @index(Global)
    @inbounds bidc[cmap[v] + 1] = bid[v]
end

@kernel function _bcount_kernel!(cnt, @Const(bid))
    v = @index(Global)
    @inbounds Atomix.@atomic cnt[bid[v]] += Int32(1)
end

@kernel function _bwsum_kernel!(ws, @Const(bid), @Const(vwgt))
    v = @index(Global)
    @inbounds Atomix.@atomic ws[bid[v]] += Int32(vwgt[v])
end

@kernel function _crossedge_kernel!(bad, @Const(xadj), @Const(adjncy), @Const(bid))
    v = @index(Global)
    @inbounds for p in (xadj[v] + 1):xadj[v + 1]
        bid[adjncy[p] + 1] != bid[v] && (bad[1] = Int32(1))
    end
end

function _percount(backend, bid, nb, n)
    cnt = KA.allocate(backend, Int32, nb)
    fill!(cnt, Int32(0))
    _bcount_kernel!(backend)(cnt, bid; ndrange = n)
    KA.synchronize(backend)
    return Int.(Array(cnt))
end

"""
    batch_invariants(backend, B) -> (cross_block_edges::Bool, nv, wsum)

Structural checks: no edge crosses blocks; per-block vertex counts and weight
sums (the weight sums must be invariant under every contraction level).
"""
function batch_invariants(backend, B::Batch)
    bad = KA.allocate(backend, Int32, 1)
    fill!(bad, Int32(0))
    _crossedge_kernel!(backend)(bad, B.dx, B.da, B.bid; ndrange = B.n)
    ws = KA.allocate(backend, Int32, B.nb)
    fill!(ws, Int32(0))
    _bwsum_kernel!(backend)(ws, B.bid, B.dv; ndrange = B.n)
    KA.synchronize(backend)
    return (Array(bad)[1] != 0, _percount(backend, B.bid, B.nb, B.n), Int.(Array(ws)))
end

"""
    coarsen_batch!(backend, B; seed = 1, levels = nothing, maxlevels = 40)
        -> (B_coarse, frozen_done)

Coarsen every block of the batch simultaneously until each reaches its METIS
CoarsenTo (clamp(nvtxs/8, 40, 100)) or stalls (shrink > COARSEN_FRACTION),
freezing finished blocks via a zero matching cap. `levels`, when a vector, is
filled with (dx, da, dv, n, cmap, bid) per level, finest first, for device
uncoarsening.
"""
function coarsen_batch!(backend, B::Batch; seed::Integer = 1,
                        levels::Union{Nothing, Vector{Any}} = nothing,
                        maxlevels::Int = 40)
    nb = B.nb
    ct_b = [clamp(B.nv_b[b] ÷ 8, 40, 100) for b in 1:nb]
    frozen = [B.nv_b[b] <= ct_b[b] for b in 1:nb]
    nv = copy(B.nv_b)
    dx, da, dw, dv, bid, n = B.dx, B.da, B.dw, B.dv, B.bid, B.n
    caps = KA.allocate(backend, Int32, nb)
    caps_h = Vector{Int32}(undef, nb)
    level = 0
    while !all(frozen) && level < maxlevels
        for b in 1:nb
            caps_h[b] = frozen[b] ? Int32(0) :
                        Int32(max(trunc(Int, 1.5 * B.tvw_b[b] / ct_b[b]), 1))
        end
        copyto!(caps, caps_h)
        mv = KA.allocate(backend, Int32, n)
        _mv_kernel!(backend)(mv, bid, caps; ndrange = n)
        KA.synchronize(backend)
        match = GpuND.gpu_match(backend, dx, da, dw, dv, mv; seed = seed + level)
        cn, cx, ca, cw, cv, cmap = GpuND.gpu_contract(backend, dx, da, dw, dv, match)
        levels === nothing ||
            push!(levels, (dx = dx, da = da, dv = dv, n = n, cmap = cmap, bid = bid))
        bidc = KA.allocate(backend, Int32, cn)
        _bidmap_kernel!(backend)(bidc, bid, cmap; ndrange = n)
        KA.synchronize(backend)
        nvc = _percount(backend, bidc, nb, cn)
        for b in 1:nb
            frozen[b] && continue
            if nvc[b] <= ct_b[b] || nvc[b] > ME.COARSEN_FRACTION * nv[b]
                frozen[b] = true              # done, or stalled at this size
            end
        end
        nv = nvc
        dx, da, dw, dv, bid, n = cx, ca, cw, cv, bidc, cn
        level += 1
    end
    return (Batch(n, nb, dx, da, dw, dv, bid, nv, copy(B.tvw_b)), frozen)
end

end # module
