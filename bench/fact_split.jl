# Prototype: split regime-B factorization — the stock fused kernel keeps a
# front's contribution-block update (F22 -= F21 F21') inside the front's single
# workgroup, which serializes ~5 MFLOP on one block for the fat fronts near the
# root (measured 1-2 ms per front). Here the panel part (assembly, F11
# Cholesky, trsm) stays one-block, and contribution blocks with m > SYRK_IN
# rows are updated by a separate tiled kernel, one 32x32 tile per workgroup,
# embarrassingly parallel, deterministic (disjoint tile writes).
#
#   CUDA_VISIBLE_DEVICES=1 julia +1.13 --project=bench bench/fact_split.jl

using SparseDirectSolver, SparseArrays, LinearAlgebra, Random, Statistics, Printf
using CUDA, CUDA.CUSPARSE
using Metis
const SDS = SparseDirectSolver
const KA = SDS.KernelAbstractions
using .KA: @kernel, @index, @localmem, @synchronize, @Const, @uniform

include(joinpath(@__DIR__, "matrices.jl"))
using .BenchMatrices: read_mtx

const SYRK_IN = 64
const TILE = 32
const WGF = 128

# panel part of the stock fused regime-B kernel: assembly + F11 Cholesky +
# trsm; the contribution block is updated here only when m <= SYRK_IN
@kernel function panel_cholesky_kernel!(factor, stack, info, @Const(nzval), @Const(amap), @Const(amap_ptr),
                                        @Const(amap_src), @Const(nodes), bm, @Const(front_ptr),
                                        @Const(front_nrows), @Const(front_ncols), @Const(cb_ptr),
                                        @Const(child_ptr), @Const(child_list), @Const(relind_ptr),
                                        @Const(relind), maxchild, ::Val{W}, ::Val{NL},
                                        ::Val{WG}) where {W, NL, WG}
    @uniform TT = eltype(factor)
    @uniform RT = real(eltype(factor))
    li = @index(Local, Linear)
    G = @index(Group, Linear)
    L11 = @localmem TT (NL,)
    st = @localmem Int32 (1,)
    piv = @localmem RT (1,)
    @inbounds s = nodes[bm.first + SDS._bm_node(bm, G) - 1]
    k = SDS._bm_gmember(bm, G)
    SDS._zero_front!(factor, SDS._mview(stack, k, bm.nbatch), s, li,
                     SDS.member_panels(front_ptr, k, bm.nbatch), front_nrows, front_ncols, cb_ptr, Val(WG))
    @synchronize
    @inbounds s = nodes[bm.first + SDS._bm_node(bm, G) - 1]
    k = SDS._bm_gmember(bm, G)
    SDS._scatter_front!(factor, SDS._mview(nzval, k, bm.nbatch), amap, amap_ptr, amap_src, s,
                        SDS._member_shift(front_ptr, s, k, bm.nbatch), li, Val(WG))
    @synchronize
    for kc in 1:maxchild
        @inbounds s = nodes[bm.first + SDS._bm_node(bm, G) - 1]
        k = SDS._bm_gmember(bm, G)
        SDS._extend_add_child!(factor, SDS._mview(stack, k, bm.nbatch), s, kc, li,
                               SDS.member_panels(front_ptr, k, bm.nbatch), front_nrows, front_ncols, cb_ptr,
                               child_ptr, child_list, relind_ptr, relind, Val(WG))
        @synchronize
    end
    @inbounds s = nodes[bm.first + SDS._bm_node(bm, G) - 1]
    k = SDS._bm_gmember(bm, G)
    SDS._front_load!(L11, st, factor, s, li, SDS.member_panels(front_ptr, k, bm.nbatch), front_nrows,
                     front_ncols, Val(W), Val(WG))
    @synchronize
    for j in 1:W
        @inbounds s = nodes[bm.first + SDS._bm_node(bm, G) - 1]
        SDS._front_chol_update!(L11, st, piv, j, s, li, front_ncols, Val(W), Val(WG))
        @synchronize
        @inbounds s = nodes[bm.first + SDS._bm_node(bm, G) - 1]
        SDS._front_chol_scale!(L11, st, piv, j, s, li, front_ncols, Val(W), Val(WG))
        @synchronize
    end
    @inbounds s = nodes[bm.first + SDS._bm_node(bm, G) - 1]
    k = SDS._bm_gmember(bm, G)
    @inbounds begin
        P = SDS.member_panels(front_ptr, k, bm.nbatch)
        f = Int(front_nrows[s])
        w = Int(front_ncols[s])
        p0 = Int(P[s])
        for q in (li - 1):WG:(w * w - 1)
            j = q ÷ w + 1
            i = q - (j - 1) * w + 1
            i >= j && (factor[p0 + (j - 1) * f + i - 1] = L11[SDS._packed(i, j, W)])
        end
        if st[1] == Int32(0) && f - w <= SYRK_IN
            for i in (w + li):WG:f
                for kk in 1:w
                    x = factor[p0 + (kk - 1) * f + i - 1]
                    for j in 1:(kk - 1)
                        x -= factor[p0 + (j - 1) * f + i - 1] * L11[SDS._packed(kk, j, W)]
                    end
                    factor[p0 + (kk - 1) * f + i - 1] = x / L11[SDS._packed(kk, kk, W)]
                end
            end
        end
        li == 1 && (SDS._iview(info, k, bm.nbatch)[s] = st[1])
    end
    @synchronize
    @inbounds s = nodes[bm.first + SDS._bm_node(bm, G) - 1]
    k = SDS._bm_gmember(bm, G)
    @inbounds if Int(front_nrows[s]) - Int(front_ncols[s]) <= SYRK_IN
        SDS._front_syrk!(factor, SDS._mview(stack, k, bm.nbatch), st, s, li,
                         SDS.member_panels(front_ptr, k, bm.nbatch), front_nrows, front_ncols, cb_ptr, Val(WG))
    end
end

# one 128-row chunk of the trsm F21 <- F21 L11^-T per workgroup (m > SYRK_IN fronts)
@kernel function trsm_chunk_kernel!(factor, @Const(info), @Const(cfront), @Const(cchunk), @Const(front_ptr),
                                    @Const(front_nrows), @Const(front_ncols), ::Val{WG}) where {WG}
    li = @index(Local, Linear)
    G = @index(Group, Linear)
    L11 = @localmem Float64 (2080,)
    @inbounds begin
        s = Int(cfront[G])
        f = Int(front_nrows[s])
        w = Int(front_ncols[s])
        p0 = Int(front_ptr[s])
        for q in li:WG:(w * (w + 1) ÷ 2)
            # unpack packed index q -> (i, j), i >= j: walk columns
            j = 1
            acc = w
            qq = q
            while qq > acc
                qq -= acc
                acc -= 1
                j += 1
            end
            i = j + qq - 1
            L11[SDS._packed(i, j, 64)] = factor[p0 + (j - 1) * f + i - 1]
        end
        @synchronize
        if info[s] == Int32(0)
            i = w + (Int(cchunk[G]) - 1) * WG + li
            if i <= f
                for kk in 1:w
                    x = factor[p0 + (kk - 1) * f + i - 1]
                    for j in 1:(kk - 1)
                        x -= factor[p0 + (j - 1) * f + i - 1] * L11[SDS._packed(kk, j, 64)]
                    end
                    factor[p0 + (kk - 1) * f + i - 1] = x / L11[SDS._packed(kk, kk, 64)]
                end
            end
        end
    end
end

# one 32x32 lower tile of F22 -= F21 F21' per workgroup (Float64, nbatch = 1)
@kernel function syrk_tile_kernel!(stack, @Const(factor), @Const(info), @Const(tfront), @Const(tti),
                                   @Const(ttj), @Const(front_ptr), @Const(front_nrows), @Const(front_ncols),
                                   @Const(cb_ptr), ::Val{WG}) where {WG}
    li = @index(Local, Linear)
    G = @index(Group, Linear)
    Ai = @localmem Float64 (TILE * SYRK_IN,)
    Aj = @localmem Float64 (TILE * SYRK_IN,)
    @inbounds begin
        s = Int(tfront[G])
        if info[s] == Int32(0) && cb_ptr[s] > 0
            f = Int(front_nrows[s])
            w = Int(front_ncols[s])
            m = f - w
            p0 = Int(front_ptr[s]) + w - 1
            c0 = Int(cb_ptr[s])
            i0 = (Int(tti[G]) - 1) * TILE
            j0 = (Int(ttj[G]) - 1) * TILE
            for q in (li - 1):WG:(TILE * w - 1)
                r = q % TILE + 1
                c = q ÷ TILE + 1
                row = i0 + r
                Ai[r + (c - 1) * TILE] = row <= m ? factor[p0 + (c - 1) * f + row] : 0.0
                row2 = j0 + r
                Aj[r + (c - 1) * TILE] = row2 <= m ? factor[p0 + (c - 1) * f + row2] : 0.0
            end
            @synchronize
            for q in (li - 1):WG:(TILE * TILE - 1)
                r = q % TILE + 1
                cc = q ÷ TILE + 1
                ii = i0 + r
                jj = j0 + cc
                if ii <= m && jj <= m && ii >= jj
                    acc = 0.0
                    for kk in 1:w
                        acc += Ai[r + (kk - 1) * TILE] * Aj[cc + (kk - 1) * TILE]
                    end
                    d = c0 + SDS._packed(ii, jj, m) - 1
                    stack[d] -= acc
                end
            end
        end
    end
end

# host: per B-group tile lists for fronts with m > SYRK_IN (built once per analysis)
function build_tiles(S, plan)
    nodes = S.schedule.group_nodes
    rows_h = S.schedule.rows
    width_h = S.schedule.width
    cbp = Array(S.cb_ptr)
    tiles = Dict{Int, Union{Nothing, NTuple{3, CuVector{Int32}}}}()
    for k in eachindex(plan.group_first)
        plan.group_width[k] > 0 || (tiles[k] = nothing; continue)
        tf = Int32[]; ti = Int32[]; tj = Int32[]
        for q in plan.group_first[k]:plan.group_last[k]
            v = nodes[q]
            m = rows_h[v] - width_h[v]
            (m > SYRK_IN && cbp[v] > 0) || continue
            T = cld(m, TILE)
            for tjj in 1:T, tii in tjj:T
                push!(tf, Int32(v)); push!(ti, Int32(tii)); push!(tj, Int32(tjj))
            end
        end
        tiles[k] = isempty(tf) ? nothing : (CuArray(tf), CuArray(ti), CuArray(tj))
    end
    chunks = Dict{Int, Union{Nothing, NTuple{2, CuVector{Int32}}}}()
    for k in eachindex(plan.group_first)
        plan.group_width[k] > 0 || (chunks[k] = nothing; continue)
        cf = Int32[]; cc = Int32[]
        for q in plan.group_first[k]:plan.group_last[k]
            v = nodes[q]
            m = rows_h[v] - width_h[v]
            m > SYRK_IN || continue
            for c in 1:cld(m, 128)
                push!(cf, Int32(v)); push!(cc, Int32(c))
            end
        end
        chunks[k] = isempty(cf) ? nothing : (CuArray(cf), CuArray(cc))
    end
    return tiles, chunks
end

function refact_split!(s, nzval, tiles, chunks)
    N, S = s.numeric, s.symbolic
    p = SDS._front_impls(N, S, :auto)
    plan = N.plan
    nodes = S.schedule.group_nodes
    backend = KA.get_backend(N.factor)
    for k in eachindex(plan.sub_first)
        a, b = plan.sub_first[k], plan.sub_last[k]
        SDS.factorize_subtrees!(N, S, nzval, a, b - a + 1, plan.sub_local[k])
    end
    for k in eachindex(plan.group_first)
        a, b = plan.group_first[k], plan.group_last[k]
        W = plan.group_width[k]
        if W > 0
            bm = SDS.batch_map(N; first = a)
            args = (N.factor, N.stack, N.info, nzval, S.amap, S.amap_ptr, S.amap_src, S.group_nodes, bm,
                    S.front_ptr, S.front_nrows, S.front_ncols, S.cb_ptr, S.child_ptr, S.child_list,
                    S.relind_ptr, S.relind, Int(plan.group_maxchild[k]))
            nd = WGF * (b - a + 1)
            if W == 8
                panel_cholesky_kernel!(backend, WGF)(args..., Val(8), Val(36), Val(WGF); ndrange = nd)
            elseif W == 16
                panel_cholesky_kernel!(backend, WGF)(args..., Val(16), Val(136), Val(WGF); ndrange = nd)
            elseif W == 32
                panel_cholesky_kernel!(backend, WGF)(args..., Val(32), Val(528), Val(WGF); ndrange = nd)
            else
                panel_cholesky_kernel!(backend, WGF)(args..., Val(64), Val(2080), Val(WGF); ndrange = nd)
            end
            c = chunks[k]
            c === nothing ||
                trsm_chunk_kernel!(backend, WGF)(N.factor, N.info, c[1], c[2], S.front_ptr, S.front_nrows,
                                                 S.front_ncols, Val(WGF); ndrange = WGF * length(c[1]))
            t = tiles[k]
            t === nothing ||
                syrk_tile_kernel!(backend, WGF)(N.stack, N.factor, N.info, t[1], t[2], t[3], S.front_ptr,
                                                S.front_nrows, S.front_ncols, S.cb_ptr, Val(WGF);
                                                ndrange = WGF * length(t[1]))
        else
            SDS.zero_fronts!(N, S, a, b - a + 1)
            SDS.scatter_A!(N, S, nzval, a, b - a + 1)
            SDS.extend_add!(N, S, a, b - a + 1, Int(plan.group_maxchild[k]))
            for q in a:b
                SDS._factor_front_c!(N, S, nodes[q], p)
            end
        end
    end
    SDS.assemble_schur!(N, S, nzval)
    SDS.cholesky_stats!(N, S)
    return nothing
end
refact_info!(s) = SDS._numeric_info!(s.numeric, s.symbolic, true)

# ---------------------------------------------------------------------------
const PATH = joinpath(@__DIR__, "data", "kkt_pglib_opf_case78484_epigrids_condensed_10.mtx")
A = SparseMatrixCSC{Float64, Int}(sparse(read_mtx(PATH)))
n = size(A, 1)
Random.seed!(666)
bh = rand(n)
Lh = tril(A)
const UP = (p = Vector{Int32}(undef, n); read!(joinpath(@__DIR__, "cudss_perm.bin"), p); Int.(p))

function run_config(label; cw = 32, rows = 512, sp = 16384)
    bd = CuArray(bh); xd = similar(bd)
    s = DirectSolver(CuSparseMatrixCSR(Lh), "SPD", 'L')
    s.options.regime_c_width = cw
    s.options.regime_c_rows = rows
    s.options.subtree_parallelism = sp
    SDS.setparam!(s, "user_perm", UP)
    ex(p) = SDS.execute!(p, s, xd, bd; asynchronous = false)
    ex("analysis"); ex("factorization"); ex("refactorization")
    CUDA.synchronize()
    t0 = median([(@elapsed ex("refactorization")) for _ in 1:10])
    ex("solve")
    rel0 = norm(bh - A * Array(xd)) / norm(bh)
    fac0 = Array(s.numeric.factor)
    nzval = SDS._factor_values(s)
    tiles, chunks = build_tiles(s.symbolic, s.numeric.plan)
    refact_split!(s, nzval, tiles, chunks); CUDA.synchronize()
    info = refact_info!(s)
    fac1 = Array(s.numeric.factor)
    ex("solve")
    rel1 = norm(bh - A * Array(xd)) / norm(bh)
    for _ in 1:2; refact_split!(s, nzval, tiles, chunks); end
    CUDA.synchronize()
    t1 = median([(@elapsed (refact_split!(s, nzval, tiles, chunks); CUDA.synchronize())) for _ in 1:10])
    nt = sum(t === nothing ? 0 : length(t[1]) for t in values(tiles))
    tg = NaN
    try
        g = CUDA.capture() do
            refact_split!(s, nzval, tiles, chunks)
        end
        exec = CUDA.instantiate(g)
        replay() = (CUDA.launch(exec); CUDA.synchronize())
        replay()
        fac2 = Array(s.numeric.factor)
        @assert fac2 == fac0 "graph replay factor differs"
        tg = median([(@elapsed replay()) for _ in 1:10])
    catch err
        println("    graph capture failed: ", sprint(showerror, err)[1:min(end, 200)])
    end
    @printf "%-26s stock %7.2f ms | split %7.2f ms | graph %7.2f ms  info %d  tiles %d  Δfactor %.2e  relres %.2e (stock %.2e)\n" label t0*1e3 t1*1e3 tg*1e3 info nt norm(fac1 - fac0) / norm(fac0) rel1 rel0
    flush(stdout)
    GC.gc(); CUDA.reclaim()
end

run_config("cw32 rows256 sp16384"; rows = 256)
run_config("cw32 rows256 sp65536"; rows = 256, sp = 65536)
println("done")
