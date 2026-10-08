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
                                        @Const(relind), maxchild, ::Val{ASM}, ::Val{W}, ::Val{NL},
                                        ::Val{WG}) where {ASM, W, NL, WG}
    @uniform TT = eltype(factor)
    @uniform RT = real(eltype(factor))
    li = @index(Local, Linear)
    G = @index(Group, Linear)
    L11 = @localmem TT (NL,)
    st = @localmem Int32 (1,)
    piv = @localmem RT (1,)
    if ASM
        @inbounds s = nodes[bm.first + SDS._bm_node(bm, G) - 1]
        k = SDS._bm_gmember(bm, G)
        SDS._zero_front!(factor, SDS._mview(stack, k, bm.nbatch), s, li,
                         SDS.member_panels(front_ptr, k, bm.nbatch), front_nrows, front_ncols, cb_ptr, Val(WG))
    end
    @synchronize
    if ASM
        @inbounds s = nodes[bm.first + SDS._bm_node(bm, G) - 1]
        k = SDS._bm_gmember(bm, G)
        SDS._scatter_front!(factor, SDS._mview(nzval, k, bm.nbatch), amap, amap_ptr, amap_src, s,
                            SDS._member_shift(front_ptr, s, k, bm.nbatch), li, Val(WG))
    end
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
    @uniform TT = eltype(factor)
    li = @index(Local, Linear)
    G = @index(Group, Linear)
    L11 = @localmem TT (2080,)
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

# one chunk of one child's contribution block per workgroup (extend-add; the
# driver launches child position 1 of every front, then position 2, ..., so the
# per-destination addition order matches the stock serial-children kernel)
const EA_CHUNK = 16384
@kernel function extend_add_chunk_kernel!(factor, stack, @Const(ef), @Const(ec), @Const(eq),
                                          @Const(front_ptr), @Const(front_nrows), @Const(front_ncols),
                                          @Const(cb_ptr), @Const(relind_ptr), @Const(relind),
                                          ::Val{WG}) where {WG}
    li = @index(Local, Linear)
    G = @index(Group, Linear)
    @inbounds begin
        s = Int(ef[G])
        c = Int(ec[G])
        cb = Int(cb_ptr[c])
        mc = Int(front_nrows[c]) - Int(front_ncols[c])
        fp = Int(front_nrows[s])
        wp = Int(front_ncols[s])
        mp = fp - wp
        pp = Int(front_ptr[s])
        cp = Int(cb_ptr[s])
        r0 = Int(relind_ptr[c]) - 1
        q0 = (Int(eq[G]) - 1) * EA_CHUNK
        qe = min(q0 + EA_CHUNK, mc * mc) - 1
        for q in (q0 + li - 1):WG:qe
            jj = q ÷ mc + 1
            ii = q - (jj - 1) * mc + 1
            if ii >= jj
                ri = Int(relind[r0 + ii])
                rj = Int(relind[r0 + jj])
                v = stack[cb + SDS._packed(ii, jj, mc) - 1]
                if rj <= wp
                    factor[pp + (rj - 1) * fp + ri - 1] += v
                else
                    stack[cp + SDS._packed(ri - wp, rj - wp, mp) - 1] += v
                end
            end
        end
    end
end

# one 32x32 lower tile of F22 -= F21 F21' per workgroup (Float64, nbatch = 1)
@kernel function syrk_tile_kernel!(stack, @Const(factor), @Const(info), @Const(tfront), @Const(tti),
                                   @Const(ttj), @Const(front_ptr), @Const(front_nrows), @Const(front_ncols),
                                   @Const(cb_ptr), ::Val{WG}) where {WG}
    @uniform TT = eltype(factor)
    li = @index(Local, Linear)
    G = @index(Group, Linear)
    Ai = @localmem TT (TILE * SYRK_IN,)
    Aj = @localmem TT (TILE * SYRK_IN,)
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
                Ai[r + (c - 1) * TILE] = row <= m ? factor[p0 + (c - 1) * f + row] : zero(TT)
                row2 = j0 + r
                Aj[r + (c - 1) * TILE] = row2 <= m ? factor[p0 + (c - 1) * f + row2] : zero(TT)
            end
            @synchronize
            for q in (li - 1):WG:(TILE * TILE - 1)
                r = q % TILE + 1
                cc = q ÷ TILE + 1
                ii = i0 + r
                jj = j0 + cc
                if ii <= m && jj <= m && ii >= jj
                    acc = zero(TT)
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
function launch_subtrees_sorted!(N, S, nzval, list, lb, backend)
    SDS._with_local_bytes(Int(lb)) do LBv
        _sorted_subtree_launch(N, S, nzval, list, LBv, Val(SDS.SUBTREE_WORKGROUP), backend)
    end
end
function _sorted_subtree_launch(N, S, nzval, list, ::Val{LB}, ::Val{WG}, backend) where {LB, WG}
    bm = SDS.batch_map(N; first = 1)
    k! = SDS.subtree_cholesky_kernel!(backend, WG)
    _sorted_subtree_launch2(N, S, nzval, list, bm, k!, Val(LB), Val(WG))
end
function _sorted_subtree_launch2(N::SDS.Numeric{T}, S, nzval, list, bm, k!, ::Val{LB}, ::Val{WG}) where {T, LB, WG}
    k!(N.factor, N.stack, N.info, nzval, S.amap, S.amap_ptr, S.amap_src, list, bm, S.subtree_ptr,
       S.subtree_nodes, S.front_ptr, S.front_nrows, S.front_ncols, S.cb_ptr, S.local_front, S.local_cb,
       S.child_ptr, S.child_list, S.relind_ptr, S.relind,
       Val((LB - SDS.SUBTREE_LOCAL_RESERVE) ÷ sizeof(T)), Val(WG); ndrange = WG * length(list) * bm.nact)
    return nothing
end

function build_tiles(S, plan)
    nodes = S.schedule.group_nodes
    rows_h = S.schedule.rows
    width_h = S.schedule.width
    cbp = Array(S.cb_ptr)
    # a C group's fronts with w <= 64 run through the split kernels (stock assembly, no vendor);
    # only wider fronts stay on the per-front vendor dense path
    narrow(k, v) = plan.group_width[k] > 0 || width_h[v] <= 64
    tiles = Dict{Int, Union{Nothing, NTuple{3, CuVector{Int32}}}}()
    chunks = Dict{Int, Union{Nothing, NTuple{2, CuVector{Int32}}}}()
    cnodes = Dict{Int, Union{Nothing, CuVector{Int32}}}()
    cwide = Dict{Int, Vector{Int}}()
    for k in eachindex(plan.group_first)
        tf = Int32[]; ti = Int32[]; tj = Int32[]
        cf = Int32[]; cc = Int32[]
        nn = Int32[]; wide = Int[]
        for q in plan.group_first[k]:plan.group_last[k]
            v = nodes[q]
            if !narrow(k, v)
                push!(wide, v)
                continue
            end
            plan.group_width[k] > 0 || push!(nn, Int32(v))
            m = rows_h[v] - width_h[v]
            m > SYRK_IN || continue
            for c in 1:cld(m, 128)
                push!(cf, Int32(v)); push!(cc, Int32(c))
            end
            cbp[v] > 0 || continue
            T = cld(m, TILE)
            for tjj in 1:T, tii in tjj:T
                push!(tf, Int32(v)); push!(ti, Int32(tii)); push!(tj, Int32(tjj))
            end
        end
        tiles[k] = isempty(tf) ? nothing : (CuArray(tf), CuArray(ti), CuArray(tj))
        chunks[k] = isempty(cf) ? nothing : (CuArray(cf), CuArray(cc))
        cnodes[k] = isempty(nn) ? nothing : CuArray(nn)
        cwide[k] = wide
    end
    chp = Array(S.child_ptr); chl = Array(S.child_list)
    ea = Dict{Int, Vector{NTuple{3, CuVector{Int32}}}}()
    for k in eachindex(plan.group_first)
        plan.group_width[k] > 0 && continue
        per_kc = NTuple{3, CuVector{Int32}}[]
        for kc in 1:plan.group_maxchild[k]
            ef = Int32[]; ec = Int32[]; eq = Int32[]
            for q in plan.group_first[k]:plan.group_last[k]
                v = nodes[q]
                chp[v] + kc - 1 < chp[v + 1] || continue
                c = chl[chp[v] + kc - 1]
                cbp[c] > 0 || continue
                mc = rows_h[c] - width_h[c]
                for ch in 1:cld(mc * mc, 16384)
                    push!(ef, Int32(v)); push!(ec, Int32(c)); push!(eq, Int32(ch))
                end
            end
            isempty(ef) || push!(per_kc, (CuArray(ef), CuArray(ec), CuArray(eq)))
        end
        ea[k] = per_kc
    end
    stp2 = Array(S.subtree_ptr)
    ssub = Dict{Int, CuVector{Int}}()
    for k in eachindex(plan.sub_first)
        ids = [nodes[q] for q in plan.sub_first[k]:plan.sub_last[k]]
        sort!(ids; by = e -> stp2[e + 1] - stp2[e], rev = true)
        ssub[k] = CuArray(ids)
    end
    return tiles, chunks, cnodes, cwide, ea, ssub
end

function refact_split!(s, nzval, tiles, chunks, cnodes, cwide, ea, ssub)
    N, S = s.numeric, s.symbolic
    p = SDS._front_impls(N, S, :auto)
    plan = N.plan
    nodes = S.schedule.group_nodes
    backend = KA.get_backend(N.factor)
    for k in eachindex(plan.sub_first)
        launch_subtrees_sorted!(N, S, nzval, ssub[k], plan.sub_local[k], backend)
    end
    for k in eachindex(plan.group_first)
        a, b = plan.group_first[k], plan.group_last[k]
        W = plan.group_width[k]
        launch_panel(nodearr, first, count, maxchild, W, asm) = begin
            bm = SDS.batch_map(N; first)
            args = (N.factor, N.stack, N.info, nzval, S.amap, S.amap_ptr, S.amap_src, nodearr, bm,
                    S.front_ptr, S.front_nrows, S.front_ncols, S.cb_ptr, S.child_ptr, S.child_list,
                    S.relind_ptr, S.relind, maxchild)
            nd = WGF * count
            if W == 8
                panel_cholesky_kernel!(backend, WGF)(args..., asm, Val(8), Val(36), Val(WGF); ndrange = nd)
            elseif W == 16
                panel_cholesky_kernel!(backend, WGF)(args..., asm, Val(16), Val(136), Val(WGF); ndrange = nd)
            elseif W == 32
                panel_cholesky_kernel!(backend, WGF)(args..., asm, Val(32), Val(528), Val(WGF); ndrange = nd)
            else
                panel_cholesky_kernel!(backend, WGF)(args..., asm, Val(64), Val(2080), Val(WGF); ndrange = nd)
            end
        end
        run_split(k) = begin
            c = chunks[k]
            c === nothing ||
                trsm_chunk_kernel!(backend, WGF)(N.factor, N.info, c[1], c[2], S.front_ptr, S.front_nrows,
                                                 S.front_ncols, Val(WGF); ndrange = WGF * length(c[1]))
            t = tiles[k]
            t === nothing ||
                syrk_tile_kernel!(backend, WGF)(N.stack, N.factor, N.info, t[1], t[2], t[3], S.front_ptr,
                                                S.front_nrows, S.front_ncols, S.cb_ptr, Val(WGF);
                                                ndrange = WGF * length(t[1]))
        end
        if W > 0
            launch_panel(S.group_nodes, a, b - a + 1, Int(plan.group_maxchild[k]), W, Val(true))
            run_split(k)
        else
            SDS.zero_fronts!(N, S, a, b - a + 1)
            SDS.scatter_A!(N, S, nzval, a, b - a + 1)
            for t3 in ea[k]
                extend_add_chunk_kernel!(backend, WGF)(N.factor, N.stack, t3[1], t3[2], t3[3], S.front_ptr,
                                                       S.front_nrows, S.front_ncols, S.cb_ptr, S.relind_ptr,
                                                       S.relind, Val(WGF); ndrange = WGF * length(t3[1]))
            end
            nn = cnodes[k]
            nn === nothing || launch_panel(nn, 1, length(nn), 0, 64, Val(false))
            run_split(k)
            for v in cwide[k]
                SDS._factor_front_c!(N, S, Int(v), p)
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

function run_config(label; cw = 32, rows = 512, sp = 16384, amalg = nothing, T = Float64, scale = false, delta = 0.0)
    if scale
        dsc = 1.0 ./ sqrt.(abs.(Vector(diag(A))))
        Dm = Diagonal(dsc)
        Ls = tril(SparseMatrixCSC{Float64, Int}(Dm * A * Dm + delta * I))
        bs = dsc .* bh
    else
        dsc = ones(n); Ls = Lh; bs = bh
    end
    bd = CuArray(T.(bs)); xd = similar(bd)
    s = DirectSolver(CuSparseMatrixCSR(SparseMatrixCSC{T, Int}(Ls)), "SPD", 'L')
    s.options.regime_c_width = cw
    s.options.regime_c_rows = rows
    s.options.subtree_parallelism = sp
    amalg === nothing || setfield!(s.options, :amalgamation,
        typeof(getfield(s.options, :amalgamation))(amalg))
    SDS.setparam!(s, "user_perm", UP)
    ex(p) = SDS.execute!(p, s, xd, bd; asynchronous = false)
    ex("analysis"); ex("factorization"); ex("refactorization")
    CUDA.synchronize()
    t0 = median([(@elapsed ex("refactorization")) for _ in 1:10])
    ex("solve")
    rel0 = norm(bh - A * (dsc .* Float64.(Array(xd)))) / norm(bh)
    fac0 = Array(s.numeric.factor)
    nzval = SDS._factor_values(s)
    tiles, chunks, cnodes, cwide, ea, ssub = build_tiles(s.symbolic, s.numeric.plan)
    refact_split!(s, nzval, tiles, chunks, cnodes, cwide, ea, ssub); CUDA.synchronize()
    info = refact_info!(s)
    fac1 = Array(s.numeric.factor)
    ex("solve")
    rel1 = norm(bh - A * (dsc .* Float64.(Array(xd)))) / norm(bh)
    for _ in 1:2; refact_split!(s, nzval, tiles, chunks, cnodes, cwide, ea, ssub); end
    CUDA.synchronize()
    t1 = median([(@elapsed (refact_split!(s, nzval, tiles, chunks, cnodes, cwide, ea, ssub); CUDA.synchronize())) for _ in 1:10])
    nt = sum(t === nothing ? 0 : length(t[1]) for t in values(tiles))
    tg = NaN
    try
        g = CUDA.capture() do
            refact_split!(s, nzval, tiles, chunks, cnodes, cwide, ea, ssub)
        end
        exec = CUDA.instantiate(g)
        replay() = (CUDA.launch(exec); CUDA.synchronize())
        replay()
        fac2 = Array(s.numeric.factor)

        tg = median([(@elapsed replay()) for _ in 1:10])
    catch err
        println("    graph capture failed: ", sprint(showerror, err)[1:min(end, 200)])
    end
    @printf "%-26s stock %7.2f ms | split %7.2f ms | graph %7.2f ms  info %d  tiles %d  Δfactor %.2e  relres %.2e (stock %.2e)\n" label t0*1e3 t1*1e3 tg*1e3 info nt norm(fac1 - fac0) / norm(fac0) rel1 rel0
    flush(stdout)
    GC.gc(); CUDA.reclaim()
end

run_config("FP32 scaled d1e-6"; rows = 64, T = Float32, scale = true, delta = 1e-6)
run_config("FP32 scaled d1e-4"; rows = 64, T = Float32, scale = true, delta = 1e-4)
run_config("FP64 scaled d1e-6"; rows = 64, scale = true, delta = 1e-6)
println("done")
