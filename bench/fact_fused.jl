# Prototype, experiment 5 for the factorization (variant i, full overlap): ONE
# dependency-counter mega-kernel whose workgroups play four roles in one
# topologically ordered grid — extend-add chunks, panels (assembly + F11
# Cholesky), trsm row-chunks, syrk tiles — while the wide (w > 64) fronts run
# concurrently on a second stream, each bracketed by a 1-block waiter (spins
# until the front's extend-add chunks arrived) and a signaler. Subtrees run as
# the stock launches before the mega-kernel. CUDA-only, SPD, nbatch = 1.
#
# Dependency state per supernode (Int32 arrays, reset per refactorization):
#   cb_done[v]    1 when v's contribution block is complete (signal)
#   arrived[v]    # completed children (B panels spin on == nchild[v])
#   ea_left[v]    # extend-add chunk workgroups still to run into v
#   panel_done[v] 1 when v's L11/writeback is done (trsm chunks spin)
#   trsm_left[v]  # trsm chunks left (syrk tiles spin on 0)
#   tiles_left[v] # syrk tiles left (last one signals)
# A signal is: threadfence, cb_done[v] = 1, arrived[parent] += 1. Deadlock
# safety: block ids are assigned in plan (topological) order, so every
# workgroup's dependencies are earlier blocks (dispatched first) or stream-2
# work that only needs single-block waiters to be schedulable.
#
#   CUDA_VISIBLE_DEVICES=1 julia +1.13 --project=bench bench/fact_fused.jl

using SparseDirectSolver, SparseArrays, LinearAlgebra, Random, Statistics, Printf
using CUDA, CUDA.CUSPARSE
using Metis
const SDS = SparseDirectSolver
const KA = SDS.KernelAbstractions
const Atomix = SDS.Atomix
using .KA: @kernel, @index, @localmem, @synchronize, @Const, @uniform

include(joinpath(@__DIR__, "matrices.jl"))
using .BenchMatrices: read_mtx

const SYRK_IN = 64
const TILE = 32
const WGF = 128
const NL64 = 2080                 # packed 64x64 lower triangle

const ROLE_PANEL_B = Int8(1)
const ROLE_PANEL_C = Int8(2)
const ROLE_TRSM = Int8(3)
const ROLE_TILE = Int8(4)
const ROLE_EA = Int8(5)

# spin with backoff on an Int32 cell until pred(value) holds; thread 1 only
@inline function _spin_until_eq!(arr, idx, target, scratch)
    pause = Int32(4)
    while (Atomix.@atomic arr[idx] += Int32(0)) != target
        z = 0.0
        for _ in 1:pause
            z += 1.0
        end
        z < 0 && (scratch[1] = z)
        pause = min(pause << 1, Int32(128))
    end
    return nothing
end

@kernel function fused_fact_kernel!(factor, stack, pcb, info, @Const(nzval), @Const(amap), @Const(amap_ptr),
                                    @Const(amap_src), @Const(cb2), @Const(wt), @Const(role), @Const(ra), @Const(rb), @Const(rc),
                                    @Const(nchild), arrived, ea_left, panel_done, trsm_left, tiles_left,
                                    cb_done, @Const(snparent), @Const(front_ptr), @Const(front_nrows),
                                    @Const(front_ncols), @Const(cb_ptr), @Const(child_ptr),
                                    @Const(child_list), @Const(relind_ptr), @Const(relind), base,
                                    ::Val{WG}) where {WG}
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
                _spin_until_eq!(cb_done, c, Int32(1), fscr)
            end
            @synchronize
            CUDA.threadfence()
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
            if li == 1
                CUDA.threadfence()
                Atomix.@atomic ea_left[v] += Int32(-1)
            end
        elseif r == ROLE_PANEL_B || r == ROLE_PANEL_C
            v = Int(ra[G])
            if li == 1
                if r == ROLE_PANEL_B
                    _spin_until_eq!(arrived, v, nchild[v], fscr)
                else
                    _spin_until_eq!(ea_left, v, Int32(0), fscr)
                end
            end
            @synchronize
            CUDA.threadfence()
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
            for j in 1:w
                SDS._front_chol_update!(sh, st, piv, j, v, li, front_ncols, Val(64), Val(WG))
                @synchronize
                SDS._front_chol_scale!(sh, st, piv, j, v, li, front_ncols, Val(64), Val(WG))
                @synchronize
            end
            f = Int(front_nrows[v])
            p0 = Int(front_ptr[v])
            for q in (li - 1):WG:(w * w - 1)
                j = q ÷ w + 1
                i = q - (j - 1) * w + 1
                i >= j && (factor[p0 + (j - 1) * f + i - 1] = sh[SDS._packed(i, j, 64)])
            end
            m = f - w
            if st[1] == Int32(0) && m <= SYRK_IN
                for i in (w + li):WG:f
                    for kk in 1:w
                        x = factor[p0 + (kk - 1) * f + i - 1]
                        for j in 1:(kk - 1)
                            x -= factor[p0 + (j - 1) * f + i - 1] * sh[SDS._packed(kk, j, 64)]
                        end
                        factor[p0 + (kk - 1) * f + i - 1] = x / sh[SDS._packed(kk, kk, 64)]
                    end
                end
            end
            li == 1 && (info[v] = st[1])
            @synchronize
            if m <= SYRK_IN
                SDS._front_syrk!(factor, pcb, st, v, li, front_ptr, front_nrows, front_ncols, cb2,
                                 Val(WG))
                @synchronize
                if li == 1
                    CUDA.threadfence()
                    Atomix.@atomic cb_done[v] += Int32(1)
                    pv = Int(snparent[v])
                    pv > 0 && (Atomix.@atomic arrived[pv] += Int32(1))
                end
            else
                if li == 1
                    CUDA.threadfence()
                    Atomix.@atomic panel_done[v] += Int32(1)
                end
            end
        elseif r == ROLE_TRSM
            v = Int(ra[G])
            if li == 1
                _spin_until_eq!(panel_done, v, Int32(1), fscr)
            end
            @synchronize
            CUDA.threadfence()
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
                sh[SDS._packed(i, j, 64)] = factor[p0 + (j - 1) * f + i - 1]
            end
            @synchronize
            if info[v] == Int32(0)
                i = w + (Int(rb[G]) - 1) * WG + li
                if i <= f
                    for kk in 1:w
                        x = factor[p0 + (kk - 1) * f + i - 1]
                        for j in 1:(kk - 1)
                            x -= factor[p0 + (j - 1) * f + i - 1] * sh[SDS._packed(kk, j, 64)]
                        end
                        factor[p0 + (kk - 1) * f + i - 1] = x / sh[SDS._packed(kk, kk, 64)]
                    end
                end
            end
            @synchronize
            if li == 1
                CUDA.threadfence()
                left = (Atomix.@atomic trsm_left[v] += Int32(-1))
                if left == Int32(1) && tiles_left[v] == Int32(0)
                    Atomix.@atomic cb_done[v] += Int32(1)
                    pv = Int(snparent[v])
                    pv > 0 && (Atomix.@atomic arrived[pv] += Int32(1))
                end
            end
        elseif r == ROLE_TILE
            v = Int(ra[G])
            if li == 1
                _spin_until_eq!(trsm_left, v, Int32(0), fscr)
            end
            @synchronize
            CUDA.threadfence()
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
                if ok
                    for q in (li - 1):WG:(TILE * kb - 1)
                        rr = q % TILE + 1
                        cc = q ÷ TILE + 1
                        row = i0 + rr
                        sh[rr + (cc - 1) * TILE] = row <= m ? factor[p0 + (kk0 + cc - 1) * f + row] : zero(TT)
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
            if li == 1
                CUDA.threadfence()
                left = (Atomix.@atomic tiles_left[v] += Int32(-1))
                if left == Int32(1)
                    Atomix.@atomic cb_done[v] += Int32(1)
                    pv = Int(snparent[v])
                    pv > 0 && (Atomix.@atomic arrived[pv] += Int32(1))
                    Int(wt[v]) > 0 && (Atomix.@atomic ea_left[Int(wt[v])] += Int32(-1))
                end
            end
        end
    end
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
        pause = Int32(4)
        @inbounds while (Atomix.@atomic ea_left[v] += Int32(0)) != Int32(0)
            z = 0.0
            for _ in 1:pause
                z += 1.0
            end
            z < 0 && (ea_left[v] = Int32(0))
            pause = min(pause << 1, Int32(128))
        end
    end
end

@kernel function signal_kernel!(cb_done, arrived, @Const(snparent), v)
    li = @index(Local, Linear)
    @inbounds if li == 1
        CUDA.threadfence()
        Atomix.@atomic cb_done[v] += Int32(1)
        pv = Int(snparent[v])
        pv > 0 && (Atomix.@atomic arrived[pv] += Int32(1))
    end
end

# subtree launcher with a custom (sorted) list, local memory sized by eltype
function launch_subtrees_sorted!(N, S, nzval, list, lb, backend)
    SDS._with_local_bytes(Int(lb)) do LBv
        _sorted_subtree_launch(N, S, nzval, list, LBv, Val(SDS.SUBTREE_WORKGROUP), backend)
    end
end
function _sorted_subtree_launch(N::SDS.Numeric{T}, S, nzval, list, ::Val{LB}, ::Val{WG}, backend) where {T, LB, WG}
    bm = SDS.batch_map(N; first = 1)
    k! = SDS.subtree_cholesky_kernel!(backend, WG)
    k!(N.factor, N.stack, N.info, nzval, S.amap, S.amap_ptr, S.amap_src, list, bm, S.subtree_ptr,
       S.subtree_nodes, S.front_ptr, S.front_nrows, S.front_ncols, S.cb_ptr, S.local_front, S.local_cb,
       S.child_ptr, S.child_list, S.relind_ptr, S.relind,
       Val((LB - SDS.SUBTREE_LOCAL_RESERVE) ÷ sizeof(T)), Val(WG); ndrange = WG * length(list) * bm.nact)
    return nothing
end

# ---------------------------------------------------------------------------
# host: the topologically ordered role list + counter bases + stream-2 program
function build_fused_plan(s)
    S = s.symbolic
    plan = s.numeric.plan
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
        for ch in 1:cld(m, WGF)
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
                push!(segments, (seg_base, length(role) - seg_base, gwides))
                seg_base = length(role)
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
    ssub = Vector{CuVector{Int}}()
    for k in eachindex(plan.sub_first)
        ids = [nodes[q] for q in plan.sub_first[k]:plan.sub_last[k]]
        sort!(ids; by = e -> stp2[e + 1] - stp2[e], rev = true)
        push!(ssub, CuArray(ids))
    end

    println("segments: ", length(segments)); flush(stdout)
    return (; role = CuArray(role), ra = CuArray(ra), rb = CuArray(rb), rc = CuArray(rc),
            nblocks = length(role), wides, ssub, segments,
            cb2 = CuArray(Int32.(cb2_h)), cb2_h, pcb = CUDA.zeros(Float64, off - 1),
            czero = CuArray(czero), wt = CuArray(wt_h),
            nchild = CuArray(nchild),
            arrived_base = CuArray(arrived_base), ea_base = CuArray(ea_left),
            trsm_base = CuArray(trsm_left), tiles_base = CuArray(tiles_left),
            cb_base = CuArray(cb_done_base),
            arrived = CUDA.zeros(Int32, ns), ea = CUDA.zeros(Int32, ns),
            trsm = CUDA.zeros(Int32, ns), tiles = CUDA.zeros(Int32, ns),
            pdone = CUDA.zeros(Int32, ns), cbd = CUDA.zeros(Int32, ns),
            s2 = CUDA.CuStream())
end

function refact_fused!(s, nzval, fp)
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
    for k in eachindex(plan.sub_first)
        launch_subtrees_sorted!(N, S, nzval, fp.ssub[k], plan.sub_local[k], backend)
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
            fused_fact_kernel!(backend, WGF)(N.factor, N.stack, fp.pcb, N.info, nzval, S.amap, S.amap_ptr,
                                             S.amap_src, fp.cb2, fp.wt, fp.role, fp.ra, fp.rb, fp.rc, fp.nchild,
                                             fp.arrived, fp.ea, fp.pdone, fp.trsm, fp.tiles, fp.cbd,
                                             S.snparent, S.front_ptr, S.front_nrows, S.front_ncols,
                                             S.cb_ptr, S.child_ptr, S.child_list, S.relind_ptr, S.relind,
                                             Int32(base), Val(WGF); ndrange = WGF * count)
        for v in gwides
            SDS._factor_panel_c!(N.factor, fp.pcb, N.work, N.info, v, S.layout.panel_ptr[v],
                                 S.schedule.rows[v], S.schedule.width[v], fp.cb2_h[v], p)
            signal_kernel!(backend, 32)(fp.cbd, fp.arrived, S.snparent, v; ndrange = 32)
        end
    end
    SDS.assemble_schur!(N, S, nzval)
    SDS.cholesky_stats!(N, S)
    return SDS._numeric_info!(N, S, true)
end

# ---------------------------------------------------------------------------
const PATH = joinpath(@__DIR__, "data", "kkt_pglib_opf_case78484_epigrids_condensed_10.mtx")
A = SparseMatrixCSC{Float64, Int}(sparse(read_mtx(PATH)))
n = size(A, 1)
Random.seed!(666)
bh = rand(n)
Lh = tril(A)
const UP = (p = Vector{Int32}(undef, n); read!(joinpath(@__DIR__, "cudss_perm.bin"), p); Int.(p))

function run_fused(label; cw = 32, rows = 64, sp = 16384, uperm = UP, alg = nothing, smf = 0)
    bd = CuArray(bh); xd = similar(bd)
    s = DirectSolver(CuSparseMatrixCSR(Lh), "SPD", 'L')
    s.options.regime_c_width = cw
    s.options.regime_c_rows = rows
    s.options.subtree_parallelism = sp
    s.options.subtree_max_fronts = smf
    uperm === nothing || SDS.setparam!(s, "user_perm", uperm)
    alg === nothing || SDS.setparam!(s, "reordering_alg", alg)
    ex(ph) = SDS.execute!(ph, s, xd, bd; asynchronous = false)
    print("analysis… "); @time ex("analysis")
    ex("factorization"); ex("refactorization"); CUDA.synchronize()
    t0 = median([(@elapsed ex("refactorization")) for _ in 1:10])
    ex("solve")
    rel0 = norm(bh - A * Array(xd)) / norm(bh)
    fac0 = Array(s.numeric.factor)
    nzval = SDS._factor_values(s)
    fp = build_fused_plan(s)
    println("mega blocks: ", fp.nblocks, "  wides: ", length(fp.wides)); flush(stdout)
    info = refact_fused!(s, nzval, fp); CUDA.synchronize()
    begin
        sc = s.symbolic.schedule
        ns = length(sc.width)
        infos = Array(s.numeric.info)[1:ns]
        bad = findall(!=(Int32(0)), infos)
        println("failing fronts: ", length(bad))
        if !isempty(bad)
            sort!(bad; by = v -> sc.level[v])
            for v in bad[1:min(6, end)]
                println("  v=", v, " level=", Int(sc.level[v]), " regime=", Int(sc.regime[v]),
                        " w=", Int(sc.width[v]), " f=", Int(sc.rows[v]), " info=", infos[v])
            end
            flush(stdout)
        end
    end
    fac1 = Array(s.numeric.factor)
    ex("solve")
    rel1 = norm(bh - A * Array(xd)) / norm(bh)
    for _ in 1:2
        refact_fused!(s, nzval, fp)
    end
    CUDA.synchronize()
    t1 = median([(@elapsed (refact_fused!(s, nzval, fp); CUDA.synchronize())) for _ in 1:10])
    @printf "%-22s stock %7.2f ms | fused %7.2f ms  info %d  Δfactor %.2e  relres %.2e (stock %.2e)\n" label t0*1e3 t1*1e3 info norm(fac1 - fac0) / norm(fac0) rel1 rel0
    flush(stdout)
    GC.gc(); CUDA.reclaim()
end

run_fused("ND smf=0 (base)"; uperm = nothing, alg = "algo4")
run_fused("ND smf=96"; uperm = nothing, alg = "algo4", smf = 96)
run_fused("ND smf=64"; uperm = nothing, alg = "algo4", smf = 64)
run_fused("ND smf=48"; uperm = nothing, alg = "algo4", smf = 48)
run_fused("ND smf=32"; uperm = nothing, alg = "algo4", smf = 32)
println("done")
