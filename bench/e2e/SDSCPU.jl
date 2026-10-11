# CPU fast path for the refactorization (prototype, stock src untouched): the
# stock schedule's groups processed in order, with
#   - subtree groups: one Julia task per subtree, the serial multifrontal stack
#     in a per-task scratch (the regime-A algorithm in plain scalar Julia);
#   - regime-B/C groups: the stock assembly kernels (KA CPU, internally
#     threaded), then one task per front running LAPACK potrf + BLAS trsm/syrk
#     on the dense panel views (BLAS pinned to one thread; task-level
#     parallelism is ours).
# Cholesky ("SPD") first; the potrf-derived LDL conversion drops in later.
module SDSCPU

using SparseDirectSolver
using LinearAlgebra
using Base.Threads
using Printf
const SDS = SparseDirectSolver

_packed(i, j, n) = SDS._packed(i, j, n)

# split a:b into at most n contiguous chunks
function _chunks(a::Int, b::Int, n::Int)
    len = b - a + 1
    len <= 0 && return UnitRange{Int}[]
    c = min(n, len)
    sz = cld(len, c)
    return [ (a + (i - 1) * sz):min(a + i * sz - 1, b) for i in 1:cld(len, sz) ]
end

# serial regime-A subtree: the GPU kernel's algorithm on a plain scratch vector
function _subtree!(factor::Vector{T}, stack::Vector{T}, info, nzval, S, t, buf::Vector{T}) where {T}
    stn = S.subtree_nodes
    @inbounds for q in S.subtree_ptr[t]:(S.subtree_ptr[t + 1] - 1)
        v = Int(stn[q])
        f = Int(S.front_nrows[v])
        w = Int(S.front_ncols[v])
        m = f - w
        lf = Int(S.local_front[v]) - 1
        # zero + scatter A
        fill!(view(buf, (lf + 1):(lf + f * (f + 1) ÷ 2)), zero(T))
        p0 = Int(S.front_ptr[v])
        for k in Int(S.amap_ptr[v]):(Int(S.amap_ptr[v + 1]) - 1)
            p = Int(S.amap_src[k])
            off = S.amap[p]
            loc = Int(abs(off)) - p0
            j = loc ÷ f + 1
            i = loc - (j - 1) * f + 1
            buf[lf + _packed(i, j, f)] += SDS._amap_value(T, nzval[p], off)
        end
        # extend-add the children (their packed CBs live lower in the scratch)
        for kc in Int(S.child_ptr[v]):(Int(S.child_ptr[v + 1]) - 1)
            c = Int(S.child_list[kc])
            mc = Int(S.front_nrows[c]) - Int(S.front_ncols[c])
            lc = Int(S.local_cb[c]) - 1
            r0 = Int(S.relind_ptr[c]) - 1
            for jj in 1:mc
                rj = Int(S.relind[r0 + jj])
                for ii in jj:mc
                    ri = Int(S.relind[r0 + ii])
                    buf[lf + _packed(ri, rj, f)] += buf[lc + _packed(ii, jj, mc)]
                end
            end
        end
        # right-looking Cholesky of the first w columns (whole packed front)
        st = Int32(0)
        for j in 1:w
            dj = real(buf[lf + _packed(j, j, f)])
            if !(dj > 0)
                st = Int32(j)
                break
            end
            c = sqrt(dj)
            buf[lf + _packed(j, j, f)] = c
            rd = inv(c)
            base = lf + _packed(j, j, f)                 # column j is contiguous below the diagonal
            @simd for i in 1:(f - j)
                buf[base + i] *= rd
            end
            for k in (j + 1):f
                lkj = conj(buf[base + k - j])
                bk = lf + _packed(k, k, f) - k
                @simd for i in k:f
                    buf[bk + i] -= buf[base + i - j] * lkj
                end
            end
        end
        # write the panel (zero strict upper), status, and the root CB to the stack
        for j in 1:w, i in 1:f
            factor[p0 + (j - 1) * f + i - 1] = i >= j ? buf[lf + _packed(i, j, f)] : zero(T)
        end
        st != 0 && (info[v] = st)
        lc = Int(S.local_cb[v]) - 1
        src = lf + w * (2 * f - w + 1) ÷ 2
        if lc < 0                                       # subtree root: CB to the update stack
            c0 = Int(S.cb_ptr[v])
            c0 > 0 && copyto!(stack, c0, buf, src + 1, m * (m + 1) ÷ 2)
        else                                            # move the CB down (src > dst: forward copy)
            copyto!(buf, lc + 1, buf, src + 1, m * (m + 1) ÷ 2)
        end
    end
    return nothing
end

# extend-add of all children into parent columns rj ∈ cols (panel) and CB columns
# rj - w ∈ cols .- w; disjoint across column slices, so safe to run sliced in parallel
function _extend_cols!(factor::Vector{T}, stack::Vector{T}, S, v, f, w, m, p0, c0, cols) where {T}
    @inbounds for kc in Int(S.child_ptr[v]):(Int(S.child_ptr[v + 1]) - 1)
        c = Int(S.child_list[kc])
        cc = Int(S.cb_ptr[c])
        cc > 0 || continue
        mc = Int(S.front_nrows[c]) - Int(S.front_ncols[c])
        r0 = Int(S.relind_ptr[c]) - 1
        for jj in 1:mc
            rj = Int(S.relind[r0 + jj])
            rj in cols || continue
            if rj <= w
                b = p0 + (rj - 1) * f - 1
                for ii in jj:mc
                    factor[b + Int(S.relind[r0 + ii])] += stack[cc + _packed(ii, jj, mc) - 1]
                end
            else
                bj = c0 + _packed(rj - w, rj - w, m) - 1 - (rj - w)
                for ii in jj:mc
                    ri = Int(S.relind[r0 + ii])
                    stack[bj + ri - w] += stack[cc + _packed(ii, jj, mc) - 1]
                end
            end
        end
    end
    return nothing
end

# one regime-B/C front: assembly (zero + scatter A + extend-add, serial and cache-local)
# then LAPACK potrf + BLAS trsm + syrk, CB packed into the stack
function _front_blas!(factor::Vector{T}, stack::Vector{T}, work::Vector{T}, info, S, nzval, v,
                      par::Bool = false) where {T}
    f = Int(S.front_nrows[v])
    w = Int(S.front_ncols[v])
    m = f - w
    p0 = Int(S.front_ptr[v])
    c0 = Int(S.cb_ptr[v])
    @inbounds begin
        if par
            @sync begin
                for r in _chunks(1, f * w, 4 * nthreads())
                    Threads.@spawn fill!(view(factor, (p0 + first(r) - 1):(p0 + last(r) - 1)), zero(T))
                end
                c0 > 0 && for r in _chunks(1, m * (m + 1) ÷ 2, 2 * nthreads())
                    Threads.@spawn fill!(view(stack, (c0 + first(r) - 1):(c0 + last(r) - 1)), zero(T))
                end
            end
        else
            fill!(view(factor, p0:(p0 + f * w - 1)), zero(T))
            c0 > 0 && fill!(view(stack, c0:(c0 + m * (m + 1) ÷ 2 - 1)), zero(T))
        end
        for k in Int(S.amap_ptr[v]):(Int(S.amap_ptr[v + 1]) - 1)
            p = Int(S.amap_src[k])
            off = S.amap[p]
            factor[Int(abs(off))] += SDS._amap_value(T, nzval[p], off)
        end
        if par
            @sync for r in _chunks(1, f, 4 * nthreads())
                Threads.@spawn _extend_cols!(factor, stack, S, v, f, w, m, p0, c0, r)
            end
        else
            _extend_cols!(factor, stack, S, v, f, w, m, p0, c0, 1:f)
        end
    end
    P = reshape(view(factor, p0:(p0 + f * w - 1)), f, w)
    F11 = view(P, 1:w, 1:w)
    _, st = LAPACK.potrf!('L', F11)
    if st != 0
        @inbounds info[v] = Int32(st)
        return nothing
    end
    m > 0 || return nothing
    F21 = view(P, (w + 1):f, 1:w)
    BLAS.trsm!('R', 'L', 'C', 'N', one(T), F11, F21)
    c0 > 0 || return nothing
    C = reshape(view(work, 1:(m * m)), m, m)
    T <: Real ? BLAS.syrk!('L', 'N', -one(T), F21, zero(T), C) :
                BLAS.herk!('L', 'N', -one(real(T)), F21, zero(real(T)), C)
    if par
        @sync for r in _chunks(1, m, 4 * nthreads())
            Threads.@spawn @inbounds for jj in r
                b = c0 + _packed(jj, jj, m) - 1 - jj
                @simd for ii in jj:m
                    stack[b + ii] += C[ii, jj]
                end
            end
        end
    else
        @inbounds for jj in 1:m
            b = c0 + _packed(jj, jj, m) - 1 - jj
            @simd for ii in jj:m
                stack[b + ii] += C[ii, jj]
            end
        end
    end
    return nothing
end

"""
    CPUCtx(s)

Persistent buffers for [`refact_cpu!`](@ref): per-task scratch and workspace
pools (task-safe channels; threadid-indexed buffers break under task
migration).
"""
struct CPUCtx{T}
    spool::Channel{Vector{T}}
    wpool::Channel{Vector{T}}
end

function CPUCtx(s)
    N = s.numeric
    T = eltype(N.factor)
    plan = N.plan
    maxloc = isempty(plan.sub_local) ? 0 : maximum(plan.sub_local)
    spool = Channel{Vector{T}}(nthreads())
    wpool = Channel{Vector{T}}(nthreads())
    for _ in 1:nthreads()
        put!(spool, Vector{T}(undef, cld(maxloc, sizeof(T))))
        put!(wpool, Vector{T}(undef, length(N.work)))
    end
    return CPUCtx{T}(spool, wpool)
end

"""
    refact_cpu!(s, nzval, ctx = CPUCtx(s)) -> info

Task-parallel CPU refactorization of a factorized "SPD" solver on the CPU
backend, writing the stock factor layout (panels + packed CB stack). Wide
levels run one task per chunk of fronts with BLAS pinned to one thread; thin
levels (the root chain) run serially under fully threaded BLAS.
"""
function refact_cpu!(s, nzval, ctx::CPUCtx = CPUCtx(s))
    N, S = s.numeric, s.symbolic
    plan = N.plan
    T = eltype(N.factor)
    nblas = BLAS.get_num_threads()
    nt = nthreads()
    fill!(N.info, Int32(0))
    spool, wpool = ctx.spool, ctx.wpool
    nch = 8 * nt                                       # chunked tasks: spawn cost amortized
    tsub = 0.0; tblas = 0.0
    wk0 = take!(wpool)                                  # serial-level workspace
    try
        BLAS.set_num_threads(1)
        tsub = @elapsed for k in eachindex(plan.sub_first)
            a, b = plan.sub_first[k], plan.sub_last[k]
            @sync for r in _chunks(a, b, nch)
                Threads.@spawn begin
                    buf = take!(spool)
                    try
                        for q in r
                            _subtree!(N.factor, N.stack, N.info, nzval, S, Int(S.group_nodes[q]), buf)
                        end
                    finally
                        put!(spool, buf)
                    end
                end
            end
        end
        for k in eachindex(plan.group_first)
            a, b = plan.group_first[k], plan.group_last[k]
            if b - a + 1 <= max(2, nt ÷ 8)             # thin level: serial fronts, threaded BLAS
                tblas += @elapsed begin
                    BLAS.set_num_threads(nt)
                    # par = true (sliced assembly) measured SLOWER here: _extend_cols! rescans
                    # every child column per slice, O(slices x columns); a per-slice column
                    # index built at plan time would be needed to make it pay
                    for q in a:b
                        _front_blas!(N.factor, N.stack, wk0, N.info, S, nzval, Int(S.group_nodes[q]), false)
                    end
                    BLAS.set_num_threads(1)
                end
            else
                tblas += @elapsed @sync for r in _chunks(a, b, nch)
                    Threads.@spawn begin
                        wk = take!(wpool)
                        try
                            for q in r
                                _front_blas!(N.factor, N.stack, wk, N.info, S, nzval, Int(S.group_nodes[q]))
                            end
                        finally
                            put!(wpool, wk)
                        end
                    end
                end
            end
        end
    finally
        put!(wpool, wk0)
        BLAS.set_num_threads(nblas)
    end
    haskey(ENV, "SDSCPU_STAGES") &&
        @printf("  stages: subtrees %7.1f ms  fronts %7.1f ms\n", tsub * 1e3, tblas * 1e3)
    SDS.assemble_schur!(N, S, nzval)
    return maximum(Array(N.info)) == 0 ? 0 : 1
end

end # module
