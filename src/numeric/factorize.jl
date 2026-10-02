# Level driver of the numeric phase (PLAN §2.4, §3.9): a fixed sequence of
# launches over the steps of the schedule, no allocation, no host
# synchronization except the final read of the reduced `info`. Regime A runs
# first, one fused subtree kernel launch per budget class
# (`src/numeric/subtree.jl`); then regime-B groups run the fused per-front kernel
# (`src/numeric/front.jl`, one launch per group) and regime-C groups the
# assembly kernels, then `potrf`/`trsm`/`syrk` (`herk`) per front through the
# dense interface, the `syrk` result packed into the front's block.

"Workgroup size of the per-phase reduction of the front statistics."
const STATS_WORKGROUP = 256

@kernel function _cholesky_stats_kernel!(stats, info, super_ptr, front_ncols, ns, ::Val{WG}, ::Val{NF},
                                         ::Val{LOG2WG}) where {WG, NF, LOG2WG}
    li = @index(Local, Linear)
    best = @localmem Int64 (WG,)
    @inbounds begin
        m = typemax(Int64)
        for s in li:WG:ns
            fi = Int64(info[s])
            base = (s - 1) * NF
            stats[base + 1] = fi == 0 ? Int64(front_ncols[s]) : fi - 1
            for k in 2:(NF - 1)
                stats[base + k] = 0
            end
            stats[base + NF] = fi
            fi > 0 && (m = min(m, Int64(super_ptr[s]) + fi - 1))
        end
        best[li] = m
    end
    @synchronize
    for lev in 1:LOG2WG
        @inbounds begin
            h = WG >> lev
            if li <= h
                best[li] = min(best[li], best[li + h])
            end
        end
        @synchronize
    end
    if li == 1
        @inbounds info[ns + 1] = best[1] == typemax(Int64) ? Int32(0) : Int32(best[1])
    end
end

"""
    cholesky_stats!(numeric, symbolic) -> numeric

Fill the per-front statistics of a Cholesky factorization from the per-front
`potrf` status `numeric.info[1:ns]` (`npos` = `w`, or the failed local column
minus one; `info` = the failed local column) and reduce the smallest failed
factor column into `numeric.info[ns + 1]` (0 = none). One workgroup, no
atomics. Asynchronous.
"""
function cholesky_stats!(N::Numeric, S::Symbolic)
    WG = STATS_WORKGROUP
    kernel! = _cholesky_stats_kernel!(KernelAbstractions.get_backend(N.factor), WG)
    kernel!(N.stats, N.info, S.super_ptr, S.front_ncols, nsupernodes(S), Val(WG), Val(FRONT_STATS_FIELDS),
            Val(_ilog2(WG)); ndrange = WG)
    return N
end

function _check_numeric(N::Numeric{T}, S::Symbolic, nzval::AbstractVector) where {T}
    _is_ldlt_structure(S.structure) ? _check_ldlt_eltype(T) : _check_reference_cholesky(S, T)
    sizeof(T) <= S.elsize ||
        throw(InvalidValueError("the analysis was built for $(S.elsize)-byte elements, the numeric storage has " *
                                "$(sizeof(T))-byte $T"))
    length(nzval) == S.nnz ||
        throw(InvalidValueError("nzval has $(length(nzval)) entries, the analysis expects $(S.nnz)"))
    length(N.factor) == S.layout.factor_len && length(N.stack) == S.layout.stack_len &&
        length(N.info) == nsupernodes(S) + 1 ||
        throw(InvalidValueError("the numeric storage was not allocated for this analysis"))
    length(N.work) == S.layout.work_len && length(N.piv) == S.n && length(N.psign) == S.n ||
        throw(InvalidValueError("the numeric storage was not allocated for this analysis"))
    backend = KernelAbstractions.get_backend(N.factor)
    for x in (nzval, S.amap, N.info)
        typeof(KernelAbstractions.get_backend(x)) == typeof(backend) ||
            throw(InvalidValueError("nzval, the device maps (adapt the Symbolic) and the numeric storage must " *
                                    "live on the same backend"))
    end
    return nothing
end

# the dense implementations of the regime-C front kernels, resolved once per phase (backend and `T` are
# fixed), so the per-front calls do no capability lookup (its lock and closure allocate);
# `factorization_alg = "algo1"` (no vendor calls) turns `:auto` into the KA kernels
function _front_impls(N::Numeric{T}, S::Symbolic, impl::Symbol) where {T}
    impl === :auto && !S.schedule.vendor_c && (impl = :ka)
    return (potrf = select_impl(:potrf, N.factor, impl), trsm = select_impl(:trsm, N.factor, impl),
            herk = select_impl(T <: Real ? :syrk : :herk, N.factor, impl))
end

# regime-C dense kernels of front `s` (assembled panel + contribution block), `p` from `_front_impls`
function _factor_front_c!(N::Numeric{T}, S::Symbolic, s::Int, p::NamedTuple) where {T}
    L, sc = S.layout, S.schedule
    _factor_panel_c!(N.factor, N.stack, N.work, N.info, s, L.panel_ptr[s], sc.rows[s], sc.width[s], L.cb_ptr[s], p)
    return nothing
end

# `potrf`/`trsm`/`syrk` (`herk`) on the `f×w` panel at `factor[p0]`; the `syrk` result goes to the
# `m×m` workspace `work[1:m^2]` and is added to the packed contribution block at `stack[c0]`
# (`c0 = 0`: none); status into `info[s]` (also used by bench/front_bins.jl)
function _factor_panel_c!(factor::AbstractVector{T}, stack::AbstractVector{T}, work::AbstractVector{T},
                          info::AbstractVector{Int32}, s::Int, p0::Int, f::Int, w::Int, c0::Int,
                          p::NamedTuple) where {T}
    m = f - w
    P = reshape(view(factor, p0:(p0 + f * w - 1)), f, w)
    F11 = view(P, 1:w, 1:w)
    _potrf_info_impl!(p.potrf, 'L', F11, info, s)
    m > 0 || return nothing
    F21 = view(P, (w + 1):f, 1:w)
    _trsm_impl!(p.trsm, 'R', 'L', 'C', 'N', one(T), F11, F21)
    c0 > 0 || return nothing                                  # no parent: nothing to update
    C = reshape(view(work, 1:(m * m)), m, m)
    _herk_impl!(p.herk, 'L', C, F21, -one(real(T)), zero(real(T)))
    pack_add!(stack, c0, work, m)
    return nothing
end

_check_ldlt_eltype(::Type{T}) where {T} = T <: LinearAlgebra.BlasFloat ||
    throw(NotSupportedError("LDLᵀ/LDLᴴ supports BLAS element types only, got $T"))

"""
    factorize!(numeric, symbolic, nzval; impl = :auto, opts = Options()) -> info::Int

Structures `"S"`/`"H"`: [`factorize_ldlt!`](@ref) with the pivoting options
`opts` (`impl` is not used). Structures `"SPD"`/`"HPD"`: multifrontal Cholesky `P A Pᵀ = L Lᴴ` on the device: `nzval` are the stored
values of A (same pattern, view and index base as the analysis) on the backend
of `numeric`, and `symbolic` has its maps on that backend
(`adapt(backend, symbolic, INT)`). For every launch group of the schedule,
in order: a regime-A group (budget class) is one [`factorize_subtrees!`](@ref)
launch (one workgroup per subtree: its fronts assembled, factored and
extend-added in local memory, panels and the root's contribution block written
to global memory); a regime-B group (width class `W ≤ 64`) is one
[`factorize_fronts_b!`](@ref) launch (fused assembly, `F11` Cholesky in local
memory, `trsm`, `syrk`/`herk` into the front's contribution block); a regime-C
group is one launch each of [`zero_fronts!`](@ref), [`scatter_A!`](@ref) and
[`extend_add!`](@ref) over its fronts, then per front [`potrf_info!`](@ref),
[`trsm!`](@ref) and [`herk!`](@ref)/[`syrk!`](@ref) (into the workspace, then
[`pack_add!`](@ref) into its packed contribution block) through the dense
interface (`impl` is passed on: `:auto`,
`:generic`, `:vendor`, `:ka`, resolved once per call; `:auto` means `:ka` when
the analysis used `factorization_alg = "algo1"`, no vendor calls). Then
[`cholesky_stats!`](@ref) and one read of the reduced status.

Returns `info = 0` on success, else the original column of the smallest
non-positive pivot of the factor (as [`ref_factorize!`](@ref)); fronts above a
failed one hold garbage. `opts.user_host_interrupt` is polled before every
launch group (a host read): when set, [`InterruptedError`](@ref) is raised and
the factor is incomplete. Structure `"SPD"` (real) or `"HPD"`. Nothing is
allocated on the device; the panels are bitwise reproducible for a fixed
`impl` and backend (regimes A and B are deterministic by construction).
"""
function factorize!(N::Numeric{T}, S::Symbolic, nzval::AbstractVector; impl::Symbol = :auto,
                    opts::Options = Options()) where {T}
    _is_ldlt_structure(S.structure) && return factorize_ldlt!(N, S, nzval; opts)
    _check_numeric(N, S, nzval)
    p = _front_impls(N, S, impl)
    plan = N.plan
    nodes = S.schedule.group_nodes
    flag = opts.user_host_interrupt
    for k in eachindex(plan.sub_first)
        _poll_interrupt(flag)
        a, b = plan.sub_first[k], plan.sub_last[k]
        factorize_subtrees!(N, S, nzval, a, b - a + 1, plan.sub_local[k])
    end
    for k in eachindex(plan.group_first)
        _poll_interrupt(flag)
        a, b = plan.group_first[k], plan.group_last[k]
        W = plan.group_width[k]
        if W > 0
            factorize_fronts_b!(N, S, nzval, a, b - a + 1, plan.group_maxchild[k], W)
        else
            zero_fronts!(N, S, a, b - a + 1)
            scatter_A!(N, S, nzval, a, b - a + 1)
            extend_add!(N, S, a, b - a + 1, plan.group_maxchild[k])
            for q in a:b
                _factor_front_c!(N, S, nodes[q], p)
            end
        end
    end
    cholesky_stats!(N, S)
    copyto!(plan.info_host, 1, N.info, nsupernodes(S) + 1, 1)  # the phase's only host synchronization
    k = Int(plan.info_host[1])
    return k == 0 ? 0 : S.partition.perm[k]
end

factorize!(N::Numeric, S::Symbolic, A::CSR; impl::Symbol = :auto, opts::Options = Options()) =
    factorize!(N, S, vec(A.nzval); impl, opts)
