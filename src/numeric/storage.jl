# Numeric storage (PLAN §3.2): the device buffers of a factorization laid out
# by the `Layout` of a `Symbolic`, plus the host launch plan of the level
# driver. Allocated once by `allocate_numeric`; the numeric phase itself
# allocates nothing (PLAN §3.9).

"""
    NumericPlan

Host launch plan of the level driver ([`factorize!`](@ref)), derived from the
[`Schedule`](@ref) once at allocation: for every step `t` of regimes B/C,
`step_first[t]:step_last[t]` is its range in `schedule.group_nodes` (empty
when `step_first[t] > step_last[t]`) and `step_maxchild[t]` the largest number
of children of one of its fronts (the trip count of the owner-pull extend-add).
The launch groups of regimes B/C, in execution order: group `k` covers
`group_first[k]:group_last[k]` of `group_nodes`, its fronts have at most
`group_maxchild[k]` children, and `group_width[k]` is the width class `W` of
its fused regime-B kernel ([`factorize_fronts_b!`](@ref)), or `0` when the
group takes the regime-C path (regime C, or a regime-B bin wider than
`REGIME_B_MAX_WIDTH`). The regime-A launch groups (one per budget class, run
first): group `k` covers the subtree ids `sub_first[k]:sub_last[k]` of
`group_nodes` and its kernel has `sub_local[k]` bytes of local memory
([`subtree_local_bytes`](@ref) of its budget). `info_host` is the host staging
buffer of the one `info` read per phase.
"""
struct NumericPlan
    step_first::Vector{Int}
    step_last::Vector{Int}
    step_maxchild::Vector{Int}
    group_first::Vector{Int}
    group_last::Vector{Int}
    group_maxchild::Vector{Int}
    group_width::Vector{Int}
    sub_first::Vector{Int}
    sub_last::Vector{Int}
    sub_local::Vector{Int}
    info_host::Vector{Int32}
end

function NumericPlan(S)
    sp, sc = S.partition, S.schedule
    ns = nsupernodes(sp)
    nchild = zeros(Int, ns)
    for s in 1:ns
        p = sp.snparent[s]
        p != 0 && (nchild[p] += 1)
    end
    first = ones(Int, sc.nsteps)
    last = zeros(Int, sc.nsteps)
    maxchild = zeros(Int, sc.nsteps)
    gfirst, glast, gmaxchild, gwidth = Int[], Int[], Int[], Int[]
    sfirst, slast, slocal = Int[], Int[], Int[]
    nf = length(sc.fclasses)
    for grp in sc.groups
        if grp.regime == REGIME_A
            isempty(gfirst) || throw(InvalidValueError("schedule: regime-A groups must come first"))
            push!(sfirst, grp.first)
            push!(slast, grp.last)
            push!(slocal, subtree_local_bytes(sc.budgets[grp.class]))
            continue
        end
        W = grp.regime == REGIME_B ? sc.wclasses[(grp.class - 1) ÷ nf + 1] : 0
        push!(gfirst, grp.first)
        push!(glast, grp.last)
        push!(gmaxchild, maximum(q -> nchild[sc.group_nodes[q]], grp.first:grp.last; init = 0))
        push!(gwidth, W <= REGIME_B_MAX_WIDTH ? W : 0)
        t = grp.step
        if last[t] < first[t]
            first[t] = grp.first
        elseif grp.first != last[t] + 1
            throw(InvalidValueError("schedule: the launch groups of step $t are not contiguous"))
        end
        last[t] = grp.last
        for q in grp.first:grp.last
            maxchild[t] = max(maxchild[t], nchild[sc.group_nodes[q]])
        end
    end
    return NumericPlan(first, last, maxchild, gfirst, glast, gmaxchild, gwidth, sfirst, slast, slocal, zeros(Int32, 1))
end

"""
    Numeric{T, VT <: AbstractVector{T}, VS <: AbstractVector{Int64}, VI <: AbstractVector{Int32},
            VK <: AbstractVector{Int8}}

Numeric storage of a factorization (PLAN §3.2), laid out by the
[`Layout`](@ref) of a [`Symbolic`](@ref):

* `factor` (`layout.factor_len` entries): the panel of supernode `s` is the
  column-major `f×w` block `factor[panel_ptr[s]:(panel_ptr[s+1]-1)]` (leading
  dimension `f`, rows `snrows(s)`, the upper triangle of its diagonal block is
  unused and kept zero);
* `d` (`layout.d_len = 2n` entries): D of LDLᵀ/LDLᴴ (unused by Cholesky):
  `d[k]` is the diagonal of D at factor column `k`, `d[n + k]` the subdiagonal
  entry `D[k+1, k]` of a 2×2 block starting at `k` (zero otherwise);
* `piv` (`n` `Int32`, LDLᵀ/LDLᴴ only): the local pivot order, `piv[k]` is the
  column of `P A Pᵀ` (supernodal numbering before pivoting) eliminated at factor
  column `k`; pivoting stays inside a supernode, so `piv` permutes each
  `sncols(s)` (see [`ref_ldlt!`](@ref));
* `pivot_kind` (`n` `Int8`, LDLᵀ/LDLᴴ only): `PIVOT_KIND_1X1`,
  `PIVOT_KIND_PERTURBED`, `PIVOT_KIND_2X2_FIRST`, `PIVOT_KIND_2X2_SECOND`
  (`0`: not factored);
* `stack` (`layout.stack_len` entries): the update stack of the device path
  (contribution block of `s` at `cb_ptr[s]`, packed lower triangle of the
  `m×m` block, `m(m+1)/2` entries);
* `work` (`layout.work_len` entries): the full `m×m` `syrk`/`herk` result of a
  regime-C front before it is packed into its block;
* `stats` (`FRONT_STATS_FIELDS × ns` `Int64`, column `s` = front `s`):
  `(npos, nneg, nzero, nperturbed, n2x2, info)`, `info` = the local column of the
  first failed pivot of the front (`0` = none);
* `info` (`ns + 1` `Int32`): device status of the numeric phase, the
  `potrf` info of front `s` at `s`, the reduced result (smallest failed factor
  column, `0` = none) at `ns + 1`;
* `plan`: the host [`NumericPlan`](@ref).
"""
struct Numeric{T, VT <: AbstractVector{T}, VS <: AbstractVector{Int64}, VI <: AbstractVector{Int32},
               VK <: AbstractVector{Int8}}
    factor::VT
    d::VT
    stack::VT
    work::VT
    stats::VS
    info::VI
    piv::VI
    pivot_kind::VK
    plan::NumericPlan
end

# `pivot_kind` codes of LDLᵀ/LDLᴴ factor columns
const PIVOT_KIND_1X1 = Int8(1)          # 1×1 pivot
const PIVOT_KIND_2X2_FIRST = Int8(2)    # first column of a 2×2 block
const PIVOT_KIND_2X2_SECOND = Int8(3)   # second column of a 2×2 block
const PIVOT_KIND_PERTURBED = Int8(4)    # 1×1 pivot replaced by ±ε

Base.eltype(::Numeric{T}) where {T} = T

Base.show(io::IO, N::Numeric{T, VT}) where {T, VT} =
    print(io, "Numeric{", T, ", ", nameof(VT), "}(factor ", length(N.factor), ", D ", length(N.d), ", stack ",
          length(N.stack), " entries)")

"""
    allocate_numeric(symbolic, T, backend = CPU()) -> Numeric{T}

Allocate (zero-filled) the factor panels, D, update stack, regime-C workspace, per-front
statistics, status vector, pivot order and pivot kinds of `symbolic`'s [`Layout`](@ref) for element type
`T` on the KernelAbstractions `backend`, and build its [`NumericPlan`](@ref).
This is the only allocation of the numeric phase.
"""
function allocate_numeric(S::Symbolic, ::Type{T}, backend::KernelAbstractions.Backend = KernelAbstractions.CPU()) where {T}
    L = S.layout
    S.n < typemax(Int32) || throw(InvalidValueError("n = $(S.n) does not fit the Int32 status vector"))
    factor = KernelAbstractions.zeros(backend, T, L.factor_len)
    d = KernelAbstractions.zeros(backend, T, L.d_len)
    stack = KernelAbstractions.zeros(backend, T, L.stack_len)
    work = KernelAbstractions.zeros(backend, T, L.work_len)
    stats = KernelAbstractions.zeros(backend, Int64, FRONT_STATS_FIELDS * nsupernodes(S))
    info = KernelAbstractions.zeros(backend, Int32, nsupernodes(S) + 1)
    piv = KernelAbstractions.zeros(backend, Int32, S.n)
    pivot_kind = KernelAbstractions.zeros(backend, Int8, S.n)
    return Numeric{T, typeof(factor), typeof(stats), typeof(info), typeof(pivot_kind)}(factor, d, stack, work, stats,
                                                                                         info, piv, pivot_kind,
                                                                                         NumericPlan(S))
end
