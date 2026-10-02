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
buffer of the one `info` read per phase (one entry per batch member).
Batch state (PLAN §3.5): `members_host[1:nact[]]` are the active members of the
last phase (mirrored in `numeric.members`), `runs` the starts of their runs of
consecutive members ([`member_runs!`](@ref), the strided batches of regime C).
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
    members_host::Vector{Int32}
    nact::Base.RefValue{Int}
    runs::Vector{Int}
end

function NumericPlan(S, nb::Integer = 1)
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
    return NumericPlan(first, last, maxchild, gfirst, glast, gmaxchild, gwidth, sfirst, slast, slocal, zeros(Int32, nb),
                       Int32.(1:nb), Ref(Int(nb)), [1, nb + 1])
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
* `totals` (`FRONT_STATS_FIELDS` `Int64`): the sums of `stats` over the fronts
  ([`reduce_stats!`](@ref); `info` field: number of failed fronts);
* `psign` (`n` `Int8`, LDLᵀ/LDLᴴ): the `pivot_sign` request per original row
  (`0` = none), copied from the options at every factorization;
* `aux` (1 entry): `max |aᵢⱼ|` of the values of the last LDLᵀ/LDLᴴ
  factorization with `pivot_epsilon_alg = "algo1"` (scaled perturbation);
* `info` (`ns + 1` `Int32`): device status of the numeric phase, the
  `potrf` info of front `s` at `s`, the reduced result (smallest failed factor
  column, `0` = none) at `ns + 1`;
* `plan`: the host [`NumericPlan`](@ref);
* `members` (`nbatch` `Int32`): the active batch members of the last phase
  (first `plan.nact[]` entries, see [`BatchMap`](@ref));
* `nbatch`: the number of matrices of the uniform batch (PLAN §3.5).

A uniform batch of `nbatch` members multiplies every length above by
`nbatch`, in the layouts of `src/numeric/batch.jl`: interleaved per front for
`factor` ([`panel_offset`](@ref)), interleaved per entry for `info`, member
after member for the others (the regime-C workspace holds the `m×m` blocks of
the members of one front).
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
    totals::VS
    psign::VK
    aux::VT
    plan::NumericPlan
    members::VI
    nbatch::Int
end

# `pivot_kind` codes of LDLᵀ/LDLᴴ factor columns
const PIVOT_KIND_1X1 = Int8(1)          # 1×1 pivot
const PIVOT_KIND_2X2_FIRST = Int8(2)    # first column of a 2×2 block
const PIVOT_KIND_2X2_SECOND = Int8(3)   # second column of a 2×2 block
const PIVOT_KIND_PERTURBED = Int8(4)    # 1×1 pivot replaced by ±ε

Base.eltype(::Numeric{T}) where {T} = T

Base.show(io::IO, N::Numeric{T, VT}) where {T, VT} =
    print(io, "Numeric{", T, ", ", nameof(VT), "}(factor ", length(N.factor), ", D ", length(N.d), ", stack ",
          length(N.stack), " entries", N.nbatch > 1 ? ", $(N.nbatch) batch members)" : ")")

"""
    allocate_numeric(symbolic, T, backend = CPU(); nbatch = 1) -> Numeric{T}

Allocate (zero-filled) the factor panels, D, update stack, regime-C workspace, per-front
statistics and their totals, status vector, pivot order, pivot kinds, pivot sign requests and `aux` of `symbolic`'s [`Layout`](@ref) for element type
`T` on the KernelAbstractions `backend`, for a uniform batch of `nbatch`
matrices (every length times `nbatch`, all members active), and build its [`NumericPlan`](@ref).
This is the only allocation of the numeric phase.
"""
function allocate_numeric(S::Symbolic{INT}, ::Type{T}, backend::KernelAbstractions.Backend = KernelAbstractions.CPU();
                          nbatch::Integer = 1) where {INT, T}
    L = S.layout
    nb = Int(nbatch)
    nb >= 1 || throw(InvalidValueError("nbatch must be ≥ 1, got $nb"))
    S.n < typemax(Int32) || throw(InvalidValueError("n = $(S.n) does not fit the Int32 status vector"))
    nb * L.factor_len < typemax(INT) && nb * (nsupernodes(S) + 1) < typemax(Int32) ||
        throw(InvalidValueError("a batch of $nb factors of $(L.factor_len) entries does not fit the index type $INT"))
    factor = KernelAbstractions.zeros(backend, T, nb * L.factor_len)
    d = KernelAbstractions.zeros(backend, T, nb * L.d_len)
    stack = KernelAbstractions.zeros(backend, T, nb * L.stack_len)
    work = KernelAbstractions.zeros(backend, T, nb * L.work_len)
    stats = KernelAbstractions.zeros(backend, Int64, nb * FRONT_STATS_FIELDS * nsupernodes(S))
    info = KernelAbstractions.zeros(backend, Int32, nb * (nsupernodes(S) + 1))
    piv = KernelAbstractions.zeros(backend, Int32, nb * S.n)
    pivot_kind = KernelAbstractions.zeros(backend, Int8, nb * S.n)
    totals = KernelAbstractions.zeros(backend, Int64, nb * FRONT_STATS_FIELDS)
    psign = KernelAbstractions.zeros(backend, Int8, S.n)
    aux = KernelAbstractions.zeros(backend, T, nb)
    members = KernelAbstractions.allocate(backend, Int32, nb)
    copyto!(members, Int32.(1:nb))
    return Numeric{T, typeof(factor), typeof(stats), typeof(info), typeof(pivot_kind)}(factor, d, stack, work, stats,
                                                                                         info, piv, pivot_kind, totals,
                                                                                         psign, aux, NumericPlan(S, nb),
                                                                                         members, nb)
end

"""
    set_members!(numeric, members) -> numeric

Make `members` (1-based, increasing, a subset of `1:numeric.nbatch`) the
active batch members of the next phases: copied to `numeric.members` (a
host-to-device copy, only when they change) and split into runs of
consecutive members. Asynchronous.
"""
function set_members!(N::Numeric, members::AbstractVector{<:Integer})
    plan = N.plan
    nact = length(members)
    1 <= nact <= N.nbatch || throw(InvalidValueError("$nact active members for a batch of $(N.nbatch)"))
    for j in 1:nact
        1 <= members[j] <= N.nbatch && (j == 1 || members[j] > members[j - 1]) ||
            throw(InvalidValueError("active members must be increasing members of 1:$(N.nbatch), got $members"))
    end
    nact == plan.nact[] && all(j -> plan.members_host[j] == members[j], 1:nact) && return N
    for j in 1:nact
        plan.members_host[j] = Int32(members[j])
    end
    plan.nact[] = nact
    member_runs!(plan.runs, plan.members_host, nact)
    copyto!(N.members, 1, plan.members_host, 1, nact)
    return N
end

"""
    batch_map(numeric; first = 1, nrhs = 1) -> BatchMap

The [`BatchMap`](@ref) of the active members of `numeric` for a launch.
"""
batch_map(N::Numeric; first::Integer = 1, nrhs::Integer = 1) =
    BatchMap(N.members, N.plan.nact[], N.nbatch; first, nrhs)
