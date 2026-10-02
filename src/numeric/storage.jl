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
of children of one of its fronts (the trip count of the owner-pull extend-add);
`info_host` is the host staging buffer of the one `info` read per phase.
"""
struct NumericPlan
    step_first::Vector{Int}
    step_last::Vector{Int}
    step_maxchild::Vector{Int}
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
    for grp in sc.groups
        grp.regime == REGIME_A && continue
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
    return NumericPlan(first, last, maxchild, zeros(Int32, 1))
end

"""
    Numeric{T, VT <: AbstractVector{T}, VS <: AbstractVector{Int64}, VI <: AbstractVector{Int32}}

Numeric storage of a factorization (PLAN §3.2), laid out by the
[`Layout`](@ref) of a [`Symbolic`](@ref):

* `factor` (`layout.factor_len` entries): the panel of supernode `s` is the
  column-major `f×w` block `factor[panel_ptr[s]:(panel_ptr[s+1]-1)]` (leading
  dimension `f`, rows `snrows(s)`, the upper triangle of its diagonal block is
  unused and kept zero);
* `d` (`layout.d_len = 2n` entries): D of LDLᵀ/LDLᴴ (unused by Cholesky);
* `stack` (`layout.stack_len` entries): the update stack of the device path
  (contribution block of `s` at `cb_ptr[s]`, `m×m`, leading dimension `m`);
* `stats` (`FRONT_STATS_FIELDS × ns` `Int64`, column `s` = front `s`):
  `(npos, nneg, nzero, nperturbed, n2x2, info)`, `info` = the local column of the
  first failed pivot of the front (`0` = none);
* `info` (`ns + 1` `Int32`): device status of the numeric phase, the
  `potrf` info of front `s` at `s`, the reduced result (smallest failed factor
  column, `0` = none) at `ns + 1`;
* `plan`: the host [`NumericPlan`](@ref).
"""
struct Numeric{T, VT <: AbstractVector{T}, VS <: AbstractVector{Int64}, VI <: AbstractVector{Int32}}
    factor::VT
    d::VT
    stack::VT
    stats::VS
    info::VI
    plan::NumericPlan
end

Base.eltype(::Numeric{T}) where {T} = T

Base.show(io::IO, N::Numeric{T, VT}) where {T, VT} =
    print(io, "Numeric{", T, ", ", nameof(VT), "}(factor ", length(N.factor), ", D ", length(N.d), ", stack ",
          length(N.stack), " entries)")

"""
    allocate_numeric(symbolic, T, backend = CPU()) -> Numeric{T}

Allocate (zero-filled) the factor panels, D, update stack, per-front
statistics and status vector of `symbolic`'s [`Layout`](@ref) for element type
`T` on the KernelAbstractions `backend`, and build its [`NumericPlan`](@ref).
This is the only allocation of the numeric phase.
"""
function allocate_numeric(S::Symbolic, ::Type{T}, backend::KernelAbstractions.Backend = KernelAbstractions.CPU()) where {T}
    L = S.layout
    S.n < typemax(Int32) || throw(InvalidValueError("n = $(S.n) does not fit the Int32 status vector"))
    factor = KernelAbstractions.zeros(backend, T, L.factor_len)
    d = KernelAbstractions.zeros(backend, T, L.d_len)
    stack = KernelAbstractions.zeros(backend, T, L.stack_len)
    stats = KernelAbstractions.zeros(backend, Int64, FRONT_STATS_FIELDS * nsupernodes(S))
    info = KernelAbstractions.zeros(backend, Int32, nsupernodes(S) + 1)
    return Numeric{T, typeof(factor), typeof(stats), typeof(info)}(factor, d, stack, stats, info, NumericPlan(S))
end
