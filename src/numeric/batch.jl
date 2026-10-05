# Uniform batch (PLAN §1.6, §3.5): `nbatch` matrices with the pattern of one
# analysis. Every device array of the numeric and solve phases gets a batch
# stride, and every kernel takes the batch index as one more grid dimension,
# flattened into its 1-D ndrange (2-D ndranges allocate on every launch of the
# KA CPU backend): the workgroups (or work items) of a launch over `count`
# fronts become `count × nact`, member slot fastest, for the `nact` active
# members of the batch (`ubatch_index`/`ubatch_mask`).
#
# Storage of member `k` (1-based) of an `nb`-member batch:
#
# * factor panels: interleaved per front, the `nb` panels of supernode `s` one
#   after the other, `f×w` each, starting at `nb (panel_ptr[s] - 1) + 1`, so a
#   regime-C front of the whole batch is one `f×w×nb` strided batch for the
#   batched dense calls ([`member_panels`](@ref));
# * `info`: interleaved, entry `s` of member `k` at `(s - 1) nb + k`, so the
#   statuses of one front are contiguous (batched `potrf`);
# * everything else (`d`, `piv`, `pivot_kind`, the update stack, the regime-C
#   workspace per launch, `stats`, `totals`, `aux`, the values `nzval`, the
#   right-hand sides): member after member (`(k - 1) len + 1 : k len`).
#
# With `nb = 1` both layouts are the single-matrix layout.

"""
    BatchMap{V <: Union{Nothing, AbstractVector{Int32}}}

Batch part of a kernel launch (an argument of every numeric and solve
kernel): `members[1:nact]` are the active batch members (1-based, increasing;
`members = nothing`: a single matrix, member 1),
`nbatch` the batch size, `first` the launch's first entry in its node list
(the former `first` argument of the kernels) and `nrhs` the right-hand sides
per member of a solve (1 in the numeric phase). Adapted to the device with its
`members` vector.
"""
struct BatchMap{V <: Union{Nothing, AbstractVector{Int32}}}
    members::V
    nact::Int
    nbatch::Int
    first::Int
    nrhs::Int
end

Adapt.@adapt_structure BatchMap

BatchMap(members::AbstractVector{Int32}, nact::Integer, nbatch::Integer; first::Integer = 1, nrhs::Integer = 1) =
    BatchMap{typeof(members)}(members, Int(nact), Int(nbatch), Int(first), Int(nrhs))

BatchMap(::Nothing, nact::Integer, nbatch::Integer; first::Integer = 1, nrhs::Integer = 1) =
    BatchMap{Nothing}(nothing, Int(nact), Int(nbatch), Int(first), Int(nrhs))

# a single matrix (launches without a `Numeric`: tests and benchmarks of the kernels)
single_batch(; first::Integer = 1, nrhs::Integer = 1) = BatchMap(nothing, 1, 1; first, nrhs)

# workgroup `G` of a launch over `nact` members: (node index g, member slot j), member slot fastest
@inline _bm_node(bm::BatchMap, G) = (G - 1) ÷ bm.nact + 1
@inline _bm_slot(bm::BatchMap, G) = (G - 1) % bm.nact + 1
@inline _bm_member(bm::BatchMap, j) = @inbounds Int(bm.members[j])
@inline _bm_member(::BatchMap{Nothing}, j) = 1
# member of workgroup `G`
@inline _bm_gmember(bm::BatchMap, G) = _bm_member(bm, _bm_slot(bm, G))
# member of the compact solve column `r` (columns of member slot j: (j - 1) nrhs + 1 : j nrhs)
@inline _bm_cmember(bm::BatchMap, r) = _bm_member(bm, (r - 1) ÷ bm.nrhs + 1)
# column of the user's right-hand side array (all `nbatch` members) holding the compact column `r`
@inline _bm_ucol(bm::BatchMap, r) = (_bm_cmember(bm, r) - 1) * bm.nrhs + (r - 1) % bm.nrhs + 1

# member-major view of member `k` of the `nb` members of `A` (`length(A) ÷ nb` entries each)
@inline function _mview(A, k, nb)
    len = length(A) ÷ nb
    return @inbounds view(A, ((k - 1) * len + 1):(k * len))
end

# interleaved view of member `k`: entry `i` at `(i - 1) nb + k`
@inline _iview(A, k, nb) = @inbounds view(A, k:nb:length(A))

"""
    MemberPanels{V}

The panel pointers of member `k` of an `nb`-member batch in the interleaved
factor layout: `P[s] = nb (front_ptr[s] - 1) + (k - 1)(front_ptr[s+1] - front_ptr[s]) + 1`
(the first entry of the `f×w` panel of supernode `s` of member `k`). Passed
to the kernels in place of `front_ptr` for the factor offsets; `P.ptr` is the
analysis's `front_ptr`. Note that `P[s + 1] - P[s]` is not a panel length.
"""
struct MemberPanels{V}
    ptr::V
    k::Int
    nb::Int
end

@inline function Base.getindex(P::MemberPanels, s::Integer)
    @inbounds a = P.ptr[s]
    @inbounds b = P.ptr[s + 1]
    return oftype(a, P.nb * (a - 1) + (P.k - 1) * (b - a) + 1)
end

@inline member_panels(front_ptr, k, nb) = MemberPanels(front_ptr, Int(k), Int(nb))

"""
    panel_offset(panel_ptr, s, k, nb) -> Int

First factor index of the panel of supernode `s` of batch member `k` in the
interleaved layout (host side, `panel_ptr` of the `Layout`).
"""
panel_offset(panel_ptr::AbstractVector, s::Integer, k::Integer, nb::Integer) =
    nb * (panel_ptr[s] - 1) + (k - 1) * (panel_ptr[s + 1] - panel_ptr[s]) + 1

"""
    active_members(nbatch, ubatch_index, ubatch_mask) -> Vector{Int}

The batch members a phase processes (1-based, increasing): all of them for
`ubatch_index = -1` and no mask, member `ubatch_index + 1` (`ubatch_index` is
0-based, as in cuDSS), or the members whose `ubatch_mask` flag is 1 (the mask
has `nbatch` entries; with both set, their intersection). Raises
[`InvalidValueError`](@ref) for an index or mask that does not fit the batch.
"""
function active_members(nb::Integer, index::Integer, mask)
    (index == -1 || 0 <= index < nb) ||
        throw(InvalidValueError("ubatch_index = $index is outside the batch (0:$(nb - 1), or -1 for all members)"))
    if mask !== nothing
        length(mask) == nb ||
            throw(InvalidValueError("ubatch_mask has $(length(mask)) entries, the batch has $nb members"))
    end
    members = Int[]
    for k in 1:nb
        index == -1 || k == index + 1 || continue
        mask === nothing || mask[k] != 0 || continue
        push!(members, k)
    end
    return members
end

# runs of consecutive members: the start positions in `members` (and length(members) + 1 at the end)
function member_runs!(starts::Vector{Int}, members::AbstractVector{<:Integer}, nact::Integer)
    empty!(starts)
    for j in 1:nact
        (j == 1 || members[j] != members[j - 1] + 1) && push!(starts, j)
    end
    push!(starts, nact + 1)
    return starts
end
