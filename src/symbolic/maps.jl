# Symbolic step 7 (PLAN §2.3): the device maps of the numeric and solve
# phases, and the `Symbolic` object that holds the whole analysis.
#
# Every map is an integer vector, built on the host as `Vector{Int}` and moved
# to the device once by `adapt(backend, symbolic, INT)` with an overflow check.
# Offsets into the factor buffer and the update stack follow `Layout`; row and
# column indices are in the supernodal numbering of `SupernodePartition` (column
# `k` is original column `perm[k]`).

"""
    DEVICE_MAPS

Names of the index vectors of a [`Symbolic`](@ref) that live on the device
(`VI`), in the order [`adapt`](@ref) moves them.
"""
const DEVICE_MAPS = (:perm, :iperm, :super_ptr, :snparent, :rowptr, :rowval, :front_ptr, :front_nrows,
                     :front_ncols, :cb_ptr, :child_ptr, :child_list, :relind_ptr, :relind, :amap, :amap_ptr,
                     :amap_src, :subtree_ptr, :subtree_nodes, :local_front, :local_cb, :group_ptr, :group_nodes)

"""
    Symbolic{INT, VI <: AbstractVector{INT}}

Result of the analysis of one sparsity pattern (PLAN §2.3, §3.2): host tree
data, schedule and layout, plus the device maps as `VI` vectors.

Host fields: `n`, `nnz` (stored entries of the user's CSR), `structure`,
`view`, `index`, `elsize` (`sizeof(T)` the schedule was built for),
`partition::SupernodePartition`, `schedule::Schedule`, `layout::Layout`.

Device maps (`DEVICE_MAPS`; supernodal numbering, 1-based, element offsets):

* `perm`, `iperm` (length `n`): the fill-reducing permutation of the factor and
  its inverse (column `k` of `L` is original column `perm[k]`);
* `super_ptr`, `snparent`: supernode column ranges and parents (`0` = root);
* solve gather lists `rowptr`, `rowval`: the (permuted) rows of supernode `s`
  are `rowval[rowptr[s]:(rowptr[s+1]-1)]`, its own columns first;
* per-front descriptors: `front_ptr` (= `layout.panel_ptr`, length `ns + 1`),
  `front_nrows` (`f`), `front_ncols` (`w`), `cb_ptr` (update-stack offset of the
  contribution block, `0` if it does not go through the stack), `snparent`;
* `child_ptr`, `child_list`: the children of `s` in the fixed order their
  contribution blocks are pulled (increasing id), for owner-pull extend-add;
* `relind_ptr`, `relind`: for a child `c` with `m = f_c - w_c > 0`,
  `relind[relind_ptr[c]:(relind_ptr[c+1]-1)]` are the positions (1-based) of
  its rows `w_c+1:f_c` among the rows of its parent;
* `amap` (length `nnz`): for every entry `p` of the user's `nzval`, the
  factor-buffer offset it is added to; `0` when the view ignores the entry, and
  `-offset` when the conjugate is added (a `"H"`/`"HPD"` entry that lands in the
  upper triangle of `P A Pᵀ` and is mirrored). Duplicated entries map to the same
  offset and are summed;
* `amap_ptr`, `amap_src`: the same map grouped by supernode for owner-pull
  assembly: supernode `s` receives the entries `amap_src[amap_ptr[s]:(amap_ptr[s+1]-1)]`
  (positions in `nzval`, sorted by destination, then by position);
* `subtree_ptr`, `subtree_nodes`: regime-A subtree descriptors (the supernodes of
  subtree `t` in processing order, see [`Schedule`](@ref));
* `local_front`, `local_cb`: local-memory offsets of the packed front and of the
  contribution block of a regime-A supernode (see [`Layout`](@ref));
* `group_ptr`, `group_nodes`: the launch groups of the schedule (group `g`
  covers `group_nodes[group_ptr[g]:(group_ptr[g+1]-1)]`: subtree ids for regime
  A, supernode ids otherwise).
"""
struct Symbolic{INT, VI <: AbstractVector{INT}}
    n::Int
    nnz::Int
    structure::Structure
    view::MatrixView
    index::IndexBase
    elsize::Int
    partition::SupernodePartition
    schedule::Schedule
    layout::Layout
    perm::VI
    iperm::VI
    super_ptr::VI
    snparent::VI
    rowptr::VI
    rowval::VI
    front_ptr::VI
    front_nrows::VI
    front_ncols::VI
    cb_ptr::VI
    child_ptr::VI
    child_list::VI
    relind_ptr::VI
    relind::VI
    amap::VI
    amap_ptr::VI
    amap_src::VI
    subtree_ptr::VI
    subtree_nodes::VI
    local_front::VI
    local_cb::VI
    group_ptr::VI
    group_nodes::VI
end

nsupernodes(S::Symbolic) = nsupernodes(S.partition)

Base.show(io::IO, S::Symbolic{INT, VI}) where {INT, VI} =
    print(io, "Symbolic{", INT, ", ", nameof(VI), "}(n = ", S.n, ", ", nsupernodes(S), " supernodes, ",
          nlaunches(S.schedule), " launches)")

"""
    relative_indices(sp::SupernodePartition) -> (relind_ptr, relind)

Extend-add maps: for every supernode `c` with a parent `p` the positions of
`snrows(sp, c)[w_c+1:end]` inside `snrows(sp, p)` (they are a subset, both
sorted), concatenated; empty for roots and fronts without a contribution block.
"""
function relative_indices(sp::SupernodePartition)
    ns = nsupernodes(sp)
    relind_ptr = Vector{Int}(undef, ns + 1)
    relind_ptr[1] = 1
    relind = Int[]
    for c in 1:ns
        p = sp.snparent[c]
        if p != 0
            prow = snrows(sp, p)
            k = 1
            for i in view(snrows(sp, c), (snwidth(sp, c) + 1):(sp.rowptr[c + 1] - sp.rowptr[c]))
                while k <= length(prow) && prow[k] < i
                    k += 1
                end
                (k <= length(prow) && prow[k] == i) ||
                    throw(InvalidValueError("row $i of supernode $c is missing from its parent $p"))
                push!(relind, k)
            end
        end
        relind_ptr[c + 1] = length(relind) + 1
    end
    return relind_ptr, relind
end

"""
    assembly_map(sp, layout, rowptr, colval, n, structure; view = 'F', index = 'O') -> amap

The `amap` of [`Symbolic`](@ref): for every stored entry `p` of the user's CSR
pattern (`rowptr`, `colval`, base `index`), its offset in the factor buffer of
`layout`, `-offset` when its conjugate is stored instead, `0` when `view` ignores
it. Entry `(r, c)` lands at `(i, j) = (iperm[r], iperm[c])` of `P A Pᵀ`, or at
`(j, i)` when `i < j`. Only symmetric structures are supported (`"G"` needs
U panels, T19).
"""
function assembly_map(sp::SupernodePartition, layout::Layout, rowptr::AbstractVector{<:Integer},
                      colval::AbstractVector{<:Integer}, n::Integer, structure; view = VIEW_FULL, index = INDEX_ONE)
    s_ = _structure(structure)
    v = _matrix_view(view)
    s_ == STRUCTURE_GENERAL &&
        throw(NotSupportedError("assembly maps for structure \"G\" (LU panels) are not implemented yet (T19)"))
    n == sp.n || throw(InvalidValueError("matrix size $n does not match the analysis size $(sp.n)"))
    rp, cv = _host_pattern(rowptr, colval, n, index)
    keep = _entry_filter(s_, v)
    herm = _is_hermitian(s_)
    amap = zeros(Int, length(cv))
    for r in 1:n, p in rp[r]:(rp[r + 1] - 1)
        c = cv[p]
        keep(r, c) || continue
        i, j = sp.iperm[r], sp.iperm[c]
        flip = i < j
        flip && ((i, j) = (j, i))
        s = sp.col2sn[j]
        rows = snrows(sp, s)
        pos = searchsortedfirst(rows, i)
        (pos <= length(rows) && rows[pos] == i) ||
            throw(InvalidValueError("entry ($r, $c) is outside the symbolic factor (pattern changed?)"))
        f = length(rows)
        off = layout.panel_ptr[s] + (j - sp.super_ptr[s]) * f + pos - 1
        amap[p] = flip && herm ? -off : off
    end
    return amap
end

# owner-pull grouping of amap by supernode
function _group_amap(amap::Vector{Int}, layout::Layout, ns::Int)
    owner(off) = searchsortedlast(layout.panel_ptr, off)
    order = [p for p in eachindex(amap) if amap[p] != 0]
    sort!(order; by = p -> (abs(amap[p]), p))
    amap_ptr = zeros(Int, ns + 1)
    amap_ptr[1] = 1
    for p in order
        amap_ptr[owner(abs(amap[p])) + 1] += 1
    end
    cumsum!(amap_ptr, amap_ptr)
    return amap_ptr, order     # sorted by destination, hence by owner
end

"""
    Symbolic(sp::SupernodePartition, schedule, layout, rowptr, colval, n, structure;
             view = 'F', index = 'O') -> Symbolic{Int, Vector{Int}}

Assemble the host [`Symbolic`](@ref) from the supernodes, schedule and layout
of a pattern and its user CSR arrays (only the pattern is read).
"""
function Symbolic(sp::SupernodePartition, sc::Schedule, layout::Layout, rowptr::AbstractVector{<:Integer},
                  colval::AbstractVector{<:Integer}, n::Integer, structure; view = VIEW_FULL, index = INDEX_ONE)
    ns = nsupernodes(sp)
    amap = assembly_map(sp, layout, rowptr, colval, n, structure; view, index)
    amap_ptr, amap_src = _group_amap(amap, layout, ns)
    child_ptr = zeros(Int, ns + 1)
    child_ptr[1] = 1
    for s in 1:ns
        p = sp.snparent[s]
        p != 0 && (child_ptr[p + 1] += 1)
    end
    cumsum!(child_ptr, child_ptr)
    child_list = Vector{Int}(undef, child_ptr[end] - 1)
    next = child_ptr[1:ns]
    for s in 1:ns                                         # increasing child id per parent
        p = sp.snparent[s]
        p == 0 && continue
        child_list[next[p]] = s
        next[p] += 1
    end
    relind_ptr, relind = relative_indices(sp)
    group_ptr = Vector{Int}(undef, length(sc.groups) + 1)
    group_ptr[1] = 1
    for (g, grp) in enumerate(sc.groups)
        group_ptr[g + 1] = grp.last + 1
    end
    return Symbolic{Int, Vector{Int}}(Int(n), length(amap), _structure(structure), _matrix_view(view),
                                      _index_base(index), sc.elsize, sp, sc, layout,
                                      copy(sp.perm), copy(sp.iperm), copy(sp.super_ptr), copy(sp.snparent),
                                      copy(sp.rowptr), copy(sp.rowval), copy(layout.panel_ptr), copy(sc.rows),
                                      copy(sc.width), copy(layout.cb_ptr), child_ptr, child_list, relind_ptr,
                                      relind, amap, amap_ptr, amap_src, copy(sc.subtree_ptr),
                                      copy(sc.subtree_nodes), copy(layout.local_front), copy(layout.local_cb),
                                      group_ptr, copy(sc.group_nodes))
end

"""
    symbolic_analysis(A::CSR, structure, view = 'F'; opts = Options(), T = eltype(A)) -> Symbolic{Int, Vector{Int}}

The whole host analysis of PLAN §2.3: [`SymmetricPattern`](@ref),
[`compute_ordering`](@ref) (with the 2×2 pivot pair candidates or fixed pairs
of [`analysis_pairs`](@ref) for `"S"`/`"H"`), [`supernode_partition`](@ref) on
[`factor_pattern`](@ref), [`build_schedule`](@ref) for element type `T`,
[`build_layout`](@ref) and the maps of [`Symbolic`](@ref). `A.rowptr`/`A.colval`
are copied to the host once; `A.nzval` too when pairs are looked for
([`pairs_enabled`](@ref)), and only then.
"""
function symbolic_analysis(A::CSR, structure, view = VIEW_FULL; opts::Options = Options(), T::Type = eltype(A))
    A.nrows == A.ncols || throw(InvalidValueError("the matrix must be square, got $(A.nrows) × $(A.ncols)"))
    rowptr = Array(A.rowptr)
    colval = Array(A.colval)
    P = SymmetricPattern(rowptr, colval, A.nrows, structure; view, index = A.index)
    pp = analysis_pairs(P, rowptr, colval, A.nzval, A.nrows, structure, opts; view, index = A.index)
    ord = compute_ordering(P, opts; T, pp.pairs, pp.candidates)
    sp = supernode_partition(factor_pattern(P, ord), ord.perm, opts)
    sc = build_schedule(sp, opts, T)
    layout = build_layout(sp, sc)
    return Symbolic(sp, sc, layout, rowptr, colval, A.nrows, structure; view, index = A.index)
end

function _index_vector(backend::KernelAbstractions.Backend, x::AbstractVector{<:Integer}, ::Type{INT},
                       name::Symbol) where {INT <: Integer}
    hi = maximum(abs, x; init = 0)
    hi <= typemax(INT) ||
        throw(InvalidValueError("symbolic map $name has an entry $hi that does not fit in $INT; use Int64 indices"))
    y = KernelAbstractions.allocate(backend, INT, length(x))
    isempty(x) || copyto!(y, Vector{INT}(x))
    return y
end

"""
    adapt(backend, symbolic::Symbolic, INT) -> Symbolic{INT}

Move the device maps ([`DEVICE_MAPS`](@ref)) of `symbolic` to the
KernelAbstractions `backend` as vectors of `INT` (`Int32` or `Int64`); the host
fields are shared. Raises [`InvalidValueError`](@ref) when an index or offset
does not fit in `INT`.
"""
function Adapt.adapt(backend::KernelAbstractions.Backend, S::Symbolic, ::Type{INT}) where {INT <: Signed}
    maps = map(f -> _index_vector(backend, Array(getfield(S, f)), INT, f), DEVICE_MAPS)
    VI = typeof(first(maps))
    return Symbolic{INT, VI}(S.n, S.nnz, S.structure, S.view, S.index, S.elsize, S.partition, S.schedule,
                             S.layout, maps...)
end

"""
    device_map_bytes(symbolic, INT = eltype of its maps) -> Int64

Bytes of the device maps ([`DEVICE_MAPS`](@ref)) as `INT` vectors.
"""
device_map_bytes(S::Symbolic{INT}, ::Type{J} = INT) where {INT, J} =
    Int64(sum(f -> length(getfield(S, f)), DEVICE_MAPS) * sizeof(J))

# per-front statistics of the numeric phase: (npos, nneg, nzero, nperturbed, n2x2, info) as Int64
const FRONT_STATS_FIELDS = 6

"""
    memory_estimates(symbolic, T[, INT]) -> Vector{Int64}

The `"memory_estimates"` data parameter (16 entries, bytes) for factors of
element type `T` and device indices `INT` (default: the index type of
`symbolic`'s maps, `Int` for a host analysis):

| slot | content |
| --- | --- |
| 1 | permanent device memory: factor panels + D + device maps + per-front statistics + regime-C workspace |
| 2 | peak device memory: slot 1 + update stack high-water mark |
| 3 | permanent host memory (the host analysis data) |
| 4 | peak host memory (= slot 3) |
| 5 | minimum device memory of the hybrid memory mode (= slot 2 until M12) |
| 6 | maximum host memory of the hybrid memory mode (0 until M12) |
| 7 | factor panels (`Layout.factor_len` entries of `T`) |
| 8 | D (`2n` entries of `T`) |
| 9 | update stack (`Layout.stack_len` entries of `T`) |
| 10 | device maps (`INT`) |
| 11 | per-front statistics (6 `Int64` per supernode) and their totals (6 `Int64`), status (`ns + 1` `Int32`), pivot order (`n` `Int32`), pivot kinds and sign requests (`n` `Int8` each), `aux` (one `T`) |
| 12 | largest regime-A local memory in use (per workgroup, [`subtree_local_bytes`](@ref) of its class) |
| 13–16 | 0 (reserved) |

Slots 1–6 follow cuDSS's `CUDSS_DATA_MEMORY_ESTIMATES`.
"""
function memory_estimates(S::Symbolic{INT0}, ::Type{T}, ::Type{INT} = INT0) where {INT0, T, INT}
    est = zeros(Int64, 16)
    L = S.layout
    sc = S.schedule
    est[7] = Int64(L.factor_len) * sizeof(T)
    est[8] = Int64(L.d_len) * sizeof(T)
    est[9] = Int64(L.stack_len) * sizeof(T)
    est[10] = device_map_bytes(S, INT)
    est[11] = Int64(nsupernodes(S) + 1) * FRONT_STATS_FIELDS * sizeof(Int64) +
              Int64(nsupernodes(S) + 1) * sizeof(Int32) + Int64(S.n) * (sizeof(Int32) + 2 * sizeof(Int8)) + sizeof(T)
    est[12] = Int64(maximum((subtree_local_bytes(sc.budgets[c]) for c in sc.subtree_class); init = 0))
    est[1] = est[7] + est[8] + est[10] + est[11] + Int64(L.work_len) * sizeof(T)
    est[2] = est[1] + est[9]
    est[3] = Int64(Base.summarysize(S.partition) + Base.summarysize(S.schedule) + Base.summarysize(S.layout))
    est[4] = est[3]
    est[5] = est[2]
    return est
end
