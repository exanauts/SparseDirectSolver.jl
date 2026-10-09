# Symbolic steps 3–4 (PLAN §2.3): fundamental supernodes, GPU-tuned relaxed
# amalgamation and the supernodal symbolic factorization.
#
# Numberings. T05 works in the *etree numbering*: column `k` is original column
# `perm[k]`. A supernode partition needs contiguous column ranges, so it
# renumbers the columns once more by a topological order of the column
# elimination tree (`ColumnPartition.order`); topological reorderings of the
# etree are equivalent orderings (same filled graph up to relabelling), so the
# column counts carry over. The *supernodal numbering* used by everything
# downstream is the final permutation `perm[order]`.

"""
    ColumnPartition

A partition of the columns of an elimination tree into supernodes (contiguous
column ranges after a renumbering), without row structure:

* `order::Vector{Int}`: column `k` of the supernodal numbering is column
  `order[k]` of the etree numbering (a topological order of the etree);
* `super_ptr::Vector{Int}`: supernode `s` owns columns `super_ptr[s]:(super_ptr[s+1]-1)`
  of the supernodal numbering;
* `snparent::Vector{Int}`: parent supernode, `0` for a root.

Returned by [`fundamental_supernodes`](@ref) and [`amalgamate`](@ref).
"""
struct ColumnPartition
    order::Vector{Int}
    super_ptr::Vector{Int}
    snparent::Vector{Int}
end

"""
    nsupernodes(cp) -> Int

Number of supernodes of a [`ColumnPartition`](@ref) or [`SupernodePartition`](@ref).
"""
nsupernodes(cp::ColumnPartition) = length(cp.super_ptr) - 1

# number of entries of the lower trapezoid of an f×w panel (diagonal block lower triangle included)
@inline _trapezoid(w::Integer, f::Integer) = w * f - (w * (w - 1)) ÷ 2

# supernode of every column from the range pointers
function _col2sn(super_ptr::AbstractVector{<:Integer})
    ns = length(super_ptr) - 1
    col2sn = Vector{Int}(undef, super_ptr[end] - 1)
    for s in 1:ns, j in super_ptr[s]:(super_ptr[s + 1] - 1)
        col2sn[j] = s
    end
    return col2sn
end

# relabel a parent array: new column k is old column order[k]
function _relabel_parent(parent::AbstractVector{<:Integer}, order::AbstractVector{<:Integer})
    inv = invperm(order)
    return [parent[j] == 0 ? 0 : inv[parent[j]] for j in order]
end

"""
    fundamental_supernodes(parent, post, counts) -> ColumnPartition

Fundamental supernodes of the elimination tree `parent` with column counts
`counts` (both in the etree numbering, from [`etree`](@ref) and
[`colcounts`](@ref)) and its postorder `post` ([`postorder`](@ref)). The columns
are renumbered by `post` (`order == post`); column `k + 1` joins the supernode
of column `k` when `k` is its only child and `counts[k] == counts[k+1] + 1`, so
the row structure of every supernode is exactly that of its first column.
"""
function fundamental_supernodes(parent::AbstractVector{<:Integer}, post::AbstractVector{<:Integer},
                                counts::AbstractVector{<:Integer})
    n = length(parent)
    (length(post) == n && length(counts) == n) ||
        throw(InvalidValueError("parent, post and counts must have the same length"))
    isperm(post) || throw(InvalidValueError("post is not a permutation"))
    par = _relabel_parent(parent, post)
    all(k -> par[k] == 0 || par[k] > k, 1:n) || throw(InvalidValueError("post is not a postorder of parent"))
    cnt = counts[post]
    nchild = zeros(Int, n)
    for k in 1:n
        par[k] != 0 && (nchild[par[k]] += 1)
    end
    super_ptr = Int[1]
    for k in 2:n
        joins = par[k - 1] == k && nchild[k] == 1 && cnt[k - 1] == cnt[k] + 1
        joins || push!(super_ptr, k)
    end
    n > 0 && push!(super_ptr, n + 1)
    col2sn = _col2sn(super_ptr)
    snparent = [par[super_ptr[s + 1] - 1] == 0 ? 0 : col2sn[par[super_ptr[s + 1] - 1]]
                for s in 1:(length(super_ptr) - 1)]
    return ColumnPartition(Vector{Int}(post), super_ptr, snparent)
end

"""
    amalgamate(sn::ColumnPartition, parent, counts, params) -> ColumnPartition

GPU-tuned relaxed amalgamation (Ashcraft–Grimes) of the *exact* supernodes `sn`
(fundamental supernodes, see [`fundamental_supernodes`](@ref)); `parent` and
`counts` are in the etree numbering. `params = (max_width, zero_fraction,
min_width)` is `opts.amalgamation`:

1. the units are the fundamental supernodes themselves. A supernode wider than
   `max_width` stays whole: it is a dense separator that goes to regime C as a
   single front; splitting it into a chain of `max_width` panels multiplied the
   update-stack footprint by the chain length and added tree levels (issue #48);
2. bottom-up over the supernodal tree, the children `c` of each supernode `p`
   are tried in decreasing order of their off-diagonal row count (the order of
   increasing explicit zeros per column, which merging does not change) and
   merged into `p` when the merged width is `≤ max_width`, the explicit zeros
   of the whole factor stay `≤ zero_fraction · nnz(L)`, and either the merged
   panel has at most `zero_fraction` explicit zeros per true nonzero or its
   width is still `≤ min_width` (tiny panels are merged more eagerly).

Merging `c` (width `w_c`, `f_c` rows) into `p` (`f_p` rows) gives a panel of
width `w_c + w_p` and `w_c + f_p` rows, adding `w_c (w_c + f_p - f_c)` explicit
zeros. The result is renumbered so that every merged supernode is contiguous
(postorder of the merged tree, columns in their previous order).
"""
function amalgamate(sn::ColumnPartition, parent::AbstractVector{<:Integer}, counts::AbstractVector{<:Integer},
                    params)
    max_width = Int(params.max_width)
    zero_fraction = Float64(params.zero_fraction)
    min_width = Int(params.min_width)
    max_width >= 1 || throw(InvalidValueError("amalgamate: max_width must be ≥ 1, got $max_width"))
    n = length(sn.order)
    (length(parent) == n && length(counts) == n) ||
        throw(InvalidValueError("parent and counts must have length $n"))
    cnt = counts[sn.order]
    # 1. units: the exact fundamental supernodes; `max_width` only caps merging (step 2)
    uptr = Int[1]
    for s in 1:nsupernodes(sn)
        first, last = sn.super_ptr[s], sn.super_ptr[s + 1] - 1
        for j in first:last
            cnt[j] == cnt[first] - (j - first) ||
                throw(InvalidValueError("amalgamate needs exact supernodes; supernode $s has a padded column $j"))
        end
        push!(uptr, last + 1)
    end
    nu = length(uptr) - 1
    par = _relabel_parent(parent, sn.order)
    col2u = _col2sn(uptr)
    uparent = [par[uptr[u + 1] - 1] == 0 ? 0 : col2u[par[uptr[u + 1] - 1]] for u in 1:nu]
    all(u -> uparent[u] == 0 || uparent[u] > u, 1:nu) ||
        throw(InvalidValueError("sn.order is not a topological order of parent"))
    # 2. bottom-up greedy merging; a unit absorbed into its parent points to it in `into`
    width = [uptr[u + 1] - uptr[u] for u in 1:nu]
    rows = [cnt[uptr[u]] for u in 1:nu]                          # f: rows of the panel
    truennz = [_trapezoid(width[u], rows[u]) for u in 1:nu]      # true nonzeros of the panel
    zeros_ = zeros(Int, nu)                                      # explicit zeros of the panel
    budget = floor(Int, zero_fraction * sum(truennz; init = 0))
    used = 0
    into = zeros(Int, nu)
    children = [Int[] for _ in 1:nu]
    for u in 1:nu
        uparent[u] != 0 && push!(children[uparent[u]], u)
    end
    for p in 1:nu
        kids = children[p]
        # most shared rows first, then narrow first, then by index (deterministic)
        sort!(kids; by = c -> (-(rows[c] - width[c]), width[c], c))
        for c in kids
            w = width[c] + width[p]
            w <= max_width || continue
            added = width[c] * (width[c] + rows[p] - rows[c])
            used + added <= budget || continue
            z = zeros_[c] + zeros_[p] + added
            t = truennz[c] + truennz[p]
            (z <= zero_fraction * t || w <= min_width) || continue
            into[c] = p
            used += added
            width[p] = w
            rows[p] += width[c]
            zeros_[p] = z
            truennz[p] = t
        end
    end
    # 3. merged supernodes = units with into == 0; their parent is the representative of uparent
    rep = collect(1:nu)
    for u in nu:-1:1                                             # parents are merged before children
        into[u] != 0 && (rep[u] = rep[into[u]])
    end
    ids = zeros(Int, nu)
    ns = 0
    for u in 1:nu
        into[u] == 0 && (ns += 1; ids[u] = ns)
    end
    gparent = zeros(Int, ns)
    members = [Int[] for _ in 1:ns]
    for u in 1:nu
        push!(members[ids[rep[u]]], u)                           # increasing unit order
        into[u] == 0 && uparent[u] != 0 && (gparent[ids[u]] = ids[rep[uparent[u]]])
    end
    gpost = postorder(gparent)
    neworder = Int[]
    super_ptr = Int[1]
    sizehint!(neworder, n)
    for g in gpost
        for u in members[g], j in uptr[u]:(uptr[u + 1] - 1)
            push!(neworder, j)
        end
        push!(super_ptr, length(neworder) + 1)
    end
    gpos = invperm(gpost)
    snparent = [gparent[g] == 0 ? 0 : gpos[gparent[g]] for g in gpost]
    return ColumnPartition(sn.order[neworder], super_ptr, snparent)
end

"""
    SupernodePartition

Supernodes of the factor with their row structure (PLAN §2.3 steps 3–4), in the
supernodal numbering: column `k` of the factor is original column `perm[k]`.

* `n`, `perm`, `iperm`: size, final fill-reducing permutation (the T05 ordering
  composed with the supernodal renumbering) and its inverse;
* `parent`, `counts`: column elimination tree and true column counts (diagonal
  included) in the supernodal numbering;
* `super_ptr`: supernode `s` owns columns `super_ptr[s]:(super_ptr[s+1]-1)`;
  `col2sn[j]` is the supernode of column `j`;
* `snparent` (`0` = root) and `snpost`, a postorder of the supernodal tree;
* `rowptr`, `rowval`: the rows of supernode `s` are
  `rowval[rowptr[s]:(rowptr[s+1]-1)]` (see [`snrows`](@ref)), sorted, so its own
  columns come first; the panel is `length(rows) × width`;
* `nnz_stored`: entries of the stored lower-trapezoidal panels, explicit zeros
  of amalgamation included; `nnz_L`: true nonzeros of `L`; `flops`: Cholesky
  flop count on the stored panels (`Σ` over panel columns of rows², real
  arithmetic, the [`cholesky_flops`](@ref) convention);
* `amalgamated`: whether [`amalgamate`](@ref) ran.
"""
struct SupernodePartition
    n::Int
    perm::Vector{Int}
    iperm::Vector{Int}
    parent::Vector{Int}
    counts::Vector{Int}
    super_ptr::Vector{Int}
    col2sn::Vector{Int}
    snparent::Vector{Int}
    snpost::Vector{Int}
    rowptr::Vector{Int}
    rowval::Vector{Int}
    nnz_stored::Int
    nnz_L::Int
    flops::Float64
    amalgamated::Bool
end

nsupernodes(sp::SupernodePartition) = length(sp.super_ptr) - 1

"""
    nsuperpanels(sp::SupernodePartition) -> Int

Number of supernodes after amalgamation (the `"nsuperpanels"` data parameter).
"""
nsuperpanels(sp::SupernodePartition) = nsupernodes(sp)

"""
    sncols(sp, s) -> UnitRange{Int}

Columns of supernode `s` (supernodal numbering).
"""
sncols(sp::SupernodePartition, s::Integer) = sp.super_ptr[s]:(sp.super_ptr[s + 1] - 1)

"""
    snrows(sp, s)

Sorted rows of supernode `s` (a view), its own columns first.
"""
snrows(sp::SupernodePartition, s::Integer) = view(sp.rowval, sp.rowptr[s]:(sp.rowptr[s + 1] - 1))

"""
    snwidth(sp, s) -> Int

Number of columns `w` of supernode `s`; its panel is `length(snrows(sp, s)) × w`.
"""
snwidth(sp::SupernodePartition, s::Integer) = sp.super_ptr[s + 1] - sp.super_ptr[s]

Base.show(io::IO, sp::SupernodePartition) =
    print(io, "SupernodePartition(n = ", sp.n, ", ", nsupernodes(sp), " supernodes, nnz_stored = ",
          sp.nnz_stored, ", nnz_L = ", sp.nnz_L, ", amalgamated = ", sp.amalgamated, ")")

"""
    SupernodePartition(P::SymmetricPattern, perm, parent, counts, cp::ColumnPartition; amalgamated)

Supernodal symbolic factorization: the rows of every supernode of `cp` are its
own columns plus the union of the rows of `A[perm, perm]` in its columns and of
the rows of its children, restricted to rows after its last column. `perm` is
the T05 ordering and `parent`, `counts` the etree and column counts in its
numbering.
"""
function SupernodePartition(P::SymmetricPattern, perm::AbstractVector{<:Integer}, parent::AbstractVector{<:Integer},
                            counts::AbstractVector{<:Integer}, cp::ColumnPartition; amalgamated::Bool)
    n = P.n
    (length(perm) == n && length(parent) == n && length(counts) == n && length(cp.order) == n) ||
        throw(InvalidValueError("perm, parent, counts and the partition must have length $n"))
    fperm = Vector{Int}(perm)[cp.order]
    fiperm = invperm(fperm)
    par = _relabel_parent(parent, cp.order)
    cnt = Vector{Int}(counts[cp.order])
    ns = nsupernodes(cp)
    super_ptr = cp.super_ptr
    col2sn = _col2sn(super_ptr)
    snparent = cp.snparent
    children = [Int[] for _ in 1:ns]
    for s in 1:ns
        snparent[s] != 0 && push!(children[snparent[s]], s)
    end
    rowptr = Vector{Int}(undef, ns + 1)
    rowptr[1] = 1
    rowval = Int[]
    mark = zeros(Int, n)
    below = Int[]
    nnz_stored = 0
    flops = 0.0
    for s in 1:ns                                  # children have smaller indices
        first, last = super_ptr[s], super_ptr[s + 1] - 1
        empty!(below)
        for j in first:last, r in neighbors(P, fperm[j])
            i = fiperm[r]
            if i > last && mark[i] != s
                mark[i] = s
                push!(below, i)
            end
        end
        for c in children[s], i in view(rowval, rowptr[c]:(rowptr[c + 1] - 1))
            if i > last && mark[i] != s
                mark[i] = s
                push!(below, i)
            end
        end
        sort!(below)
        append!(rowval, first:last)
        append!(rowval, below)
        rowptr[s + 1] = length(rowval) + 1
        w = last - first + 1
        f = w + length(below)
        nnz_stored += _trapezoid(w, f)
        for k in 0:(w - 1)
            flops += Float64(f - k)^2
        end
    end
    snpost = postorder(snparent)
    return SupernodePartition(n, fperm, fiperm, par, cnt, Vector{Int}(super_ptr), col2sn, Vector{Int}(snparent),
                              snpost, rowptr, rowval, nnz_stored, nnz_L(cnt), flops, amalgamated)
end

"""
    supernode_partition(P::SymmetricPattern, perm, opts::Options = Options()) -> SupernodePartition

The whole supernode step for the ordering `perm` (e.g. `compute_ordering(P, opts).perm`):
[`etree`](@ref), [`postorder`](@ref) (siblings by original column, so the result depends only on the
etree: an analysis under its own output permutation `sp.perm` reproduces `sp`), [`colcounts`](@ref),
[`fundamental_supernodes`](@ref), [`amalgamate`](@ref) with `opts.amalgamation`
unless `opts.use_superpanels == 0`, and the supernodal symbolic factorization.
"""
function supernode_partition(P::SymmetricPattern, perm::AbstractVector{<:Integer}, opts::Options = Options())
    parent = etree(P, perm)
    post = postorder(parent; key = perm)          # canonical: the result depends only on the etree (T24)
    counts = colcounts(P, perm, parent, post)
    cp = fundamental_supernodes(parent, post, counts)
    amalgamated = opts.use_superpanels != 0
    amalgamated && (cp = amalgamate(cp, parent, counts, opts.amalgamation))
    return SupernodePartition(P, perm, parent, counts, cp; amalgamated)
end

"""
    schur_supernode_partition(P::SymmetricPattern, perm, ns, opts::Options = Options()) -> SupernodePartition

[`supernode_partition`](@ref) for Schur complement mode (PLAN §3.6): the last
`ns` columns of `perm` (the Schur block, [`compute_schur_ordering`](@ref)) form
one dense root supernode, the last one, with its columns in the order of
`perm`; its etree is the chain of those columns and its column counts are the
dense ones (`ns, ns - 1, …, 1`). The other columns are partitioned as usual
(fundamental supernodes and [`amalgamate`](@ref) on their elimination forest,
so nothing is merged into the Schur root); a supernode whose last column has
its etree parent in the Schur block becomes a child of the Schur root.
"""
function schur_supernode_partition(P::SymmetricPattern, perm::AbstractVector{<:Integer}, ns::Integer,
                                   opts::Options = Options())
    n = P.n
    1 <= ns <= n || throw(InvalidValueError("schur_supernode_partition: ns = $ns, expected 1:$n"))
    m = n - ns
    parent = etree(P, perm)
    post = postorder(parent)
    counts = colcounts(P, perm, parent, post)
    subparent = [parent[j] > m ? 0 : parent[j] for j in 1:m]
    for j in (m + 1):n                          # the Schur block: a dense chain
        parent[j] = j < n ? j + 1 : 0
        counts[j] = n - j + 1
    end
    subcounts = counts[1:m]
    cp = fundamental_supernodes(subparent, postorder(subparent; key = view(perm, 1:m)), subcounts)
    amalgamated = opts.use_superpanels != 0
    amalgamated && m > 0 && (cp = amalgamate(cp, subparent, subcounts, opts.amalgamation))
    nsub = nsupernodes(cp)
    snparent = copy(cp.snparent)
    for s in 1:nsub
        snparent[s] == 0 && parent[cp.order[cp.super_ptr[s + 1] - 1]] > m && (snparent[s] = nsub + 1)
    end
    push!(snparent, 0)
    sp = ColumnPartition([cp.order; (m + 1):n], [cp.super_ptr; n + 1], snparent)
    return SupernodePartition(P, perm, parent, counts, sp; amalgamated)
end
