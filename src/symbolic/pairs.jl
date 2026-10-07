# 2×2 pivot candidate pairs for "S"/"H" (issue #64; MA57/HSL_MA97 style, Duff &
# Pralet 2005): a row whose diagonal is zero or negligible is matched with a
# neighbour and the ordering runs on the graph compressed by these pairs, so the
# partner is eliminated right before the candidate and both share a supernode,
# where in-front Bunch–Kaufman can take them as a 2×2 pivot.
#
# Which candidates are paired (`pivot_pairs`): "default" pairs only those whose
# pivot is structurally zero in the ordering (`zero_pivot_pairs!`, iterated with
# re-ordering; issue #66), "all" pairs every candidate (`pivot_pairs`, about
# 2× nnz(L) on KKT systems).
#
# Host only, plain `Int` arrays. The values are read once, at analysis.

"""
    PIVOT_PAIR_TOLERANCE

Default relative tolerance of the 2×2 candidate test of [`pivot_pairs`](@ref):
row `i` is a candidate when `|aᵢᵢ| ≤ τ · maxⱼ≠ᵢ |aᵢⱼ|`, `τ =` the parameter
`"pivot_pair_tolerance"` (default `1e-6`).
"""
const PIVOT_PAIR_TOLERANCE = 1.0e-6

_pairs_structure(s::Structure) = s == STRUCTURE_SYMMETRIC || s == STRUCTURE_HERMITIAN

"""
    pairs_enabled(structure, opts::Options) -> Bool

Whether the analysis looks for 2×2 pivot candidate pairs: `opts.pivot_pairs` is
`"default"` or `"all"`, structure `"S"` or `"H"`, no `user_perm` and not the
natural ordering (`reordering_alg = "algo5"`).
"""
pairs_enabled(structure, opts::Options) =
    opts.pivot_pairs != PIVOT_PAIRS_NONE && _pairs_structure(_structure(structure)) &&
    opts.user_perm === nothing && opts.reordering_alg != REORDERING_NATURAL

"""
    PairCandidates

2×2 pivot candidates of a matrix, from [`pivot_candidates`](@ref) (host arrays):

* `candidate::BitVector`: row `i` has an absent diagonal or
  `|aᵢᵢ| ≤ τ maxⱼ≠ᵢ |aᵢⱼ|`, `τ =` [`PIVOT_PAIR_TOLERANCE`](@ref);
* `w::Vector{R}`: `|aᵢⱼ|` (duplicates summed first), aligned with `P.rowval`
  of the pattern `P` it was computed on (`R = real(T)`);
* `amax::Vector{R}`: `maxⱼ≠ᵢ |aᵢⱼ|` per row (`0` for a row without off-diagonal entries).
"""
struct PairCandidates{R <: Real}
    candidate::BitVector
    w::Vector{R}
    amax::Vector{R}
end

"""
    pivot_candidates(P::SymmetricPattern, rowptr, colval, nzval, n, structure; view = 'F', index = 'O')
        -> PairCandidates
    pivot_candidates(P::SymmetricPattern, A::CSR, structure; view = 'F') -> PairCandidates

The 2×2 pivot candidates ([`PairCandidates`](@ref)) of the CSR matrix
(`rowptr`, `colval`, `nzval`; the first batch member when `nzval` is a matrix)
whose symmetric pattern is `P`, read with the view and index rules of
[`SymmetricPattern`](@ref) (duplicates summed), with the candidate tolerance
`τ = tolerance` (the analysis passes `opts.pivot_pair_tolerance`). The only
read of the values at analysis.
"""
function pivot_candidates(P::SymmetricPattern, rowptr::AbstractVector{<:Integer}, colval::AbstractVector{<:Integer},
                          nzval::AbstractArray, n::Integer, structure; view = VIEW_FULL, index = INDEX_ONE,
                          tolerance::Real = PIVOT_PAIR_TOLERANCE)
    P.n == n || throw(InvalidValueError("the pattern has size $(P.n), the matrix $n"))
    s = _structure(structure)
    rp, cv = _host_pattern(rowptr, colval, n, index)
    nz = Array(nzval)
    nz isa AbstractMatrix && (nz = nz[:, 1])
    length(nz) >= length(cv) ||
        throw(InvalidValueError("nzval has $(length(nz)) entries, the pattern $(length(cv))"))
    rows, cols, srcs = _selected_entries(rp, cv, n, s, _matrix_view(view))
    T = eltype(nz)
    dval = zeros(T, n)
    dpresent = falses(n)
    wval = zeros(T, nnz(P))                       # aligned with P.rowval
    for k in eachindex(rows)
        r, c, v = rows[k], cols[k], nz[srcs[k]]
        if r == c
            dval[r] += v
            dpresent[r] = true
        else                                      # (r, c) in column c and (c, r) in column r
            wval[P.colptr[c] - 1 + searchsortedfirst(neighbors(P, c), r)] += v
            wval[P.colptr[r] - 1 + searchsortedfirst(neighbors(P, r), c)] += v
        end
    end
    w = abs.(wval)
    R = eltype(w)
    amax = zeros(R, n)
    candidate = falses(n)
    for i in 1:n
        amax[i] = maximum(p -> w[p], P.colptr[i]:(P.colptr[i + 1] - 1); init = zero(R))
        candidate[i] = !dpresent[i] || abs(dval[i]) <= tolerance * amax[i]
    end
    return PairCandidates{R}(candidate, w, amax)
end

pivot_candidates(P::SymmetricPattern, A::CSR, structure; view = VIEW_FULL, tolerance::Real = PIVOT_PAIR_TOLERANCE) =
    pivot_candidates(P, A.rowptr, A.colval, A.nzval, A.nrows, structure; view, index = A.index, tolerance)

"""
    pivot_pairs(P::SymmetricPattern, rowptr, colval, nzval, n, structure; view = 'F', index = 'O')
        -> Vector{Tuple{Int,Int}}
    pivot_pairs(P::SymmetricPattern, A::CSR, structure; view = 'F')
    pivot_pairs(P::SymmetricPattern, cands::PairCandidates)

The pairs of `pivot_pairs = "all"`: every 2×2 pivot candidate of
[`pivot_candidates`](@ref) matched to a neighbour, as `(partner, candidate)`,
sorted by candidate.

* Pairing: a greedy matching of candidates to non-candidate neighbours with
  `aᵢⱼ ≠ 0`. Candidates are taken by increasing number of such neighbours (then
  by index); each takes the free neighbour with the largest `|aᵢⱼ|`, ties by
  smaller degree in `P`, then smaller index. Each non-candidate is matched at
  most once; unmatched candidates stay single columns.

On KKT systems about half of the rows are candidates, and ordering with all of
them paired roughly doubles `nnz(L)` (issue #66); `"default"` pairs only the
candidates that need it, see [`zero_pivot_pairs!`](@ref). Does not look at `structure`
beyond the view rules; [`pairs_enabled`](@ref) decides whether the analysis
calls it.
"""
function pivot_pairs(P::SymmetricPattern, cands::PairCandidates)
    n = P.n
    candidate, w = cands.candidate, cands.w
    eligible(i, p) = !candidate[P.rowval[p]] && w[p] > 0
    cs = [i for i in 1:n if candidate[i]]
    nelig = [count(p -> eligible(i, p), P.colptr[i]:(P.colptr[i + 1] - 1)) for i in cs]
    order = sortperm(collect(zip(nelig, cs)))
    matched = falses(n)
    pairs = Tuple{Int, Int}[]
    deg(j) = P.colptr[j + 1] - P.colptr[j]
    for t in order
        i = cs[t]
        best = 0
        for p in P.colptr[i]:(P.colptr[i + 1] - 1)
            j = P.rowval[p]
            (eligible(i, p) && !matched[j]) || continue
            if best == 0 || w[p] > w[best] || (w[p] == w[best] && deg(j) < deg(P.rowval[best]))
                best = p                          # equal weight and degree: the smaller index (seen first)
            end
        end
        best == 0 && continue
        matched[P.rowval[best]] = true
        push!(pairs, (P.rowval[best], i))
    end
    sort!(pairs; by = last)
    return pairs
end

pivot_pairs(P::SymmetricPattern, rowptr::AbstractVector{<:Integer}, colval::AbstractVector{<:Integer},
            nzval::AbstractArray, n::Integer, structure; view = VIEW_FULL, index = INDEX_ONE) =
    pivot_pairs(P, pivot_candidates(P, rowptr, colval, nzval, n, structure; view, index))

pivot_pairs(P::SymmetricPattern, A::CSR, structure; view = VIEW_FULL) =
    pivot_pairs(P, A.rowptr, A.colval, A.nzval, A.nrows, structure; view, index = A.index)

"""
    PIVOT_PAIRS_MAX_ROUNDS

Maximum number of [`zero_pivot_pairs!`](@ref) rounds (each followed by a
re-ordering) of the `pivot_pairs = "default"` analysis (`8`).
"""
const PIVOT_PAIRS_MAX_ROUNDS = 8

"""
    structural_zero_pivots(cands::PairCandidates, P::SymmetricPattern, perm, pairs = []) -> Vector{Int}

The candidates of `cands`, not in `pairs`, whose pivot is structurally zero when
`A[perm, perm]` is factored column by column, in elimination order.

With generic values, the `k`-th pivot is nonzero exactly when the leading block
`A[perm[1:k], perm[1:k]]` is structurally nonsingular (has a perfect matching
on its pattern) given that the previous one is; the diagonal of a candidate
counts as absent. The test runs incrementally: a perfect matching of the prefix
is kept, a non-candidate is matched to its own diagonal, and a candidate looks
for an augmenting path from its row to its column through earlier columns
(`|aᵢⱼ| > 0` edges, and the diagonal of a non-candidate row that an earlier path
moved off it). A candidate without one is reported and matched to itself,
as if a partner fixed it. A pair `(a, c)` with `c` right after `a` is matched
crosswise (a 2×2 block). An elimination-tree leaf candidate is always reported;
so is a candidate whose earlier neighbours are all used up by earlier
candidates (two dual rows coupled to the same single primal).
"""
function structural_zero_pivots(cands::PairCandidates, P::SymmetricPattern, perm::AbstractVector{<:Integer},
                                pairs = Tuple{Int, Int}[])
    n = P.n
    length(perm) == n || throw(InvalidValueError("perm has length $(length(perm)), the pattern $n"))
    candidate, w = cands.candidate, cands.w
    length(candidate) == n || throw(InvalidValueError("the candidates have length $(length(candidate)), the pattern $n"))
    pos = invperm(perm)
    mate = zeros(Int, n)
    for (a, b) in pairs
        mate[a] = b
        mate[b] = a
    end
    colm = zeros(Int, n)                          # column j is matched to row colm[j]
    seen = zeros(Int, n)                          # columns visited by the current search (stamp)
    srow = Int[]                                  # DFS stack: row, next position in its adjacency, column taken
    sptr = Int[]
    scol = Int[]
    out = Int[]
    for k in 1:n
        v = perm[k]
        m = mate[v]
        if m != 0 && pos[m] == k - 1              # second column of a pair: the 2×2 block, crosswise
            colm[v] = m; colm[m] = v
            continue
        end
        if !candidate[v] || m != 0                # nonzero diagonal, or the first column of a pair
            colm[v] = v
            continue
        end
        # augmenting path from row v to column v through the columns of the prefix (iterative DFS)
        empty!(srow); empty!(sptr); empty!(scol)
        push!(srow, v); push!(sptr, P.colptr[v]); push!(scol, 0)
        found = false
        while !isempty(srow)
            r, p = srow[end], sptr[end]
            if p > P.colptr[r + 1]                # row r exhausted: backtrack
                pop!(srow); pop!(sptr); pop!(scol)
                continue
            end
            sptr[end] = p + 1
            if p == P.colptr[r + 1]               # last slot: the diagonal of a non-candidate row
                candidate[r] && continue          # (not stored; r may have been displaced from it)
                j = r
            else
                j = P.rowval[p]
                w[p] > 0 || continue
            end
            (pos[j] <= k && seen[j] != k) || continue
            seen[j] = k
            scol[end] = j
            if j == v                             # reached the free column: flip the path
                for t in eachindex(srow)
                    colm[scol[t]] = srow[t]
                end
                found = true
                break
            end
            push!(srow, colm[j]); push!(sptr, P.colptr[colm[j]]); push!(scol, 0)
        end
        if !found
            push!(out, v)
            colm[v] = v
        end
    end
    return out
end

"""
    zero_pivot_pairs!(pairs, cands::PairCandidates, P::SymmetricPattern, perm, u) -> (added, unmatched)

One round of the `pivot_pairs = "default"` selection (issue #66): pair the 2×2
candidates whose pivot is structurally zero under the ordering `perm`
([`structural_zero_pivots`](@ref)), appending `(partner, candidate)` to `pairs`
(kept sorted by candidate).

Why only those: every other candidate receives a structurally nonzero update of
its diagonal from earlier columns before it is eliminated, and in-front pivoting
deals with it. A structurally zero pivot stays zero (or tiny) whatever the
values, so it needs a partner in its fully-summed block. On MadNLP K2 systems
this pairs about a fifth of the candidates and costs 1.3–1.4× `nnz(L)` (instead
of ~2× for every candidate, [`pivot_pairs`](@ref)), and leaves a few dozen
perturbed pivots instead of about a thousand.

Partners: a neighbour `j` that is in no pair yet with `|aᵢⱼ| ≥ u · maxₖ≠ᵢ |aᵢₖ|`
(and `aᵢⱼ ≠ 0`), another candidate allowed (a `[0 a; a 0]` block is a good 2×2
pivot, the slack–dual case of K2 systems). The candidates are taken by
increasing number of such free partners at the start of the round, then by
position in `perm`; each takes the free partner with the largest `|aᵢⱼ|` (ties:
earliest in `perm`), as a weak coupling makes a poor 2×2 pivot. Returns the
number of pairs added and the number of structurally zero pivots left without a
partner.

Pairing moves the partner, so the analysis re-orders on the compressed graph and
calls it again until no pair is added ([`compute_ordering`](@ref)).
"""
function zero_pivot_pairs!(pairs::Vector{Tuple{Int, Int}}, cands::PairCandidates, P::SymmetricPattern,
                           perm::AbstractVector{<:Integer}, u::Real)
    zs = structural_zero_pivots(cands, P, perm, pairs)
    n = P.n
    w, amax = cands.w, cands.amax
    inpair = falses(n)
    for (a, b) in pairs
        inpair[a] = inpair[b] = true
    end
    pos = invperm(perm)
    eligible(i, p) = !inpair[P.rowval[p]] && w[p] > 0 && w[p] >= u * amax[i]
    nfree(i) = count(p -> eligible(i, p), P.colptr[i]:(P.colptr[i + 1] - 1))
    order = sortperm(collect(zip([nfree(i) for i in zs], [pos[i] for i in zs])))
    added = unmatched = 0
    for t in order
        i = zs[t]
        inpair[i] && continue                     # taken as the partner of an earlier candidate
        best = 0                                  # position in P.rowval
        for p in P.colptr[i]:(P.colptr[i + 1] - 1)
            eligible(i, p) || continue
            (best == 0 || w[p] > w[best] || (w[p] == w[best] && pos[P.rowval[p]] < pos[P.rowval[best]])) &&
                (best = p)
        end
        if best == 0
            unmatched += 1
            continue
        end
        j = P.rowval[best]
        inpair[j] = inpair[i] = true
        push!(pairs, (j, i))
        added += 1
    end
    sort!(pairs; by = last)
    return added, unmatched
end

# The pairing loop of `pivot_pairs = "default"` for one ordering algorithm: order,
# pair the structurally zero pivots, re-order the compressed graph, until no pair is added.
function _pair_ordering(P::SymmetricPattern, a::Symbol, opts::Options, cands::PairCandidates)
    pairs = Tuple{Int, Int}[]
    perm = _ordering_perm(P, a, opts)
    rounds = 0
    while rounds < PIVOT_PAIRS_MAX_ROUNDS
        added, _ = zero_pivot_pairs!(pairs, cands, P, perm, opts.pivot_threshold)
        added == 0 && break
        rounds += 1
        Pc, gptr, members = compressed_pattern(P, pairs)
        perm = expand_permutation(_ordering_perm(Pc, a, opts), gptr, members)
    end
    return (a, perm, pairs, rounds)
end

"""
    analysis_pairs(P::SymmetricPattern, rowptr, colval, nzval, n, structure, opts; view, index)
        -> (; pairs, candidates)

The 2×2 pivot pair input of [`compute_ordering`](@ref) for the analysis:
nothing when [`pairs_enabled`](@ref) is false; [`pivot_candidates`](@ref) for
`pivot_pairs = "default"` (pairs for the structurally zero pivots); the fixed [`pivot_pairs`](@ref) for
`"all"`. Reads `nzval` on the host once. Pairs are decided from the values present at `"analysis"` (the first batch
member): an all-zero `nzval` gives no pairs and an undefined one arbitrary pairs,
so run `"analysis"` after the first assembly of the matrix (MadNLP: after the
first KKT assembly); `"all"` reads the values as well.
"""
function analysis_pairs(P::SymmetricPattern, rowptr, colval, nzval, n::Integer, structure, opts::Options;
                        view = VIEW_FULL, index = INDEX_ONE)
    none = Tuple{Int, Int}[]
    pairs_enabled(structure, opts) || return (pairs = none, candidates = nothing)
    cands = pivot_candidates(P, rowptr, colval, nzval, n, structure; view, index,
                             tolerance = opts.pivot_pair_tolerance)
    opts.pivot_pairs == PIVOT_PAIRS_ALL && return (pairs = pivot_pairs(P, cands), candidates = nothing)
    return (pairs = none, candidates = cands)
end

function _check_pairs(pairs, n)
    seen = falses(n)
    for (a, b) in pairs
        (1 <= a <= n && 1 <= b <= n && a != b && !seen[a] && !seen[b]) ||
            throw(InvalidValueError("pivot pairs must be disjoint pairs of distinct indices in 1:$n"))
        seen[a] = seen[b] = true
    end
    return nothing
end

# Compressed vertices: a pair is one vertex (partner first), every other column
# one vertex, numbered by increasing first column (the partner for a pair).
# Returns (ptr, members).
function _pair_groups(n::Int, pairs)
    _check_pairs(pairs, n)
    mate = zeros(Int, n)                          # partner → candidate
    paired = falses(n)
    for (a, b) in pairs
        mate[a] = b
        paired[a] = paired[b] = true
    end
    ptr = Int[1]
    members = Int[]
    for v in 1:n
        if mate[v] != 0                           # (partner, candidate)
            push!(members, v, mate[v])
        elseif !paired[v]
            push!(members, v)
        else
            continue                              # a candidate: emitted with its partner
        end
        push!(ptr, length(members) + 1)
    end
    return ptr, members
end

"""
    compressed_pattern(P::SymmetricPattern, pairs) -> (Pc::SymmetricPattern, ptr, members)

The pattern compressed by the 2×2 pairs `(partner, candidate)`: a pair is one
vertex whose adjacency is the union of both rows (self-loops removed), every other
column is one vertex. Compressed vertex `g` stands for the columns
`members[ptr[g]:(ptr[g+1]-1)]` (partner first); vertices are numbered by
increasing first column (the partner for a pair).
"""
function compressed_pattern(P::SymmetricPattern, pairs)
    n = P.n
    ptr, members = _pair_groups(n, pairs)
    ng = length(ptr) - 1
    group = Vector{Int}(undef, n)
    for g in 1:ng, k in ptr[g]:(ptr[g + 1] - 1)
        group[members[k]] = g
    end
    I = Int[]
    J = Int[]
    for j in 1:n, i in neighbors(P, j)
        gi, gj = group[i], group[j]
        gi == gj && continue
        push!(I, gi); push!(J, gj)
    end
    colptr, rowval = _csc_unique(ng, I, J)
    return SymmetricPattern(ng, colptr, rowval), ptr, members
end

"""
    expand_permutation(cperm, ptr, members) -> Vector{Int}

The ordering of the original columns given by an ordering `cperm` of the
compressed vertices of [`compressed_pattern`](@ref): the columns of each vertex in
turn, partner before candidate.
"""
function expand_permutation(cperm::AbstractVector{<:Integer}, ptr::AbstractVector{<:Integer},
                            members::AbstractVector{<:Integer})
    perm = Int[]
    sizehint!(perm, length(members))
    for g in cperm, k in ptr[g]:(ptr[g + 1] - 1)
        push!(perm, members[k])
    end
    return perm
end

"""
    pair_pattern(P::SymmetricPattern, pairs) -> SymmetricPattern

The pattern the symbolic factorization runs on when `pairs` is not empty (`P`
itself otherwise): the expansion of [`compressed_pattern`](@ref), where both
columns of a pair have the union structure of the pair (explicit structural zeros
of the factor) and are adjacent to each other. With the partner ordered right
before its candidate, the candidate is the only child of the partner in the
elimination tree and their columns have the same structure below the pair, so the
pair is (part of) one fundamental supernode; amalgamation only merges supernodes,
so the pair is never split.
"""
function pair_pattern(P::SymmetricPattern, pairs)
    isempty(pairs) && return P
    Pc, ptr, members = compressed_pattern(P, pairs)
    I = Int[]
    J = Int[]
    for g in 1:Pc.n
        mg = view(members, ptr[g]:(ptr[g + 1] - 1))
        for a in mg, b in mg
            a != b && (push!(I, a); push!(J, b))
        end
        for h in neighbors(Pc, g), a in mg, b in view(members, ptr[h]:(ptr[h + 1] - 1))
            push!(I, b); push!(J, a)
        end
    end
    colptr, rowval = _csc_unique(P.n, I, J)
    return SymmetricPattern(P.n, colptr, rowval)
end

"""
    factor_pattern(P::SymmetricPattern, ord::Ordering) -> SymmetricPattern

The pattern of the symbolic factorization for the ordering `ord`:
[`pair_pattern`](@ref)`(P, ord.pairs)`.
"""
factor_pattern(P::SymmetricPattern, ord::Ordering) = pair_pattern(P, ord.pairs)

