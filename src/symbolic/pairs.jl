# 2×2 pivot candidate pairs for "S"/"H" (issue #64; MA57/HSL_MA97 style, Duff &
# Pralet 2005): a row whose diagonal is zero or negligible is matched with a
# neighbour and the ordering runs on the graph compressed by these pairs, so the
# partner is eliminated right before the candidate and both share a supernode,
# where in-front Bunch–Kaufman can take them as a 2×2 pivot.
#
# Host only, plain `Int` arrays. The values are read once, at analysis.

"""
    PIVOT_PAIR_TOLERANCE

Relative tolerance of the 2×2 candidate test of [`pivot_pairs`](@ref): row `i`
is a candidate when `|aᵢᵢ| ≤ PIVOT_PAIR_TOLERANCE · maxⱼ≠ᵢ |aᵢⱼ|` (`1e-6`).
"""
const PIVOT_PAIR_TOLERANCE = 1.0e-6

_pairs_structure(s::Structure) = s == STRUCTURE_SYMMETRIC || s == STRUCTURE_HERMITIAN

"""
    pairs_enabled(structure, opts::Options) -> Bool

Whether the analysis looks for 2×2 pivot candidate pairs: `opts.pivot_pairs ==
"default"`, structure `"S"` or `"H"`, no `user_perm` and not the natural ordering
(`reordering_alg = "algo5"`).
"""
pairs_enabled(structure, opts::Options) =
    opts.pivot_pairs == PIVOT_PAIRS_DEFAULT && _pairs_structure(_structure(structure)) &&
    opts.user_perm === nothing && opts.reordering_alg != REORDERING_NATURAL

"""
    pivot_pairs(P::SymmetricPattern, rowptr, colval, nzval, n, structure; view = 'F', index = 'O')
        -> Vector{Tuple{Int,Int}}

2×2 pivot candidate pairs `(partner, candidate)` of the CSR matrix (`rowptr`,
`colval`, `nzval`; the first batch member when `nzval` is a matrix) whose
symmetric pattern is `P`, read with the view and index rules of
[`SymmetricPattern`](@ref) (duplicates summed). Sorted by candidate.

* Candidates: rows `i` whose diagonal is absent from the pattern, or with
  `|aᵢᵢ| ≤ τ maxⱼ≠ᵢ |aᵢⱼ|`, `τ =` [`PIVOT_PAIR_TOLERANCE`](@ref) (an exact zero
  included).
* Pairing: a greedy matching of candidates to non-candidate neighbours with
  `aᵢⱼ ≠ 0`. Candidates are taken by increasing number of such neighbours (then
  by index); each takes the free neighbour with the largest `|aᵢⱼ|`, ties by
  smaller degree in `P`, then smaller index. Each non-candidate is matched at
  most once; unmatched candidates stay single columns.

Does not look at `structure` beyond the view rules; [`pairs_enabled`](@ref)
decides whether the analysis calls it.
"""
function pivot_pairs(P::SymmetricPattern, rowptr::AbstractVector{<:Integer}, colval::AbstractVector{<:Integer},
                     nzval::AbstractArray, n::Integer, structure; view = VIEW_FULL, index = INDEX_ONE)
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
    candidate = falses(n)
    for i in 1:n
        amax = maximum(p -> w[p], P.colptr[i]:(P.colptr[i + 1] - 1); init = zero(eltype(w)))
        candidate[i] = !dpresent[i] || abs(dval[i]) <= PIVOT_PAIR_TOLERANCE * amax
    end
    eligible(i, p) = !candidate[P.rowval[p]] && w[p] > 0
    cands = [i for i in 1:n if candidate[i]]
    nelig = [count(p -> eligible(i, p), P.colptr[i]:(P.colptr[i + 1] - 1)) for i in cands]
    order = sortperm(collect(zip(nelig, cands)))
    matched = falses(n)
    pairs = Tuple{Int, Int}[]
    deg(j) = P.colptr[j + 1] - P.colptr[j]
    for t in order
        i = cands[t]
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

pivot_pairs(P::SymmetricPattern, A::CSR, structure; view = VIEW_FULL) =
    pivot_pairs(P, A.rowptr, A.colval, A.nzval, A.nrows, structure; view, index = A.index)

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

