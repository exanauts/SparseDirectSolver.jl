# Matching and scaling (PLAN §1.3 `matching_alg`, §1.4 `perm_matching`,
# `scale_row`/`scale_col`, milestone M9): MC64-style weighted bipartite matching
# of the rows of the stored matrix to its columns (Duff & Koster 2001), on the
# host, at analysis, from the values of the first batch member.
#
# * `"G"`: the matrix factored is `M = (Dr A Dc)[:, q]`: the matched entry of
#   row `i` (column `q[i]`) moves to the diagonal, and with the scalings of job 5
#   `|mᵢᵢ| = 1 ≥ |mᵢⱼ|`. The symmetric ordering and LU then run on `M`; its CSR
#   pattern is the stored one with relabeled column indices (no value moves).
# * `"S"`/`"H"`/`"SPD"`/`"HPD"`: symmetric scaling `D A D` with
#   `dᵢ = sqrt(rᵢ cᵢ)` from the matching of the full matrix (Duff & Pralet
#   2005), which keeps the symmetry and, by Sylvester's law, the inertia; the
#   cycles of the matching permutation give the 2×2 pivot pairs of `"S"`/`"H"`
#   ([`matching_pairs`](@ref)).
#
# Host only, plain `Int`/`Float64` arrays.

"""
    Matching

Result of [`compute_matching`](@ref) (host):

* `alg::MatchingAlg`: the job that ran (`MATCHING_AUTO` resolved to job 5);
* `perm::Vector{Int}`: the matching permutation, row `i` matched to column
  `perm[i]` (the `"perm_matching"` data parameter; for a structurally singular
  matrix the unmatched rows get the unmatched columns in increasing order);
* `matched::BitVector`: row `i` has a matched entry (`false` only for a
  structurally singular matrix);
* `rscale`, `cscale::Vector{Float64}`: row and column scaling factors
  (`"scale_row"`, `"scale_col"`; ones for jobs 1–4, equal for the symmetric
  structures);
* `symmetric::Bool`: symmetric scaling only (structures other than `"G"`): the
  factored matrix is `D A D`, not permuted.
"""
struct Matching
    alg::MatchingAlg
    perm::Vector{Int}
    matched::BitVector
    rscale::Vector{Float64}
    cscale::Vector{Float64}
    symmetric::Bool
end

Base.show(io::IO, m::Matching) =
    print(io, "Matching(", convert(String, m.alg), ", n = ", length(m.perm), ", matched = ", count(m.matched),
          m.symmetric ? ", symmetric" : "", ")")

"""
    matching_job(alg::MatchingAlg) -> Int

The MC64 job of `matching_alg` (PLAN §1.3): `"algo1"`–`"algo5"` are jobs 1–5,
`"algo6"` (auto) is job 5, `"default"` is `0` (no matching).
"""
matching_job(alg::MatchingAlg) = alg == MATCHING_AUTO ? 5 : Int(alg)

# --- binary heap of (key, vertex), lazy deletion ---------------------------------

function _heap_push!(keys::Vector{Float64}, vals::Vector{Int}, k::Float64, v::Int)
    push!(keys, k)
    push!(vals, v)
    i = length(keys)
    while i > 1
        p = i >> 1
        keys[p] <= keys[i] && break
        keys[p], keys[i] = keys[i], keys[p]
        vals[p], vals[i] = vals[i], vals[p]
        i = p
    end
    return nothing
end

function _heap_pop!(keys::Vector{Float64}, vals::Vector{Int})
    k, v = keys[1], vals[1]
    last_k, last_v = pop!(keys), pop!(vals)
    m = length(keys)
    if m > 0
        keys[1], vals[1] = last_k, last_v
        i = 1
        while true
            l = 2i
            l > m && break
            c = (l + 1 <= m && keys[l + 1] < keys[l]) ? l + 1 : l
            keys[i] <= keys[c] && break
            keys[i], keys[c] = keys[c], keys[i]
            vals[i], vals[c] = vals[c], vals[i]
            i = c
        end
    end
    return k, v
end

"""
    min_cost_matching(ptr, adj, cost, n) -> (lmate, u, v)

Minimum-cost matching of maximum cardinality between the left vertices `1:n`
(adjacency `adj[ptr[i]:ptr[i+1]-1]`, costs `cost` ≥ 0 aligned with `adj`) and
the right vertices `1:n`, by successive shortest augmenting paths (Dijkstra
on the reduced costs `cost - u[i] - v[j] ≥ 0`, a binary heap; the sparse
Hungarian method of MC64). Returns `lmate[i]` (the right vertex matched to `i`,
`0` if none) and the dual variables `u` (left) and `v` (right): `u[i] + v[j] ≤
cost` on every edge, with equality on the matched ones.
"""
function min_cost_matching(ptr::AbstractVector{Int}, adj::AbstractVector{Int}, cost::AbstractVector{Float64},
                           n::Int)
    lmate = zeros(Int, n)
    rmate = zeros(Int, n)
    u = zeros(n)
    v = zeros(n)
    # feasible start: u = row minimum, v = 0; then the cheap matching of the tight edges
    for i in 1:n
        u[i] = minimum(p -> cost[p], ptr[i]:(ptr[i + 1] - 1); init = 0.0)
        for p in ptr[i]:(ptr[i + 1] - 1)
            j = adj[p]
            if rmate[j] == 0 && cost[p] - u[i] <= 0
                lmate[i] = j
                rmate[j] = i
                break
            end
        end
    end
    dist = fill(Inf, n)
    pred = zeros(Int, n)
    done = falses(n)
    touched = Int[]
    finals = Int[]
    hk = Float64[]
    hv = Int[]
    for i0 in 1:n
        lmate[i0] == 0 || continue
        empty!(touched); empty!(finals); empty!(hk); empty!(hv)
        cur, dcur = i0, 0.0
        jfree, L = 0, Inf
        while true
            for p in ptr[cur]:(ptr[cur + 1] - 1)
                j = adj[p]
                done[j] && continue
                nd = dcur + max(cost[p] - u[cur] - v[j], 0.0)
                if nd < dist[j]
                    isinf(dist[j]) && push!(touched, j)
                    dist[j] = nd
                    pred[j] = cur
                    _heap_push!(hk, hv, nd, j)
                end
            end
            j = 0
            while !isempty(hk)
                d, jj = _heap_pop!(hk, hv)
                (done[jj] || d > dist[jj]) && continue
                j = jj
                break
            end
            j == 0 && break                           # no augmenting path: i0 stays unmatched
            done[j] = true
            push!(finals, j)
            if rmate[j] == 0
                jfree, L = j, dist[j]
                break
            end
            cur, dcur = rmate[j], dist[j]             # the matched edge has reduced cost 0
        end
        if jfree != 0
            u[i0] += L
            for j in finals
                j == jfree && continue
                u[rmate[j]] += L - dist[j]
                v[j] -= L - dist[j]
            end
            j = jfree
            while true                                # flip the path
                i = pred[j]
                nj = lmate[i]
                lmate[i] = j
                rmate[j] = i
                i == i0 && break
                j = nj
            end
        end
        for j in touched
            dist[j] = Inf
            done[j] = false
        end
    end
    return lmate, u, v
end

# rows → columns adjacency with |aᵢⱼ| (duplicates summed first, explicit zeros kept): CSR of the full matrix
function _abs_csr(rowptr, colval, nzval, n::Int, s::Structure, v::MatrixView, index)
    rp, cv = _host_pattern(rowptr, colval, n, index)
    nz = Array(nzval)
    nz isa AbstractMatrix && (nz = nz[:, 1])
    I = Int[]
    J = Int[]
    V = eltype(nz)[]
    if s == STRUCTURE_GENERAL
        for r in 1:n, p in rp[r]:(rp[r + 1] - 1)
            push!(I, r); push!(J, cv[p]); push!(V, nz[p])
        end
    else
        rows, cols, srcs = _selected_entries(rp, cv, n, s, v)
        herm = _is_hermitian(s)
        for k in eachindex(rows)
            r, c, x = rows[k], cols[k], nz[srcs[k]]
            push!(I, r); push!(J, c); push!(V, x)
            r == c && continue
            push!(I, c); push!(J, r); push!(V, herm ? conj(x) : x)
        end
    end
    A = sparse(J, I, V, n, n)                      # CSC of the transpose = CSR of the matrix, duplicates summed
    return A.colptr, A.rowval, Float64.(abs.(A.nzval))
end

# maximum cardinality matching on the edges with `keep[p]` (zero costs)
function _cardinality_matching(ptr, adj, keep::AbstractVector{Bool}, n)
    sub_ptr = zeros(Int, n + 1)
    sub_ptr[1] = 1
    sub_adj = Int[]
    for i in 1:n
        for p in ptr[i]:(ptr[i + 1] - 1)
            keep[p] && push!(sub_adj, adj[p])
        end
        sub_ptr[i + 1] = length(sub_adj) + 1
    end
    lmate, _, _ = min_cost_matching(sub_ptr, sub_adj, zeros(length(sub_adj)), n)
    return lmate
end

# MC64 jobs 2/3: maximize the smallest matched |aᵢⱼ| (bottleneck), by bisection over the distinct values
function _bottleneck_matching(ptr, adj, w, n)
    full = _cardinality_matching(ptr, adj, trues(length(w)), n)
    target = count(!iszero, full)
    vals = sort!(unique(filter(>(0), w)))
    isempty(vals) && return full
    lo, hi = 1, length(vals)                       # the largest threshold index that keeps `target` matches
    best = _cardinality_matching(ptr, adj, w .>= vals[1], n)
    count(!iszero, best) == target || return full
    while lo < hi
        mid = (lo + hi + 1) >> 1
        m = _cardinality_matching(ptr, adj, w .>= vals[mid], n)
        if count(!iszero, m) == target
            lo, best = mid, m
        else
            hi = mid - 1
        end
    end
    return best
end

"""
    mc64(ptr, adj, w, n, job) -> (lmate, rscale, cscale)

MC64-style matching of the rows `1:n` of a square matrix (CSR `ptr`/`adj`,
absolute values `w`) to its columns:

* job 1: maximum cardinality (structural; every stored entry counts);
* jobs 2 and 3: maximize the smallest matched `|aᵢⱼ|` (bottleneck);
* job 4: maximize the sum of the matched `|aᵢⱼ|`;
* job 5: maximize their product, with the scalings `rᵢ = exp(uᵢ) / maxⱼ |aᵢⱼ|`,
  `cⱼ = exp(vⱼ)` from the duals of the assignment on the costs
  `log maxₖ |aᵢₖ| - log |aᵢⱼ|`, so that `|rᵢ aᵢⱼ cⱼ| ≤ 1`, with equality on the
  matched entries.

Jobs 2–5 use the nonzero entries only. `lmate[i]` is the column matched to row
`i` (`0` for an unmatched row of a structurally singular matrix); the scalings
are ones for jobs 1–4.
"""
function mc64(ptr::AbstractVector{Int}, adj::AbstractVector{Int}, w::AbstractVector{Float64}, n::Int, job::Int)
    rs = ones(n)
    cs = ones(n)
    if job == 1
        return _cardinality_matching(ptr, adj, trues(length(w)), n), rs, cs
    elseif job == 2 || job == 3
        return _bottleneck_matching(ptr, adj, w, n), rs, cs
    elseif job != 4 && job != 5
        throw(InvalidValueError("unknown matching job $job; expected 1 to 5"))
    end
    # jobs 4 and 5: assignment on the nonzero entries
    rowmax = [maximum(p -> w[p], ptr[i]:(ptr[i + 1] - 1); init = 0.0) for i in 1:n]
    sptr = zeros(Int, n + 1)
    sptr[1] = 1
    sadj = Int[]
    cost = Float64[]
    for i in 1:n
        for p in ptr[i]:(ptr[i + 1] - 1)
            w[p] > 0 || continue
            push!(sadj, adj[p])
            push!(cost, job == 5 ? log(rowmax[i]) - log(w[p]) : rowmax[i] - w[p])
        end
        sptr[i + 1] = length(sadj) + 1
    end
    lmate, u, v = min_cost_matching(sptr, sadj, cost, n)
    if job == 5
        for i in 1:n
            rs[i] = rowmax[i] > 0 ? exp(u[i]) / rowmax[i] : 1.0
            cs[i] = exp(v[i])
        end
        _finite_scaling!(rs)
        _finite_scaling!(cs)
    end
    return lmate, rs, cs
end

# an empty row or column (structurally singular) or an over/underflow keeps the factor 1
function _finite_scaling!(s::Vector{Float64})
    for i in eachindex(s)
        (isfinite(s[i]) && s[i] > 0) || (s[i] = 1.0)
    end
    return s
end

# complete a partial matching to a permutation: unmatched rows take the unmatched columns in increasing order
function _complete_matching(lmate::Vector{Int}, n::Int)
    perm = copy(lmate)
    used = falses(n)
    for j in lmate
        j != 0 && (used[j] = true)
    end
    free = [j for j in 1:n if !used[j]]
    k = 0
    for i in 1:n
        perm[i] == 0 || continue
        perm[i] = free[k += 1]
    end
    return perm
end

"""
    compute_matching(rowptr, colval, nzval, n, structure, opts::Options; view = 'F', index = 'O')
        -> Union{Nothing, Matching}

The [`Matching`](@ref) of the analysis for `opts.matching_alg` (`nothing` for
`"default"`), from the CSR matrix (`rowptr`, `colval`, `nzval`; the first
batch member when `nzval` is a matrix) read with the view and index rules of
[`SymmetricPattern`](@ref):

* `"G"`: [`mc64`](@ref) on the stored rows (`rscale`, `cscale` from job 5);
* the symmetric structures: [`mc64`](@ref) on the full matrix (both triangles),
  then the symmetric scaling `dᵢ = sqrt(rᵢ cᵢ)` of Duff & Pralet (2005), so
  `|dᵢ aᵢⱼ dⱼ| ≤ 1` (`rscale = cscale = d`). An index left unmatched by a
  structurally singular matrix gets `dᵢ = 1 / maxⱼ |aᵢⱼ| dⱼ` over its matched
  neighbours (`1` without one).

Reads `nzval` on the host once.
"""
function compute_matching(rowptr, colval, nzval, n::Integer, structure, opts::Options; view = VIEW_FULL,
                          index = INDEX_ONE)
    opts.matching_alg == MATCHING_NONE && return nothing
    s = _structure(structure)
    n = Int(n)
    job = matching_job(opts.matching_alg)
    alg = MatchingAlg(job)
    ptr, adj, w = _abs_csr(rowptr, colval, nzval, n, s, _matrix_view(view), _index_base(index))
    lmate, rs, cs = mc64(ptr, adj, w, n, job)
    matched = BitVector(lmate .!= 0)
    perm = _complete_matching(lmate, n)
    s == STRUCTURE_GENERAL && return Matching(alg, perm, matched, rs, cs, false)
    d = sqrt.(rs .* cs)
    if !all(matched)
        for i in 1:n
            matched[i] && continue
            m = maximum(p -> matched[adj[p]] ? w[p] * d[adj[p]] : 0.0, ptr[i]:(ptr[i + 1] - 1); init = 0.0)
            d[i] = m > 0 ? 1 / m : 1.0
        end
    end
    _finite_scaling!(d)
    return Matching(alg, perm, matched, d, copy(d), true)
end

"""
    matching_pairs(m::Matching, rowptr, colval, nzval, n, structure, opts; view = 'F', index = 'O')
        -> Vector{Tuple{Int,Int}}

The 2×2 pivot pairs `(partner, candidate)` of `"S"`/`"H"` derived from the
symmetric matching `m` (issue #67, Duff & Pralet 2005), sorted by candidate,
for [`compute_ordering`](@ref)`(…; pairs)`:

* the matching permutation is split into cycles (only cycles of matched
  indices); a 2-cycle `(i, j)` is a pair; a longer cycle is split into pairs of
  consecutive indices (adjacent in the matrix, since `aᵢ,perm[i] ≠ 0`): for an
  even length the alternative with the larger product of scaled `|dᵢ aᵢⱼ dⱼ|`,
  for an odd length leaving out the index with the largest scaled diagonal;
* `opts.pivot_pairs = "default"` keeps a pair only when one of its indices is a
  2×2 candidate of the scaled matrix: `|dᵢ² aᵢᵢ| ≤ τ maxⱼ≠ᵢ |dᵢ aᵢⱼ dⱼ|`, `τ =
  opts.pivot_pair_tolerance`; `"all"` keeps every pair; `"none"` none;
* the candidate of a pair is its index with the smaller relative scaled
  diagonal, the other one the partner (eliminated first).
"""
function matching_pairs(m::Matching, rowptr, colval, nzval, n::Integer, structure, opts::Options;
                        view = VIEW_FULL, index = INDEX_ONE)
    none = Tuple{Int, Int}[]
    opts.pivot_pairs == PIVOT_PAIRS_NONE && return none
    s = _structure(structure)
    n = Int(n)
    ptr, adj, w = _abs_csr(rowptr, colval, nzval, n, s, _matrix_view(view), _index_base(index))
    d = m.rscale
    # scaled diagonal, scaled off-diagonal row maximum, and the scaled |a| of an entry (i, j)
    dg = zeros(n)
    offmax = zeros(n)
    for i in 1:n, p in ptr[i]:(ptr[i + 1] - 1)
        j = adj[p]
        x = d[i] * w[p] * d[j]
        j == i ? (dg[i] = x) : (offmax[i] = max(offmax[i], x))
    end
    function entry(i, j)
        r = ptr[i]:(ptr[i + 1] - 1)
        k = findfirst(p -> adj[p] == j, r)
        return k === nothing ? 0.0 : d[i] * w[r[k]] * d[j]
    end
    ratio(i) = offmax[i] > 0 ? dg[i] / offmax[i] : Inf
    candidate(i) = dg[i] <= opts.pivot_pair_tolerance * offmax[i]
    seen = falses(n)
    pairs = Tuple{Int, Int}[]
    cyc = Int[]
    for i0 in 1:n
        (seen[i0] || !m.matched[i0]) && continue
        empty!(cyc)
        i = i0
        ok = true
        while !seen[i]
            seen[i] = true
            push!(cyc, i)
            m.matched[i] || (ok = false)
            i = m.perm[i]
        end
        (ok && i == i0 && length(cyc) >= 2) || continue
        L = length(cyc)
        if iseven(L)
            p0 = prod(entry(cyc[k], cyc[k + 1]) for k in 1:2:L)
            p1 = prod(entry(cyc[k + 1], cyc[mod1(k + 2, L)]) for k in 1:2:L)
            start = p1 > p0 ? 2 : 1
        else
            start = mod1(argmax([dg[c] for c in cyc]) + 1, L)   # the index after the singleton
        end
        for t in 0:2:(L - 2)
            a, b = cyc[mod1(start + t, L)], cyc[mod1(start + t + 1, L)]
            opts.pivot_pairs == PIVOT_PAIRS_ALL || candidate(a) || candidate(b) || continue
            push!(pairs, ratio(a) < ratio(b) ? (b, a) : (a, b))
        end
    end
    sort!(pairs; by = last)
    return pairs
end

"""
    matched_colval(m::Matching, colval, index) -> Vector

The column indices of the CSR pattern of the matched matrix `M = (Dr A Dc)[:, q]`
of `"G"` (`q = m.perm`): column `c` of the stored matrix becomes column
`q⁻¹[c]`, in the index base `index` and the element type of `colval`.
"""
function matched_colval(m::Matching, colval::AbstractVector{INT}, index) where {INT}
    qinv = invperm(m.perm)
    off = _index_base(index) == INDEX_ZERO ? 1 : 0
    return INT[INT(qinv[c + off] - off) for c in colval]
end

"""
    entry_scaling(m::Matching, rowptr, colval, n; index = 'O') -> Vector{Float64}

The factor `rᵢ cⱼ` of every stored entry `(i, j)` of the CSR pattern (original
column numbering), aligned with `colval`: the factored values are
`rᵢ aᵢⱼ cⱼ` ([`scale_values!`](@ref)).
"""
function entry_scaling(m::Matching, rowptr, colval, n::Integer; index = INDEX_ONE)
    rp, cv = _host_pattern(rowptr, colval, n, index)
    w = Vector{Float64}(undef, length(cv))
    for r in 1:n, p in rp[r]:(rp[r + 1] - 1)
        w[p] = m.rscale[r] * m.cscale[cv[p]]
    end
    return w
end
