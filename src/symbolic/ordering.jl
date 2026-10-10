# Symbolic step 2 (PLAN §2.3): fill-reducing ordering through CliqueTrees.jl,
# user permutations, and the automatic AMD/ND choice by a cost model.
#
# AMD and MMD are always available (CliqueTrees + AMD.jl). Nested dissection
# needs Metis.jl: `ext/SparseDirectSolverMetisExt.jl` stores the function that
# builds the CliqueTrees algorithm object in `ND_PROVIDER` when Metis is loaded.

"""
    ND_PROVIDER

`Ref` holding `nothing` or a function `(nd_nlevels, nd_ubfactor, nd_nseps, nd_seed) -> alg` that
returns the CliqueTrees nested-dissection algorithm. Set by the Metis extension.
"""
const ND_PROVIDER = Ref{Any}(nothing)

"""
    nd_available() -> Bool

Whether nested dissection can be used (Metis.jl is loaded, which activates
`SparseDirectSolverMetisExt`).
"""
nd_available() = ND_PROVIDER[] !== nothing

function _nd_algorithm(opts::Options)
    nd_available() ||
        throw(NotSupportedError("nested dissection (reordering_alg = \"algo4\") needs Metis.jl; " *
                                "run `using Metis` to load SparseDirectSolverMetisExt"))
    return ND_PROVIDER[](opts.nd_nlevels, opts.nd_ubfactor, opts.nd_nseps, opts.nd_seed)
end

"""
    OrderingCandidate

Evaluation of one ordering by the cost model of [`compute_ordering`](@ref)
([`evaluate_ordering`](@ref)): fields `alg::Symbol`, `nnz_L::Int`, `flops::Float64`,
`nlevels::Int` (column-etree depth), `sdepth::Int` (supernodal schedule depth),
`nsupernodes::Int` (fundamental supernodes), `cost::Float64` ([`ordering_cost`](@ref)).
"""
const OrderingCandidate = @NamedTuple{alg::Symbol, nnz_L::Int, flops::Float64, nlevels::Int, sdepth::Int,
                                      nsupernodes::Int, cost::Float64}

"""
    Ordering

Result of [`compute_ordering`](@ref):

* `perm`, `iperm`: the fill-reducing permutation (`perm[k]` is the original index
  of the `k`-th pivot, so the factorized matrix is `A[perm, perm]`) and its inverse;
* `alg_used`: `:natural`, `:amd`, `:mmd`, `:nd` or `:user`;
* `stats::NamedTuple`: `nnz_L`, `flops`, `nlevels` (column-etree depth),
  `sdepth` (supernodal schedule depth), `nsupernodes`, `cost` of the chosen
  ordering, `candidates::Vector{OrderingCandidate}` (every ordering evaluated,
  the chosen one included), `auto::Bool` (whether the automatic choice ran),
  `level_flops` (the depth weight of [`ordering_cost`](@ref) used) and
  `nd_available::Bool`;
* `pairs::Vector{Tuple{Int,Int}}`: the 2×2 pivot candidate pairs `(partner,
  candidate)` the ordering was computed with (empty for no pair): `perm` puts
  each candidate right after its partner, and the symbolic factorization runs on
  [`factor_pattern`](@ref)`(P, ordering)`. `stats.pair_rounds` is the number of
  [`zero_pivot_pairs!`](@ref) rounds that added pairs for the chosen ordering (`0`
  without that pairing).
"""
struct Ordering
    perm::Vector{Int}
    iperm::Vector{Int}
    alg_used::Symbol
    stats::@NamedTuple{nnz_L::Int, flops::Float64, nlevels::Int, sdepth::Int, nsupernodes::Int, cost::Float64,
                       candidates::Vector{OrderingCandidate}, auto::Bool, level_flops::Float64,
                       nd_available::Bool, pair_rounds::Int}
    pairs::Vector{Tuple{Int, Int}}
end

Base.show(io::IO, o::Ordering) =
    print(io, "Ordering($(o.alg_used), n = $(length(o.perm)), nnz_L = $(o.stats.nnz_L), ",
          "flops = $(o.stats.flops), sdepth = $(o.stats.sdepth)",
          isempty(o.pairs) ? "" : ", npairs = $(length(o.pairs))", ")")

_flop_factor(::Type{T}) where {T} = T <: Complex ? 4.0 : 1.0

"""
    ORDERING_LEVEL_FLOPS

Default depth weight of [`ordering_cost`](@ref): flops charged per level of the
supernodal schedule, `1e8`. See [`ordering_cost`](@ref) for the model.
"""
const ORDERING_LEVEL_FLOPS = 1.0e8

"""
    ordering_cost(flops, sdepth, level_flops = ORDERING_LEVEL_FLOPS) -> Float64

Cost model of the automatic ordering choice (PLAN §2.3 step 2, issue #108):
the time of one factorization and its solves, in flops,

    cost = flops + level_flops × sdepth.

*Terms.* `flops` is the real-arithmetic Cholesky operation count of the
candidate ([`cholesky_flops`](@ref)), the throughput-bound part. `sdepth` is
the depth of its supernodal schedule ([`schedule_depth`](@ref)): the numeric
phase launches the fronts level by level and both solve sweeps walk the same
levels, so every level costs a latency (launches, synchronization, a tail of
few fronts) whatever its size, and the solve, nearly all latency, scales with
the depth alone. `level_flops` is that latency times the throughput.

*Why the supernodal schedule and not the column etree.* The column-etree depth
(`nlevels`) counts every column of a chain, and chains collapse into
supernodes, so it does not predict the schedule depth (issue #108, 78k-bus
condensed KKT: cuDSS's ordering 1279 column levels → 30 schedule levels,
METIS 1216 → 29, AMD a comparable column depth → 65). The fundamental
partition does not predict it either: on the random KKT generators it ranks
AMD 33 vs ND 63 where the amalgamated schedules both have 12 levels, and it
ranks AMD ahead of ND on `apache2` (48 vs 57) where the amalgamated schedule
ranks ND ahead (35 vs 39). So `sdepth` is the height of the amalgamated
supernodal tree, the `nlevels` of the [`Schedule`](@ref) the analysis builds.

*Weight.* The term is absolute, not relative to `n` (the previous model,
`flops × (1 + nlevels / n)`, weighed depth at 0.2% at `n ≈ 7e5` and was a flop
contest). `level_flops = 1e8` ([`ORDERING_LEVEL_FLOPS`](@ref)) is ~100 µs per
level at ~1 Tflop/s: one level of the level-batched factorization (several
launches) plus a forward and a backward sweep for each of a few solves per
factorization (refinement), as measured on the 78k-bus KKT in PR #107 (ND's
36 fewer levels: solve −36%, factorization −11 to −17% despite AMD's lower
flop count). Consequences: matrices below ~1e10 flops choose the shallower
schedule, unless the depths are within a few levels; 3-D problems, where ND
saves 10¹⁰ flops and more, choose the fewer flops. One weight serves every
backend, so that the CPU and GPU backends factor the same ordering. Ties keep
the first candidate (AMD).
"""
ordering_cost(flops::Real, sdepth::Integer, level_flops::Real = ORDERING_LEVEL_FLOPS) =
    Float64(flops) + Float64(level_flops) * sdepth

"""
    schedule_depth(parent, post, counts, amalgamation = DEFAULT_AMALGAMATION) -> (sdepth::Int, nsupernodes::Int)

Depth of the supernodal schedule of an ordering and its number of supernodes:
the [`fundamental_supernodes`](@ref) of the etree `parent` with postorder
`post` and column counts `counts` (etree numbering, from [`etree`](@ref),
[`postorder`](@ref), [`colcounts`](@ref)), relaxed by [`amalgamate`](@ref) with
the `amalgamation` limits (`nothing`: no amalgamation, `use_superpanels = 0`),
and the number of levels of their tree ([`tree_levels`](@ref)). It equals the
`nlevels` of the [`Schedule`](@ref) of the analysis of that ordering (outside
Schur complement mode).
"""
function schedule_depth(parent::AbstractVector{<:Integer}, post::AbstractVector{<:Integer},
                        counts::AbstractVector{<:Integer}, amalgamation = DEFAULT_AMALGAMATION)
    cp = fundamental_supernodes(parent, post, counts)
    amalgamation === nothing || (cp = amalgamate(cp, parent, counts, amalgamation))
    _, sdepth = tree_levels(cp.snparent)
    return sdepth, nsupernodes(cp)
end

# the amalgamation limits of the analysis (`nothing` without amalgamation)
_amalgamation(opts::Options) = opts.use_superpanels != 0 ? opts.amalgamation : nothing

"""
    evaluate_ordering(P::SymmetricPattern, perm; T = Float64, level_flops = ORDERING_LEVEL_FLOPS,
                      amalgamation = DEFAULT_AMALGAMATION) -> (; nnz_L, flops, nlevels, sdepth, nsupernodes, cost)

Elimination tree, column counts, column-etree height (`nlevels`) and
supernodal schedule depth and supernode count (`sdepth`, `nsupernodes`,
[`schedule_depth`](@ref) with `amalgamation`) of `A[perm, perm]`, condensed
into the cost of [`ordering_cost`](@ref). `flops` counts real operations (`4×`
for complex `T`); `cost` uses the real-arithmetic count whatever `T`, so the
element type does not change the choice.
"""
function evaluate_ordering(P::SymmetricPattern, perm::AbstractVector{<:Integer}; T::Type = Float64,
                           level_flops::Real = ORDERING_LEVEL_FLOPS, amalgamation = DEFAULT_AMALGAMATION)
    parent = etree(P, perm)
    post = postorder(parent)
    counts = colcounts(P, perm, parent, post)
    _, nlevels = tree_levels(parent)
    sdepth, nsn = schedule_depth(parent, post, counts, amalgamation)
    rflops = cholesky_flops(counts)
    return (nnz_L = nnz_L(counts), flops = rflops * _flop_factor(T), nlevels = nlevels, sdepth = sdepth,
            nsupernodes = nsn, cost = ordering_cost(rflops, sdepth, level_flops))
end

function _validate_user_perm(v::AbstractVector{<:Integer}, n::Integer)
    length(v) == n ||
        throw(InvalidValueError("user_perm has length $(length(v)), expected the matrix size $n"))
    p = Vector{Int}(v)
    isperm(p) && return p                       # 1-based (contains n, not 0)
    p .+= 1
    isperm(p) && return p                       # 0-based (contains 0, not n)
    throw(InvalidValueError("user_perm is not a permutation of 1:$n or 0:$(n - 1)"))
end

function _cliquetrees_perm(P::SymmetricPattern, alg)
    P.n == 0 && return Int[]
    order, _ = CliqueTrees.permutation(SparseMatrixCSC(P); alg)
    return Vector{Int}(order)
end

function _ordering_perm(P::SymmetricPattern, alg::Symbol, opts::Options)
    alg === :natural && return collect(1:P.n)
    alg === :amd && return _cliquetrees_perm(P, CliqueTrees.AMD())
    alg === :mmd && return _cliquetrees_perm(P, CliqueTrees.MMD())
    alg === :nd && return _cliquetrees_perm(P, _nd_algorithm(opts))
    throw(InvalidValueError("unknown ordering algorithm $(repr(alg)); expected :natural, :amd, :mmd, :nd or :auto"))
end

# Ordering requested by the options (user_perm first, then reordering_alg).
function _requested_alg(opts::Options)
    opts.user_perm !== nothing && return :user
    a = opts.reordering_alg
    a == REORDERING_NATURAL && return :natural
    a == REORDERING_AMD && return :amd
    a == REORDERING_ND && return :nd
    # COLAMD variants were warned about in `setparam!`; the symmetric-pattern path uses AMD
    (a == REORDERING_BTF_COLAMD || a == REORDERING_COLAMD) && return :amd
    return :auto
end

"""
    compute_ordering(P::SymmetricPattern, opts::Options; T = Float64, alg = nothing, pairs = [], candidates = nothing,
                     level_flops = ORDERING_LEVEL_FLOPS) -> Ordering

Fill-reducing ordering of the pattern `P` (PLAN §2.3 step 2):

* `opts.user_perm` set: that permutation, validated, accepted 0- or 1-based and
  returned 1-based (`alg_used = :user`);
* `reordering_alg = "algo5"`: natural ordering; `"algo3"`: AMD
  (`CliqueTrees.AMD()`); `"algo4"`: nested dissection with Metis (needs
  `using Metis`, otherwise [`NotSupportedError`](@ref)): `METIS_NodeND` with
  `ufactor = nd_ubfactor`, `nseps = nd_nseps` and `seed = nd_seed` (`-1`: METIS defaults; on
  fill-bound problems `nd_nseps = 4` trades a slower ordering for less fill);
  `nd_nlevels` is the cuDSS
  *minimum* number of dissection levels, which METIS' full recursion meets on
  every graph large enough to be split that often;
  `"algo1"`/`"algo2"`: AMD on the symmetric pattern;
* `"default"`: AMD and, if Metis is loaded, ND are both evaluated with
  [`evaluate_ordering`](@ref) and the one with the lower
  [`ordering_cost`](@ref) `flops + level_flops × sdepth` (supernodal schedule
  depth) is chosen; `stats.candidates` lists both with their schedule depths.

The keyword `alg` (`:natural`, `:amd`, `:mmd`, `:nd`, `:auto`) overrides
`reordering_alg` (MMD has no cuDSS spelling); `T` scales the flop counts in
`stats` (complex: `4×`) and does not change the choice. `level_flops` is the
depth weight of [`ordering_cost`](@ref); the schedule depth is that of the
amalgamation of `opts` (`opts.amalgamation`, none with `use_superpanels = 0`).

2×2 pivot pairs (`"S"`/`"H"`, issues #64 and #66), from [`analysis_pairs`](@ref)
in the analysis when [`pairs_enabled`](@ref):

* `pairs`: disjoint `(partner, candidate)` pairs fixed in advance
  (`pivot_pairs = "all"`, [`pivot_pairs`](@ref));
* `candidates`: a [`PairCandidates`](@ref) (`pivot_pairs = "default"`): for every
  ordering algorithm evaluated, the plain ordering is computed, the candidates
  whose pivot is structurally zero in it are paired ([`zero_pivot_pairs!`](@ref),
  partners accepted down to `opts.pivot_threshold` of the row maximum) and the
  graph compressed by the pairs is re-ordered, until a round adds no pair (at
  most [`PIVOT_PAIRS_MAX_ROUNDS`](@ref)). Each algorithm keeps its own pairs.
  The automatic choice takes the ordering with the fewest structurally zero
  pivots left ([`structural_zero_pivots`](@ref)) and compares the cost only among
  those: a zero pivot is perturbed in the factorization, an accuracy loss that no
  saving in flops or depth pays for (on the slack KKT generators ND's search
  stops with zero pivots left where AMD's has none).

Both come from the values of the matrix at analysis: an all-zero `nzval` gives
no pairs and an undefined one arbitrary pairs, so the analysis must run after the
first assembly of the values (MadNLP: after the first KKT assembly), for
`"default"` and `"all"` alike.

AMD, MMD and ND then order the graph compressed by the pairs
([`compressed_pattern`](@ref): a pair is one vertex with the union adjacency),
and the order is expanded with each partner right before its candidate
([`expand_permutation`](@ref)). Every ordering is evaluated, and the factor is
built, on [`pair_pattern`](@ref), where both columns of a pair have the union
structure, so the pair lies in one fundamental supernode (never split by
[`amalgamate`](@ref)) and in-front Bunch–Kaufman can take it as a 2×2 pivot.
With `user_perm` (and with the natural ordering) the pairs are not applied, the
user owns the order: `ordering.pairs` is empty. Without pairs (no candidate, or
no structurally zero pivot) the result is the one of the plain ordering, bitwise.
"""
function compute_ordering(P::SymmetricPattern, opts::Options; T::Type = Float64,
                          alg::Union{Nothing, Symbol} = nothing, pairs = Tuple{Int, Int}[],
                          candidates = nothing, level_flops::Real = ORDERING_LEVEL_FLOPS)
    requested = alg === nothing ? _requested_alg(opts) : alg
    (candidates === nothing || isempty(pairs)) ||
        throw(InvalidValueError("compute_ordering: pass either pairs or candidates, not both"))
    fixed = requested === :user || requested === :natural
    fixed_pairs = fixed ? Tuple{Int, Int}[] : Vector{Tuple{Int, Int}}(pairs)
    search = !fixed && candidates !== nothing      # pair the structurally zero pivots per algorithm
    # (alg, perm, pairs, rounds) of one ordering algorithm
    function order_on(a)
        search && return _pair_ordering(P, a, opts, candidates)
        isempty(fixed_pairs) && return (a, _ordering_perm(P, a, opts), fixed_pairs, 0)
        Pc, gptr, members = compressed_pattern(P, fixed_pairs)
        return (a, expand_permutation(_ordering_perm(Pc, a, opts), gptr, members), fixed_pairs, 0)
    end
    if requested === :user
        opts.user_perm === nothing && throw(InvalidValueError("alg = :user needs opts.user_perm"))
        perms = [(:user, _validate_user_perm(opts.user_perm, P.n), fixed_pairs, 0)]
    elseif requested === :auto
        algs = nd_available() ? (:amd, :nd) : (:amd,)
        perms = [order_on(a) for a in algs]
    else
        perms = [order_on(requested)]
    end
    evaluated = OrderingCandidate[]
    best = 0
    best_zeros = 0                                 # structurally zero pivots the pair search left in `best`
    Q = search ? nothing : pair_pattern(P, perms[1][3])   # the fixed pairs are shared by all algorithms
    for (k, (a, perm, prs, _)) in enumerate(perms)
        e = evaluate_ordering(search ? pair_pattern(P, prs) : Q, perm; T, level_flops, amalgamation = _amalgamation(opts))
        push!(evaluated, (alg = a, nnz_L = e.nnz_L, flops = e.flops, nlevels = e.nlevels, sdepth = e.sdepth,
                          nsupernodes = e.nsupernodes, cost = e.cost))
        zeros_left = search && length(perms) > 1 ? length(structural_zero_pivots(candidates, P, perm, prs)) : 0
        if best == 0 || zeros_left < best_zeros || (zeros_left == best_zeros && e.cost < evaluated[best].cost)
            best, best_zeros = k, zeros_left
        end
    end
    alg_used, perm, used_pairs, rounds = perms[best]
    c = evaluated[best]
    stats = (nnz_L = c.nnz_L, flops = c.flops, nlevels = c.nlevels, sdepth = c.sdepth, nsupernodes = c.nsupernodes,
             cost = c.cost, candidates = evaluated, auto = requested === :auto, level_flops = Float64(level_flops),
             nd_available = nd_available(), pair_rounds = rounds)
    return Ordering(perm, invperm(perm), alg_used, stats, used_pairs)
end

# ---------------------------------------------------------------------------
# Schur complement mode (PLAN §3.6)

"""
    schur_flags(indices, n) -> BitVector

The Schur set of `user_schur_indices` (`n` flags 0/1, `1` = the row and column
belong to the Schur complement) as a `BitVector`; raises
[`InvalidValueError`](@ref) when `indices` is missing, has the wrong length or
selects no index.
"""
function schur_flags(indices, n::Integer)
    indices === nothing &&
        throw(InvalidValueError("schur_mode = 1 needs \"user_schur_indices\" (n flags 0/1)"))
    length(indices) == n ||
        throw(InvalidValueError("user_schur_indices has $(length(indices)) entries, the matrix has $n rows"))
    flags = BitVector(x != 0 for x in indices)
    any(flags) || throw(InvalidValueError("user_schur_indices selects no row: the Schur complement would be empty"))
    return flags
end

"""
    induced_pattern(P::SymmetricPattern, vertices) -> SymmetricPattern

The pattern of `P` restricted to `vertices` (sorted, distinct), renumbered
`1:length(vertices)` in their order.
"""
function induced_pattern(P::SymmetricPattern, vertices::AbstractVector{<:Integer})
    loc = zeros(Int, P.n)
    for (k, v) in enumerate(vertices)
        loc[v] = k
    end
    colptr = Vector{Int}(undef, length(vertices) + 1)
    colptr[1] = 1
    rowval = Int[]
    for (k, v) in enumerate(vertices)
        for i in neighbors(P, v)
            loc[i] > 0 && push!(rowval, loc[i])
        end
        colptr[k + 1] = length(rowval) + 1
    end
    return SymmetricPattern(length(vertices), colptr, rowval)
end

"""
    compute_schur_ordering(P::SymmetricPattern, opts::Options, schur::AbstractVector{Bool}; T = Float64,
                           level_flops = ORDERING_LEVEL_FLOPS) -> Ordering

Schur-constrained ordering (PLAN §3.6): the rows and columns flagged in
`schur` ([`schur_flags`](@ref)) are ordered last, in increasing original
order, and the others first. The fill-reducing ordering of
[`compute_ordering`](@ref) (`reordering_alg`) runs on the pattern induced by the
other vertices ([`induced_pattern`](@ref)); since the Schur vertices come
last they are never intermediate vertices of a fill path, so this is the
ordering of the factored part. A `user_perm` keeps its relative order of the
other vertices. The 2×2 pivot pairs of `"S"`/`"H"` are not used (`pairs`
empty). `stats` describe the whole ordering (the Schur block included).
"""
function compute_schur_ordering(P::SymmetricPattern, opts::Options, schur::AbstractVector{Bool}; T::Type = Float64,
                                level_flops::Real = ORDERING_LEVEL_FLOPS)
    n = P.n
    length(schur) == n || throw(InvalidValueError("schur flags have length $(length(schur)), expected $n"))
    inner = findall(!, schur)
    outer = findall(schur)
    if opts.user_perm !== nothing
        up = _validate_user_perm(opts.user_perm, n)
        perm = [filter(i -> !schur[i], up); outer]
        alg_used = :user
        candidates = OrderingCandidate[]
        auto = false
    else
        o = compute_ordering(induced_pattern(P, inner), opts; T, level_flops)
        perm = [inner[o.perm]; outer]
        alg_used = o.alg_used
        candidates = o.stats.candidates
        auto = o.stats.auto
    end
    e = evaluate_ordering(P, perm; T, level_flops, amalgamation = _amalgamation(opts))
    stats = (nnz_L = e.nnz_L, flops = e.flops, nlevels = e.nlevels, sdepth = e.sdepth, nsupernodes = e.nsupernodes,
             cost = e.cost, candidates = candidates, auto = auto, level_flops = Float64(level_flops),
             nd_available = nd_available(), pair_rounds = 0)
    return Ordering(perm, invperm(perm), alg_used, stats, Tuple{Int, Int}[])
end

"""
    schur_pattern(P::SymmetricPattern, schur::AbstractVector{Bool}) -> (rowptr, colval)

Symbolic pattern of the Schur complement `S = A₂₂ − A₂₁ A₁₁⁻¹ A₁₂` of the
flagged rows and columns (`A₂₂`), in their increasing original order, as
1-based CSR arrays with sorted columns, both triangles and the whole diagonal:
`(i, j)` is in the pattern when `i == j`, when `(i, j)` is in `P`, or when
both are adjacent to one connected component of the graph of the other
vertices (a fill path through eliminated vertices only). `P` is the symmetric
pattern of the analysis (`A + Aᵀ` for `"G"`), so for LU the pattern may hold
entries that are numerically zero by structure (explicit zeros).
"""
function schur_pattern(P::SymmetricPattern, schur::AbstractVector{Bool})
    n = P.n
    loc = zeros(Int, n)
    ns = 0
    for v in 1:n
        schur[v] && (ns += 1; loc[v] = ns)
    end
    adj = [Int[k] for k in 1:ns]
    for v in 1:n
        loc[v] > 0 || continue
        for i in neighbors(P, v)
            loc[i] > 0 && push!(adj[loc[v]], loc[i])
        end
    end
    # connected components of the eliminated vertices: their Schur neighbours form a clique
    seen = falses(n)
    mark = zeros(Int, ns)
    stack = Int[]
    nbrs = Int[]
    ncomp = 0
    for v0 in 1:n
        (loc[v0] == 0 && !seen[v0]) || continue
        ncomp += 1
        empty!(nbrs)
        seen[v0] = true
        push!(stack, v0)
        while !isempty(stack)
            v = pop!(stack)
            for i in neighbors(P, v)
                if loc[i] > 0
                    if mark[loc[i]] != ncomp
                        mark[loc[i]] = ncomp
                        push!(nbrs, loc[i])
                    end
                elseif !seen[i]
                    seen[i] = true
                    push!(stack, i)
                end
            end
        end
        for a in nbrs, b in nbrs
            a == b || push!(adj[a], b)
        end
    end
    rowptr = Vector{Int}(undef, ns + 1)
    rowptr[1] = 1
    colval = Int[]
    for k in 1:ns
        cols = sort!(unique!(adj[k]))
        append!(colval, cols)
        rowptr[k + 1] = length(colval) + 1
    end
    return rowptr, colval
end
