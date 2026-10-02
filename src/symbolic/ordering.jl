# Symbolic step 2 (PLAN §2.3): fill-reducing ordering through CliqueTrees.jl,
# user permutations, and the automatic AMD/ND choice by a cost model.
#
# AMD and MMD are always available (CliqueTrees + AMD.jl). Nested dissection
# needs Metis.jl: `ext/SparseDirectSolverMetisExt.jl` stores the function that
# builds the CliqueTrees algorithm object in `ND_PROVIDER` when Metis is loaded.

"""
    ND_PROVIDER

`Ref` holding `nothing` or a function `(nd_nlevels, nd_ubfactor) -> alg` that
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
    return ND_PROVIDER[](opts.nd_nlevels, opts.nd_ubfactor)
end

"""
    OrderingCandidate

Evaluation of one ordering by the cost model of [`compute_ordering`](@ref):
fields `alg::Symbol`, `nnz_L::Int`, `flops::Float64`, `nlevels::Int`, `cost::Float64`.
"""
const OrderingCandidate = @NamedTuple{alg::Symbol, nnz_L::Int, flops::Float64, nlevels::Int, cost::Float64}

"""
    Ordering

Result of [`compute_ordering`](@ref):

* `perm`, `iperm`: the fill-reducing permutation (`perm[k]` is the original index
  of the `k`-th pivot, so the factorized matrix is `A[perm, perm]`) and its inverse;
* `alg_used`: `:natural`, `:amd`, `:mmd`, `:nd` or `:user`;
* `stats::NamedTuple`: `nnz_L`, `flops`, `nlevels`, `cost` of the chosen
  ordering, `candidates::Vector{OrderingCandidate}` (every ordering evaluated,
  the chosen one included), `auto::Bool` (whether the automatic choice ran) and
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
    stats::@NamedTuple{nnz_L::Int, flops::Float64, nlevels::Int, cost::Float64,
                       candidates::Vector{OrderingCandidate}, auto::Bool, nd_available::Bool, pair_rounds::Int}
    pairs::Vector{Tuple{Int, Int}}
end

Base.show(io::IO, o::Ordering) =
    print(io, "Ordering($(o.alg_used), n = $(length(o.perm)), nnz_L = $(o.stats.nnz_L), ",
          "flops = $(o.stats.flops), nlevels = $(o.stats.nlevels)",
          isempty(o.pairs) ? "" : ", npairs = $(length(o.pairs))", ")")

_flop_factor(::Type{T}) where {T} = T <: Complex ? 4.0 : 1.0

"""
    ordering_cost(flops, nlevels, n) -> Float64

Cost model of the automatic ordering choice (PLAN §2.3 step 2): predicted
flops weighted by the relative critical path, `flops × (1 + nlevels / n)`.
"""
ordering_cost(flops::Real, nlevels::Integer, n::Integer) = Float64(flops) * (1 + nlevels / max(n, 1))

"""
    evaluate_ordering(P::SymmetricPattern, perm; T = Float64) -> (; nnz_L, flops, nlevels, cost)

Elimination tree, column counts and tree height of `A[perm, perm]`, condensed
into the quantities of the cost model ([`ordering_cost`](@ref)). `flops` counts
real operations (`4×` for complex `T`).
"""
function evaluate_ordering(P::SymmetricPattern, perm::AbstractVector{<:Integer}; T::Type = Float64)
    parent = etree(P, perm)
    post = postorder(parent)
    counts = colcounts(P, perm, parent, post)
    _, nlevels = tree_levels(parent)
    flops = cholesky_flops(counts) * _flop_factor(T)
    return (nnz_L = nnz_L(counts), flops = flops, nlevels = nlevels, cost = ordering_cost(flops, nlevels, P.n))
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
    compute_ordering(P::SymmetricPattern, opts::Options; T = Float64, alg = nothing, pairs = [], candidates = nothing)
        -> Ordering

Fill-reducing ordering of the pattern `P` (PLAN §2.3 step 2):

* `opts.user_perm` set: that permutation, validated, accepted 0- or 1-based and
  returned 1-based (`alg_used = :user`);
* `reordering_alg = "algo5"`: natural ordering; `"algo3"`: AMD
  (`CliqueTrees.AMD()`); `"algo4"`: nested dissection with Metis (needs
  `using Metis`, otherwise [`NotSupportedError`](@ref)): `METIS_NodeND` with
  `ufactor = nd_ubfactor` (`-1`: METIS default); `nd_nlevels` is the cuDSS
  *minimum* number of dissection levels, which METIS' full recursion meets on
  every graph large enough to be split that often;
  `"algo1"`/`"algo2"`: AMD on the symmetric pattern;
* `"default"`: AMD and, if Metis is loaded, ND are both evaluated with
  [`evaluate_ordering`](@ref) and the one with the lower
  `flops × (1 + nlevels / n)` is chosen; `stats.candidates` lists both.

The keyword `alg` (`:natural`, `:amd`, `:mmd`, `:nd`, `:auto`) overrides
`reordering_alg` (MMD has no cuDSS spelling); `T` scales the flop counts in
`stats` (complex: `4×`) and does not change the choice.

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
                          candidates = nothing)
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
    for (k, (a, perm, prs, _)) in enumerate(perms)
        e = evaluate_ordering(pair_pattern(P, prs), perm; T)
        push!(evaluated, (alg = a, nnz_L = e.nnz_L, flops = e.flops, nlevels = e.nlevels, cost = e.cost))
        (best == 0 || e.cost < evaluated[best].cost) && (best = k)
    end
    alg_used, perm, used_pairs, rounds = perms[best]
    c = evaluated[best]
    stats = (nnz_L = c.nnz_L, flops = c.flops, nlevels = c.nlevels, cost = c.cost,
             candidates = evaluated, auto = requested === :auto, nd_available = nd_available(),
             pair_rounds = rounds)
    return Ordering(perm, invperm(perm), alg_used, stats, used_pairs)
end
