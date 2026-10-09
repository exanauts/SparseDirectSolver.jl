# Symbolic step 3 (PLAN §2.3): elimination tree, postorder, column counts and
# the derived statistics (nnz(L), flops, tree height).
#
# All functions work in the *permuted* numbering: column `k` of the factor is
# original column `perm[k]`. Roots have parent 0.

"""
    etree(P::SymmetricPattern, perm) -> parent::Vector{Int}

Elimination tree of `A[perm, perm]` for the symmetric pattern `P` (Liu's
algorithm with path compression). `parent[k]` is the parent of column `k` of
the permuted matrix, `0` for a root; `parent[k] > k` for every non-root.
"""
function etree(P::SymmetricPattern, perm::AbstractVector{<:Integer})
    n = P.n
    length(perm) == n || throw(InvalidValueError("perm has length $(length(perm)), expected $n"))
    iperm = invperm(perm)
    parent = zeros(Int, n)
    ancestor = zeros(Int, n)
    for k in 1:n
        for r in neighbors(P, perm[k])
            i = iperm[r]
            i < k || continue
            # climb from i to the root of its current subtree, compressing the path to k
            while i != 0 && i < k
                inext = ancestor[i]
                ancestor[i] = k
                if inext == 0
                    parent[i] = k
                end
                i = inext
            end
        end
    end
    return parent
end

"""
    postorder(parent; key = nothing) -> post::Vector{Int}

A postorder of the forest `parent` (`0` = root): `post[k]` is the node visited
`k`-th; the roots and the children of every node are visited in increasing
index order, or in increasing `key` order when a vector `key` of distinct values
is given, and every node comes after all its descendants. Non-recursive.

With `key = perm` (the original column of every node of the etree of
`A[perm, perm]`) the visit sequence, in original columns, depends only on the
tree and not on which topological order `perm` lists it in: the supernode step
([`supernode_partition`](@ref)) uses it so that an analysis under its own
output permutation reproduces itself (T24).
"""
function postorder(parent::AbstractVector{<:Integer}; key::Union{Nothing, AbstractVector{<:Integer}} = nothing)
    n = length(parent)
    key === nothing || length(key) == n ||
        throw(InvalidValueError("postorder: key has length $(length(key)), expected $n"))
    seq = key === nothing ? (1:n) : sortperm(key)     # nodes in visiting order among siblings
    head = zeros(Int, n)     # first child
    next = zeros(Int, n)     # next sibling
    # insert in reverse so the child lists come out in increasing order
    for j in Iterators.reverse(seq)
        p = parent[j]
        p == 0 && continue
        next[j] = head[p]
        head[p] = j
    end
    post = Vector{Int}(undef, n)
    stack = Int[]
    k = 0
    for root in seq
        parent[root] == 0 || continue
        push!(stack, root)
        while !isempty(stack)
            p = stack[end]
            c = head[p]
            if c == 0
                pop!(stack)
                k += 1
                post[k] = p
            else
                head[p] = next[c]   # consume the child
                push!(stack, c)
            end
        end
    end
    k == n || throw(InvalidValueError("parent does not describe a forest (cycle detected)"))
    return post
end

# Gilbert–Ng–Peyton least-common-ancestor test of the skeleton matrix (CSparse `cs_leaf`).
@inline function _gnp_leaf(i, j, first, maxfirst, prevleaf, ancestor)
    (i <= j || first[j] <= maxfirst[i]) && return (0, 0)
    maxfirst[i] = first[j]
    jprev = prevleaf[i]
    prevleaf[i] = j
    jprev == 0 && return (i, 1)            # first leaf of row subtree i
    q = jprev
    while q != ancestor[q]
        q = ancestor[q]
    end
    s = jprev
    while s != q                           # path compression
        sparent = ancestor[s]
        ancestor[s] = q
        s = sparent
    end
    return (q, 2)                          # subsequent leaf; q is the lca
end

"""
    colcounts(P::SymmetricPattern, perm, parent, post) -> counts::Vector{Int}

Number of nonzeros in each column of the Cholesky factor `L` of `A[perm, perm]`
(diagonal included), by the Gilbert–Ng–Peyton algorithm in `O(nnz(A) α)`.
`parent` comes from [`etree`](@ref) and `post` from [`postorder`](@ref).
"""
function colcounts(P::SymmetricPattern, perm::AbstractVector{<:Integer}, parent::AbstractVector{<:Integer},
                   post::AbstractVector{<:Integer})
    n = P.n
    (length(perm) == n && length(parent) == n && length(post) == n) ||
        throw(InvalidValueError("perm, parent and post must have length $n"))
    iperm = invperm(perm)
    delta = zeros(Int, n)
    first = zeros(Int, n)
    maxfirst = zeros(Int, n)
    prevleaf = zeros(Int, n)
    ancestor = collect(1:n)
    # first[j]: postorder position of the first descendant of j; leaves get delta = 1
    for k in 1:n
        j = post[k]
        delta[j] = first[j] == 0 ? 1 : 0
        while j != 0 && first[j] == 0
            first[j] = k
            j = parent[j]
        end
    end
    for k in 1:n
        j = post[k]
        parent[j] != 0 && (delta[parent[j]] -= 1)
        for r in neighbors(P, perm[j])
            i = iperm[r]
            q, jleaf = _gnp_leaf(i, j, first, maxfirst, prevleaf, ancestor)
            jleaf >= 1 && (delta[j] += 1)
            jleaf == 2 && (delta[q] -= 1)
        end
        parent[j] != 0 && (ancestor[j] = parent[j])
    end
    # accumulate along the tree, children before parents
    counts = delta
    for j in post
        parent[j] != 0 && (counts[parent[j]] += counts[j])
    end
    return counts
end

"""
    nnz_L(counts) -> Int

Number of nonzeros of the Cholesky factor `L` (diagonal included): `sum(counts)`.
"""
nnz_L(counts::AbstractVector{<:Integer}) = Int(sum(counts; init = 0))

"""
    cholesky_flops(counts) -> Float64

Floating-point operation count of the Cholesky factorization with column counts
`counts` (diagonal included), `Σⱼ cⱼ²` as reported by CHOLMOD (one division
and `cⱼ - 1` multiply–adds per entry of the update, real arithmetic). Multiply by
4 for complex element types.
"""
cholesky_flops(counts::AbstractVector{<:Integer}) = sum(c -> Float64(c)^2, counts; init = 0.0)

"""
    tree_levels(parent) -> (height::Vector{Int}, nlevels::Int)

Height of every node of the forest `parent` (leaves have height 1, a node is
one above its highest child) and the number of levels, the maximum height
(0 for an empty forest). Nodes of equal height can be eliminated concurrently,
so `nlevels` is the length of the critical path in the level schedule.
"""
function tree_levels(parent::AbstractVector{<:Integer})
    n = length(parent)
    height = ones(Int, n)
    for j in postorder(parent)
        p = parent[j]
        p != 0 && (height[p] = max(height[p], height[j] + 1))
    end
    return height, (n == 0 ? 0 : maximum(height))
end
