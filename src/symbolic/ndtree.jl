# Nested-dissection partition tree in the cuDSS encoding (PLAN §1.4, T24):
# `"nd_partition_tree"` (export) and `"user_nd_partition_tree"` (import, with
# `"user_perm"`), so an ordering can be cached between runs.
#
# Encoding (cuDSS `CUDSS_DATA_ND_PARTITION_TREE`): a complete binary tree with
# `k = nd_nlevels` levels stored as a flat array of `2^k - 1` sizes in level
# order from the bottom to the top: the `2^(k-1)` leaves first, the root last.
# Every node owns a contiguous range of columns of the permuted matrix; the
# columns of a node's subtree are its left subtree, its right subtree, then the
# node itself (the separator), so a parent is eliminated after both children.
#
# Host only, plain `Int` arrays.

"""
    ND_TREE_MAX_LEVELS

Largest `nd_nlevels` for which [`nd_partition_tree`](@ref) builds the tree
(`2^24 - 1` entries); larger values raise [`InvalidValueError`](@ref).
"""
const ND_TREE_MAX_LEVELS = 24

function _check_nd_levels(k::Integer)
    1 <= k <= ND_TREE_MAX_LEVELS ||
        throw(InvalidValueError("nd_nlevels = $k: the ND partition tree needs 1 ≤ nd_nlevels ≤ $ND_TREE_MAX_LEVELS"))
    return Int(k)
end

# flat (1-based) position of the node with heap index `h` (root 1, children 2h and 2h + 1) in a tree of `k` levels
@inline function _nd_flat(k::Int, h::Int)
    d = 8 * sizeof(Int) - 1 - leading_zeros(h)          # depth, root 0
    return (1 << k) - (1 << (d + 1)) + (h - (1 << d)) + 1
end

"""
    nd_partition_tree(sp::SupernodePartition, nlevels) -> Vector{Int}

The partition tree of the analysis `sp` in the cuDSS encoding with
`k = nlevels` levels (`2^k - 1` sizes, leaves first, root last), for the
permutation `sp.perm` (the `"perm_reorder_row"` of the analysis). It is read
off the supernodal elimination tree, whose numbering is a postorder: a node's
separator is the chain of supernodes at the top of its subtree forest down to
the first supernode with two or more children (empty when the forest already
has several roots), and the remaining subtrees are split into a left and a
right child at the root boundary that best balances their columns. A node at
the last level takes all the columns of its subtree forest. Nodes past the
bottom of the elimination tree have size `0`. The tree satisfies the
dependency rule that [`check_nd_partition_tree`](@ref) tests: every column's
etree parent lies in the column's node or in one of its ancestors. It exists for
every ordering, not only nested dissection.
"""
function nd_partition_tree(sp::SupernodePartition, nlevels::Integer)
    k = _check_nd_levels(nlevels)
    ns = nsupernodes(sp)
    first = collect(1:ns)                 # first supernode of the subtree of s (postorder: first[s]:s)
    sizes = ones(Int, ns)
    for s in 1:ns
        p = sp.snparent[s]
        p == 0 && continue
        p > s || throw(InvalidValueError("nd_partition_tree: the supernodal numbering is not topological"))
        first[p] = min(first[p], first[s])
        sizes[p] += sizes[s]
    end
    all(s -> s - first[s] + 1 == sizes[s], 1:ns) ||
        throw(InvalidValueError("nd_partition_tree: the supernodal numbering is not a postorder"))
    tree = zeros(Int, (1 << k) - 1)
    cols(lo, hi) = lo > hi ? 0 : sp.super_ptr[hi + 1] - sp.super_ptr[lo]
    # supernodes lo:hi (whole subtrees) -> node with heap index h
    function build!(lo, hi, h)
        idx = _nd_flat(k, h)
        if h >= (1 << (k - 1))                       # last level: the whole forest
            tree[idx] = cols(lo, hi)
            return
        end
        top = hi
        while hi >= lo && first[hi] == lo            # a single tree: its root joins the separator
            hi -= 1
        end
        tree[idx] = cols(hi + 1, top)
        m = lo                                       # left lo:(m - 1), right m:hi
        if hi >= lo                                  # two or more roots: split at a root boundary
            total = cols(lo, hi)
            best = typemax(Int)
            r = hi
            while r >= lo
                f = first[r]
                if f > lo
                    e = abs(2 * cols(lo, f - 1) - total)
                    e < best && (best = e; m = f)
                end
                r = f - 1
            end
        end
        build!(lo, m - 1, 2h)
        build!(m, hi, 2h + 1)
        return
    end
    build!(1, ns, 1)
    return tree
end

"""
    nd_tree_nodes(tree, n) -> node::Vector{Int}

The node of every column `1:n` of the permuted matrix under the partition tree
`tree` (cuDSS encoding, `length(tree) = 2^k - 1`), as a heap index (root `1`,
children of `h` at `2h` and `2h + 1`). Raises [`InvalidValueError`](@ref) when
the length is not `2^k - 1`, an entry is negative or the sizes do not add up to
`n`.
"""
function nd_tree_nodes(tree::AbstractVector{<:Integer}, n::Integer)
    len = length(tree)
    (len >= 1 && ispow2(len + 1)) ||
        throw(InvalidValueError("user_nd_partition_tree has $len entries, expected 2^k - 1 (k = nd_nlevels)"))
    k = trailing_zeros(len + 1)
    t = Vector{Int}(tree)
    all(>=(0), t) || throw(InvalidValueError("user_nd_partition_tree has a negative entry"))
    sum(t) == n ||
        throw(InvalidValueError("user_nd_partition_tree sizes add up to $(sum(t)), the matrix has $n columns"))
    nh = (1 << k) - 1
    total = zeros(Int, nh)                           # columns of the subtree of heap node h
    for h in nh:-1:1
        total[h] = t[_nd_flat(k, h)] + (2h <= nh ? total[2h] + total[2h + 1] : 0)
    end
    node = Vector{Int}(undef, n)
    start = zeros(Int, nh)                           # first column of the subtree of h
    start[1] = 1
    for h in 1:nh
        if 2h <= nh
            start[2h] = start[h]
            start[2h + 1] = start[h] + total[2h]
        end
        s0 = start[h] + total[h] - t[_nd_flat(k, h)]   # the node's own columns come last
        for j in s0:(start[h] + total[h] - 1)
            node[j] = h
        end
    end
    return node
end

"""
    check_nd_partition_tree(tree, nlevels, parent) -> nothing

Validate an imported partition tree (`"user_nd_partition_tree"`) against
`nd_nlevels = nlevels` and the elimination tree `parent` of the matrix permuted
by `"user_perm"` ([`etree`](@ref)): `2^nlevels - 1` non-negative sizes adding up
to `n` ([`nd_tree_nodes`](@ref)), and every column's etree parent in the
column's node or in an ancestor node, so that the dependencies follow the tree
(cuDSS: "dependencies between the subsets should correspond to the tree
structure"). Raises [`InvalidValueError`](@ref) otherwise.
"""
function check_nd_partition_tree(tree::AbstractVector{<:Integer}, nlevels::Integer, parent::AbstractVector{<:Integer})
    k = _check_nd_levels(nlevels)
    length(tree) == (1 << k) - 1 ||
        throw(InvalidValueError("user_nd_partition_tree has $(length(tree)) entries, expected 2^nd_nlevels - 1 = " *
                                "$((1 << k) - 1) (nd_nlevels = $k)"))
    node = nd_tree_nodes(tree, length(parent))
    for j in eachindex(parent)
        p = parent[j]
        p == 0 && continue
        a, b = node[p], node[j]
        while b > a
            b >>= 1
        end
        b == a || throw(InvalidValueError("user_nd_partition_tree does not match user_perm: column $j of the " *
                                          "permuted matrix updates column $p, which is not in its node or an " *
                                          "ancestor node"))
    end
    return nothing
end
