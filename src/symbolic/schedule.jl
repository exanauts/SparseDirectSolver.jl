# Symbolic step 5 (PLAN §2.2, §2.3): the assembly tree (= supernodal tree) is
# split into the three regimes of the numeric phase and turned into a fixed
# sequence of launch groups.
#
# * Regime A: whole leaf subtrees whose serial multifrontal stack (packed
#   lower-triangular fronts and contribution blocks, in the subtree's processing
#   order) fits the local memory of a budget class; one workgroup per subtree,
#   one launch per budget class.
# * Regime B: the remaining fronts with `w ≤ regime_c_width` and
#   `f ≤ regime_c_rows`, binned by (width class, row class); one launch per
#   (level, chunk, bin).
# * Regime C: the wide/tall fronts (vendor dense calls, KA tiled fallback).
#
# B/C fronts are scheduled by their *schedule level*: the height above the
# regime-A subtrees (a B/C front whose children are all in regime A, or that has
# no children, is on schedule level 1). A schedule level is split into chunks
# when the contribution blocks it produces exceed `opts.memory_budget` bytes.
# Every (level, chunk) pair is a *step* `1:nsteps`; regime A is step 0.

"""
    REGIME_A, REGIME_B, REGIME_C

Regime codes (`Int8`) of [`Schedule`](@ref)`.regime`: fused subtree kernels,
fused per-front level kernels, vendor dense calls on large fronts (PLAN §2.2).
"""
const REGIME_A = Int8(1)
const REGIME_B = Int8(2)
const REGIME_C = Int8(3)

"Largest front width with a fused regime-B kernel; wider bins (raised `regime_c_width`) take the regime-C path."
const REGIME_B_MAX_WIDTH = 64

"""
`@localmem` sizes (bytes) of the regime-A kernel instances; a budget uses the largest one it holds.
Capped at 48 KiB: `@localmem` is static shared memory on CUDA, where ptxas rejects more than 48 KiB.
"""
const SUBTREE_LOCAL_SIZES = (8192, 16384, 32768, 49152)

"Bytes of a regime-A kernel's local memory kept for its control words and pivot (not for fronts)."
const SUBTREE_LOCAL_RESERVE = 256

"""
    subtree_local_bytes(budget) -> Int

`@localmem` bytes of the regime-A kernel of a budget class: the largest of
`SUBTREE_LOCAL_SIZES` (8, 16, 32, 48 KiB) that is `≤ budget`, `0` below
8 KiB. The size is a `Val` parameter of the kernel, so a short list of
instances serves every budget.
"""
subtree_local_bytes(budget::Integer) = foldl((acc, b) -> b <= budget ? b : acc, SUBTREE_LOCAL_SIZES; init = 0)

"""
    subtree_capacity(budget, elsize) -> Int

Entries of `elsize` bytes a regime-A subtree may keep in local memory under
`budget`: [`subtree_local_bytes`](@ref) minus `SUBTREE_LOCAL_RESERVE`, divided
by `elsize` (`0` when the budget is too small).
"""
subtree_capacity(budget::Integer, elsize::Integer) =
    max(subtree_local_bytes(budget) - SUBTREE_LOCAL_RESERVE, 0) ÷ Int(elsize)

"""
    packed_length(m) -> Int

Entries of the packed lower triangle of an `m×m` matrix, `m(m+1)/2`: the
storage of a contribution block on the update stack and of a regime-A front in
local memory (column-major packed, see `_packed`).
"""
packed_length(m::Integer) = Int(m) * (Int(m) + 1) ÷ 2

# position (1-based) of (i, j), i ≥ j, in the column-major packed lower triangle of an m×m matrix:
# column j starts after (j - 1)(2m - j + 2)/2 entries
@inline _packed(i, j, m) = (j - 1) * (2 * m - j + 2) ÷ 2 + i - j + 1

"""
    takes_c_path(schedule, s) -> Bool

Whether supernode `s` runs on the regime-C path of the numeric phase (vendor
or KA dense calls per front): regime C, or a regime-B bin wider than
`REGIME_B_MAX_WIDTH` (no fused kernel).
"""
function takes_c_path(sc, s::Integer)
    sc.regime[s] == REGIME_C && return true
    sc.regime[s] == REGIME_B || return false
    return sc.wclasses[(sc.bin[s] - 1) ÷ length(sc.fclasses) + 1] > REGIME_B_MAX_WIDTH
end

"""
    ScheduleGroup

One launch group of a [`Schedule`](@ref): `step` (0 for regime A), schedule
`level` and `chunk` (0 for regime A), `regime`, `class` (regime A: index into
`budgets`; regime B: bin id; regime C: 0) and the range `first:last` of
`Schedule.group_nodes` it covers. For regime A the range holds subtree ids, for
B and C supernode ids.
"""
struct ScheduleGroup
    step::Int
    level::Int
    chunk::Int
    regime::Int8
    class::Int
    first::Int
    last::Int
end

"""
    Schedule

Regime assignment and launch order of the supernodes of a
[`SupernodePartition`](@ref) for element type `T` (budgets are bytes):

* `width[s]`, `rows[s]`: front width `w` and number of rows `f`;
* `level[s]`: height of supernode `s` above the leaves of the supernodal tree
  (leaves 1, `level[parent] > level[child]`), `nlevels` the maximum;
* `regime[s]` ∈ (`REGIME_A`, `REGIME_B`, `REGIME_C`);
* `slevel[s]`: schedule level of a B/C front (height counting only B/C fronts),
  `0` in regime A; `nslevels` the maximum;
* `step[s]`: launch step of a B/C front (`1:nsteps`, one per (level, chunk)),
  `0` in regime A; `step_level[t]`, `step_chunk[t]` describe step `t`;
* `bin[s]`: regime-B bin id (`(wi - 1) * length(fclasses) + fi` for width class
  `wclasses[wi]` and row class `fclasses[fi]`), `0` otherwise;
* regime A subtrees: `subtree[s]` (`0` outside A), `subtree_ptr`/`subtree_nodes`
  (the nodes of subtree `t`, in processing order: a postorder whose children are
  visited by decreasing `peak - cb`, which minimizes the stack), `subtree_root`,
  `subtree_peak` (bytes of the serial stack of packed fronts and contribution
  blocks), `subtree_class` (index of the smallest budget of `budgets` whose
  [`subtree_capacity`](@ref) holds the peak);
* `groups` and `group_nodes`: the launch groups in execution order (regime A
  classes first, then step by step: B bins in increasing bin id, then C);
* `wclasses`, `fclasses`, `budgets`, `regime_c_width`, `regime_c_rows`,
  `memory_budget`, `elsize = sizeof(T)`, `vendor_c` (whether regime C uses vendor
  calls; `factorization_alg = "algo1"` turns them off).
"""
struct Schedule
    width::Vector{Int}
    rows::Vector{Int}
    level::Vector{Int}
    nlevels::Int
    regime::Vector{Int8}
    slevel::Vector{Int}
    nslevels::Int
    step::Vector{Int}
    nsteps::Int
    step_level::Vector{Int}
    step_chunk::Vector{Int}
    bin::Vector{Int}
    subtree::Vector{Int}
    subtree_ptr::Vector{Int}
    subtree_nodes::Vector{Int}
    subtree_root::Vector{Int}
    subtree_peak::Vector{Int}
    subtree_class::Vector{Int}
    groups::Vector{ScheduleGroup}
    group_nodes::Vector{Int}
    wclasses::Vector{Int}
    fclasses::Vector{Int}
    budgets::Vector{Int}
    regime_c_width::Int
    regime_c_rows::Int
    memory_budget::Int64
    elsize::Int
    vendor_c::Bool
end

"""
    nsubtrees(schedule) -> Int

Number of regime-A subtrees.
"""
nsubtrees(sc::Schedule) = length(sc.subtree_root)

Base.show(io::IO, sc::Schedule) =
    print(io, "Schedule(", length(sc.regime), " fronts: ", count(==(REGIME_A), sc.regime), " A in ",
          nsubtrees(sc), " subtrees, ", count(==(REGIME_B), sc.regime), " B, ", count(==(REGIME_C), sc.regime),
          " C; ", sc.nslevels, " schedule levels, ", sc.nsteps, " steps, ", nlaunches(sc), " launches)")

# powers of two from `lo` up to the first one ≥ `hi`
function _size_classes(lo::Int, hi::Int)
    c = [lo]
    while c[end] < hi
        push!(c, 2 * c[end])
    end
    return c
end

"""
    tree_height(parent) -> Vector{Int}

Height of every node of a forest given by `parent` (`0` = root) with
`parent[s] > s`: leaves 1, otherwise one more than the highest child.
"""
function tree_height(parent::AbstractVector{<:Integer})
    n = length(parent)
    h = ones(Int, n)
    for s in 1:n
        p = parent[s]
        p == 0 && continue
        p > s || throw(InvalidValueError("tree_height needs parent[s] > s, got parent[$s] = $p"))
        h[p] = max(h[p], h[s] + 1)
    end
    return h
end

# number of launches of a C front: potrf, plus trsm, syrk/herk and pack_add! when it has a contribution block
_c_front_launches(f::Int, w::Int) = f > w ? 4 : 1

"""
    nlaunches(schedule) -> Int

Number of kernel and vendor launches of one numeric factorization following
`schedule`: one per regime-A budget class, one per regime-B group (fused
assemble + factor + update kernel), and per regime-C group one batched
assembly/extend-add launch plus, per front, `potrf` and, when the front has a
contribution block, `trsm`, `syrk`/`herk` and the `pack_add!` of the
workspace product into the packed contribution block.
"""
function nlaunches(sc::Schedule)
    total = 0
    for g in sc.groups
        if g.regime == REGIME_C
            total += 1
            for k in g.first:g.last
                s = sc.group_nodes[k]
                total += _c_front_launches(sc.rows[s], sc.width[s])
            end
        else
            total += 1
        end
    end
    return total
end

"""
    front_flops(f, w) -> Int64

Multiply-add count of the partial dense factorization of an `f×f` front with
`w` pivot columns: `Σ_{j=f-w+1}^{f} j²` (the elimination of column `k` updates
the trailing `(f-k)×(f-k)` block). The work measure of the regime-A split.
"""
function front_flops(f::Integer, w::Integer)
    sq(n) = Int64(n) * (n + 1) * (2n + 1) ÷ 6
    return sq(f) - sq(f - w)
end

"""
    build_schedule(sp::SupernodePartition, opts::Options = Options(), ::Type{T} = Float64) -> Schedule

Regime assignment, binning, levels and launch groups (PLAN §2.3 step 5) for
the supernodes of `sp`, with byte budgets for the element type `T`:

1. regime C: fronts with `w > opts.regime_c_width` or `f > opts.regime_c_rows`
   (every non-A front with `factorization_alg = "algo2"`);
2. regime A: a front is eligible when it is not C, its children are eligible
   and its subtree's serial stack peak (packed lower-triangular `f×f` fronts
   plus the packed `m×m` contribution blocks waiting for their parent,
   `m = f - w`) fits the [`subtree_capacity`](@ref) of the largest of
   `opts.subtree_budgets`, and its subtree's flops ([`front_flops`](@ref))
   are at most `total ÷ opts.subtree_parallelism` (`0`: no flop limit), so
   that a large tree of small fronts is split into enough subtrees to fill
   the device instead of running on one workgroup; the maximal
   eligible subtrees are the regime-A
   subtrees, each with the smallest budget class that holds it (empty
   `subtree_budgets` disables regime A);
3. regime B: the rest, binned by the smallest width class in `8, 16, 32, 64, …`
   `≥ w` and row class in `64, 128, 256, 512, …` `≥ f` (up to the regime-C
   thresholds);
4. schedule levels of the B/C fronts, split into chunks whose produced
   (packed) contribution-block bytes stay `≤ opts.memory_budget` (a single front larger
   than the budget gets its own chunk; a negative budget means no chunking).
"""
function build_schedule(sp::SupernodePartition, opts::Options = Options(), ::Type{T} = Float64) where {T}
    ns = nsupernodes(sp)
    elsize = sizeof(T)
    cw, cr = opts.regime_c_width, opts.regime_c_rows
    budgets = sort(opts.subtree_budgets)
    capacity = [subtree_capacity(b, elsize) for b in budgets]
    maxcap = isempty(budgets) ? 0 : maximum(capacity)
    alg = opts.factorization_alg
    width = [snwidth(sp, s) for s in 1:ns]
    rows = [sp.rowptr[s + 1] - sp.rowptr[s] for s in 1:ns]
    cb = [packed_length(rows[s] - width[s]) for s in 1:ns]
    children = [Int[] for _ in 1:ns]
    for s in 1:ns
        sp.snparent[s] != 0 && push!(children[sp.snparent[s]], s)
    end
    level = tree_height(sp.snparent)
    # flops of every subtree; children precede parents in the supernode numbering
    work = [front_flops(rows[s], width[s]) for s in 1:ns]
    total = sum(work; init = Int64(0))
    for s in 1:ns
        p = sp.snparent[s]
        p != 0 && (work[p] += work[s])
    end
    par = opts.subtree_parallelism
    maxwork = par <= 0 ? typemax(Int64) : total ÷ par
    big = [width[s] > cw || rows[s] > cr for s in 1:ns]
    # 2. regime A: serial stack peak (entries) with children ordered by decreasing peak - cb (Liu)
    peak = zeros(Int, ns)
    eligible = falses(ns)
    for s in 1:ns
        kids = children[s]
        sort!(kids; by = c -> (-(peak[c] - cb[c]), c))
        acc = 0
        pk = 0
        for c in kids
            pk = max(pk, acc + peak[c])
            acc += cb[c]
        end
        peak[s] = max(pk, acc + packed_length(rows[s]))
        eligible[s] = !big[s] && all(c -> eligible[c], kids) && peak[s] <= maxcap && work[s] <= maxwork
    end
    regime = fill(REGIME_B, ns)
    subtree = zeros(Int, ns)
    subtree_ptr = Int[1]
    subtree_nodes = Int[]
    subtree_root = Int[]
    subtree_peak = Int[]
    subtree_class = Int[]
    for s in 1:ns
        eligible[s] || continue
        p = sp.snparent[s]
        (p == 0 || !eligible[p]) || continue
        push!(subtree_root, s)
        t = length(subtree_root)
        # postorder of the subtree, children in the (sorted) order of `children`
        stack = [(s, 1)]
        while !isempty(stack)
            v, i = pop!(stack)
            if i <= length(children[v])
                push!(stack, (v, i + 1))
                push!(stack, (children[v][i], 1))
            else
                push!(subtree_nodes, v)
                subtree[v] = t
                regime[v] = REGIME_A
            end
        end
        push!(subtree_ptr, length(subtree_nodes) + 1)
        push!(subtree_peak, peak[s] * elsize)
        push!(subtree_class, findfirst(>=(peak[s]), capacity))
    end
    for s in 1:ns
        regime[s] == REGIME_A && continue
        (big[s] || alg == FACTORIZATION_VENDOR) && (regime[s] = REGIME_C)
    end
    # 3. regime-B bins
    wclasses = _size_classes(8, max(cw, 8))
    fclasses = _size_classes(64, max(cr, 64))
    nf = length(fclasses)
    bin = zeros(Int, ns)
    for s in 1:ns
        if regime[s] == REGIME_B
            wi = findfirst(>=(width[s]), wclasses)
            fi = findfirst(>=(rows[s]), fclasses)
            bin[s] = (wi - 1) * nf + fi
        end
    end
    # 4. schedule levels and chunks
    slevel = zeros(Int, ns)
    for s in 1:ns
        regime[s] == REGIME_A && continue
        slevel[s] = 1 + maximum((slevel[c] for c in children[s]); init = 0)
    end
    nslevels = maximum(slevel; init = 0)
    bylevel = [Int[] for _ in 1:nslevels]
    for s in 1:ns
        slevel[s] > 0 && push!(bylevel[slevel[s]], s)
    end
    groups = ScheduleGroup[]
    group_nodes = Int[]
    for (ci, b) in enumerate(budgets)
        ts = [t for t in eachindex(subtree_root) if subtree_class[t] == ci]
        isempty(ts) && continue
        first = length(group_nodes) + 1
        append!(group_nodes, ts)
        push!(groups, ScheduleGroup(0, 0, 0, REGIME_A, ci, first, length(group_nodes)))
    end
    step = zeros(Int, ns)
    step_level = Int[]
    step_chunk = Int[]
    membudget = opts.memory_budget
    for l in 1:nslevels
        fronts = bylevel[l]
        # B fronts by bin, then C fronts; ties by supernode id (deterministic)
        sort!(fronts; by = s -> (regime[s], regime[s] == REGIME_B ? bin[s] : 0, s))
        chunks = Vector{Int}[]
        current = Int[]
        bytes = 0
        for s in fronts
            b = cb[s] * elsize
            if membudget >= 0 && !isempty(current) && bytes + b > membudget
                push!(chunks, current)
                current = Int[]
                bytes = 0
            end
            push!(current, s)
            bytes += b
        end
        isempty(current) || push!(chunks, current)
        for (k, chunk) in enumerate(chunks)
            push!(step_level, l)
            push!(step_chunk, k)
            t = length(step_level)
            i = 1
            while i <= length(chunk)
                s = chunk[i]
                j = i
                key = (regime[s], regime[s] == REGIME_B ? bin[s] : 0)
                while j < length(chunk) && (regime[chunk[j + 1]], regime[chunk[j + 1]] == REGIME_B ? bin[chunk[j + 1]] : 0) == key
                    j += 1
                end
                first = length(group_nodes) + 1
                for q in i:j
                    push!(group_nodes, chunk[q])
                    step[chunk[q]] = t
                end
                push!(groups, ScheduleGroup(t, l, k, key[1], key[2], first, length(group_nodes)))
                i = j + 1
            end
        end
    end
    return Schedule(width, rows, level, maximum(level; init = 0), regime, slevel, nslevels, step, length(step_level), step_level,
                    step_chunk, bin, subtree, subtree_ptr, subtree_nodes, subtree_root, subtree_peak, subtree_class,
                    groups, group_nodes, wclasses, fclasses, budgets, cw, cr, membudget, elsize,
                    alg != FACTORIZATION_VERY_SPARSE)
end
