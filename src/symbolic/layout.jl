# Symbolic steps 6 and 8 (PLAN §2.3): the static memory layout of the numeric
# phase. Offsets are 1-based element offsets (not bytes) into three device
# buffers, so they do not depend on the element type:
#
# * factor buffer: the panel of supernode `s` is the contiguous column-major
#   `f×w` block `panel_ptr[s]:(panel_ptr[s+1]-1)` (leading dimension `f`); the
#   upper triangle of its diagonal block is stored but unused;
# * D buffer (length `2n`): the diagonal of D for column `j` at `j`, the
#   subdiagonal of a 2×2 pivot starting at column `j` at `n + j`;
# * update stack: the `m×m` contribution block (`m = f - w`, column-major,
#   leading dimension `m`) of a B/C front, or of the root of a regime-A subtree,
#   at `cb_ptr[s]:(cb_ptr[s] + m^2 - 1)`. It lives from the step that produces it
#   to the step of its parent (both included); offsets come from a first-fit
#   allocation over those intervals, so contribution blocks live at the same time
#   never overlap. Contribution blocks inside a regime-A subtree stay in local
#   memory (`cb_ptr = 0`), as do those of roots of the tree (`m = 0`).

"""
    Layout

Static memory layout of the numeric phase (PLAN §2.3 steps 6 and 8), element
offsets, 1-based:

* `panel_ptr` (length `ns + 1`): panel of supernode `s` at
  `panel_ptr[s]:(panel_ptr[s+1]-1)`, column-major `f×w`; `factor_len` entries in all;
* `d_ptr` (length `ns + 1`): D entries of supernode `s` at `d_ptr[s]:(d_ptr[s+1]-1)`
  (its columns); the subdiagonal of 2×2 pivots at `n` more; `d_len = 2n`;
* `cb_ptr`, `cb_len`: contribution block of `s` on the update stack (`0`/`0` if
  it has none there); `cb_first`, `cb_last`: the steps it lives in (step 0 is
  regime A, see [`Schedule`](@ref));
* `step_top[t + 1]`: last update-stack entry in use during step `t`
  (`t = 0:nsteps`); `stack_len = maximum(step_top)` is the high-water mark.
"""
struct Layout
    panel_ptr::Vector{Int}
    factor_len::Int
    d_ptr::Vector{Int}
    d_len::Int
    cb_ptr::Vector{Int}
    cb_len::Vector{Int}
    cb_first::Vector{Int}
    cb_last::Vector{Int}
    step_top::Vector{Int}
    stack_len::Int
end

Base.show(io::IO, L::Layout) =
    print(io, "Layout(factor ", L.factor_len, ", D ", L.d_len, ", update stack ", L.stack_len, " entries)")

# first fit in a sorted free list of (offset, length) holes over an unbounded buffer
function _first_fit!(free::Vector{Tuple{Int, Int}}, top::Base.RefValue{Int}, len::Int)
    for (k, (off, l)) in enumerate(free)
        if l >= len
            if l == len
                deleteat!(free, k)
            else
                free[k] = (off + len, l - len)
            end
            return off
        end
    end
    # extend the buffer, reusing a hole that ends at the top
    if !isempty(free) && free[end][1] + free[end][2] == top[] + 1
        off = free[end][1]
        pop!(free)
    else
        off = top[] + 1
    end
    top[] = off + len - 1
    return off
end

function _release!(free::Vector{Tuple{Int, Int}}, off::Int, len::Int)
    k = searchsortedfirst(free, (off, len))
    insert!(free, k, (off, len))
    # merge with the neighbours
    if k < length(free) && free[k][1] + free[k][2] == free[k + 1][1]
        free[k] = (free[k][1], free[k][2] + free[k + 1][2])
        deleteat!(free, k + 1)
    end
    if k > 1 && free[k - 1][1] + free[k - 1][2] == free[k][1]
        free[k - 1] = (free[k - 1][1], free[k - 1][2] + free[k][2])
        deleteat!(free, k)
    end
    return nothing
end

"""
    build_layout(sp::SupernodePartition, schedule::Schedule) -> Layout

Panel offsets in supernode order, D offsets, and the update-stack offsets of
the contribution blocks that leave their front through global memory (B/C
fronts and regime-A subtree roots that have a parent), allocated first-fit over
their lifetimes `[step(s), step(parent)]`.
"""
function build_layout(sp::SupernodePartition, sc::Schedule)
    ns = nsupernodes(sp)
    n = sp.n
    panel_ptr = Vector{Int}(undef, ns + 1)
    panel_ptr[1] = 1
    for s in 1:ns
        panel_ptr[s + 1] = panel_ptr[s] + sc.rows[s] * sc.width[s]
    end
    d_ptr = copy(sp.super_ptr)
    # contribution blocks on the update stack and their lifetimes
    cb_ptr = zeros(Int, ns)
    cb_len = zeros(Int, ns)
    cb_first = zeros(Int, ns)
    cb_last = zeros(Int, ns)
    produced = [Int[] for _ in 0:sc.nsteps]
    for s in 1:ns
        p = sp.snparent[s]
        m = sc.rows[s] - sc.width[s]
        (p == 0 || m == 0) && continue
        sc.regime[s] == REGIME_A && sc.subtree_root[sc.subtree[s]] != s && continue
        cb_len[s] = m * m
        cb_first[s] = sc.step[s]
        cb_last[s] = sc.step[p]
        cb_last[s] > cb_first[s] ||
            throw(InvalidValueError("schedule: front $s (step $(cb_first[s])) is not before its parent (step $(cb_last[s]))"))
        push!(produced[cb_first[s] + 1], s)
    end
    # sweep the steps: release the blocks consumed by the previous step, then allocate
    released = [Int[] for _ in 0:(sc.nsteps + 1)]
    free = Tuple{Int, Int}[]
    top = Ref(0)
    step_top = zeros(Int, sc.nsteps + 1)
    live = Set{Int}()
    for t in 0:sc.nsteps
        if t > 0
            for s in released[t]
                _release!(free, cb_ptr[s], cb_len[s])
                delete!(live, s)
            end
        end
        for s in produced[t + 1]
            cb_ptr[s] = _first_fit!(free, top, cb_len[s])
            push!(live, s)
            push!(released[cb_last[s] + 1], s)          # free after its last step
        end
        step_top[t + 1] = maximum((cb_ptr[s] + cb_len[s] - 1 for s in live); init = 0)
    end
    return Layout(panel_ptr, panel_ptr[end] - 1, d_ptr, 2n, cb_ptr, cb_len, cb_first, cb_last, step_top,
                  maximum(step_top; init = 0))
end
