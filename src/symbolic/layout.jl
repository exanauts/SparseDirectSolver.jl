# Symbolic steps 6 and 8 (PLAN §2.3): the static memory layout of the numeric
# phase. Offsets are 1-based element offsets (not bytes) into three device
# buffers, so they do not depend on the element type:
#
# * factor buffer: the panel of supernode `s` is the contiguous column-major
#   `f×w` block `panel_ptr[s]:(panel_ptr[s+1]-1)` (leading dimension `f`); the
#   upper triangle of its diagonal block is stored but unused;
# * D buffer (length `2n`): the diagonal of D for column `j` at `j`, the
#   subdiagonal of a 2×2 pivot starting at column `j` at `n + j`;
# * update stack: the contribution block (`m = f - w`) of a B/C front, or of the
#   root of a regime-A subtree, as the packed lower triangle of the `m×m` block
#   (column-major packed, `m(m+1)/2` entries, column `j` from offset
#   `(j-1)(2m-j+2)/2`, see `_packed`) at `cb_ptr[s]:(cb_ptr[s] + m(m+1)/2 - 1)`.
#   It lives from the step that produces it to the step of its parent (both
#   included); offsets come from a first-fit allocation over those intervals, so
#   contribution blocks live at the same time never overlap. Contribution blocks
#   inside a regime-A subtree stay in local memory (`cb_ptr = 0`), as do those of
#   roots of the tree (`m = 0`);
# * regime-C workspace: the full `m×m` result of the vendor `syrk`/`herk` of a
#   front on the regime-C path, packed and added to its block afterwards; one
#   buffer of the largest such `m^2`;
# * local memory of a regime-A subtree (per workgroup): the serial stack of the
#   subtree in processing order. The front of `v` (packed `f×f` lower triangle)
#   is placed on top of its children's contribution blocks; once factored, its
#   trailing `m×m` triangle (which is the packed contribution block) moves down
#   to where the first child's block started.

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
  (`t = 0:nsteps`); `stack_len = maximum(step_top)` is the high-water mark;
* `work_len`: entries of the regime-C `syrk` workspace (largest `m^2` of a
  front on the regime-C path, [`takes_c_path`](@ref), with a block on the stack);
* `local_front`, `local_cb` (regime A, `0` elsewhere): local-memory offset of the
  packed front of `s` and of its contribution block after the move (`0` for a
  subtree root, whose block goes to the update stack); `local_len[t]`: entries
  of the serial stack of subtree `t` (its peak).
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
    work_len::Int
    local_front::Vector{Int}
    local_cb::Vector{Int}
    local_len::Vector{Int}
end

Base.show(io::IO, L::Layout) =
    print(io, "Layout(factor ", L.factor_len, ", D ", L.d_len, ", update stack ", L.stack_len, ", workspace ",
          L.work_len, " entries)")

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
fronts and regime-A subtree roots that have a parent; packed lower triangles),
allocated first-fit over their lifetimes `[step(s), step(parent)]`, the
regime-C workspace, and the local-memory offsets of the regime-A subtrees
([`subtree_local_layout`](@ref)).
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
        cb_len[s] = packed_length(m)
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
    work_len = maximum((cb_len[s] > 0 && takes_c_path(sc, s) ? (sc.rows[s] - sc.width[s])^2 : 0 for s in 1:ns);
                       init = 0)
    local_front, local_cb, local_len = subtree_local_layout(sp, sc)
    return Layout(panel_ptr, panel_ptr[end] - 1, d_ptr, 2n, cb_ptr, cb_len, cb_first, cb_last, step_top,
                  maximum(step_top; init = 0), work_len, local_front, local_cb, local_len)
end

"""
    subtree_local_layout(sp, schedule) -> (local_front, local_cb, local_len)

Local-memory layout (1-based entry offsets) of the regime-A subtrees: a serial
stack in each subtree's processing order. The packed front of `v`
(`f(f+1)/2` entries) starts at `local_front[v]`, right above the contribution
blocks of its children (which are the top of the stack, contiguous); after the
factorization its packed `m(m+1)/2` contribution block (the trailing triangle
of the front) moves down to `local_cb[v]`, the start of the first child's
block, or of the front when it has no children. Subtree roots keep
`local_cb = 0` (their block goes to the update stack). `local_len[t]` is the
peak of subtree `t` in entries (`schedule.subtree_peak[t] / elsize`).
"""
function subtree_local_layout(sp::SupernodePartition, sc::Schedule)
    ns = nsupernodes(sp)
    local_front = zeros(Int, ns)
    local_cb = zeros(Int, ns)
    local_len = zeros(Int, nsubtrees(sc))
    for t in 1:nsubtrees(sc)
        stack = Int[]                                  # nodes whose blocks are live, bottom to top
        top = 0                                        # entries in use
        peak = 0
        for k in sc.subtree_ptr[t]:(sc.subtree_ptr[t + 1] - 1)
            v = sc.subtree_nodes[k]
            base = top
            while !isempty(stack) && sp.snparent[stack[end]] == v
                c = pop!(stack)
                base = local_cb[c] - 1
            end
            any(c -> sp.snparent[c] == v, stack) &&
                throw(InvalidValueError("subtree $t: the children of $v are not on top of the stack"))
            local_front[v] = top + 1
            peak = max(peak, top + packed_length(sc.rows[v]))
            m = sc.rows[v] - sc.width[v]
            if v == sc.subtree_root[t]
                top = base
            else
                local_cb[v] = base + 1
                top = base + packed_length(m)
                push!(stack, v)
            end
        end
        local_len[t] = peak
        peak * sc.elsize == sc.subtree_peak[t] ||
            throw(InvalidValueError("subtree $t: local stack peak $(peak * sc.elsize) B, schedule says $(sc.subtree_peak[t]) B"))
    end
    return local_front, local_cb, local_len
end
