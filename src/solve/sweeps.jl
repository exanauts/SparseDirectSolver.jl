# Forward, diagonal and backward sweeps of the solve phase (PLAN §2.5, §3.4) on
# the permuted right-hand side `Y` (`n × nrhs`, supernodal numbering), with the
# schedule of the numeric phase:
#
# * regime A: one launch over every subtree, one workgroup per (subtree,
#   right-hand side) walking the subtree's supernodes in processing order
#   (forward) or in reverse (backward);
# * regime B: one launch per step, one workgroup per (front, right-hand side):
#   TRSV with `L11`, GEMV with `L21` (forward), gather + GEMV with `L21ᴴ`, TRSV
#   with `L11ᴴ` (backward);
# * regime C (the fronts that take the regime-C path in the numeric phase):
#   per front `trsm` and `gemm` on all right-hand sides through the dense
#   interface (vendor by default), plus a scatter / gather kernel.
#
# The right-hand sides are the second grid dimension of every kernel (flattened
# into the 1-D ndrange, see `_sv_front`).
#
# The forward sweep has write conflicts: the fronts of one launch update shared
# ancestor rows. Two variants:
#
# * atomic (default): the updates `Y[rows below] -= L21 y` go straight into `Y`,
#   with `Atomix.@atomic` where two workgroups can meet (regime B, and rows of a
#   regime-A subtree that lie above its root); regime-C fronts run one at a time
#   and scatter without atomics;
# * deterministic (`deterministic = true`, and whenever the backend has no
#   atomic add for `T`, which includes every complex `T`, issue #36): every front
#   writes its update to its own buffer `U` (rows `w+1:f` of its gather list,
#   `U[rowptr[s] + i - 1, r]`), and the owner pulls its children's buffers
#   through `relind` in `child_list` order, as the extend-add of the numeric
#   phase. No atomics; bitwise reproducible.
#
# The backward sweep only gathers from finished ancestor rows: conflict-free.
# The diagonal sweep is the identity for Cholesky (D is used by LDLᵀ, T15).
#
# As in the numeric kernels, per-node values sit in a `@localmem` control array
# written by work item 1 before a barrier, so every loop containing a barrier
# has a workgroup-uniform trip count that the KA CPU backend can evaluate.

"Workgroup size of the solve kernels."
const SOLVE_WORKGROUP = 64

# launch kinds of a `SolvePlan`
const SOLVE_SUBTREES = 1    # regime A: subtree ids
const SOLVE_FRONTS = 2      # regime B: one workgroup per front
const SOLVE_DENSE = 3       # regime-C path: dense calls per front

"""
    SolvePlan

Host launch plan of the sweeps, in forward order (the backward sweep runs it
in reverse): launch `k` has kind `kind[k]` (`SOLVE_SUBTREES`: the subtree ids
`group_nodes[first[k]:last[k]]`, all regime-A subtrees in one launch;
`SOLVE_FRONTS`: the regime-B fronts of one step; `SOLVE_DENSE`: the
regime-C-path fronts of one step, solved one after the other with dense calls).
`maxm` is the largest `m = f - w` of a regime-C-path front.
"""
struct SolvePlan
    kind::Vector{Int}
    first::Vector{Int}
    last::Vector{Int}
    maxm::Int
end

function SolvePlan(S::Symbolic)
    sc = S.schedule
    kind, first, last, step = Int[], Int[], Int[], Int[]
    maxm = 0
    for g in sc.groups
        if g.regime == REGIME_A
            k = SOLVE_SUBTREES
        else
            k = takes_c_path(sc, sc.group_nodes[g.first]) ? SOLVE_DENSE : SOLVE_FRONTS
            if k == SOLVE_DENSE
                for q in g.first:g.last
                    s = sc.group_nodes[q]
                    maxm = max(maxm, sc.rows[s] - sc.width[s])
                end
            end
        end
        # groups of the same kind and step are adjacent in `group_nodes`: one launch
        if !isempty(kind) && kind[end] == k && step[end] == g.step && last[end] + 1 == g.first
            last[end] = g.last
        else
            push!(kind, k)
            push!(first, g.first)
            push!(last, g.last)
            push!(step, g.step)
        end
    end
    return SolvePlan(kind, first, last, maxm)
end

"""
    SolveWorkspace{T, MT <: AbstractMatrix{T}}

Device storage of the solve phase for up to `nrhs` right-hand sides,
allocated once by [`allocate_solve`](@ref):

* `Y` (`n × nrhs`): the permuted right-hand side, overwritten by the solution;
* `U` (`length(rowval) × nrhs`): the per-front update buffers of the
  deterministic forward sweep (front `s` uses rows `rowptr[s] + w : rowptr[s+1] - 1`);
* `tmp` (`maxm × nrhs`): the `gemm` result of a regime-C-path front;
* `atomic`: whether the backend has an atomic add for `T`
  ([`capabilities`](@ref)); without one the forward sweep is always the
  deterministic variant;
* `plan`: the host [`SolvePlan`](@ref).
"""
struct SolveWorkspace{T, MT <: AbstractMatrix{T}}
    Y::MT
    U::MT
    tmp::MT
    atomic::Bool
    plan::SolvePlan
end

Base.eltype(::SolveWorkspace{T}) where {T} = T

"""
    max_rhs(ws::SolveWorkspace) -> Int

Number of right-hand sides `ws` was allocated for.
"""
max_rhs(ws::SolveWorkspace) = size(ws.Y, 2)

Base.show(io::IO, ws::SolveWorkspace{T, MT}) where {T, MT} =
    print(io, "SolveWorkspace{", T, ", ", nameof(MT), "}(n = ", size(ws.Y, 1), ", nrhs ≤ ", max_rhs(ws), ", ",
          ws.atomic ? "atomic" : "deterministic only", ")")

"""
    allocate_solve(symbolic, T, backend = CPU(), nrhs = 1) -> SolveWorkspace{T}

Allocate (zero-filled) the solve storage of [`SolveWorkspace`](@ref) for `nrhs`
right-hand sides of element type `T` on `backend`, build the
[`SolvePlan`](@ref) and look up the atomic capability once. This is the only
allocation of the solve phase.
"""
function allocate_solve(S::Symbolic, ::Type{T}, backend::KernelAbstractions.Backend = KernelAbstractions.CPU(),
                        nrhs::Integer = 1) where {T}
    nrhs >= 1 || throw(InvalidValueError("nrhs must be ≥ 1, got $nrhs"))
    plan = SolvePlan(S)
    Y = KernelAbstractions.zeros(backend, T, S.n, nrhs)
    U = KernelAbstractions.zeros(backend, T, length(S.partition.rowval), nrhs)
    tmp = KernelAbstractions.zeros(backend, T, plan.maxm, nrhs)
    atomic = capabilities(backend, T).atomic_add
    return SolveWorkspace{T, typeof(Y)}(Y, U, tmp, atomic, plan)
end

"""
    solve_memory(symbolic, T, nrhs = 1) -> Int64

Bytes of the [`SolveWorkspace`](@ref) of `nrhs` right-hand sides.
"""
solve_memory(S::Symbolic, ::Type{T}, nrhs::Integer = 1) where {T} =
    Int64(S.n + length(S.partition.rowval) + SolvePlan(S).maxm) * nrhs * sizeof(T)

# ---------------------------------------------------------------------------
# device helpers (work item `li` of `WG`, right-hand side `r`)

# slots of the control array of the solve kernels
const _SV_FIRST = 1      # first node: index into subtree_nodes (subtrees) or the list (fronts)
const _SV_COUNT = 2      # nodes walked by the workgroup
const _SV_LIMIT = 3      # atomic sweep: rows ≤ LIMIT belong to the workgroup alone (no atomics)
const _SV_NODE = 4       # current supernode v
const _SV_F = 5          # rows f
const _SV_W = 6          # columns w
const _SV_P0 = 7         # panel offset front_ptr[v]
const _SV_RP = 8         # gather-list offset rowptr[v]
const _SV_C0 = 9         # first column super_ptr[v]
const _SV_NCHILD = 10    # number of children
const _SV_CTL = 10

# workgroup `G` of a launch over `count` fronts (or subtrees) and the right-hand sides: front `g` of right-hand
# side `r`, G = g + count (r - 1) (the right-hand side is the second grid dimension, flattened into the 1-D
# ndrange: 2-D ndranges allocate on every launch of the KA CPU backend)
@inline _sv_front(G, count) = (G - 1) % count + 1
@inline _sv_rhs(G, count) = (G - 1) ÷ count + 1

# work item 1: workgroup `g` of a launch over `list[first:...]`
@inline function _sv_start!(ctl, g, list, first, subtree_ptr, subtree_nodes, super_ptr, ::Val{SUB}) where {SUB}
    @inbounds begin
        IT = eltype(ctl)
        e = list[first + g - 1]
        if SUB
            a = subtree_ptr[e]
            b = subtree_ptr[e + 1]
            root = subtree_nodes[b - 1]                   # processing order is a postorder: the root is last
            ctl[_SV_FIRST] = a % IT
            ctl[_SV_COUNT] = (b - a) % IT
            ctl[_SV_LIMIT] = (super_ptr[root + 1] - 1) % IT
        else
            ctl[_SV_FIRST] = (first + g - 1) % IT
            ctl[_SV_COUNT] = one(IT)
            ctl[_SV_LIMIT] = zero(IT)
        end
    end
    return nothing
end

# work item 1: load the descriptors of supernode v
@inline function _sv_setup!(ctl, v, super_ptr, rowptr, front_ptr, front_nrows, front_ncols, child_ptr)
    @inbounds begin
        IT = eltype(ctl)
        ctl[_SV_NODE] = v % IT
        ctl[_SV_F] = front_nrows[v] % IT
        ctl[_SV_W] = front_ncols[v] % IT
        ctl[_SV_P0] = front_ptr[v] % IT
        ctl[_SV_RP] = rowptr[v] % IT
        ctl[_SV_C0] = super_ptr[v] % IT
        ctl[_SV_NCHILD] = (child_ptr[v + 1] - child_ptr[v]) % IT
    end
    return nothing
end

# the k-th node of the workgroup: forward order, or reverse order for the backward sweep
@inline function _sv_node(ctl, k, list, subtree_nodes, ::Val{SUB}, ::Val{REV}) where {SUB, REV}
    @inbounds begin
        SUB || return list[ctl[_SV_FIRST]]
        return REV ? subtree_nodes[ctl[_SV_FIRST] + ctl[_SV_COUNT] - k] : subtree_nodes[ctl[_SV_FIRST] + k - 1]
    end
end

# deterministic forward: zero the update buffer of the front
@inline function _sv_zero_u!(U, ctl, r, li, ::Val{WG}) where {WG}
    @inbounds begin
        rp = ctl[_SV_RP] - 1
        z = zero(eltype(U))
        for i in (ctl[_SV_W] + li):WG:ctl[_SV_F]
            U[rp + i, r] = z
        end
    end
    return nothing
end

# deterministic forward: add the update buffer of the kc-th child (rows of one child are distinct)
@inline function _sv_pull!(Y, U, ctl, kc, r, child_ptr, child_list, rowptr, front_nrows, front_ncols, relind_ptr,
                           relind, li, ::Val{WG}) where {WG}
    @inbounds begin
        v = ctl[_SV_NODE]
        w = ctl[_SV_W]
        rp = ctl[_SV_RP] - 1
        c0 = ctl[_SV_C0] - 1
        c = child_list[child_ptr[v] + kc - 1]
        wc = front_ncols[c]
        mc = front_nrows[c] - wc
        src = rowptr[c] + wc - 1
        r0 = relind_ptr[c] - 1
        for k in li:WG:mc
            t = relind[r0 + k]
            x = U[src + k, r]
            if t <= w
                Y[c0 + t, r] += x
            else
                U[rp + t, r] += x
            end
        end
    end
    return nothing
end

# forward TRSV, column j: Y[c0 + k] -= L[k, j] y_j / L[j, j] for k > j (the division by the diagonal comes last)
@inline function _sv_fwd_col!(Y, factor, ctl, j, r, li, ::Val{WG}) where {WG}
    @inbounds begin
        f = ctl[_SV_F]
        c0 = ctl[_SV_C0] - 1
        col = ctl[_SV_P0] - 1 + (j - 1) * f
        yj = Y[c0 + j, r] / factor[col + j]
        for k in (j + li):WG:ctl[_SV_W]
            Y[c0 + k, r] -= factor[col + k] * yj
        end
    end
    return nothing
end

@inline function _sv_fwd_diag!(Y, factor, ctl, r, li, ::Val{WG}) where {WG}
    @inbounds begin
        f = ctl[_SV_F]
        c0 = ctl[_SV_C0] - 1
        p0 = ctl[_SV_P0] - 1
        for j in li:WG:ctl[_SV_W]
            Y[c0 + j, r] /= factor[p0 + (j - 1) * f + j]
        end
    end
    return nothing
end

# forward GEMV: t = L21[i, :] y for every row i below the diagonal block, subtracted from the row's
# update buffer (deterministic) or from Y (atomically above the workgroup's own rows)
@inline function _sv_fwd_update!(Y, U, factor, rowval, ctl, r, li, ::Val{DET}, ::Val{WG}) where {DET, WG}
    @inbounds begin
        f = ctl[_SV_F]
        w = ctl[_SV_W]
        c0 = ctl[_SV_C0] - 1
        p0 = ctl[_SV_P0] - 1
        rp = ctl[_SV_RP] - 1
        limit = ctl[_SV_LIMIT]
        for i in (w + li):WG:f
            t = zero(eltype(Y))
            for j in 1:w
                t += factor[p0 + (j - 1) * f + i] * Y[c0 + j, r]
            end
            if DET
                U[rp + i, r] -= t
            else
                row = rowval[rp + i]
                if row > limit
                    Atomix.@atomic Y[row + (r - 1) * size(Y, 1)] += -t
                else
                    Y[row, r] -= t
                end
            end
        end
    end
    return nothing
end

# backward gather + GEMV: y_j -= Σ_i conj(L21[i, j]) x[row_i] (the rows below belong to finished ancestors)
@inline function _sv_bwd_gather!(Y, factor, rowval, ctl, r, li, ::Val{WG}) where {WG}
    @inbounds begin
        f = ctl[_SV_F]
        w = ctl[_SV_W]
        c0 = ctl[_SV_C0] - 1
        p0 = ctl[_SV_P0] - 1
        rp = ctl[_SV_RP] - 1
        for j in li:WG:w
            col = p0 + (j - 1) * f
            t = zero(eltype(Y))
            for i in (w + 1):f
                t += conj(factor[col + i]) * Y[rowval[rp + i], r]
            end
            Y[c0 + j, r] -= t
        end
    end
    return nothing
end

# backward TRSV with L11ᴴ, step jj (column j = w - jj + 1): Y[c0 + k] -= conj(L[j, k]) x_j for k < j
@inline function _sv_bwd_col!(Y, factor, ctl, jj, r, li, ::Val{WG}) where {WG}
    @inbounds begin
        f = ctl[_SV_F]
        j = ctl[_SV_W] - jj + 1
        c0 = ctl[_SV_C0] - 1
        p0 = ctl[_SV_P0] - 1
        xj = Y[c0 + j, r] / conj(factor[p0 + (j - 1) * f + j])
        for k in li:WG:(j - 1)
            Y[c0 + k, r] -= conj(factor[p0 + (k - 1) * f + j]) * xj
        end
    end
    return nothing
end

@inline function _sv_bwd_diag!(Y, factor, ctl, r, li, ::Val{WG}) where {WG}
    @inbounds begin
        f = ctl[_SV_F]
        c0 = ctl[_SV_C0] - 1
        p0 = ctl[_SV_P0] - 1
        for j in li:WG:ctl[_SV_W]
            Y[c0 + j, r] /= conj(factor[p0 + (j - 1) * f + j])
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# kernels

"""
    solve_fwd_kernel!(backend, WG)(Y, U, factor, list, first, count, subtree_ptr, subtree_nodes, super_ptr,
                                   rowptr, rowval, front_ptr, front_nrows, front_ncols, child_ptr, child_list,
                                   relind_ptr, relind, Val(SUB), Val(DET), Val(WG); ndrange = WG * count * nrhs)

Forward sweep `L z = y` of right-hand side `r` by workgroup `G = g + count (r - 1)`
over its fronts: the subtree `list[first + g - 1]`, its
supernodes in processing order (`SUB = true`, regime A), or the single front
`list[first + g - 1]` (regime B). Per front: (deterministic, `DET`) zero its
update buffer and pull the children's; TRSV with `L11` on its columns of `Y`;
GEMV with `L21` into its update buffer or (atomic variant) into the rows of
`Y` below.
"""
@kernel function solve_fwd_kernel!(Y, U, factor, list, first, count, subtree_ptr, subtree_nodes, super_ptr, rowptr,
                                   rowval, front_ptr, front_nrows, front_ncols, child_ptr, child_list, relind_ptr,
                                   relind, ::Val{SUB}, ::Val{DET}, ::Val{WG}) where {SUB, DET, WG}
    @uniform IT = eltype(front_ptr)
    li = @index(Local, Linear)
    G = @index(Group, Linear)
    ctl = @localmem IT (_SV_CTL,)
    if li == 1
        _sv_start!(ctl, _sv_front(G, count), list, first, subtree_ptr, subtree_nodes, super_ptr, Val(SUB))
    end
    @synchronize
    for k in 1:ctl[_SV_COUNT]
        if li == 1
            v = _sv_node(ctl, k, list, subtree_nodes, Val(SUB), Val(false))
            _sv_setup!(ctl, v, super_ptr, rowptr, front_ptr, front_nrows, front_ncols, child_ptr)
        end
        @synchronize
        if DET
            _sv_zero_u!(U, ctl, _sv_rhs(G, count), li, Val(WG))
            @synchronize
            for kc in 1:ctl[_SV_NCHILD]
                _sv_pull!(Y, U, ctl, kc, _sv_rhs(G, count), child_ptr, child_list, rowptr, front_nrows, front_ncols,
                          relind_ptr, relind, li, Val(WG))
                @synchronize
            end
        end
        for j in 1:ctl[_SV_W]
            _sv_fwd_col!(Y, factor, ctl, j, _sv_rhs(G, count), li, Val(WG))
            @synchronize
        end
        _sv_fwd_diag!(Y, factor, ctl, _sv_rhs(G, count), li, Val(WG))
        @synchronize
        _sv_fwd_update!(Y, U, factor, rowval, ctl, _sv_rhs(G, count), li, Val(DET), Val(WG))
        @synchronize
    end
end

"""
    solve_bwd_kernel!(backend, WG)(Y, factor, list, first, count, subtree_ptr, subtree_nodes, super_ptr, rowptr,
                                   rowval, front_ptr, front_nrows, front_ncols, child_ptr, Val(SUB), Val(WG);
                                   ndrange = WG * count * nrhs)

Backward sweep `Lᴴ x = z` of right-hand side `r` by workgroup `G = g + count (r - 1)`
over its fronts (the subtree `list[first + g - 1]` in reverse processing order, `SUB = true`, or one
front): gather the finished rows below, GEMV with `L21ᴴ`, TRSV with `L11ᴴ`.
"""
@kernel function solve_bwd_kernel!(Y, factor, list, first, count, subtree_ptr, subtree_nodes, super_ptr, rowptr, rowval,
                                   front_ptr, front_nrows, front_ncols, child_ptr, ::Val{SUB},
                                   ::Val{WG}) where {SUB, WG}
    @uniform IT = eltype(front_ptr)
    li = @index(Local, Linear)
    G = @index(Group, Linear)
    ctl = @localmem IT (_SV_CTL,)
    if li == 1
        _sv_start!(ctl, _sv_front(G, count), list, first, subtree_ptr, subtree_nodes, super_ptr, Val(SUB))
    end
    @synchronize
    for k in 1:ctl[_SV_COUNT]
        if li == 1
            v = _sv_node(ctl, k, list, subtree_nodes, Val(SUB), Val(true))
            _sv_setup!(ctl, v, super_ptr, rowptr, front_ptr, front_nrows, front_ncols, child_ptr)
        end
        @synchronize
        _sv_bwd_gather!(Y, factor, rowval, ctl, _sv_rhs(G, count), li, Val(WG))
        @synchronize
        for jj in 1:ctl[_SV_W]
            _sv_bwd_col!(Y, factor, ctl, jj, _sv_rhs(G, count), li, Val(WG))
            @synchronize
        end
        _sv_bwd_diag!(Y, factor, ctl, _sv_rhs(G, count), li, Val(WG))
        @synchronize
    end
end

# deterministic forward of the regime-C path: zero the fronts' update buffers and pull their children's
@kernel function _sv_pull_kernel!(Y, U, list, first, count, rowptr, super_ptr, front_ptr, front_nrows, front_ncols,
                                  child_ptr, child_list, relind_ptr, relind, ::Val{WG}) where {WG}
    @uniform IT = eltype(front_ptr)
    li = @index(Local, Linear)
    G = @index(Group, Linear)
    ctl = @localmem IT (_SV_CTL,)
    if li == 1
        @inbounds v = list[first + _sv_front(G, count) - 1]
        _sv_setup!(ctl, v, super_ptr, rowptr, front_ptr, front_nrows, front_ncols, child_ptr)
    end
    @synchronize
    _sv_zero_u!(U, ctl, _sv_rhs(G, count), li, Val(WG))
    @synchronize
    for kc in 1:ctl[_SV_NCHILD]
        _sv_pull!(Y, U, ctl, kc, _sv_rhs(G, count), child_ptr, child_list, rowptr, front_nrows, front_ncols,
                  relind_ptr, relind, li, Val(WG))
        @synchronize
    end
end

# regime-C path, forward: subtract tmp (= L21 y) from the update buffer (DET) or from the rows of Y below
@kernel function _sv_scatter_kernel!(Y, U, tmp, rowval, rp, m, nrhs, ::Val{DET}) where {DET}
    q = @index(Global, Linear)
    k = (q - 1) % m + 1
    r = (q - 1) ÷ m + 1
    @inbounds if r <= nrhs
        if DET
            U[rp + k - 1, r] -= tmp[k, r]
        else
            Y[rowval[rp + k - 1], r] -= tmp[k, r]
        end
    end
end

# regime-C path, backward: gather the rows of Y below the diagonal block into tmp
@kernel function _sv_gather_kernel!(tmp, Y, rowval, rp, m, nrhs)
    q = @index(Global, Linear)
    k = (q - 1) % m + 1
    r = (q - 1) ÷ m + 1
    @inbounds if r <= nrhs
        tmp[k, r] = Y[rowval[rp + k - 1], r]
    end
end

# ---------------------------------------------------------------------------
# drivers

function _check_solve(ws::SolveWorkspace{T}, S::Symbolic, N::Numeric{T}, nrhs::Integer) where {T}
    _check_reference_cholesky(S, T)
    length(N.factor) == S.layout.factor_len ||
        throw(InvalidValueError("the numeric storage was not allocated for this analysis"))
    size(ws.Y, 1) == S.n && size(ws.U, 1) == length(S.partition.rowval) && size(ws.tmp, 1) == ws.plan.maxm ||
        throw(InvalidValueError("the solve workspace was not allocated for this analysis"))
    0 <= nrhs <= max_rhs(ws) ||
        throw(InvalidValueError("$nrhs right-hand sides, the solve workspace holds $(max_rhs(ws))"))
    backend = typeof(KernelAbstractions.get_backend(ws.Y))
    typeof(KernelAbstractions.get_backend(N.factor)) == backend &&
        typeof(KernelAbstractions.get_backend(S.rowval)) == backend ||
        throw(InvalidValueError("the solve workspace, the numeric storage and the device maps (adapt the Symbolic) " *
                                "must live on the same backend"))
    return nothing
end

# dense implementations of the regime-C-path fronts, resolved once per sweep (as `_front_impls`)
function _solve_impls(N::Numeric, S::Symbolic, impl::Symbol)
    impl === :auto && !S.schedule.vendor_c && (impl = :ka)
    return (trsm = select_impl(:trsm, N.factor, impl), gemm = select_impl(:gemm, N.factor, impl))
end

function _launch_fwd!(ws::SolveWorkspace, S::Symbolic, N::Numeric, first::Int, count::Int, nrhs::Int, sub::Val,
                      det::Val)
    WG = SOLVE_WORKGROUP
    solve_fwd_kernel!(KernelAbstractions.get_backend(ws.Y), WG)(
        ws.Y, ws.U, N.factor, S.group_nodes, first, count, S.subtree_ptr, S.subtree_nodes, S.super_ptr, S.rowptr,
        S.rowval, S.front_ptr, S.front_nrows, S.front_ncols, S.child_ptr, S.child_list, S.relind_ptr, S.relind, sub,
        det, Val(WG); ndrange = WG * count * nrhs)
    return nothing
end

function _launch_bwd!(ws::SolveWorkspace, S::Symbolic, N::Numeric, first::Int, count::Int, nrhs::Int, sub::Val)
    WG = SOLVE_WORKGROUP
    solve_bwd_kernel!(KernelAbstractions.get_backend(ws.Y), WG)(
        ws.Y, N.factor, S.group_nodes, first, count, S.subtree_ptr, S.subtree_nodes, S.super_ptr, S.rowptr,
        S.rowval, S.front_ptr, S.front_nrows, S.front_ncols, S.child_ptr, sub, Val(WG); ndrange = WG * count * nrhs)
    return nothing
end

# the flags as compile-time constants through explicit branches (closures over runtime flags allocate)
function _launch_sweep!(ws::SolveWorkspace, S::Symbolic, N::Numeric, first::Int, count::Int, nrhs::Int,
                        sub::Bool, det::Bool, forward::Bool)
    if !forward
        sub ? _launch_bwd!(ws, S, N, first, count, nrhs, Val(true)) :
              _launch_bwd!(ws, S, N, first, count, nrhs, Val(false))
    elseif sub
        det ? _launch_fwd!(ws, S, N, first, count, nrhs, Val(true), Val(true)) :
              _launch_fwd!(ws, S, N, first, count, nrhs, Val(true), Val(false))
    else
        det ? _launch_fwd!(ws, S, N, first, count, nrhs, Val(false), Val(true)) :
              _launch_fwd!(ws, S, N, first, count, nrhs, Val(false), Val(false))
    end
    return nothing
end

# the f×w panel of s and its blocks L11, L21, and s's columns of Y
function _dense_front(ws::SolveWorkspace, S::Symbolic, N::Numeric, s::Int, nrhs::Int)
    sc = S.schedule
    f, w = sc.rows[s], sc.width[s]
    p0 = S.layout.panel_ptr[s]
    c0 = S.partition.super_ptr[s]
    P = reshape(view(N.factor, p0:(p0 + f * w - 1)), f, w)
    return f, w, view(P, 1:w, 1:w), view(P, (w + 1):f, 1:w), view(ws.Y, c0:(c0 + w - 1), 1:nrhs)
end

function _fwd_dense!(ws::SolveWorkspace{T}, S::Symbolic, N::Numeric, s::Int, nrhs::Int, det::Bool,
                     p::NamedTuple) where {T}
    f, w, L11, L21, Yc = _dense_front(ws, S, N, s, nrhs)
    _trsm_impl!(p.trsm, 'L', 'L', 'N', 'N', one(T), L11, Yc)
    m = f - w
    m > 0 || return nothing
    Tm = view(ws.tmp, 1:m, 1:nrhs)
    _gemm_impl!(p.gemm, 'N', 'N', one(T), L21, Yc, zero(T), Tm)
    rp = S.partition.rowptr[s] + w
    kernel! = _sv_scatter_kernel!(KernelAbstractions.get_backend(ws.Y), SOLVE_WORKGROUP)
    if det
        kernel!(ws.Y, ws.U, ws.tmp, S.rowval, rp, m, nrhs, Val(true); ndrange = m * nrhs)
    else
        kernel!(ws.Y, ws.U, ws.tmp, S.rowval, rp, m, nrhs, Val(false); ndrange = m * nrhs)
    end
    return nothing
end

function _bwd_dense!(ws::SolveWorkspace{T}, S::Symbolic, N::Numeric, s::Int, nrhs::Int, p::NamedTuple) where {T}
    f, w, L11, L21, Yc = _dense_front(ws, S, N, s, nrhs)
    m = f - w
    if m > 0
        rp = S.partition.rowptr[s] + w
        kernel! = _sv_gather_kernel!(KernelAbstractions.get_backend(ws.Y), SOLVE_WORKGROUP)
        kernel!(ws.tmp, ws.Y, S.rowval, rp, m, nrhs; ndrange = m * nrhs)
        _gemm_impl!(p.gemm, 'C', 'N', -one(T), L21, view(ws.tmp, 1:m, 1:nrhs), one(T), Yc)
    end
    _trsm_impl!(p.trsm, 'L', 'L', 'C', 'N', one(T), L11, Yc)
    return nothing
end

"""
    forward_sweep!(ws, symbolic, numeric; nrhs = max_rhs(ws), deterministic = false, impl = :auto) -> ws

Forward sweep `L Z = Y` in place on the first `nrhs` columns of `ws.Y` (the
permuted right-hand side, [`permute_rhs!`](@ref)), with the factor of
[`factorize!`](@ref) in `numeric` and the device maps of `symbolic`: one launch
over all regime-A subtrees, then step by step one launch over the regime-B
fronts and, for each regime-C-path front, `trsm` + `gemm` through the dense
interface (`impl` as in [`factorize!`](@ref)) and a scatter kernel. The atomic
variant runs unless `deterministic` is set or the backend lacks an atomic add
for `T` (complex `T`, issue #36); the deterministic variant goes through the
per-front buffers `ws.U` and an owner-pull of the children (one extra launch
per regime-C step). Asynchronous; allocates nothing on the device.
"""
function forward_sweep!(ws::SolveWorkspace, S::Symbolic, N::Numeric; nrhs::Integer = max_rhs(ws),
                        deterministic::Bool = false, impl::Symbol = :auto)
    _check_solve(ws, S, N, nrhs)
    nrhs > 0 || return ws
    det = deterministic || !ws.atomic
    p = _solve_impls(N, S, impl)
    plan = ws.plan
    nodes = S.schedule.group_nodes
    for k in eachindex(plan.kind)
        a, b = plan.first[k], plan.last[k]
        if plan.kind[k] == SOLVE_DENSE
            if det
                WG = SOLVE_WORKGROUP
                _sv_pull_kernel!(KernelAbstractions.get_backend(ws.Y), WG)(
                    ws.Y, ws.U, S.group_nodes, a, b - a + 1, S.rowptr, S.super_ptr, S.front_ptr, S.front_nrows,
                    S.front_ncols, S.child_ptr, S.child_list, S.relind_ptr, S.relind, Val(WG);
                    ndrange = WG * (b - a + 1) * Int(nrhs))
            end
            for q in a:b
                _fwd_dense!(ws, S, N, nodes[q], Int(nrhs), det, p)
            end
        else
            _launch_sweep!(ws, S, N, a, b - a + 1, Int(nrhs), plan.kind[k] == SOLVE_SUBTREES, det, true)
        end
    end
    return ws
end

"""
    diagonal_sweep!(ws, symbolic, numeric; nrhs = max_rhs(ws)) -> ws

Diagonal step `D W = Z` between the sweeps: the identity for Cholesky
(`"SPD"`/`"HPD"`); LDLᵀ/LDLᴴ fill it in T15.
"""
function diagonal_sweep!(ws::SolveWorkspace, S::Symbolic, N::Numeric; nrhs::Integer = max_rhs(ws))
    _check_solve(ws, S, N, nrhs)
    return ws
end

"""
    backward_sweep!(ws, symbolic, numeric; nrhs = max_rhs(ws), impl = :auto) -> ws

Backward sweep `Lᴴ X = W` in place on the first `nrhs` columns of `ws.Y`, the
forward plan in reverse: per step the regime-C-path fronts (gather kernel,
`gemm` with `L21ᴴ`, `trsm` with `L11ᴴ`) and one launch over the regime-B
fronts, then one launch over all regime-A subtrees (supernodes in reverse
processing order). Gather-based and conflict-free: no atomics, deterministic.
Asynchronous.
"""
function backward_sweep!(ws::SolveWorkspace, S::Symbolic, N::Numeric; nrhs::Integer = max_rhs(ws),
                         impl::Symbol = :auto)
    _check_solve(ws, S, N, nrhs)
    nrhs > 0 || return ws
    p = _solve_impls(N, S, impl)
    plan = ws.plan
    nodes = S.schedule.group_nodes
    for k in reverse(eachindex(plan.kind))
        a, b = plan.first[k], plan.last[k]
        if plan.kind[k] == SOLVE_DENSE
            for q in b:-1:a
                _bwd_dense!(ws, S, N, nodes[q], Int(nrhs), p)
            end
        else
            _launch_sweep!(ws, S, N, a, b - a + 1, Int(nrhs), plan.kind[k] == SOLVE_SUBTREES, false, false)
        end
    end
    return ws
end

"""
    sweep_solve!(X, ws, symbolic, numeric, B; transposed = false, deterministic = false, impl = :auto) -> X

Solve `A X = B` with the factor `P A Pᵀ = L Lᴴ` of [`factorize!`](@ref):
[`permute_rhs!`](@ref) into `ws.Y`, [`forward_sweep!`](@ref),
[`diagonal_sweep!`](@ref), [`backward_sweep!`](@ref),
[`unpermute_solution!`](@ref) into `X`. `B` and `X` are device arrays in the
layouts of [`rhs_count`](@ref) (vector, `n × nrhs` matrix, strided vector;
row-major with `transposed = true`), with at most `max_rhs(ws)` right-hand
sides; `X === B` is allowed. Asynchronous; allocates nothing on the device.
"""
function sweep_solve!(X::AbstractVecOrMat, ws::SolveWorkspace, S::Symbolic, N::Numeric, B::AbstractVecOrMat;
                      transposed::Bool = false, deterministic::Bool = false, impl::Symbol = :auto)
    nrhs = rhs_count(B, S.n; transposed)
    rhs_count(X, S.n; transposed) == nrhs ||
        throw(DimensionMismatch("X has $(rhs_count(X, S.n; transposed)) right-hand sides, B has $nrhs"))
    _check_solve(ws, S, N, nrhs)
    permute_rhs!(ws.Y, B, S.perm; transposed)
    forward_sweep!(ws, S, N; nrhs, deterministic, impl)
    diagonal_sweep!(ws, S, N; nrhs)
    backward_sweep!(ws, S, N; nrhs, impl)
    unpermute_solution!(X, ws.Y, S.perm; transposed)
    return X
end
