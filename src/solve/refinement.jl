# Iterative refinement of the solve phase (PLAN §2.5, §1.3 `ir_n_steps`,
# `ir_tol`): the residual `R = B - op(A) X` by a KA CSR SpMV over the
# full-pattern map of the stored matrix (`full_pattern_map`: the symmetric
# expansion of the triangle given by the view, mirrored entries conjugated for
# `"H"`/`"HPD"`), its column norms reduced on the device, and the correction
# `X += A⁻¹ R` through the solve sweeps. Everything runs in the input
# precision; the residual uses the solver's *current* values (`update!` without
# a refactorization refines towards the new matrix with the old factor).
#
# `op(A)` is `A`, `Aᵀ` or `Aᴴ` (`solve_mode`) of the user matrix, which is the
# stored CSR matrix `M` or, for a CSC input, `Mᵀ`. For symmetric or Hermitian
# `M` every `op(A)` is `M` or `conj(M)`; `conj(M) x = b` is solved as
# `x = conj(M⁻¹ conj(b))` ([`solve_conjugated`](@ref)). For a general `M`
# (`"G"`, LU) `op(A)` is `M`, `Mᵀ` or `conj(Mᵀ)` ([`solve_transposed`](@ref)):
# the refinement map then also holds the rows of `Mᵀ` (`transpose_matrix`
# selects them) and the sweeps solve with the transposed factors.
#
# User arrays come in the layouts of `src/solve/permute.jl` (vector, matrix,
# strided vector, row-major when transposed); the residual `R` is a column-major
# `n × nrhs` device matrix in the original numbering. Uniform batch: `R` holds
# the compact columns of the active members (as the solve workspace), each
# residual uses its member's values (`nzval` member after member).

"Workgroup size of the residual-norm reduction (one workgroup per right-hand side)."
const REFINE_WORKGROUP = 256

"""
    RefinementWorkspace{T, R, VI, MT, VR}

Device storage of iterative refinement, allocated at the first solve that
refines ([`allocate_refinement`](@ref)) and grown with the number of right-hand
sides:

* `rowptr`, `colval`, `src` (`INT` vectors): the full matrix `M` as a CSR over
  the contributions of the user's `nzval` (`src[c] > 0`: `nzval[src[c]]`,
  `src[c] < 0`: `conj(nzval[-src[c]])`), from [`refinement_map`](@ref); for
  `"G"` the rows `n+1:2n` of `rowptr` are the rows of `Mᵀ`;
* `R` (`n × nrhs`): the residual, then the correction;
* `Bc` (`n × nrhs`): a copy of the right-hand side when `X` and `B` alias;
* `norms` (`2 nrhs`, real): `‖Rₖ‖₂²` and `‖Bₖ‖₂²` per right-hand side, and
  `norms_host`, its host copy (the early-exit test of `ir_tol`);
* `krylov`: the storage of FGMRES-IR (`ir_mode = "fgmres"`), owned by the
  Krylov.jl extension (`nothing` until its first solve).
"""
struct RefinementWorkspace{T, R <: Real, VI <: AbstractVector, MT <: AbstractMatrix{T}, VR <: AbstractVector{R}}
    rowptr::VI
    colval::VI
    src::VI
    R::MT
    Bc::MT
    norms::VR
    norms_host::Vector{R}
    krylov::Base.RefValue{Any}
end

max_rhs(W::RefinementWorkspace) = size(W.R, 2)

"""
    refinement_map(F::FullPatternMap; transpose = false) -> (rowptr, colval, src)

Host CSR of the full matrix of `F` over its contributions (duplicates are not
merged; their products are summed by the SpMV): row `i` holds the entries
`rowptr[i]:rowptr[i+1]-1`, column `colval[c]`, value `nzval[src[c]]`, conjugated
when `src[c] < 0`. 1-based. With `transpose = true` (structure `"G"`) the rows
of `Mᵀ` follow as rows `n+1:2n` (`rowptr` has `2n + 1` entries).
"""
function refinement_map(F::FullPatternMap; transpose::Bool = false)
    n = F.n
    rowptr = Vector{Int}(undef, (transpose ? 2n : n) + 1)
    rowptr[1] = 1
    colval = Int[]
    src = Int[]
    for i in 1:n
        for e in F.rowptr[i]:(F.rowptr[i + 1] - 1), s in F.srcptr[e]:(F.srcptr[e + 1] - 1)
            push!(colval, F.colval[e])
            push!(src, F.conjflag[s] ? -F.src[s] : F.src[s])
        end
        rowptr[i + 1] = length(colval) + 1
    end
    if transpose
        # the entries of M by column: row j of Mᵀ holds the entries (i, j) of M, rows i in order
        cols = [Tuple{Int, Int}[] for _ in 1:n]
        for i in 1:n, e in F.rowptr[i]:(F.rowptr[i + 1] - 1), s in F.srcptr[e]:(F.srcptr[e + 1] - 1)
            push!(cols[F.colval[e]], (i, F.conjflag[s] ? -F.src[s] : F.src[s]))
        end
        for j in 1:n
            for (i, v) in cols[j]
                push!(colval, i)
                push!(src, v)
            end
            rowptr[n + j + 1] = length(colval) + 1
        end
    end
    return rowptr, colval, src
end

"""
    allocate_refinement(map, T, INT, backend, nrhs; n = rows of map) -> RefinementWorkspace

Move the host [`refinement_map`](@ref) `map` to `backend` as `INT` vectors
(overflow-checked) and allocate the residual storage for `nrhs` right-hand
sides of element type `T` and `n` rows (pass `n` for a `"G"` map, which holds `2n` rows).
"""
function allocate_refinement(map::NTuple{3, Vector{Int}}, ::Type{T}, ::Type{INT},
                             backend::KernelAbstractions.Backend, nrhs::Integer;
                             n::Integer = length(map[1]) - 1) where {T, INT}
    rowptr, colval, src = map
    dev(v) = (v_ = _to_index_type(INT, v); d = KernelAbstractions.allocate(backend, INT, length(v_)); copyto!(d, v_); d)
    R = KernelAbstractions.zeros(backend, T, n, nrhs)
    Bc = KernelAbstractions.zeros(backend, T, n, nrhs)
    norms = KernelAbstractions.zeros(backend, real(T), 2 * nrhs)
    return RefinementWorkspace{T, real(T), typeof(dev(Int[])), typeof(R), typeof(norms)}(
        dev(rowptr), dev(colval), dev(src), R, Bc, norms, zeros(real(T), 2 * nrhs), Ref{Any}(nothing))
end

function _to_index_type(::Type{INT}, v::Vector{Int}) where {INT}
    all(x -> typemin(INT) <= x <= typemax(INT), v) ||
        throw(InvalidValueError("the refinement map does not fit index type $INT"))
    return convert(Vector{INT}, v)
end

# a fresh workspace for `nrhs` right-hand sides that reuses the device map of `W`
function _grow_refinement(W::RefinementWorkspace{T, R, VI, MT, VR}, nrhs::Integer) where {T, R, VI, MT, VR}
    backend = KernelAbstractions.get_backend(W.R)
    n = size(W.R, 1)
    Rm = KernelAbstractions.zeros(backend, T, n, nrhs)
    Bc = KernelAbstractions.zeros(backend, T, n, nrhs)
    norms = KernelAbstractions.zeros(backend, R, 2 * nrhs)
    return RefinementWorkspace{T, R, VI, MT, VR}(W.rowptr, W.colval, W.src, Rm, Bc, norms, zeros(R, 2 * nrhs),
                                            Ref{Any}(nothing))
end

# compact column r (user column `_bm_ucol(bm, r)` of the `nrhs` user columns, values of member
# `_bm_cmember(bm, r)`)
# (`roff = n`: the rows of Mᵀ of a "G" map)
@kernel function _residual_kernel!(R, rowptr, colval, src, nzval_all, X, B, n, nrhs, ncols, bm, roff, ::Val{XT},
                                   ::Val{BT}, ::Val{CJ}) where {XT, BT, CJ}
    q = @index(Global, Linear)
    i = (q - 1) % n + 1
    r = (q - 1) ÷ n + 1
    @inbounds if r <= ncols
        nzval = _mview(nzval_all, _bm_cmember(bm, r), bm.nbatch)
        u = _bm_ucol(bm, r)
        acc = zero(eltype(R))
        for c in Int(rowptr[roff + i]):(Int(rowptr[roff + i + 1]) - 1)
            s = Int(src[c])
            v = nzval[abs(s)]
            v = xor(s < 0, CJ) ? conj(v) : v
            acc += v * _rhs_get(X, Int(colval[c]), u, n, nrhs, Val(XT))
        end
        R[i, r] = _rhs_get(B, i, u, n, nrhs, Val(BT)) - acc
    end
end

@kernel function _residual_norms_kernel!(norms, R, B, n, nrhs, bm, ::Val{BT}, ::Val{WG},
                                         ::Val{LOG2WG}) where {BT, WG, LOG2WG}
    @uniform RT = eltype(norms)
    li = @index(Local, Linear)
    r = @index(Group, Linear)
    nr = @localmem RT (WG,)
    nb = @localmem RT (WG,)
    @inbounds begin
        u = _bm_ucol(bm, r)
        sr = zero(RT)
        sb = zero(RT)
        for i in li:WG:n
            sr += abs2(R[i, r])
            sb += abs2(_rhs_get(B, i, u, n, nrhs, Val(BT)))
        end
        nr[li] = sr
        nb[li] = sb
    end
    @synchronize
    for lev in 1:LOG2WG
        @inbounds begin
            h = WG >> lev
            if li <= h
                nr[li] += nr[li + h]
                nb[li] += nb[li + h]
            end
        end
        @synchronize
    end
    if li == 1
        @inbounds norms[2 * r - 1] = nr[1]
        @inbounds norms[2 * r] = nb[1]
    end
end

@kernel function _add_correction_kernel!(X, Y, perm, scale, n, nrhs, ncols, bm, ::Val{TR}, ::Val{CJ}) where {TR, CJ}
    q = @index(Global, Linear)
    k = (q - 1) % n + 1
    r = (q - 1) ÷ n + 1
    @inbounds if r <= ncols
        i = Int(perm[k])
        u = _bm_ucol(bm, r)
        y = _scaled(Y[k, r], scale, i)
        _rhs_set!(X, _rhs_get(X, i, u, n, nrhs, Val(TR)) + (CJ ? conj(y) : y), i, u, n, nrhs, Val(TR))
    end
end

@kernel function _copy_rhs_kernel!(C, B, n, nrhs, ::Val{TR}) where {TR}
    q = @index(Global, Linear)
    i = (q - 1) % n + 1
    r = (q - 1) ÷ n + 1
    @inbounds if r <= nrhs
        C[i, r] = _rhs_get(B, i, r, n, nrhs, Val(TR))
    end
end

# the batch map of `nrhs` user columns: given, or every column in order
_user_map(bm::BatchMap, nrhs) = bm
_user_map(::Nothing, nrhs) = single_batch(; nrhs)

"""
    residual!(W, nzval, X, B; nrhs, transposed = false, b_transposed = transposed, conjugate = false,
              bm = nothing, transpose_matrix = false) -> W.R

`W.R[:, 1:nrhs] = B - M X` (`conj(M)` when `conjugate`; `Mᵀ` when
`transpose_matrix`, which needs a `"G"` map) with the full matrix
`M` of the map of `W` and the user values `nzval`; `X`, `B` in the layouts of
[`rhs_count`](@ref) (row-major when `transposed`/`b_transposed`) with `nrhs`
right-hand sides. Uniform batch ([`BatchMap`](@ref) `bm`): `nzval` holds the
values of the `bm.nbatch` members, and the residuals of the active members go
to the compact columns `1:(bm.nrhs bm.nact)` of `W.R`. One launch,
one work item per (row, right-hand side), gather-based (no atomics).
Asynchronous.
"""
function residual!(W::RefinementWorkspace, nzval::AbstractVector, X::AbstractVecOrMat, B::AbstractVecOrMat;
                   nrhs::Integer, transposed::Bool = false, b_transposed::Bool = transposed,
                   conjugate::Bool = false, bm::Union{Nothing, BatchMap} = nothing, transpose_matrix::Bool = false)
    n = size(W.R, 1)
    bm = _user_map(bm, nrhs)
    ncols = bm.nrhs * bm.nact
    n * ncols > 0 || return W.R
    roff = _map_row_offset(W, transpose_matrix)
    kernel! = _residual_kernel!(KernelAbstractions.get_backend(W.R), PERMUTE_WORKGROUP)
    _with_flags(transposed, b_transposed) do xt, bt
        if conjugate
            kernel!(W.R, W.rowptr, W.colval, W.src, nzval, X, B, n, Int(nrhs), ncols, bm, roff, xt, bt, Val(true);
                    ndrange = n * ncols)
        else
            kernel!(W.R, W.rowptr, W.colval, W.src, nzval, X, B, n, Int(nrhs), ncols, bm, roff, xt, bt, Val(false);
                    ndrange = n * ncols)
        end
    end
    return W.R
end

"""
    residual_norms!(W, B; nrhs, transposed = false, bm = nothing) -> W.norms_host

`‖W.R[:, k]‖₂²` and `‖B[:, k]‖₂²` for the compact columns `k` (`1:nrhs`, or
the active members' columns of the [`BatchMap`](@ref) `bm`) reduced on the
device (one workgroup per right-hand side, `@localmem` tree reduction) into
`W.norms`, then copied to `W.norms_host` (one host synchronization).
"""
function residual_norms!(W::RefinementWorkspace, B::AbstractVecOrMat; nrhs::Integer, transposed::Bool = false,
                         bm::Union{Nothing, BatchMap} = nothing)
    n = size(W.R, 1)
    bm = _user_map(bm, nrhs)
    ncols = bm.nrhs * bm.nact
    WG = REFINE_WORKGROUP
    kernel! = _residual_norms_kernel!(KernelAbstractions.get_backend(W.R), WG)
    if transposed
        kernel!(W.norms, W.R, B, n, Int(nrhs), bm, Val(true), Val(WG), Val(_ilog2(WG)); ndrange = WG * ncols)
    else
        kernel!(W.norms, W.R, B, n, Int(nrhs), bm, Val(false), Val(WG), Val(_ilog2(WG)); ndrange = WG * ncols)
    end
    copyto!(W.norms_host, 1, W.norms, 1, 2 * ncols)
    return W.norms_host
end

# first row of the map of `op(A)`: the rows of Mᵀ start at n (a "G" map)
function _map_row_offset(W::RefinementWorkspace, transpose_matrix::Bool)
    transpose_matrix || return 0
    n = size(W.R, 1)
    length(W.rowptr) == 2n + 1 ||
        throw(InvalidValueError("the refinement map has no rows of Mᵀ (only structure \"G\" solves with Aᵀ)"))
    return n
end

# largest relative residual ‖Rₖ‖/‖Bₖ‖ over the right-hand sides (‖Rₖ‖ when Bₖ = 0)
function _max_relative_residual(norms::Vector{R}, nrhs::Int) where {R}
    m = zero(R)
    for k in 1:nrhs
        rk, bk = sqrt(norms[2k - 1]), sqrt(norms[2k])
        m = max(m, bk > 0 ? rk / bk : rk)
    end
    return m
end

"""
    copy_rhs!(C, B; nrhs, transposed = false) -> C

`C[:, 1:nrhs]` = the right-hand sides of the user array `B` (column-major copy;
the refinement keeps `B` when the solution overwrites it). One launch.
"""
function copy_rhs!(C::AbstractMatrix, B::AbstractVecOrMat; nrhs::Integer, transposed::Bool = false)
    n = size(C, 1)
    n * nrhs > 0 || return C
    kernel! = _copy_rhs_kernel!(KernelAbstractions.get_backend(C), PERMUTE_WORKGROUP)
    if transposed
        kernel!(C, B, n, Int(nrhs), Val(true); ndrange = n * Int(nrhs))
    else
        kernel!(C, B, n, Int(nrhs), Val(false); ndrange = n * Int(nrhs))
    end
    return C
end

"""
    add_correction!(X, Y, perm; nrhs, transposed = false, conjugate = false, bm = nothing, scale = nothing) -> X

`X[perm[k], r] += Y[k, r]` (`conj(Y[k, r])` when `conjugate`; `Y[k, r]` times
`scale[perm[k]]` first with a matching scale vector): the solution of
the correction system, in factor order in the workspace `Y`, added to the user
array `X` of `nrhs` right-hand sides (uniform batch: the compact columns of `Y`
to the active members' columns of `X`, [`BatchMap`](@ref) `bm`). One launch,
conflict-free (`perm` is a permutation).
"""
function add_correction!(X::AbstractVecOrMat, Y::AbstractMatrix, perm::AbstractVector; nrhs::Integer,
                         transposed::Bool = false, conjugate::Bool = false, bm::Union{Nothing, BatchMap} = nothing,
                         scale::Union{Nothing, AbstractVector} = nothing)
    n = length(perm)
    bm = _user_map(bm, nrhs)
    ncols = bm.nrhs * bm.nact
    n * ncols > 0 || return X
    kernel! = _add_correction_kernel!(KernelAbstractions.get_backend(Y), PERMUTE_WORKGROUP)
    _with_flags(transposed, conjugate) do tr, cj
        kernel!(X, Y, perm, scale, n, Int(nrhs), ncols, bm, tr, cj; ndrange = n * ncols)
    end
    return X
end

"""
    refine!(X, B, W, ws, symbolic, numeric, nzval; nsteps, tol = 0, transposed = false,
            b_transposed = transposed, conjugate = false, deterministic = false,
            interrupt = nothing, progress = Ref(0), transpose_matrix = false, scaling = nothing) -> steps

Plain iterative refinement of the solution `X` of `op(A) X = B` (`op(A) = M`,
or `conj(M)` when `conjugate`; `Mᵀ` or `conj(Mᵀ)` with `transpose_matrix`, LU
only), at most `nsteps` steps of

    R = B - op(A) X            (residual!)
    stop if tol > 0 and maxₖ ‖Rₖ‖₂ / ‖Bₖ‖₂ ≤ tol   (residual_norms!, one host synchronization)
    X += op(A)⁻¹ R             (permute, forward / diagonal / backward sweeps, add_correction!)

with the factor in `numeric`, the solve workspace `ws` and the refinement
workspace `W` (with the permutations and scalings of matching when `scaling`
is a [`SolveScaling`](@ref), [`solve_io`](@ref); the residual is always the one
of the original matrix values `nzval`). `tol = 0` (the default of `ir_tol`) never stops early and never
synchronizes. `interrupt` (a `Threads.Atomic{Bool}` or `nothing`) is polled
before every step ([`InterruptedError`](@ref); `X` then holds the last
completed iterate). Returns the number of corrections applied, which is also
kept in `progress[]` after every step (so it survives an interrupt).
"""
function refine!(X::AbstractVecOrMat, B::AbstractVecOrMat, W::RefinementWorkspace, ws::SolveWorkspace, S::Symbolic,
                 N::Numeric, nzval::AbstractVector; nsteps::Integer, tol::Real = 0, transposed::Bool = false,
                 b_transposed::Bool = transposed, conjugate::Bool = false, deterministic::Bool = false,
                 interrupt::Union{Nothing, Threads.Atomic{Bool}} = nothing,
                 progress::Base.RefValue{Int} = Ref(0), transpose_matrix::Bool = false, scaling = nothing)
    nu = rhs_count(X, S.n; transposed)                        # user columns: nrhs per member × nbatch
    nu % N.nbatch == 0 || throw(DimensionMismatch("$nu right-hand sides for a batch of $(N.nbatch) members"))
    nrhs = nu ÷ N.nbatch
    ncols = _ncols(N, nrhs)
    max_rhs(W) >= ncols && max_rhs(ws) >= ncols ||
        throw(DimensionMismatch("the refinement workspace holds $(max_rhs(W)) right-hand sides, need $ncols"))
    bm = batch_map(N; nrhs)
    pin, sin, pout, sout = solve_io(S, scaling, transpose_matrix)
    steps = 0
    progress[] = 0
    for _ in 1:nsteps
        _poll_interrupt(interrupt)
        residual!(W, nzval, X, B; nrhs = nu, transposed, b_transposed, conjugate, bm, transpose_matrix)
        if tol > 0
            norms = residual_norms!(W, B; nrhs = nu, transposed = b_transposed, bm)
            rel = _max_relative_residual(norms, ncols)
            _log(LOG_DEBUG, () -> "refinement: step $steps, relative residual $rel")
            rel <= tol && break
        end
        permute_rhs!(ws.Y, W.R, pin; conjugate, scale = sin)
        forward_sweep!(ws, S, N; nrhs, deterministic, transpose = transpose_matrix)
        diagonal_sweep!(ws, S, N; nrhs)
        backward_sweep!(ws, S, N; nrhs, transpose = transpose_matrix)
        add_correction!(X, ws.Y, pout; nrhs = nu, transposed, conjugate, bm, scale = sout)
        steps += 1
        progress[] = steps
    end
    return steps
end

# --- FGMRES-IR (`ir_mode = "fgmres"`, Krylov.jl extension) ----------------------
#
# FGMRES on the correction system `op(A) D = R₀` (`R₀ = B - op(A) X₀`, `X₀` the
# solution of the sweeps), right-preconditioned by the factorization, then
# `X = X₀ + D`. Krylov.jl works on vectors: the compact columns of all
# right-hand sides (and active batch members) are stacked into one vector of
# length `n ncols`, i.e. FGMRES runs on the block-diagonal system `I ⊗ op(A)`
# with the operators below; both apply all columns per launch.

"""
    FGMRES_PROVIDER

`Ref` to the FGMRES driver of the Krylov.jl extension
(`SparseDirectSolverKrylovExt`), set in its `__init__`; `nothing` while
Krylov.jl is not loaded (`ir_mode = "fgmres"` then raises
[`NotSupportedError`](@ref)). Called as
`provider(cache, A, P, R, len; atol, itmax) -> (D, iterations)`: FGMRES for
`A D = R[1:len]` with right preconditioner `P` (`mul!(y, P, x)`), absolute
residual tolerance `atol`, at most `itmax` iterations, storage kept in `cache`
(a `Ref{Any}`).
"""
const FGMRES_PROVIDER = Ref{Any}(nothing)

"""
    fgmres_available() -> Bool

Whether the Krylov.jl extension is loaded (`ir_mode = "fgmres"` works).
"""
fgmres_available() = FGMRES_PROVIDER[] !== nothing

"""
    RefinementOperator(W, nzval, n, ncols, bm, conjugate, roff = 0)

`op(A)` (the full matrix `M` of the [`RefinementWorkspace`](@ref) `W` with
the values `nzval`, `conj(M)` when `conjugate`; `roff = n`: `Mᵀ` from the
rows of a `"G"` map) acting on the `n × ncols`
compact columns stacked into vectors of length `n ncols` (column `r` uses
the values of the member of compact column `r` of the [`BatchMap`](@ref)
`bm`). Supports `size`, `eltype` and `mul!(y, op, x)` (one launch,
asynchronous): the matrix operator of FGMRES-IR.
"""
struct RefinementOperator{T, RW <: RefinementWorkspace{T}, V <: AbstractVector, BM <: BatchMap}
    W::RW
    nzval::V
    n::Int
    ncols::Int
    bm::BM
    conjugate::Bool
    roff::Int
end

RefinementOperator(W::RefinementWorkspace, nzval::AbstractVector, n::Integer, ncols::Integer, bm::BatchMap,
                   conjugate::Bool) = RefinementOperator(W, nzval, Int(n), Int(ncols), bm, conjugate, 0)

Base.size(op::RefinementOperator) = (op.n * op.ncols, op.n * op.ncols)
Base.size(op::RefinementOperator, d::Integer) = d <= 2 ? op.n * op.ncols : 1
Base.eltype(::RefinementOperator{T}) where {T} = T

@kernel function _spmv_kernel!(y, rowptr, colval, src, nzval_all, x, n, ncols, bm, roff, ::Val{CJ}) where {CJ}
    q = @index(Global, Linear)
    i = (q - 1) % n + 1
    r = (q - 1) ÷ n + 1
    @inbounds if r <= ncols
        nzval = _mview(nzval_all, _bm_cmember(bm, r), bm.nbatch)
        off = (r - 1) * n
        acc = zero(eltype(y))
        for c in Int(rowptr[roff + i]):(Int(rowptr[roff + i + 1]) - 1)
            s = Int(src[c])
            v = nzval[abs(s)]
            v = xor(s < 0, CJ) ? conj(v) : v
            acc += v * x[Int(colval[c]) + off]
        end
        y[i + off] = acc
    end
end

function LinearAlgebra.mul!(y::AbstractVector, op::RefinementOperator, x::AbstractVector)
    m = op.n * op.ncols
    length(x) == m && length(y) == m ||
        throw(DimensionMismatch("the operator has size $m, x has $(length(x)) and y $(length(y)) entries"))
    m > 0 || return y
    W = op.W
    kernel! = _spmv_kernel!(KernelAbstractions.get_backend(W.R), PERMUTE_WORKGROUP)
    if op.conjugate
        kernel!(y, W.rowptr, W.colval, W.src, op.nzval, x, op.n, op.ncols, op.bm, op.roff, Val(true); ndrange = m)
    else
        kernel!(y, W.rowptr, W.colval, W.src, op.nzval, x, op.n, op.ncols, op.bm, op.roff, Val(false); ndrange = m)
    end
    return y
end

"""
    FactorPreconditioner(ws, S, N, nrhs, ncols, conjugate, deterministic, interrupt, transpose = false,
                         scaling = nothing)

`op(A)⁻¹` through the factorization in `N` (permutation, forward, diagonal
and backward sweeps in the solve workspace `ws`; `conj(M⁻¹ conj(x))` when
`conjugate`; the sweeps of `Mᵀ` when `transpose`, LU only) on the stacked compact columns (`nrhs` per active member,
`ncols` in all) of the [`RefinementOperator`](@ref); with the matching
permutations and scalings of a [`SolveScaling`](@ref) `scaling` ([`solve_io`](@ref)). `mul!(y, P, x)` polls
`interrupt` first ([`InterruptedError`](@ref)); asynchronous otherwise. The
right preconditioner of FGMRES-IR.
"""
struct FactorPreconditioner{T, WS <: SolveWorkspace{T}, SY <: Symbolic, NU <: Numeric}
    ws::WS
    S::SY
    N::NU
    nrhs::Int
    ncols::Int
    conjugate::Bool
    deterministic::Bool
    interrupt::Union{Nothing, Threads.Atomic{Bool}}
    transpose::Bool
    scaling::Any
end

FactorPreconditioner(ws::SolveWorkspace, S::Symbolic, N::Numeric, nrhs::Integer, ncols::Integer, conjugate::Bool,
                     deterministic::Bool, interrupt, transpose::Bool = false, scaling = nothing) =
    FactorPreconditioner(ws, S, N, Int(nrhs), Int(ncols), conjugate, deterministic, interrupt, transpose, scaling)

Base.size(P::FactorPreconditioner) = (P.S.n * P.ncols, P.S.n * P.ncols)
Base.size(P::FactorPreconditioner, d::Integer) = d <= 2 ? P.S.n * P.ncols : 1
Base.eltype(::FactorPreconditioner{T}) where {T} = T

function LinearAlgebra.mul!(y::AbstractVector, P::FactorPreconditioner, x::AbstractVector)
    _poll_interrupt(P.interrupt)
    pin, sin, pout, sout = solve_io(P.S, P.scaling, P.transpose)
    permute_rhs!(P.ws.Y, x, pin; conjugate = P.conjugate, scale = sin)
    forward_sweep!(P.ws, P.S, P.N; nrhs = P.nrhs, deterministic = P.deterministic, transpose = P.transpose)
    diagonal_sweep!(P.ws, P.S, P.N; nrhs = P.nrhs)
    backward_sweep!(P.ws, P.S, P.N; nrhs = P.nrhs, transpose = P.transpose)
    unpermute_solution!(y, P.ws.Y, pout; conjugate = P.conjugate, scale = sout)
    return y
end

# smallest ‖Bₖ‖ over the right-hand sides (1 for Bₖ = 0, as in `_max_relative_residual`)
function _min_rhs_norm(norms::Vector{R}, nrhs::Int) where {R}
    m = typemax(R)
    for k in 1:nrhs
        bk = sqrt(norms[2k])
        m = min(m, bk > 0 ? bk : one(R))
    end
    return m
end

"""
    fgmres_refine!(X, B, W, ws, symbolic, numeric, nzval; nsteps, tol = 0, transposed = false,
                   b_transposed = transposed, conjugate = false, deterministic = false,
                   interrupt = nothing, progress = Ref(0), transpose_matrix = false, scaling = nothing)
        -> iterations

FGMRES-IR (`ir_mode = "fgmres"`, needs Krylov.jl) of the solution `X` of
`op(A) X = B`, with the arguments of [`refine!`](@ref):

    R₀ = B - op(A) X                         (residual!)
    stop if tol > 0 and maxₖ ‖R₀ₖ‖₂ / ‖Bₖ‖₂ ≤ tol
    D = FGMRES(op(A), R₀; right preconditioner op(A)⁻¹ by the factor,
               at most nsteps iterations, ‖R₀ - op(A) D‖₂ ≤ tol minₖ ‖Bₖ‖₂)
    X += D                                   (add_correction!)

on the stacked compact columns ([`RefinementOperator`](@ref),
[`FactorPreconditioner`](@ref)). Each iteration costs one SpMV and one solve,
as a step of plain refinement. `tol = 0` runs `nsteps` iterations (FGMRES
stops earlier only on an exact residual). The joint criterion bounds every
column's relative residual by `tol`. Returns the iterations performed (also
in `progress[]`); an interrupt (polled before every preconditioner
application) leaves `X` unchanged and `progress[] = 0`. FGMRES synchronizes
with the host every iteration (Krylov.jl's dot products and norms).
"""
function fgmres_refine!(X::AbstractVecOrMat, B::AbstractVecOrMat, W::RefinementWorkspace, ws::SolveWorkspace,
                        S::Symbolic, N::Numeric, nzval::AbstractVector; nsteps::Integer, tol::Real = 0,
                        transposed::Bool = false, b_transposed::Bool = transposed, conjugate::Bool = false,
                        deterministic::Bool = false, interrupt::Union{Nothing, Threads.Atomic{Bool}} = nothing,
                        progress::Base.RefValue{Int} = Ref(0), transpose_matrix::Bool = false,
                        scaling = nothing)
    provider = FGMRES_PROVIDER[]
    provider === nothing &&
        throw(NotSupportedError("ir_mode = \"fgmres\" needs Krylov.jl: load it with `using Krylov`"))
    nu = rhs_count(X, S.n; transposed)
    nu % N.nbatch == 0 || throw(DimensionMismatch("$nu right-hand sides for a batch of $(N.nbatch) members"))
    nrhs = nu ÷ N.nbatch
    ncols = _ncols(N, nrhs)
    max_rhs(W) >= ncols && max_rhs(ws) >= ncols ||
        throw(DimensionMismatch("the refinement workspace holds $(max_rhs(W)) right-hand sides, need $ncols"))
    progress[] = 0
    nsteps > 0 && S.n * ncols > 0 || return 0
    bm = batch_map(N; nrhs)
    _poll_interrupt(interrupt)
    residual!(W, nzval, X, B; nrhs = nu, transposed, b_transposed, conjugate, bm, transpose_matrix)
    R = real(eltype(W.R))
    atol = zero(R)
    if tol > 0
        norms = residual_norms!(W, B; nrhs = nu, transposed = b_transposed, bm)
        rel = _max_relative_residual(norms, ncols)
        _log(LOG_DEBUG, () -> "fgmres refinement: initial relative residual $rel")
        rel <= tol && return 0
        atol = R(tol) * _min_rhs_norm(norms, ncols)
    end
    A = RefinementOperator(W, nzval, S.n, ncols, bm, conjugate, _map_row_offset(W, transpose_matrix))
    P = FactorPreconditioner(ws, S, N, nrhs, ncols, conjugate, deterministic, interrupt, transpose_matrix, scaling)
    D, iters = provider(W.krylov, A, P, W.R, S.n * ncols; atol, itmax = Int(nsteps))
    permute_rhs!(ws.Y, D, S.perm)            # D is in the original numbering: scatter into the user columns
    add_correction!(X, ws.Y, S.perm; nrhs = nu, transposed, bm)
    progress[] = iters
    return iters
end

"""
    solve_conjugated(structure, T, csc_input::Bool, solve_mode) -> Bool

Whether `op(A)` (`solve_mode` 0: `A`, 1: `Aᵀ`, 2: `Aᴴ`) of the user matrix is
the conjugate of the stored, factorized matrix `M` (`A = M`, or `A = Mᵀ` for a
CSC input). Real `T`: never. Complex symmetric `"S"` (`M = Mᵀ`): for `Aᴴ`.
Hermitian `"H"`/`"HPD"` (`Mᵀ = conj(M)`): for `Aᵀ` of a CSR input, and for `A`
and `Aᴴ` of a CSC input. General `"G"`: for `Aᴴ` (`op(A) = conj(Mᵀ)` or
`conj(M)`, see [`solve_transposed`](@ref)).
"""
function solve_conjugated(structure::Structure, ::Type{T}, csc_input::Bool, solve_mode::Integer) where {T}
    T <: Complex || return false
    _is_hermitian(structure) && return xor(csc_input, solve_mode == 1)
    return solve_mode == 2
end

"""
    solve_transposed(structure, csc_input::Bool, solve_mode) -> Bool

Whether the solve of `op(A)` (`solve_mode` 0: `A`, 1: `Aᵀ`, 2: `Aᴴ`) uses the
transpose `Mᵀ` of the stored, factorized matrix `M` (`A = M`, or `A = Mᵀ` for
a CSC input), conjugated or not ([`solve_conjugated`](@ref)). Only for
structure `"G"` (LU): `Mᵀ` for `Aᵀ`/`Aᴴ` of a CSR input and for `A` of a CSC
input. Symmetric and Hermitian structures never need it.
"""
solve_transposed(structure::Structure, csc_input::Bool, solve_mode::Integer) =
    structure == STRUCTURE_GENERAL && xor(solve_mode != 0, csc_input)
