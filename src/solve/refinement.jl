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
# stored CSR matrix `M` or, for a CSC input, `Mᵀ`. `M` is symmetric or
# Hermitian, so every `op(A)` is `M` or `conj(M)`; `conj(M) x = b` is solved as
# `x = conj(M⁻¹ conj(b))` ([`solve_conjugated`](@ref)).
#
# User arrays come in the layouts of `src/solve/permute.jl` (vector, matrix,
# strided vector, row-major when transposed); the residual `R` is a column-major
# `n × nrhs` device matrix in the original numbering.

"Workgroup size of the residual-norm reduction (one workgroup per right-hand side)."
const REFINE_WORKGROUP = 256

"""
    RefinementWorkspace{T, R, VI, MT, VR}

Device storage of iterative refinement, allocated at the first solve that
refines ([`allocate_refinement`](@ref)) and grown with the number of right-hand
sides:

* `rowptr`, `colval`, `src` (`INT` vectors): the full matrix `M` as a CSR over
  the contributions of the user's `nzval` (`src[c] > 0`: `nzval[src[c]]`,
  `src[c] < 0`: `conj(nzval[-src[c]])`), from [`refinement_map`](@ref);
* `R` (`n × nrhs`): the residual, then the correction;
* `Bc` (`n × nrhs`): a copy of the right-hand side when `X` and `B` alias;
* `norms` (`2 nrhs`, real): `‖Rₖ‖₂²` and `‖Bₖ‖₂²` per right-hand side, and
  `norms_host`, its host copy (the early-exit test of `ir_tol`).
"""
struct RefinementWorkspace{T, R <: Real, VI <: AbstractVector, MT <: AbstractMatrix{T}, VR <: AbstractVector{R}}
    rowptr::VI
    colval::VI
    src::VI
    R::MT
    Bc::MT
    norms::VR
    norms_host::Vector{R}
end

max_rhs(W::RefinementWorkspace) = size(W.R, 2)

"""
    refinement_map(F::FullPatternMap) -> (rowptr, colval, src)

Host CSR of the full matrix of `F` over its contributions (duplicates are not
merged; their products are summed by the SpMV): row `i` holds the entries
`rowptr[i]:rowptr[i+1]-1`, column `colval[c]`, value `nzval[src[c]]`, conjugated
when `src[c] < 0`. 1-based.
"""
function refinement_map(F::FullPatternMap)
    n = F.n
    rowptr = Vector{Int}(undef, n + 1)
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
    return rowptr, colval, src
end

"""
    allocate_refinement(map, T, INT, backend, nrhs) -> RefinementWorkspace

Move the host [`refinement_map`](@ref) `map` to `backend` as `INT` vectors
(overflow-checked) and allocate the residual storage for `nrhs` right-hand
sides of element type `T`.
"""
function allocate_refinement(map::NTuple{3, Vector{Int}}, ::Type{T}, ::Type{INT},
                             backend::KernelAbstractions.Backend, nrhs::Integer) where {T, INT}
    rowptr, colval, src = map
    n = length(rowptr) - 1
    dev(v) = (v_ = _to_index_type(INT, v); d = KernelAbstractions.allocate(backend, INT, length(v_)); copyto!(d, v_); d)
    R = KernelAbstractions.zeros(backend, T, n, nrhs)
    Bc = KernelAbstractions.zeros(backend, T, n, nrhs)
    norms = KernelAbstractions.zeros(backend, real(T), 2 * nrhs)
    return RefinementWorkspace{T, real(T), typeof(dev(Int[])), typeof(R), typeof(norms)}(
        dev(rowptr), dev(colval), dev(src), R, Bc, norms, zeros(real(T), 2 * nrhs))
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
    return RefinementWorkspace{T, R, VI, MT, VR}(W.rowptr, W.colval, W.src, Rm, Bc, norms, zeros(R, 2 * nrhs))
end

@kernel function _residual_kernel!(R, rowptr, colval, src, nzval, X, B, n, nrhs, ::Val{XT}, ::Val{BT},
                                   ::Val{CJ}) where {XT, BT, CJ}
    q = @index(Global, Linear)
    i = (q - 1) % n + 1
    r = (q - 1) ÷ n + 1
    @inbounds if r <= nrhs
        acc = zero(eltype(R))
        for c in Int(rowptr[i]):(Int(rowptr[i + 1]) - 1)
            s = Int(src[c])
            v = nzval[abs(s)]
            v = xor(s < 0, CJ) ? conj(v) : v
            acc += v * _rhs_get(X, Int(colval[c]), r, n, nrhs, Val(XT))
        end
        R[i, r] = _rhs_get(B, i, r, n, nrhs, Val(BT)) - acc
    end
end

@kernel function _residual_norms_kernel!(norms, R, B, n, nrhs, ::Val{BT}, ::Val{WG}, ::Val{LOG2WG}) where {BT, WG, LOG2WG}
    @uniform RT = eltype(norms)
    li = @index(Local, Linear)
    r = @index(Group, Linear)
    nr = @localmem RT (WG,)
    nb = @localmem RT (WG,)
    @inbounds begin
        sr = zero(RT)
        sb = zero(RT)
        for i in li:WG:n
            sr += abs2(R[i, r])
            sb += abs2(_rhs_get(B, i, r, n, nrhs, Val(BT)))
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

@kernel function _add_correction_kernel!(X, Y, perm, n, nrhs, ::Val{TR}, ::Val{CJ}) where {TR, CJ}
    q = @index(Global, Linear)
    k = (q - 1) % n + 1
    r = (q - 1) ÷ n + 1
    @inbounds if r <= nrhs
        i = Int(perm[k])
        y = Y[k, r]
        _rhs_set!(X, _rhs_get(X, i, r, n, nrhs, Val(TR)) + (CJ ? conj(y) : y), i, r, n, nrhs, Val(TR))
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

"""
    residual!(W, nzval, X, B; nrhs, transposed = false, b_transposed = transposed, conjugate = false) -> W.R

`W.R[:, 1:nrhs] = B - M X` (`conj(M)` when `conjugate`) with the full matrix
`M` of the map of `W` and the user values `nzval`; `X`, `B` in the layouts of
[`rhs_count`](@ref) (row-major when `transposed`/`b_transposed`). One launch,
one work item per (row, right-hand side), gather-based (no atomics).
Asynchronous.
"""
function residual!(W::RefinementWorkspace, nzval::AbstractVector, X::AbstractVecOrMat, B::AbstractVecOrMat;
                   nrhs::Integer, transposed::Bool = false, b_transposed::Bool = transposed,
                   conjugate::Bool = false)
    n = size(W.R, 1)
    n * nrhs > 0 || return W.R
    kernel! = _residual_kernel!(KernelAbstractions.get_backend(W.R), PERMUTE_WORKGROUP)
    _with_flags(transposed, b_transposed) do xt, bt
        if conjugate
            kernel!(W.R, W.rowptr, W.colval, W.src, nzval, X, B, n, Int(nrhs), xt, bt, Val(true); ndrange = n * Int(nrhs))
        else
            kernel!(W.R, W.rowptr, W.colval, W.src, nzval, X, B, n, Int(nrhs), xt, bt, Val(false); ndrange = n * Int(nrhs))
        end
    end
    return W.R
end

"""
    residual_norms!(W, B; nrhs, transposed = false) -> W.norms_host

`‖W.R[:, k]‖₂²` and `‖B[:, k]‖₂²` for `k = 1:nrhs` reduced on the device (one
workgroup per right-hand side, `@localmem` tree reduction) into `W.norms`, then
copied to `W.norms_host` (one host synchronization).
"""
function residual_norms!(W::RefinementWorkspace, B::AbstractVecOrMat; nrhs::Integer, transposed::Bool = false)
    n = size(W.R, 1)
    WG = REFINE_WORKGROUP
    kernel! = _residual_norms_kernel!(KernelAbstractions.get_backend(W.R), WG)
    if transposed
        kernel!(W.norms, W.R, B, n, Int(nrhs), Val(true), Val(WG), Val(_ilog2(WG)); ndrange = WG * Int(nrhs))
    else
        kernel!(W.norms, W.R, B, n, Int(nrhs), Val(false), Val(WG), Val(_ilog2(WG)); ndrange = WG * Int(nrhs))
    end
    copyto!(W.norms_host, 1, W.norms, 1, 2 * Int(nrhs))
    return W.norms_host
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
    add_correction!(X, Y, perm; nrhs, transposed = false, conjugate = false) -> X

`X[perm[k], r] += Y[k, r]` (`conj(Y[k, r])` when `conjugate`): the solution of
the correction system, in factor order in the workspace `Y`, added to the user
array `X`. One launch, conflict-free (`perm` is a permutation).
"""
function add_correction!(X::AbstractVecOrMat, Y::AbstractMatrix, perm::AbstractVector; nrhs::Integer,
                         transposed::Bool = false, conjugate::Bool = false)
    n = length(perm)
    n * nrhs > 0 || return X
    kernel! = _add_correction_kernel!(KernelAbstractions.get_backend(Y), PERMUTE_WORKGROUP)
    _with_flags(transposed, conjugate) do tr, cj
        kernel!(X, Y, perm, n, Int(nrhs), tr, cj; ndrange = n * Int(nrhs))
    end
    return X
end

"""
    refine!(X, B, W, ws, symbolic, numeric, nzval; nsteps, tol = 0, transposed = false,
            b_transposed = transposed, conjugate = false, deterministic = false,
            interrupt = nothing, progress = Ref(0)) -> steps

Plain iterative refinement of the solution `X` of `op(A) X = B` (`op(A) = M`,
or `conj(M)` when `conjugate`), at most `nsteps` steps of

    R = B - op(A) X            (residual!)
    stop if tol > 0 and maxₖ ‖Rₖ‖₂ / ‖Bₖ‖₂ ≤ tol   (residual_norms!, one host synchronization)
    X += op(A)⁻¹ R             (permute, forward / diagonal / backward sweeps, add_correction!)

with the factor in `numeric`, the solve workspace `ws` and the refinement
workspace `W`. `tol = 0` (the default of `ir_tol`) never stops early and never
synchronizes. `interrupt` (a `Threads.Atomic{Bool}` or `nothing`) is polled
before every step ([`InterruptedError`](@ref); `X` then holds the last
completed iterate). Returns the number of corrections applied, which is also
kept in `progress[]` after every step (so it survives an interrupt).
"""
function refine!(X::AbstractVecOrMat, B::AbstractVecOrMat, W::RefinementWorkspace, ws::SolveWorkspace, S::Symbolic,
                 N::Numeric, nzval::AbstractVector; nsteps::Integer, tol::Real = 0, transposed::Bool = false,
                 b_transposed::Bool = transposed, conjugate::Bool = false, deterministic::Bool = false,
                 interrupt::Union{Nothing, Threads.Atomic{Bool}} = nothing,
                 progress::Base.RefValue{Int} = Ref(0))
    nrhs = rhs_count(X, S.n; transposed)
    max_rhs(W) >= nrhs && max_rhs(ws) >= nrhs ||
        throw(DimensionMismatch("the refinement workspace holds $(max_rhs(W)) right-hand sides, need $nrhs"))
    steps = 0
    progress[] = 0
    for _ in 1:nsteps
        _poll_interrupt(interrupt)
        residual!(W, nzval, X, B; nrhs, transposed, b_transposed, conjugate)
        if tol > 0
            norms = residual_norms!(W, B; nrhs, transposed = b_transposed)
            rel = _max_relative_residual(norms, nrhs)
            _log(LOG_DEBUG, () -> "refinement: step $steps, relative residual $rel")
            rel <= tol && break
        end
        permute_rhs!(ws.Y, W.R, S.perm; conjugate)
        forward_sweep!(ws, S, N; nrhs, deterministic)
        diagonal_sweep!(ws, S, N; nrhs)
        backward_sweep!(ws, S, N; nrhs)
        add_correction!(X, ws.Y, S.perm; nrhs, transposed, conjugate)
        steps += 1
        progress[] = steps
    end
    return steps
end

"""
    solve_conjugated(structure, T, csc_input::Bool, solve_mode) -> Bool

Whether `op(A)` (`solve_mode` 0: `A`, 1: `Aᵀ`, 2: `Aᴴ`) of the user matrix is
the conjugate of the stored, factorized matrix `M` (`A = M`, or `A = Mᵀ` for a
CSC input). Real `T`: never. Complex symmetric `"S"` (`M = Mᵀ`): for `Aᴴ`.
Hermitian `"H"`/`"HPD"` (`Mᵀ = conj(M)`): for `Aᵀ` of a CSR input, and for `A`
and `Aᴴ` of a CSC input.
"""
function solve_conjugated(structure::Structure, ::Type{T}, csc_input::Bool, solve_mode::Integer) where {T}
    T <: Complex || return false
    _is_hermitian(structure) && return xor(csc_input, solve_mode == 1)
    return solve_mode == 2
end
