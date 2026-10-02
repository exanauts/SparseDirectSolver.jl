# Julia-native layer (PLAN §1.5, §3.1) on top of the handle layer: `cholesky`,
# `cholesky!`, `ldlt`, `ldlt!`, `ldiv!`, `\`, `logabsdet`, `diag`, `nnz`. Methods take the
# in-package `CSR`; the CUDA extension adds `CuSparseMatrixCSR` and its
# `Symmetric`/`Hermitian` wrappers, as CUDSS.jl does. A `SparseMatrixCSC` is not
# accepted by `cholesky` (that method belongs to CHOLMOD): wrap it with `CSR(A)`.
#
# PLAN §3.1 turns iterative refinement on in this layer (`ir_n_steps = 2` with
# early exit). Refinement arrives in T16; until then the layer keeps the
# handle-layer default `ir_n_steps = 0`, and a value set by the user is stored
# and ignored (the solve warns once).

"""
    cholesky(A::CSR, NoPivot(); view = 'F', check = false) -> DirectSolver

LLᵀ (real `T`, structure `"SPD"`) or LLᴴ (complex `T`, `"HPD"`) factorization of
the sparse matrix `A` on its backend: a [`DirectSolver`](@ref) after
`"analysis"` and `"factorization"` (synchronized). `view` selects the triangle
of `A` that is read (`'L'`, `'U'`, `'F'`). As in CUDSS.jl a failed
factorization does not throw unless `check = true` (then
[`FactorizationError`](@ref)); otherwise check `getparam(solver, "info")`.
"""
function LinearAlgebra.cholesky(A::CSR{T}, ::NoPivot = NoPivot(); view::Char = 'F', check::Bool = false) where {T}
    solver = DirectSolver(A, T <: Real ? "SPD" : "HPD", view)
    execute!("analysis", solver, nothing, nothing)
    execute!("factorization", solver, nothing, nothing; asynchronous = false)
    _check_info(solver, check)
    return solver
end

function _check_info(solver::DirectSolver, check::Bool)
    check && solver.info != 0 &&
        throw(FactorizationError(solver.info, "the matrix is not positive definite (pivot $(solver.info))"))
    return nothing
end

"""
    ldlt(A::CSR; view = 'F', check = false) -> DirectSolver

LDLᵀ (real `T`, structure `"S"`) or LDLᴴ (complex `T`, `"H"`) factorization of
the sparse symmetric/Hermitian matrix `A` on its backend (≅ CUDSS.jl's `ldlt`):
a [`DirectSolver`](@ref) after `"analysis"` and `"factorization"`
(synchronized), with in-front Bunch–Kaufman pivoting and static perturbation of
tiny pivots (read `getparam(solver, "npivots")` and `"inertia"`). `view` as in
[`cholesky`](@ref). The factorization does not fail (`"info"` stays 0), so
`check` has no effect; it is accepted for symmetry with `cholesky`.
"""
function LinearAlgebra.ldlt(A::CSR{T}; view::Char = 'F', check::Bool = false) where {T}
    solver = DirectSolver(A, T <: Real ? "S" : "H", view)
    execute!("analysis", solver, nothing, nothing)
    execute!("factorization", solver, nothing, nothing; asynchronous = false)
    _check_info(solver, check)
    return solver
end

"""
    ldlt!(solver::DirectSolver, A; check = false) -> solver

Factorize the new values `A` reusing the analysis of `solver`, as
[`cholesky!`](@ref) (`"factorization"` the first time after an analysis,
`"refactorization"` afterwards, synchronized).
"""
function LinearAlgebra.ldlt!(solver::DirectSolver, A; check::Bool = false)
    update!(solver, A)
    phase = solver.fresh_factorization ? "factorization" : "refactorization"
    execute!(phase, solver, nothing, nothing; asynchronous = false)
    _check_info(solver, check)
    return solver
end

"""
    cholesky!(solver::DirectSolver, A; check = false) -> solver

Factorize the new values `A` (a [`CSR`](@ref), `SparseMatrixCSC`, raw-array
triple through [`update!`](@ref), or a device sparse matrix of an extension)
reusing the analysis of `solver`: [`update!`](@ref), then `"factorization"` the
first time after an analysis (`solver.fresh_factorization`) and
`"refactorization"` afterwards, synchronized. `check` as in [`cholesky`](@ref).
"""
function LinearAlgebra.cholesky!(solver::DirectSolver, A; check::Bool = false)
    update!(solver, A)
    phase = solver.fresh_factorization ? "factorization" : "refactorization"
    execute!(phase, solver, nothing, nothing; asynchronous = false)
    _check_info(solver, check)
    return solver
end

# one method per argument kind: LinearAlgebra has `ldiv!` methods for `Factorization` with
# vectors and with matrices, so a `Union` here would be ambiguous
for RHS in (AbstractVector, AbstractMatrix, MatrixDescriptor)
    @eval function LinearAlgebra.ldiv!(solver::DirectSolver, B::$RHS)
        execute!("solve", solver, B, B; asynchronous = false)
        return B
    end
    @eval function LinearAlgebra.ldiv!(X::$RHS, solver::DirectSolver, B::$RHS)
        execute!("solve", solver, X, B; asynchronous = false)
        return X
    end
end

@doc """
    ldiv!(solver::DirectSolver, B) -> B
    ldiv!(X, solver::DirectSolver, B) -> X

Solve `A X = B` with the factorization in `solver` (`"solve"`, synchronized),
in place in `B` or into `X`. `B`, `X`: vectors, matrices or
[`MatrixDescriptor`](@ref)s on the solver's backend.
""" LinearAlgebra.ldiv!(::DirectSolver, ::AbstractVector)

"""
    solver \\ B

`ldiv!(similar(B), solver, B)` for a vector or matrix `B`.
"""
Base.:\(solver::DirectSolver, B::AbstractVector) = ldiv!(similar(B), solver, B)
Base.:\(solver::DirectSolver, B::AbstractMatrix) = ldiv!(similar(B), solver, B)
# disambiguation with LinearAlgebra's real-factorization / complex-RHS method: the element types must match
Base.:\(solver::DirectSolver{T}, B::Union{Array{Complex{T}, 1}, Array{Complex{T}, 2}}) where {T <: Union{Float32, Float64}} =
    ldiv!(similar(B), solver, B)

"""
    diag(solver::DirectSolver)

Diagonal of the factor `L` (Cholesky) or of `D` (LDLᵀ/LDLᴴ) in factor order
(the `"diag"` data parameter), a vector on the solver's backend.
"""
LinearAlgebra.diag(solver::DirectSolver) = getparam(solver, "diag")

"""
    nnz(solver::DirectSolver)

Nonzeros of the factor `L` (the `"lu_nnz"` data parameter).
"""
SparseArrays.nnz(solver::DirectSolver) = Int(getparam(solver, "lu_nnz"))

"""
    logabsdet(solver::DirectSolver) -> (log|det A|, sign)

Cholesky: from the diagonal of the factor, `log|det A| = 2 Σ log Lₖₖ`, `sign =
one(T)` (`A` is positive definite). LDLᵀ/LDLᴴ: `det A = det D` (the
permutations cancel, `det L = 1`), the product of the 1×1 pivots and of the
determinants of the 2×2 blocks; `sign` is `±1` (real or Hermitian) or `det D /
|det D|` (complex symmetric). With perturbed pivots (`"npivots" > 0`) this is
the determinant of `A + E`. Raises [`FactorizationError`](@ref) when the
factorization failed (`"info" ≠ 0`).
"""
function LinearAlgebra.logabsdet(solver::DirectSolver{T}) where {T}
    solver.stage >= STAGE_FACTORIZED && solver.info == 0 ||
        throw(FactorizationError(solver.info, "logabsdet needs a successful factorization"))
    _is_ldlt_structure(solver.structure) && return _logabsdet_ldlt(solver)
    d = Array(getparam(solver, "diag"))
    return 2 * sum(x -> log(abs(x)), d; init = zero(real(T))), one(T)
end

function _logabsdet_ldlt(solver::DirectSolver{T}) where {T}
    N = solver.numeric
    n = size(solver, 1)
    d, kind = Array(N.d), Array(N.pivot_kind)
    herm = !(T <: Complex) || solver.structure == STRUCTURE_HERMITIAN
    la, sgn = zero(real(T)), one(T)
    k = 1
    while k <= n
        if kind[k] == PIVOT_KIND_2X2_FIRST
            b = d[n + k]
            x = d[k] * d[k + 1] - (herm ? conj(b) : b) * b
            herm && (x = T(real(x)))
            k += 2
        else
            x = d[k]
            k += 1
        end
        la += log(abs(x))
        sgn *= herm ? T(sign(real(x))) : x / abs(x)
    end
    return la, sgn
end

"""
    logdet(solver::DirectSolver)

`log det A` of the positive definite matrix factorized in `solver` (the first
value of [`logabsdet`](@ref)).
"""
LinearAlgebra.logdet(solver::DirectSolver) = first(logabsdet(solver))
