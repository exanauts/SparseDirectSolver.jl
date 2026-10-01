# Matrix generators shared by every test file (TASKS.md "Shared test conventions").
#
# Every generator returns a `SparseMatrixCSC{T,Int}` (the doc examples return
# NamedTuples holding them). Symmetric generators return exactly symmetric
# (real) or exactly Hermitian (complex) matrices. Random generators draw from
# `rng` (default: the global RNG, which the test runner seeds with 666). The
# methods without a `T` argument use `Float64`.

using LinearAlgebra
using Random
using SparseArrays

# tridiag(-1, 2, -1): 1-D Laplacian with Dirichlet boundary conditions.
function _laplacian1d(::Type{T}, n::Integer) where {T}
    return spdiagm(-1 => fill(-one(T), n - 1), 0 => fill(T(2), n), 1 => fill(-one(T), n - 1))
end

_speye(::Type{T}, n::Integer) where {T} = sparse(one(T) * I, n, n)

# Sum of |a_ij| over the stored entries of each row.
function _row_abs_sums(A::SparseMatrixCSC)
    r = zeros(real(eltype(A)), size(A, 1))
    rows = rowvals(A)
    vals = nonzeros(A)
    for j in axes(A, 2), k in nzrange(A, j)
        r[rows[k]] += abs(vals[k])
    end
    return r
end

# A without its diagonal entries (structurally removed).
_offdiag(A::SparseMatrixCSC) = dropzeros!(A - spdiagm(0 => diag(A)))

"""
    laplacian2d(T, nx, ny)

5-point Laplacian on an `nx × ny` grid with Dirichlet boundary conditions
(SPD, `n = nx·ny`, natural row-major grid numbering).
"""
function laplacian2d(::Type{T}, nx::Integer, ny::Integer) where {T}
    return kron(_speye(T, ny), _laplacian1d(T, nx)) + kron(_laplacian1d(T, ny), _speye(T, nx))
end
laplacian2d(nx::Integer, ny::Integer) = laplacian2d(Float64, nx, ny)

"""
    laplacian3d(T, nx, ny, nz)

7-point Laplacian on an `nx × ny × nz` grid with Dirichlet boundary conditions (SPD).
"""
function laplacian3d(::Type{T}, nx::Integer, ny::Integer, nz::Integer) where {T}
    Ix, Iy, Iz = _speye(T, nx), _speye(T, ny), _speye(T, nz)
    Lx, Ly, Lz = _laplacian1d(T, nx), _laplacian1d(T, ny), _laplacian1d(T, nz)
    return kron(Iz, kron(Iy, Lx)) + kron(Iz, kron(Ly, Ix)) + kron(Lz, kron(Iy, Ix))
end
laplacian3d(nx::Integer, ny::Integer, nz::Integer) = laplacian3d(Float64, nx, ny, nz)

"""
    random_spd(T, n, density; rng)

`B Bᴴ + n I` with `B = sprand(T, n, n, density)`: symmetric positive definite
for real `T`, Hermitian positive definite for complex `T`.
"""
function random_spd(::Type{T}, n::Integer, density::Real; rng::AbstractRNG = Random.default_rng()) where {T}
    B = sprand(rng, T, n, n, density)
    A = B * B' + n * I
    return (A + A') / 2  # exact symmetry; the sparse product may round (i,j) and (j,i) differently
end
random_spd(n::Integer, density::Real; kwargs...) = random_spd(Float64, n, density; kwargs...)

"""
    random_hpd(T, n, density; rng)

Hermitian positive definite `B Bᴴ + n I` for complex `T`.
"""
function random_hpd(::Type{T}, n::Integer, density::Real; kwargs...) where {T <: Complex}
    return random_spd(T, n, density; kwargs...)
end
random_hpd(n::Integer, density::Real; kwargs...) = random_hpd(ComplexF64, n, density; kwargs...)

"""
    random_symindef(T, n, density; hermitian = true, rng)

Symmetric (real `T`) or Hermitian (complex `T`, `hermitian = true`) indefinite
matrix: random off-diagonal part `B + Bᴴ` plus a diagonal `dᵢ = ±(rᵢ + 1)`,
where `rᵢ` is the off-diagonal absolute row sum and the signs alternate in a
random order. By Gershgorin, every eigenvalue satisfies `|λ| ≥ 1` and the
inertia is `(count(d .> 0), count(d .< 0))`: nonsingular, both signs present
for `n ≥ 2`, condition number `≤ 2 max(r) + 1`. With `hermitian = false` and
complex `T` the result is complex symmetric (`B + Bᵀ`) instead.
"""
function random_symindef(::Type{T}, n::Integer, density::Real; hermitian::Bool = true,
                         rng::AbstractRNG = Random.default_rng()) where {T}
    B = sprand(rng, T, n, n, density)
    S = _offdiag(hermitian ? B + B' : B + transpose(B))
    r = _row_abs_sums(S)
    signs = shuffle(rng, [isodd(i) ? 1 : -1 for i in 1:n])
    return S + spdiagm(0 => T[signs[i] * (r[i] + 1) for i in 1:n])
end
random_symindef(n::Integer, density::Real; kwargs...) = random_symindef(Float64, n, density; kwargs...)

"""
    kkt_matrix(T, nh, nj, δ; hessian = :spd, density_h, density_j, rng)

KKT matrix `[H Jᴴ; J -δI]` of size `nh + nj` (`nj ≤ nh`). `H` is
`random_spd(T, nh, density_h)` (`hessian = :spd`) or
`random_symindef(T, nh, density_h)` (`hessian = :indefinite`). `J` has full row
rank (its first `nj` columns are row diagonally dominant). The `-δI` block is
always stored, also for `δ = 0`, as in MadNLP's KKT systems. With
`hessian = :spd` and `δ ≥ 0` the inertia is `(nh, nj, 0)`.
"""
function kkt_matrix(::Type{T}, nh::Integer, nj::Integer, δ::Real; hessian::Symbol = :spd,
                    density_h::Real = min(1.0, 5 / nh), density_j::Real = min(1.0, 3 / nh),
                    rng::AbstractRNG = Random.default_rng()) where {T}
    nj <= nh || throw(ArgumentError("kkt_matrix needs nj ≤ nh for a full-row-rank J"))
    H = if hessian === :spd
        random_spd(T, nh, density_h; rng)
    elseif hessian === :indefinite
        random_symindef(T, nh, density_h; rng)
    else
        throw(ArgumentError("hessian must be :spd or :indefinite"))
    end
    R = sprand(rng, T, nj, nh, density_j)
    J = R + spdiagm(nj, nh, 0 => T.(_row_abs_sums(R) .+ 1))
    D = sparse(1:nj, 1:nj, fill(T(-δ), nj), nj, nj)
    return [H J'; J D]
end
kkt_matrix(nh::Integer, nj::Integer, δ::Real; kwargs...) = kkt_matrix(Float64, nh, nj, δ; kwargs...)

"""
    random_general(T, n, density; rng)

Unsymmetric, row diagonally dominant matrix `B + diag(r + 1)` (`r`: absolute
row sums of the off-diagonal part of `B = sprand(T, n, n, density)`), so
`‖A⁻¹‖∞ ≤ 1`.
"""
function random_general(::Type{T}, n::Integer, density::Real; rng::AbstractRNG = Random.default_rng()) where {T}
    B = _offdiag(sprand(rng, T, n, n, density))
    return B + spdiagm(0 => T.(_row_abs_sums(B) .+ 1))
end
random_general(n::Integer, density::Real; kwargs...) = random_general(Float64, n, density; kwargs...)

"""
    singular_block_matrix(T, n, j; stored_zero = false)

Symmetric (Hermitian for complex `T`) `n × n` matrix whose row and column `j`
are zero: an SPD pentadiagonal matrix (diagonal 6, couplings of modulus ≤ 1)
on the other `n - 1` indices, and a structurally zero pivot at `j` (stored as
an explicit zero when `stored_zero = true`). Any elimination order meets a zero
pivot at column `j`, so Cholesky reports `info = j` and LDLᵀ must perturb that
pivot; `b = A x` is a consistent right-hand side. Setting `A[j, j]` to a
positive value makes the matrix SPD, a negative value makes Cholesky fail at `j`.
"""
function singular_block_matrix(::Type{T}, n::Integer, j::Integer; stored_zero::Bool = false) where {T}
    1 <= j <= n || throw(ArgumentError("j must be in 1:n"))
    idx = [1:(j - 1); (j + 1):n]
    c1 = -one(T)
    c2 = T <: Complex ? T(-0.5 + 0.5im) : T(-0.5)
    rows, cols, vals = Int[], Int[], T[]
    for p in eachindex(idx)
        push!(rows, idx[p]); push!(cols, idx[p]); push!(vals, T(6))
        for (off, c) in ((1, c1), (2, c2))
            p + off <= length(idx) || continue
            push!(rows, idx[p + off]); push!(cols, idx[p]); push!(vals, c)
            push!(rows, idx[p]); push!(cols, idx[p + off]); push!(vals, conj(c))
        end
    end
    if stored_zero
        push!(rows, j); push!(cols, j); push!(vals, zero(T))
    end
    return sparse(rows, cols, vals, n, n)
end
singular_block_matrix(n::Integer, j::Integer; kwargs...) = singular_block_matrix(Float64, n, j; kwargs...)

# --- examples from ../CUDSS.jl/docs/src ------------------------------------------

"""
    schur_example_lu(T)

The 5×5 "Schur complement -- LU" example of the CUDSS.jl docs, as a NamedTuple:
`A` (general), `b` (with solution `x = ones`), `schur_indices` (`[0,0,1,1,1]`),
`S = A₂₂ − A₂₁ A₁₁⁻¹ A₁₂` (dense) and `x`.
"""
function schur_example_lu(::Type{T}) where {T}
    rows = [1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5]
    cols = [1, 3, 5, 2, 4, 2, 3, 1, 4, 5, 4, 5]
    vals = T[4, 1, 2, 5, 3, 6, 8, 7, 9, 1, 1, 10]
    A = sparse(rows, cols, vals, 5, 5)
    b = T[7, 8, 14, 17, 11]
    S = T[8 -3.6 0; -1.75 9 -2.5; 0 1 10]
    return (A = A, b = b, schur_indices = [0, 0, 1, 1, 1], S = S, x = ones(T, 5))
end

"""
    schur_example_ldlt(T)

The 5×5 "Schur complement -- LDLᵀ and LDLᴴ" example of the CUDSS.jl docs: `A`
(full symmetric matrix; the docs pass `tril(A)` with view `'L'`), `b` (solution
`ones`), `schur_indices = [1,1,0,0,0]`, `S = A₁₁ − A₁₂ A₂₂⁻¹ A₂₁ = [1 2; 2 -4]`, `x`.
"""
function schur_example_ldlt(::Type{T}) where {T}
    rows = [1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5]
    cols = [1, 3, 5, 2, 4, 1, 3, 2, 4, 5, 1, 4, 5]
    vals = T[4, 1, 2, 2, 3, 1, 3, 3, 2, 1, 2, 1, 2]
    A = sparse(rows, cols, vals, 5, 5)
    b = T[7, 5, 4, 6, 5]
    S = T[1 2; 2 -4]
    return (A = A, b = b, schur_indices = [1, 1, 0, 0, 0], S = S, x = ones(T, 5))
end

"""
    schur_example_cholesky(T)

The 5×5 "Schur complement -- LLᵀ and LLᴴ" example of the CUDSS.jl docs: `A`
(full SPD matrix; the docs pass `triu(A)` with view `'U'`), `b` (solution
`ones`), `schur_indices = [1,1,0,0,0]`, `S = A₁₁ − A₁₂ A₂₂⁻¹ A₂₁ = [2 1; 1 2]`, `x`.
"""
function schur_example_cholesky(::Type{T}) where {T}
    rows = [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5]
    cols = [1, 2, 3, 1, 2, 4, 1, 3, 2, 4, 5]
    vals = T[2.5, 1, 1, 1, 2.5, 1, 1, 2, 1, 2, 2]
    A = sparse(rows, cols, vals, 5, 5)
    b = T[4.5, 4.5, 3, 3, 2]
    S = T[2 1; 1 2]
    return (A = A, b = b, schur_indices = [1, 1, 0, 0, 0], S = S, x = ones(T, 5))
end

"""
    ubatch_example(T)

The 3×3 uniform-batch example of the CUDSS.jl docs,
`A(λ) = [1+λ 0 3; 4 5+λ 0; 2 6 2+λ]` for `λ ∈ Λ = (1, 10, -20)`, as a NamedTuple:
`A` (vector of the three matrices), the shared 1-based CSR pattern `rowptr`,
`colval`, the strided values `nzval` (length `3·7`), the strided right-hand
side `b = 1:9`, `Λ`, `n = 3` and `nbatch = 3`.
"""
function ubatch_example(::Type{T}) where {T}
    Λ = (1, 10, -20)
    rowptr = [1, 3, 5, 8]
    colval = [1, 3, 1, 2, 1, 2, 3]
    nzval = T[v for λ in Λ for v in (1 + λ, 3, 4, 5 + λ, 2, 6, 2 + λ)]
    A = [sparse(T[1+λ 0 3; 4 5+λ 0; 2 6 2+λ]) for λ in Λ]
    b = T.(collect(1:9))
    return (A = A, rowptr = rowptr, colval = colval, nzval = nzval, b = b, Λ = Λ, n = 3, nbatch = 3)
end

# ---------------------------------------------------------------------------
# dense generators (T03)

"""
    dense_hpd(T, n; rng)

Dense Hermitian (real: symmetric) positive definite `n × n` matrix `G Gᴴ + n I`
with `G` standard normal; exactly Hermitian with a real diagonal.
"""
function dense_hpd(::Type{T}, n::Integer; rng::AbstractRNG = Random.default_rng()) where {T}
    G = randn(rng, T, n, n)
    return Matrix{T}(Hermitian(G * G' + n * I))
end

"""
    dense_triangular(T, n; uplo = :L, unit = false, rng)

Well-conditioned dense triangular `n × n` matrix (the other triangle is random
garbage that a triangular solve must ignore): off-diagonal entries standard
normal scaled by `1/n`, diagonal in `[1, 2]` (`unit = true`: the stored
diagonal is random too and must be ignored).
"""
function dense_triangular(::Type{T}, n::Integer; uplo::Symbol = :L, unit::Bool = false,
                          rng::AbstractRNG = Random.default_rng()) where {T}
    A = randn(rng, T, n, n) ./ n
    tri = uplo === :L ? tril(A, -1) : triu(A, 1)
    garbage = uplo === :L ? triu(randn(rng, T, n, n), 1) : tril(randn(rng, T, n, n), -1)
    d = unit ? randn(rng, T, n) : one(T) .+ rand(rng, real(T), n)
    return tri + garbage + Diagonal(d)
end
