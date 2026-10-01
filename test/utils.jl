# Element types, tolerances and residuals shared by every test file
# (TASKS.md "Shared test conventions").

using LinearAlgebra

const ELTYPES = (Float32, Float64, ComplexF32, ComplexF64)
const REAL_ELTYPES = (Float32, Float64)
const COMPLEX_ELTYPES = (ComplexF32, ComplexF64)
const INTTYPES = (Int32, Int64)

"""
    tol(T)

Residual tolerance on well-conditioned generators: `sqrt(eps(real(T)))`.
"""
tol(::Type{T}) where {T} = sqrt(eps(real(T)))

"""
    relres(A, x, b)

Relative residual `‖b - A x‖ / max(‖b‖, 1)`.
"""
relres(A, x, b) = norm(b - A * x) / max(norm(b), one(real(eltype(b))))

"""
    spd_structure(T)

`"SPD"` for real `T`, `"HPD"` for complex `T`.
"""
spd_structure(::Type{T}) where {T} = T <: Real ? "SPD" : "HPD"

"""
    sym_structure(T)

`"S"` for real `T`, `"H"` for complex `T`.
"""
sym_structure(::Type{T}) where {T} = T <: Real ? "S" : "H"

"""
    eigen_inertia(A; atol = 1e-8) -> (npos, nneg, nzero)

Eigenvalue sign counts of a small symmetric/Hermitian matrix (dense eigensolver).
"""
function eigen_inertia(A; atol = 1.0e-8)
    λ = eigvals(Hermitian(Matrix(A)))
    return (count(>(atol), λ), count(<(-atol), λ), count(x -> abs(x) <= atol, λ))
end

"""
    thrown(f) -> Union{Exception, Nothing}

The exception `f()` throws, or `nothing`; for checking exception messages.
"""
function thrown(f)
    try
        f()
    catch e
        return e
    end
    return nothing
end

# ---------------------------------------------------------------------------
# dense layer (T03)

"""
    dense_tol(T)

Relative tolerance of the dense-op tests: `50·eps(real(T))`, applied to a
Frobenius norm of the operands (TASKS.md T03).
"""
dense_tol(::Type{T}) where {T} = 50 * eps(real(T))

"""
    DENSE_SIZES

`(m, n, k)` shapes every dense op is tested on (TASKS.md T03).
"""
const DENSE_SIZES = ((1, 1, 1), (7, 5, 3), (32, 32, 32), (100, 64, 33), (257, 17, 9))

"""
    ipiv_permutation(ipiv, m) -> Vector{Int}

Row permutation `p` of the LAPACK pivot sequence `ipiv` (host vector): applying
the interchanges `i ↔ ipiv[i]` to `1:m` in order, so that `A[p, :] == P * A`.
"""
function ipiv_permutation(ipiv::AbstractVector{<:Integer}, m::Integer)
    p = collect(1:m)
    for (i, q) in enumerate(ipiv)
        p[i], p[q] = p[q], p[i]
    end
    return p
end
