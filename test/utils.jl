# Element types, tolerances and residuals shared by every test file
# (TASKS.md "Shared test conventions").

using LinearAlgebra
using SparseArrays

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

# ---------------------------------------------------------------------------
# symbolic engine (T05, T06)

"""
    brute_force_symbolic(A, perm) -> (parent, counts, F)

Dense symbolic elimination of `A[perm, perm]`: the etree (parent = first
off-diagonal nonzero of column `k` of `L`), the column counts (diagonal
included) and the filled pattern `F` (a `BitMatrix`, both triangles).
"""
function brute_force_symbolic(A::SparseMatrixCSC, perm::AbstractVector{<:Integer})
    n = size(A, 1)
    B = A[perm, perm]
    F = falses(n, n)
    for j in 1:n, p in nzrange(B, j)
        F[rowvals(B)[p], j] = true
    end
    for k in 1:n
        F[k, k] = true
    end
    for k in 1:n
        rows = [i for i in (k + 1):n if F[i, k]]
        for i in rows, j in rows
            F[i, j] = true
        end
    end
    parent = zeros(Int, n)
    counts = zeros(Int, n)
    for k in 1:n
        below = findfirst(view(F, (k + 1):n, k))
        parent[k] = below === nothing ? 0 : k + below
        counts[k] = count(view(F, k:n, k))
    end
    return parent, counts, F
end

"""
    offdiag_pattern(A) -> SparseMatrixCSC{Bool}

Off-diagonal stored pattern of `A` as a Boolean sparse matrix.
"""
function offdiag_pattern(A::SparseMatrixCSC)
    I, J, _ = findnz(A)
    keep = I .!= J
    return sparse(I[keep], J[keep], trues(count(keep)), size(A)...)
end

# ---------------------------------------------------------------------------
# numeric phase (T09)

"""
    panel_tol(T)

Elementwise tolerance of device panels against the reference panels, relative
to `max |L|`: `100·eps(real(T))` (TASKS.md T09).
"""
panel_tol(::Type{T}) where {T} = 100 * eps(real(T))

"""
    ka_cpu_alloc_budget(launches, localmem_bytes) -> Int

Bytes the KernelAbstractions CPU backend itself may allocate for `launches`
kernel launches whose `@localmem` buffers total `localmem_bytes`, on top of
the 1024 B allowed to an allocation-free phase (PLAN §3.9). Measured (T09
review round 1): 0 on Julia ≥ 1.12 without coverage (the local runs); on Julia
1.10 every launch boxes its arguments behind KA's `__run` inference barrier
(80–288 B per launch), hence 320 B per launch below 1.12 (1.11 not measured);
with `--code-coverage` (CI's `julia-runtest`) the `@localmem` `MArray`s are
heap-allocated, hence `localmem_bytes` plus 64 B of header per launch.
"""
function ka_cpu_alloc_budget(launches::Integer, localmem_bytes::Integer)
    b = 1024
    VERSION < v"1.12" && (b += 320 * launches)
    Base.JLOptions().code_coverage != 0 && (b += localmem_bytes + 64 * launches)
    return b
end
