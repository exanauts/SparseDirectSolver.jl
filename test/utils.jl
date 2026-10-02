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

"""
    triangle_view(A, view) -> SparseMatrixCSC

The part of `A` stored for matrix view `view` (`'L'`: `tril(A)`, `'U'`: `triu(A)`, `'F'`: `A`).
"""
triangle_view(A, view) = view == 'L' ? tril(A) : view == 'U' ? triu(A) : A

"""
    numeric_setup(backend, A, INT = Int32; view = 'L', index = 'O', opts) -> (S, Nr, info_ref, Sd, Nd, nz)

Host analysis `S` of the `view` triangle of the SPD/HPD matrix `A` (index base
`index`, options `opts`), its reference factor `Nr` and `info_ref`
(`ref_factorize!`), the analysis adapted to `backend` with `INT` maps, device
storage `Nd` and the values `nz` on `backend` (T09, T10).
"""
function numeric_setup(backend, A::SparseMatrixCSC{T}, ::Type{INT} = Int32; view = 'L', index = 'O',
                       opts = Options(subtree_budgets = Int[])) where {T, INT}
    C = SparseDirectSolver.CSR(triangle_view(A, view); index)
    S = SparseDirectSolver.symbolic_analysis(C, spd_structure(T), view; opts)
    Nr = SparseDirectSolver.allocate_numeric(S, T)
    info_ref = SparseDirectSolver.ref_factorize!(Nr, S, C.nzval)
    Sd = SparseDirectSolver.adapt(backend, S, INT)
    Nd = SparseDirectSolver.allocate_numeric(Sd, T, backend)
    return S, Nr, info_ref, Sd, Nd, to_device(backend, C.nzval)
end

"""
    panel_error(Nh, Nr)

`max |L_device - L_ref| / max |L_ref|` of a host copy `Nh` of a device factor
and the reference factor `Nr`; compare with [`panel_tol`](@ref).
"""
panel_error(Nh, Nr) = maximum(abs, Nh.factor - Nr.factor) / maximum(abs, Nr.factor)

"""
    numeric_alloc_budget(S, T) -> Int

[`ka_cpu_alloc_budget`](@ref) for one `factorize!` of the analysis `S` with
element type `T` and `impl = :auto` on the CPU backend: at most three
assembly launches per launch group, one `ka_chol_check_info!` and one
`pack_add!` per front and the statistics launch; `@localmem` of the statistics
kernel plus, per group, that of the largest fused regime-B kernel (packed
64×64 triangle, status, pivot) or of the largest regime-A kernel (48 KiB).
"""
function numeric_alloc_budget(S, ::Type{T}) where {T}
    ngroups = length(S.schedule.groups)
    launches = 3 * ngroups + 2 * SparseDirectSolver.nsupernodes(S) + 1
    localmem = SparseDirectSolver.STATS_WORKGROUP * sizeof(Int64) +
               ngroups * max(64 * 65 ÷ 2 * sizeof(T) + sizeof(Int32) + sizeof(real(T)),
                             maximum(SparseDirectSolver.SUBTREE_LOCAL_SIZES))
    return ka_cpu_alloc_budget(launches, localmem)
end

"""
    pack_lower(C) -> Vector

Column-major packed lower triangle of the square matrix `C` (`m(m+1)/2`
entries), the storage of contribution blocks on the update stack (T11).
"""
pack_lower(C::AbstractMatrix) = [C[i, j] for j in axes(C, 2) for i in j:size(C, 1)]

"""
    unpack_lower(v, m) -> Matrix

The `m×m` lower-triangular matrix (zero strict upper triangle) of the packed
vector `v` ([`pack_lower`](@ref)).
"""
function unpack_lower(v::AbstractVector{T}, m::Integer) where {T}
    C = zeros(T, m, m)
    q = 1
    for j in 1:m, i in j:m
        C[i, j] = v[q]
        q += 1
    end
    return C
end

# ---------------------------------------------------------------------------
# solve phase (T12)

"""
    variant_tol(T)

Relative distance allowed between the atomic and the deterministic forward
sweep, and between a multi-RHS solve and column-by-column solves:
`10·eps(real(T))` of `‖x‖` (TASKS.md T12).
"""
variant_tol(::Type{T}) where {T} = 10 * eps(real(T))

"""
    solve_setup(backend, A, INT = Int32; opts, nrhs = 5) -> (S, Sd, Nd, ws)

[`numeric_setup`](@ref) of the SPD/HPD matrix `A` (lower triangle) followed
by the device factorization (asserted to succeed) and a solve workspace for
`nrhs` right-hand sides on `backend` (T12).
"""
function solve_setup(backend, A::SparseMatrixCSC{T}, ::Type{INT} = Int32; opts = Options(),
                     nrhs::Integer = 5) where {T, INT}
    S, _, _, Sd, Nd, nz = numeric_setup(backend, A, INT; opts)
    SparseDirectSolver.factorize!(Nd, Sd, nz) == 0 || error("solve_setup: the factorization failed")
    return S, Sd, Nd, SparseDirectSolver.allocate_solve(Sd, T, backend, nrhs)
end

"""
    device_solve(backend, ws, Sd, Nd, b; kwargs...) -> host solution

Copy `b` to `backend`, run `sweep_solve!` with `kwargs` and return the solution
on the host.
"""
function device_solve(backend, ws, Sd, Nd, b::AbstractArray; kwargs...)
    bd = to_device(backend, b)
    return to_host(SparseDirectSolver.sweep_solve!(similar(bd), ws, Sd, Nd, bd; kwargs...))
end

"""
    solve_alloc_budget(S, ws) -> Int

[`ka_cpu_alloc_budget`](@ref) for one `sweep_solve!` with the plan of `ws`
on the CPU backend: two permutation launches, a forward and a backward launch
per kernel launch of the plan, and per regime-C-path front up to three dense
calls or kernels each way plus one pull launch per dense step; `@localmem` of
the control array (`_SV_CTL` indices) per launch.
"""
function solve_alloc_budget(S, ws)
    plan = ws.plan
    launches = 2
    for k in eachindex(plan.kind)
        launches += plan.kind[k] == SparseDirectSolver.SOLVE_DENSE ? 6 * (plan.last[k] - plan.first[k] + 1) + 1 : 2
    end
    return ka_cpu_alloc_budget(launches, launches * SparseDirectSolver._SV_CTL * sizeof(Int64))
end
