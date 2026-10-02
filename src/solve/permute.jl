# Permutation kernels of the solve phase (PLAN §2.5): the right-hand side
# enters the permuted (supernodal) numbering of the factor, `Y[k, r] =
# B[perm[k], r]`, and the solution leaves it, `X[perm[k], r] = Y[k, r]`. One
# work item per entry (entry `(k, r)` is work item `k + n (r - 1)`; 2-D
# ndranges allocate on every launch of the KA CPU backend).
#
# User arrays come in the layouts of a cuDSS dense matrix (`MatrixDescriptor`):
# an `n` vector, a column-major `n × nrhs` matrix, a strided vector of
# `n * nrhs` entries (column-major), or, with `transposed = true`, row-major
# data, i.e. a column-major `nrhs × n` matrix (or the strided vector of one).

"Workgroup size of the permutation kernels."
const PERMUTE_WORKGROUP = 256

# position of entry (k, r) of the logical n × nrhs right-hand side in the user array
@inline _rhs_get(B::AbstractVector, k, r, n, nrhs, ::Val{TR}) where {TR} =
    @inbounds TR ? B[r + (k - 1) * nrhs] : B[k + (r - 1) * n]
@inline _rhs_get(B::AbstractMatrix, k, r, n, nrhs, ::Val{TR}) where {TR} = @inbounds TR ? B[r, k] : B[k, r]
@inline function _rhs_set!(B::AbstractVector, v, k, r, n, nrhs, ::Val{TR}) where {TR}
    @inbounds TR ? (B[r + (k - 1) * nrhs] = v) : (B[k + (r - 1) * n] = v)
    return nothing
end
@inline function _rhs_set!(B::AbstractMatrix, v, k, r, n, nrhs, ::Val{TR}) where {TR}
    @inbounds TR ? (B[r, k] = v) : (B[k, r] = v)
    return nothing
end

"""
    rhs_count(B, n; transposed = false) -> nrhs

Number of right-hand sides of the user array `B` for a system of size `n`:
`1` for an `n` vector, `size(B, 2)` for an `n × nrhs` matrix (`size(B, 1)` for
a transposed, i.e. row-major, `nrhs × n` one), `length(B) ÷ n` for a strided
vector. Raises `DimensionMismatch` when the shape does not fit `n`.
"""
function rhs_count(B::AbstractVecOrMat, n::Integer; transposed::Bool = false)
    if B isa AbstractVector
        (n > 0 ? length(B) % n == 0 : isempty(B)) ||
            throw(DimensionMismatch("a strided right-hand side needs a multiple of n = $n entries, got $(length(B))"))
        return n > 0 ? length(B) ÷ n : 0
    end
    rows, nrhs = transposed ? (size(B, 2), size(B, 1)) : size(B)
    rows == n || throw(DimensionMismatch("the right-hand side is $(size(B))$(transposed ? " (transposed)" : ""), " *
                                         "the system has n = $n"))
    return nrhs
end

@kernel function _permute_rhs_kernel!(Y, B, perm, n, nrhs, ::Val{TR}) where {TR}
    q = @index(Global, Linear)
    k = (q - 1) % n + 1
    r = (q - 1) ÷ n + 1
    @inbounds if r <= nrhs
        Y[k, r] = _rhs_get(B, perm[k], r, n, nrhs, Val(TR))
    end
end

@kernel function _unpermute_solution_kernel!(X, Y, perm, n, nrhs, ::Val{TR}) where {TR}
    q = @index(Global, Linear)
    k = (q - 1) % n + 1
    r = (q - 1) ÷ n + 1
    @inbounds if r <= nrhs
        _rhs_set!(X, Y[k, r], perm[k], r, n, nrhs, Val(TR))
    end
end

function _check_permute(Y, B, perm, transposed)
    n = length(perm)
    nrhs = rhs_count(B, n; transposed)
    size(Y, 1) == n && size(Y, 2) >= nrhs ||
        throw(DimensionMismatch("the permuted right-hand side is $(size(Y)), needs $n × ≥ $nrhs"))
    return n, nrhs
end

"""
    permute_rhs!(Y, B, perm; transposed = false) -> Y

`Y[k, r] = B[perm[k], r]` for the `nrhs` right-hand sides of the user array
`B` ([`rhs_count`](@ref): vector, matrix, strided vector; row-major when
`transposed`) into the first `nrhs` columns of the `n × ≥ nrhs` device matrix
`Y`. One launch; asynchronous.
"""
function permute_rhs!(Y::AbstractMatrix, B::AbstractVecOrMat, perm::AbstractVector; transposed::Bool = false)
    n, nrhs = _check_permute(Y, B, perm, transposed)
    n * nrhs > 0 || return Y
    kernel! = _permute_rhs_kernel!(KernelAbstractions.get_backend(Y), PERMUTE_WORKGROUP)
    if transposed
        kernel!(Y, B, perm, n, nrhs, Val(true); ndrange = n * nrhs)
    else
        kernel!(Y, B, perm, n, nrhs, Val(false); ndrange = n * nrhs)
    end
    return Y
end

"""
    unpermute_solution!(X, Y, perm; transposed = false) -> X

`X[perm[k], r] = Y[k, r]`: the inverse of [`permute_rhs!`](@ref), from the
first `nrhs` columns of `Y` into the user array `X` (same layouts). One launch;
asynchronous.
"""
function unpermute_solution!(X::AbstractVecOrMat, Y::AbstractMatrix, perm::AbstractVector; transposed::Bool = false)
    n, nrhs = _check_permute(Y, X, perm, transposed)
    n * nrhs > 0 || return X
    kernel! = _unpermute_solution_kernel!(KernelAbstractions.get_backend(Y), PERMUTE_WORKGROUP)
    if transposed
        kernel!(X, Y, perm, n, nrhs, Val(true); ndrange = n * nrhs)
    else
        kernel!(X, Y, perm, n, nrhs, Val(false); ndrange = n * nrhs)
    end
    return X
end
