# Unblocked right-looking LU with partial pivoting in one workgroup per matrix,
# and the row interchanges `laswp`. Pivots follow LAPACK: `ipiv[j]` is the row
# (1-based, relative to the matrix) swapped with row j at step j; ties pick the
# smallest row index; a zero pivot sets `info` to its column, skips the scaling
# and continues, as `getrf`.

@inline _abs1(x::Real) = abs(x)
@inline _abs1(x::Complex) = abs(real(x)) + abs(imag(x))

@inline _pget(p::AbstractVector, j, b) = @inbounds p[j]
@inline _pget(p::AbstractMatrix, j, b) = @inbounds p[j, b]
@inline _pset!(p::AbstractVector, v, j, b) = (@inbounds p[j] = v; nothing)
@inline _pset!(p::AbstractMatrix, v, j, b) = (@inbounds p[j, b] = v; nothing)

@kernel function _ka_getrf_kernel!(A, ipiv, info, ::Val{WG}, ::Val{LOG2}, m, n) where {WG, LOG2}
    @uniform T = eltype(A)
    @uniform R = real(T)
    rv = @localmem R (WG,)
    ri = @localmem Int32 (WG,)
    st = @localmem Int32 (1,)
    li = @index(Local, Linear)
    gi = @index(Group, NTuple)
    if li == 1
        @inbounds st[1] = Int32(0)
    end
    for j in 1:min(m, n)
        best = R(-1)
        bi = Int32(j)
        for i in (j + li - 1):WG:m
            v = _abs1(_get(A, i, j, gi[2]))
            if v > best
                best = v
                bi = Int32(i)
            end
        end
        @inbounds rv[li] = best
        @inbounds ri[li] = bi
        @synchronize
        for lev in 1:LOG2
            s = WG >> lev
            if li <= s
                o = li + s
                @inbounds if rv[o] > rv[li] || (rv[o] == rv[li] && ri[o] < ri[li])
                    rv[li] = rv[o]
                    ri[li] = ri[o]
                end
            end
            @synchronize
        end
        @inbounds p = Int(ri[1])
        if li == 1
            _pset!(ipiv, p, j, gi[2])
            @inbounds if !(rv[1] > 0) && st[1] == 0
                st[1] = Int32(j)
            end
        end
        if p != j
            for c in li:WG:n
                tmp = _get(A, j, c, gi[2])
                _set!(A, _get(A, p, c, gi[2]), j, c, gi[2])
                _set!(A, tmp, p, c, gi[2])
            end
        end
        @synchronize
        piv = _get(A, j, j, gi[2])
        if !iszero(piv)
            for i in (j + li):WG:m
                _set!(A, _get(A, i, j, gi[2]) / piv, i, j, gi[2])
            end
        end
        @synchronize
        r = m - j
        for q in (li - 1):WG:(r * (n - j) - 1)
            i = j + 1 + q % r
            k = j + 1 + q ÷ r
            _set!(A, _get(A, i, k, gi[2]) - _get(A, i, j, gi[2]) * _get(A, j, k, gi[2]), i, k, gi[2])
        end
        @synchronize
    end
    if li == 1
        @inbounds info[gi[2]] = st[1]
    end
end

"""
    ka_getrf!(A, ipiv, info; workgroup = Val(128)) -> info

KernelAbstractions fallback LU with partial pivoting of the `m × n` matrix `A`
in place (`P A = L U`, unit `L` below the diagonal, `U` on and above), one 1-D
workgroup per matrix. `A` is a matrix (`ipiv` a vector of length `min(m, n)`)
or a 3-D strided batch (`ipiv` a `min(m, n) × count` matrix). `info` is a
device `Int32` vector with one entry per matrix: 0, or the first column with
an exactly zero pivot, as LAPACK `getrf`. Asynchronous.
"""
function ka_getrf!(A::AbstractArray, ipiv::AbstractVecOrMat{<:Integer}, info::AbstractVector{Int32};
                   workgroup::Val{WG} = Val(KA_WORKGROUP)) where {WG}
    m, n = size(A, 1), size(A, 2)
    nb = _nbatch(A)
    size(ipiv, 1) >= min(m, n) || throw(DimensionMismatch("ipiv has $(size(ipiv, 1)) rows < min(m, n) = $(min(m, n))"))
    size(ipiv, 2) >= nb || throw(DimensionMismatch("ipiv has $(size(ipiv, 2)) columns < batch count $nb"))
    length(info) >= nb || throw(DimensionMismatch("info has length $(length(info)) < batch count $nb"))
    nb == 0 && return info
    backend = KernelAbstractions.get_backend(A)
    kernel! = _ka_getrf_kernel!(backend, (WG, 1))
    kernel!(A, ipiv, info, workgroup, Val(_ilog2(WG)), m, n; ndrange = (WG, nb))
    return info
end

@kernel function _ka_laswp_kernel!(A, ipiv, ::Val{REV}, npiv) where {REV}
    c, b = @index(Global, NTuple)
    for t in 1:npiv
        i = REV ? npiv - t + 1 : t
        p = Int(_pget(ipiv, i, b))
        if p != i
            tmp = _get(A, i, c, b)
            _set!(A, _get(A, p, c, b), i, c, b)
            _set!(A, tmp, p, c, b)
        end
    end
end

"""
    ka_laswp!(A, ipiv; reverse = false, workgroup = 128) -> A

KernelAbstractions row interchanges: for `i = 1:length(ipiv)` (in reverse
order if `reverse`), swap rows `i` and `ipiv[i]` of `A`, as LAPACK `laswp`
with `k1 = 1`, `k2 = length(ipiv)`. Applied after [`ka_getrf!`](@ref) this
computes `P A`; `reverse = true` applies `Pᵀ`. One work item per column; `A` may
be a 3-D strided batch with `ipiv` a matrix (one column per member).
"""
function ka_laswp!(A::AbstractArray, ipiv::AbstractVecOrMat{<:Integer}; reverse::Bool = false,
                   workgroup::Integer = KA_WORKGROUP)
    nb = _nbatch(A)
    npiv = size(ipiv, 1)
    npiv <= size(A, 1) || throw(DimensionMismatch("ipiv has $npiv entries but A has $(size(A, 1)) rows"))
    size(ipiv, 2) >= nb || throw(DimensionMismatch("ipiv has $(size(ipiv, 2)) columns < batch count $nb"))
    (size(A, 2) == 0 || nb == 0 || npiv == 0) && return A
    backend = KernelAbstractions.get_backend(A)
    kernel! = _ka_laswp_kernel!(backend, (min(workgroup, size(A, 2)), 1))
    kernel!(A, ipiv, Val(reverse), npiv; ndrange = (size(A, 2), nb))
    return A
end
