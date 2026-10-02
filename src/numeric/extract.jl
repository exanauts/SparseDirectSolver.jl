# Copy a device factorization back to the host (tests, the oracle comparison,
# `extract_L`). Not part of the numeric phase: it allocates and synchronizes.

"""
    host_numeric(numeric) -> Numeric{T, Vector{T}}

Host copy of `numeric` (panels, D, update stack, statistics and status; the
plan is shared), usable by [`ref_solve!`](@ref) and [`extract_L`](@ref).
A host `numeric` is returned as is.
"""
host_numeric(N::Numeric{T, Vector{T}, Vector{Int64}, Vector{Int32}}) where {T} = N
function host_numeric(N::Numeric{T}) where {T}
    factor, d, stack = Array(N.factor), Array(N.d), Array(N.stack)
    stats, info = Array(N.stats), Array(N.info)
    return Numeric{T, Vector{T}, Vector{Int64}, Vector{Int32}}(factor, d, stack, stats, info, N.plan)
end

"""
    panel(symbolic, numeric, s) -> Matrix

Host copy of the `f×w` panel of supernode `s` (rows `snrows(s)`, the factor
columns of `s`; the strict upper triangle of its first `w` rows is zero).
"""
function panel(S::Symbolic, N::Numeric{T}, s::Integer) where {T}
    L, sc = S.layout, S.schedule
    f, w = sc.rows[s], sc.width[s]
    return reshape(Array(view(N.factor, L.panel_ptr[s]:(L.panel_ptr[s + 1] - 1))), f, w)
end
