# Copy a device factorization back to the host (tests, the oracle comparison,
# `extract_L`). Not part of the numeric phase: it allocates and synchronizes.

"""
    host_numeric(numeric) -> Numeric{T, Vector{T}}

Host copy of `numeric` (panels, D, update stack, workspace, statistics and totals, status, pivot order, kinds and sign
requests, `aux`; the
plan is shared), usable by [`ref_solve!`](@ref) and [`extract_L`](@ref).
A host `numeric` is returned as is.
"""
host_numeric(N::Numeric{T, Vector{T}, Vector{Int64}, Vector{Int32}, Vector{Int8}}) where {T} = N
function host_numeric(N::Numeric{T}) where {T}
    factor, d, stack, work = Array(N.factor), Array(N.d), Array(N.stack), Array(N.work)
    stats, info, piv, pivot_kind = Array(N.stats), Array(N.info), Array(N.piv), Array(N.pivot_kind)
    return Numeric{T, Vector{T}, Vector{Int64}, Vector{Int32}, Vector{Int8}}(factor, d, stack, work, stats, info, piv,
                                                                             pivot_kind, Array(N.totals),
                                                                             Array(N.psign), Array(N.aux), N.plan,
                                                                             Array(N.members), N.nbatch)
end

"""
    panel(symbolic, numeric, s, k = 1) -> Matrix

Host copy of the `f×w` panel of supernode `s` (rows `snrows(s)`, the factor
columns of `s`; the strict upper triangle of its first `w` rows is zero) of
batch member `k`.
"""
function panel(S::Symbolic, N::Numeric{T}, s::Integer, k::Integer = 1) where {T}
    L, sc = S.layout, S.schedule
    f, w = sc.rows[s], sc.width[s]
    p0 = panel_offset(L.panel_ptr, s, k, N.nbatch)
    return reshape(Array(view(N.factor, p0:(p0 + f * w - 1))), f, w)
end

"""
    member_numeric(numeric, symbolic, k) -> Numeric{T, Vector{T}}

Host copy of batch member `k` of `numeric` as a single-matrix storage (the
layout of `allocate_numeric(symbolic, T)`), for the reference comparisons and
[`extract_L`](@ref).
"""
function member_numeric(N::Numeric{T}, S::Symbolic, k::Integer) where {T}
    nb = N.nbatch
    1 <= k <= nb || throw(InvalidValueError("batch member $k outside 1:$nb"))
    H = host_numeric(N)
    nb == 1 && return H
    L = S.layout
    ns = nsupernodes(S)
    factor = zeros(T, L.factor_len)
    for s in 1:ns
        len = L.panel_ptr[s + 1] - L.panel_ptr[s]
        p0 = panel_offset(L.panel_ptr, s, k, nb)
        copyto!(factor, L.panel_ptr[s], H.factor, p0, len)
    end
    mv(A) = A[((k - 1) * (length(A) ÷ nb) + 1):(k * (length(A) ÷ nb))]
    info = H.info[k:nb:end]
    return Numeric{T, Vector{T}, Vector{Int64}, Vector{Int32}, Vector{Int8}}(factor, mv(H.d), mv(H.stack), mv(H.work),
                                                                             mv(H.stats), info, mv(H.piv),
                                                                             mv(H.pivot_kind), mv(H.totals),
                                                                             copy(H.psign), mv(H.aux),
                                                                             NumericPlan(S), Int32[1], 1)
end
