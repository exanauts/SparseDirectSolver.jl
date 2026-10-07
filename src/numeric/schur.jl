# Schur complement mode (PLAN §3.6): the Schur rows and columns are ordered
# last and form the root supernode (`schedule.schur`), a regime-C front alone in
# the last launch group. The numeric phase factors every other front as usual
# and only *assembles* the root: zero, scatter of A, extend-add of the children's
# contribution blocks. The assembled root panel is then the Schur complement
# `S = A₂₂ − A₂₁ A₁₁⁻¹ A₁₂` (its lower triangle; LU: the lower triangle in the
# `L` panel and the strict upper triangle transposed in the `Uᵀ` panel), in the
# increasing original order of the Schur indices. Its factor columns get the
# identity pivot order and unit 1×1 pivots, so the diagonal sweep is the
# identity on the Schur block, and its statistics are zero.

"Workgroup size of the Schur kernels (metadata, LU assembly, export)."
const SCHUR_WORKGROUP = 256

# the Schur root's factor columns c0:(c0 + w - 1): identity local order, unit 1×1 pivots, zero statistics
@kernel function _schur_meta_kernel!(piv, d, pivot_kind, stats, info, s, c0, w, n, ::Val{NF}) where {NF}
    q = @index(Global, Linear)
    @inbounds begin
        if q <= w
            c = c0 + q - 1
            piv[c] = Int32(c)
            d[c] = one(eltype(d))
            d[n + c] = zero(eltype(d))
            pivot_kind[c] = PIVOT_KIND_1X1
        end
        q <= NF && (stats[(s - 1) * NF + q] = 0)
        q == 1 && (info[s] = Int32(0))
    end
end

# LU: zero both structures of the root, scatter A, extend-add the children's two contribution blocks
@kernel function _schur_assemble_lu_kernel!(factor, ufactor, stack, ustack, nzval, amap, amap_ptr, amap_src, s,
                                            front_ptr, front_nrows, front_ncols, cb_ptr, child_ptr, child_list,
                                            relind_ptr, relind, maxchild, ::Val{WG}) where {WG}
    li = @index(Local, Linear)
    _zero_front!(factor, stack, s, li, front_ptr, front_nrows, front_ncols, cb_ptr, Val(WG))
    _zero_front!(ufactor, ustack, s, li, front_ptr, front_nrows, front_ncols, cb_ptr, Val(WG))
    @synchronize
    _scatter_front_lu!(factor, ufactor, nzval, amap, amap_ptr, amap_src, s, 0, li, Val(WG))
    @synchronize
    for kc in 1:maxchild
        _extend_add_child!(factor, stack, s, kc, li, front_ptr, front_nrows, front_ncols, cb_ptr, child_ptr,
                           child_list, relind_ptr, relind, Val(WG))
        _extend_add_child!(ufactor, ustack, s, kc, li, front_ptr, front_nrows, front_ncols, cb_ptr, child_ptr,
                           child_list, relind_ptr, relind, Val(WG))
        @synchronize
    end
end

"""
    assemble_schur!(numeric, symbolic, nzval) -> numeric

Schur complement mode (PLAN §3.6): assemble the Schur root front
(`symbolic.schedule.schur`, run by [`factorize!`](@ref) after every other
front) without factoring it: zero its panel, scatter the values `nzval` of A
into it and extend-add its children's contribution blocks
([`zero_fronts!`](@ref), [`scatter_A!`](@ref), [`extend_add!`](@ref); for LU
one kernel doing the same on the `L` and `Uᵀ` structures). The panel then
holds the Schur complement `S` ([`schur_matrix!`](@ref)). The root's factor
columns get the identity local pivot order, `d = 1` (unit 1×1 pivots, so
[`diagonal_sweep!`](@ref) leaves the Schur block unchanged), zero statistics
and status `0`. A no-op without a Schur root. Single matrices only (Schur
mode refuses uniform batches). Asynchronous.
"""
function assemble_schur!(N::Numeric, S::Symbolic, nzval::AbstractVector)
    plan = N.plan
    a = plan.schur_first
    a == 0 && return N
    N.nbatch == 1 || throw(NotSupportedError("Schur complement mode does not support uniform batches"))
    s = S.schedule.schur
    backend = KernelAbstractions.get_backend(N.factor)
    WG = SCHUR_WORKGROUP
    if S.structure == STRUCTURE_GENERAL
        _schur_assemble_lu_kernel!(backend, WG)(N.factor, N.ufactor, N.stack, N.ustack, nzval, S.amap, S.amap_ptr,
                                                S.amap_src, s, S.front_ptr, S.front_nrows, S.front_ncols, S.cb_ptr,
                                                S.child_ptr, S.child_list, S.relind_ptr, S.relind,
                                                plan.schur_maxchild, Val(WG); ndrange = WG)
    else
        zero_fronts!(N, S, a, 1)
        scatter_A!(N, S, nzval, a, 1)
        extend_add!(N, S, a, 1, plan.schur_maxchild)
    end
    sp = S.partition
    w = snwidth(sp, s)
    NF = FRONT_STATS_FIELDS
    _schur_meta_kernel!(backend, WG)(N.piv, N.d, N.pivot_kind, N.stats, N.info, s, sp.super_ptr[s], w, S.n, Val(NF);
                                     ndrange = max(w, NF))
    return N
end

# ---------------------------------------------------------------------------
# export

# entry (i, j) of the Schur complement of the stored matrix from the root panel at p0 (ns × ns, column-major):
# MODE 0 symmetric (mirror the lower triangle), 1 Hermitian (mirror conjugated), 2 LU (strict upper in Uᵀ)
@inline function _schur_stored(factor, ufactor, p0, ns, i, j, ::Val{MODE}) where {MODE}
    @inbounds begin
        i >= j && return factor[p0 + (j - 1) * ns + i - 1]
        v = MODE == 2 ? ufactor[p0 + (i - 1) * ns + j - 1] : factor[p0 + (i - 1) * ns + j - 1]
        return MODE == 1 ? conj(v) : v
    end
end

# the user's Schur complement: the stored one, transposed for a transposed (CSC) input (TR)
@inline _schur_entry(factor, ufactor, p0, ns, i, j, mode::Val, ::Val{TR}) where {TR} =
    TR ? _schur_stored(factor, ufactor, p0, ns, j, i, mode) : _schur_stored(factor, ufactor, p0, ns, i, j, mode)

@kernel function _schur_dense_kernel!(D, factor, ufactor, p0, ns, mode::Val, tr::Val)
    q = @index(Global, Linear)
    @inbounds if q <= ns * ns
        j = (q - 1) ÷ ns + 1
        i = q - (j - 1) * ns
        D[i, j] = _schur_entry(factor, ufactor, p0, ns, i, j, mode, tr)
    end
end

# one work item per row of the CSR destination (its rowptr/colval already written, base `base`)
@kernel function _schur_csr_kernel!(nzval, rowptr, colval, base, factor, ufactor, p0, ns, mode::Val, tr::Val)
    i = @index(Global, Linear)
    @inbounds if i <= ns
        for p in (Int(rowptr[i]) + 1 - base):(Int(rowptr[i + 1]) - base)
            j = Int(colval[p]) + 1 - base
            nzval[p] = _schur_entry(factor, ufactor, p0, ns, i, j, mode, tr)
        end
    end
end

_schur_mode(S::Symbolic, ::Type{T}) where {T} =
    S.structure == STRUCTURE_GENERAL ? Val(2) : (_is_hermitian(S.structure) && T <: Complex) ? Val(1) : Val(0)

"""
    schur_matrix!(dest, numeric, symbolic; transposed = false) -> dest

Write the Schur complement assembled by [`assemble_schur!`](@ref) into
`dest`: an `ns × ns` dense matrix on the backend of `numeric` (every entry), or a
[`CSR`](@ref) whose `rowptr` and `colval` already hold the pattern to export
(the solver writes them, see `getparam(solver, "schur_matrix")`): only its
values are written. `transposed`: the solver's matrix is a transposed CSR (a
CSC input), whose stored Schur complement is the transpose of the user's.
One launch; asynchronous.
"""
function schur_matrix!(D::AbstractMatrix, N::Numeric{T}, S::Symbolic; transposed::Bool = false) where {T}
    s = S.schedule.schur
    s > 0 || throw(InvalidValueError("the analysis has no Schur complement (schur_mode = 0)"))
    ns = snwidth(S.partition, s)
    size(D) == (ns, ns) || throw(DimensionMismatch("the Schur complement is $ns × $ns, the matrix $(size(D))"))
    ns > 0 || return D
    p0 = S.layout.panel_ptr[s]
    kernel! = _schur_dense_kernel!(KernelAbstractions.get_backend(D), SCHUR_WORKGROUP)
    if transposed
        kernel!(D, N.factor, N.ufactor, p0, ns, _schur_mode(S, T), Val(true); ndrange = ns * ns)
    else
        kernel!(D, N.factor, N.ufactor, p0, ns, _schur_mode(S, T), Val(false); ndrange = ns * ns)
    end
    return D
end

function schur_matrix!(D::CSR, N::Numeric{T}, S::Symbolic; transposed::Bool = false) where {T}
    s = S.schedule.schur
    s > 0 || throw(InvalidValueError("the analysis has no Schur complement (schur_mode = 0)"))
    ns = snwidth(S.partition, s)
    size(D) == (ns, ns) || throw(DimensionMismatch("the Schur complement is $ns × $ns, the matrix $(size(D))"))
    ns > 0 || return D
    p0 = S.layout.panel_ptr[s]
    base = D.index == INDEX_ZERO ? 0 : 1
    kernel! = _schur_csr_kernel!(KernelAbstractions.get_backend(D.nzval), SCHUR_WORKGROUP)
    if transposed
        kernel!(D.nzval, D.rowptr, D.colval, base, N.factor, N.ufactor, p0, ns, _schur_mode(S, T), Val(true);
                ndrange = ns)
    else
        kernel!(D.nzval, D.rowptr, D.colval, base, N.factor, N.ufactor, p0, ns, _schur_mode(S, T), Val(false);
                ndrange = ns)
    end
    return D
end
