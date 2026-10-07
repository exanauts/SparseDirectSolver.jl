# CPU reference multifrontal LU (PLAN §2.2, §3.3, M7; the oracle of the device
# LU of `src/numeric/lu.jl`): plain Julia on host arrays, the traversal,
# assembly and extend-add of the reference Cholesky on the symmetric pattern of
# `A + Aᵀ`, with a dense in-front `L D U` factorization that pivots on rows
# inside the fully-summed block of each front (threshold partial pivoting) and
# replaces pivots that are too small by `±ε` (static perturbation, as
# SuperLU_DIST's GESP). No delayed pivots: the symbolic structure never changes.
#
# Storage (see `Numeric`): two lower-triangular structures with the layout of
# the Cholesky factor. `factor` holds the unit-lower `L` (front rows after the
# local row interchanges), `ufactor` holds `Uᵀ` (unit diagonal; the panel of `s`
# is `U[c0:c0+w-1, rows of s]ᵀ`, columns not permuted), `d[k]` the pivot `D[k, k]`
# and `piv[k]` the row of `P A Pᵀ` (supernodal numbering) eliminated at factor
# column `k`. Contribution blocks of the device path are two packed triangles
# (the lower triangle with the diagonal in `stack`, the transposed strict upper
# triangle in `ustack`); the reference keeps full host matrices.

# resolved pivoting parameters of one LU factorization
struct _LUParams{R}
    pivot::Bool            # in-block threshold partial pivoting (row interchanges)
    u::R                   # pivot_threshold
    eps::R                 # effective perturbation ε (scaled when pivot_epsilon_alg = "algo1")
end

"""
    lu_pivoting(pivot_type::PivotType) -> Bool

Whether the LU of structure `"G"` interchanges rows inside the fully-summed
block of a front: `'A'` (default), `'L'` and `'B'` mean in-block threshold
partial pivoting; `'N'` and `'D'` keep the diagonal pivots (perturbed when
tiny). Global pivoting (`'C'`, `'R'`) is not planned (PLAN §3.3).
"""
function lu_pivoting(p::PivotType)
    (p == PIVOT_NONE || p == PIVOT_DIAGONAL) && return false
    p == PIVOT_AUTO || p == PIVOT_BUNCH_KAUFMAN || p == PIVOT_LOCAL_BLOCK ||
        throw(NotSupportedError("pivot_type = $(repr(convert(Char, p))) is not supported by LU"))
    return true
end

function _lu_params(S::Symbolic, ::Type{T}, opts::Options, nz::AbstractVector) where {T}
    S.structure == STRUCTURE_GENERAL ||
        throw(InvalidValueError("LU needs structure \"G\", got \"$(convert(String, S.structure))\""))
    R = real(T)
    eps = R(resolved_pivot_epsilon(opts, R))
    if opts.pivot_epsilon_alg == PIVOT_EPSILON_SCALED
        amax = maximum(abs, nz; init = zero(R))
        amax > 0 && (eps *= R(amax))
    end
    return _LUParams{R}(lu_pivoting(opts.pivot_type), R(opts.pivot_threshold), eps)
end

"""
    ref_lu!(numeric::Numeric{T, Vector{T}}, symbolic, nzval; opts = Options()) -> info

Reference multifrontal `L D U` of the general matrix whose stored values are
`nzval` (structure `"G"`, same CSR pattern and index base as the analysis)
into the host `numeric`. The supernodes of the symmetric pattern of `A + Aᵀ`
are processed in postorder: the full `f×f` front is assembled (A through the
`amap`, whose negative offsets are the entries above the diagonal; the
children's full contribution blocks through `relind`), its `w` fully-summed
columns are factored with row interchanges inside the fully-summed rows, and
the contribution block is updated once with `F₂₂ ← F₂₂ − (L₂₁ D) U₁₂`.

Pivot choice at step `k` (`opts.pivot_type`, see [`lu_pivoting`](@ref)): with
`γ = maxᵢ |F[i, k]|` over all remaining rows of the front (fully-summed or
not), the diagonal is kept when `|F[k, k]| ≥ u γ` (`u = pivot_threshold`),
otherwise the fully-summed row `r ≥ k` with the largest `|F[r, k]|` (the first
one) is swapped into row `k`; it is taken even when it fails the threshold
(static pivoting: no delayed pivots). A pivot with `|d| < ε` becomes
`d ← sign(d) ε` (`+ε` for `d = 0`; `d/|d|` for complex `T`), with `ε` as in
[`ref_ldlt!`](@ref) (`pivot_epsilon`, scaled by `max |aᵢⱼ|` with
`pivot_epsilon_alg = "algo1"`); `pivot_sign` is not used by LU.

Fills `numeric.factor` (unit-lower `L` panels), `numeric.ufactor` (unit-lower
`Uᵀ` panels), `numeric.d`, `numeric.piv` (local row order), `numeric.pivot_kind`
(`PIVOT_KIND_1X1` or `PIVOT_KIND_PERTURBED`) and the per-front `numeric.stats`
(`nzero`, `nperturbed`; `npos`, `nneg` and `n2x2` stay 0: a general matrix has
no inertia). An exactly zero pivot is always tiny; with `ε = 0` it stays zero
and the factorization fails there: the front's `info` statistic is the failed
local column and the return value the original column of the failed pivot with
the smallest factor column (as [`ref_ldlt!`](@ref)). Otherwise returns 0: with
static perturbation `ε > 0` the factorization always completes.
"""
function ref_lu!(N::Numeric{T, Vector{T}}, S::Symbolic, nzval::AbstractVector; opts::Options = Options()) where {T}
    length(nzval) == S.nnz ||
        throw(InvalidValueError("nzval has $(length(nzval)) entries, the analysis expects $(S.nnz)"))
    nz = _host_vector(nzval)
    prm = _lu_params(S, T, opts, nz)
    amap = _host_vector(S.amap)
    amap_ptr = _host_vector(S.amap_ptr)
    amap_src = _host_vector(S.amap_src)
    child_ptr = _host_vector(S.child_ptr)
    child_list = _host_vector(S.child_list)
    relind_ptr = _host_vector(S.relind_ptr)
    relind = _host_vector(S.relind)
    sp, L = S.partition, S.layout
    ns = nsupernodes(sp)
    n = S.n
    length(N.ufactor) == length(N.factor) ||
        throw(InvalidValueError("the numeric storage was not allocated for structure \"G\""))
    fill!(N.factor, zero(T))
    fill!(N.ufactor, zero(T))
    fill!(N.d, zero(T))
    fill!(N.stats, 0)
    fill!(N.piv, Int32(0))
    fill!(N.pivot_kind, Int8(0))
    cbs = Vector{Matrix{T}}(undef, ns)
    for s in sp.snpost
        f = sp.rowptr[s + 1] - sp.rowptr[s]
        w = snwidth(sp, s)
        m = f - w
        c0 = sp.super_ptr[s]
        base = L.panel_ptr[s]
        F = zeros(T, f, f)
        for k in amap_ptr[s]:(amap_ptr[s + 1] - 1)
            p = amap_src[k]
            off = Int(amap[p])
            loc = abs(off) - base
            i, j = loc % f + 1, loc ÷ f + 1
            off > 0 ? (F[i, j] += T(nz[p])) : (F[j, i] += T(nz[p]))
        end
        for k in child_ptr[s]:(child_ptr[s + 1] - 1)
            c = child_list[k]
            isassigned(cbs, c) || continue
            C = cbs[c]
            ri = view(relind, relind_ptr[c]:(relind_ptr[c + 1] - 1))
            for jj in axes(C, 2), ii in axes(C, 1)
                F[ri[ii], ri[jj]] += C[ii, jj]
            end
            cbs[c] = Matrix{T}(undef, 0, 0)
        end
        lp, kind, dd, st = _lu_front!(F, w, prm)
        for k in 1:w
            g = c0 + k - 1
            N.piv[g] = Int32(c0 + lp[k] - 1)
            N.pivot_kind[g] = kind[k]
            N.d[g] = dd[k]
        end
        for q in 1:FRONT_STATS_FIELDS
            N.stats[(s - 1) * FRONT_STATS_FIELDS + q] = st[q]
        end
        if m > 0
            # F₂₂ ← F₂₂ − (L₂₁ D) U₁₂, summed in the order of the device kernels
            for jj in 1:m, ii in 1:m
                acc = zero(T)
                for k in 1:w
                    acc += F[w + ii, k] * dd[k] * F[k, w + jj]
                end
                F[w + ii, w + jj] -= acc
            end
            sp.snparent[s] != 0 && (cbs[s] = F[(w + 1):f, (w + 1):f])
        end
        # unit-lower panels of L and Uᵀ
        for j in 1:w
            N.factor[base + (j - 1) * f + j - 1] = one(T)
            N.ufactor[base + (j - 1) * f + j - 1] = one(T)
            for i in (j + 1):f
                N.factor[base + (j - 1) * f + i - 1] = F[i, j]
                N.ufactor[base + (j - 1) * f + i - 1] = F[j, i]
            end
        end
    end
    _host_totals!(N)
    return _host_info(N, S)
end

ref_lu!(N::Numeric, S::Symbolic, A::CSR; kwargs...) = ref_lu!(N, S, vec(A.nzval); kwargs...)

# Dense L D U of the fully-summed columns 1:w of the full front F with row interchanges among the rows 1:w
# (columns 1:f of the fully-summed rows and rows 1:f of the fully-summed columns are read and updated; the
# contribution block is left to the caller). On return the strict lower part of F[:, 1:w] is L, the strict
# upper part of F[1:w, :] is U (both unit, scaled by D), F[k, k] = D[k]. Returns the local row order `lp`
# (`lp[k]` = assembled row eliminated at k), the pivot kinds, D and the statistics.
function _lu_front!(F::AbstractMatrix{T}, w::Int, prm::_LUParams) where {T}
    f = size(F, 1)
    lp = collect(1:w)
    kind = fill(PIVOT_KIND_1X1, w)
    dd = zeros(T, w)
    st = zeros(Int64, FRONT_STATS_FIELDS)                # nzero, nperturbed (and the zeros), info
    for k in 1:w
        r = prm.pivot ? _lu_choose_row(F, k, w, prm.u) : k
        if r != k
            for j in 1:f
                F[k, j], F[r, j] = F[r, j], F[k, j]
            end
            lp[k], lp[r] = lp[r], lp[k]
        end
        d = F[k, k]
        if !(abs(d) >= prm.eps) || iszero(d)             # tiny (or NaN, or exactly zero): perturb
            iszero(d) && (st[STAT_NZERO] += 1)
            d = _perturbation_sign(d, 0, !(T <: Complex)) * prm.eps
            kind[k] = PIVOT_KIND_PERTURBED
            st[STAT_NPERTURBED] += 1
            iszero(d) && st[STAT_INFO] == 0 && (st[STAT_INFO] = k)       # ε = 0: failed column
        end
        F[k, k] = d
        dd[k] = d
        for j in (k + 1):w, i in (k + 1):f
            F[i, j] -= F[i, k] / d * F[k, j]
        end
        for j in (w + 1):f, i in (k + 1):w
            F[i, j] -= F[i, k] / d * F[k, j]
        end
        for i in (k + 1):f
            F[i, k] /= d
        end
        for j in (k + 1):f
            F[k, j] /= d
        end
    end
    return lp, kind, dd, st
end

# in-block threshold partial pivoting: the diagonal when |F[k, k]| ≥ u maxᵢ≥ₖ |F[i, k]| (all rows of the front),
# else the first fully-summed row with the largest |F[r, k]|
function _lu_choose_row(F::AbstractMatrix{T}, k::Int, w::Int, u) where {T}
    γ = zero(real(T))
    for i in k:size(F, 1)
        γ = max(γ, abs(F[i, k]))
    end
    akk = abs(F[k, k])
    akk >= u * γ && return k
    r, best = k, akk
    for i in (k + 1):w
        a = abs(F[i, k])
        if a > best
            r, best = i, a
        end
    end
    return r
end

"""
    extract_lu(symbolic, numeric) -> (L, D, U, p, q)

The LU factor in `numeric` (host or device, batch member 1) as host sparse
matrices in factor order: `L` unit lower and `U` unit upper triangular,
`D` the pivots, `p` and `q` the row and column permutations (original
indices, `p[k] = perm[piv[k]]`, `q = perm`), so that
`A[p, q] ≈ L * Diagonal(D) * U` (with the perturbed pivots, `A + E`). Rows of `L`
below a diagonal block are the rows of later supernodes, which their own local
row interchanges moved to `ipiv[row]`.
"""
function extract_lu(S::Symbolic, N::Numeric{T}) where {T}
    S.structure == STRUCTURE_GENERAL || throw(InvalidValueError("extract_lu needs structure \"G\""))
    H = N.nbatch == 1 ? host_numeric(N) : member_numeric(N, S, 1)
    sp, L = S.partition, S.layout
    n = S.n
    piv = Int.(H.piv[1:n])
    ipiv = invperm(piv)
    Li, Lj, Lv = Int[], Int[], T[]
    Ui, Uj, Uv = Int[], Int[], T[]
    for s in 1:nsupernodes(sp)
        rows = snrows(sp, s)
        f = length(rows)
        c0 = sp.super_ptr[s]
        w = snwidth(sp, s)
        base = L.panel_ptr[s]
        for j in 1:w, i in j:f
            g = c0 + j - 1
            l = i == j ? one(T) : H.factor[base + (j - 1) * f + i - 1]
            u = i == j ? one(T) : H.ufactor[base + (j - 1) * f + i - 1]
            push!(Li, i <= w ? c0 + i - 1 : ipiv[rows[i]]); push!(Lj, g); push!(Lv, l)
            push!(Ui, g); push!(Uj, i <= w ? c0 + i - 1 : rows[i]); push!(Uv, u)
        end
    end
    perm = sp.perm
    return sparse(Li, Lj, Lv, n, n), H.d[1:n], sparse(Ui, Uj, Uv, n, n), perm[piv], copy(perm)
end

"""
    ref_solve_lu!(X, symbolic, numeric, B; transpose = false) -> X

Host solve of `A X = B` (`Aᵀ X = B` when `transpose`) with the LU factor in
`numeric` through [`extract_lu`](@ref) (sparse triangular solves; tests and
debugging). `B`, `X`: vectors or matrices.
"""
function ref_solve_lu!(X::AbstractVecOrMat, S::Symbolic, N::Numeric{T}, B::AbstractVecOrMat;
                       transpose::Bool = false) where {T}
    L, D, U, p, q = extract_lu(S, N)
    Bh = Array(B)
    Bm = reshape(Bh, S.n, :)
    Y = similar(Bm)
    if transpose
        # Aᵀ[q, p] = Uᵀ D Lᵀ
        Z = UnitLowerTriangular(sparse(Base.transpose(U))) \ Bm[q, :]
        Y[p, :] = UnitUpperTriangular(sparse(Base.transpose(L))) \ (Diagonal(D) \ Z)
    else
        Z = UnitLowerTriangular(L) \ Bm[p, :]
        Y[q, :] = UnitUpperTriangular(U) \ (Diagonal(D) \ Z)
    end
    copyto!(X, vec(Y))
    return X
end
