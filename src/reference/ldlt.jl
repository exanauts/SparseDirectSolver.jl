# CPU reference multifrontal LDLᵀ/LDLᴴ (PLAN §3.3, §7, the oracle of T15):
# plain Julia on host arrays, the same traversal, assembly and extend-add as
# the reference Cholesky (`src/reference/cholesky.jl`), with a dense in-front
# factorization that pivots inside the fully-summed block of each front
# (Bunch–Kaufman 1×1/2×2, `pivot_threshold` acceptance) and replaces pivots
# that are too small by `±ε` (static perturbation with the `pivot_sign`
# policy). No delayed pivots: the symbolic structure never changes.
#
# Storage (see `Numeric`): unit-lower panels; `d[k]` the diagonal of D at
# factor column `k`, `d[n + k]` the subdiagonal entry of a 2×2 block starting at
# `k`; `pivot_kind[k]`; `piv[k]` the column of `P A Pᵀ` (supernodal numbering
# before pivoting) eliminated at factor column `k`. Pivoting permutes only the
# columns of one supernode; the rows of a panel below its diagonal block keep the
# numbering before pivoting, and the solve applies each supernode's local
# order to its slice of the right-hand side when it reaches that supernode (as
# LAPACK's `sytrs` applies the interchanges of `sytrf`).

"""
    BUNCH_KAUFMAN_ALPHA

`α = (1 + √17)/8`, the Bunch–Kaufman constant that minimizes the element growth
bound of 1×1/2×2 pivoting.
"""
const BUNCH_KAUFMAN_ALPHA = (1 + sqrt(17)) / 8

# indices of the per-front statistics (column `s` of `numeric.stats`)
const STAT_NPOS = 1
const STAT_NNEG = 2
const STAT_NZERO = 3
const STAT_NPERTURBED = 4
const STAT_N2X2 = 5
const STAT_INFO = 6

# resolved pivoting parameters of one factorization
struct _LDLTParams{R}
    pivot::PivotType       # PIVOT_NONE, PIVOT_DIAGONAL or PIVOT_BUNCH_KAUFMAN
    u::R                   # pivot_threshold
    eps::R                 # effective perturbation ε (scaled when pivot_epsilon_alg = "algo1")
    herm::Bool             # Hermitian (or real symmetric) vs complex symmetric
end

_is_ldlt_structure(s::Structure) = s == STRUCTURE_SYMMETRIC || s == STRUCTURE_HERMITIAN

function _ldlt_pivot_type(p::PivotType)
    p == PIVOT_NONE && return PIVOT_NONE
    p == PIVOT_DIAGONAL && return PIVOT_DIAGONAL
    # 'A' (auto), 'B' and 'L' (local block): Bunch–Kaufman inside the fully-summed block
    p == PIVOT_AUTO || p == PIVOT_BUNCH_KAUFMAN || p == PIVOT_LOCAL_BLOCK || throw(NotSupportedError(
        "pivot_type = $(repr(convert(Char, p))) is not supported by LDLᵀ/LDLᴴ"))
    return PIVOT_BUNCH_KAUFMAN
end

function _ldlt_params(S::Symbolic, ::Type{T}, opts::Options, nz::AbstractVector) where {T}
    _is_ldlt_structure(S.structure) ||
        throw(InvalidValueError("LDLᵀ/LDLᴴ needs structure \"S\" or \"H\", got \"$(convert(String, S.structure))\""))
    R = real(T)
    ps = opts.pivot_sign
    ps === nothing || length(ps) == S.n ||
        throw(InvalidValueError("pivot_sign has $(length(ps)) entries, the matrix has $(S.n) rows"))
    eps = R(resolved_pivot_epsilon(opts, R))
    if opts.pivot_epsilon_alg == PIVOT_EPSILON_SCALED
        amax = maximum(abs, nz; init = zero(R))
        amax > 0 && (eps *= R(amax))
    end
    herm = !(T <: Complex) || S.structure == STRUCTURE_HERMITIAN
    return _LDLTParams{R}(_ldlt_pivot_type(opts.pivot_type), R(opts.pivot_threshold), eps, herm)
end

"""
    ref_ldlt!(numeric::Numeric{T, Vector{T}}, symbolic, nzval; opts = Options()) -> info

Reference multifrontal `LDLᵀ` (real `T`, or complex `T` with structure `"S"`,
complex symmetric) or `LDLᴴ` (structure `"H"`) of the matrix whose stored values
are `nzval` (same CSR pattern, view and index base as the analysis), into the
host `numeric`. Assembly and extend-add are those of [`ref_factorize!`](@ref);
each front is then factored densely with pivots chosen inside its fully-summed
block (its first `w` columns), and the contribution block is updated once with
`F₂₂ ← F₂₂ − (L₂₁ D) L₂₁ᴴ` (`ᵀ` for complex symmetric).

Pivoting (`opts.pivot_type`; `'A'`, `'B'` and `'L'` are Bunch–Kaufman, the
default):

* `'B'`: Bunch–Kaufman with `α = (1+√17)/8` on the remaining block, choosing a
  1×1 pivot at the current column, a 1×1 pivot at the row `r` of the largest
  block entry of that column, or the 2×2 pivot on both. The choice is
  *acceptable* when it is not tiny (`|d| ≥ ε`, or for a 2×2 block a nonzero
  determinant and an entry `≥ ε`) and passes the threshold test with
  `u = opts.pivot_threshold` against the whole remaining column of the front,
  fully-summed or not (`|d| ≥ u maxᵢ |aᵢc|`; for a 2×2 block, every entry of
  the resulting `L` columns bounded by `1/u` through `|D⁻¹|`). Otherwise the
  acceptable 1×1 pivot of the block with the largest `|aⱼⱼ|` is taken; if there
  is none, the Bunch–Kaufman choice is used anyway when it is not tiny (stable
  inside the block, only the bound on the rows outside fails); else the column
  is perturbed.
* `'D'`: 1×1 pivots only: the current column if acceptable, else the
  acceptable diagonal with the largest `|aⱼⱼ|`, else the largest diagonal if not
  tiny, else the current column, perturbed.
* `'N'`: no search: the current column, perturbed when tiny.

Perturbation: a 1×1 pivot with `|d| < ε` becomes `d ← sign · ε`, where `ε` is
`pivot_epsilon` ([`default_pivot_epsilon`](@ref) if unset), multiplied by
`max |aᵢⱼ|` of `nzval` when `pivot_epsilon_alg = "algo1"` (scaled; `"default"`
and `"algo2"` are static), and `sign` is `opts.pivot_sign[row]` for the
original row of that pivot when given and nonzero, else the sign of `d`
(`+1` for `d == 0`; for complex symmetric `T`, `d/|d|`).

Fills `numeric.factor` (unit-lower panels), `numeric.d`, `numeric.piv`,
`numeric.pivot_kind` and the per-front `numeric.stats` (`npos, nneg, nzero,
nperturbed, n2x2, info`; `nzero` counts pivots that were exactly zero before
their perturbation; `npos`/`nneg` count the signs of D after perturbation, a 2×2
block with negative determinant counting one of each, and stay `0` for complex
symmetric matrices, which have no inertia). Returns `info = 0`: with static
perturbation the factorization always completes.
"""
function ref_ldlt!(N::Numeric{T, Vector{T}}, S::Symbolic, nzval::AbstractVector; opts::Options = Options()) where {T}
    length(nzval) == S.nnz ||
        throw(InvalidValueError("nzval has $(length(nzval)) entries, the analysis expects $(S.nnz)"))
    nz = _host_vector(nzval)
    prm = _ldlt_params(S, T, opts, nz)
    psign = opts.pivot_sign
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
    fill!(N.factor, zero(T))
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
        # assembly of A (lower trapezoid of the panel) and extend-add, as in the Cholesky
        for k in amap_ptr[s]:(amap_ptr[s + 1] - 1)
            p = amap_src[k]
            off = Int(amap[p])
            loc = abs(off) - base
            F[loc % f + 1, loc ÷ f + 1] += off < 0 ? conj(T(nz[p])) : T(nz[p])
        end
        for k in child_ptr[s]:(child_ptr[s + 1] - 1)
            c = child_list[k]
            isassigned(cbs, c) || continue
            C = cbs[c]
            ri = view(relind, relind_ptr[c]:(relind_ptr[c + 1] - 1))
            for jj in axes(C, 2), ii in jj:size(C, 1)
                F[ri[ii], ri[jj]] += C[ii, jj]
            end
            cbs[c] = Matrix{T}(undef, 0, 0)
        end
        # the in-front factorization reads both triangles of the fully-summed columns
        for j in 1:w, i in (j + 1):w
            F[j, i] = prm.herm ? conj(F[i, j]) : F[i, j]
        end
        # sign requested for the pivot at local column c (original row perm[c0 + c - 1])
        signof = c -> psign === nothing ? 0 : Int(psign[sp.perm[c0 + c - 1]])
        lp, kind, dd, ee, st = _ldlt_front!(F, w, prm, signof)
        for k in 1:w
            g = c0 + k - 1
            N.piv[g] = Int32(c0 + lp[k] - 1)
            N.pivot_kind[g] = kind[k]
            N.d[g] = dd[k]
            N.d[n + g] = ee[k]
        end
        for q in 1:5
            N.stats[(s - 1) * FRONT_STATS_FIELDS + q] = st[q]
        end
        if m > 0
            # F₂₂ ← F₂₂ − W L₂₁ᴴ with W = L₂₁ D
            L21 = F[(w + 1):f, 1:w]
            W = _times_d(L21, dd, ee, kind, prm.herm)
            F22 = view(F, (w + 1):f, (w + 1):f)
            mul!(F22, W, prm.herm ? L21' : transpose(L21), -one(T), one(T))
            sp.snparent[s] != 0 && (cbs[s] = F[(w + 1):f, (w + 1):f])
        end
        # unit-lower panel
        for j in 1:w
            N.factor[base + (j - 1) * f + j - 1] = one(T)
            for i in (j + 1):f
                N.factor[base + (j - 1) * f + i - 1] = F[i, j]
            end
        end
    end
    _host_totals!(N)
    return 0
end

# the totals of the per-front statistics (host storage; `reduce_stats!` on the device)
function _host_totals!(N::Numeric)
    for q in 1:FRONT_STATS_FIELDS
        N.totals[q] = sum((N.stats[(s - 1) * FRONT_STATS_FIELDS + q] for s in 1:(length(N.stats) ÷ FRONT_STATS_FIELDS));
                          init = Int64(0))
    end
    N.totals[STAT_INFO] = count(s -> N.stats[(s - 1) * FRONT_STATS_FIELDS + STAT_INFO] != 0,
                                1:(length(N.stats) ÷ FRONT_STATS_FIELDS))
    return N
end

ref_ldlt!(N::Numeric, S::Symbolic, A::CSR; kwargs...) = ref_ldlt!(N, S, vec(A.nzval); kwargs...)

# X D for X with the columns of the factor and D block diagonal (1×1 and 2×2 blocks)
function _times_d(X::AbstractMatrix{T}, dd, ee, kind, herm::Bool) where {T}
    W = similar(X)
    k = 1
    w = size(X, 2)
    while k <= w
        if kind[k] == PIVOT_KIND_2X2_FIRST
            a, b, c = dd[k], ee[k], dd[k + 1]
            up = herm ? conj(b) : b
            for i in axes(X, 1)
                x1, x2 = X[i, k], X[i, k + 1]
                W[i, k] = x1 * a + x2 * b
                W[i, k + 1] = x1 * up + x2 * c
            end
            k += 2
        else
            for i in axes(X, 1)
                W[i, k] = X[i, k] * dd[k]
            end
            k += 1
        end
    end
    return W
end

# largest |F[i, c]| over the remaining rows k:f of column c, without rows c and `skip`
function _colmax(F::AbstractMatrix{T}, c::Int, k::Int, skip::Int = 0) where {T}
    m = zero(real(T))
    for i in k:size(F, 1)
        (i == c || i == skip) && continue
        m = max(m, abs(F[i, c]))
    end
    return m
end

# acceptable 1×1 pivot at column c: not tiny and passing the threshold test
function _accept_1x1(F, c, k, prm::_LDLTParams)
    d = abs(F[c, c])
    return d >= prm.eps && d >= prm.u * _colmax(F, c, k)
end

# the 2×2 block on columns (k, r) is nonsingular and not tiny
function _nonsingular_2x2(F, k, r, prm::_LDLTParams)
    det = F[k, k] * F[r, r] - F[k, r] * F[r, k]
    return !iszero(det) && isfinite(det) && max(abs(F[k, k]), abs(F[r, k]), abs(F[r, r])) >= prm.eps
end

# acceptable 2×2 pivot on columns (k, r): every entry of its two L columns bounded by 1/u
function _accept_2x2(F, k, r, prm::_LDLTParams)
    _nonsingular_2x2(F, k, r, prm) || return false
    det = F[k, k] * F[r, r] - F[k, r] * F[r, k]
    e11, e12, e21, e22 = abs(F[r, r] / det), abs(F[k, r] / det), abs(F[r, k] / det), abs(F[k, k] / det)
    m1, m2 = _colmax(F, k, k, r), _colmax(F, r, k, k)
    return prm.u * (e11 * m1 + e21 * m2) <= 1 && prm.u * (e12 * m1 + e22 * m2) <= 1
end

# the acceptable 1×1 pivot of the block k:w with the largest |aⱼⱼ| (0 if none)
function _best_1x1(F, k, w, prm::_LDLTParams)
    best = 0
    for j in k:w
        _accept_1x1(F, j, k, prm) || continue
        (best == 0 || abs(F[j, j]) > abs(F[best, best])) && (best = j)
    end
    return best
end

# Pivot choice at column k of the block k:w: `(c, 0)` for a 1×1 pivot at column
# c, `(k, r)` for the 2×2 pivot on columns k and r.
function _choose_pivot(F::AbstractMatrix{T}, k::Int, w::Int, prm::_LDLTParams) where {T}
    prm.pivot == PIVOT_NONE && return (k, 0)
    if prm.pivot == PIVOT_DIAGONAL
        _accept_1x1(F, k, k, prm) && return (k, 0)
        best = _best_1x1(F, k, w, prm)
        best != 0 && return (best, 0)
        big = k
        for j in (k + 1):w
            abs(F[j, j]) > abs(F[big, big]) && (big = j)
        end
        return (abs(F[big, big]) >= prm.eps ? big : k, 0)
    end
    # Bunch–Kaufman inside the block (α in the working precision, as the device kernels: no Float64 on Metal)
    α = real(T)(BUNCH_KAUFMAN_ALPHA)
    λ, r = zero(real(T)), 0
    for i in (k + 1):w
        a = abs(F[i, k])
        a > λ && ((λ, r) = (a, i))
    end
    akk = abs(F[k, k])
    bk = if r == 0 || akk >= α * λ
        (k, 0)
    else
        σ = zero(λ)                              # largest off-diagonal of column r in the block
        for i in k:w
            i != r && (σ = max(σ, abs(F[i, r])))
        end
        if akk * σ >= α * λ^2
            (k, 0)
        elseif abs(F[r, r]) >= α * σ
            (r, 0)
        else
            (k, r)
        end
    end
    c, r2 = bk
    (r2 == 0 ? _accept_1x1(F, c, k, prm) : _accept_2x2(F, k, r2, prm)) && return bk
    best = _best_1x1(F, k, w, prm)
    best != 0 && return (best, 0)
    r2 != 0 && _nonsingular_2x2(F, k, r2, prm) && return bk
    return (c, 0)
end

# symmetric interchange of the columns/rows p and q (≤ w) of the first w columns of F
function _swap_front!(F::AbstractMatrix, p::Int, q::Int, w::Int, lp::Vector{Int})
    p == q && return nothing
    for i in axes(F, 1)
        F[i, p], F[i, q] = F[i, q], F[i, p]
    end
    for j in 1:w
        F[p, j], F[q, j] = F[q, j], F[p, j]
    end
    lp[p], lp[q] = lp[q], lp[p]
    return nothing
end

function _perturbation_sign(d::T, requested::Int, herm::Bool) where {T}
    requested != 0 && return T(requested)
    if herm
        return real(d) < 0 ? -one(T) : one(T)
    end
    return iszero(d) ? one(T) : d / abs(d)
end

# Dense LDLᵀ/LDLᴴ of the fully-summed columns 1:w of the front F (both triangles
# of F[1:w, 1:w] and rows w+1:f of columns 1:w are read and updated; the
# contribution block is left to the caller). Returns the local pivot order `lp`
# (`lp[k]` = assembled column eliminated at k), the pivot kinds, D's diagonal
# `dd` and 2×2 subdiagonal `ee`, and the statistics (npos, nneg, nzero,
# nperturbed, n2x2). On return F[i, j] (i > j, j ≤ w) holds L.
function _ldlt_front!(F::Matrix{T}, w::Int, prm::_LDLTParams, signof) where {T}
    f = size(F, 1)
    lp = collect(1:w)
    kind = zeros(Int8, w)
    dd = zeros(T, w)
    ee = zeros(T, w)
    st = zeros(Int, 5)
    k = 1
    while k <= w
        c, r = _choose_pivot(F, k, w, prm)
        if r == 0
            _swap_front!(F, k, c, w, lp)
            d = prm.herm ? T(real(F[k, k])) : F[k, k]
            kind[k] = PIVOT_KIND_1X1
            if !(abs(d) >= prm.eps)                   # tiny (or NaN): perturb
                iszero(d) && (st[STAT_NZERO] += 1)
                d = _perturbation_sign(d, signof(lp[k]), prm.herm) * prm.eps
                kind[k] = PIVOT_KIND_PERTURBED
                st[STAT_NPERTURBED] += 1
            end
            F[k, k] = d
            dd[k] = d
            if prm.herm
                real(d) > 0 ? (st[STAT_NPOS] += 1) : real(d) < 0 && (st[STAT_NNEG] += 1)
            end
            for i in (k + 1):f
                l = F[i, k] / d
                for j in (k + 1):w
                    F[i, j] -= l * F[k, j]
                end
                F[i, k] = l
            end
            k += 1
        else
            _swap_front!(F, k + 1, r, w, lp)
            a, b, c2 = F[k, k], F[k + 1, k], F[k + 1, k + 1]
            prm.herm && ((a, c2) = (T(real(a)), T(real(c2))))
            up = prm.herm ? conj(b) : b
            F[k, k], F[k + 1, k + 1], F[k, k + 1] = a, c2, up
            det = a * c2 - up * b
            e11, e12, e21, e22 = c2 / det, -up / det, -b / det, a / det
            kind[k], kind[k + 1] = PIVOT_KIND_2X2_FIRST, PIVOT_KIND_2X2_SECOND
            dd[k], dd[k + 1], ee[k] = a, c2, b
            st[STAT_N2X2] += 1
            if prm.herm
                if real(det) < 0
                    st[STAT_NPOS] += 1
                    st[STAT_NNEG] += 1
                elseif real(a) > 0
                    st[STAT_NPOS] += 2
                else
                    st[STAT_NNEG] += 2
                end
            end
            for i in (k + 2):f
                x1, x2 = F[i, k], F[i, k + 1]
                l1 = x1 * e11 + x2 * e21
                l2 = x1 * e12 + x2 * e22
                for j in (k + 2):w
                    F[i, j] -= l1 * F[k, j] + l2 * F[k + 1, j]
                end
                F[i, k], F[i, k + 1] = l1, l2
            end
            F[k + 1, k] = zero(T)                     # the 2×2 block lives in D, L's block is the identity
            k += 2
        end
    end
    return lp, kind, dd, ee, st
end

"""
    ref_solve_ldlt!(X, symbolic, numeric, B) -> X

Solve `A X = B` with the reference factor of [`ref_ldlt!`](@ref): `Y = B[perm, :]`;
forward sweep over the supernodes in `snpost` order (reorder the supernode's
slice of `Y` by its local pivot order `piv`, unit-lower `trsm`, update of the
rows below); diagonal step `Y ← D⁻¹ Y` (1×1 and 2×2 blocks); backward sweep in
reverse order (update from the rows below, `trsm` with `L₁₁ᴴ` (`ᵀ` for complex
symmetric), undo the local pivot order); `X[perm, :] = Y`. Called by
[`ref_solve!`](@ref) for structures `"S"`/`"H"`. `X === B` is allowed.
"""
function ref_solve_ldlt!(X::AbstractVecOrMat, S::Symbolic, N::Numeric{T, Vector{T}}, B::AbstractVecOrMat) where {T}
    n = S.n
    size(B, 1) == n && size(X, 1) == n && size(X, 2) == size(B, 2) ||
        throw(DimensionMismatch("A is $n×$n, X is $(size(X)), B is $(size(B))"))
    herm = !(T <: Complex) || S.structure == STRUCTURE_HERMITIAN
    sp, L = S.partition, S.layout
    perm = sp.perm
    piv = N.piv
    nrhs = size(B, 2)
    Y = Matrix{T}(undef, n, nrhs)
    for r in 1:nrhs, k in 1:n
        Y[k, r] = B[perm[k], r]
    end
    tmp = Matrix{T}(undef, maximum(s -> sp.rowptr[s + 1] - sp.rowptr[s] - snwidth(sp, s), 1:nsupernodes(sp);
                                   init = 0), nrhs)
    wmax = maximum(s -> snwidth(sp, s), 1:nsupernodes(sp); init = 0)
    blk = Matrix{T}(undef, wmax, nrhs)
    for s in sp.snpost                                   # forward: L Z = Y
        f, w, cols, below, L11, L21 = _ref_panel(sp, L, N, s)
        Yc = view(Y, cols, :)
        for r in 1:nrhs, (k, g) in enumerate(cols)
            blk[k, r] = Y[piv[g], r]
        end
        Yc .= view(blk, 1:w, :)
        BLAS.trsm!('L', 'L', 'N', 'U', one(T), L11, Yc)
        if f > w
            t = view(tmp, 1:(f - w), :)
            mul!(t, L21, Yc)
            for r in 1:nrhs, (k, i) in enumerate(below)
                Y[i, r] -= t[k, r]
            end
        end
    end
    k = 1                                                # diagonal: D W = Z
    while k <= n
        if N.pivot_kind[k] == PIVOT_KIND_2X2_FIRST
            a, b, c = N.d[k], N.d[n + k], N.d[k + 1]
            up = herm ? conj(b) : b
            det = a * c - up * b
            for r in 1:nrhs
                y1, y2 = Y[k, r], Y[k + 1, r]
                Y[k, r] = (c * y1 - up * y2) / det
                Y[k + 1, r] = (a * y2 - b * y1) / det
            end
            k += 2
        else
            for r in 1:nrhs
                Y[k, r] /= N.d[k]
            end
            k += 1
        end
    end
    for s in Iterators.reverse(sp.snpost)                 # backward: Lᴴ V = W
        f, w, cols, below, L11, L21 = _ref_panel(sp, L, N, s)
        Yc = view(Y, cols, :)
        if f > w
            t = view(tmp, 1:(f - w), :)
            for r in 1:nrhs, (k, i) in enumerate(below)
                t[k, r] = Y[i, r]
            end
            mul!(Yc, herm ? L21' : transpose(L21), t, -one(T), one(T))
        end
        BLAS.trsm!('L', 'L', herm ? 'C' : 'T', 'U', one(T), L11, Yc)
        copyto!(view(blk, 1:w, :), Yc)
        for r in 1:nrhs, (k, g) in enumerate(cols)
            Y[piv[g], r] = blk[k, r]
        end
    end
    for r in 1:nrhs, k in 1:n
        X[perm[k], r] = Y[k, r]
    end
    return X
end

"""
    extract_ldlt(symbolic, numeric) -> (L, D, p)

The factors of an LDLᵀ/LDLᴴ factorization ([`ref_ldlt!`](@ref) or a device
`numeric`, copied to the host once) as sparse matrices with
`A[p, p] ≈ L D Lᴴ` (`Lᵀ` for complex symmetric `"S"`): `p` is the analysis
permutation composed with the local pivot orders, `L` unit lower triangular
with every stored panel entry, `D` block diagonal with 1×1 and 2×2 blocks.
"""
function extract_ldlt(S::Symbolic, N::Numeric{T}) where {T}
    sp, L = S.partition, S.layout
    factor = _host_vector(N.factor)
    d = _host_vector(N.d)
    piv = Int.(_host_vector(N.piv))
    kind = _host_vector(N.pivot_kind)
    herm = !(T <: Complex) || S.structure == STRUCTURE_HERMITIAN
    n = sp.n
    isperm(piv) || throw(InvalidValueError("numeric.piv is not a permutation (not an LDLᵀ/LDLᴴ factorization?)"))
    pos = invperm(piv)                  # column of P A Pᵀ → factor position
    I, J, V = Int[], Int[], T[]
    for s in 1:nsupernodes(sp)
        rows = snrows(sp, s)
        f = length(rows)
        w = snwidth(sp, s)
        for (j, col) in enumerate(sncols(sp, s)), i in j:f
            push!(I, i <= w ? sp.super_ptr[s] + i - 1 : pos[rows[i]])
            push!(J, col)
            push!(V, factor[L.panel_ptr[s] + (j - 1) * f + i - 1])
        end
    end
    Lf = sparse(I, J, V, n, n)
    DI, DJ, DV = collect(1:n), collect(1:n), d[1:n]
    for k in 1:n
        kind[k] == PIVOT_KIND_2X2_FIRST || continue
        append!(DI, (k + 1, k))
        append!(DJ, (k, k + 1))
        append!(DV, (d[n + k], herm ? conj(d[n + k]) : d[n + k]))
    end
    return Lf, sparse(DI, DJ, DV, n, n), sp.perm[piv]
end

"""
    pivot_stats(numeric) -> (npos, nneg, nzero, nperturbed, n2x2)

Sum of the per-front statistics of an LDLᵀ/LDLᴴ factorization (PLAN §1.7
`pivot_stats`): positive and negative eigenvalues of D (after perturbation; a
2×2 block with negative determinant counts one of each), pivots that were
exactly zero, perturbed pivots, 2×2 blocks. A device `numeric.stats` is copied
to the host. `npos`/`nneg` are `0` for complex symmetric (`"S"`) matrices.
"""
function pivot_stats(N::Numeric)
    st = _host_vector(N.stats)
    ns = length(st) ÷ FRONT_STATS_FIELDS
    tot(q) = sum((st[(s - 1) * FRONT_STATS_FIELDS + q] for s in 1:ns); init = Int64(0))
    return (npos = tot(STAT_NPOS), nneg = tot(STAT_NNEG), nzero = tot(STAT_NZERO),
            nperturbed = tot(STAT_NPERTURBED), n2x2 = tot(STAT_N2X2))
end

"""
    inertia(numeric) -> (npos, nneg)

Inertia of D of an LDLᵀ/LDLᴴ factorization (PLAN §1.4 `inertia`). With
perturbed pivots this is the inertia of `A + E`; read it together with
[`npivots`](@ref).
"""
function inertia(N::Numeric)
    st = pivot_stats(N)
    return (st.npos, st.nneg)
end

"""
    npivots(numeric) -> Int64

Number of perturbed pivots of an LDLᵀ/LDLᴴ factorization (PLAN §1.4 `npivots`).
"""
npivots(N::Numeric) = pivot_stats(N).nperturbed
