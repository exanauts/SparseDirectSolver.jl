# CPU reference multifrontal Cholesky (PLAN §7, the oracle): plain Julia on
# host arrays with BLAS/LAPACK, processing the supernodes of a `Symbolic` in
# postorder with the same maps the device path uses (`amap_ptr`/`amap_src` for
# the assembly of A, `child_list`/`relind` for the extend-add, `Layout` for the
# panels). It is deliberately independent of the dense interface and of the KA
# kernels it is meant to check. Contribution blocks live in host matrices, not
# on the update stack of the layout (regime-A blocks have no stack slot).

"""
    Numeric{T, VT <: AbstractVector{T}, VS <: AbstractVector{Int64}}

Numeric storage of a factorization (PLAN §3.2), laid out by the
[`Layout`](@ref) of a [`Symbolic`](@ref):

* `factor` (`layout.factor_len` entries): the panel of supernode `s` is the
  column-major `f×w` block `factor[panel_ptr[s]:(panel_ptr[s+1]-1)]` (leading
  dimension `f`, rows `snrows(s)`, the upper triangle of its diagonal block is
  unused and kept zero);
* `d` (`layout.d_len = 2n` entries): D of LDLᵀ/LDLᴴ (unused by Cholesky);
* `stack` (`layout.stack_len` entries): the update stack of the device path;
* `stats` (`FRONT_STATS_FIELDS × ns` `Int64`, column `s` = front `s`):
  `(npos, nneg, nzero, nperturbed, n2x2, info)`, `info` = the local column of the
  first failed pivot of the front (`0` = none).
"""
struct Numeric{T, VT <: AbstractVector{T}, VS <: AbstractVector{Int64}}
    factor::VT
    d::VT
    stack::VT
    stats::VS
end

Base.eltype(::Numeric{T}) where {T} = T

Base.show(io::IO, N::Numeric{T, VT}) where {T, VT} =
    print(io, "Numeric{", T, ", ", nameof(VT), "}(factor ", length(N.factor), ", D ", length(N.d), ", stack ",
          length(N.stack), " entries)")

"""
    allocate_numeric(symbolic, T, backend = CPU()) -> Numeric{T}

Allocate (zero-filled) the factor panels, D, update stack and per-front
statistics of `symbolic`'s [`Layout`](@ref) for element type `T` on the
KernelAbstractions `backend`. This is the only allocation of the numeric phase.
"""
function allocate_numeric(S::Symbolic, ::Type{T}, backend::KernelAbstractions.Backend = KernelAbstractions.CPU()) where {T}
    L = S.layout
    factor = KernelAbstractions.zeros(backend, T, L.factor_len)
    d = KernelAbstractions.zeros(backend, T, L.d_len)
    stack = KernelAbstractions.zeros(backend, T, L.stack_len)
    stats = KernelAbstractions.zeros(backend, Int64, FRONT_STATS_FIELDS * nsupernodes(S))
    return Numeric{T, typeof(factor), typeof(stats)}(factor, d, stack, stats)
end

_host_vector(x::Vector) = x
_host_vector(x::AbstractVector) = Array(x)

function _check_reference_cholesky(S::Symbolic, ::Type{T}) where {T}
    S.structure == STRUCTURE_SPD || S.structure == STRUCTURE_HPD ||
        throw(InvalidValueError("Cholesky needs structure \"SPD\" or \"HPD\", got \"$(convert(String, S.structure))\""))
    T <: Complex && S.structure != STRUCTURE_HPD &&
        throw(InvalidValueError("complex Cholesky needs structure \"HPD\" (LLᴴ of a Hermitian matrix)"))
    T <: LinearAlgebra.BlasFloat ||
        throw(NotSupportedError("the reference Cholesky supports BLAS element types only, got $T"))
    return nothing
end

"""
    ref_factorize!(numeric::Numeric{T, Vector{T}}, symbolic, nzval) -> info

Reference multifrontal Cholesky `P A Pᵀ = L Lᴴ` of the matrix whose stored
values are `nzval` (same CSR pattern, view and index base as the analysis; a
host vector or anything `Array` converts) into the host `numeric`. The
supernodes are processed in `snpost` order: zero the `f×f` front, add A through
`amap_ptr`/`amap_src`/`amap` (conjugating negative offsets), extend-add the
children's contribution blocks through `relind` in `child_list` order, then
`potrf` on the `w×w` block, `trsm` on the `(f-w)×w` block and `syrk`/`herk` on
the contribution block. The front's first `w` columns are stored in its panel.

Returns `info = 0` on success, else the first non-positive pivot (the smallest
failed column of the factor, as the column of the *original* matrix); the
factorization stops there and later panels stay zero. Requires structure
`"SPD"` (real `T`) or `"HPD"`.
"""
function ref_factorize!(N::Numeric{T, Vector{T}}, S::Symbolic, nzval::AbstractVector) where {T}
    _check_reference_cholesky(S, T)
    length(nzval) == S.nnz ||
        throw(InvalidValueError("nzval has $(length(nzval)) entries, the analysis expects $(S.nnz)"))
    nz = _host_vector(nzval)
    amap = _host_vector(S.amap)
    amap_ptr = _host_vector(S.amap_ptr)
    amap_src = _host_vector(S.amap_src)
    child_ptr = _host_vector(S.child_ptr)
    child_list = _host_vector(S.child_list)
    relind_ptr = _host_vector(S.relind_ptr)
    relind = _host_vector(S.relind)
    sp, L = S.partition, S.layout
    ns = nsupernodes(sp)
    fill!(N.factor, zero(T))
    fill!(N.stats, 0)
    cbs = Vector{Matrix{T}}(undef, ns)
    for s in sp.snpost
        f = sp.rowptr[s + 1] - sp.rowptr[s]
        w = snwidth(sp, s)
        m = f - w
        base = L.panel_ptr[s]
        F = zeros(T, f, f)
        # assembly of A (entries already in the lower trapezoid of the panel)
        for k in amap_ptr[s]:(amap_ptr[s + 1] - 1)
            p = amap_src[k]
            off = Int(amap[p])
            loc = abs(off) - base
            F[loc % f + 1, loc ÷ f + 1] += off < 0 ? conj(T(nz[p])) : T(nz[p])
        end
        # extend-add of the children's contribution blocks (lower triangles)
        for k in child_ptr[s]:(child_ptr[s + 1] - 1)
            c = child_list[k]
            isassigned(cbs, c) || continue                # child without a contribution block
            C = cbs[c]
            ri = view(relind, relind_ptr[c]:(relind_ptr[c + 1] - 1))
            for jj in axes(C, 2), ii in jj:size(C, 1)
                F[ri[ii], ri[jj]] += C[ii, jj]
            end
            cbs[c] = Matrix{T}(undef, 0, 0)               # consumed
        end
        # dense kernels of the front
        F11 = view(F, 1:w, 1:w)
        _, linfo = LAPACK.potrf!('L', F11)
        if linfo > 0
            N.stats[(s - 1) * FRONT_STATS_FIELDS + 1] = linfo - 1
            N.stats[s * FRONT_STATS_FIELDS] = linfo
            return sp.perm[sp.super_ptr[s] + linfo - 1]
        end
        if m > 0
            F21 = view(F, (w + 1):f, 1:w)
            BLAS.trsm!('R', 'L', 'C', 'N', one(T), F11, F21)
            F22 = view(F, (w + 1):f, (w + 1):f)
            if T <: Complex
                BLAS.herk!('L', 'N', -one(real(T)), F21, one(real(T)), F22)
            else
                BLAS.syrk!('L', 'N', -one(T), F21, one(T), F22)
            end
            sp.snparent[s] != 0 && (cbs[s] = F[(w + 1):f, (w + 1):f])
        end
        # store the panel (lower trapezoid; the strict upper triangle of F11 is zero)
        for j in 1:w, i in j:f
            N.factor[base + (j - 1) * f + i - 1] = F[i, j]
        end
        N.stats[(s - 1) * FRONT_STATS_FIELDS + 1] = w
    end
    return 0
end

ref_factorize!(N::Numeric, S::Symbolic, A::CSR) = ref_factorize!(N, S, vec(A.nzval))

"""
    ref_solve!(X, symbolic, numeric, B) -> X

Solve `A X = B` with the reference factor `P A Pᵀ = L Lᴴ` of
[`ref_factorize!`](@ref): `Y = B[perm, :]`, forward sweep `L Z = Y` over the
supernodes in `snpost` order (`trsm` on the diagonal block, `gemm` update of
the rows below), backward sweep `Lᴴ W = Z` in reverse order, `X[perm, :] = W`.
`B` and `X` are host vectors or `n × nrhs` matrices (`X === B` is allowed).
"""
function ref_solve!(X::AbstractVecOrMat, S::Symbolic, N::Numeric{T, Vector{T}}, B::AbstractVecOrMat) where {T}
    n = S.n
    size(B, 1) == n && size(X, 1) == n && size(X, 2) == size(B, 2) ||
        throw(DimensionMismatch("A is $n×$n, X is $(size(X)), B is $(size(B))"))
    sp, L = S.partition, S.layout
    perm = sp.perm
    nrhs = size(B, 2)
    Y = Matrix{T}(undef, n, nrhs)
    for r in 1:nrhs, k in 1:n
        Y[k, r] = B[perm[k], r]
    end
    tmp = Matrix{T}(undef, maximum(s -> sp.rowptr[s + 1] - sp.rowptr[s] - snwidth(sp, s), 1:nsupernodes(sp);
                                   init = 0), nrhs)
    for s in sp.snpost                                  # forward: L Z = Y
        f, w, cols, below, L11, L21 = _ref_panel(sp, L, N, s)
        Yc = view(Y, cols, :)
        BLAS.trsm!('L', 'L', 'N', 'N', one(T), L11, Yc)
        if f > w
            t = view(tmp, 1:(f - w), :)
            mul!(t, L21, Yc)
            for r in 1:nrhs, (k, i) in enumerate(below)
                Y[i, r] -= t[k, r]
            end
        end
    end
    for s in Iterators.reverse(sp.snpost)                # backward: Lᴴ W = Z
        f, w, cols, below, L11, L21 = _ref_panel(sp, L, N, s)
        Yc = view(Y, cols, :)
        if f > w
            t = view(tmp, 1:(f - w), :)
            for r in 1:nrhs, (k, i) in enumerate(below)
                t[k, r] = Y[i, r]
            end
            mul!(Yc, L21', t, -one(T), one(T))
        end
        BLAS.trsm!('L', 'L', 'C', 'N', one(T), L11, Yc)
    end
    for r in 1:nrhs, k in 1:n
        X[perm[k], r] = Y[k, r]
    end
    return X
end

function _ref_panel(sp::SupernodePartition, L::Layout, N::Numeric, s::Integer)
    f = sp.rowptr[s + 1] - sp.rowptr[s]
    w = snwidth(sp, s)
    P = reshape(view(N.factor, L.panel_ptr[s]:(L.panel_ptr[s + 1] - 1)), f, w)
    below = view(snrows(sp, s), (w + 1):f)
    return f, w, sncols(sp, s), below, view(P, 1:w, 1:w), view(P, (w + 1):f, 1:w)
end

"""
    extract_L(symbolic, numeric) -> SparseMatrixCSC

The Cholesky factor `L` of `P A Pᵀ = L Lᴴ` (`P` = `symbolic.partition.perm`,
supernodal numbering) as a sparse lower-triangular matrix with every stored
panel entry, explicit zeros of amalgamation included (`nnz == nnz_stored`).
A device `numeric.factor` is copied to the host once.
"""
function extract_L(S::Symbolic, N::Numeric{T}) where {T}
    sp, L = S.partition, S.layout
    factor = _host_vector(N.factor)
    n = sp.n
    colptr = Vector{Int}(undef, n + 1)
    colptr[1] = 1
    rowval = Vector{Int}(undef, sp.nnz_stored)
    nzval = Vector{T}(undef, sp.nnz_stored)
    q = 1
    for s in 1:nsupernodes(sp)
        rows = snrows(sp, s)
        f = length(rows)
        for (j, col) in enumerate(sncols(sp, s))
            for i in j:f
                rowval[q] = rows[i]
                nzval[q] = factor[L.panel_ptr[s] + (j - 1) * f + i - 1]
                q += 1
            end
            colptr[col + 1] = q
        end
    end
    q == sp.nnz_stored + 1 || throw(InvalidValueError("panels hold $(q - 1) entries, expected $(sp.nnz_stored)"))
    return SparseMatrixCSC(n, n, colptr, rowval, nzval)
end
