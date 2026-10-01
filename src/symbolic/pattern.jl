# Symbolic step 1 (PLAN §2.3): the symmetric adjacency pattern the ordering and
# the elimination tree work on, and the full-pattern map used by the residual
# SpMV of iterative refinement (T16).
#
# Everything here runs on the host with plain `Int` arrays. Only `rowptr` and
# `colval` are copied from the device (analysis is a phase boundary); values
# are never read.
#
# Conventions. The `view` and `index` arguments describe the *stored* CSR arrays
# (as in cuDSS, which never sees the `transposed` flag of a `CSR`): MadNLP passes
# the CSC arrays of the lower triangle as a CSR with view `'U'`. The symmetric
# pattern of the stored matrix and of its transpose are the same, so
# `SymmetricPattern` ignores `transposed`; `full_pattern_map` describes the stored
# matrix and the solver applies `transposed` through `solve_mode` (PLAN §1.1).

"""
    SymmetricPattern

Adjacency structure of an `n × n` symmetric sparsity pattern on the host:
`colptr` (length `n + 1`) and `rowval` hold, for every column `j`, the sorted
row indices `i ≠ j` with `(i, j)` in the pattern, 1-based, without duplicates
and without the diagonal, both triangles stored (`(i, j)` present iff `(j, i)`
present).

    SymmetricPattern(n, colptr, rowval)
    SymmetricPattern(A::CSR, structure; view = 'F')
    SymmetricPattern(rowptr, colval, n, structure; view = 'F', index = 'O')

The first form wraps arrays that already satisfy the invariants (checked). The
others build the pattern from a CSR pattern, following PLAN §2.3 step 1:

* `structure` `"S"`, `"H"`, `"SPD"`, `"HPD"` (or a [`Structure`](@ref)): only the
  triangle selected by `view` is read (`'L'`: `col ≤ row`, `'U'`: `col ≥ row`;
  `'F'` reads the lower triangle, as cuDSS does) and mirrored;
* `"G"`: the pattern of `A + Aᵀ`; `view` must be `'F'`;
* duplicates are dropped, `index` (`'O'`/`'Z'`) is the base of `rowptr`/`colval`,
  non-square matrices and out-of-range column indices raise [`InvalidValueError`](@ref).
"""
struct SymmetricPattern
    n::Int
    colptr::Vector{Int}
    rowval::Vector{Int}

    function SymmetricPattern(n::Integer, colptr::Vector{Int}, rowval::Vector{Int})
        n >= 0 || throw(InvalidValueError("pattern size must be nonnegative, got $n"))
        length(colptr) == n + 1 ||
            throw(InvalidValueError("length(colptr) = $(length(colptr)) does not match n + 1 = $(n + 1)"))
        (colptr[1] == 1 && colptr[end] == length(rowval) + 1) ||
            throw(InvalidValueError("colptr must start at 1 and end at length(rowval) + 1"))
        for j in 1:n
            colptr[j] <= colptr[j + 1] || throw(InvalidValueError("colptr is not nondecreasing at column $j"))
            prev = 0
            for p in colptr[j]:(colptr[j + 1] - 1)
                i = rowval[p]
                (prev < i <= n && i != j) ||
                    throw(InvalidValueError("column $j of the pattern is not sorted, unique, in range and off-diagonal"))
                prev = i
            end
        end
        return new(Int(n), colptr, rowval)
    end
end

Base.size(P::SymmetricPattern) = (P.n, P.n)
Base.size(P::SymmetricPattern, d::Integer) = d <= 2 ? P.n : 1

"""
    nnz(P::SymmetricPattern)

Number of stored off-diagonal entries (both triangles).
"""
SparseArrays.nnz(P::SymmetricPattern) = length(P.rowval)

"""
    neighbors(P::SymmetricPattern, j)

Sorted indices `i ≠ j` adjacent to `j` (a view into `P.rowval`).
"""
neighbors(P::SymmetricPattern, j::Integer) = view(P.rowval, P.colptr[j]:(P.colptr[j + 1] - 1))

Base.:(==)(P::SymmetricPattern, Q::SymmetricPattern) =
    P.n == Q.n && P.colptr == Q.colptr && P.rowval == Q.rowval

Base.hash(P::SymmetricPattern, h::UInt) = hash(P.rowval, hash(P.colptr, hash(P.n, hash(:SymmetricPattern, h))))

Base.show(io::IO, P::SymmetricPattern) = print(io, "SymmetricPattern(n = $(P.n), nnz = $(nnz(P)))")

"""
    SparseMatrixCSC(P::SymmetricPattern) -> SparseMatrixCSC{Bool,Int}

The pattern as a Boolean sparse matrix (no diagonal), for tests and debugging.
"""
SparseArrays.SparseMatrixCSC(P::SymmetricPattern) =
    SparseMatrixCSC(P.n, P.n, copy(P.colptr), copy(P.rowval), fill(true, length(P.rowval)))

_structure(s::Structure) = s
_structure(s::AbstractString) = convert(Structure, s)
_structure(s) = throw(InvalidValueError("structure must be \"G\", \"S\", \"H\", \"SPD\", \"HPD\" or a Structure, got $(repr(s))"))

_matrix_view(v::MatrixView) = v
_matrix_view(v::AbstractChar) = convert(MatrixView, v)
_matrix_view(v) = throw(InvalidValueError("view must be 'L', 'U', 'F' or a MatrixView, got $(repr(v))"))

_is_hermitian(s::Structure) = s == STRUCTURE_HERMITIAN || s == STRUCTURE_HPD

# Which stored entries (r, c) are read, and whether the mirrored entry is generated.
# For symmetric structures 'F' reads the lower triangle, as in cuDSS (PLAN §1.1).
function _entry_filter(structure::Structure, view::MatrixView)
    if structure == STRUCTURE_GENERAL
        view == VIEW_FULL ||
            throw(InvalidValueError("structure \"G\" needs view 'F', got '$(convert(Char, view))'"))
        return (r, c) -> true
    end
    view == VIEW_UPPER && return (r, c) -> c >= r
    return (r, c) -> c <= r
end

# Host copies of a CSR pattern with 1-based indices; validates sizes and ranges.
function _host_pattern(rowptr::AbstractVector{<:Integer}, colval::AbstractVector{<:Integer}, n::Integer, index)
    base = _index_base(index) == INDEX_ZERO ? 1 : 0
    length(rowptr) == n + 1 ||
        throw(InvalidValueError("length(rowptr) = $(length(rowptr)) does not match n + 1 = $(n + 1)"))
    rp = Vector{Int}(Array(rowptr)) .+ base
    cv = Vector{Int}(Array(colval)) .+ base
    (rp[1] == 1 && rp[end] == length(cv) + 1) ||
        throw(InvalidValueError("rowptr must start at the index base and end at nnz + base"))
    for r in 1:n
        rp[r] <= rp[r + 1] || throw(InvalidValueError("rowptr is not nondecreasing at row $r"))
    end
    for c in cv
        1 <= c <= n || throw(InvalidValueError("column index $(c - base) out of range for n = $n"))
    end
    return rp, cv
end

# Triplets (row, col, nzval index) of the stored entries selected by the view.
function _selected_entries(rp, cv, n, structure, view)
    keep = _entry_filter(structure, view)
    rows = Int[]
    cols = Int[]
    srcs = Int[]
    for r in 1:n, p in rp[r]:(rp[r + 1] - 1)
        c = cv[p]
        keep(r, c) || continue
        push!(rows, r)
        push!(cols, c)
        push!(srcs, p)
    end
    return rows, cols, srcs
end

function SymmetricPattern(rowptr::AbstractVector{<:Integer}, colval::AbstractVector{<:Integer}, n::Integer,
                          structure; view = VIEW_FULL, index = INDEX_ONE)
    s = _structure(structure)
    v = _matrix_view(view)
    rp, cv = _host_pattern(rowptr, colval, n, index)
    rows, cols, _ = _selected_entries(rp, cv, n, s, v)
    # off-diagonal entries in both orientations, then column-sort and deduplicate
    I = Int[]
    J = Int[]
    sizehint!(I, 2 * length(rows))
    sizehint!(J, 2 * length(rows))
    for (r, c) in zip(rows, cols)
        r == c && continue
        push!(I, r); push!(J, c)
        push!(I, c); push!(J, r)
    end
    colptr, rowval = _csc_unique(n, I, J)
    return SymmetricPattern(n, colptr, rowval)
end

function SymmetricPattern(A::CSR, structure; view = VIEW_FULL)
    A.nrows == A.ncols ||
        throw(InvalidValueError("the matrix must be square, got $(A.nrows) × $(A.ncols)"))
    return SymmetricPattern(A.rowptr, A.colval, A.nrows, structure; view, index = A.index)
end

# Column-sorted, duplicate-free CSC arrays of the entries (I[k], J[k]) (counting sort).
function _csc_unique(n::Int, I::Vector{Int}, J::Vector{Int})
    count = zeros(Int, n + 1)
    for j in J
        count[j + 1] += 1
    end
    colptr = cumsum!(count, count) .+ 1
    rowval = Vector{Int}(undef, length(I))
    next = colptr[1:n]
    for k in eachindex(I)
        j = J[k]
        rowval[next[j]] = I[k]
        next[j] += 1
    end
    # sort each column and squeeze out duplicates in place
    out = 1
    newptr = Vector{Int}(undef, n + 1)
    newptr[1] = 1
    for j in 1:n
        seg = view(rowval, colptr[j]:(colptr[j + 1] - 1))
        sort!(seg)
        prev = 0
        for i in seg
            i == prev && continue
            rowval[out] = i
            out += 1
            prev = i
        end
        newptr[j + 1] = out
    end
    resize!(rowval, out - 1)
    return newptr, rowval
end

"""
    FullPatternMap

Host description of the full matrix `M` behind a user CSR input (PLAN §2.3
step 1, used by the residual SpMV of refinement): `M` as a 1-based CSR
(`rowptr`, `colval`, sorted columns, duplicates merged) plus, for every entry
`e` of `M`, the user `nzval` entries that make it up:

    M.nzval[e] = Σ_{s ∈ srcptr[e]:srcptr[e+1]-1}  conjflag[s] ? conj(nzval[src[s]]) : nzval[src[s]]

`src` holds 1-based positions in the user's `nzval` (first batch member;
member `k` adds `(k - 1) * nnz`). For symmetric structures the triangle given by
the view is mirrored; for `"H"`/`"HPD"` the mirrored entries are conjugated.
The map describes the *stored* CSR matrix; the `transposed` flag of a [`CSR`](@ref)
is applied by the solver (through `solve_mode`).
"""
struct FullPatternMap
    n::Int
    rowptr::Vector{Int}
    colval::Vector{Int}
    srcptr::Vector{Int}
    src::Vector{Int}
    conjflag::Vector{Bool}
end

SparseArrays.nnz(F::FullPatternMap) = length(F.colval)

"""
    full_pattern_map(A::CSR, structure; view = 'F') -> FullPatternMap
    full_pattern_map(rowptr, colval, n, structure; view = 'F', index = 'O') -> FullPatternMap

Build the [`FullPatternMap`](@ref) of a CSR input with the same structure and
view rules as [`SymmetricPattern`](@ref) (for `"G"` the matrix is taken as is).
Duplicated user entries are summed.
"""
function full_pattern_map(rowptr::AbstractVector{<:Integer}, colval::AbstractVector{<:Integer}, n::Integer,
                          structure; view = VIEW_FULL, index = INDEX_ONE)
    s = _structure(structure)
    v = _matrix_view(view)
    rp, cv = _host_pattern(rowptr, colval, n, index)
    rows, cols, srcs = _selected_entries(rp, cv, n, s, v)
    herm = _is_hermitian(s)
    # (row, col, src, conj) of every contribution to M
    R = Int[]; C = Int[]; S = Int[]; F = Bool[]
    for k in eachindex(rows)
        r, c, p = rows[k], cols[k], srcs[k]
        push!(R, r); push!(C, c); push!(S, p); push!(F, false)
        if s != STRUCTURE_GENERAL && r != c
            push!(R, c); push!(C, r); push!(S, p); push!(F, herm)
        end
    end
    # sort contributions by (row, col, src): row-major CSR of M with merged duplicates
    order = sortperm(collect(zip(R, C, S)))
    nrow = zeros(Int, n + 1)
    mrowptr = Vector{Int}(undef, n + 1)
    mcolval = Int[]
    srcptr = Int[1]
    src = S[order]
    conjflag = F[order]
    prev = (0, 0)
    for (t, k) in enumerate(order)
        key = (R[k], C[k])
        if key != prev
            push!(mcolval, C[k])
            nrow[R[k] + 1] += 1
            t > 1 && push!(srcptr, t)
            prev = key
        end
    end
    push!(srcptr, length(order) + 1)
    isempty(order) && (srcptr = [1])
    mrowptr[1] = 1
    for r in 1:n
        mrowptr[r + 1] = mrowptr[r] + nrow[r + 1]
    end
    return FullPatternMap(Int(n), mrowptr, mcolval, srcptr, src, conjflag)
end

function full_pattern_map(A::CSR, structure; view = VIEW_FULL)
    A.nrows == A.ncols ||
        throw(InvalidValueError("the matrix must be square, got $(A.nrows) × $(A.ncols)"))
    return full_pattern_map(A.rowptr, A.colval, A.nrows, structure; view, index = A.index)
end

"""
    full_values(F::FullPatternMap, nzval::AbstractVector) -> Vector

Host evaluation of the map: the values of the full matrix `M` in the order of
`F.colval` (see [`FullPatternMap`](@ref)).
"""
function full_values(F::FullPatternMap, nzval::AbstractVector{T}) where {T}
    x = Array(nzval)
    out = zeros(T, nnz(F))
    for e in 1:nnz(F)
        acc = zero(T)
        for s in F.srcptr[e]:(F.srcptr[e + 1] - 1)
            val = x[F.src[s]]
            acc += F.conjflag[s] ? conj(val) : val
        end
        out[e] = acc
    end
    return out
end

"""
    SparseMatrixCSC(F::FullPatternMap, nzval) -> SparseMatrixCSC

The full matrix `M` described by `F` with the values taken from the user's
`nzval` (host copy; for tests and debugging).
"""
function SparseArrays.SparseMatrixCSC(F::FullPatternMap, nzval::AbstractVector)
    vals = full_values(F, nzval)
    rows = Int[]
    for r in 1:F.n, _ in F.rowptr[r]:(F.rowptr[r + 1] - 1)
        push!(rows, r)
    end
    return sparse(rows, F.colval, vals, F.n, F.n)
end
