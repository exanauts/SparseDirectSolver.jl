# Input containers (PLAN §1.1, §3.1): the in-package CSR matrix shared by every
# backend, and the dense descriptor behind `MatrixDescriptor(T, n; nbatch)` +
# `update!`. Backend sparse types (`CuSparseMatrixCSR`, …) are wrapped without
# copies by the extensions in `ext/`.

"""
    CSR{T,INT,VI,VT}

Compressed sparse row matrix on any backend: `rowptr::VI`, `colval::VI`
(`VI <: AbstractVector{INT}`), `nzval::VT` (`VT <: AbstractVecOrMat{T}`),
`nrows`, `ncols`, `index::IndexBase` and `transposed::Bool`.

* `index` is the base of `rowptr`/`colval` (`INDEX_ONE` or `INDEX_ZERO`); the
  arrays are never rebased in place.
* `transposed = true` means the stored CSR arrays describe `transpose(M)` of the
  matrix `M` the caller means (for example a CSC matrix reinterpreted as the CSR
  of its transpose, see [`csr_of_transpose`](@ref)). `size`, `nnz` and
  `SparseMatrixCSC(A)` refer to the *stored* CSR matrix; consumers honor the flag.
* A uniform batch shares `rowptr`/`colval`; its values are either a vector of
  length `nbatch * nnz` (members one after the other) or an `nnz × nbatch` matrix.

Constructors:

    CSR(rowptr, colval, nzval[, nrows, ncols]; index = 'O', transposed = false)
    CSR(A::SparseMatrixCSC; index = 'O')

The first wraps the arrays without copying (square `nrows = ncols =
length(rowptr) - 1` when the sizes are omitted); the second builds the CSR
arrays of `A` on the host. Only lengths are validated, never array contents, so
no device memory is read.
"""
struct CSR{T, INT <: Integer, VI <: AbstractVector{INT}, VT <: AbstractVecOrMat{T}}
    rowptr::VI
    colval::VI
    nzval::VT
    nrows::Int
    ncols::Int
    index::IndexBase
    transposed::Bool

    function CSR{T, INT, VI, VT}(rowptr::VI, colval::VI, nzval::VT, nrows::Integer, ncols::Integer,
                                 index::IndexBase, transposed::Bool) where {T, INT, VI, VT}
        nrows >= 0 && ncols >= 0 || throw(InvalidValueError("CSR sizes must be nonnegative, got $nrows × $ncols"))
        length(rowptr) == nrows + 1 ||
            throw(InvalidValueError("length(rowptr) = $(length(rowptr)) does not match nrows + 1 = $(nrows + 1)"))
        nz = length(colval)
        if nzval isa AbstractMatrix
            size(nzval, 1) == nz ||
                throw(InvalidValueError("batched nzval has $(size(nzval, 1)) rows, expected nnz = $nz"))
        elseif nz == 0
            isempty(nzval) || throw(InvalidValueError("nzval has $(length(nzval)) entries but colval is empty"))
        else
            length(nzval) > 0 && length(nzval) % nz == 0 ||
                throw(InvalidValueError("length(nzval) = $(length(nzval)) is not a positive multiple of nnz = $nz"))
        end
        return new{T, INT, VI, VT}(rowptr, colval, nzval, nrows, ncols, index, transposed)
    end
end

_index_base(index::IndexBase) = index
_index_base(index::AbstractChar) = convert(IndexBase, index)
_index_base(index) = throw(InvalidValueError("index base must be 'O', 'Z' or an IndexBase, got $(repr(index))"))

function CSR(rowptr::VI, colval::VI, nzval::VT, nrows::Integer, ncols::Integer;
             index = INDEX_ONE, transposed::Bool = false) where {T, INT <: Integer, VI <: AbstractVector{INT},
                                                                 VT <: AbstractVecOrMat{T}}
    return CSR{T, INT, VI, VT}(rowptr, colval, nzval, nrows, ncols, _index_base(index), transposed)
end

function CSR(rowptr::AbstractVector{<:Integer}, colval::AbstractVector{<:Integer}, nzval::AbstractVecOrMat,
             nrows::Integer, ncols::Integer; kwargs...)
    throw(InvalidValueError("rowptr ($(typeof(rowptr))) and colval ($(typeof(colval))) must have the same array type"))
end

function CSR(rowptr::AbstractVector{<:Integer}, colval::AbstractVector{<:Integer}, nzval::AbstractVecOrMat; kwargs...)
    n = length(rowptr) - 1
    return CSR(rowptr, colval, nzval, n, n; kwargs...)
end

function CSR(A::SparseMatrixCSC{T, INT}; index = INDEX_ONE) where {T, INT}
    base = _index_base(index)
    At = copy(transpose(A))   # CSC of Aᵀ = CSR of A (plain transpose, no conjugation)
    rowptr, colval = At.colptr, At.rowval
    if base == INDEX_ZERO
        rowptr .-= one(INT)
        colval .-= one(INT)
    end
    return CSR(rowptr, colval, At.nzval, size(A, 1), size(A, 2); index = base)
end

"""
    csr_of_transpose(A::SparseMatrixCSC) -> CSR

Zero-copy reinterpretation of the CSC matrix `A` (`m × n`) as the CSR matrix of
`transpose(A)` (`n × m`): `A.colptr` becomes `rowptr`, `A.rowval` becomes
`colval`, the arrays are shared and `transposed = true` records that the caller
means `A`. This is how MadNLP passes `colPtr`/`rowVal` (PLAN §1.1).
"""
function csr_of_transpose(A::SparseMatrixCSC)
    m, n = size(A)
    return CSR(A.colptr, A.rowval, A.nzval, n, m; index = INDEX_ONE, transposed = true)
end

"""
    to_backend(A, backend) -> CSR

`A` (a `SparseMatrixCSC` or a [`CSR`](@ref)) as a `CSR` whose arrays live on the
KernelAbstractions `backend`. A `SparseMatrixCSC` is first converted on the host
(`index` keyword as in `CSR(A; index)`); arrays already on `backend` are not copied.
"""
to_backend(A::SparseMatrixCSC, backend::KernelAbstractions.Backend; index = INDEX_ONE) =
    to_backend(CSR(A; index), backend)

function to_backend(A::CSR, backend::KernelAbstractions.Backend)
    return CSR(_to_backend_array(backend, A.rowptr), _to_backend_array(backend, A.colval),
               _to_backend_array(backend, A.nzval), A.nrows, A.ncols; index = A.index, transposed = A.transposed)
end

function _to_backend_array(backend::KernelAbstractions.Backend, x::AbstractArray)
    typeof(KernelAbstractions.get_backend(x)) == typeof(backend) && return x
    y = KernelAbstractions.allocate(backend, eltype(x), size(x))
    copyto!(y, x)
    return y
end

function Adapt.adapt_structure(to, A::CSR)
    return CSR(adapt(to, A.rowptr), adapt(to, A.colval), adapt(to, A.nzval), A.nrows, A.ncols;
               index = A.index, transposed = A.transposed)
end

Base.size(A::CSR) = (A.nrows, A.ncols)
Base.size(A::CSR, d::Integer) = d <= 2 ? size(A)[d] : 1
Base.eltype(::Type{<:CSR{T}}) where {T} = T

"""
    nnz(A::CSR)

Number of stored entries of one batch member (`length(A.colval)`).
"""
SparseArrays.nnz(A::CSR) = length(A.colval)

"""
    nbatch(A::CSR)
    nbatch(desc::MatrixDescriptor)

Number of matrices in a uniform batch: `size(nzval, 2)` for matrix values,
`length(nzval) ÷ nnz` for vector values (1 for a plain matrix).
"""
function nbatch(A::CSR)
    A.nzval isa AbstractMatrix && return size(A.nzval, 2)
    nz = nnz(A)
    return nz == 0 ? 1 : length(A.nzval) ÷ nz
end

KernelAbstractions.get_backend(A::CSR) = KernelAbstractions.get_backend(A.nzval)

"""
    SparseMatrixCSC(A::CSR[, k = 1])

Host copy of batch member `k` of the stored CSR matrix (`size(A)`, index type
`INT`, base rebased to one). The `transposed` flag is not applied: for
`B = csr_of_transpose(M)`, `SparseMatrixCSC(B) == transpose(M)`.
"""
function SparseArrays.SparseMatrixCSC(A::CSR{T, INT}, k::Integer = 1) where {T, INT}
    nb = nbatch(A)
    1 <= k <= nb || throw(InvalidValueError("batch member $k out of range 1:$nb"))
    nz = nnz(A)
    rowptr = Array{INT}(A.rowptr)
    colval = Array{INT}(A.colval)
    if A.nzval isa AbstractMatrix
        nzval = Array{T}(A.nzval[:, k])
    else
        nzval = Array{T}(A.nzval[((k - 1) * nz + 1):(k * nz)])
    end
    if A.index == INDEX_ZERO
        rowptr .+= one(INT)
        colval .+= one(INT)
    end
    At = SparseMatrixCSC{T, INT}(A.ncols, A.nrows, rowptr, colval, nzval)
    return copy(transpose(At))
end

function Base.show(io::IO, A::CSR{T, INT}) where {T, INT}
    print(io, A.nrows, "×", A.ncols, " CSR{", T, ", ", INT, "} with ", nnz(A), " stored entries")
    nbatch(A) > 1 && print(io, " × ", nbatch(A), " batch members")
    print(io, " (", A.index == INDEX_ONE ? "one" : "zero", "-based", A.transposed ? ", transposed" : "", ")")
    return nothing
end

"""
    MatrixDescriptor{T,A}

Dense right-hand side / solution descriptor (≅ `CudssMatrix` for dense data):
`data::Union{Nothing,A}`, `nrows`, `ncols`, `nbatch`, `transposed`. `nrows ×
ncols` is the logical size of one batch member. With `transposed = true` the
data is row-major, i.e. a column-major array of size `(ncols, nrows)`, as in
CUDSS.jl.

    MatrixDescriptor(T, n; nbatch = 1)                       # n × 1, no data yet
    MatrixDescriptor(T, m, n; nbatch = 1, transposed = false) # as CUDSS.jl: transposed gives n × m
    MatrixDescriptor(x::AbstractArray; transposed = false)    # vector, matrix or (n, p, nbatch) array

[`update!`](@ref) re-points the descriptor to a new buffer without copying.
Descriptors created without data accept any `AbstractArray{T}`; descriptors
created from an array accept arrays of the same type.
"""
mutable struct MatrixDescriptor{T, A <: AbstractArray{T}}
    data::Union{Nothing, A}
    nrows::Int
    ncols::Int
    nbatch::Int
    transposed::Bool
end

function _check_descriptor_sizes(m, n, nb)
    m >= 0 && n >= 0 || throw(InvalidValueError("descriptor sizes must be nonnegative, got $m × $n"))
    nb >= 1 || throw(InvalidValueError("nbatch must be ≥ 1, got $nb"))
    return nothing
end

function MatrixDescriptor(::Type{T}, n::Integer; nbatch::Integer = 1) where {T}
    _check_descriptor_sizes(n, 1, nbatch)
    return MatrixDescriptor{T, AbstractArray{T}}(nothing, n, 1, nbatch, false)
end

function MatrixDescriptor(::Type{T}, m::Integer, n::Integer; nbatch::Integer = 1, transposed::Bool = false) where {T}
    _check_descriptor_sizes(m, n, nbatch)
    nrows, ncols = transposed ? (n, m) : (m, n)
    return MatrixDescriptor{T, AbstractArray{T}}(nothing, nrows, ncols, nbatch, transposed)
end

function MatrixDescriptor(x::AbstractArray{T}; transposed::Bool = false) where {T}
    ndims(x) <= 3 || throw(InvalidValueError("MatrixDescriptor takes a vector, matrix or 3-D array, got $(ndims(x)) dimensions"))
    if ndims(x) == 1
        transposed && throw(InvalidValueError("a vector descriptor cannot be transposed"))
        nrows, ncols, nb = length(x), 1, 1
    else
        nrows, ncols = transposed ? (size(x, 2), size(x, 1)) : (size(x, 1), size(x, 2))
        nb = size(x, 3)
    end
    return MatrixDescriptor{T, typeof(x)}(x, nrows, ncols, nb, transposed)
end

Base.size(desc::MatrixDescriptor) = (desc.nrows, desc.ncols)
Base.eltype(::Type{<:MatrixDescriptor{T}}) where {T} = T
nbatch(desc::MatrixDescriptor) = desc.nbatch

function KernelAbstractions.get_backend(desc::MatrixDescriptor)
    desc.data === nothing && throw(InvalidValueError("MatrixDescriptor has no data; call update! first"))
    return KernelAbstractions.get_backend(desc.data)
end

"""
    update!(desc::MatrixDescriptor, x::AbstractArray) -> desc

Point `desc` at `x` without copying (≅ `cudss_update(matrix, x)`). `x` must hold
`nrows * ncols * nbatch` entries of type `T`: either as a strided vector, or with
the descriptor's shape (`(nrows, ncols)`, `(ncols, nrows)` when transposed, plus
a trailing `nbatch` dimension for 3-D arrays). Anything else throws
`InvalidValueError`.
"""
function update!(desc::MatrixDescriptor{T, A}, x::AbstractArray) where {T, A}
    eltype(x) === T || throw(InvalidValueError("descriptor holds $T, got an array of $(eltype(x))"))
    x isa A || throw(InvalidValueError("descriptor holds arrays of type $A, got $(typeof(x))"))
    len = desc.nrows * desc.ncols * desc.nbatch
    length(x) == len ||
        throw(InvalidValueError("array has $(length(x)) entries, descriptor expects $len " *
                                "($(desc.nrows) × $(desc.ncols) × $(desc.nbatch))"))
    if ndims(x) > 1
        shape = desc.transposed ? (desc.ncols, desc.nrows) : (desc.nrows, desc.ncols)
        expected = ndims(x) == 2 ? shape : (shape..., desc.nbatch)
        ndims(x) <= 3 && size(x) == expected ||
            throw(InvalidValueError("array of size $(size(x)) does not match descriptor shape $expected"))
    end
    desc.data = x
    return desc
end
