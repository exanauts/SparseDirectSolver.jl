# Vendor bindings of the dense layer (PLAN §2.6 item 2). The core declares one
# function per vendor routine; the fallback method raises `NotSupportedError`
# and each backend adds methods for its own array types: the host BLAS/LAPACK
# of LinearAlgebra for the KA CPU backend (below), cuBLAS/cuSOLVER in
# `ext/SparseDirectSolverCUDAExt.jl`. `capabilities` probes which of them work.
#
# Conventions follow BLAS/LAPACK: `Char` flags, `ipiv` 1-based and relative to
# the matrix, `info` as an `Int` (or a vector for batched routines).

const VENDOR_FUNCTIONS = (:vendor_gemm!, :vendor_syrk!, :vendor_herk!, :vendor_trsm!, :vendor_potrf!,
                          :vendor_potrf_info!, :vendor_getrf!, :vendor_gemm_strided_batched!,
                          :vendor_trsm_batched!, :vendor_potrf_batched!,
                          :vendor_batch_pointers, :vendor_trsm_batched_ptrs!, :vendor_potrf_batched_ptrs!)

for f in VENDOR_FUNCTIONS
    @eval $f(args...) = throw(NotSupportedError(string($(string(f)), " has no vendor binding for argument types ",
                                                       join(map(typeof, args), ", "))))
end

@doc """
    vendor_gemm!(transA::Char, transB::Char, α, A, B, β, C) -> C
    vendor_syrk!(uplo::Char, α, A, β, C) -> C                  # uplo triangle of α A Aᵀ + β C
    vendor_herk!(uplo::Char, α, A, β, C) -> C                  # uplo triangle of α A Aᴴ + β C (complex)
    vendor_trsm!(side, uplo, trans, diag, α, A, B) -> B
    vendor_potrf!(uplo::Char, A) -> info::Int
    vendor_potrf_info!(uplo::Char, A, info, idx) -> info       # LAPACK info into the device Int32 vector info[idx]
    vendor_getrf!(A, ipiv) -> info::Int
    vendor_gemm_strided_batched!(transA, transB, α, A, B, β, C) -> C     # 3-D arrays
    vendor_trsm_batched!(side, uplo, trans, diag, α, A, B) -> B          # 3-D arrays
    vendor_potrf_batched!(uplo, A, info) -> info                         # 3-D A, Int32 info
    vendor_batch_pointers(A) -> ptrs                                     # device vector of the member pointers of 3-D A
    vendor_trsm_batched_ptrs!(side, uplo, trans, diag, α, m, n, Aptrs, lda, Bptrs, ldb, count)
    vendor_potrf_batched_ptrs!(uplo, n, Aptrs, lda, info, count) -> info # Int32 info, count entries

The `_ptrs` variants take device vectors of member pointers built once by
`vendor_batch_pointers` (the uniform-batch regime C, T17: no allocation in the
numeric phase); `vendor_trsm_batched!`/`vendor_potrf_batched!` build them per call.

Shared bodies for the extensions (internal; an extension keeps only its vendor calls
and forwards to these):

    _check_square_batch(A) -> n                          # order of the square members of 3-D A
    vendor_free!(x) -> nothing                           # release a device temporary early (hook, no-op default)
    _trsm_batched_via_ptrs!(side, uplo, trans, diag, α, A, B) -> B
    _potrf_batched_via_ptrs!(uplo, A, info) -> info
    _with_pivot_vector(f, ipiv, k, PI) -> f(p)           # f on a pivot vector of eltype PI, copied back to ipiv[1:k]

`_trsm_batched_via_ptrs!`/`_potrf_batched_via_ptrs!` build the member pointers with
`vendor_batch_pointers`, call the `_ptrs` binding and free the pointer vectors with
`vendor_free!`; the extension overloads `vendor_batch_pointers` and `vendor_free!`
for its array types. `_with_pivot_vector` passes `ipiv` through when its element
type is the vendor's pivot type `PI`, else runs `f` on a `similar(ipiv, PI, k)`
temporary and copies it back.

Vendor dense routines behind the `impl = :vendor` path of the dense interface.
The generic methods throw `NotSupportedError`; backends add methods for their
array types (host BLAS/LAPACK for `Array`s, cuBLAS/cuSOLVER for `CuArray`s).
""" vendor_gemm!

# Host arrays the CPU BLAS/LAPACK accept: plain matrices, the panel views
# `reshape(view(buf, a:b), f, w)` and `view(A, i, j)` of host arrays or panels with
# `Int` or range indices (strided; `view(A, [1, 3], :)` is not and falls outside).
# Backends whose arrays are `DenseArray`s (GPU arrays) must not reach host BLAS,
# hence the explicit `Array` parent.
"""
    vendor_free!(x) -> nothing

Release the device temporary `x` without waiting for the garbage collector. The
default does nothing; backends overload it for their array types (`unsafe_free!`).
"""
vendor_free!(x) = nothing

function _check_square_batch(A::AbstractArray{<:Any, 3})
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("batch members are $(size(A, 1))×$(size(A, 2)), not square"))
    return size(A, 1)
end

# the per-call batched wrappers of the backends: member pointers built here, freed after the call
function _trsm_batched_via_ptrs!(side::Char, uplo::Char, trans::Char, diag::Char, α, A::AbstractArray{<:Any, 3},
                                 B::AbstractArray{<:Any, 3})
    m, n, nb = size(B)
    nb == 0 && return B
    GC.@preserve A B begin
        Aptrs = vendor_batch_pointers(A)
        Bptrs = vendor_batch_pointers(B)
        vendor_trsm_batched_ptrs!(side, uplo, trans, diag, α, m, n, Aptrs, stride(A, 2), Bptrs, stride(B, 2), nb)
        vendor_free!(Aptrs)
        vendor_free!(Bptrs)
    end
    return B
end

function _potrf_batched_via_ptrs!(uplo::Char, A::AbstractArray{<:Any, 3}, info::AbstractVector)
    n = _check_square_batch(A)
    nb = size(A, 3)
    nb == 0 && return info
    GC.@preserve A begin
        Aptrs = vendor_batch_pointers(A)
        vendor_potrf_batched_ptrs!(uplo, n, Aptrs, stride(A, 2), info, nb)
        vendor_free!(Aptrs)
    end
    return info
end

# vendor pivots have one index type `PI` (`Cint` for cuSOLVER/rocSOLVER, `Int64` for
# oneMKL); other index vectors go through a temporary, copied back to `ipiv[1:k]`
function _with_pivot_vector(f, ipiv::AbstractVector, k::Integer, ::Type{PI}) where {PI <: Integer}
    eltype(ipiv) === PI && length(ipiv) == k && return f(ipiv)   # LAPACK writes exactly k pivots
    p = similar(ipiv, PI, k)
    info = f(p)
    view(ipiv, 1:k) .= p
    vendor_free!(p)
    return info
end

const HostStridedIndex = Tuple{Vararg{Union{Int, AbstractRange{Int}}}}
const HostPanel{T} = Base.ReshapedArray{T, 2, <:SubArray{T, 1, <:Array{T}}}
const HostMatrix{T} = Union{Matrix{T}, SubArray{T, 2, <:Array{T}, <:HostStridedIndex}, HostPanel{T},
                            SubArray{T, 2, <:HostPanel{T}, <:HostStridedIndex}}

const BlasT = LinearAlgebra.BlasFloat

vendor_gemm!(tA::Char, tB::Char, α, A::HostMatrix{T}, B::HostMatrix{T}, β, C::HostMatrix{T}) where {T <: BlasT} =
    BLAS.gemm!(tA, tB, T(α), A, B, T(β), C)
vendor_syrk!(uplo::Char, α, A::HostMatrix{T}, β, C::HostMatrix{T}) where {T <: BlasT} =
    BLAS.syrk!(uplo, 'N', T(α), A, T(β), C)
vendor_herk!(uplo::Char, α, A::HostMatrix{T}, β, C::HostMatrix{T}) where {T <: Complex{<:Union{Float32, Float64}}} =
    BLAS.herk!(uplo, 'N', real(T)(α), A, real(T)(β), C)
vendor_trsm!(side::Char, uplo::Char, trans::Char, diag::Char, α, A::HostMatrix{T}, B::HostMatrix{T}) where {T <: BlasT} =
    BLAS.trsm!(side, uplo, trans, diag, T(α), A, B)
vendor_potrf!(uplo::Char, A::HostMatrix{<:BlasT}) = Int(LAPACK.potrf!(uplo, A)[2])
function vendor_potrf_info!(uplo::Char, A::HostMatrix{<:BlasT}, info::Vector{Int32}, idx::Integer)
    info[idx] = Int32(LAPACK.potrf!(uplo, A)[2])
    return info
end
function vendor_getrf!(A::HostMatrix{<:BlasT}, ipiv::AbstractVector{<:Integer})
    _, p, info = LAPACK.getrf!(A)
    view(ipiv, 1:length(p)) .= p
    return Int(info)
end
