# Vendor bindings of the dense layer (PLAN §2.6 item 2). The core declares one
# function per vendor routine; the fallback method raises `NotSupportedError`
# and each backend adds methods for its own array types: the host BLAS/LAPACK
# of LinearAlgebra for the KA CPU backend (below), cuBLAS/cuSOLVER in
# `ext/SparseDirectSolverCUDAExt.jl`. `capabilities` probes which of them work.
#
# Conventions follow BLAS/LAPACK: `Char` flags, `ipiv` 1-based and relative to
# the matrix, `info` as an `Int` (or a vector for batched routines).

const VENDOR_FUNCTIONS = (:vendor_gemm!, :vendor_syrk!, :vendor_herk!, :vendor_trsm!, :vendor_potrf!,
                          :vendor_getrf!, :vendor_sytrf!, :vendor_gemm_strided_batched!,
                          :vendor_trsm_batched!, :vendor_potrf_batched!, :vendor_getrf_batched!)

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
    vendor_getrf!(A, ipiv) -> info::Int
    vendor_sytrf!(uplo::Char, A, ipiv) -> info::Int            # Bunch–Kaufman (capability probe only)
    vendor_gemm_strided_batched!(transA, transB, α, A, B, β, C) -> C     # 3-D arrays
    vendor_trsm_batched!(side, uplo, trans, diag, α, A, B) -> B          # 3-D arrays
    vendor_potrf_batched!(uplo, A, info) -> info                         # 3-D A, Int32 info
    vendor_getrf_batched!(A, ipiv, info) -> info                         # 3-D A, Int32 ipiv matrix

Vendor dense routines behind the `impl = :vendor` path of the dense interface.
The generic methods throw `NotSupportedError`; backends add methods for their
array types (host BLAS/LAPACK for `Array`s, cuBLAS/cuSOLVER for `CuArray`s).
""" vendor_gemm!

# Host arrays the CPU BLAS/LAPACK accept: plain matrices and the panel views
# `reshape(view(buf, a:b), f, w)` / `view(A, i, j)` of host arrays. Backends whose
# arrays are `DenseArray`s (GPU arrays) must not reach host BLAS, hence the
# explicit `Array` parent.
const HostMatrix{T} = Union{Matrix{T}, SubArray{T, 2, <:Array{T}},
                            Base.ReshapedArray{T, 2, <:SubArray{T, 1, <:Array{T}}}}

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
function vendor_getrf!(A::HostMatrix{<:BlasT}, ipiv::AbstractVector{<:Integer})
    _, p, info = LAPACK.getrf!(A)
    view(ipiv, 1:length(p)) .= p
    return Int(info)
end
function vendor_sytrf!(uplo::Char, A::HostMatrix{<:BlasT}, ipiv::AbstractVector{<:Integer})
    _, p, info = LAPACK.sytrf!(uplo, A)
    view(ipiv, 1:length(p)) .= p
    return Int(info)
end
