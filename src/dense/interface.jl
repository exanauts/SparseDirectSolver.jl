# Dense-op interface of the numeric and solve phases (PLAN §2.6). Every op takes
# `impl ∈ (:auto, :generic, :vendor, :ka)`:
#
# * `:generic` — the LinearAlgebra entry points each backend routes to its
#   vendor library (`mul!`, triangular `ldiv!`/`rdiv!`, `cholesky!`, `lu!`);
# * `:vendor`  — the `vendor_*` bindings of `src/dense/vendor.jl` and `ext/`;
# * `:ka`      — the KernelAbstractions fallbacks of `src/dense/fallback/`;
# * `:auto`    — vendor if `capabilities(backend, T)` says so, else generic if
#   available, else KA.
#
# Flags accept BLAS characters or symbols (`'N'`/`:N`, `'L'`/`:L`, …). Batched
# ops work on 3-D arrays whose third dimension is the batch, usually
# [`strided_batch`](@ref) views (data pointer + stride + count) of a flat buffer.

const DENSE_IMPLS = (:auto, :generic, :vendor, :ka)

# capability field behind the :generic and :vendor paths of each op (nothing = no such path)
const DENSE_OPS = (
    gemm = (generic = :generic_mul, vendor = :vendor_gemm),
    syrk = (generic = :generic_mul, vendor = :vendor_syrk),
    herk = (generic = :generic_mul, vendor = :vendor_herk),
    trsm = (generic = :generic_trsm, vendor = :vendor_trsm),
    potrf = (generic = :generic_cholesky, vendor = :vendor_potrf),
    getrf = (generic = :generic_lu, vendor = :vendor_getrf),
    laswp = (generic = nothing, vendor = nothing),
    gemm_strided_batched = (generic = nothing, vendor = :vendor_gemm_strided_batched),
    trsm_strided_batched = (generic = nothing, vendor = :vendor_trsm_batched),
)

# DENSE_OPS as a concretely typed table (`:none` = no such path), for `select_impl`
const _DENSE_OP_FIELDS = Dict{Symbol, Tuple{Symbol, Symbol}}(
    op => (something(spec.generic, :none), something(spec.vendor, :none)) for (op, spec) in pairs(DENSE_OPS))

function _flag_char(x, allowed::String, what::String)
    c = x isa Symbol ? (length(String(x)) == 1 ? only(String(x)) : '?') : x isa AbstractChar ? Char(x) : '?'
    c = uppercase(c)
    c in allowed || throw(InvalidValueError("invalid $what flag $(repr(x)); expected one of $(join(collect(allowed), ", "))"))
    return c
end
_trans_char(x) = _flag_char(x, "NTC", "trans")
_uplo_char(x) = _flag_char(x, "LU", "uplo")
_side_char(x) = _flag_char(x, "LR", "side")
_diag_char(x) = _flag_char(x, "NU", "diag")

function _check_gemm_dims(C, A, B, tA::Char, tB::Char)
    m, k = tA == 'N' ? (size(A, 1), size(A, 2)) : (size(A, 2), size(A, 1))
    kb, n = tB == 'N' ? (size(B, 1), size(B, 2)) : (size(B, 2), size(B, 1))
    (k == kb && size(C, 1) == m && size(C, 2) == n) ||
        throw(DimensionMismatch("gemm: C is $(size(C, 1))×$(size(C, 2)), op(A) is $m×$k, op(B) is $kb×$n"))
    return nothing
end

function _check_syrk_dims(C, A)
    size(C, 1) == size(C, 2) == size(A, 1) ||
        throw(DimensionMismatch("syrk: C is $(size(C, 1))×$(size(C, 2)), A has $(size(A, 1)) rows"))
    return nothing
end

function _check_trsm_dims(side::Char, A, B)
    n = side == 'L' ? size(B, 1) : size(B, 2)
    size(A, 1) == size(A, 2) == n ||
        throw(DimensionMismatch("trsm: A is $(size(A, 1))×$(size(A, 2)), B is $(size(B, 1))×$(size(B, 2)) (side $side)"))
    return nothing
end

function _check_batch(C, As...)
    for A in As
        size(A, 3) == size(C, 3) || throw(DimensionMismatch("batch counts differ: $(size(A, 3)) and $(size(C, 3))"))
    end
    return nothing
end

"""
    dense_impls(op::Symbol, backend, T) -> Vector{Symbol}

The implementations of dense op `op` (`:gemm`, `:syrk`, `:herk`, `:trsm`,
`:potrf`, `:getrf`, `:laswp`, `:gemm_strided_batched`, `:trsm_strided_batched`)
available for element type `T` on `backend` according to
[`capabilities`](@ref), in `:auto` preference order (`:vendor`, `:generic`,
`:ka`). `:ka` is always present.
"""
function dense_impls(op::Symbol, backend, ::Type{T}) where {T}
    haskey(DENSE_OPS, op) || throw(InvalidValueError("unknown dense op :$op; expected one of $(keys(DENSE_OPS))"))
    spec = DENSE_OPS[op]
    caps = capabilities(backend, T)
    impls = Symbol[]
    spec.vendor === nothing || getfield(caps, spec.vendor) && push!(impls, :vendor)
    spec.generic === nothing || getfield(caps, spec.generic) && push!(impls, :generic)
    push!(impls, :ka)
    return impls
end

"""
    select_impl(op::Symbol, X::AbstractArray, impl::Symbol) -> Symbol

Resolve `impl` for dense op `op` on the backend and element type of `X`:
`:auto` becomes the first entry of [`dense_impls`](@ref); an explicit `:generic`
or `:vendor` the capability table does not list raises `NotSupportedError`;
anything else raises `InvalidValueError`.
"""
function select_impl(op::Symbol, X::AbstractArray, impl::Symbol)
    impl in DENSE_IMPLS || throw(InvalidValueError("invalid dense impl :$impl; expected one of $DENSE_IMPLS"))
    impl === :ka && return :ka
    haskey(DENSE_OPS, op) || throw(InvalidValueError("unknown dense op :$op; expected one of $(keys(DENSE_OPS))"))
    # same answer as `dense_impls`, without building the vector (called per front)
    gfield, vfield = _DENSE_OP_FIELDS[op]
    caps = capabilities(KernelAbstractions.get_backend(X), eltype(X))
    vendor = vfield !== :none && getfield(caps, vfield)::Bool
    generic = gfield !== :none && getfield(caps, gfield)::Bool
    impl === :auto && return vendor ? :vendor : generic ? :generic : :ka
    (impl === :vendor ? vendor : generic) ||
        throw(NotSupportedError("impl = :$impl is not available for $op with $(eltype(X)) on $(KernelAbstractions.get_backend(X))"))
    return impl
end

_op_wrap(X, t::Char) = t == 'N' ? X : t == 'T' ? transpose(X) : adjoint(X)

"""
    gemm!(C, A, B, α = 1, β = 0; transA = :N, transB = :N, impl = :auto) -> C

`C ← α op(A) op(B) + β C` with `op` from `:N`, `:T`, `:C`.
"""
function gemm!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, α = true, β = false;
               transA = 'N', transB = 'N', impl::Symbol = :auto)
    tA, tB = _trans_char(transA), _trans_char(transB)
    _check_gemm_dims(C, A, B, tA, tB)
    T = eltype(C)
    p = select_impl(:gemm, C, impl)
    if p === :vendor
        vendor_gemm!(tA, tB, T(α), A, B, T(β), C)
    elseif p === :generic
        mul!(C, _op_wrap(A, tA), _op_wrap(B, tB), T(α), T(β))
    else
        ka_gemm!(C, A, B, α, β; transA = tA, transB = tB)
    end
    return C
end

"""
    syrk!(C, A, α, β; uplo = :L, impl = :auto) -> C

Symmetric rank-k update of the `uplo` triangle of `C`: `C ← α A Aᵀ + β C`
(plain transpose, also for complex `T`; see [`herk!`](@ref) for `A Aᴴ`).
`:vendor` and `:ka` leave the other triangle untouched; `:generic` (`mul!`)
overwrites it with the same symmetric update.
"""
function syrk!(C::AbstractMatrix, A::AbstractMatrix, α, β; uplo = 'L', impl::Symbol = :auto)
    ul = _uplo_char(uplo)
    _check_syrk_dims(C, A)
    return _syrk_impl!(select_impl(:syrk, C, impl), ul, C, A, α, β)
end

# `syrk!` with an already resolved implementation `p` (no capability lookup)
function _syrk_impl!(p::Symbol, ul::Char, C::AbstractMatrix, A::AbstractMatrix, α, β)
    T = eltype(C)
    if p === :vendor
        vendor_syrk!(ul, T(α), A, T(β), C)
    elseif p === :generic
        mul!(C, A, transpose(A), T(α), T(β))
    else
        ka_syrk!(C, A, α, β; uplo = ul)
    end
    return C
end

"""
    herk!(C, A, α, β; uplo = :L, impl = :auto) -> C

Hermitian rank-k update of the `uplo` triangle of `C`: `C ← α A Aᴴ + β C` with
real `α`, `β` (BLAS `herk`; the diagonal of the result is real). For real `T`
this is [`syrk!`](@ref). Triangle semantics as in `syrk!`.
"""
function herk!(C::AbstractMatrix, A::AbstractMatrix, α::Real, β::Real; uplo = 'L', impl::Symbol = :auto)
    T = eltype(C)
    T <: Real && return syrk!(C, A, α, β; uplo, impl)
    ul = _uplo_char(uplo)
    _check_syrk_dims(C, A)
    return _herk_impl!(select_impl(:herk, C, impl), ul, C, A, α, β)
end

# `herk!` with an already resolved implementation `p` (`:syrk` for real `T`), no capability lookup
function _herk_impl!(p::Symbol, ul::Char, C::AbstractMatrix, A::AbstractMatrix, α::Real, β::Real)
    T = eltype(C)
    T <: Real && return _syrk_impl!(p, ul, C, A, α, β)
    if p === :vendor
        vendor_herk!(ul, real(T)(α), A, real(T)(β), C)
    elseif p === :generic
        mul!(C, A, adjoint(A), T(α), T(β))
    else
        ka_syrk!(C, A, α, β; uplo = ul, conjugate = true)
    end
    return C
end

function _generic_trsm!(side::Char, uplo::Char, trans::Char, diag::Char, α, A, B)
    T = eltype(B)
    if iszero(α)
        fill!(B, zero(T))
        return B
    end
    isone(α) || (B .*= T(α))
    tri = uplo == 'L' ? (diag == 'U' ? UnitLowerTriangular(A) : LowerTriangular(A)) :
          (diag == 'U' ? UnitUpperTriangular(A) : UpperTriangular(A))
    M = _op_wrap(tri, trans)
    side == 'L' ? ldiv!(M, B) : rdiv!(B, M)
    return B
end

"""
    trsm!(side, uplo, trans, diag, α, A, B; impl = :auto) -> B

Triangular solve with multiple right-hand sides, BLAS `trsm`: `B ← α op(A)⁻¹ B`
for `side = :L`, `B ← α B op(A)⁻¹` for `side = :R`; `A` is lower (`uplo = :L`)
or upper triangular, with unit diagonal if `diag = :U`; `trans ∈ (:N, :T, :C)`.
"""
function trsm!(side, uplo, trans, diag, α, A::AbstractMatrix, B::AbstractMatrix; impl::Symbol = :auto)
    sd, ul, tr, dg = _side_char(side), _uplo_char(uplo), _trans_char(trans), _diag_char(diag)
    _check_trsm_dims(sd, A, B)
    return _trsm_impl!(select_impl(:trsm, B, impl), sd, ul, tr, dg, α, A, B)
end

# `trsm!` with an already resolved implementation `p` (no capability lookup)
function _trsm_impl!(p::Symbol, sd::Char, ul::Char, tr::Char, dg::Char, α, A::AbstractMatrix, B::AbstractMatrix)
    if p === :vendor
        vendor_trsm!(sd, ul, tr, dg, eltype(B)(α), A, B)
    elseif p === :generic
        _generic_trsm!(sd, ul, tr, dg, α, A, B)
    else
        ka_trsm!(sd, ul, tr, dg, α, A, B)
    end
    return B
end

_device_info(A, n::Integer = 1) = KernelAbstractions.zeros(KernelAbstractions.get_backend(A), Int32, n)

"""
    potrf!(uplo, A; impl = :auto) -> info::Int

Cholesky factorization of the `uplo` triangle of the square matrix `A` in
place (`A = LLᴴ` or `UᴴU`). Returns 0 on success, or the first column `j`
(1-based) whose leading minor is not positive definite, as LAPACK `potrf`.
Reading `info` synchronizes with the device.
"""
function potrf!(uplo, A::AbstractMatrix; impl::Symbol = :auto)
    ul = _uplo_char(uplo)
    LinearAlgebra.checksquare(A)
    p = select_impl(:potrf, A, impl)
    if p === :vendor || p === :generic
        info = p === :vendor ? vendor_potrf!(ul, A) :
               Int(cholesky!(Hermitian(A, ul == 'L' ? :L : :U), NoPivot(); check = false).info)
        info == 0 || return info
        # cuSOLVER zpotrf (n = 32) returns info = 0 when only the last pivot is
        # not positive (T03 CI); accept the factor only if its diagonal is.
        return Int(only(Array(ka_chol_diag_info!(_device_info(A), A))))
    else
        info = ka_potrf!(ul, A, _device_info(A))
        return Int(only(Array(info)))
    end
end

"""
    potrf_info!(uplo, A, info, idx = 1; impl = :auto) -> info

Asynchronous [`potrf!`](@ref): Cholesky of the `uplo` triangle of `A` in place,
with the LAPACK `info` (0, or the first non-positive pivot column) written to
`info[idx]` of the device `Int32` vector `info` instead of being returned, so
the numeric phase can factor many fronts and read their status once
(PLAN §3.9). `:vendor` (`vendor_potrf_info!`) and `:generic` results are
validated on the device by [`ka_chol_check_info!`](@ref) (cuSOLVER can miss a
non-positive last pivot, see `potrf!`); `:ka` is [`ka_potrf!`](@ref). The
`:vendor` and `:ka` paths do not synchronize with the host; `:generic` does
when the backend's `cholesky!` reads its `info` (CUDA).
"""
function potrf_info!(uplo, A::AbstractMatrix, info::AbstractVector{Int32}, idx::Integer = 1; impl::Symbol = :auto)
    ul = _uplo_char(uplo)
    LinearAlgebra.checksquare(A)
    1 <= idx <= length(info) || throw(DimensionMismatch("info index $idx outside 1:$(length(info))"))
    return _potrf_info_impl!(select_impl(:potrf, A, impl), ul, A, info, idx)
end

# `potrf_info!` with an already resolved implementation `p` (no capability lookup)
function _potrf_info_impl!(p::Symbol, ul::Char, A::AbstractMatrix, info::AbstractVector{Int32}, idx::Integer)
    if p === :vendor
        vendor_potrf_info!(ul, A, info, idx)
        ka_chol_check_info!(info, idx, A)
    elseif p === :generic
        cholesky!(Hermitian(A, ul == 'L' ? :L : :U), NoPivot(); check = false)
        fill!(view(info, idx:idx), Int32(0))
        ka_chol_check_info!(info, idx, A)
    else
        ka_potrf!(ul, A, info; offset = idx - 1)
    end
    return info
end

"""
    getrf!(A, ipiv; impl = :auto) -> info::Int

LU factorization with partial pivoting of the `m × n` matrix `A` in place,
`P A = L U` (unit `L`), LAPACK `getrf`: `ipiv[1:min(m, n)]` receives the row
interchanges (1-based, see [`laswp!`](@ref)); `info` is 0 or the first column
with an exactly zero pivot. `ipiv` lives on the backend of `A`. Reading `info`
synchronizes with the device.
"""
function getrf!(A::AbstractMatrix, ipiv::AbstractVector{<:Integer}; impl::Symbol = :auto)
    length(ipiv) >= min(size(A)...) ||
        throw(DimensionMismatch("ipiv has length $(length(ipiv)) < min(m, n) = $(min(size(A)...))"))
    p = select_impl(:getrf, A, impl)
    if p === :vendor
        return vendor_getrf!(A, ipiv)
    elseif p === :generic
        F = lu!(A, RowMaximum(); check = false)
        view(ipiv, 1:length(F.ipiv)) .= F.ipiv
        return Int(F.info)
    else
        info = ka_getrf!(A, ipiv, _device_info(A))
        return Int(only(Array(info)))
    end
end

"""
    laswp!(A, ipiv; npiv = length(ipiv), reverse = false, impl = :auto) -> A

Apply the row interchanges of `ipiv` (as returned by [`getrf!`](@ref)) to `A`:
for `i = 1:npiv` swap rows `i` and `ipiv[i]`, so that `laswp!(A, ipiv)`
computes `P A`; `reverse = true` applies them backwards (`Pᵀ A`). `npiv` is
LAPACK's `k2` (with `k1 = 1`): pass `npiv = min(m, n)` when `ipiv` is an
oversized buffer filled by `getrf!`. Only `:ka` exists (no generic entry point;
vendor bindings may be added per backend).
"""
function laswp!(A::AbstractMatrix, ipiv::AbstractVector{<:Integer}; npiv::Integer = length(ipiv),
                reverse::Bool = false, impl::Symbol = :auto)
    select_impl(:laswp, A, impl)
    return ka_laswp!(A, ipiv; npiv, reverse)
end

"""
    gemm_strided_batched!(C, A, B, α = 1, β = 0; transA = :N, transB = :N, impl = :auto) -> C

Uniform strided batch of [`gemm!`](@ref): `C[:, :, i] ← α op(A[:, :, i]) op(B[:, :, i]) + β C[:, :, i]`
for every member `i` of the 3-D arrays (see [`strided_batch`](@ref)). Implementations:
`:vendor`, `:ka`.
"""
function gemm_strided_batched!(C::AbstractArray{<:Any, 3}, A::AbstractArray{<:Any, 3}, B::AbstractArray{<:Any, 3},
                               α = true, β = false; transA = 'N', transB = 'N', impl::Symbol = :auto)
    tA, tB = _trans_char(transA), _trans_char(transB)
    _check_gemm_dims(C, A, B, tA, tB)
    _check_batch(C, A, B)
    T = eltype(C)
    size(C, 3) == 0 && return C
    if select_impl(:gemm_strided_batched, C, impl) === :vendor
        vendor_gemm_strided_batched!(tA, tB, T(α), A, B, T(β), C)
    else
        ka_gemm_strided_batched!(C, A, B, α, β; transA = tA, transB = tB)
    end
    return C
end

"""
    trsm_strided_batched!(side, uplo, trans, diag, α, A, B; impl = :auto) -> B

Uniform strided batch of [`trsm!`](@ref) on 3-D arrays `A` (`n × n × count`) and
`B`. Implementations: `:vendor`, `:ka`.
"""
function trsm_strided_batched!(side, uplo, trans, diag, α, A::AbstractArray{<:Any, 3}, B::AbstractArray{<:Any, 3};
                               impl::Symbol = :auto)
    sd, ul, tr, dg = _side_char(side), _uplo_char(uplo), _trans_char(trans), _diag_char(diag)
    _check_trsm_dims(sd, A, B)
    _check_batch(B, A)
    size(B, 3) == 0 && return B
    if select_impl(:trsm_strided_batched, B, impl) === :vendor
        vendor_trsm_batched!(sd, ul, tr, dg, eltype(B)(α), A, B)
    else
        ka_trsm_strided_batched!(sd, ul, tr, dg, α, A, B)
    end
    return B
end

"""
    strided_batch(buf, offset, m, n, stride, count) -> AbstractArray{T,3}

The uniform batch of `count` column-major `m × n` matrices stored in the flat
vector `buf`, the first starting after `offset` entries and each `stride ≥ m n`
entries after the previous one, as an `m × n × count` view sharing `buf`
(no copy). `stride` must be a multiple of `m`, and when `stride > m n` the
buffer must extend over the full `stride` of the last member too
(`offset + stride * count ≤ length(buf)`).
"""
function strided_batch(buf::AbstractVector, offset::Integer, m::Integer, n::Integer, stride::Integer, count::Integer)
    min(offset, m, n, count) >= 0 || throw(InvalidValueError("strided_batch: negative offset, size or count"))
    stride >= m * n || throw(InvalidValueError("strided_batch: stride $stride < m n = $(m * n)"))
    if stride == m * n || m == 0 || n == 0
        offset + m * n * count <= length(buf) || throw(InvalidValueError("strided_batch: range exceeds the buffer"))
        return reshape(view(buf, (offset + 1):(offset + m * n * count)), m, n, count)
    end
    stride % m == 0 || throw(InvalidValueError("strided_batch: stride $stride is not a multiple of m = $m"))
    offset + stride * count <= length(buf) || throw(InvalidValueError("strided_batch: range exceeds the buffer"))
    # each member padded to `stride`: a strided view of a reshaped contiguous range
    padded = reshape(view(buf, (offset + 1):(offset + stride * count)), m, stride ÷ m, count)
    return view(padded, :, 1:n, :)
end
