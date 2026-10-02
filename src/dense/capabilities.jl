# Capability audit of the dense layer (PLAN §2.6, milestone M0): which generic
# LinearAlgebra entry points and vendor bindings work on a backend for an
# element type, probed once with tiny try/catch calls whose results are also
# checked against the host LinearAlgebra reference, then cached per
# `(backend, T)`. `select_impl` reads the table for `impl = :auto`.

"""
    DenseCapabilities

Result of [`capabilities`](@ref): one `Bool` per probe.

* `generic_mul`, `generic_trsm` (triangular `ldiv!`/`rdiv!`), `generic_cholesky`
  (`cholesky!`), `generic_lu` (`lu!`): LinearAlgebra entry points.
* `vendor_gemm`, `vendor_syrk`, `vendor_herk`, `vendor_trsm`, `vendor_potrf`,
  `vendor_getrf`, `vendor_sytrf`, `vendor_gemm_strided_batched`,
  `vendor_trsm_batched`, `vendor_potrf_batched`, `vendor_getrf_batched`: the
  `vendor_*` bindings of the backend.
* `reshape_view_mul`: `mul!` accepts a panel view `reshape(view(buf, r), f, w)`.
* `atomic_add`: `Atomix.@atomic x[i] += v` works on `T` in a KA kernel.
"""
struct DenseCapabilities
    generic_mul::Bool
    generic_trsm::Bool
    generic_cholesky::Bool
    generic_lu::Bool
    vendor_gemm::Bool
    vendor_syrk::Bool
    vendor_herk::Bool
    vendor_trsm::Bool
    vendor_potrf::Bool
    vendor_getrf::Bool
    vendor_sytrf::Bool
    vendor_gemm_strided_batched::Bool
    vendor_trsm_batched::Bool
    vendor_potrf_batched::Bool
    vendor_getrf_batched::Bool
    reshape_view_mul::Bool
    atomic_add::Bool
end

function Base.show(io::IO, caps::DenseCapabilities)
    on = [String(f) for f in fieldnames(DenseCapabilities) if getfield(caps, f)]
    print(io, "DenseCapabilities(", join(on, ", "), ")")
end

const _CAPABILITIES = Dict{Tuple{Any, DataType}, DenseCapabilities}()
const _CAPABILITIES_LOCK = ReentrantLock()

"""
    capabilities(backend, T) -> DenseCapabilities

Dense-layer capabilities of `backend` (a KernelAbstractions backend) for
element type `T`, probed on first use with small try/catch calls and cached
per `(backend, T)`. A probe counts as available only if it neither throws nor
returns a wrong result.
"""
function capabilities(backend, ::Type{T}) where {T}
    key = (backend, T)
    lock(_CAPABILITIES_LOCK) do
        get!(() -> _probe_capabilities(backend, T), _CAPABILITIES, key)
    end
end

"""
    print_capabilities([io = stdout], backend; eltypes = (Float32, Float64, ComplexF32, ComplexF64))

Print the [`capabilities`](@ref) table of `backend`: one row per probe, one
column per element type.
"""
function print_capabilities(io::IO, backend; eltypes = (Float32, Float64, ComplexF32, ComplexF64))
    caps = [capabilities(backend, T) for T in eltypes]
    names = fieldnames(DenseCapabilities)
    w = maximum(length ∘ String, names)
    cw = max(maximum(length ∘ string, eltypes), 3)
    println(io, "Dense capabilities of ", backend)
    println(io, rpad("capability", w), " | ", join((rpad(string(T), cw) for T in eltypes), " | "))
    println(io, repeat('-', w), "-|-", join((repeat('-', cw) for _ in eltypes), "-|-"))
    for f in names
        println(io, rpad(String(f), w), " | ", join((rpad(getfield(c, f) ? "yes" : "no", cw) for c in caps), " | "))
    end
    return nothing
end
print_capabilities(backend; kwargs...) = print_capabilities(stdout, backend; kwargs...)

# ---------------------------------------------------------------------------
# probes

const _PROBE_N = 4

function _probe(f)
    try
        return f() === true
    catch
        return false
    end
end

function _to_dev(backend, x::Array{T}) where {T}
    y = KernelAbstractions.allocate(backend, T, size(x))
    copyto!(y, x)
    return y
end

_probe_close(x, ref) = (h = Array(x); size(h) == size(ref) && isapprox(h, ref; rtol = sqrt(eps(real(eltype(ref))))))

# deterministic, well-conditioned probe data
function _probe_general(::Type{T}, m, n, shift = 0) where {T}
    A = [T(1 // (i + j + shift)) for i in 1:m, j in 1:n]
    T <: Complex && (A .+= im .* T[T(1 // (2i + j + shift)) for i in 1:m, j in 1:n])
    for i in 1:min(m, n)
        A[i, i] += T(m + n)
    end
    return A
end
_probe_hpd(::Type{T}, n) where {T} = (G = _probe_general(T, n, n); Matrix{T}(Hermitian(G * G' + n * I)))

@kernel function _atomic_probe_kernel!(x, v)
    i = @index(Global)
    Atomix.@atomic x[1] += v
end

function _probe_atomic(backend, ::Type{T}) where {T}
    x = _to_dev(backend, zeros(T, 1))
    _atomic_probe_kernel!(backend, 64)(x, one(T); ndrange = 256)
    KernelAbstractions.synchronize(backend)
    return Array(x)[1] == T(256)
end

function _probe_capabilities(backend, ::Type{T}) where {T}
    n = _PROBE_N
    A = _probe_general(T, n, n)
    B = _probe_general(T, n, n, 1)
    S = _probe_hpd(T, n)
    L = Matrix(cholesky(Hermitian(S, :L)).L)
    F = lu(A)
    dev(x) = _to_dev(backend, x)
    lower(x) = tril(Array(x))
    batch(X, Y) = cat(X, Y; dims = 3)

    generic_mul = _probe() do
        C = dev(zeros(T, n, n))
        mul!(C, dev(A), dev(B))
        D = dev(zeros(T, n, n))
        mul!(D, transpose(dev(A)), adjoint(dev(B)), T(2), T(0))
        _probe_close(C, A * B) && _probe_close(D, 2 * transpose(A) * B')
    end
    generic_trsm = _probe() do
        X = dev(copy(B))
        ldiv!(LowerTriangular(dev(A)), X)
        Y = dev(copy(B))
        rdiv!(Y, transpose(UpperTriangular(dev(A))))
        _probe_close(X, LowerTriangular(A) \ B) && _probe_close(Y, B / transpose(UpperTriangular(A)))
    end
    generic_cholesky = _probe() do
        X = dev(copy(S))
        cholesky!(Hermitian(X, :L), NoPivot(); check = false).info == 0 && isapprox(lower(X), L; rtol = sqrt(eps(real(T))))
    end
    generic_lu = _probe() do
        X = dev(copy(A))
        G = lu!(X, RowMaximum(); check = false)
        G.info == 0 && _probe_close(X, F.factors) && Array(G.ipiv) == F.ipiv
    end
    vendor_gemm = _probe() do
        C = dev(zeros(T, n, n))
        vendor_gemm!('N', 'C', one(T), dev(A), dev(B), zero(T), C)
        _probe_close(C, A * B')
    end
    vendor_syrk = _probe() do
        C = dev(zeros(T, n, n))
        vendor_syrk!('L', one(T), dev(A), zero(T), C)
        isapprox(lower(C), tril(A * transpose(A)); rtol = sqrt(eps(real(T))))
    end
    vendor_herk = T <: Complex && _probe() do
        C = dev(zeros(T, n, n))
        vendor_herk!('L', one(real(T)), dev(A), zero(real(T)), C)
        isapprox(lower(C), tril(A * A'); rtol = sqrt(eps(real(T))))
    end
    vendor_trsm = _probe() do
        X = dev(copy(B))
        vendor_trsm!('L', 'L', 'N', 'N', one(T), dev(A), X)
        _probe_close(X, LowerTriangular(A) \ B)
    end
    vendor_potrf = _probe() do
        X = dev(copy(S))
        ok = vendor_potrf!('L', X) == 0 && isapprox(lower(X), L; rtol = sqrt(eps(real(T))))
        # the asynchronous variant of the numeric phase (device info, no host read)
        Y = dev(copy(S))
        info = KernelAbstractions.zeros(backend, Int32, 2)
        vendor_potrf_info!('L', Y, info, 2)
        ok && Array(info) == Int32[0, 0] && isapprox(lower(Y), L; rtol = sqrt(eps(real(T))))
    end
    vendor_getrf = _probe() do
        X = dev(copy(A))
        ipiv = KernelAbstractions.zeros(backend, Int32, n)
        vendor_getrf!(X, ipiv) == 0 && _probe_close(X, F.factors) && Array(ipiv) == F.ipiv
    end
    vendor_sytrf = _probe() do
        X = dev(Matrix{T}(Symmetric(A + transpose(A))))
        ipiv = KernelAbstractions.zeros(backend, Int32, n)
        vendor_sytrf!('L', X, ipiv) == 0
    end
    vendor_gemm_strided_batched = _probe() do
        C = dev(zeros(T, n, n, 2))
        vendor_gemm_strided_batched!('N', 'N', one(T), dev(batch(A, B)), dev(batch(B, A)), zero(T), C)
        _probe_close(C, batch(A * B, B * A))
    end
    vendor_trsm_batched = _probe() do
        X = dev(batch(B, A))
        vendor_trsm_batched!('L', 'L', 'N', 'N', one(T), dev(batch(A, B)), X)
        _probe_close(X, batch(LowerTriangular(A) \ B, LowerTriangular(B) \ A))
    end
    vendor_potrf_batched = _probe() do
        X = dev(batch(S, S))
        info = KernelAbstractions.zeros(backend, Int32, 2)
        vendor_potrf_batched!('L', X, info)
        H = Array(X)
        all(iszero, Array(info)) && isapprox(tril(H[:, :, 2]), L; rtol = sqrt(eps(real(T))))
    end
    vendor_getrf_batched = _probe() do
        X = dev(batch(A, A))
        ipiv = KernelAbstractions.zeros(backend, Int32, n, 2)
        info = KernelAbstractions.zeros(backend, Int32, 2)
        vendor_getrf_batched!(X, ipiv, info)
        all(iszero, Array(info)) && _probe_close(X, batch(F.factors, F.factors)) && Array(ipiv)[:, 2] == F.ipiv
    end
    reshape_view_mul = _probe() do
        f, w = 6, 3
        P = _probe_general(T, f, w)
        buf = dev(vcat(zeros(T, 3), vec(P), zeros(T, 2)))
        Pv = reshape(view(buf, 4:(3 + f * w)), f, w)
        C = dev(zeros(T, f, 2))
        mul!(C, Pv, dev(B[1:w, 1:2]))
        _probe_close(C, P * B[1:w, 1:2])
    end
    atomic_add = _probe(() -> _probe_atomic(backend, T))

    return DenseCapabilities(generic_mul, generic_trsm, generic_cholesky, generic_lu,
                             vendor_gemm, vendor_syrk, vendor_herk, vendor_trsm, vendor_potrf, vendor_getrf, vendor_sytrf,
                             vendor_gemm_strided_batched, vendor_trsm_batched, vendor_potrf_batched, vendor_getrf_batched,
                             reshape_view_mul, atomic_add)
end
