# T03: dense-op interface, capability audit, KA fallbacks.
#
# Every op is checked for every impl the capability table lists on the backend
# (`dense_impls`, always including :ka), on plain device matrices and on panel
# views `reshape(view(buf, a:b), f, w)` of a flat buffer.

dense_op(X, t::Char) = t == 'N' ? X : t == 'T' ? transpose(X) : adjoint(X)
dense_tri(A, uplo::Char, diag::Char) =
    uplo == 'L' ? (diag == 'U' ? UnitLowerTriangular(A) : LowerTriangular(A)) :
    (diag == 'U' ? UnitUpperTriangular(A) : UpperTriangular(A))
dense_part(C, uplo::Char) = uplo == 'L' ? tril(C) : triu(C)

@testset "capability audit ($(backend_name(backend)))" for backend in BACKENDS
    SDS.print_capabilities(stdout, backend)
    table = sprint(io -> SDS.print_capabilities(io, backend))
    @test occursin("generic_mul", table) && occursin("ComplexF64", table)
    for T in ELTYPES
        caps = SDS.capabilities(backend, T)
        @test caps isa SDS.DenseCapabilities
        @test SDS.capabilities(backend, T) === caps   # cached
        @test caps.generic_mul
        if backend_name(backend) == "CUDA"
            @test caps.vendor_syrk
            @test caps.vendor_gemm_strided_batched
            @test caps.vendor_potrf
        end
        for op in keys(SDS.DENSE_OPS)
            impls = SDS.dense_impls(op, backend, T)
            @test last(impls) === :ka
        end
    end
end

@testset "dense ops ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    tl = dense_tol(T)
    α, β = T(0.75), T(-1.25)
    for (m, n, k) in DENSE_SIZES, panel in (false, true)
        dev(X) = panel ? to_panel(backend, X) : to_device(backend, X)

        @testset "gemm $m×$n×$k panel=$panel" begin
            for (tA, tB) in (('N', 'N'), ('T', 'N'), ('N', 'C'), ('C', 'T'))
                A = randn(T, tA == 'N' ? (m, k) : (k, m))
                B = randn(T, tB == 'N' ? (k, n) : (n, k))
                C0 = randn(T, m, n)
                ref = α * dense_op(A, tA) * dense_op(B, tB) + β * C0
                bound = tl * (abs(α) * norm(A) * norm(B) + abs(β) * norm(C0))
                for impl in SDS.dense_impls(:gemm, backend, T)
                    C = dev(C0)
                    @test SDS.gemm!(C, dev(A), dev(B), α, β; transA = tA, transB = tB, impl) === C
                    @test norm(to_host(C) - ref) <= bound
                end
            end
        end

        @testset "syrk/herk $m×$k panel=$panel" begin
            A = randn(T, m, k)
            C0 = Matrix{T}(Hermitian(randn(T, m, m)))
            for uplo in ('L', 'U')
                ref = α * A * transpose(A) + β * C0
                bound = tl * (abs(α) * norm(A)^2 + abs(β) * norm(C0))
                for impl in SDS.dense_impls(:syrk, backend, T)
                    C = dev(C0)
                    SDS.syrk!(C, dev(A), α, β; uplo, impl)
                    Ch = to_host(C)
                    @test norm(dense_part(Ch, uplo) - dense_part(ref, uplo)) <= bound
                    # :vendor and :ka leave the strict other triangle untouched
                    impl === :generic || @test (uplo == 'L' ? triu(Ch, 1) == triu(C0, 1) : tril(Ch, -1) == tril(C0, -1))
                end
                a, b = real(T)(0.75), real(T)(-1.25)
                ref = a * A * A' + b * C0
                for impl in SDS.dense_impls(:herk, backend, T)
                    C = dev(C0)
                    SDS.herk!(C, dev(A), a, b; uplo, impl)
                    @test norm(dense_part(to_host(C), uplo) - dense_part(ref, uplo)) <= bound
                end
            end
        end

        @testset "trsm $m×$n panel=$panel" begin
            B = randn(T, m, n)
            for side in ('L', 'R'), uplo in ('L', 'U'), trans in ('N', 'T', 'C'), diag in ('N', 'U')
                na = side == 'L' ? m : n
                A = dense_triangular(T, na; uplo = Symbol(uplo), unit = diag == 'U')
                M = dense_op(dense_tri(A, uplo, diag), trans)
                ref = side == 'L' ? α * (M \ B) : α * (B / M)
                for impl in SDS.dense_impls(:trsm, backend, T)
                    X = dev(B)
                    @test SDS.trsm!(side, uplo, trans, diag, α, dev(A), X; impl) === X
                    @test norm(to_host(X) - ref) <= tl * norm(ref)
                end
            end
        end

        @testset "potrf $m panel=$panel" begin
            S = dense_hpd(T, m)
            for uplo in ('L', 'U'), impl in SDS.dense_impls(:potrf, backend, T)
                X = dev(S)
                @test SDS.potrf!(uplo, X; impl) == 0
                F = uplo == 'L' ? LowerTriangular(to_host(X)) : UpperTriangular(to_host(X))'
                @test norm(F * F' - S) <= tl * norm(F)^2
            end
            # not positive definite at column j: info == j
            for j in unique((1, cld(m, 2), m)), uplo in ('L', 'U'), impl in SDS.dense_impls(:potrf, backend, T)
                Sj = copy(S)
                Sj[j, j] = -one(T)
                @test SDS.potrf!(uplo, dev(Sj); impl) == j
            end
        end

        @testset "getrf/laswp $m×$n panel=$panel" begin
            for (r, c) in unique(((m, n), (m, m)))
                A = randn(T, r, c)
                kmin = min(r, c)
                for impl in SDS.dense_impls(:getrf, backend, T), IP in (Int32, Int64)
                    X = dev(A)
                    ipiv = to_device(backend, zeros(IP, kmin))
                    @test SDS.getrf!(X, ipiv; impl) == 0
                    F = to_host(X)
                    L = tril(F[:, 1:kmin], -1) + Matrix{T}(I, r, kmin)
                    U = triu(F[1:kmin, :])
                    p = to_host(ipiv)
                    @test all(i -> i <= p[i] <= r, 1:kmin)
                    perm = ipiv_permutation(p, r)
                    @test norm(L * U - A[perm, :]) <= tl * norm(L) * norm(U)
                    # laswp: P A on the device, and back with reverse = true
                    Y = dev(A)
                    @test SDS.laswp!(Y, ipiv) === Y
                    @test to_host(Y) == A[perm, :]
                    SDS.laswp!(Y, ipiv; reverse = true)
                    @test to_host(Y) == A
                    # oversized pivot buffer: only the first npiv = min(m, n) entries are applied
                    X = dev(A)
                    ipiv2 = to_device(backend, zeros(IP, kmin + 3))
                    @test SDS.getrf!(X, ipiv2; impl) == 0
                    Y = dev(A)
                    SDS.laswp!(Y, ipiv2; npiv = kmin)
                    @test to_host(Y) == A[perm, :]
                end
            end
        end
    end

    @testset "getrf zero pivot" begin
        A = randn(T, 6, 6)
        A[:, 3] .= 0
        for impl in SDS.dense_impls(:getrf, backend, T)
            ipiv = to_device(backend, zeros(Int32, 6))
            @test SDS.getrf!(to_device(backend, A), ipiv; impl) == 3
        end
    end
end

@testset "strided batched ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    tl = dense_tol(T)
    α, β = T(0.5), T(2)
    for (m, n, k) in ((1, 1, 1), (7, 5, 3), (32, 32, 32)), count in (1, 4, 33), padded in (false, true)
        # device 3-D batch: contiguous, or members `pad·m` entries apart in a flat buffer
        function devb(X::Array{<:Any, 3})
            padded || return to_device(backend, X)
            r, c, nb = size(X)
            stride = r * (c + 2)
            buf = to_device(backend, zeros(T, 5 + stride * nb))
            Y = SDS.strided_batch(buf, 5, r, c, stride, nb)
            for i in 1:nb
                copyto!(view(buf, (5 + (i - 1) * stride + 1):(5 + (i - 1) * stride + r * c)), to_device(backend, vec(X[:, :, i])))
            end
            @test to_host(Y) == X
            return Y
        end
        @testset "gemm $m×$n×$k count=$count padded=$padded" begin
            for (tA, tB) in (('N', 'N'), ('T', 'C'))
                A = randn(T, (tA == 'N' ? (m, k) : (k, m))..., count)
                B = randn(T, (tB == 'N' ? (k, n) : (n, k))..., count)
                C0 = randn(T, m, n, count)
                for impl in SDS.dense_impls(:gemm_strided_batched, backend, T)
                    C = devb(C0)
                    @test SDS.gemm_strided_batched!(C, devb(A), devb(B), α, β; transA = tA, transB = tB, impl) === C
                    Ch = to_host(C)
                    for i in 1:count
                        Ai, Bi = A[:, :, i], B[:, :, i]
                        ref = α * dense_op(Ai, tA) * dense_op(Bi, tB) + β * C0[:, :, i]
                        @test norm(Ch[:, :, i] - ref) <= tl * (abs(α) * norm(Ai) * norm(Bi) + abs(β) * norm(C0[:, :, i]))
                    end
                end
            end
        end
        @testset "trsm $m×$n count=$count padded=$padded" begin
            for (side, uplo, trans, diag) in (('L', 'L', 'N', 'N'), ('R', 'U', 'C', 'N'), ('L', 'U', 'T', 'U'), ('R', 'L', 'N', 'U'))
                na = side == 'L' ? m : n
                A = cat((dense_triangular(T, na; uplo = Symbol(uplo), unit = diag == 'U') for _ in 1:count)...; dims = 3)
                B = randn(T, m, n, count)
                for impl in SDS.dense_impls(:trsm_strided_batched, backend, T)
                    X = devb(B)
                    @test SDS.trsm_strided_batched!(side, uplo, trans, diag, α, devb(A), X; impl) === X
                    Xh = to_host(X)
                    for i in 1:count
                        M = dense_op(dense_tri(A[:, :, i], uplo, diag), trans)
                        ref = side == 'L' ? α * (M \ B[:, :, i]) : α * (B[:, :, i] / M)
                        @test norm(Xh[:, :, i] - ref) <= tl * norm(ref)
                    end
                end
            end
        end
    end
end

@testset "KA batched factorizations and tiles ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    nb, n = 3, 20
    S = cat((dense_hpd(T, n) for _ in 1:nb)...; dims = 3)
    X = to_device(backend, S)
    info = to_device(backend, zeros(Int32, nb))
    SDS.ka_potrf!('L', X, info)
    @test to_host(info) == zeros(Int32, nb)
    @test to_host(SDS.ka_chol_diag_info!(to_device(backend, ones(Int32, nb)), X)) == zeros(Int32, nb)
    Xh = to_host(X)
    for i in 1:nb
        L = LowerTriangular(Xh[:, :, i])
        @test norm(L * L' - S[:, :, i]) <= dense_tol(T) * norm(L)^2
    end
    # diagonal pivot check of a factor: first non-positive / non-finite pivot per member
    D = copy(Xh)
    D[7, 7, 1] = -one(T)
    D[n, n, 3] = T(NaN)
    @test to_host(SDS.ka_chol_diag_info!(to_device(backend, zeros(Int32, nb)), to_device(backend, D))) == Int32[7, 0, n]
    A = randn(T, n, n, nb)
    X = to_device(backend, A)
    ipiv = to_device(backend, zeros(Int32, n, nb))
    SDS.ka_getrf!(X, ipiv, info)
    @test to_host(info) == zeros(Int32, nb)
    Y = to_device(backend, A)
    SDS.ka_laswp!(Y, ipiv)
    Xh, Yh = to_host(X), to_host(Y)
    for i in 1:nb
        L = UnitLowerTriangular(Xh[:, :, i])
        U = UpperTriangular(Xh[:, :, i])
        @test norm(L * U - Yh[:, :, i]) <= dense_tol(T) * norm(L) * norm(U)
    end
    # both tile sizes of the KA gemm
    A, B = randn(T, 70, 40), randn(T, 40, 66)
    for tile in (Val(16), Val(32))
        C = to_device(backend, zeros(T, 70, 66))
        SDS.ka_gemm!(C, to_device(backend, A), to_device(backend, B), 1, 0; tile)
        @test norm(to_host(C) - A * B) <= dense_tol(T) * norm(A) * norm(B)
    end
end

@testset "impl = :auto ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    A = to_device(backend, randn(T, 5, 5))
    for op in (:gemm, :potrf, :laswp, :trsm_strided_batched)
        @test SDS.select_impl(op, A, :auto) === first(SDS.dense_impls(op, backend, T))
    end
    @test SDS.select_impl(:laswp, A, :auto) === :ka
    C = to_device(backend, zeros(T, 5, 5))
    SDS.gemm!(C, A, A)
    @test to_host(C) ≈ to_host(A) * to_host(A)
end

@testset "dense interface errors" begin
    A = zeros(4, 4)
    @test_throws InvalidValueError SDS.gemm!(A, A, A; impl = :bogus)
    @test_throws InvalidValueError SDS.gemm!(A, A, A; transA = :X)
    @test_throws InvalidValueError SDS.trsm!(:L, :L, :N, :Q, 1, A, A)
    @test_throws NotSupportedError SDS.laswp!(A, [1, 2]; impl = :vendor)
    @test_throws DimensionMismatch SDS.laswp!(A, [1, 2]; npiv = 3)
    # host BLAS accepts strided views only; a non-strided view is not a vendor operand
    @test_throws NotSupportedError SDS.gemm!(A, view(A, [1, 3, 2, 4], :), A; impl = :vendor)
    @test SDS.gemm!(zeros(4, 4), view(A, [1, 3, 2, 4], :), A; impl = :generic) == zeros(4, 4)
    @test_throws NotSupportedError SDS.gemm_strided_batched!(zeros(2, 2, 1), zeros(2, 2, 1), zeros(2, 2, 1); impl = :generic)
    @test_throws NotSupportedError SDS.vendor_potrf_batched!('L', zeros(2, 2, 1), zeros(Int32, 1))
    @test_throws DimensionMismatch SDS.gemm!(zeros(3, 4), A, A)
    @test_throws DimensionMismatch SDS.trsm!(:R, :L, :N, :N, 1, A, zeros(4, 3))
    @test_throws InvalidValueError SDS.dense_impls(:nope, CPU(), Float64)
    buf = collect(1.0:40.0)
    Bt = SDS.strided_batch(buf, 2, 3, 2, 9, 4)
    @test size(Bt) == (3, 2, 4)
    @test Bt[:, :, 2] == reshape(buf[(2 + 9 + 1):(2 + 9 + 6)], 3, 2)
    @test SDS.strided_batch(buf, 0, 2, 3, 6, 5) == reshape(buf[1:30], 2, 3, 5)
    @test_throws InvalidValueError SDS.strided_batch(buf, 0, 3, 2, 5, 2)    # stride < m n
    @test_throws InvalidValueError SDS.strided_batch(buf, 0, 3, 2, 8, 2)    # not a multiple of m
    @test_throws InvalidValueError SDS.strided_batch(buf, 10, 3, 2, 9, 4)   # beyond the buffer
end

# T09: the asynchronous Cholesky of the numeric phase writes its status into a
# slot of a device vector; the matrix is the diagonal block of a panel view.
@testset "potrf_info! ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    tl = dense_tol(T)
    for m in (1, 7, 32, 100)
        S = dense_hpd(T, m)
        f = m + 5
        for uplo in ('L', 'U'), impl in SDS.dense_impls(:potrf, backend, T)
            P = to_panel(backend, vcat(S, randn(T, f - m, m)))
            F11 = view(P, 1:m, 1:m)
            info = to_device(backend, fill(Int32(7), 4))
            @test SDS.potrf_info!(uplo, F11, info, 3; impl) === info
            @test to_host(info) == Int32[7, 7, 0, 7]
            Fh = to_host(F11)
            F = uplo == 'L' ? LowerTriangular(Fh) : UpperTriangular(Fh)'
            @test norm(F * F' - S) <= tl * norm(F)^2
            for j in unique((1, cld(m, 2), m))
                Sj = copy(S)
                Sj[j, j] = -one(T)
                X = to_device(backend, Sj)
                SDS.potrf_info!(uplo, X, info, 2; impl)
                @test to_host(info) == Int32[7, j, 0, 7]
            end
        end
    end
    @test_throws DimensionMismatch SDS.potrf_info!('L', to_device(backend, dense_hpd(T, 3)),
                                                   to_device(backend, zeros(Int32, 2)), 3)
    # ka_potrf! with an offset into the status vector
    X = to_device(backend, cat(dense_hpd(T, 5), dense_hpd(T, 5); dims = 3))
    info = to_device(backend, fill(Int32(7), 4))
    SDS.ka_potrf!('L', X, info; offset = 2)
    @test to_host(info) == Int32[7, 7, 0, 0]
    @test_throws DimensionMismatch SDS.ka_potrf!('L', X, info; offset = 3)
end
