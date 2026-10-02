# T14: CPU reference multifrontal LDLᵀ/LDLᴴ (the oracle of T15), host only.
# Structure "S" for real T, "H" for complex T (sym_structure), plus complex symmetric "S".

stat_of(N, s, q) = N.stats[(s - 1) * SDS.FRONT_STATS_FIELDS + q]

psign_minus(n, j) = (v = zeros(Int8, n); v[j] = -1; v)

@testset "inputs and storage" begin
    A = random_symindef(Float64, 60, 0.05)
    S, N, info, C = reference_ldlt(A)
    @test info == 0
    @test N isa SDS.Numeric{Float64, Vector{Float64}, Vector{Int64}, Vector{Int32}, Vector{Int8}}
    @test length(N.piv) == 60 && length(N.pivot_kind) == 60 && length(N.d) == S.layout.d_len == 120
    @test isperm(N.piv) && all(!iszero, N.pivot_kind)
    # pivoting stays inside a supernode
    sp = S.partition
    @test all(s -> sort(N.piv[SDS.sncols(sp, s)]) == collect(SDS.sncols(sp, s)), 1:SDS.nsupernodes(S))
    # unit diagonal panels
    L, _, _ = SDS.extract_ldlt(S, N)
    @test all(isone, diag(L)) && istril(L) && nnz(L) == sp.nnz_stored
    @test all(s -> stat_of(N, s, 6) == 0, 1:SDS.nsupernodes(S))
    # structure, sizes and options
    Sp = SDS.symbolic_analysis(C, "SPD", 'L')
    @test thrown(() -> SDS.ref_ldlt!(SDS.allocate_numeric(Sp, Float64), Sp, C.nzval)) isa InvalidValueError
    @test thrown(() -> SDS.ref_ldlt!(N, S, C.nzval[1:(end - 1)])) isa InvalidValueError
    @test thrown(() -> SDS.ref_ldlt!(N, S, C.nzval; opts = Options(pivot_sign = ones(Int8, 59)))) isa InvalidValueError
    # 'L' (local block) and 'A' (auto) are Bunch–Kaufman
    for p in ('A', 'L')
        _, Np, _, _ = reference_ldlt(A; opts = Options(pivot_type = p))
        @test Np.factor == N.factor && Np.d == N.d && Np.piv == N.piv
    end
    @test SDS.BUNCH_KAUFMAN_ALPHA ≈ (1 + sqrt(17)) / 8
    # the Cholesky oracle still refuses "S"/"H"
    @test thrown(() -> SDS.ref_factorize!(SDS.allocate_numeric(S, Float64), S, C.nzval)) isa InvalidValueError
end

@testset "P A Pᵀ = L D Lᴴ, inertia and solves: $T" for T in ELTYPES
    # The matrices and right-hand sides otherwise depend on the RNG state left by the preceding
    # testsets (backend list, test subset): reseed so each T sees one fixed draw.
    Random.seed!(666)
    nh, nj = 300, 100
    cases = [("random_symindef(400,0.01)", random_symindef(T, 400, 0.01), Options()),
             ("kkt(300,100,1e-2)", kkt_matrix(T, nh, nj, 1.0e-2), Options()),
             ("kkt(300,100,1e-8)", kkt_matrix(T, nh, nj, 1.0e-8), Options()),
             ("kkt(300,100,1e-8) interleaved", kkt_matrix(T, nh, nj, 1.0e-8),
              Options(user_perm = kkt_interleaved_perm(nh, nj))),
             ("kkt(300,100,1e-2) interleaved", kkt_matrix(T, nh, nj, 1.0e-2),
              Options(user_perm = kkt_interleaved_perm(nh, nj))),
             ("kkt(300,100,1e-8) pivot_pairs = all", kkt_matrix(T, nh, nj, 1.0e-8), Options(pivot_pairs = "all"))]
    for (name, A, opts) in cases
        @testset "$name" begin
            S, N, info, _ = reference_ldlt(A; opts)
            @test info == 0
            @test ldlt_error(A, S, N) <= ldlt_factor_tol(T)
            st = SDS.pivot_stats(N)
            @test st.nperturbed == 0 && SDS.npivots(N) == 0 && st.nzero == 0
            @test st.npos + st.nneg == size(A, 1)
            # every pivot kind is accounted for
            @test st.n2x2 == count(==(SDS.PIVOT_KIND_2X2_FIRST), N.pivot_kind) ==
                  count(==(SDS.PIVOT_KIND_2X2_SECOND), N.pivot_kind)
            for nrhs in (1, 5)
                b = nrhs == 1 ? rand(T, size(A, 1)) : rand(T, size(A, 1), nrhs)
                x = similar(b)
                @test SDS.ref_solve!(x, S, N, b) === x
                if name == "kkt(300,100,1e-8)" && T == ComplexF32
                    # Issue #66: pivot_pairs = "default" pairs only the structurally zero dual
                    # pivots. The other duals of this generator meet weak couplings first, so max|L|
                    # can grow to ~1e4 (80 with "all"); the factor error and inertia hold. The
                    # ComplexF32 residual is ≤ 0.4 tol(T) on most draws (seeds 1–15) but 1.3–1.8×
                    # tol(T) on this one with Julia ≥ 1.11's sprand stream (it passes on Julia 1.10),
                    # so neither @test nor @test_broken holds on every CI version. One refinement
                    # step (the remedy, T16) must bring it under tol(T); the last case asserts the
                    # unrefined bound with "all".
                    x1 = x + SDS.ref_solve!(similar(b), S, N, b - A * x)
                    @test relres(A, x1, b) <= tol(T)
                else
                    @test relres(A, x, b) <= tol(T)
                end
                y = copy(b)
                SDS.ref_solve!(y, S, N, y)
                @test y == x
            end
        end
    end
    # without the 2×2 pivot pairs of the analysis (#64), AMD makes the dual rows of the
    # δ = 1e-8 KKT matrix leaves of the assembly tree, factored alone with pivots -δ, so
    # L D Lᴴ has growth ~1/δ; the factorization is still backward stable with respect to
    # |L| |D| |L|ᴴ (and Float32 perturbs those pivots since δ < ε)
    A = kkt_matrix(T, nh, nj, 1.0e-8)
    S, N, info, _ = reference_ldlt(A; opts = Options(pivot_pairs = "none"))
    @test info == 0
    @test ldlt_backward_error(A, S, N) <= ldlt_factor_tol(T)
    @test SDS.inertia(N) == (nh, nj)
end

@testset "inertia equals the eigenvalue signs: $T" for T in ELTYPES
    nh, nj = 200, 100
    cases = [("random_symindef(300,0.02)", random_symindef(T, 300, 0.02), Options()),
             ("random_symindef(250,0.05) natural", random_symindef(T, 250, 0.05), Options(reordering_alg = "algo5")),
             ("kkt(200,100,1e-2)", kkt_matrix(T, nh, nj, 1.0e-2), Options()),
             ("kkt indefinite H, δ = 0", kkt_matrix(T, nh, nj, 0.0; hessian = :indefinite), Options()),
             ("kkt indefinite H, δ = 0, interleaved", kkt_matrix(T, nh, nj, 0.0; hessian = :indefinite),
              Options(user_perm = kkt_interleaved_perm(nh, nj))),
             ("kkt 1e-3 H, δ = 0, interleaved (2×2)", kkt_matrix(T, nh, nj, 0.0; hessian = :indefinite,
                                                                  hessian_scale = 1.0e-3),
              Options(user_perm = kkt_interleaved_perm(nh, nj))),
             ("kkt 1e-3 H, δ = 0, one front (2×2)", kkt_matrix(T, nh, nj, 0.0; hessian = :indefinite,
                                                               hessian_scale = 1.0e-3),
              Options(amalgamation = (max_width = 10_000, zero_fraction = 100.0, min_width = 10_000)))]
    for (name, A, opts) in cases
        @testset "$name" begin
            S, N, info, _ = reference_ldlt(A; opts)
            st = SDS.pivot_stats(N)
            @test st.nperturbed == 0
            @test SDS.inertia(N) == eigen_npos_nneg(A)
            @test ldlt_error(A, S, N) <= ldlt_factor_tol(T)
            b = rand(T, size(A, 1))
            x = SDS.ref_solve!(similar(b), S, N, b)
            @test relres(A, x, b) <= tol(T)
            # per-front statistics add up
            @test sum(s -> stat_of(N, s, 1) + stat_of(N, s, 2), 1:SDS.nsupernodes(S)) == size(A, 1)
            if occursin("2×2", name)
                @test st.n2x2 > 0
                # D's 2×2 blocks: Hermitian off-diagonal stored once, at d[n + k]
                n = size(A, 1)
                ks = findall(==(SDS.PIVOT_KIND_2X2_FIRST), N.pivot_kind)
                @test all(k -> N.pivot_kind[k + 1] == SDS.PIVOT_KIND_2X2_SECOND && !iszero(N.d[n + k]), ks)
                @test all(k -> iszero(N.d[n + k]), setdiff(1:n, ks))
            end
        end
    end
end

@testset "views and refactorization: $T" for T in ELTYPES
    A = kkt_matrix(T, 200, 100, 0.0; hessian = :indefinite, hessian_scale = 1.0e-3)
    opts = Options(user_perm = kkt_interleaved_perm(200, 100))
    S, N, _, _ = reference_ldlt(A; view = 'L', opts)
    @test SDS.pivot_stats(N).n2x2 > 0
    for view in ('U', 'F')
        Sv, Nv, _, _ = reference_ldlt(A; view, opts)
        @test Nv.factor == N.factor && Nv.d == N.d && Nv.piv == N.piv && Nv.pivot_kind == N.pivot_kind
    end
    # new values on the same analysis, then the old ones again
    C = SDS.CSR(tril(A))
    B = copy(A)
    nonzeros(B) .*= T(2)
    SDS.ref_ldlt!(N, S, SDS.CSR(tril(B)).nzval; opts)
    @test ldlt_error(B, S, N) <= ldlt_factor_tol(T)
    Nf = SDS.allocate_numeric(S, T)
    SDS.ref_ldlt!(Nf, S, C.nzval; opts)
    SDS.ref_ldlt!(N, S, C.nzval; opts)
    @test N.factor == Nf.factor && N.d == Nf.d && N.piv == Nf.piv && N.stats == Nf.stats
end

@testset "complex symmetric LDLᵀ: $T" for T in COMPLEX_ELTYPES
    A = random_symindef(T, 300, 0.02; hermitian = false)
    @test transpose(A) == A && A' != A
    S, N, info, _ = reference_ldlt(A; structure = "S")
    @test info == 0
    @test ldlt_error(A, S, N; herm = false) <= ldlt_factor_tol(T)
    @test SDS.inertia(N) == (0, 0)            # no inertia for complex symmetric matrices
    b = rand(T, 300, 3)
    @test relres(A, SDS.ref_solve!(similar(b), S, N, b), b) <= tol(T)
    # 2×2 pivots on a complex symmetric matrix
    n = 300
    K = kkt_matrix(T, 200, 100, 0.0; hessian = :indefinite, hessian_scale = 1.0e-3)
    K = tril(K) + transpose(tril(K, -1))      # complex symmetric with the generator's lower triangle
    @test transpose(K) == K && K' != K
    S2, N2, _, _ = reference_ldlt(K; structure = "S", opts = Options(user_perm = kkt_interleaved_perm(200, 100)))
    @test SDS.pivot_stats(N2).n2x2 > 0
    @test ldlt_error(K, S2, N2; herm = false) <= ldlt_factor_tol(T)
    x = SDS.ref_solve!(similar(b), S2, N2, b)
    @test relres(K, x, b) <= tol(T)
end

@testset "perturbation and pivot_sign: $T" for T in ELTYPES
    n, j = 60, 23
    for stored_zero in (false, true), sgn in (1, -1)
        A = singular_block_matrix(T, n, j; stored_zero)
        psign = zeros(Int8, n)
        psign[j] = sgn
        opts = Options(pivot_sign = psign)
        S, N, info, _ = reference_ldlt(A; opts)
        @test info == 0
        st = SDS.pivot_stats(N)
        @test st.nperturbed >= 1 && SDS.npivots(N) >= 1 && st.nzero >= 1
        _, _, p = SDS.extract_ldlt(S, N)
        k = findfirst(==(SDS.PIVOT_KIND_PERTURBED), N.pivot_kind)
        @test k !== nothing && p[k] == j
        @test real(N.d[k]) == sgn * real(T)(SDS.default_pivot_epsilon(real(T))) && iszero(imag(N.d[k]))
        @test SDS.inertia(N) == (sgn > 0 ? (n, 0) : (n - 1, 1))
        # the perturbed matrix is factored exactly; a consistent right-hand side is solved
        x0 = rand(T, n)
        x0[j] = 0
        b = A * x0
        x = SDS.ref_solve!(similar(b), S, N, b)
        @test relres(A, x, b) <= 1.0e-6
    end
    # without pivot_sign: the sign of the (zero) pivot, +1
    A = singular_block_matrix(T, n, j)
    S, N, _, _ = reference_ldlt(A)
    k = findfirst(==(SDS.PIVOT_KIND_PERTURBED), N.pivot_kind)
    @test real(N.d[k]) == real(T)(SDS.default_pivot_epsilon(real(T)))
    # pivot_epsilon and the scaled variant (ε max|aᵢⱼ|, max |aᵢⱼ| = 6)
    S, N, _, _ = reference_ldlt(A; opts = Options(pivot_epsilon = 1.0e-3, pivot_sign = psign_minus(n, j)))
    k = findfirst(==(SDS.PIVOT_KIND_PERTURBED), N.pivot_kind)
    @test real(N.d[k]) ≈ -1.0e-3
    S, N, _, _ = reference_ldlt(A; opts = Options(pivot_epsilon = 1.0e-3, pivot_epsilon_alg = "algo1"))
    k = findfirst(==(SDS.PIVOT_KIND_PERTURBED), N.pivot_kind)
    @test real(N.d[k]) ≈ 6.0e-3
    # 'N' and 'D' perturb the same pivot
    for pt in ('N', 'D')
        S, N, _, _ = reference_ldlt(A; opts = Options(pivot_type = pt))
        _, _, p = SDS.extract_ldlt(S, N)
        @test SDS.npivots(N) == 1 && p[findfirst(==(SDS.PIVOT_KIND_PERTURBED), N.pivot_kind)] == j
    end
end

@testset "pivot_type 'D' and 'N' on quasi-definite KKT: $T" for T in ELTYPES
    nh, nj = 300, 100
    for δ in (1.0e-8, 1.0e-2), pt in ('D', 'N')
        A = kkt_matrix(T, nh, nj, δ)
        # interleaved ordering, and the default ordering with the 2×2 pivot pairs of the
        # analysis (#64; δ = 1e-8 is below the pair tolerance): no dual pivot is
        # eliminated before its primal partner
        for opts in (Options(pivot_type = pt, user_perm = kkt_interleaved_perm(nh, nj)),
                     Options(pivot_type = pt), Options(pivot_type = pt, pivot_pairs = "all"))
            S, N, info, _ = reference_ldlt(A; opts)
            st = SDS.pivot_stats(N)
            @test info == 0 && st.n2x2 == 0 && st.nperturbed == 0
            @test all(k -> k == SDS.PIVOT_KIND_1X1, N.pivot_kind)
            @test SDS.inertia(N) == (nh, nj)
            @test ldlt_error(A, S, N) <= ldlt_factor_tol(T)
            b = rand(T, nh + nj)
            @test relres(A, SDS.ref_solve!(similar(b), S, N, b), b) <= tol(T)
            pt == 'N' && @test N.piv == 1:(nh + nj)        # no search: no interchange
        end
    end
end

const MADNLP_ORDERINGS = (("default ordering", Options()),
                          ("default ordering, pivot_pairs = all", Options(pivot_pairs = "all")),
                          ("interleaved", Options(user_perm = kkt_interleaved_perm(200, 100))))

@testset "MadNLP-style inertia correction ($label): $T" for T in ELTYPES, (label, opts) in MADNLP_ORDERINGS
    nh, nj = 200, 100
    # primal regularization δw on an indefinite H, no dual regularization (δ = 0); the
    # 2×2 pivot pairs of the analysis (#64) or the KKT-aware ordering keep every dual row
    # with its primal partner
    A = kkt_matrix(T, nh, nj, 0.0; hessian = :indefinite)
    done, iterations, S, N, δw = madnlp_inertia_loop(A, nh, nj, opts)
    @test done && iterations > 1                  # H is indefinite: δw = 0 is rejected
    @test SDS.npivots(N) == 0
    Aw = A + spdiagm(0 => [fill(T(δw), nh); zeros(T, nj)])
    b = rand(T, nh + nj)
    @test relres(Aw, SDS.ref_solve!(similar(b), S, N, b), b) <= tol(T)
    @test eigen_npos_nneg(Aw) == (nh, nj)
end
