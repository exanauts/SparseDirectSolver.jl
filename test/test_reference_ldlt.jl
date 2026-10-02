# T14: CPU reference multifrontal LDLᵀ/LDLᴴ (the oracle of T15), host only.
# Structure "S" for real T, "H" for complex T (sym_structure), plus complex symmetric "S".

ldlt_triangle(A, view) = view == 'L' ? tril(A) : view == 'U' ? triu(A) : A

# analysis + numeric storage + LDLᵀ/LDLᴴ factorization of A given through `view`
function reference_ldlt(A::SparseMatrixCSC{T}; view = 'L', opts = Options(), structure = sym_structure(T)) where {T}
    C = SDS.CSR(ldlt_triangle(A, view))
    S = SDS.symbolic_analysis(C, structure, view; opts)
    N = SDS.allocate_numeric(S, T, CPU())
    info = SDS.ref_ldlt!(N, S, C.nzval; opts)
    return S, N, info, C
end

# A[p, p] − L D Lᴴ (Lᵀ for complex symmetric), and |L| |D| |L|ᴴ
function ldlt_residual(A, S, N; herm = true)
    L, D, p = SDS.extract_ldlt(S, N)
    Lc = herm ? L' : transpose(L)
    return A[p, p] - L * D * Lc, abs.(L) * abs.(D) * abs.(L)'
end

# ‖P A Pᵀ − L D Lᴴ‖_F / ‖A‖_F
ldlt_error(A, S, N; herm = true) = norm(first(ldlt_residual(A, S, N; herm))) / norm(A)

# the same error relative to ‖|L| |D| |L|ᴴ‖_F (backward error of the factorization with its growth)
function ldlt_backward_error(A, S, N; herm = true)
    R, G = ldlt_residual(A, S, N; herm)
    return norm(R) / norm(G)
end

ldlt_factor_tol(::Type{T}) where {T} = real(T) == Float64 ? 1.0e-10 : tol(T)

# (npos, nneg) of the eigenvalues (test/utils.jl), requiring no zero eigenvalue
function eigen_npos_nneg(A)
    npos, nneg, nzero = eigen_inertia(A)
    @test nzero == 0
    return (npos, nneg)
end

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
    nh, nj = 300, 100
    cases = [("random_symindef(400,0.01)", random_symindef(T, 400, 0.01), Options()),
             ("kkt(300,100,1e-2)", kkt_matrix(T, nh, nj, 1.0e-2), Options()),
             ("kkt(300,100,1e-8) interleaved", kkt_matrix(T, nh, nj, 1.0e-8),
              Options(user_perm = kkt_interleaved_perm(nh, nj))),
             ("kkt(300,100,1e-2) interleaved", kkt_matrix(T, nh, nj, 1.0e-2),
              Options(user_perm = kkt_interleaved_perm(nh, nj)))]
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
                @test relres(A, x, b) <= tol(T)
                y = copy(b)
                SDS.ref_solve!(y, S, N, y)
                @test y == x
            end
        end
    end
    # default (AMD) ordering of the δ = 1e-8 KKT matrix: the dual rows are leaves of
    # the assembly tree, factored alone with pivots -δ (no 2×2 partner in the block),
    # so L D Lᴴ has growth ~1/δ; the factorization is still backward stable with
    # respect to |L| |D| |L|ᴴ (and Float32 perturbs those pivots since δ < ε)
    A = kkt_matrix(T, nh, nj, 1.0e-8)
    S, N, info, _ = reference_ldlt(A)
    @test info == 0
    @test ldlt_backward_error(A, S, N) <= ldlt_factor_tol(T)
    @test SDS.inertia(N) == (nh, nj)
end

@testset "inertia equals the eigenvalue signs: $T" for T in ELTYPES
    nh, nj = 200, 100
    cases = [("random_symindef(300,0.02)", random_symindef(T, 300, 0.02), Options()),
             ("random_symindef(250,0.05) natural", random_symindef(T, 250, 0.05), Options(reordering_alg = "algo5")),
             ("kkt(200,100,1e-2)", kkt_matrix(T, nh, nj, 1.0e-2), Options()),
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
        # interleaved ordering: no dual pivot is eliminated before its primal partner;
        # the default ordering is checked for δ = 1e-2 (bounded growth)
        for opts in (Options(pivot_type = pt, user_perm = kkt_interleaved_perm(nh, nj)),
                     Options(pivot_type = pt))
            δ == 1.0e-8 && opts.user_perm === nothing && continue
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

@testset "MadNLP-style inertia correction: $T" for T in ELTYPES
    nh, nj = 200, 100
    # primal regularization δw on an indefinite H, no dual regularization (δ = 0);
    # the KKT-aware ordering keeps every dual row with its primal partner
    A = kkt_matrix(T, nh, nj, 0.0; hessian = :indefinite)
    opts = Options(user_perm = kkt_interleaved_perm(nh, nj))
    C = SDS.CSR(tril(A))
    S = SDS.symbolic_analysis(C, sym_structure(T), 'L'; opts)
    N = SDS.allocate_numeric(S, T)
    dpos = [C.rowptr[i] - 1 + findfirst(==(i), C.colval[C.rowptr[i]:(C.rowptr[i + 1] - 1)]) for i in 1:nh]
    δw, iterations, done = 0.0, 0, false
    nz = similar(C.nzval)
    while iterations < 50
        iterations += 1
        nz .= C.nzval
        nz[dpos] .+= T(δw)
        SDS.ref_ldlt!(N, S, nz; opts)
        st = SDS.pivot_stats(N)
        if SDS.inertia(N) == (nh, nj) && st.nperturbed == 0
            done = true
            break
        end
        δw = δw == 0 ? 1.0e-4 : 8 * δw
    end
    @test done && iterations > 1                  # H is indefinite: δw = 0 is rejected
    @test SDS.npivots(N) == 0
    Aw = A + spdiagm(0 => [fill(T(δw), nh); zeros(T, nj)])
    b = rand(T, nh + nj)
    @test relres(Aw, SDS.ref_solve!(similar(b), S, N, b), b) <= tol(T)
    @test eigen_npos_nneg(Aw) == (nh, nj)
end
