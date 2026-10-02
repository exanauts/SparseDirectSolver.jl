# #64: 2×2 pivot candidate pairs in the analysis of "S"/"H" matrices (host only).
# A row with a zero or negligible diagonal is ordered right after a partner and
# shares its supernode, so in-front Bunch–Kaufman can pair them.

# pattern, pairs and ordering of the lower triangle of A as the analysis computes them
function pairs_ordering(A::SparseMatrixCSC{T}; opts = Options(), structure = sym_structure(T), view = 'L') where {T}
    C = SDS.CSR(triangle_view(A, view))
    P = SDS.SymmetricPattern(C, structure; view)
    pairs = SDS.pairs_enabled(structure, opts) ? SDS.pivot_pairs(P, C, structure; view) : Tuple{Int, Int}[]
    return P, SDS.compute_ordering(P, opts; T, pairs), C
end

# both columns of every pair in one supernode, partner right before its candidate
pairs_in_supernodes(sp, pairs) =
    all(((a, b),) -> sp.iperm[b] == sp.iperm[a] + 1 && sp.col2sn[sp.iperm[a]] == sp.col2sn[sp.iperm[b]], pairs)

same_partition(S, R) = S.partition.perm == R.partition.perm && S.partition.super_ptr == R.partition.super_ptr &&
                       S.partition.rowval == R.partition.rowval && S.partition.nnz_L == R.partition.nnz_L

@testset "candidates and greedy pairing" begin
    # lower triangle, stored zeros kept: diagonal 1 stored zero, 2 absent, 3 tiny (≤ 1e-6 max|a₃ⱼ|),
    # 4–7 regular, 8 stored zero without any off-diagonal entry
    n = 8
    I = [1, 3, 4, 5, 6, 7, 8, 2, 3, 3, 4, 5, 6, 7, 5, 6]
    J = [1, 3, 4, 5, 6, 7, 8, 1, 1, 2, 1, 1, 2, 3, 4, 4]
    V = [0.0, 1.0e-9, 4.0, 4.0, 4.0, 4.0, 0.0, 5.0, 7.0, 2.0, 3.0, 3.0, 1.0, 9.0, 1.0, 1.0]
    L = sparse(I, J, V, n, n)
    @test nnz(L) == length(I)
    C = SDS.CSR(L)
    P = SDS.SymmetricPattern(C, "S"; view = 'L')
    pairs = SDS.pivot_pairs(P, C, "S"; view = 'L')
    # candidates 1, 2, 3, 8; free non-candidate neighbours 1 → {4, 5}, 2 → {6}, 3 → {7}, 8 → {}.
    # 8 stays alone; 2 takes 6, 3 takes 7; 1 sees |a₁₄| = |a₁₅| and takes 5 (degree 2 < 3)
    @test pairs == [(5, 1), (6, 2), (7, 3)]
    U = copy(transpose(L))
    CU = SDS.CSR(U)
    @test SDS.pivot_pairs(SDS.SymmetricPattern(CU, "S"; view = 'U'), CU, "S"; view = 'U') == pairs
    # a non-candidate is matched at most once: 1 and 2 both have only 3; 1 comes first (index)
    B = sparse([1, 2, 3, 3, 3], [1, 2, 3, 1, 2], [0.0, 0.0, 5.0, 1.0, 2.0], 3, 3)
    CB = SDS.CSR(B)
    @test SDS.pivot_pairs(SDS.SymmetricPattern(CB, "S"; view = 'L'), CB, "S"; view = 'L') == [(3, 1)]
    # compressed graph and the padded pattern
    Pc, ptr, members = SDS.compressed_pattern(P, pairs)
    @test Pc.n == n - length(pairs) && sort(members) == 1:n
    groups2 = [members[ptr[g]:(ptr[g + 1] - 1)] for g in 1:Pc.n if ptr[g + 1] - ptr[g] == 2]
    @test sort(groups2) == sort([[a, b] for (a, b) in pairs])
    Q = SDS.pair_pattern(P, pairs)
    for (a, b) in pairs
        @test b in SDS.neighbors(Q, a)
        @test setdiff(SDS.neighbors(Q, a), [b]) == setdiff(SDS.neighbors(Q, b), [a])
        @test issubset(union(SDS.neighbors(P, a), SDS.neighbors(P, b)), union(SDS.neighbors(Q, a), [a]))
    end
    @test all(j -> issubset(SDS.neighbors(P, j), SDS.neighbors(Q, j)), 1:n)
    @test SDS.pair_pattern(P, Tuple{Int, Int}[]) === P
    @test thrown(() -> SDS.compressed_pattern(P, [(1, 2), (2, 3)])) isa InvalidValueError
    @test thrown(() -> SDS.compressed_pattern(P, [(1, 1)])) isa InvalidValueError
    # the ordering expands every pair partner-first
    ord = SDS.compute_ordering(P, Options(); pairs)
    @test ord.pairs == pairs && isperm(ord.perm)
    @test all(((a, b),) -> ord.iperm[b] == ord.iperm[a] + 1, pairs)
end

@testset "KKT matrices, default ordering: $T" for T in ELTYPES
    cases = [("kkt(300,100,1e-8)", 300, 100, kkt_matrix(T, 300, 100, 1.0e-8)),
             ("kkt(200,100,0) indefinite H", 200, 100, kkt_matrix(T, 200, 100, 0.0; hessian = :indefinite)),
             ("kkt(200,100,0) indefinite 1e-3 H", 200, 100,
              kkt_matrix(T, 200, 100, 0.0; hessian = :indefinite, hessian_scale = 1.0e-3))]
    for (name, nh, nj, A) in cases
        @testset "$name" begin
            P, ord, C = pairs_ordering(A)
            # every dual row is a candidate, paired with the primal of its dominant J entry
            @test ord.pairs == [(i, nh + i) for i in 1:nj]
            @test all(((a, b),) -> ord.iperm[b] == ord.iperm[a] + 1, ord.pairs)
            # the analysis keeps each pair in one supernode, also without amalgamation
            S = SDS.symbolic_analysis(C, sym_structure(T), 'L')
            @test pairs_in_supernodes(S.partition, ord.pairs)
            @test S.partition.nnz_L == ord.stats.nnz_L
            S0 = SDS.symbolic_analysis(C, sym_structure(T), 'L'; opts = Options(use_superpanels = 0))
            @test pairs_in_supernodes(S0.partition, ord.pairs)
            @test all(s -> SDS.snwidth(S0.partition, s) >= 2,
                      unique(S0.partition.col2sn[S0.partition.iperm[b]] for (_, b) in ord.pairs))
            # the factorization: no perturbation, no growth, exact inertia
            S, N, info, _ = reference_ldlt(A)
            st = SDS.pivot_stats(N)
            @test info == 0
            @test st.nperturbed == 0 && st.nzero == 0 && SDS.npivots(N) == 0
            @test ldlt_error(A, S, N) <= ldlt_factor_tol(T)
            @test SDS.inertia(N) == (occursin("1e-8", name) ? (nh, nj) : eigen_npos_nneg(A))
            # Bunch–Kaufman takes 2×2 pivots where 1×1 pivots on the primal partners are not stable
            occursin("1e-3", name) && @test st.n2x2 >= nj ÷ 2
            b = rand(T, nh + nj, 2)
            @test relres(A, SDS.ref_solve!(similar(b), S, N, b), b) <= tol(T)
            # without pairs the dual pivots are factored alone (growth 1/δ, or perturbed zeros)
            Sn, Nn, _, _ = reference_ldlt(A; opts = Options(pivot_pairs = "none"))
            @test ldlt_error(A, Sn, Nn) > ldlt_factor_tol(T) || SDS.pivot_stats(Nn).nperturbed > 0
        end
    end
end

@testset "MadNLP-style inertia correction, default ordering: $T" for T in ELTYPES
    nh, nj = 200, 100
    A = kkt_matrix(T, nh, nj, 0.0; hessian = :indefinite)
    done, iterations, S, N, δw = madnlp_inertia_loop(A, nh, nj, Options())
    @test done && iterations > 1
    @test SDS.npivots(N) == 0
    Aw = A + spdiagm(0 => [fill(T(δw), nh); zeros(T, nj)])
    b = rand(T, nh + nj)
    @test relres(Aw, SDS.ref_solve!(similar(b), S, N, b), b) <= tol(T)
    @test eigen_npos_nneg(Aw) == (nh, nj)
    # without pairs the zero dual pivots are always perturbed: the loop never ends
    @test !first(madnlp_inertia_loop(A, nh, nj, Options(pivot_pairs = "none")))
end

@testset "fill: $T" for T in ELTYPES
    # The constraint "dual next to its partner" costs fill on these KKT generators: under plain AMD a
    # dual row is a cheap leaf. Measured nnz_L(pairs)/nnz_L(none): 1.73–1.76 (δ = 1e-8), 2.02–2.15
    # (δ = 0); the T14 interleaved user_perm is worse. The 1.15 bound of #64's request does not hold.
    for (nh, nj, A) in ((300, 100, kkt_matrix(T, 300, 100, 1.0e-8)),
                        (200, 100, kkt_matrix(T, 200, 100, 0.0; hessian = :indefinite)))
        _, ord, _ = pairs_ordering(A)
        _, none, _ = pairs_ordering(A; opts = Options(pivot_pairs = "none"))
        _, inter, _ = pairs_ordering(A; opts = Options(user_perm = kkt_interleaved_perm(nh, nj)))
        @test isempty(none.pairs) && isempty(inter.pairs) && length(ord.pairs) == nj
        @test ord.stats.nnz_L <= 2.5 * none.stats.nnz_L
        @test ord.stats.nnz_L <= inter.stats.nnz_L
    end
    # no candidate (Gershgorin-dominant diagonal): no pair, bitwise the same analysis
    A = random_symindef(T, 400, 0.01)
    P, ord, C = pairs_ordering(A)
    _, none, _ = pairs_ordering(A; opts = Options(pivot_pairs = "none"))
    @test isempty(ord.pairs) && ord.perm == none.perm && ord.stats == none.stats
    @test same_partition(SDS.symbolic_analysis(C, sym_structure(T), 'L'),
                         SDS.symbolic_analysis(C, sym_structure(T), 'L'; opts = Options(pivot_pairs = "none")))
end

@testset "structures without pairs, user_perm, natural ordering: $T" for T in ELTYPES
    nh, nj = 200, 100
    A = kkt_matrix(T, nh, nj, 0.0; hessian = :indefinite)
    C = SDS.CSR(tril(A))
    # "SPD"/"HPD" (pattern only) and "G": never paired, bitwise the analysis with pivot_pairs = "none"
    for structure in (spd_structure(T), "G")
        @test !SDS.pairs_enabled(structure, Options())
        CG = structure == "G" ? SDS.CSR(A) : C
        view = structure == "G" ? 'F' : 'L'
        P = SDS.SymmetricPattern(CG, structure; view)
        @test SDS.compute_ordering(P, Options()).perm == SDS.compute_ordering(P, Options(pivot_pairs = "none")).perm
        structure == "G" && continue
        @test same_partition(SDS.symbolic_analysis(C, structure, 'L'),
                             SDS.symbolic_analysis(C, structure, 'L'; opts = Options(pivot_pairs = "none")))
    end
    # user_perm: the user owns the order, no pair; natural ordering: no pair either
    for opts in (Options(user_perm = kkt_interleaved_perm(nh, nj)), Options(reordering_alg = "algo5"))
        @test !SDS.pairs_enabled(sym_structure(T), opts)
        P, ord, _ = pairs_ordering(A; opts)
        @test isempty(ord.pairs)
        @test isempty(SDS.compute_ordering(P, opts; pairs = [(1, nh + 1)]).pairs)
        opts.user_perm === nothing || @test ord.perm == opts.user_perm
        @test same_partition(SDS.symbolic_analysis(C, sym_structure(T), 'L'; opts),
                             SDS.symbolic_analysis(C, sym_structure(T), 'L';
                                                   opts = Options(pivot_pairs = "none", user_perm = opts.user_perm,
                                                                  reordering_alg = opts.reordering_alg)))
    end
    # nested dissection (Metis is loaded by the test driver) and MMD also order the compressed graph
    P, _, _ = pairs_ordering(A)
    pairs = SDS.pivot_pairs(P, C, sym_structure(T); view = 'L')
    for alg in (:nd, :mmd, :auto)
        ord = SDS.compute_ordering(P, Options(); alg, pairs)
        @test ord.pairs == pairs && all(((a, b),) -> ord.iperm[b] == ord.iperm[a] + 1, pairs)
    end
end

@testset "index types and bases: $INT" for INT in INTTYPES
    T = Float64
    A = kkt_matrix(T, 200, 100, 0.0; hessian = :indefinite)
    L = tril(A)
    S1 = SDS.symbolic_analysis(SDS.CSR(SparseMatrixCSC{T, INT}(L)), "S", 'L')
    for index in ('O', 'Z')
        C = SDS.CSR(SparseMatrixCSC{T, INT}(L); index)
        @test eltype(C.rowptr) == INT
        P = SDS.SymmetricPattern(C, "S"; view = 'L')
        @test SDS.pivot_pairs(P, C, "S"; view = 'L') == [(i, 200 + i) for i in 1:100]
        S = SDS.symbolic_analysis(C, "S", 'L')
        @test same_partition(S, S1)
        Sd = SDS.adapt(CPU(), S, INT)
        @test eltype(Sd.amap) == INT && Array(Sd.perm) == S.partition.perm
        N = SDS.allocate_numeric(S, T)
        SDS.ref_ldlt!(N, S, C.nzval)
        @test SDS.pivot_stats(N).nperturbed == 0 && ldlt_error(A, S, N) <= ldlt_factor_tol(T)
    end
end

@testset "pivot_pairs parameter" begin
    opts = Options()
    @test getparam(opts, "pivot_pairs") == "default" && opts.pivot_pairs == SDS.PIVOT_PAIRS_DEFAULT
    setparam!(opts, "pivot_pairs", "none")
    @test getparam(opts, "pivot_pairs") == "none" && !SDS.pairs_enabled("S", opts)
    @test thrown(() -> setparam!(opts, "pivot_pairs", "auto")) isa InvalidValueError
    @test thrown(() -> setparam!(opts, "pivot_pairs", 1)) isa InvalidValueError
    @test "pivot_pairs" in EXTRA_PARAMETERS
    @test SDS.pairs_enabled("S", Options()) && SDS.pairs_enabled("H", Options())
end
