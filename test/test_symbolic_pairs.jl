# #64, #66: 2×2 pivot candidate pairs in the analysis of "S"/"H" matrices (host only).
# A row with a zero or negligible diagonal is ordered right after a partner and
# shares its supernode, so in-front Bunch–Kaufman can pair them. "default" pairs
# the candidates whose pivot is structurally zero in the ordering, "all" every candidate.

# pattern, pairs and ordering of the lower triangle of A as the analysis computes them
function pairs_ordering(A::SparseMatrixCSC{T}; opts = Options(), structure = sym_structure(T), view = 'L') where {T}
    C = SDS.CSR(triangle_view(A, view))
    P = SDS.SymmetricPattern(C, structure; view)
    pp = SDS.analysis_pairs(P, C.rowptr, C.colval, C.nzval, C.nrows, structure, opts; view, index = C.index)
    return P, SDS.compute_ordering(P, opts; T, pp.pairs, pp.candidates), C
end

# both columns of every pair in one supernode, partner right before its candidate
pairs_in_supernodes(sp, pairs) =
    all(((a, b),) -> sp.iperm[b] == sp.iperm[a] + 1 && sp.col2sn[sp.iperm[a]] == sp.col2sn[sp.iperm[b]], pairs)

same_partition(S, R) = S.partition.perm == R.partition.perm && S.partition.super_ptr == R.partition.super_ptr &&
                       S.partition.rowval == R.partition.rowval && S.partition.nnz_L == R.partition.nnz_L

# no unpaired candidate has a structurally zero pivot in the final ordering
no_zero_pivot(P, cands, ord) = isempty(SDS.structural_zero_pivots(cands, P, ord.perm, ord.pairs))

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
    cands = SDS.pivot_candidates(P, C, "S"; view = 'L')
    @test findall(cands.candidate) == [1, 2, 3, 8]
    @test cands.amax == [7.0, 5.0, 9.0, 3.0, 3.0, 1.0, 9.0, 0.0]
    @test length(cands.w) == SDS.nnz(P) && all(>(0), cands.w)
    @test SDS.pivot_pairs(P, cands) == pairs
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
    @test ord.pairs == pairs && isperm(ord.perm) && ord.stats.pair_rounds == 0
    @test all(((a, b),) -> ord.iperm[b] == ord.iperm[a] + 1, pairs)
    @test thrown(() -> SDS.compute_ordering(P, Options(); pairs, candidates = cands)) isa InvalidValueError
end

@testset "structurally zero pivots and their pairs" begin
    # path 1 – 2 – 3 – 4 – 5 – 6, eliminated in the order 1, 6, 2, 3, 4, 5. Candidates 1 and 6 (no
    # earlier neighbour: elimination-tree leaves) have structurally zero pivots; 3 (zero diagonal) is
    # reached through 2, whose diagonal is free: 3 → column 2, row 2 → column 3. The only entry of row
    # 6, |a₆₅| = 1e-3, is its row maximum, so 5 is an acceptable partner.
    n = 6
    L = sparse([1, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5],
               [0.0, 4.0, 0.0, 4.0, 4.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0e-3], n, n)
    C = SDS.CSR(L)
    P = SDS.SymmetricPattern(C, "S"; view = 'L')
    cands = SDS.pivot_candidates(P, C, "S"; view = 'L')
    @test findall(cands.candidate) == [1, 3, 6]
    order = [1, 6, 2, 3, 4, 5]
    @test SDS.structural_zero_pivots(cands, P, order) == [1, 6]
    pairs = Tuple{Int, Int}[]
    @test SDS.zero_pivot_pairs!(pairs, cands, P, order, 0.01) == (2, 0)
    @test pairs == [(2, 1), (5, 6)]
    @test isempty(SDS.structural_zero_pivots(cands, P, order, pairs))
    # a second round adds nothing
    @test SDS.zero_pivot_pairs!(pairs, cands, P, order, 0.01) == (0, 0)
    @test thrown(() -> SDS.zero_pivot_pairs!(pairs, cands, P, [1, 2], 0.01)) isa InvalidValueError
    # in the natural order 1 still has no earlier neighbour, 6 now comes after 5
    @test SDS.structural_zero_pivots(cands, P, collect(1:n)) == [1]
    # two zero-diagonal rows whose only neighbour is the same primal: the first one takes it, the
    # second pivot is structurally zero although it is not a leaf of the elimination tree
    # (rows 1 = primal, 2 and 3 = duals; order 1, 2, 3)
    G = sparse([1, 2, 3, 2, 3], [1, 2, 3, 1, 1], [4.0, 0.0, 0.0, 1.0, 2.0], 3, 3)
    CG = SDS.CSR(G)
    PG = SDS.SymmetricPattern(CG, "S"; view = 'L')
    cg = SDS.pivot_candidates(PG, CG, "S"; view = 'L')
    @test SDS.etree(PG, [1, 2, 3]) == [2, 3, 0]                             # 3 has a child
    @test SDS.structural_zero_pivots(cg, PG, [1, 2, 3]) == [3]
    # a partner must reach u times the row maximum: row 1, (0, 1e-3, 1), skips the weak entry
    B = sparse([1, 2, 3, 2, 3], [1, 2, 3, 1, 1], [0.0, 4.0, 4.0, 1.0e-3, 1.0], 3, 3)
    CB = SDS.CSR(B)
    PB = SDS.SymmetricPattern(CB, "S"; view = 'L')
    cb = SDS.pivot_candidates(PB, CB, "S"; view = 'L')
    pb = Tuple{Int, Int}[]
    @test SDS.zero_pivot_pairs!(pb, cb, PB, [1, 2, 3], 0.01) == (1, 0) && pb == [(3, 1)]
    pb = Tuple{Int, Int}[]
    @test SDS.zero_pivot_pairs!(pb, cb, PB, [1, 2, 3], 0.0) == (1, 0) && pb == [(3, 1)]  # the largest |a|
    pb = Tuple{Int, Int}[]
    @test SDS.zero_pivot_pairs!(pb, cb, PB, [1, 2, 3], 2.0) == (0, 1) && isempty(pb)    # no acceptable partner
    # two candidates may pair with each other: [0 a; a 0] is a 2×2 pivot
    D = sparse([1, 2, 2], [1, 2, 1], [0.0, 0.0, 3.0], 2, 2)
    CD = SDS.CSR(D)
    PD = SDS.SymmetricPattern(CD, "S"; view = 'L')
    cd = SDS.pivot_candidates(PD, CD, "S"; view = 'L')
    @test SDS.structural_zero_pivots(cd, PD, [1, 2]) == [1]                 # 2 is reached through 1
    pd = Tuple{Int, Int}[]
    @test SDS.zero_pivot_pairs!(pd, cd, PD, [1, 2], 0.01) == (1, 0) && pd == [(2, 1)]
    @test isempty(SDS.structural_zero_pivots(cd, PD, [2, 1], pd))           # the 2×2 block, crosswise
    @test SDS.pivot_pairs(PD, CD, "S"; view = 'L') == []                   # "all" never pairs two candidates
    # a zero pivot without a free partner stays alone and is counted
    E = sparse([1, 2, 3, 3, 3], [1, 2, 3, 1, 2], [0.0, 0.0, 4.0, 1.0, 1.0], 3, 3)
    CE = SDS.CSR(E)
    PE = SDS.SymmetricPattern(CE, "S"; view = 'L')
    pe = Tuple{Int, Int}[]
    @test SDS.zero_pivot_pairs!(pe, SDS.pivot_candidates(PE, CE, "S"; view = 'L'), PE, [1, 2, 3], 0.01) == (1, 1)
    @test pe == [(3, 1)]
end

@testset "KKT matrices, default ordering, pivot_pairs = $mode: $T" for T in ELTYPES, mode in ("default", "all")
    opts = Options(pivot_pairs = mode)
    cases = [("kkt(300,100,1e-8)", 300, 100, kkt_matrix(T, 300, 100, 1.0e-8)),
             ("kkt(200,100,0) indefinite H", 200, 100, kkt_matrix(T, 200, 100, 0.0; hessian = :indefinite)),
             ("kkt(200,100,0) indefinite 1e-3 H", 200, 100,
              kkt_matrix(T, 200, 100, 0.0; hessian = :indefinite, hessian_scale = 1.0e-3))]
    for (name, nh, nj, A) in cases
        @testset "$name" begin
            P, ord, C = pairs_ordering(A; opts)
            cands = SDS.pivot_candidates(P, C, sym_structure(T); view = 'L')
            @test findall(cands.candidate) == (nh + 1):(nh + nj)                # the dual rows
            if mode == "all"
                # every dual row is a candidate, paired with the primal of its dominant J entry
                @test ord.pairs == [(i, nh + i) for i in 1:nj]
                @test ord.stats.pair_rounds == 0
            else
                # duals with a structurally zero pivot, each with a primal partner; none is left
                @test !isempty(ord.pairs) && ord.stats.pair_rounds >= 1
                @test all(((a, b),) -> a <= nh < b, ord.pairs)
                @test no_zero_pivot(P, cands, ord)
            end
            @test all(((a, b),) -> ord.iperm[b] == ord.iperm[a] + 1, ord.pairs)
            @test ord.stats.nnz_L == SDS.evaluate_ordering(SDS.pair_pattern(P, ord.pairs), ord.perm).nnz_L
            # the analysis keeps each pair in one supernode, also without amalgamation
            S = SDS.symbolic_analysis(C, sym_structure(T), 'L'; opts)
            @test S.partition.perm[S.partition.iperm] == 1:(nh + nj)
            @test pairs_in_supernodes(S.partition, ord.pairs)
            @test S.partition.nnz_L == ord.stats.nnz_L
            S0 = SDS.symbolic_analysis(C, sym_structure(T), 'L';
                                       opts = Options(use_superpanels = 0, pivot_pairs = mode))
            @test pairs_in_supernodes(S0.partition, ord.pairs)
            @test all(s -> SDS.snwidth(S0.partition, s) >= 2,
                      unique(S0.partition.col2sn[S0.partition.iperm[b]] for (_, b) in ord.pairs))
            # the factorization: no perturbation, no growth, exact inertia
            S, N, info, _ = reference_ldlt(A; opts)
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

@testset "KKT with slacks, pivot_pairs = default: $T" for T in ELTYPES
    # MadNLP K2 with inequality slacks: half of the slacks have no barrier term. A zero-Σ slack is a
    # zero pivot whose only neighbour is its dual (itself a candidate): "default" pairs them as a
    # [0 −1; −1 0] block at a few percent of fill; "all" matches candidates to non-candidates only
    # and leaves those slacks alone (perturbed).
    nh, ns = 200, 100
    A = kkt_slack_matrix(T, nh, ns, 0.0)
    P, ord, C = pairs_ordering(A)
    cands = SDS.pivot_candidates(P, C, sym_structure(T); view = 'L')
    _, none, _ = pairs_ordering(A; opts = Options(pivot_pairs = "none"))
    _, all_, _ = pairs_ordering(A; opts = Options(pivot_pairs = "all"))
    @test count(cands.candidate) == ns + ns ÷ 2                            # duals and zero-Σ slacks
    @test length(ord.pairs) < count(cands.candidate) ÷ 2
    @test no_zero_pivot(P, cands, ord)
    @test ord.stats.nnz_L <= 1.15 * none.stats.nnz_L
    @test ord.stats.nnz_L < all_.stats.nnz_L
    S, N, info, _ = reference_ldlt(A)
    st = SDS.pivot_stats(N)
    @test info == 0 && st.nperturbed == 0 && st.nzero == 0
    @test ldlt_error(A, S, N) <= ldlt_factor_tol(T)
    @test SDS.inertia(N) == eigen_npos_nneg(A)
    b = rand(T, nh + 2ns)
    @test relres(A, SDS.ref_solve!(similar(b), S, N, b), b) <= tol(T)
    # without pairs some zero-Σ slacks are eliminated alone and perturbed
    _, Nn, _, _ = reference_ldlt(A; opts = Options(pivot_pairs = "none"))
    @test SDS.pivot_stats(Nn).nperturbed > 0
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
    # The constraint "dual next to its partner" costs fill: under plain AMD a dual row is a cheap
    # leaf. On these generators almost every dual pivot is structurally zero, so "default" pairs
    # most of them and costs 1.6–2.2× nnz_L, like "all"; the T14 interleaved user_perm is worse. On
    # MadNLP K2 systems "default" costs 1.3–1.4× and "all" ~2× (issue #66, bench/pivot_pairs.jl).
    # Pairing fewer duals is not monotone in nnz_L under AMD: over 15 seeds "default"/"all" was
    # 0.96–1.025 (1.012 on Julia 1.10's draw), hence the 5% margin.
    for (nh, nj, A) in ((300, 100, kkt_matrix(T, 300, 100, 1.0e-8)),
                        (200, 100, kkt_matrix(T, 200, 100, 0.0; hessian = :indefinite)))
        _, ord, _ = pairs_ordering(A)
        _, all_, _ = pairs_ordering(A; opts = Options(pivot_pairs = "all"))
        _, none, _ = pairs_ordering(A; opts = Options(pivot_pairs = "none"))
        _, inter, _ = pairs_ordering(A; opts = Options(user_perm = kkt_interleaved_perm(nh, nj)))
        @test isempty(none.pairs) && isempty(inter.pairs) && length(all_.pairs) == nj
        @test 0 < length(ord.pairs) <= nj
        @test ord.stats.nnz_L <= 1.05 * all_.stats.nnz_L && all_.stats.nnz_L <= 2.5 * none.stats.nnz_L
        @test all_.stats.nnz_L <= inter.stats.nnz_L
    end
    # no candidate (Gershgorin-dominant diagonal): no pair, bitwise the same analysis
    A = random_symindef(T, 400, 0.01)
    P, ord, C = pairs_ordering(A)
    _, none, _ = pairs_ordering(A; opts = Options(pivot_pairs = "none"))
    @test isempty(ord.pairs) && ord.perm == none.perm && ord.stats == none.stats
    _, all_, _ = pairs_ordering(A; opts = Options(pivot_pairs = "all"))
    @test isempty(all_.pairs) && all_.perm == none.perm && all_.stats == none.stats
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

# MadNLP K2 dump (bench/dump_madnlp_kkt.jl; bench/data is not in git): checked when present
const K2_DUMP = joinpath(@__DIR__, "..", "bench", "data", "kkt_pglib_opf_case118_ieee_k2_10.mtx")
isdefined(@__MODULE__, :BenchMatrices) || include(joinpath(@__DIR__, "..", "bench", "matrices.jl"))

@testset "MadNLP K2 dump (case118, if present)" begin
    if !isfile(K2_DUMP)
        @info "skipped: $K2_DUMP not found (run bench/dump_madnlp_kkt.jl)"
    else
        A = BenchMatrices.symmetrize_triangle(tril(BenchMatrices.read_mtx(K2_DUMP)))
        res = Dict{String, Any}()
        for mode in ("none", "default", "all")
            S, N, info, _ = reference_ldlt(A; opts = Options(pivot_pairs = mode))
            @test info == 0
            res[mode] = (nnz_L = S.partition.nnz_L, nperturbed = SDS.pivot_stats(N).nperturbed)
        end
        # measured (issue #66): default 1.26× nnz_L, 5 perturbed pivots; all 2.18×, 65; none 147
        @test res["default"].nnz_L <= 1.5 * res["none"].nnz_L
        @test res["default"].nnz_L < res["all"].nnz_L
        @test res["default"].nperturbed < res["all"].nperturbed < res["none"].nperturbed
        @test res["default"].nperturbed <= res["none"].nperturbed ÷ 10
    end
end
