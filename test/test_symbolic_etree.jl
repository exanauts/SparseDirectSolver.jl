# T05: symmetric pattern, orderings, elimination tree, column counts (host only).

@testset "elimination tree and column counts vs brute force" begin
    for trial in 1:200
        n = rand(5:60)
        density = rand((0.02, 0.05, 0.1, 0.2, 0.4))
        A = random_symindef(n, density)
        P = SDS.SymmetricPattern(SDS.CSR(A), "S"; view = 'F')
        perm = isodd(trial) ? randperm(n) : collect(1:n)
        parent = SDS.etree(P, perm)
        post = SDS.postorder(parent)
        counts = SDS.colcounts(P, perm, parent, post)
        ref_parent, ref_counts, _ = brute_force_symbolic(A, perm)
        @test parent == ref_parent
        @test counts == ref_counts
        @test isperm(post)
    end
end

@testset "postorder and tree levels" begin
    # chain 1 → 2 → 3 → 4, star (1, 2, 3 → 4), forest {1 → 3, 2 → 3}, {4}
    for (parent, levels, height) in (([2, 3, 4, 0], 4, [1, 2, 3, 4]),
                                     ([4, 4, 4, 0], 2, [1, 1, 1, 2]),
                                     ([3, 3, 0, 0], 2, [1, 1, 2, 1]),
                                     (Int[], 0, Int[]))
        post = SDS.postorder(parent)
        @test isperm(post)
        pos = invperm(post)
        @test all(parent[j] == 0 || pos[j] < pos[parent[j]] for j in eachindex(parent))
        h, nlev = SDS.tree_levels(parent)
        @test h == height
        @test nlev == levels
    end
    @test SDS.postorder([2, 3, 4, 0]) == [1, 2, 3, 4]
    @test thrown(() -> SDS.postorder([2, 1])) isa InvalidValueError
    @test SDS.cholesky_flops([1, 2, 3]) == 14.0
    @test SDS.nnz_L(Int[]) == 0
end

@testset "nnz(L) vs CHOLMOD" begin
    opts = Options()
    for (name, A) in (("laplacian2d(50, 50)", laplacian2d(50, 50)),
                      ("laplacian3d(12, 12, 12)", laplacian3d(12, 12, 12)),
                      ("random_spd(2000, 0.002)", random_spd(2000, 0.002)))
        P = SDS.SymmetricPattern(SDS.CSR(A), "SPD"; view = 'F')
        for alg in (:amd, :natural)
            ord = SDS.compute_ordering(P, opts; alg)
            parent = SDS.etree(P, ord.perm)
            counts = SDS.colcounts(P, ord.perm, parent, SDS.postorder(parent))
            L = dropzeros!(sparse(cholesky(A; perm = ord.perm).L))
            @testset "$name $alg" begin
                @test SDS.nnz_L(counts) == nnz(L)
                @test ord.stats.nnz_L == nnz(L)
            end
        end
    end
end

@testset "orderings" begin
    lap2 = laplacian2d(30, 30)
    lap3 = laplacian3d(10, 10, 10)
    kkt = kkt_matrix(300, 100, 1.0e-8)
    @test SDS.nd_available()
    for (name, A) in (("lap2d", lap2), ("lap3d", lap3), ("kkt", kkt))
        n = size(A, 1)
        P = SDS.SymmetricPattern(SDS.CSR(A), "S"; view = 'F')
        @testset "$name" begin
            results = Dict{Symbol, Any}()
            for alg in (:natural, :amd, :mmd, :nd, :auto)
                ord = SDS.compute_ordering(P, Options(); alg)
                @test isperm(ord.perm)
                @test length(ord.perm) == n
                @test ord.iperm == invperm(ord.perm)
                alg === :auto || @test ord.alg_used === alg
                results[alg] = ord
            end
            @test results[:natural].perm == 1:n
            # the options select the same algorithms
            for (s, alg) in (("algo5", :natural), ("algo3", :amd), ("algo4", :nd))
                ord = SDS.compute_ordering(P, Options(reordering_alg = s))
                @test ord.alg_used === alg
                @test ord.perm == results[alg].perm
                @test !ord.stats.auto
            end
            # AMD never fills more than natural ordering on these matrices
            @test results[:amd].stats.nnz_L <= results[:natural].stats.nnz_L
            # automatic choice: both candidates evaluated, flops within 1.1× of the better one
            auto = SDS.compute_ordering(P, Options())
            @test auto.stats.auto
            @test auto.alg_used in (:amd, :nd)
            @test sort([c.alg for c in auto.stats.candidates]) == [:amd, :nd]
            best = min(results[:amd].stats.flops, results[:nd].stats.flops)
            @test auto.stats.flops <= 1.1 * best
            chosen = only(c for c in auto.stats.candidates if c.alg === auto.alg_used)
            @test chosen.cost == minimum(c.cost for c in auto.stats.candidates)
            @test auto.stats.flops == chosen.flops
            # complex element types scale the flop count only
            cplx = SDS.compute_ordering(P, Options(); T = ComplexF64)
            @test cplx.perm == auto.perm
            @test cplx.stats.flops == 4 * auto.stats.flops
            # ND parameters are passed to the algorithm object and give valid orderings
            for (nlev, ub) in ((0, -1), (1, -1), (3, 50), (10, 200))
                ord = SDS.compute_ordering(P, Options(reordering_alg = "algo4", nd_nlevels = nlev, nd_ubfactor = ub))
                @test isperm(ord.perm)
                @test ord.alg_used === :nd
            end
        end
    end
    @test SDS.ND_PROVIDER[](10, 50, -1).ufactor == 50
    @test SDS.ND_PROVIDER[](0, -1, -1).ufactor == -1
    @test SDS.ND_PROVIDER[](10, -1, 4).nseps == 4
    @test SDS.ND_PROVIDER[](10, -1, 0).nseps == -1

    @testset "user_perm" begin
        A = laplacian2d(7, 6)
        n = size(A, 1)
        P = SDS.SymmetricPattern(SDS.CSR(A), "SPD"; view = 'L')
        p = randperm(n)
        for given in (p, p .- 1, Int32.(p), Int32.(p .- 1))
            ord = SDS.compute_ordering(P, Options(user_perm = given, reordering_alg = "algo3"))
            @test ord.alg_used === :user
            @test ord.perm == p
            @test ord.iperm == invperm(p)
        end
        @test thrown(() -> SDS.compute_ordering(P, Options(user_perm = p[1:(end - 1)]))) isa InvalidValueError
        bad = copy(p); bad[1] = bad[2]
        @test thrown(() -> SDS.compute_ordering(P, Options(user_perm = bad))) isa InvalidValueError
        @test thrown(() -> SDS.compute_ordering(P, Options(user_perm = p .+ 1))) isa InvalidValueError
        @test thrown(() -> SDS.compute_ordering(P, Options(); alg = :user)) isa InvalidValueError
        @test thrown(() -> SDS.compute_ordering(P, Options(); alg = :colamd)) isa InvalidValueError
    end

    @testset "without Metis" begin
        A = laplacian2d(12, 12)
        P = SDS.SymmetricPattern(SDS.CSR(A), "SPD"; view = 'F')
        saved = SDS.ND_PROVIDER[]
        try
            SDS.ND_PROVIDER[] = nothing
            @test !SDS.nd_available()
            @test thrown(() -> SDS.compute_ordering(P, Options(reordering_alg = "algo4"))) isa NotSupportedError
            ord = SDS.compute_ordering(P, Options())
            @test ord.alg_used === :amd
            @test [c.alg for c in ord.stats.candidates] == [:amd]
            @test !ord.stats.nd_available
        finally
            SDS.ND_PROVIDER[] = saved
        end
        @test SDS.nd_available()
    end

    @testset "empty and tiny patterns" begin
        for n in (0, 1)
            P = SDS.SymmetricPattern(SDS.CSR(sparse(1.0I, n, n)), "SPD")
            for alg in (:natural, :amd, :mmd, :nd, :auto)
                ord = SDS.compute_ordering(P, Options(); alg)
                @test ord.perm == 1:n
                @test ord.stats.nnz_L == n
            end
        end
    end
end

@testset "SymmetricPattern" begin
    for T in ELTYPES
        A = random_symindef(T, 40, 0.1; hermitian = false)
        H = random_symindef(T, 40, 0.1)
        ref = offdiag_pattern(A)
        for (structure, M) in (("S", A), (T <: Real ? "SPD" : "HPD", H), ("H", H))
            pats = [SDS.SymmetricPattern(SDS.CSR(B; index), structure; view = vw)
                    for (B, vw) in ((M, 'F'), (sparse(tril(M)), 'L'), (sparse(triu(M)), 'U'))
                    for index in ('O', 'Z')]
            @test all(==(pats[1]), pats)
            @test SparseMatrixCSC(pats[1]) == offdiag_pattern(M)
        end
        @test SparseMatrixCSC(SDS.SymmetricPattern(SDS.CSR(A), "S")) == ref
        # 'F' on a symmetric structure reads only the lower triangle
        U = sparse(triu(A))
        @test nnz(SDS.SymmetricPattern(SDS.CSR(U), "S"; view = 'F')) == 0
        # "G": the pattern of A + Aᵀ
        G = random_general(T, 50, 0.05)
        Gp = SparseMatrixCSC(size(G)..., G.colptr, G.rowval, ones(nnz(G)))   # structure only
        @test !issymmetric(Gp)
        for index in ('O', 'Z')
            PG = SDS.SymmetricPattern(SDS.CSR(G; index), "G")
            @test SparseMatrixCSC(PG) == offdiag_pattern(Gp + Gp')
            # "G" ignores the view (cuDSS, #84): every stored entry is read
            @test SDS.SymmetricPattern(SDS.CSR(G; index), "G"; view = 'L') == PG
            @test SDS.SymmetricPattern(SDS.CSR(G; index), "G"; view = 'U') == PG
            FL = SDS.full_pattern_map(SDS.CSR(G; index), "G"; view = 'L')
            FF = SDS.full_pattern_map(SDS.CSR(G; index), "G"; view = 'F')
            @test all(getfield(FL, f) == getfield(FF, f) for f in fieldnames(SDS.FullPatternMap))
        end
    end

    # duplicates are dropped; enums and strings are interchangeable
    rowptr = [1, 3, 6, 8]
    colval = [1, 1, 2, 1, 1, 1, 3]   # row 1: (1,1) twice; row 2: (2,2), (2,1) twice; row 3: (3,1), (3,3)
    P = SDS.SymmetricPattern(rowptr, colval, 3, SDS.STRUCTURE_SYMMETRIC; view = SDS.VIEW_LOWER)
    @test P == SDS.SymmetricPattern(3, [1, 3, 4, 5], [2, 3, 1, 1])
    @test P == SDS.SymmetricPattern(rowptr .- 1, colval .- 1, 3, "S"; view = 'L', index = 'Z')
    @test nnz(P) == 4
    @test collect(SDS.neighbors(P, 1)) == [2, 3]

    # rejected inputs
    R = sprand(4, 5, 0.5)
    @test thrown(() -> SDS.SymmetricPattern(SDS.CSR(R), "G")) isa InvalidValueError
    @test thrown(() -> SDS.full_pattern_map(SDS.CSR(R), "G")) isa InvalidValueError
    @test thrown(() -> SDS.SymmetricPattern([1, 2, 3], [1, 4], 2, "S")) isa InvalidValueError
    @test thrown(() -> SDS.SymmetricPattern([1, 2, 3], [1, 0], 2, "S")) isa InvalidValueError
    @test thrown(() -> SDS.SymmetricPattern([1, 2], [1, 1], 2, "S")) isa InvalidValueError
    @test thrown(() -> SDS.SymmetricPattern([1, 2, 3], [1, 2], 2, "X")) isa InvalidValueError
    @test thrown(() -> SDS.SymmetricPattern([1, 2, 3], [1, 2], 2, "S"; view = 'Q')) isa InvalidValueError
    @test thrown(() -> SDS.SymmetricPattern(2, [1, 2, 3], [2, 2])) isa InvalidValueError   # (2,2) is diagonal
    @test thrown(() -> SDS.SymmetricPattern(2, [1, 2, 2], [1])) isa InvalidValueError      # (1,1) is diagonal
    @test thrown(() -> SDS.SymmetricPattern(3, [1, 3, 3, 3], [3, 2])) isa InvalidValueError # unsorted
end

@testset "full_pattern_map" begin
    for T in ELTYPES
        S = random_symindef(T, 30, 0.15; hermitian = false)   # complex symmetric for complex T
        H = random_symindef(T, 30, 0.15)                      # Hermitian
        G = random_general(T, 30, 0.1)
        cases = Any[("S", S), ("H", H), (T <: Real ? "SPD" : "HPD", H)]
        for (structure, M) in cases, (B, vw) in ((M, 'F'), (sparse(tril(M)), 'L'), (sparse(triu(M)), 'U')),
            index in ('O', 'Z')
            C = SDS.CSR(B; index)
            F = SDS.full_pattern_map(C, structure; view = vw)
            @test SparseMatrixCSC(F, C.nzval) == M
            @test any(F.conjflag) == (structure in ("H", "HPD"))
        end
        for index in ('O', 'Z')
            C = SDS.CSR(G; index)
            F = SDS.full_pattern_map(C, "G")
            @test SparseMatrixCSC(F, C.nzval) == G
            @test !any(F.conjflag)
            @test F.srcptr == 1:(nnz(G) + 1)
        end
    end
    # duplicated user entries are summed, mirrored entries of "H" are conjugated
    rowptr = [1, 2, 5]
    colval = [1, 1, 2, 1]
    nzval = ComplexF64[1, 2 + 1im, 3, 4 - 2im]   # (1,1)=1, (2,1)=2+i, (2,2)=3, (2,1)+=4-2i
    F = SDS.full_pattern_map(rowptr, colval, 2, "H"; view = 'L')
    @test nnz(F) == 4
    @test SparseMatrixCSC(F, nzval) == sparse(ComplexF64[1 6+1im; 6-1im 3])
    @test SparseMatrixCSC(SDS.full_pattern_map(rowptr, colval, 2, "S"; view = 'L'), nzval) ==
          sparse(ComplexF64[1 6-1im; 6-1im 3])
    # a batch member is addressed by offsetting the map's source indices
    C = SDS.CSR(sparse(tril(laplacian2d(4, 4))))
    F = SDS.full_pattern_map(C, "SPD"; view = 'L')
    batch = [C.nzval; 2 .* C.nzval]
    @test SDS.full_values(F, view(batch, (nnz(C) + 1):(2 * nnz(C)))) == 2 .* SDS.full_values(F, C.nzval)
end
