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
            # automatic choice: both candidates evaluated; the chosen one pays at most `level_flops` flops per
            # schedule level it saves (T22: `flops + level_flops × sdepth`; the T05 model was within 1.1× of the
            # best flops, which ND on lap2d, 10 levels shallower for 1.4× the flops, no longer is)
            auto = SDS.compute_ordering(P, Options())
            @test auto.stats.auto
            @test auto.alg_used in (:amd, :nd)
            @test sort([c.alg for c in auto.stats.candidates]) == [:amd, :nd]
            best = min(results[:amd].stats.flops, results[:nd].stats.flops)
            deepest = max(results[:amd].stats.sdepth, results[:nd].stats.sdepth)
            @test auto.stats.flops <= best + auto.stats.level_flops * (deepest - auto.stats.sdepth)
            chosen = only(c for c in auto.stats.candidates if c.alg === auto.alg_used)
            @test chosen.cost == minimum(c.cost for c in auto.stats.candidates)
            @test auto.stats.flops == chosen.flops
            @test auto.stats.sdepth == chosen.sdepth
            @test auto.perm == results[auto.alg_used].perm
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

@testset "ordering chooser: supernodal schedule depth (T22)" begin
    # the KKT generators of test/matrices.jl: the chooser takes the candidate with the smaller schedule depth
    # (ties: the smaller cost), reported through `Ordering.stats`
    kkts = (("kkt_matrix(60, 20)", kkt_matrix(60, 20, 1.0e-8)), ("kkt_matrix(200, 80)", kkt_matrix(200, 80, 1.0e-8)),
            ("kkt_matrix(300, 100)", kkt_matrix(300, 100, 1.0e-8)),
            ("kkt_matrix(2000, 800)", kkt_matrix(2000, 800, 1.0e-8)),
            ("kkt_slack_matrix(200, 60)", kkt_slack_matrix(Float64, 200, 60, 0.0)),
            ("kkt_matrix(300, 100, indefinite)", kkt_matrix(300, 100, 1.0e-8; hessian = :indefinite)))
    strict = 0
    for (name, A) in kkts
        @testset "$name" begin
            P = SDS.SymmetricPattern(SDS.CSR(A), "S"; view = 'F')
            auto = SDS.compute_ordering(P, Options())
            cands = auto.stats.candidates
            @test sort([c.alg for c in cands]) == [:amd, :nd]
            chosen = only(c for c in cands if c.alg === auto.alg_used)
            @test auto.stats.sdepth == chosen.sdepth == minimum(c.sdepth for c in cands)
            ties = [c for c in cands if c.sdepth == chosen.sdepth]
            @test chosen.cost == minimum(c.cost for c in ties)
            strict += length(ties) == 1
            opts = Options()
            for c in cands
                ord = SDS.compute_ordering(P, opts; alg = c.alg)
                @test (ord.stats.sdepth, ord.stats.nnz_L, ord.stats.flops) == (c.sdepth, c.nnz_L, c.flops)
                # the schedule depth is the height of the supernodal tree the analysis builds
                sp = SDS.supernode_partition(P, ord.perm, opts)
                @test SDS.tree_levels(sp.snparent)[2] == c.sdepth
                @test SDS.nsupernodes(sp) == c.nsupernodes
                @test c.cost == SDS.ordering_cost(c.flops, c.sdepth)
                # without amalgamation: the fundamental supernodes
                flat = SDS.compute_ordering(P, Options(use_superpanels = 0); alg = c.alg)
                spf = SDS.supernode_partition(P, ord.perm, Options(use_superpanels = 0))
                @test flat.stats.sdepth == SDS.tree_levels(spf.snparent)[2] >= c.sdepth
            end
        end
    end
    @test strict >= 2                # the generators include strict depth differences, not only ties

    # explicit orderings reproduce the CliqueTrees/Metis permutations of T05 bit for bit
    for (name, A) in (kkts[3], kkts[5], ("lap2d", laplacian2d(30, 30)))
        P = SDS.SymmetricPattern(SDS.CSR(A), "S"; view = 'F')
        G = SparseMatrixCSC(P)
        amd = Vector{Int}(first(SDS.CliqueTrees.permutation(G; alg = SDS.CliqueTrees.AMD())))
        nd = Vector{Int}(first(SDS.CliqueTrees.permutation(G; alg = SDS.ND_PROVIDER[](10, -1, -1))))
        for (s, ref) in (("algo1", amd), ("algo2", amd), ("algo3", amd), ("algo4", nd))
            ord = SDS.compute_ordering(P, Options(reordering_alg = s))
            @test ord.perm == ref
            @test !ord.stats.auto && length(ord.stats.candidates) == 1
        end
    end

    # the depth weight: 0 is a flop contest, a huge weight a depth contest
    A = laplacian2d(30, 30)
    P = SDS.SymmetricPattern(SDS.CSR(A), "SPD"; view = 'F')
    flop = SDS.compute_ordering(P, Options(); level_flops = 0)
    deep = SDS.compute_ordering(P, Options(); level_flops = 1.0e12)
    @test flop.stats.level_flops == 0 && deep.stats.level_flops == 1.0e12
    @test flop.stats.flops == minimum(c.flops for c in flop.stats.candidates)
    @test deep.stats.sdepth == minimum(c.sdepth for c in deep.stats.candidates)
    @test flop.alg_used === :amd && deep.alg_used === :nd      # AMD: fewer flops, ND: 10 fewer levels
    @test SDS.compute_ordering(P, Options()).stats.level_flops == SDS.ORDERING_LEVEL_FLOPS
    @test SDS.ordering_cost(2.0e6, 3, 1.0e5) == 2.3e6

    # schedule_depth: chain of single-child columns with nested structure = one supernode
    @test SDS.schedule_depth([2, 3, 4, 0], [1, 2, 3, 4], [4, 3, 2, 1], nothing) == (1, 1)
    @test SDS.schedule_depth([4, 4, 4, 0], [1, 2, 3, 4], [2, 2, 2, 1], nothing) == (2, 4)

    # the analysis reports the schedule depth it scored: `Schedule.nlevels`
    for backend in BACKENDS, T in eltypes_among((Float64, ComplexF32))
        A = kkt_matrix(T, 200, 80, 1.0e-8)
        solver = DirectSolver(api_matrix(backend, triangle_view(A, 'L'), Int32), "S", 'L')
        execute!("analysis", solver, nothing, nothing)
        @test solver.ordering.stats.auto
        @test solver.host_symbolic.schedule.nlevels == solver.ordering.stats.sdepth ==
              minimum(c.sdepth for c in solver.ordering.stats.candidates)
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
