# T11: regime A, the fused subtree kernel (one launch per budget class, one
# workgroup per leaf subtree, fronts and contribution blocks in local memory),
# together with regimes B and C, checked against the T08 reference panels.

# default budgets (16, 32, 48 KiB): regimes A, B and C on the 3-D Laplacian and the random SPD matrix; regime-A
# subtrees as large as the budgets allow (`subtree_parallelism = 0`; the default split leaves these small
# matrices few or no subtrees, see "subtree_parallelism" in test_symbolic_schedule.jl)
const NUMERIC_A_OPTS = Options(subtree_parallelism = 0)
# small budgets and regime-C thresholds: all three regimes on a 2-D Laplacian
const NUMERIC_ABC_OPTS = Options(subtree_budgets = [8192, 16384], regime_c_width = 16, regime_c_rows = 128,
                                 subtree_parallelism = 0)

numeric_a_matrices(::Type{T}) where {T} =
    (("laplacian2d(40,40) A+B+C", laplacian2d(T, 40, 40), NUMERIC_ABC_OPTS),
     ("laplacian2d(40,40) A only", laplacian2d(T, 40, 40), NUMERIC_A_OPTS),
     ("random_spd(500,0.01)", random_spd(T, 500, 0.01), NUMERIC_A_OPTS),
     ("laplacian3d(10,10,10)", laplacian3d(T, 10, 10, 10), NUMERIC_A_OPTS))

numeric_a_allocated(N, S, nz) = @allocated SDS.factorize!(N, S, nz)

nregime_a(S, r) = count(==(r), S.schedule.regime)

# rounds of the local move of the contribution block of a non-root regime-A node (host replica of the kernel)
function subtree_move_rounds(S, v)
    sc, L = S.schedule, S.layout
    f, w = sc.rows[v], sc.width[v]
    m = f - w
    (L.local_cb[v] == 0 || m == 0) && return 0
    d = L.local_front[v] + w * (2f - w + 1) ÷ 2 - L.local_cb[v]
    return cld(m * (m + 1) ÷ 2, d)
end

RUN_SHARED && @testset "plan: regime-A groups" begin
    Random.seed!(666)
    for (name, A, opts) in numeric_a_matrices(Float64)
        S, _, _, _, Nd, _ = numeric_setup(CPU(), A; opts)
        sc, plan = S.schedule, Nd.plan
        @test SDS.nsubtrees(sc) > 0
        # one launch per budget class in use, before every B/C group; every subtree once
        @test length(plan.sub_first) == length(unique(sc.subtree_class))
        @test all(g -> g.regime == SDS.REGIME_A, sc.groups[1:length(plan.sub_first)])
        trees = reduce(vcat, [sc.group_nodes[plan.sub_first[k]:plan.sub_last[k]] for k in eachindex(plan.sub_first)])
        @test sort(trees) == 1:SDS.nsubtrees(sc)
        for k in eachindex(plan.sub_first), t in sc.group_nodes[plan.sub_first[k]:plan.sub_last[k]]
            @test plan.sub_local[k] in SDS.SUBTREE_LOCAL_SIZES
            @test S.layout.local_len[t] <= (plan.sub_local[k] - SDS.SUBTREE_LOCAL_RESERVE) ÷ sizeof(Float64)
        end
        @test SDS.memory_estimates(S, Float64)[12] == maximum(plan.sub_local)
        # B/C groups cover the other fronts
        seen = copy(sc.subtree_nodes)
        foreach(k -> append!(seen, sc.group_nodes[plan.group_first[k]:plan.group_last[k]]), eachindex(plan.group_first))
        @test sort(seen) == 1:SDS.nsupernodes(S)
    end
    # the budget classes map to the kernel sizes, smaller budgets disable regime A
    @test SDS.subtree_local_bytes(16 * 1024) == 16384
    @test SDS.subtree_local_bytes(20000) == 16384
    @test SDS.subtree_local_bytes(1 << 20) == 65536
    @test SDS.subtree_local_bytes(4096) == 0 && SDS.subtree_capacity(4096, 8) == 0
    S = SDS.symbolic_analysis(SDS.CSR(tril(laplacian2d(Float64, 10, 10))), "SPD", 'L';
                              opts = Options(subtree_budgets = [4096]))
    @test SDS.nsubtrees(S.schedule) == 0
    # the local capacity depends on the element size: wider numeric elements than the analysis's are rejected
    C = SDS.CSR(tril(laplacian2d(Float32, 10, 10)))
    S32 = SDS.symbolic_analysis(C, "SPD", 'L'; opts = NUMERIC_A_OPTS)
    @test S32.elsize == sizeof(Float32) && SDS.nsubtrees(S32.schedule) > 0
    @test thrown(() -> SDS.factorize!(SDS.allocate_numeric(S32, Float64), S32, Float64.(C.nzval))) isa
          InvalidValueError
    S64 = SDS.symbolic_analysis(SDS.CSR(tril(laplacian2d(Float64, 10, 10))), "SPD", 'L'; opts = NUMERIC_A_OPTS)
    @test SDS.factorize!(SDS.allocate_numeric(S64, Float32), S64, Float32.(C.nzval)) == 0
end

@testset "panels, solves, determinism ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    Random.seed!(666)
    for (name, A, opts) in numeric_a_matrices(T), INT in (name == "laplacian2d(40,40) A+B+C" ? INTTYPES : (Int32,))
        @testset "$name $INT" begin
            S, Nr, info_ref, Sd, Nd, nz = numeric_setup(backend, A, INT; opts)
            @test nregime_a(S, SDS.REGIME_A) > 0
            if opts === NUMERIC_ABC_OPTS
                @test nregime_a(S, SDS.REGIME_B) > 0 && nregime_a(S, SDS.REGIME_C) > 0
            end
            @test info_ref == 0
            @test SDS.factorize!(Nd, Sd, nz) == 0
            Nh = SDS.host_numeric(Nd)
            F1 = copy(Nh.factor)                         # `host_numeric` aliases host storage
            @test panel_error(Nh, Nr) <= panel_tol(T)
            @test Nh.stats == Nr.stats
            @test SDS.extract_L(Sd, Nd) == SDS.extract_L(S, Nh)
            for nrhs in (1, 5)
                b = nrhs == 1 ? rand(T, size(A, 1)) : rand(T, size(A, 1), nrhs)
                @test relres(A, SDS.ref_solve!(similar(b), S, Nh, b), b) <= tol(T)
            end
            # determinism: the same values give bitwise identical panels
            @test SDS.factorize!(Nd, Sd, nz) == 0
            @test SDS.host_numeric(Nd).factor == F1
        end
    end
end

@testset "budget classes and kernel sizes ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    Random.seed!(666)
    # the same matrix under different budgets: other subtrees, other local sizes, same factor
    A = laplacian3d(T, 8, 8, 8)
    for opts in (Options(subtree_budgets = [8192], subtree_parallelism = 0),
                 Options(subtree_budgets = [49152], subtree_parallelism = 0),
                 Options(subtree_budgets = [65536], subtree_parallelism = 0),
                 Options(subtree_budgets = [8192, 20000, 49152], regime_c_width = 32, subtree_parallelism = 0),
                 Options(subtree_budgets = [8192, 20000, 1 << 20], regime_c_width = 32, subtree_parallelism = 0),
                 Options(subtree_budgets = [32768], factorization_alg = "algo2", subtree_parallelism = 0))
        # the 64 KiB class (issue #60) runs where the backend has the local memory (CPU, ROCm); elsewhere (CUDA)
        # the allocation of an analysis that uses it is rejected
        if SDS.subtree_local_bytes(maximum(opts.subtree_budgets)) > SDS.max_local_bytes(backend)
            @test thrown(() -> numeric_setup(backend, A; opts)) isa InvalidValueError
            continue
        end
        S, Nr, _, Sd, Nd, nz = numeric_setup(backend, A; opts)
        @test SDS.nsubtrees(S.schedule) > 0
        @test SDS.factorize!(Nd, Sd, nz) == 0
        Nh = SDS.host_numeric(Nd)
        @test panel_error(Nh, Nr) <= panel_tol(T)
        @test Nh.stats == Nr.stats
        b = rand(T, size(A, 1))
        @test relres(A, SDS.ref_solve!(similar(b), S, Nh, b), b) <= tol(T)
    end
end

@testset "nlaunches on laplacian2d(100, 100) with AMD ($(backend_name(backend)), $T)" for backend in BACKENDS,
                                                                                            T in ELTYPES
    Random.seed!(666)
    A = laplacian2d(T, 100, 100)
    b = rand(T, size(A, 1), 2)
    x = map((Options(reordering_alg = "algo3", subtree_parallelism = 0),
             Options(reordering_alg = "algo3", subtree_budgets = Int[]))) do opts
        S, Nr, _, Sd, Nd, nz = numeric_setup(backend, A; opts)
        @test SDS.factorize!(Nd, Sd, nz) == 0
        Nh = SDS.host_numeric(Nd)
        @test panel_error(Nh, Nr) <= panel_tol(T)
        xa = SDS.ref_solve!(similar(b), S, Nh, b)
        @test relres(A, xa, b) <= tol(T)
        (SDS.nlaunches(S.schedule), xa, S)
    end
    (nA, xA, SA), (n0, x0, _) = x
    backend isa CPU && println("  nlaunches laplacian2d(100, 100) AMD, $T: $nA with regime A, $n0 without")
    @test SDS.nsubtrees(SA.schedule) > 0
    # the budgets are bytes: 16-byte elements (ComplexF64) fit half the fronts per subtree (measured 35 vs 80)
    @test 3 * nA <= n0 || (sizeof(T) == 16 && nA < n0)
    @test norm(xA - x0) <= tol(T) * norm(x0)
    # local moves in several rounds (source and destination overlap) occur here
    @test any(v -> subtree_move_rounds(SA, v) > 1, SA.schedule.subtree_nodes)
end

@testset "views, index bases, refactorization ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    Random.seed!(666)
    A = random_spd(T, 300, 0.02)
    for opts in (NUMERIC_A_OPTS, NUMERIC_ABC_OPTS)
        S, Nr, _, Sd, Nd, nz = numeric_setup(backend, A; opts)
        @test nregime_a(S, SDS.REGIME_A) > 0
        @test SDS.factorize!(Nd, Sd, nz) == 0
        F0 = copy(SDS.host_numeric(Nd).factor)
        @test maximum(abs, F0 - Nr.factor) <= panel_tol(T) * maximum(abs, Nr.factor)
        for view in ('U', 'F'), index in ('O', 'Z')
            _, _, _, Sv, Nv, nzv = numeric_setup(backend, A; view, index, opts)
            @test SDS.factorize!(Nv, Sv, nzv) == 0
            @test SDS.host_numeric(Nv).factor == F0
        end
        # new values on the same pattern and the same Symbolic and storage
        B = copy(A)
        nonzeros(B) .= rand(T, nnz(B))
        B = (B + B') / 2 + 2 * size(B, 1) * I
        @test SparseMatrixCSC(B .!= 0) == SparseMatrixCSC(A .!= 0)
        Cb = SDS.CSR(tril(B))
        S2, Nr2, _, _, _, _ = numeric_setup(CPU(), B; opts)
        @test SDS.factorize!(Nd, Sd, to_device(backend, Cb.nzval)) == 0
        Nh = SDS.host_numeric(Nd)
        @test panel_error(Nh, Nr2) <= panel_tol(T)
        b = rand(T, size(B, 1), 2)
        @test relres(B, SDS.ref_solve!(similar(b), S2, Nh, b), b) <= tol(T)
        @test SDS.factorize!(Nd, Sd, nz) == 0            # and back
        @test SDS.host_numeric(Nd).factor == F0
        if backend isa CPU
            # no allocation in the numeric phase (the first call compiles), see `ka_cpu_alloc_budget`
            numeric_a_allocated(Nd, Sd, nz)
            @test numeric_a_allocated(Nd, Sd, nz) <= numeric_alloc_budget(S, T)
        end
    end
end

@testset "info: first non-positive pivot ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    Random.seed!(666)
    n, j = 60, 23
    for opts in (NUMERIC_A_OPTS, Options(reordering_alg = "algo5", subtree_parallelism = 0),
                 Options(use_superpanels = 0, subtree_parallelism = 0), NUMERIC_ABC_OPTS)
        A = singular_block_matrix(T, n, j; stored_zero = true)
        A[j, j] = 3
        _, Nr, info_ref, Sd, Nd, nz = numeric_setup(backend, A; opts)
        @test info_ref == 0
        @test SDS.factorize!(Nd, Sd, nz) == 0
        @test panel_error(SDS.host_numeric(Nd), Nr) <= panel_tol(T)
        for d in (-3, 0)                                # negative and zero pivot at column j
            A[j, j] = d
            S, Nr, info_ref, Sd, Nd, nz = numeric_setup(backend, A; opts)
            s = S.partition.col2sn[S.partition.iperm[j]]
            @test S.schedule.regime[s] == SDS.REGIME_A  # the failing front is in a subtree
            @test info_ref == j
            @test SDS.factorize!(Nd, Sd, nz) == j
            stats = to_host(Nd.stats)
            @test stats[s * SDS.FRONT_STATS_FIELDS] == Nr.stats[s * SDS.FRONT_STATS_FIELDS] ==
                  S.partition.iperm[j] - S.partition.super_ptr[s] + 1
            @test stats[(s - 1) * SDS.FRONT_STATS_FIELDS + 1] == Nr.stats[(s - 1) * SDS.FRONT_STATS_FIELDS + 1]
        end
    end
    # a non-positive pivot that only appears after elimination: [1 2; 2 1] block at columns 3, 4
    A = sparse(T[4 0 0 0; 0 4 0 0; 0 0 1 2; 0 0 2 1])
    _, _, info_ref, Sd, Nd, nz = numeric_setup(backend, A; opts = Options(reordering_alg = "algo5", subtree_parallelism = 0))
    @test SDS.nsubtrees(Sd.schedule) > 0
    @test info_ref == 4
    @test SDS.factorize!(Nd, Sd, nz) == 4
end
