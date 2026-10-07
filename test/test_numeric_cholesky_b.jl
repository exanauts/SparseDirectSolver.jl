# T10: regime B, the fused per-front kernel (one launch per (step, bin), one
# workgroup per front), checked against the T08 reference panels and against
# the regime-C path.

# regime B below the regime-C thresholds, no regime-A subtrees (T11)
const NUMERIC_B_OPTS = Options(subtree_budgets = Int[])
# both regimes in one analysis: small thresholds send the larger fronts to regime C
const NUMERIC_BC_OPTS = Options(subtree_budgets = Int[], regime_c_width = 16, regime_c_rows = 128)

numeric_b_matrices(::Type{T}) where {T} =
    (("laplacian2d(40,40)", laplacian2d(T, 40, 40), NUMERIC_B_OPTS),
     ("random_spd(500,0.01)", random_spd(T, 500, 0.01), NUMERIC_B_OPTS),
     ("laplacian3d(10,10,10)", laplacian3d(T, 10, 10, 10), NUMERIC_B_OPTS),
     ("laplacian3d(10,10,10) B+C", laplacian3d(T, 10, 10, 10), NUMERIC_BC_OPTS))

numeric_b_allocated(N, S, nz) = @allocated SDS.factorize!(N, S, nz)

nregime(S, r) = count(==(r), S.schedule.regime)

RUN_SHARED && @testset "plan: regime-B groups" begin
    for opts in (NUMERIC_B_OPTS, NUMERIC_BC_OPTS, Options(subtree_budgets = Int[], regime_c_width = 128))
        A = laplacian3d(Float64, 10, 10, 10)
        S, _, _, _, Nd, _ = numeric_setup(CPU(), A; opts)
        sc, plan = S.schedule, Nd.plan
        @test nregime(S, SDS.REGIME_B) > 0
        seen = Int[]
        for k in eachindex(plan.group_first)
            nodes = sc.group_nodes[plan.group_first[k]:plan.group_last[k]]
            W = plan.group_width[k]
            if W > 0
                @test W in SDS.REGIME_B_WIDTHS
                @test all(s -> sc.regime[s] == SDS.REGIME_B && sc.width[s] <= W, nodes)
            else
                @test all(s -> sc.regime[s] == SDS.REGIME_C || sc.width[s] > SDS.REGIME_B_MAX_WIDTH, nodes)
            end
            @test plan.group_maxchild[k] == maximum(s -> count(==(s), S.partition.snparent), nodes; init = 0)
            append!(seen, nodes)
        end
        @test sort(seen) == 1:SDS.nsupernodes(S)
    end
    # factorization_alg: "algo2" sends every non-A front to regime C, "algo1" keeps regime B
    A = laplacian2d(Float64, 30, 30)
    S2, _, _, _, N2, _ = numeric_setup(CPU(), A; opts = Options(subtree_budgets = Int[], factorization_alg = "algo2"))
    @test nregime(S2, SDS.REGIME_C) == SDS.nsupernodes(S2) && all(==(0), N2.plan.group_width)
    S1, _, _, _, N1, _ = numeric_setup(CPU(), A; opts = Options(subtree_budgets = Int[], factorization_alg = "algo1"))
    @test nregime(S1, SDS.REGIME_B) == SDS.nsupernodes(S1) && all(>(0), N1.plan.group_width)
    @test !S1.schedule.vendor_c
end

# synthetic assembled fronts through the dense part of the kernel, against LAPACK on the host
@testset "front_cholesky! on synthetic fronts ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    Random.seed!(666)
    for W in SDS.REGIME_B_WIDTHS
        shapes = [(w, m, cb) for w in unique((1, W ÷ 2 + 1, W)) for (m, cb) in ((0, false), (1, true), (37, true),
                                                                               (5, false))]
        nf = length(shapes)
        front_ptr, cb_ptr = [1], zeros(Int, nf)
        stack_len = 1
        for (s, (w, m, cb)) in enumerate(shapes)
            push!(front_ptr, front_ptr[end] + (w + m) * w)
            if cb
                cb_ptr[s] = stack_len
                stack_len += m * (m + 1) ÷ 2                 # packed lower triangle (T11)
            end
        end
        factor = zeros(T, front_ptr[end] - 1)
        stack = zeros(T, stack_len - 1)
        fronts = Matrix{T}[]
        for (s, (w, m, cb)) in enumerate(shapes)
            M = rand(T, w + m, w + m)
            F = M * M' + (w + m) * I
            push!(fronts, F)
            P = reshape(view(factor, front_ptr[s]:(front_ptr[s + 1] - 1)), w + m, w)
            P .= tril(F[:, 1:w])
            cb && (stack[cb_ptr[s]:(cb_ptr[s] + m * (m + 1) ÷ 2 - 1)] .= pack_lower(F[(w + 1):end, (w + 1):end]))
        end
        # the fifth front (w = W ÷ 2 + 1, no contribution block) gets a negative pivot at its last column
        bad = 5
        wb = shapes[bad][1]
        Pb = reshape(view(factor, front_ptr[bad]:(front_ptr[bad + 1] - 1)), :, wb)
        Pb[wb, wb] = -abs(Pb[wb, wb]) - 10 * (wb + shapes[bad][2])^2
        fronts[bad][wb, wb] = Pb[wb, wb]
        dev(x) = to_device(backend, x)
        nodes = Int32.(collect(nf:-1:1))                 # any order of the fronts
        info = dev(fill(Int32(-1), nf))
        dfactor, dstack = dev(factor), dev(stack)
        SDS.front_cholesky!(dfactor, dstack, info, dev(nodes), 1, nf, dev(Int32.(front_ptr)),
                            dev(Int32[w + m for (w, m, _) in shapes]), dev(Int32[w for (w, _, _) in shapes]),
                            dev(Int32.(cb_ptr)); width = W)
        hfactor, hstack, hinfo = to_host(dfactor), to_host(dstack), to_host(info)
        for (s, (w, m, cb)) in enumerate(shapes)
            F = copy(fronts[s])
            _, linfo = LAPACK.potrf!('L', view(F, 1:w, 1:w))
            @test hinfo[s] == linfo
            @test hinfo[s] == (s == bad ? wb : 0)
            linfo == 0 || continue
            if m > 0
                BLAS.trsm!('R', 'L', 'C', 'N', one(T), view(F, 1:w, 1:w), view(F, (w + 1):(w + m), 1:w))
                F22 = view(F, (w + 1):(w + m), (w + 1):(w + m))
                T <: Complex ? BLAS.herk!('L', 'N', -one(real(T)), view(F, (w + 1):(w + m), 1:w), one(real(T)), F22) :
                BLAS.syrk!('L', 'N', -one(T), view(F, (w + 1):(w + m), 1:w), one(T), F22)
            end
            P = reshape(hfactor[front_ptr[s]:(front_ptr[s + 1] - 1)], w + m, w)
            Lref = tril(F[:, 1:w])
            @test maximum(abs, P - Lref) <= panel_tol(T) * maximum(abs, Lref)
            if cb
                C = unpack_lower(hstack[cb_ptr[s]:(cb_ptr[s] + m * (m + 1) ÷ 2 - 1)], m)
                Cref = tril(F[(w + 1):end, (w + 1):end])
                @test maximum(abs, C - Cref) <= panel_tol(T) * maximum(abs, Cref)
                @test all(k -> isreal(C[k, k]), 1:m)
            end
        end
    end
    @test thrown(() -> SDS.front_cholesky!(zeros(T, 1), zeros(T, 0), zeros(Int32, 1), [1], 1, 1, [1, 2], [1], [1],
                                           [0]; width = 128)) isa InvalidValueError
end

@testset "panels, solves, determinism ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    Random.seed!(666)
    for (name, A, opts) in numeric_b_matrices(T), INT in (name == "laplacian2d(40,40)" ? INTTYPES : (Int32,))
        @testset "$name $INT" begin
            S, Nr, info_ref, Sd, Nd, nz = numeric_setup(backend, A, INT; opts)
            @test nregime(S, SDS.REGIME_B) > 0
            opts === NUMERIC_BC_OPTS && @test nregime(S, SDS.REGIME_C) > 0
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

@testset "factorization_alg algo1 vs algo2 ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    Random.seed!(666)
    for A in (laplacian2d(T, 40, 40), random_spd(T, 500, 0.01))
        b = rand(T, size(A, 1), 2)
        x = map(("algo1", "algo2")) do alg
            S, Nr, _, Sd, Nd, nz = numeric_setup(backend, A; opts = Options(subtree_budgets = Int[],
                                                                               factorization_alg = alg))
            @test SDS.factorize!(Nd, Sd, nz) == 0
            Nh = SDS.host_numeric(Nd)
            @test panel_error(Nh, Nr) <= panel_tol(T)
            xa = SDS.ref_solve!(similar(b), S, Nh, b)
            @test relres(A, xa, b) <= tol(T)
            xa
        end
        @test norm(x[1] - x[2]) <= tol(T) * norm(x[2])
    end
end

@testset "views, index bases, refactorization ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    Random.seed!(666)
    A = random_spd(T, 300, 0.02)
    for opts in (NUMERIC_B_OPTS, NUMERIC_BC_OPTS)
        S, Nr, _, Sd, Nd, nz = numeric_setup(backend, A; opts)
        @test nregime(S, SDS.REGIME_B) > 0
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
            numeric_b_allocated(Nd, Sd, nz)
            @test numeric_b_allocated(Nd, Sd, nz) <= numeric_alloc_budget(S, T)
        end
    end
end

@testset "info: first non-positive pivot ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    Random.seed!(666)
    n, j = 60, 23
    for opts in (NUMERIC_B_OPTS, Options(subtree_budgets = Int[], reordering_alg = "algo5"),
                 Options(subtree_budgets = Int[], use_superpanels = 0), NUMERIC_BC_OPTS)
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
            @test S.schedule.regime[s] == SDS.REGIME_B  # the failing front is a fused one
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
    opts = Options(subtree_budgets = Int[], reordering_alg = "algo5")
    _, _, info_ref, Sd, Nd, nz = numeric_setup(backend, A; opts)
    @test info_ref == 4
    @test SDS.factorize!(Nd, Sd, nz) == 4
end
