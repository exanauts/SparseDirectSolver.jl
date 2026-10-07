# T12: the solve phase on the device: permutation kernels, forward sweep
# (atomic and deterministic variants), diagonal hook, backward sweep, with
# regimes A, B and C active, multiple right-hand sides and both RHS layouts.

# small budgets and regime-C thresholds: regimes A, B and C on every T09 matrix
const SOLVE_OPTS = Options(subtree_budgets = [8192, 16384], regime_c_width = 16, regime_c_rows = 128)

solve_matrices(::Type{T}) where {T} =
    (("laplacian2d(40,40)", laplacian2d(T, 40, 40)), ("random_spd(500,0.01)", random_spd(T, 500, 0.01)),
     ("laplacian3d(10,10,10)", laplacian3d(T, 10, 10, 10)))

solve_rhs(::Type{T}, n, nrhs) where {T} = nrhs == 1 ? rand(T, n) : rand(T, n, nrhs)

nregime(S, r) = count(==(r), S.schedule.regime)

solve_allocated(x, ws, S, N, b; kwargs...) = @allocated SDS.sweep_solve!(x, ws, S, N, b; kwargs...)

RUN_SHARED && @testset "solve plan and workspace" begin
    for (name, A) in solve_matrices(Float64)
        C = SDS.CSR(tril(A))
        S = SDS.symbolic_analysis(C, "SPD", 'L'; opts = SOLVE_OPTS)
        sc = S.schedule
        plan = SDS.SolvePlan(S)
        # one regime-A launch first, then one launch per (step, kind); every front exactly once
        @test plan.kind[1] == SDS.SOLVE_SUBTREES && count(==(SDS.SOLVE_SUBTREES), plan.kind) == 1
        @test sort(sc.group_nodes[plan.first[1]:plan.last[1]]) == 1:SDS.nsubtrees(sc)
        fronts = Int[]
        for k in 2:length(plan.kind)
            nodes = sc.group_nodes[plan.first[k]:plan.last[k]]
            @test length(unique(sc.step[nodes])) == 1
            @test all(s -> SDS.takes_c_path(sc, s) == (plan.kind[k] == SDS.SOLVE_DENSE), nodes)
            append!(fronts, nodes)
        end
        @test sort(vcat(fronts, sc.subtree_nodes)) == 1:SDS.nsupernodes(S)
        @test plan.maxm == maximum(s -> SDS.takes_c_path(sc, s) ? sc.rows[s] - sc.width[s] : 0, 1:SDS.nsupernodes(S))
        ws = SDS.allocate_solve(S, Float64, CPU(), 3)
        @test size(ws.Y) == (S.n, 3) && size(ws.U) == (length(S.partition.rowval), 3) && size(ws.tmp) == (plan.maxm, 3)
        @test SDS.max_rhs(ws) == 3 && ws.atomic == SDS.capabilities(CPU(), Float64).atomic_add
        @test SDS.solve_memory(S, Float64, 3) == sizeof(Float64) * (length(ws.Y) + length(ws.U) + length(ws.tmp))
    end
    S = SDS.symbolic_analysis(SDS.CSR(tril(laplacian2d(Float64, 5, 5))), "SPD", 'L')
    @test thrown(() -> SDS.allocate_solve(S, Float64, CPU(), 0)) isa InvalidValueError
end

@testset "permutation kernels ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    Random.seed!(666)
    n, nrhs = 37, 4
    perm = randperm(n)
    pd = to_device(backend, perm)
    B = rand(T, n, nrhs)
    Y = KernelAbstractions.zeros(backend, T, n, nrhs + 1)
    SDS.permute_rhs!(Y, to_device(backend, B), pd)
    @test to_host(Y)[:, 1:nrhs] == B[perm, :]
    # transposed (row-major) data and a strided vector give the same permuted matrix
    fill!(Y, zero(T))
    SDS.permute_rhs!(Y, to_device(backend, permutedims(B)), pd; transposed = true)
    @test to_host(Y)[:, 1:nrhs] == B[perm, :]
    fill!(Y, zero(T))
    SDS.permute_rhs!(Y, to_device(backend, vec(B)), pd)
    @test to_host(Y)[:, 1:nrhs] == B[perm, :]
    fill!(Y, zero(T))
    SDS.permute_rhs!(Y, to_device(backend, vec(permutedims(B))), pd; transposed = true)
    @test to_host(Y)[:, 1:nrhs] == B[perm, :]
    # and back
    for transposed in (false, true)
        X = to_device(backend, zeros(T, transposed ? (nrhs, n) : (n, nrhs)))
        SDS.unpermute_solution!(X, Y, pd; transposed)
        @test to_host(X) == (transposed ? permutedims(B) : B)
    end
    @test SDS.rhs_count(zeros(T, n), n) == 1 && SDS.rhs_count(zeros(T, 3n), n) == 3
    @test SDS.rhs_count(zeros(T, 2, n), n; transposed = true) == 2
    @test thrown(() -> SDS.rhs_count(zeros(T, n + 1), n)) isa DimensionMismatch
    @test thrown(() -> SDS.rhs_count(zeros(T, n, 2), n; transposed = true)) isa DimensionMismatch
    @test thrown(() -> SDS.permute_rhs!(Y, to_device(backend, zeros(T, n, nrhs + 2)), pd)) isa DimensionMismatch
end

@testset "residuals, all regimes ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    for (name, A) in solve_matrices(T), INT in (name == "laplacian2d(40,40)" ? INTTYPES : (Int32,))
        @testset "$name $INT" begin
            S, Sd, Nd, ws = solve_setup(backend, A, INT; opts = SOLVE_OPTS)
            @test nregime(S, SDS.REGIME_A) > 0 && nregime(S, SDS.REGIME_B) > 0 && nregime(S, SDS.REGIME_C) > 0
            Nh = SDS.host_numeric(Nd)
            for nrhs in (1, 2, 5), deterministic in (false, true)
                b = solve_rhs(T, size(A, 1), nrhs)
                x = device_solve(backend, ws, Sd, Nd, b; deterministic)
                @test relres(A, x, b) <= tol(T)
                # the same solution as the reference sweeps on the copied-back factor
                xr = SDS.ref_solve!(similar(b), S, Nh, b)
                @test norm(x - xr) <= tol(T) * norm(xr)
            end
        end
    end
end

@testset "forward and backward sweeps against L ($(backend_name(backend)), $T)" for backend in BACKENDS,
                                                                                       T in ELTYPES
    Random.seed!(666)
    A = laplacian3d(T, 10, 10, 10)
    S, Sd, Nd, ws = solve_setup(backend, A; opts = SOLVE_OPTS, nrhs = 2)
    L = SDS.extract_L(S, SDS.host_numeric(Nd))
    perm = S.partition.perm
    b = rand(T, size(A, 1), 2)
    for deterministic in (false, true)
        SDS.permute_rhs!(ws.Y, to_device(backend, b), Sd.perm)
        @test to_host(ws.Y) == b[perm, :]
        SDS.forward_sweep!(ws, Sd, Nd; deterministic)
        z = to_host(ws.Y)
        zr = LowerTriangular(Matrix(L)) \ b[perm, :]
        @test norm(z - zr) <= tol(T) * norm(zr)
        SDS.diagonal_sweep!(ws, Sd, Nd)                 # identity for Cholesky
        @test to_host(ws.Y) == z
        SDS.backward_sweep!(ws, Sd, Nd)
        y = to_host(ws.Y)
        yr = LowerTriangular(Matrix(L))' \ z
        @test norm(y - yr) <= tol(T) * norm(yr)
    end
    # the first nrhs columns only
    fill!(ws.Y, zero(T))
    SDS.permute_rhs!(ws.Y, to_device(backend, b[:, 1]), Sd.perm)
    SDS.forward_sweep!(ws, Sd, Nd; nrhs = 1)
    @test iszero(to_host(ws.Y)[:, 2])
end

@testset "deterministic and atomic variants ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    for (name, A) in solve_matrices(T)
        S, Sd, Nd, ws = solve_setup(backend, A; opts = SOLVE_OPTS)
        @test ws.atomic == SDS.capabilities(backend, T).atomic_add
        T <: Complex && @test !ws.atomic                 # no complex atomics (issue #36): always deterministic
        b = solve_rhs(T, size(A, 1), 5)
        xd = device_solve(backend, ws, Sd, Nd, b; deterministic = true)
        xa = device_solve(backend, ws, Sd, Nd, b; deterministic = false)
        @test norm(xa - xd) <= variant_tol(T) * norm(xd)
        ws.atomic || @test xa == xd
        # the deterministic variant is bitwise reproducible
        @test device_solve(backend, ws, Sd, Nd, b; deterministic = true) == xd
        @test relres(A, xa, b) <= tol(T) && relres(A, xd, b) <= tol(T)
    end
end

@testset "right-hand-side layouts and columns ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    Random.seed!(666)
    A = random_spd(T, 500, 0.01)
    S, Sd, Nd, ws = solve_setup(backend, A; opts = SOLVE_OPTS)
    n = size(A, 1)
    b = rand(T, n, 5)
    x = device_solve(backend, ws, Sd, Nd, b; deterministic = true)
    # transposed (row-major) layout and the strided vector: the same solution
    @test permutedims(device_solve(backend, ws, Sd, Nd, permutedims(b); deterministic = true, transposed = true)) == x
    @test device_solve(backend, ws, Sd, Nd, vec(b); deterministic = true) == vec(x)
    xt = permutedims(device_solve(backend, ws, Sd, Nd, permutedims(b); transposed = true))
    @test norm(xt - x) <= variant_tol(T) * norm(x)
    # in place (X === B)
    bd = to_device(backend, b)
    SDS.sweep_solve!(bd, ws, Sd, Nd, bd; deterministic = true)
    @test to_host(bd) == x
    # the n × nrhs matrix equals the columns solved one at a time (vector and n × 1 matrix)
    for r in 1:5
        xr = device_solve(backend, ws, Sd, Nd, b[:, r]; deterministic = true)
        @test norm(xr - x[:, r]) <= variant_tol(T) * norm(x[:, r])
        @test device_solve(backend, ws, Sd, Nd, b[:, r:r]; deterministic = true) == reshape(xr, n, 1)
    end
    # more right-hand sides than the workspace holds, mismatched X
    @test thrown(() -> SDS.sweep_solve!(similar(bd, n, 6), ws, Sd, Nd, to_device(backend, rand(T, n, 6)))) isa
          InvalidValueError
    @test thrown(() -> SDS.sweep_solve!(similar(bd, n, 2), ws, Sd, Nd, bd)) isa DimensionMismatch
    # a workspace of another analysis
    S2, Sd2, Nd2, ws2 = solve_setup(backend, laplacian2d(T, 6, 6))
    @test thrown(() -> SDS.sweep_solve!(similar(bd), ws2, Sd, Nd, bd)) isa InvalidValueError
end

@testset "schedules and dense implementations ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    Random.seed!(666)
    A = laplacian3d(T, 8, 8, 8)
    b = rand(T, size(A, 1), 2)
    for opts in (Options(), Options(subtree_budgets = Int[]), Options(factorization_alg = "algo1", regime_c_width = 16),
                 Options(factorization_alg = "algo2", subtree_budgets = Int[]))
        S, Sd, Nd, ws = solve_setup(backend, A; opts, nrhs = 2)
        for deterministic in (false, true)
            @test relres(A, device_solve(backend, ws, Sd, Nd, b; deterministic), b) <= tol(T)
        end
    end
    # every dense implementation of the regime-C path
    S, Sd, Nd, ws = solve_setup(backend, A; opts = SOLVE_OPTS, nrhs = 2)
    x = device_solve(backend, ws, Sd, Nd, b; deterministic = true)
    for impl in union(SDS.dense_impls(:trsm, backend, T), SDS.dense_impls(:gemm, backend, T))
        impl in SDS.dense_impls(:trsm, backend, T) && impl in SDS.dense_impls(:gemm, backend, T) || continue
        xi = device_solve(backend, ws, Sd, Nd, b; deterministic = true, impl)
        @test norm(xi - x) <= variant_tol(T) * norm(x)
    end
end

@testset "no allocations (CPU, $T)" for T in ELTYPES
    Random.seed!(666)
    A = laplacian2d(T, 40, 40)
    S, Sd, Nd, ws = solve_setup(CPU(), A; opts = SOLVE_OPTS)
    b = rand(T, size(A, 1), 5)
    x = similar(b)
    for deterministic in (false, true)
        SDS.sweep_solve!(x, ws, Sd, Nd, b; deterministic)
        @test solve_allocated(x, ws, Sd, Nd, b; deterministic) <= solve_alloc_budget(Sd, ws)
        @test relres(A, x, b) <= tol(T)
    end
end
