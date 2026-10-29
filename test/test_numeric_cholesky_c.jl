# T09: multifrontal Cholesky on the device with the assembly kernels, every
# front through the regime-C path (dense interface), checked against the T08
# reference panels.

# every front through the level driver: no regime-A subtrees (T11); since T10 the
# regime-C path on every front needs `factorization_alg = "algo2"` (regime B: test_numeric_cholesky_b)
const NUMERIC_C_OPTS = Options(subtree_budgets = Int[], factorization_alg = "algo2")

numeric_c_matrices(::Type{T}) where {T} =
    (("laplacian2d(40,40)", laplacian2d(T, 40, 40)), ("random_spd(500,0.01)", random_spd(T, 500, 0.01)),
     ("laplacian3d(10,10,10)", laplacian3d(T, 10, 10, 10)))

# shared with T10: `numeric_setup`, `panel_error`, `numeric_alloc_budget` (test/utils.jl)
numeric_c_setup(backend, A, INT = Int32; opts = NUMERIC_C_OPTS, kw...) = numeric_setup(backend, A, INT; opts, kw...)

RUN_SHARED && @testset "storage and plan" begin
    A = laplacian2d(Float64, 20, 20)
    S, _, _, _, Nd, _ = numeric_c_setup(CPU(), A)
    @test Nd isa SDS.Numeric{Float64, Vector{Float64}, Vector{Int64}, Vector{Int32}}
    @test length(Nd.info) == SDS.nsupernodes(S) + 1
    @test all(==(SDS.REGIME_C), S.schedule.regime) && all(==(0), Nd.plan.group_width)
    # the plan covers every front exactly once, step by step
    plan = Nd.plan
    sc = S.schedule
    seen = Int[]
    for t in 1:sc.nsteps
        nodes = sc.group_nodes[plan.step_first[t]:plan.step_last[t]]
        @test all(s -> sc.step[s] == t, nodes)
        @test plan.step_maxchild[t] == maximum(s -> count(==(s), S.partition.snparent), nodes; init = 0)
        append!(seen, nodes)
    end
    @test sort(seen) == 1:SDS.nsupernodes(S)
    # regime-A analyses run since T11: the default analysis schedules subtrees here; factorizing them is
    # test_numeric_cholesky_a's job (the former `factorize!` of this default analysis compiled the whole
    # regime-A/B kernel family inside this regime-C file, ~4 minutes per element type; test-suite diet).
    # Wrong sizes are rejected below.
    C = SDS.CSR(tril(A))
    SA = SDS.symbolic_analysis(C, "SPD", 'L')
    @test SDS.nsubtrees(SA.schedule) > 0
    S0 = SDS.symbolic_analysis(C, "SPD", 'L'; opts = NUMERIC_C_OPTS)
    N0 = SDS.allocate_numeric(S0, Float64)
    @test thrown(() -> SDS.factorize!(N0, S0, C.nzval[1:(end - 1)])) isa InvalidValueError
    @test thrown(() -> SDS.factorize!(SDS.allocate_numeric(S0, ComplexF64), S0, ComplexF64.(C.nzval))) isa
          InvalidValueError
    @test SDS.factorize!(N0, S0, C) == 0                 # CSR method
end

@testset "panels, solves, determinism ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    Random.seed!(666)
    for (name, A) in numeric_c_matrices(T), INT in (name == "laplacian2d(40,40)" ? INTTYPES : (Int32,))
        @testset "$name $INT" begin
            S, Nr, info_ref, Sd, Nd, nz = numeric_c_setup(backend, A, INT)
            @test info_ref == 0
            @test SDS.factorize!(Nd, Sd, nz) == 0
            Nh = SDS.host_numeric(Nd)
            F1 = copy(Nh.factor)                         # `host_numeric` aliases host storage
            @test panel_error(Nh, Nr) <= panel_tol(T)
            @test Nh.stats == Nr.stats
            @test SDS.extract_L(Sd, Nd) == SDS.extract_L(S, Nh)
            s = SDS.nsupernodes(S)
            @test SDS.panel(Sd, Nd, s) == reshape(Nh.factor[S.layout.panel_ptr[s]:end], S.schedule.rows[s], :)
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

@testset "dense implementations ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    Random.seed!(666)
    A = laplacian2d(T, 25, 25)
    S, Nr, _, Sd, Nd, nz = numeric_c_setup(backend, A)
    b = rand(T, size(A, 1), 2)
    for impl in SDS.dense_impls(:potrf, backend, T)
        @test SDS.factorize!(Nd, Sd, nz; impl) == 0
        Nh = SDS.host_numeric(Nd)
        F1 = copy(Nh.factor)
        @test panel_error(Nh, Nr) <= panel_tol(T)
        @test relres(A, SDS.ref_solve!(similar(b), S, Nh, b), b) <= tol(T)
        @test SDS.factorize!(Nd, Sd, nz; impl) == 0
        @test SDS.host_numeric(Nd).factor == F1
    end
end

@testset "views, index bases, refactorization ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    Random.seed!(666)
    A = random_spd(T, 300, 0.02)
    S, Nr, _, Sd, Nd, nz = numeric_c_setup(backend, A)
    @test SDS.factorize!(Nd, Sd, nz) == 0
    F0 = copy(SDS.host_numeric(Nd).factor)
    @test maximum(abs, F0 - Nr.factor) <= panel_tol(T) * maximum(abs, Nr.factor)
    for view in ('U', 'F'), index in ('O', 'Z')
        _, _, _, Sv, Nv, nzv = numeric_c_setup(backend, A; view, index)
        @test SDS.factorize!(Nv, Sv, nzv) == 0
        @test SDS.host_numeric(Nv).factor == F0
    end
    # new values on the same pattern and the same Symbolic and storage
    B = copy(A)
    nonzeros(B) .= rand(T, nnz(B))
    B = (B + B') / 2 + 2 * size(B, 1) * I
    @test SparseMatrixCSC(B .!= 0) == SparseMatrixCSC(A .!= 0)
    Cb = SDS.CSR(tril(B))
    S2, Nr2, _, _, _, _ = numeric_c_setup(CPU(), B)
    @test SDS.factorize!(Nd, Sd, to_device(backend, Cb.nzval)) == 0
    Nh = SDS.host_numeric(Nd)
    @test panel_error(Nh, Nr2) <= panel_tol(T)
    b = rand(T, size(B, 1), 2)
    @test relres(B, SDS.ref_solve!(similar(b), S2, Nh, b), b) <= tol(T)
    @test SDS.factorize!(Nd, Sd, nz) == 0                # and back
    @test SDS.host_numeric(Nd).factor == F0
    if backend isa CPU
        # no allocation in the numeric phase (the first call compiles) beyond what the KA CPU backend
        # itself allocates per launch under coverage (`ka_cpu_alloc_budget`)
        numeric_c_allocated(Nd, Sd, nz)
        @test numeric_c_allocated(Nd, Sd, nz) <= numeric_alloc_budget(S, T)
    end
end

@testset "info: first non-positive pivot ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    Random.seed!(666)
    n, j = 60, 23
    for opts in (NUMERIC_C_OPTS,
                 Options(subtree_budgets = Int[], factorization_alg = "algo2", reordering_alg = "algo5"),
                 Options(subtree_budgets = Int[], factorization_alg = "algo2", use_superpanels = 0))
        A = singular_block_matrix(T, n, j; stored_zero = true)
        A[j, j] = 3
        _, Nr, info_ref, Sd, Nd, nz = numeric_c_setup(backend, A; opts)
        @test info_ref == 0
        @test SDS.factorize!(Nd, Sd, nz) == 0
        @test panel_error(SDS.host_numeric(Nd), Nr) <= panel_tol(T)
        for d in (-3, 0)                                # negative and zero pivot at column j
            A[j, j] = d
            S, Nr, info_ref, Sd, Nd, nz = numeric_c_setup(backend, A; opts)
            @test info_ref == j
            # pivot_epsilon = 0: failure REPORTING is under test; the default epsilon's
            # input-diagonal clamp would lift the stored-zero pivot before potrf (#127)
            @test SDS.factorize!(Nd, Sd, nz; opts = Options(pivot_epsilon = 0.0)) == j
            s = S.partition.col2sn[S.partition.iperm[j]]
            stats = to_host(Nd.stats)
            @test stats[s * SDS.FRONT_STATS_FIELDS] == Nr.stats[s * SDS.FRONT_STATS_FIELDS] ==
                  S.partition.iperm[j] - S.partition.super_ptr[s] + 1
        end
    end
    # a non-positive pivot that only appears after elimination: [1 2; 2 1] block at columns 3, 4
    A = sparse(T[4 0 0 0; 0 4 0 0; 0 0 1 2; 0 0 2 1])
    opts = Options(subtree_budgets = Int[], factorization_alg = "algo2", reordering_alg = "algo5")
    _, _, info_ref, Sd, Nd, nz = numeric_c_setup(backend, A; opts)
    @test info_ref == 4
    @test SDS.factorize!(Nd, Sd, nz) == 4
end

# a failing pivot inside a regime-C front at least 64 wide: on Metal the offset-aware MPS Cholesky
# (`_offset_potrf_plan`) takes such a front under `impl = :auto`, and its status is folded into `info`;
# MPS names no column, so there `info` is only required to fall into the failing front
@testset "info: failing pivot in a wide front ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    Random.seed!(666)
    n, j = 96, 70
    A = sparse(dense_hpd(T, n))
    S, Nr, info_ref, Sd, Nd, nz = numeric_c_setup(backend, A)
    @test info_ref == 0
    @test SDS.factorize!(Nd, Sd, nz) == 0
    @test panel_error(SDS.host_numeric(Nd), Nr) <= panel_tol(T)
    for d in (-1, 0)
        A[j, j] = d
        S, Nr, info_ref, Sd, Nd, nz = numeric_c_setup(backend, A)
        sc, P = S.schedule, S.partition
        @test info_ref > 0
        s = P.col2sn[P.iperm[info_ref]]
        @test SDS.takes_c_path(sc, s) && sc.width[s] >= 64
        info = SDS.factorize!(Nd, Sd, nz; opts = Options(pivot_epsilon = 0.0))
        @test info > 0 && P.col2sn[P.iperm[info]] == s
        backend_name(backend) == "Metal" || @test info == info_ref
    end
end

# the vendor potrf of a backend (MPS on Metal) is an `impl = :auto` decision: `impl = :ka` and
# `factorization_alg = "algo1"` (no vendor calls, PLAN §1.3) keep the KA kernels
@testset "offset-aware vendor potrf only under :auto ($(backend_name(backend)), $T)" for backend in BACKENDS,
                                                                                         T in ELTYPES
    A = sparse(dense_hpd(T, 96))
    _, _, _, Sd, Nd, _ = numeric_c_setup(backend, A)
    @test SDS._front_impls(Nd, Sd, :auto).mps == SDS._offset_potrf_enabled(backend)
    @test !SDS._front_impls(Nd, Sd, :ka).mps
    _, _, _, Sd1, Nd1, nz1 = numeric_c_setup(backend, A; opts = Options(subtree_budgets = Int[],
                                                                      factorization_alg = "algo1",
                                                                      regime_c_width = 16))
    @test !Sd1.schedule.vendor_c && !SDS._front_impls(Nd1, Sd1, :auto).mps
    @test all(isnothing, Nd1.plan.vendor_ptrs)
    @test SDS.factorize!(Nd1, Sd1, nz1) == 0
end
