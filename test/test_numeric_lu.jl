# T19: general LU (structure "G"): the CPU reference `ref_lu!` (multifrontal L D U on the pattern of A + Aᵀ,
# row pivoting inside the fully-summed block, static perturbation), the device kernels of regimes A/B/C against
# it, the solves with A and Aᵀ, uniform batches and the handle / LinearAlgebra layers.

# regime mixes of the analysis (merged into each case's options)
const LU_REGIMES = (("default", (;)),
                    ("A+B+C, small budgets", (subtree_budgets = [8192, 16384], regime_c_width = 16,
                                              regime_c_rows = 128)),
                    ("C only", (factorization_alg = "algo2", subtree_budgets = Int[])))

# (name, matrix, option keywords, whether the in-block pivoting must interchange rows)
function lu_cases(::Type{T}) where {T}
    return (("random_general(400,0.01)", random_general(T, 400, 0.01), (;), false),
            ("random_general(300,0.02), natural order", random_general(T, 300, 0.02), (reordering_alg = "algo5",),
             false),
            ("weak diagonal", weak_diagonal_general(T, 300, 0.02), (;), false),
            ("weak diagonal, pivot_threshold = 1", weak_diagonal_general(T, 300, 0.02), (pivot_threshold = 1.0,),
             true),
            ("weak diagonal, pivot_type 'N'", weak_diagonal_general(T, 300, 0.02),
             (pivot_type = 'N', pivot_threshold = 1.0), false))
end

@testset "reference LU ($T)" for T in ELTYPES
    Random.seed!(666)
    for (name, A, kw, swaps) in lu_cases(T)
        @testset "$name" begin
            opts = Options(; kw...)
            S, Nr, _, _, _ = lu_setup(CPU(), A; opts)
            n = size(A, 1)
            @test lu_error(A, S, Nr) <= tol(T)
            # the local row order permutes the columns of each supernode
            sp = S.partition
            @test all(s -> sort(Int.(Nr.piv[SDS.sncols(sp, s)])) == SDS.sncols(sp, s), 1:SDS.nsupernodes(sp))
            swaps && @test Nr.piv != 1:n
            haskey(kw, :pivot_type) && @test Nr.piv == 1:n
            @test all(==(SDS.PIVOT_KIND_1X1), Nr.pivot_kind)
            @test SDS.pivot_stats(Nr) == (npos = 0, nneg = 0, nzero = 0, nperturbed = 0, n2x2 = 0)
            for transpose in (false, true)
                b = rand(T, n, 2)
                x = similar(b)
                SDS.ref_solve_lu!(x, S, Nr, b; transpose)
                @test relres(transpose ? Base.transpose(A) : A, x, b) <= tol(T)
            end
        end
    end
end

@testset "device LU equals the reference ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    Random.seed!(666)
    for (name, A, kw, _) in lu_cases(T), (rname, rkw) in LU_REGIMES
        @testset "$name, $rname" begin
            opts = Options(; kw..., rkw...)
            S, Nr, Sd, Nd, nz = lu_setup(backend, A; opts)
            rname == "C only" && @test all(s -> SDS.takes_c_path(S.schedule, s), 1:SDS.nsupernodes(S))
            @test SDS.factorize!(Nd, Sd, nz; opts) == 0
            Nh = SDS.host_numeric(Nd)
            # the same pivot sequence: local row orders and kinds equal, D, L and Uᵀ to rounding
            @test Nh.piv == Nr.piv
            @test Nh.pivot_kind == Nr.pivot_kind
            @test d_error(Nh, Nr) <= growth_tol(T, Nr)
            @test panel_error(Nh, Nr) <= growth_tol(T, Nr)
            @test upanel_error(Nh, Nr) <= growth_tol(T, Nr)
            # the backward error of the device factor: A[p, q] = L D U
            @test lu_error(A, S, Nh) <= tol(T)
            @test Nh.stats == Nr.stats
            @test SDS.pivot_totals(Nd) == SDS.pivot_stats(Nr) && Nh.totals == Nr.totals
            # device solves with A (L, D, U) and Aᵀ (Uᵀ, D, Lᵀ)
            ws = SDS.allocate_solve(Sd, T, backend, 5)
            for nrhs in (1, 5), det in (false, true), transpose in (false, true)
                b = nrhs == 1 ? rand(T, size(A, 1)) : rand(T, size(A, 1), nrhs)
                x = device_solve(backend, ws, Sd, Nd, b; deterministic = det, transpose)
                @test relres(transpose ? Base.transpose(A) : A, x, b) <= tol(T)
            end
        end
    end
end

@testset "regime-A groups and Int64 maps ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    Random.seed!(666)
    A = weak_diagonal_general(T, 400, 0.01)
    opts = Options(pivot_threshold = 1.0, subtree_budgets = [8192, 16384, 32768, 49152])
    # the 64 KiB regime-A class (issue #60) where the backend has the local memory (CPU, ROCm)
    if SDS.max_local_bytes(backend) >= 65536
        o64 = Options(pivot_threshold = 1.0, subtree_budgets = [65536], subtree_parallelism = 0)
        S, Nr, Sd, Nd, nz = lu_setup(backend, A, Int32; opts = o64)
        @test 65536 in Nd.plan.sub_local
        SDS.factorize!(Nd, Sd, nz; opts = o64)
        Nh = SDS.host_numeric(Nd)
        @test Nh.piv == Nr.piv && Nh.pivot_kind == Nr.pivot_kind
        gt = growth_tol(T, Nr)
        @test d_error(Nh, Nr) <= gt && panel_error(Nh, Nr) <= gt && upanel_error(Nh, Nr) <= gt
    end
    S, Nr, Sd, Nd, nz = lu_setup(backend, A, Int64; opts)
    @test !isempty(Nd.plan.sub_first) && Nr.piv != 1:400
    SDS.factorize!(Nd, Sd, nz; opts)
    Nh = SDS.host_numeric(Nd)
    @test Nh.piv == Nr.piv && Nh.pivot_kind == Nr.pivot_kind
    gt = growth_tol(T, Nr)
    @test d_error(Nh, Nr) <= gt && panel_error(Nh, Nr) <= gt && upanel_error(Nh, Nr) <= gt
    ws = SDS.allocate_solve(Sd, T, backend, 2)
    b = rand(T, 400, 2)
    @test relres(A, device_solve(backend, ws, Sd, Nd, b), b) <= tol(T)
    # the extracted factors: A[p, q] = L D U
    @test lu_error(A, S, Nh) <= tol(T)
end

@testset "pivot_epsilon = 0: a zero pivot fails ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    Random.seed!(666)
    n, j = 60, 23
    F = SDS.FRONT_STATS_FIELDS
    # an exactly zero pivot is tiny whatever ε; with ε = 0 it stays zero and info is its original column (#65)
    for (rname, rkw) in (("default", (;)), ("regime-C root", (factorization_alg = "algo2", subtree_budgets = Int[]))),
        stored_zero in (false, true)
        A = singular_block_matrix(T, n, j; stored_zero)
        opts = Options(; pivot_epsilon = 0.0, rkw...)
        C = SDS.CSR(A)
        S = SDS.symbolic_analysis(C, "G", 'F'; opts)
        Nr = SDS.allocate_numeric(S, T)
        @test SDS.ref_lu!(Nr, S, C.nzval; opts) == j
        Sd = SDS.adapt(backend, S, Int32)
        Nd = SDS.allocate_numeric(Sd, T, backend)
        @test SDS.factorize!(Nd, Sd, to_device(backend, C.nzval); opts) == j
        Nh = SDS.host_numeric(Nd)
        @test Nh.pivot_kind == Nr.pivot_kind && Nh.piv == Nr.piv
        @test d_error(Nh, Nr) <= panel_tol(T)
        @test to_host(Nd.stats)[F:F:end] == Nr.stats[F:F:end]
        @test count(!iszero, Nr.stats[F:F:end]) == 1
        st = SDS.pivot_totals(Nd)
        @test st.nzero == 1 && st.nperturbed == 1
    end
    solver = DirectSolver(api_matrix(backend, singular_block_matrix(T, n, j)), "G", 'F')
    setparam!(solver, "pivot_epsilon", 0.0)
    execute!("analysis", solver, nothing, nothing)
    execute!("factorization", solver, nothing, nothing)
    @test getparam(solver, "info") == j
end

@testset "perturbation ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    Random.seed!(666)
    n, j = 60, 23
    R = real(T)
    # a zero row and column j: no row of the front can replace the zero pivot, it is perturbed
    for (rname, rkw) in (("default", (;)), ("regime-C root", (factorization_alg = "algo2", subtree_budgets = Int[]))),
        stored_zero in (false, true)
        A = singular_block_matrix(T, n, j; stored_zero)
        opts = Options(; rkw...)
        S, Nr, Sd, Nd, nz = lu_setup(backend, A; opts)
        @test SDS.factorize!(Nd, Sd, nz; opts) == 0
        Nh = SDS.host_numeric(Nd)
        @test Nh.pivot_kind == Nr.pivot_kind && Nh.piv == Nr.piv
        @test d_error(Nh, Nr) <= panel_tol(T)
        st = SDS.pivot_totals(Nd)
        @test st.nperturbed == 1 && st.nzero == 1
        k = findfirst(==(SDS.PIVOT_KIND_PERTURBED), Nh.pivot_kind)
        _, _, _, p, q = SDS.extract_lu(S, Nh)
        @test k !== nothing && p[k] == j && q[k] == j
        @test Nh.d[k] == R(SDS.default_pivot_epsilon(R))
        x0 = rand(T, n)
        x0[j] = 0
        b = A * x0
        ws = SDS.allocate_solve(Sd, T, backend, 1)
        @test relres(A, device_solve(backend, ws, Sd, Nd, b), b) <= 1.0e-6
        @test relres(Base.transpose(A), device_solve(backend, ws, Sd, Nd, b; transpose = true), b) <= 1.0e-6
    end
    # pivot_epsilon and the scaled variant (ε max|aᵢⱼ| on the device)
    A = singular_block_matrix(T, n, j)
    for opts in (Options(pivot_epsilon = 1.0e-3), Options(pivot_epsilon = 1.0e-3, pivot_epsilon_alg = "algo1"))
        S, Nr, Sd, Nd, nz = lu_setup(backend, A; opts)
        SDS.factorize!(Nd, Sd, nz; opts)
        Nh = SDS.host_numeric(Nd)
        @test Nh.pivot_kind == Nr.pivot_kind && Nh.piv == Nr.piv && d_error(Nh, Nr) <= panel_tol(T)
        pert = Nr.pivot_kind .== SDS.PIVOT_KIND_PERTURBED
        @test Nh.d[1:n][pert] == Nr.d[1:n][pert]
        scale = opts.pivot_epsilon_alg == SDS.PIVOT_EPSILON_SCALED ? maximum(abs, A) : 1
        @test Nh.d[1:n][pert] == [R(R(1.0e-3) * R(scale))]
    end
    # global pivoting is not planned
    @test thrown(() -> Options(pivot_type = 'C')) isa NotSupportedError
end

@testset "determinism and refactorization ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    Random.seed!(666)
    A = weak_diagonal_general(T, 300, 0.02)
    opts = Options(pivot_threshold = 1.0, subtree_budgets = [8192, 16384], regime_c_width = 16, regime_c_rows = 128)
    S, Nr, Sd, Nd, nz = lu_setup(backend, A; opts)
    SDS.factorize!(Nd, Sd, nz; opts)
    factor, ufactor, d, piv = to_host(Nd.factor), to_host(Nd.ufactor), to_host(Nd.d), to_host(Nd.piv)
    # new values, then the old ones again: bitwise the same factor
    nz2 = to_device(backend, 2 .* to_host(nz))
    SDS.factorize!(Nd, Sd, nz2; opts)
    @test to_host(Nd.d) ≈ 2 .* d
    SDS.factorize!(Nd, Sd, nz; opts)
    @test to_host(Nd.factor) == factor && to_host(Nd.ufactor) == ufactor && to_host(Nd.d) == d &&
          to_host(Nd.piv) == piv
    ws = SDS.allocate_solve(Sd, T, backend, 2)
    b = rand(T, 300, 2)
    for transpose in (false, true)
        @test device_solve(backend, ws, Sd, Nd, b; deterministic = true, transpose) ==
              device_solve(backend, ws, Sd, Nd, b; deterministic = true, transpose)
    end
    # no allocation in the numeric phase and the solve (CPU backend)
    if backend isa CPU
        SDS.factorize!(Nd, Sd, nz; opts)
        @test (@allocated SDS.factorize!(Nd, Sd, nz; opts)) <= ldlt_alloc_budget(S)
        x = similar(b)
        for transpose in (false, true)
            SDS.sweep_solve!(x, ws, Sd, Nd, b; transpose)
            @test (@allocated SDS.sweep_solve!(x, ws, Sd, Nd, b; transpose)) <= solve_alloc_budget(S, ws; ldlt = true)
        end
    end
end

@testset "solve sweeps: one at a time and dense path ($(backend_name(backend)), $T)" for backend in BACKENDS,
                                                                                      T in ELTYPES
    Random.seed!(666)
    A = weak_diagonal_general(T, 300, 0.02)
    for (rname, rkw) in LU_REGIMES
        opts = Options(; pivot_threshold = 1.0, rkw...)
        S, Nr, Sd, Nd, nz = lu_setup(backend, A; opts)
        SDS.factorize!(Nd, Sd, nz; opts)
        ws = SDS.allocate_solve(Sd, T, backend, 3)
        b = rand(T, 300, 3)
        bd = to_device(backend, b)
        L, D, U, _, _ = SDS.extract_lu(S, SDS.host_numeric(Nd))
        piv = Int.(to_host(Nd.piv))
        for transpose in (false, true)
            # Y = P b; A: L Z = Y[piv] (local row orders), D W = Z, U V = W; Aᵀ: Uᵀ Z = Y, D W = Z, Lᵀ V = W
            SDS.permute_rhs!(ws.Y, bd, Sd.perm)
            Y = to_host(ws.Y)
            SDS.forward_sweep!(ws, Sd, Nd; deterministic = true, transpose)
            Z = to_host(ws.Y)
            SDS.diagonal_sweep!(ws, Sd, Nd)
            W = to_host(ws.Y)
            SDS.backward_sweep!(ws, Sd, Nd; transpose)
            V = to_host(ws.Y)
            xd = similar(bd)
            SDS.unpermute_solution!(xd, ws.Y, Sd.perm)
            @test to_host(xd) == device_solve(backend, ws, Sd, Nd, b; deterministic = true, transpose)
            @test relres(transpose ? Base.transpose(A) : A, to_host(xd), b) <= tol(T)
            if transpose
                @test norm(Base.transpose(U) * Z - Y) <= tol(T) * norm(Y)
                @test norm(Base.transpose(L) * V[piv, :] - W) <= tol(T) * norm(W)
            else
                @test norm(L * Z - Y[piv, :]) <= tol(T) * norm(Y)
                @test norm(U * V - W) <= tol(T) * norm(W)
            end
            @test norm(Diagonal(D) * W - Z) <= tol(T) * norm(Z)
            # every dense implementation of the regime-C path
            if rname == "C only"
                for impl in SDS.dense_impls(:trsm, backend, T)
                    impl in SDS.dense_impls(:gemm, backend, T) || continue
                    x = device_solve(backend, ws, Sd, Nd, b; impl, transpose)
                    @test relres(transpose ? Base.transpose(A) : A, x, b) <= tol(T)
                end
            end
        end
    end
end

@testset "uniform batch ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    Random.seed!(666)
    A = weak_diagonal_general(T, 200, 0.03)
    nb, nrhs = 3, 2
    members = batch_members(A, nb)
    solver = DirectSolver(api_batch_matrix(backend, members, 'F'), "G", 'F')
    setparam!(solver, "pivot_threshold", 1.0)
    execute!("analysis", solver, nothing, nothing)
    execute!("factorization", solver, nothing, nothing)
    @test solver.nbatch == nb
    @test getparam(solver, "info") == zeros(Int, nb)
    @test getparam(solver, "npivots") == zeros(Int32, nb)
    B = rand(T, 200, nrhs, nb)
    X = to_device(backend, zeros(T, 200, nrhs, nb))
    execute!("solve", solver, X, to_device(backend, B))
    @test maximum(batch_relres(members, to_host(X), B)) <= tol(T)
    setparam!(solver, "solve_mode", 1)
    execute!("solve", solver, X, to_device(backend, B))
    @test maximum(batch_relres(map(M -> sparse(Base.transpose(M)), members), to_host(X), B)) <= tol(T)
    setparam!(solver, "solve_mode", 0)
    # each member equals its own reference factorization
    for k in 1:nb
        _, Nr, _, _, _ = lu_setup(CPU(), members[k]; opts = solver.options)
        Nk = SDS.member_numeric(solver.numeric, solver.symbolic, k)
        @test Nk.piv == Nr.piv && d_error(Nk, Nr) <= growth_tol(T, Nr) && upanel_error(Nk, Nr) <= growth_tol(T, Nr)
    end
    # ubatch_index: only member 2 is refactorized
    before = [SDS.member_numeric(solver.numeric, solver.symbolic, k) for k in 1:nb]
    setparam!(solver, "ubatch_index", 1)
    update!(solver, api_batch_matrix(backend, [2 .* M for M in members], 'F'))
    execute!("refactorization", solver, nothing, nothing)
    for k in 1:nb
        Nk = SDS.member_numeric(solver.numeric, solver.symbolic, k)
        if k == 2
            @test Nk.d ≈ 2 .* before[k].d
        else
            @test Nk.factor == before[k].factor && Nk.ufactor == before[k].ufactor && Nk.d == before[k].d
        end
    end
    setparam!(solver, "ubatch_index", -1)
end

@testset "public API ($(backend_name(backend)), $T, $INT)" for backend in BACKENDS, T in ELTYPES, INT in INTTYPES
    Random.seed!(666)
    A = weak_diagonal_general(T, 300, 0.02)
    n = size(A, 1)
    At = sparse(Base.transpose(A))
    solver = DirectSolver(api_matrix(backend, A, INT), "G", 'F')
    setparam!(solver, "pivot_threshold", 1.0)
    setparam!(solver, "deterministic_mode", 1)
    b = to_device(backend, rand(T, n, 2))
    x, y = similar(b), similar(b)
    execute!("analysis", solver, x, b)
    @test getparam(solver, "perm_row") == getparam(solver, "perm_col") == getparam(solver, "perm_reorder_row")
    execute!("factorization", solver, x, b)
    @test getparam(solver, "info") == 0
    # the reference factorization of the same analysis
    S = solver.host_symbolic
    Nr = SDS.allocate_numeric(S, T)
    SDS.ref_lu!(Nr, S, SDS.CSR(A).nzval; opts = solver.options)
    @test getparam(solver, "npivots") == 0 && getparam(solver, "npivots") isa INT
    @test getparam(solver, "inertia") == (0, 0)
    @test getparam(solver, "pivot_stats") == SDS.pivot_stats(Nr)
    d = getparam(solver, "diag")
    @test norm(to_host(d) - Nr.d[1:n]) <= panel_tol(T) * norm(Nr.d)
    # perm_row/perm_col: A[perm_row, perm_col] = L D U
    _, _, _, p, q = SDS.extract_lu(S, Nr)
    pr, pc = getparam(solver, "perm_row"), getparam(solver, "perm_col")
    @test pr == p && pc == q && pr != pc
    buf = zeros(INT, n)
    getparam!(buf, solver, "perm_row")
    @test buf == pr
    # solve_mode 0, 1, 2: A, Aᵀ, Aᴴ; the sub-phases compose to "solve"
    for (mode, M) in ((0, A), (1, At), (2, sparse(adjoint(A))))
        setparam!(solver, "solve_mode", mode)
        execute!("solve", solver, x, b)
        @test relres(M, to_host(x), to_host(b)) <= tol(T)
        for phase in ("solve_fwd_perm", "solve_fwd", "solve_diag", "solve_bwd", "solve_bwd_perm")
            execute!(phase, solver, y, b)
        end
        @test to_host(y) == to_host(x)
        # refinement with the residual of op(A)
        setparam!(solver, "ir_n_steps", 2)
        execute!("solve", solver, x, b)
        @test relres(M, to_host(x), to_host(b)) <= tol(T)
        @test getparam(solver, "ir_n_steps") == 2
        if SDS.fgmres_available()
            setparam!(solver, "ir_mode", "fgmres")
            execute!("solve", solver, x, b)
            @test relres(M, to_host(x), to_host(b)) <= tol(T)
            setparam!(solver, "ir_mode", "ir")
        end
        setparam!(solver, "ir_n_steps", 0)
    end
    setparam!(solver, "solve_mode", 0)
    # logabsdet = log|det A| with the sign of det A
    la, sg = logabsdet(solver)
    la_ref, sg_ref = logabsdet(Matrix(A))
    @test la ≈ la_ref rtol = sqrt(eps(real(T)))
    @test sg ≈ sg_ref rtol = sqrt(eps(real(T)))
    # CSC input: the stored CSR is Aᵀ; solve_mode 0 solves A, 1 solves Aᵀ
    Ct = csr_of_transpose(SparseMatrixCSC{T, INT}(A))
    C = CSR(to_device(backend, Ct.rowptr), to_device(backend, Ct.colval), to_device(backend, Ct.nzval), n, n;
            transposed = true)
    cs = DirectSolver(C, "G", 'F')
    execute!("analysis", cs, nothing, nothing)
    execute!("factorization", cs, nothing, nothing)
    for (mode, M) in ((0, A), (1, At), (2, sparse(adjoint(A))))
        setparam!(cs, "solve_mode", mode)
        setparam!(cs, "ir_n_steps", mode)
        execute!("solve", cs, x, b)
        @test relres(M, to_host(x), to_host(b)) <= tol(T)
    end
    # "G" ignores the view (cuDSS, #84): 'L' and 'U' read the full matrix, as 'F'
    for view in ('L', 'U')
        vs = DirectSolver(api_matrix(backend, A, INT), "G", view)
        setparam!(vs, "pivot_threshold", 1.0)
        for phase in ("analysis", "factorization", "solve")
            execute!(phase, vs, x, b)
        end
        @test getparam(vs, "info") == 0 && relres(A, to_host(x), to_host(b)) <= tol(T)
    end
    # the LinearAlgebra layer
    F = lu(SDS.CSR(SparseMatrixCSC{T, INT}(A)))
    @test F.structure == SDS.STRUCTURE_GENERAL && getparam(F, "ir_n_steps") == 2
    backend isa CPU && @test relres(A, F \ to_host(b), to_host(b)) <= tol(T)
    backend isa CPU && @test relres(2 * A, lu!(F, SDS.CSR(SparseMatrixCSC{T, INT}(2 * A))) \ to_host(b), to_host(b)) <=
                             tol(T)
end
