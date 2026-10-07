# T21: matching and scaling (PLAN §1.3 `matching_alg`, §1.4 `perm_matching`,
# `scale_row`/`scale_col`, `inertia`; issues #67 and #71): the host MC64 jobs
# against brute force, the symmetric scaling and the matching-based 2×2 pivot
# pairs, LU on a badly scaled unsymmetric matrix (no perturbed pivot with
# matching, some without), every solve path with matching, and the inertia of
# LDLᵀ/LDLᴴ with matching enabled (the cuDSS defect).

const MATCHING_ALGS = ("algo1", "algo2", "algo3", "algo4", "algo5", "algo6")

# the host matching of a full SparseMatrixCSC
host_matching(A::SparseMatrixCSC, structure, alg; view = 'F') =
    (C = CSR(triangle_view(A, view)); SDS.compute_matching(C.rowptr, C.colval, C.nzval, size(A, 1), structure,
                                                          Options(matching_alg = alg); view))

RUN_SHARED && @testset "MC64 jobs against brute force" begin
    Random.seed!(666)
    for trial in 1:12
        n = 6
        A = sprand(n, n, 0.4) + sparse(randperm(n), 1:n, rand(n) .+ 0.5)   # structurally nonsingular
        W = abs.(Matrix(A))
        perms = [p for p in all_permutations(n) if all(i -> W[i, p[i]] > 0, 1:n)]
        best_min = maximum(p -> minimum(i -> W[i, p[i]], 1:n), perms)
        best_sum = maximum(p -> sum(i -> W[i, p[i]], 1:n), perms)
        best_prod = maximum(p -> prod(i -> W[i, p[i]], 1:n), perms)
        for alg in MATCHING_ALGS
            m = host_matching(A, "G", alg)
            q = m.perm
            @test isperm(q) && all(m.matched)
            @test all(i -> W[i, q[i]] > 0, 1:n)
            alg in ("algo2", "algo3") && @test minimum(i -> W[i, q[i]], 1:n) == best_min
            alg == "algo4" && @test sum(i -> W[i, q[i]], 1:n) ≈ best_sum rtol = 1.0e-12
            if alg in ("algo5", "algo6")
                @test prod(i -> W[i, q[i]], 1:n) ≈ best_prod rtol = 1.0e-12
                M = Diagonal(m.rscale) * W * Diagonal(m.cscale)
                @test maximum(M) <= 1 + 1.0e-12
                @test all(i -> isapprox(M[i, q[i]], 1; rtol = 1.0e-12), 1:n)
            else
                @test m.rscale == ones(n) && m.cscale == ones(n)
            end
            @test m.alg == SDS.MatchingAlg(alg == "algo6" ? 5 : parse(Int, alg[end:end]))
        end
    end
    # structurally singular: an empty column; the matching is completed to a permutation
    A = sparse([1, 2, 3, 3], [1, 1, 2, 3], [1.0, 2.0, 3.0, 4.0], 3, 3)
    for alg in MATCHING_ALGS
        m = host_matching(A, "G", alg)
        @test isperm(m.perm) && count(m.matched) == 2
        @test all(isfinite, m.rscale) && all(>(0), m.rscale) && all(isfinite, m.cscale) && all(>(0), m.cscale)
    end
    @test host_matching(A, "G", "default") === nothing
    # matched_colval relabels the columns, entry_scaling is rᵢ cⱼ per stored entry
    A = badly_scaled_general(Float64, 30, 0.1)
    C = CSR(A)
    m = host_matching(A, "G", "algo5")
    for index in ('O', 'Z')
        cv = index == 'O' ? C.colval : C.colval .- 1
        @test SDS.matched_colval(m, cv, index) == (index == 'O' ? invperm(m.perm)[C.colval] :
                                                   invperm(m.perm)[C.colval] .- 1)
    end
    w = SDS.entry_scaling(m, C.rowptr, C.colval, 30)
    M = Diagonal(m.rscale) * A * Diagonal(m.cscale)
    @test CSR(M).nzval ≈ w .* C.nzval
    @test maximum(abs, M) <= 1 + 1.0e-12
end

RUN_SHARED && @testset "symmetric scaling and matching pairs" begin
    Random.seed!(666)
    nh, nj = 40, 15
    K = kkt_matrix(Float64, nh, nj, 1.0e-10)
    for view in ('L', 'U', 'F')
        m = host_matching(K, "S", "algo5"; view)
        @test m.symmetric && m.rscale == m.cscale && all(m.matched) && isperm(m.perm)
        D = Diagonal(m.rscale)
        @test maximum(abs, D * K * D) <= 1 + 1.0e-12
    end
    # badly scaled: the symmetric scaling equilibrates it
    S = badly_scaled_spd(Float64, 50, 0.1)
    m = host_matching(S, "SPD", "algo5")
    D = Diagonal(m.rscale)
    @test maximum(abs, D * S * D) <= 1 + 1.0e-12
    @test minimum(abs, diag(D * S * D)) > 1.0e-3
    # 2×2 pairs from the cycles of the matching (issue #67)
    m = host_matching(K, "S", "algo5"; view = 'L')
    C = CSR(tril(K))
    pairs_of(opts) = SDS.matching_pairs(m, C.rowptr, C.colval, C.nzval, nh + nj, "S", opts; view = 'L')
    def = pairs_of(Options(matching_alg = "algo5"))
    all_ = pairs_of(Options(matching_alg = "algo5", pivot_pairs = "all"))
    @test !isempty(def) && length(def) <= length(all_)
    @test isempty(pairs_of(Options(matching_alg = "algo5", pivot_pairs = "none")))
    for pairs in (def, all_)
        idx = reduce(vcat, ([a, b] for (a, b) in pairs))
        @test allunique(idx)                                      # disjoint
        @test all(((a, b),) -> K[a, b] != 0 && a != b, pairs)     # matrix entries
        @test issorted(pairs; by = last)
    end
    # "default" pairs contain a candidate (a dual row, diagonal -1e-10) as candidate
    @test all(((a, b),) -> b > nh, def)
    # the candidate tolerance is a parameter: τ = 0 leaves no candidate (the diagonal is -1e-10, not 0)
    @test isempty(pairs_of(Options(matching_alg = "algo5", pivot_pair_tolerance = 0)))
    P = SDS.SymmetricPattern(C, "S"; view = 'L')
    @test count(SDS.pivot_candidates(P, C, "S"; view = 'L').candidate) == nj
    @test count(SDS.pivot_candidates(P, C, "S"; view = 'L', tolerance = 0).candidate) == 0
end

@testset "LU with matching: badly scaled ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    Random.seed!(666)
    n = 80
    INT = Int32
    A = badly_scaled_general(T, n, 0.05)
    b = to_device(backend, rand(T, n))
    x = to_device(backend, zeros(T, n))
    # without matching: zero diagonals, static pivoting perturbs
    plain = DirectSolver(api_matrix(backend, A, INT), "G", 'F')
    execute!("analysis", plain, x, b)
    execute!("factorization", plain, x, b)
    @test getparam(plain, "pivot_stats").nperturbed > 0
    @test thrown(() -> getparam(plain, "perm_matching")) isa InvalidValueError
    # with matching (job 5): no perturbed pivot, the solution to tolerance
    solver = DirectSolver(api_matrix(backend, A, INT), "G", 'F')
    setparam!(solver, "matching_alg", "algo5")
    @test thrown(() -> getparam(solver, "scale_row")) isa FactorizationError
    execute!("analysis", solver, x, b)
    execute!("factorization", solver, x, b)
    @test getparam(solver, "pivot_stats").nperturbed == 0
    @test getparam(solver, "npivots") == 0
    execute!("solve", solver, x, b)
    @test relres(A, to_host(x), to_host(b)) <= tol(T)
    q = getparam(solver, "perm_matching")
    r, c = getparam(solver, "scale_row"), getparam(solver, "scale_col")
    @test isperm(q) && eltype(r) == real(T) && eltype(c) == real(T)
    M = Diagonal(r) * A * Diagonal(c)
    @test maximum(abs, M) <= 1 + 10 * eps(real(T))
    @test all(i -> isapprox(abs(M[i, q[i]]), 1; rtol = 10 * eps(real(T))), 1:n)
    # the final permutations: perm_col composes the matching with the reordering
    @test getparam(solver, "perm_col") == q[getparam(solver, "perm_reorder_col")]
    @test isperm(getparam(solver, "perm_row"))
    # the sub-phases compose to "solve" bitwise; both under the deterministic forward sweep
    # (the atomic one reorders sums between runs on GPUs)
    setparam!(solver, "deterministic_mode", 1)
    execute!("solve", solver, x, b)
    x2 = to_device(backend, zeros(T, n))
    for phase in SOLVE_SUBPHASES
        execute!(phase, solver, x2, b)
    end
    setparam!(solver, "deterministic_mode", 0)
    @test to_host(x2) == to_host(x)
    # solve_mode 1 (Aᵀ), 2 (Aᴴ), with and without refinement
    for (mode, Aop) in ((1, sparse(transpose(A))), (2, sparse(adjoint(A)))), steps in (0, 2)
        setparam!(solver, "solve_mode", mode)
        setparam!(solver, "ir_n_steps", steps)
        execute!("solve", solver, x, b)
        @test relres(Aop, to_host(x), to_host(b)) <= tol(T)
    end
    setparam!(solver, "solve_mode", 0)
    # refinement measures the residual of A itself (plain and FGMRES)
    for mode in ("ir", "fgmres")
        setparam!(solver, "ir_mode", mode)
        setparam!(solver, "ir_n_steps", 3)
        execute!("solve", solver, x, b)
        @test relres(A, to_host(x), to_host(b)) <= tol(T)
        @test getparam(solver, "ir_n_steps") >= 1
    end
    setparam!(solver, "ir_n_steps", 0)
    setparam!(solver, "ir_mode", "ir")
    # several right-hand sides
    B = to_device(backend, rand(T, n, 3))
    X = to_device(backend, zeros(T, n, 3))
    execute!("solve", solver, X, B)
    @test relres(A, to_host(X), to_host(B)) <= tol(T)
    # refactorization with new values: the scaling of the analysis is kept
    update!(solver, api_matrix(backend, 2 * A, INT))
    execute!("refactorization", solver, x, b)
    execute!("solve", solver, x, b)
    @test relres(2 * A, to_host(x), to_host(b)) <= tol(T)
    # every job: a perfect matching, accurate solves
    for alg in MATCHING_ALGS
        s = DirectSolver(api_matrix(backend, A, INT), "G", 'F')
        setparam!(s, "matching_alg", alg)
        setparam!(s, "ir_n_steps", 3)
        execute!("analysis", s, x, b)
        execute!("factorization", s, x, b)
        execute!("solve", s, x, b)
        @test isperm(getparam(s, "perm_matching"))
        alg in ("algo5", "algo6") && @test getparam(s, "npivots") == 0
        @test relres(A, to_host(x), to_host(b)) <= tol(T)
    end
    # CSC input: the stored CSR is Aᵀ, the matching is computed on it
    Ct = csr_of_transpose(SparseMatrixCSC{T, INT}(A))
    Cd = CSR(to_device(backend, Ct.rowptr), to_device(backend, Ct.colval), to_device(backend, Ct.nzval), n, n;
             transposed = true)
    cs = DirectSolver(Cd, "G", 'F')
    setparam!(cs, "matching_alg", "algo5")
    execute!("analysis", cs, x, b)
    execute!("factorization", cs, x, b)
    @test getparam(cs, "npivots") == 0
    for (mode, Aop) in ((0, A), (1, sparse(transpose(A))), (2, sparse(adjoint(A))))
        setparam!(cs, "solve_mode", mode)
        execute!("solve", cs, x, b)
        @test relres(Aop, to_host(x), to_host(b)) <= tol(T)
    end
    # a stronger scaling (entries over 10^±8): still no perturbed pivot
    if real(T) == Float64
        A4 = badly_scaled_general(T, n, 0.05; exponent = 4)
        s = DirectSolver(api_matrix(backend, A4, INT), "G", 'F')
        setparam!(s, "matching_alg", "algo6")
        execute!("analysis", s, x, b)
        execute!("factorization", s, x, b)
        @test getparam(s, "npivots") == 0
        execute!("solve", s, x, b)
        @test relres(A4, to_host(x), to_host(b)) <= tol(T)
    end
end

@testset "inertia with matching ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    Random.seed!(666)
    INT = Int32
    structure = sym_structure(T)
    for (name, A) in (("random_symindef(100)", random_symindef(T, 100, 0.05)),
                      ("kkt_matrix(60, 30, 1e-8)", kkt_matrix(T, 60, 30, 1.0e-8)))
        npos, nneg, nzero = eigen_inertia(A)
        @test nzero == 0
        b = to_device(backend, rand(T, size(A, 1)))
        x = to_device(backend, zeros(T, size(A, 1)))
        @testset "$name, $alg, view $view" for alg in ("algo1", "algo5"), view in ('L', 'F')
            solver = DirectSolver(api_matrix(backend, triangle_view(A, view), INT), structure, view)
            setparam!(solver, "matching_alg", alg)
            execute!("analysis", solver, x, b)
            execute!("factorization", solver, x, b)
            @test getparam(solver, "inertia") == (INT(npos), INT(nneg))
            @test getparam(solver, "npivots") == 0
            execute!("solve", solver, x, b)
            @test relres(A, to_host(x), to_host(b)) <= tol(T)
            r = getparam(solver, "scale_row")
            @test r == getparam(solver, "scale_col")
            alg == "algo1" && @test all(==(1), r)
            @test getparam(solver, "perm_col") == getparam(solver, "perm_row")
        end
    end
    # complex symmetric "S" with matching (no inertia, but the solve)
    if T <: Complex
        A = random_symindef(T, 80, 0.05; hermitian = false)
        b = to_device(backend, rand(T, 80))
        x = to_device(backend, zeros(T, 80))
        solver = DirectSolver(api_matrix(backend, tril(A), INT), "S", 'L')
        setparam!(solver, "matching_alg", "algo5")
        execute!("analysis", solver, x, b)
        execute!("factorization", solver, x, b)
        execute!("solve", solver, x, b)
        @test relres(A, to_host(x), to_host(b)) <= tol(T)
    end
    # SPD/HPD with matching: the symmetric scaling of a badly scaled matrix
    A = badly_scaled_spd(T, 60, 0.1; exponent = 2)
    b = to_device(backend, rand(T, 60))
    x = to_device(backend, zeros(T, 60))
    solver = DirectSolver(api_matrix(backend, tril(A), INT), spd_structure(T), 'L')
    setparam!(solver, "matching_alg", "algo5")
    setparam!(solver, "ir_n_steps", 2)
    execute!("analysis", solver, x, b)
    execute!("factorization", solver, x, b)
    @test getparam(solver, "info") == 0
    @test getparam(solver, "inertia") == (INT(60), INT(0))
    execute!("solve", solver, x, b)
    @test relres(A, to_host(x), to_host(b)) <= tol(T)
end

@testset "uniform batch with matching ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    Random.seed!(666)
    n = 50
    A = badly_scaled_general(T, n, 0.08)
    members = [A, 2 * A, 3 * A]
    solver = DirectSolver(api_batch_matrix(backend, members, 'F', Int32), "G", 'F')
    setparam!(solver, "matching_alg", "algo5")
    Bh = rand(T, n * 3)
    B = to_device(backend, Bh)
    X = to_device(backend, zeros(T, n * 3))
    execute!("analysis", solver, X, B)
    execute!("factorization", solver, X, B)
    @test getparam(solver, "npivots") == zeros(Int32, 3)
    execute!("solve", solver, X, B)
    @test all(<=(tol(T)), batch_relres(members, to_host(X), Bh))
end
