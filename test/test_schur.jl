# T20: Schur complement mode (PLAN §3.6) beyond the CUDSS.jl examples (ported in
# test/ported/cudss_schur.jl): every structure on larger matrices whose Schur
# root has children in every regime, the symbolic pattern of `"schur_shape"`, the
# dense and sparse exports with every view, the Schur solve phases with one and
# several right-hand sides, the deterministic forward sweep, refactorization,
# `solve_mode`, transposed (CSC) input, the statistics of the factored part, and
# the errors.

# the structures under test for T, with their generator (diagonally dominant or SPD: A₁₁ is well conditioned)
function schur_cases(::Type{T}) where {T}
    cases = Any[(spd_structure(T), () -> laplacian2d(T, 12, 10)),
                (sym_structure(T), () -> random_symindef(T, 120, 0.04)),
                ("G", () -> random_general(T, 120, 0.04))]
    T <: Complex && push!(cases, ("S", () -> random_symindef(T, 120, 0.04; hermitian = false)))
    return cases
end

# default analysis (regime B, at this size), every factored front in regime A, and regimes B and C
schur_regimes() = (("default", nothing), ("A", Options(subtree_parallelism = 0)),
                   ("B/C", Options(subtree_budgets = Int[], regime_c_width = 4, regime_c_rows = 16)))

# 14 Schur indices spread over the matrix
schur_test_flags(n) = (f = falses(n); f[randperm(n)[1:14]] .= true; f)

schur_ldlt(structure) = structure in ("S", "H")

@testset "Schur complement ($(backend_name(backend)), $T, $structure, $rname)" for backend in BACKENDS,
                                                                                 T in ELTYPES,
                                                                                 (structure, gen) in schur_cases(T),
                                                                                 (rname, opts) in schur_regimes()
    A = gen()
    n = size(A, 1)
    flags = schur_test_flags(n)
    s, r = findall(flags), findall(!, flags)
    ns = length(s)
    INT = rname == "B/C" ? Int64 : Int32
    solver = schur_solver(backend, A, structure, flags, INT; opts)
    sc = solver.host_symbolic.schedule
    @test sc.schur == SDS.nsupernodes(solver.host_symbolic) && sc.regime[sc.schur] == SDS.REGIME_C
    @test solver.host_symbolic.partition.perm[(n - ns + 1):n] == s
    rname == "A" && @test count(==(SDS.REGIME_A), sc.regime) > length(sc.regime) ÷ 2
    # Laplacian: a regime-A subtree root hands its contribution block to the Schur root
    rname == "A" && structure in ("SPD", "HPD") &&
        @test any(t -> solver.host_symbolic.partition.snparent[t] == sc.schur, sc.subtree_root)
    rname == "B/C" && @test any(==(SDS.REGIME_B), sc.regime) && count(==(SDS.REGIME_C), sc.regime) > 1
    Sref = schur_reference(A, flags)
    P = schur_pattern_reference(A, flags)
    sym = structure != "G"

    # shape: the exact symbolic pattern (one triangle for the symmetric structures)
    shape = getparam(solver, "schur_shape")
    @test shape == (ns, ns, sym ? count(tril(P)) : count(P)) && shape isa NTuple{3, Int64}
    @test SDS.schur_pattern(SDS.SymmetricPattern(solver.host_rowptr, solver.host_colval, n, structure;
                                                 view = sym ? 'L' : 'F'), flags) ==
          (sp = sparse(P); At = sparse(transpose(sp)); (At.colptr, At.rowval))

    # dense export (the full matrix for every structure), with and without a registered destination
    S = getparam(solver, "schur_matrix")
    @test S isa typeof(to_device(backend, zeros(T, 1, 1))) && size(S) == (ns, ns)
    @test norm(to_host(S) - Sref) <= tol(T) * norm(Sref)
    D = to_device(backend, zeros(T, ns, ns))
    setparam!(solver, "schur_matrix", D)
    @test getparam(solver, "schur_matrix") === D && to_host(D) == to_host(S)
    E = to_device(backend, zeros(T, ns, ns))
    @test getparam!(E, solver, "schur_matrix") === E && to_host(E) == to_host(S)

    # sparse export: the pattern of the view, zero- and one-based
    for v in (sym ? ('L', 'U', 'F') : ('F', 'L')), index in ('O', 'Z')
        keep = v == 'L' ? tril(P) : v == 'U' ? triu(P) : P
        nz = count(keep)
        C = CSR(to_device(backend, zeros(INT, ns + 1)), to_device(backend, zeros(INT, nz)),
                to_device(backend, zeros(T, nz)), ns, ns; index)
        @test getparam!((C, v), solver, "schur_matrix")[1] === C
        Ch = SparseMatrixCSC(C)
        @test nnz(Ch) == nz && all(keep[i, j] for (i, j, _) in zip(findnz(Ch)...))
        @test norm(Matrix(Ch) - Sref .* keep) <= tol(T) * norm(Sref)
    end

    # solves: condensed right-hand side and solution, 1 and 3 right-hand sides
    Sh = to_host(S)
    for B in (rand(T, n), rand(T, n, 3))
        X, Bs = schur_solve(backend, solver, Sh, B; diag = schur_ldlt(structure))
        Bs_ref = B[s, :] - Matrix(A[s, r]) * (Matrix(A[r, r]) \ B[r, :])
        @test norm(Bs - (B isa AbstractVector ? vec(Bs_ref) : Bs_ref)) <= tol(T) * norm(Bs_ref)
        @test relres(A, X, B) <= tol(T)
        # the LU diagonal is part of "solve_bwd_schur": "solve_diag" is the identity for "G" (and Cholesky);
        # compared bitwise, so both solves take the deterministic forward sweep (the atomic one reorders sums)
        if !schur_ldlt(structure)
            setparam!(solver, "deterministic_mode", 1)
            X1, _ = schur_solve(backend, solver, Sh, B; diag = false)
            X2, _ = schur_solve(backend, solver, Sh, B; diag = true)
            setparam!(solver, "deterministic_mode", 0)
            @test X2 == X1
        end
    end
    # the deterministic forward sweep pulls the children's updates into the Schur block
    b = rand(T, n)
    x1, _ = schur_solve(backend, solver, Sh, b; diag = schur_ldlt(structure))
    setparam!(solver, "deterministic_mode", 1)
    x2, _ = schur_solve(backend, solver, Sh, b; diag = schur_ldlt(structure))
    @test norm(x2 - x1) <= tol(T) * norm(x1)
    setparam!(solver, "deterministic_mode", 0)

    # statistics count the factored part A₁₁ only
    stats = getparam(solver, "pivot_stats")
    if structure in ("SPD", "HPD")
        @test getparam(solver, "inertia") == (n - ns, 0)
    elseif structure == "G" || !(T <: Real || structure == "H")
        @test stats.nperturbed == 0
    else
        @test getparam(solver, "inertia") == eigen_npos_nneg(A[r, r])
    end
    @test stats.npos + stats.nneg + stats.nzero <= n - ns && getparam(solver, "npivots") == 0
    @test all(==(one(T)), to_host(getparam(solver, "diag"))[(n - ns + 1):n])

    # refactorization with new values
    A2 = A + spdiagm(0 => fill(T(n), n))
    update!(solver, api_matrix(backend, triangle_view(A2, sym ? 'L' : 'F'), INT))
    execute!("refactorization", solver, nothing, nothing)
    S2 = to_host(getparam(solver, "schur_matrix"))
    Sref2 = schur_reference(A2, flags)
    @test norm(S2 - Sref2) <= tol(T) * norm(Sref2)
    B = rand(T, n)
    X, _ = schur_solve(backend, solver, S2, B; diag = schur_ldlt(structure))
    @test relres(A2, X, B) <= tol(T)

    # the other solve phases need the full factorization
    x, bd = to_device(backend, zeros(T, n)), to_device(backend, rand(T, n))
    for phase in ("solve", "solve_fwd_perm", "solve_fwd", "solve_bwd", "solve_bwd_perm", "solve_refinement")
        @test thrown(() -> execute!(phase, solver, x, bd)) isa NotSupportedError
    end
end

@testset "Schur complement: solve_mode and transposed input ($(backend_name(backend)), $T)" for backend in BACKENDS,
                                                                                               T in ELTYPES
    A = random_general(T, 90, 0.05)
    n = size(A, 1)
    flags = schur_test_flags(n)
    s, r = findall(flags), findall(!, flags)
    Sref = schur_reference(A, flags)
    # solve_mode 1 (Aᵀ) and 2 (Aᴴ): the condensed system is op(S)
    solver = schur_solver(backend, A, "G", flags)
    for (mode, op) in ((1, transpose), (2, adjoint))
        setparam!(solver, "solve_mode", mode)
        B = rand(T, n, 2)
        X, Bs = schur_solve(backend, solver, Matrix(op(Sref)), B; diag = false)
        M = Matrix(op(A))
        @test norm(Bs - (B[s, :] - M[s, r] * (M[r, r] \ B[r, :]))) <= tol(T) * norm(B)
        @test relres(op(A), X, B) <= tol(T)
    end
    # a transposed CSR (the CSC of A) exports the Schur complement of A, also into a transposed destination
    for (structure, M) in (("G", A), (sym_structure(T), random_symindef(T, 90, 0.05)))
        X = structure == "G" ? M : tril(M)        # its CSC arrays: the CSR of Xᵀ, as MadNLP passes them
        csc = CSR(to_device(backend, Vector{Int32}(X.colptr)), to_device(backend, Vector{Int32}(X.rowval)),
                  to_device(backend, X.nzval), n, n; transposed = true)
        solver = DirectSolver(csc, structure, structure == "G" ? 'F' : 'L')
        setparam!(solver, "schur_mode", 1)
        setparam!(solver, "user_schur_indices", Int.(flags))
        execute!("analysis", solver, nothing, nothing)
        execute!("factorization", solver, nothing, nothing)
        Sm = schur_reference(M, flags)
        S = to_host(getparam(solver, "schur_matrix"))
        @test norm(S - Sm) <= tol(T) * norm(Sm)
        B = rand(T, n)
        X, _ = schur_solve(backend, solver, S, B; diag = structure != "G")
        @test relres(M, X, B) <= tol(T)
        P = schur_pattern_reference(M, flags)
        nz = count(P)
        St = sparse(transpose(sparse(P)))
        C = CSR(to_device(backend, zeros(Int32, size(P, 1) + 1)), to_device(backend, zeros(Int32, nz)),
                to_device(backend, zeros(T, nz)), size(P)...; transposed = true)
        getparam!(C, solver, "schur_matrix")
        @test norm(transpose(Matrix(SparseMatrixCSC(C))) - Sm) <= tol(T) * norm(Sm)    # C stores Sᵀ
    end
end

RUN_SHARED && @testset "Schur complement: orderings, edge cases, errors ($(backend_name(backend)))" for backend in BACKENDS
    T = Float64
    # two disconnected blocks, Schur indices in the second one only: the first block is a separate tree
    A = blockdiag(laplacian2d(T, 6, 6), laplacian2d(T, 7, 5))
    n = size(A, 1)
    flags = falses(n)
    flags[[40, 45, 50, 55, 60, 70]] .= true
    Sref = schur_reference(A, flags)
    for (name, params) in (("algo3", ("reordering_alg" => "algo3",)), ("algo4", ("reordering_alg" => "algo4",)),
                           ("algo5", ("reordering_alg" => "algo5",)), ("user_perm", ("user_perm" => randperm(n),)),
                           ("no amalgamation", ("use_superpanels" => 0,)))
        solver = schur_solver(backend, A, "SPD", flags; params)
        @test norm(to_host(getparam(solver, "schur_matrix")) - Sref) <= tol(T) * norm(Sref)
        perm = getparam(solver, "perm_reorder_row")
        @test perm[(n - 5):n] == findall(flags)
        if name == "user_perm"
            up = params[1][2]
            @test solver.ordering.perm == [filter(i -> !flags[i], up); findall(flags)]   # before the renumbering
        end
        @test getparam(solver, "schur_shape")[3] == count(tril(schur_pattern_reference(A, flags)))
        B = rand(T, n)
        X, _ = schur_solve(backend, solver, Sref, B)
        @test relres(A, X, B) <= tol(T)
    end
    # every index in the Schur set: nothing is factored, S = A, the phases only permute
    A = random_general(T, 30, 0.1)
    solver = schur_solver(backend, A, "G", trues(30))
    @test to_host(getparam(solver, "schur_matrix")) ≈ Matrix(A)
    B = rand(T, 30)
    X, Bs = schur_solve(backend, solver, Matrix(A), B)
    @test Bs == B && relres(A, X, B) <= tol(T)

    # errors
    A = laplacian2d(T, 6, 6)
    n = size(A, 1)
    flags = falses(n)
    flags[1:4] .= true
    mk() = DirectSolver(api_matrix(backend, tril(A)), "SPD", 'L')
    for (ind, err) in ((nothing, InvalidValueError), (zeros(Int, n), InvalidValueError),
                       (ones(Int, n - 1), InvalidValueError))
        solver = mk()
        setparam!(solver, "schur_mode", 1)
        ind === nothing || setparam!(solver, "user_schur_indices", ind)
        @test thrown(() -> execute!("analysis", solver, nothing, nothing)) isa err
    end
    rowptr, colval, nzval = batch_csr([A, 2A], 'L', Int32)
    batch = DirectSolver(api_csr(backend, rowptr, colval, nzval, n), "SPD", 'L')
    setparam!(batch, "schur_mode", 1)
    setparam!(batch, "user_schur_indices", Int.(flags))
    @test thrown(() -> execute!("analysis", batch, nothing, nothing)) isa NotSupportedError
    solver = mk()
    setparam!(solver, "schur_mode", 1)
    setparam!(solver, "user_schur_indices", Int.(flags))
    execute!("analysis", solver, nothing, nothing)
    @test getparam(solver, "schur_shape") == (4, 4, 10)
    @test thrown(() -> getparam(solver, "schur_matrix")) isa FactorizationError
    for bad in (to_device(backend, zeros(T, 3, 3)), to_device(backend, zeros(Float32, 4, 4)), sparse(1.0I, 4, 4),
                (to_device(backend, zeros(T, 4, 4)), 'L'), "S",
                CSR(to_device(backend, ones(Int32, 5)), to_device(backend, ones(Int32, 9)),
                    to_device(backend, zeros(T, 9)), 4, 4))
        @test thrown(() -> setparam!(solver, "schur_matrix", bad)) isa InvalidValueError
    end
    D = to_device(backend, zeros(T, 4, 4))
    setparam!(solver, "schur_matrix", MatrixDescriptor(D))
    setparam!(solver, "schur_matrix", nothing)
    @test solver.schur.dest === nothing
    execute!("factorization", solver, nothing, nothing)
    @test getparam(solver, "schur_matrix") isa typeof(D)
    # a new analysis without schur_mode leaves Schur mode
    setparam!(solver, "schur_mode", 0)
    execute!("analysis", solver, nothing, nothing)
    @test solver.schur === nothing && solver.host_symbolic.schedule.schur == 0
    @test thrown(() -> getparam(solver, "schur_shape")) isa InvalidValueError
end
