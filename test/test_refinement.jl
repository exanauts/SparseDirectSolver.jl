# T16: iterative refinement ("solve_refinement", `ir_n_steps`, `ir_tol`), the
# solve sub-phases composed, `solve_mode`, `user_host_interrupt` and logging.

# a logger that sets `flag` when the refinement logs the residual after `step` corrections (an interrupt
# raised between two refinement steps, deterministically)
struct InterruptAtStepLogger <: Base.CoreLogging.AbstractLogger
    flag::Threads.Atomic{Bool}
    step::Int
end
Base.CoreLogging.min_enabled_level(::InterruptAtStepLogger) = Base.CoreLogging.Debug
Base.CoreLogging.shouldlog(::InterruptAtStepLogger, args...) = true
Base.CoreLogging.catch_exceptions(::InterruptAtStepLogger) = false
function Base.CoreLogging.handle_message(L::InterruptAtStepLogger, level, message, args...; kwargs...)
    startswith(string(message), "refinement: step $(L.step),") && (L.flag[] = true)
    return nothing
end

RUN_SHARED && @testset "refinement on a badly scaled SPD matrix ($(backend_name(backend)))" for backend in BACKENDS
    Random.seed!(666)
    T = Float64
    A = badly_scaled_spd(T, 300, 0.02)       # rows scaled by up to 10^(±8)
    b = A * rand(T, 300)
    solver = ir_solver(backend, A, "SPD")
    r0 = relres(A, ir_solve(backend, solver, b; steps = 0), b)
    @test getparam(solver, "ir_n_steps") == 0
    r1 = relres(A, ir_solve(backend, solver, b; steps = 1), b)
    @test getparam(solver, "ir_n_steps") == 1
    @test r1 <= tol(T)
    # Cholesky is backward stable under symmetric scaling: the unrefined residual is already at
    # rounding level (≈ 1.3 eps; one step gains 3× to 5.4× over 16 seeds on the CPU backend, 2× to 7×
    # in the T16 report), so the 100× reduction of the task text can't be observed on an SPD matrix;
    # it is checked on the perturbed LDLᵀ factorization below. What holds on every backend: r0 at
    # rounding level and a step that does not lose accuracy.
    @test r0 <= 100 * eps(T)
    @test r1 <= r0
    # ir_tol: early exit, the data parameter reports the steps performed
    x = ir_solve(backend, solver, b; steps = 10, tol = 1.0e-14)
    @test getparam(solver, "ir_n_steps") < 10
    @test relres(A, x, b) <= 1.0e-14
    @test getparam(solver.options, "ir_n_steps") == 10
end

RUN_SHARED &&
@testset "refinement with static pivot perturbation ($(backend_name(backend)), $INT)" for backend in BACKENDS,
                                                                                           INT in INTTYPES
    Random.seed!(666)
    # KKT matrix without 2×2 pairs: the zero (2,2) block is perturbed (pivot_epsilon), and refinement
    # removes the perturbation error (the case of the MadNLP K2 systems, issue #71)
    T = Float64
    K = kkt_matrix(T, 150, 60, 0.0)
    b = K * rand(T, size(K, 1))
    solver = ir_solver(backend, K, "S", INT; params = (("pivot_pairs", "none"),))
    @test getparam(solver, "npivots") > 0
    r0 = relres(K, ir_solve(backend, solver, b; steps = 0), b)
    r1 = relres(K, ir_solve(backend, solver, b; steps = 1), b)
    @test r0 > 1.0e-8
    @test r1 <= r0 / 100
    x = ir_solve(backend, solver, b; steps = 10, tol = 1.0e-14)
    steps = getparam(solver, "ir_n_steps")
    @test 1 < steps < 10
    @test relres(K, x, b) <= 1.0e-14
    # the same steps without the early exit reach the same residual level
    @test relres(K, ir_solve(backend, solver, b; steps, tol = 0.0), b) <= 1.0e-14
    @test getparam(solver, "ir_n_steps") == steps
    # setting ir_n_steps reports the requested value until the next solve
    setparam!(solver, "ir_n_steps", 7)
    @test getparam(solver, "ir_n_steps") == 7
end

@testset "refinement: layouts, aliasing, element types ($(backend_name(backend)), $T)" for backend in BACKENDS,
                                                                                           T in ELTYPES
    Random.seed!(666)
    n = 200
    for (A, structure) in ((random_spd(T, n, 0.02), spd_structure(T)), (random_symindef(T, n, 0.02), sym_structure(T)))
        # bitwise comparisons between solves: the atomic forward sweep (default on GPUs for real T)
        # sums in a run-dependent order, so these solves use the deterministic variant
        solver = ir_solver(backend, A, structure; params = (("ir_n_steps", 2), ("deterministic_mode", 1)))
        for nrhs in (1, 3)
            b = nrhs == 1 ? rand(T, n) : rand(T, n, nrhs)
            x = ir_solve(backend, solver, b)
            @test relres(A, x, b) <= tol(T)
            @test getparam(solver, "ir_n_steps") == 2
            # X === B: "solve" keeps a copy of B for the residual, the result is the same
            bd = to_device(backend, b)
            execute!("solve", solver, bd, bd; asynchronous = false)
            @test to_host(bd) == x
            # "solve_refinement" needs the original B
            @test thrown(() -> execute!("solve_refinement", solver, bd, bd)) isa InvalidValueError
            nrhs == 1 && continue
            # row-major and strided layouts
            bt = to_device(backend, permutedims(b))
            Xt = MatrixDescriptor(similar(bt); transposed = true)
            execute!("solve", solver, Xt, MatrixDescriptor(bt; transposed = true); asynchronous = false)
            @test permutedims(to_host(Xt.data)) == x
            bs = to_device(backend, vec(b))
            xs = similar(bs)
            execute!("solve", solver, xs, bs; asynchronous = false)
            @test reshape(to_host(xs), n, nrhs) == x
        end
        @test SDS.max_rhs(solver.refinement) >= 3
        # a row-major strided vector with fewer right-hand sides than the workspace, ir_tol > 0:
        # the norm reduction indexes B with the solve's nrhs, not the workspace capacity
        b2 = rand(T, n, 2)
        Bd = MatrixDescriptor(T, n, 2; transposed = true)
        update!(Bd, to_device(backend, vec(permutedims(b2))))
        Xd = MatrixDescriptor(T, n, 2; transposed = true)
        update!(Xd, similar(Bd.data))
        setparam!(solver, "ir_tol", 1.0e-30)
        execute!("solve", solver, Xd, Bd; asynchronous = false)
        x2 = permutedims(reshape(to_host(Xd.data), 2, n))
        @test relres(A, x2, b2) <= tol(T)
        W = solver.refinement
        SDS.residual!(W, vec(solver.A.nzval), Xd.data, Bd.data; nrhs = 2, transposed = true)
        nh = SDS.residual_norms!(W, Bd.data; nrhs = 2, transposed = true)
        @test [nh[2], nh[4]] ≈ [sum(abs2, b2[:, 1]), sum(abs2, b2[:, 2])]
        setparam!(solver, "ir_tol", 0.0)
        # "solve_refinement" with ir_n_steps = 0 is a no-op
        setparam!(solver, "ir_n_steps", 0)
        b = rand(T, n)
        x = ir_solve(backend, solver, b)
        xd = to_device(backend, x)
        execute!("solve_refinement", solver, xd, to_device(backend, b); asynchronous = false)
        @test to_host(xd) == x && getparam(solver, "ir_n_steps") == 0
        # the residual uses the current values: update! without refactorization refines towards them
        if structure == spd_structure(T)
            A2 = A + T(n / 100) * I
            M2 = api_matrix(backend, triangle_view(A2, 'L'), Int32)
            update!(solver, M2)
            x2 = ir_solve(backend, solver, b; steps = 20, tol = 0.0)
            @test relres(A2, x2, b) <= tol(T)
        end
    end
end

@testset "solve sub-phases compose to \"solve\" ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    Random.seed!(666)
    n = 150
    for (A, structure) in ((random_spd(T, n, 0.03), spd_structure(T)), (random_symindef(T, n, 0.03), sym_structure(T)))
        # bitwise comparison: deterministic forward sweep (the atomic one is run-dependent on GPUs)
        solver = ir_solver(backend, A, structure; params = (("deterministic_mode", 1),))
        for steps in (0, 2), nrhs in (1, 2)
            setparam!(solver, "ir_n_steps", steps)
            b = to_device(backend, nrhs == 1 ? rand(T, n) : rand(T, n, nrhs))
            x1 = similar(b)
            execute!("solve", solver, x1, b; asynchronous = false)
            x2 = similar(b)
            fill!(x2, zero(T))
            for phase in SOLVE_SUBPHASES
                execute!(phase, solver, x2, b)
            end
            KernelAbstractions.synchronize(backend)
            @test to_host(x2) == to_host(x1)
            @test getparam(solver, "ir_n_steps") == steps
            @test relres(A, to_host(x1), to_host(b)) <= tol(T)
        end
    end
end

@testset "solve_mode ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    Random.seed!(666)
    n = 120
    b = rand(T, n)
    cases = Any[(random_spd(T, n, 0.03), spd_structure(T)), (random_symindef(T, n, 0.03), sym_structure(T))]
    T <: Complex && push!(cases, (random_symindef(T, n, 0.03; hermitian = false), "S"))   # complex symmetric
    for (A, structure) in cases, steps in (0, 1)
        solver = ir_solver(backend, A, structure; params = (("ir_n_steps", steps),))
        for (mode, op) in ((0, A), (1, transpose(A)), (2, adjoint(A)))
            setparam!(solver, "solve_mode", mode)
            @test relres(sparse(op), ir_solve(backend, solver, b), b) <= tol(T)
        end
    end
    # the task's case: complex symmetric "S", solve_mode = 2 solves Aᴴ x = b (Aᴴ ≠ A)
    if T <: Complex
        A = random_symindef(T, n, 0.03; hermitian = false)
        solver = ir_solver(backend, A, "S"; params = (("solve_mode", 2),))
        x = ir_solve(backend, solver, b)
        @test relres(sparse(A'), x, b) <= tol(T)
        @test relres(A, x, b) > 100 * tol(T)
    end
    # CSC input (the CSR of the transpose): for Hermitian matrices that is the conjugate
    A = random_spd(T, n, 0.03)
    Ct = csr_of_transpose(tril(A))
    C = CSR(to_device(backend, Ct.rowptr), to_device(backend, Ct.colval), to_device(backend, Ct.nzval), n, n;
            transposed = true)
    for structure in (spd_structure(T), sym_structure(T)), mode in 0:2, steps in (0, 2)
        solver = DirectSolver(C, structure, 'L')
        setparam!(solver, "solve_mode", mode)
        setparam!(solver, "ir_n_steps", steps)
        execute!("analysis", solver, nothing, nothing)
        execute!("factorization", solver, nothing, nothing)
        op = mode == 1 ? sparse(transpose(A)) : A
        @test relres(op, ir_solve(backend, solver, b), b) <= tol(T)
    end
end

@testset "user_host_interrupt ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    Random.seed!(666)
    A = random_symindef(T, 300, 0.02)
    b = rand(T, 300)
    for structure in (sym_structure(T), spd_structure(T))
        M = structure == spd_structure(T) ? random_spd(T, 300, 0.02) : A
        flag = Threads.Atomic{Bool}(false)
        solver = DirectSolver(api_matrix(backend, tril(M), Int32), structure, 'L')
        setparam!(solver, "user_host_interrupt", flag)
        @test getparam(solver, "user_host_interrupt") === flag
        flag[] = true
        @test thrown(() -> execute!("analysis", solver, nothing, nothing)) isa InterruptedError
        flag[] = false
        execute!("analysis", solver, nothing, nothing)
        flag[] = true
        @test thrown(() -> factorize!(solver)) isa InterruptedError
        @test thrown(() -> execute!("solve", solver, to_device(backend, b), to_device(backend, b))) isa
              FactorizationError
        flag[] = false
        factorize!(solver)
        @test getparam(solver, "info") == 0
        @test relres(M, ir_solve(backend, solver, b), b) <= tol(T)
        # an interrupted refactorization: the solver needs a "factorization" again
        flag[] = true
        @test thrown(() -> refactorize!(solver)) isa InterruptedError
        @test thrown(() -> refactorize!(solver)) isa FactorizationError
        flag[] = false
        factorize!(solver)
        refactorize!(solver)
        # the LinearAlgebra layer recovers too: cholesky!/ldlt! pick "factorization" again
        flag[] = true
        @test thrown(() -> refactorize!(solver)) isa InterruptedError
        @test solver.fresh_factorization
        flag[] = false
        structure == spd_structure(T) ? cholesky!(solver, api_matrix(backend, tril(M), Int32)) :
            ldlt!(solver, api_matrix(backend, tril(M), Int32))
        @test !solver.fresh_factorization
        @test relres(M, ir_solve(backend, solver, b), b) <= tol(T)
        # an interrupt after two refinement steps: X holds that iterate, ir_n_steps reports 2
        setparam!(solver, "ir_n_steps", 5)
        setparam!(solver, "ir_tol", 1.0e-30)
        @test Base.CoreLogging.with_logger(() -> thrown(() -> ir_solve(backend, solver, b)),
                                           InterruptAtStepLogger(flag, 1)) isa InterruptedError
        @test getparam(solver, "ir_n_steps") == 2
        flag[] = false
        setparam!(solver, "ir_tol", 0.0)
        # refinement polls the flag between steps
        setparam!(solver, "ir_n_steps", 2)
        flag[] = true
        @test thrown(() -> ir_solve(backend, solver, b)) isa InterruptedError
        flag[] = false
        @test relres(M, ir_solve(backend, solver, b), b) <= tol(T)
        setparam!(solver, "user_host_interrupt", nothing)
    end
end

@testset "LinearAlgebra layer refines ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    Random.seed!(666)
    A = random_spd(T, 100, 0.03)
    b = rand(T, 100)
    for F in (cholesky(api_matrix(backend, tril(A), Int32); view = 'L'), ldlt(api_matrix(backend, tril(A), Int32); view = 'L'))
        @test getparam(F, "ir_n_steps") == 2
        @test relres(A, to_host(F \ to_device(backend, b)), b) <= tol(T)
        @test getparam(F, "ir_n_steps") == 2
        bd = to_device(backend, b)
        ldiv!(F, bd)
        @test relres(A, to_host(bd), b) <= tol(T)
    end
    # the handle layer keeps the cuDSS default
    @test getparam(DirectSolver(api_matrix(backend, tril(A), Int32), spd_structure(T), 'L'), "ir_n_steps") == 0
end

RUN_SHARED && @testset "logging" begin
    Random.seed!(666)
    backend = first(BACKENDS)
    A = random_spd(Float64, 80, 0.05)
    b = rand(80)
    old = SDS.set_log_level!("info")
    try
        solver = DirectSolver(api_matrix(backend, tril(A), Int32), "SPD", 'L')
        @test_logs (:info, r"^reordering") (:info, r"^symbolic_factorization") match_mode = :any analyze!(solver)
        @test_logs (:info, r"^factorization: info = 0") match_mode = :any factorize!(solver)
        setparam!(solver, "ir_n_steps", 3)
        setparam!(solver, "ir_tol", 1.0e-30)
        @test_logs (:info, r"^solve_refinement: 3 of 3 steps") match_mode = :any ir_solve(backend, solver, b)
        SDS.set_log_level!(2)
        @test_logs (:info, r"^refinement: step 0, relative residual") match_mode = :any ir_solve(backend, solver, b)
        SDS.set_log_level!(0)
        @test_logs min_level = Base.CoreLogging.Info ir_solve(backend, solver, b)     # silent
    finally
        SDS.set_log_level!(old)
    end
    @test thrown(() -> SDS.set_log_level!("verbose")) isa InvalidValueError
    @test SDS._parse_log_level("DEBUG") == SDS.LOG_DEBUG && SDS._parse_log_level("") == SDS.LOG_NONE
end
