# T18: FGMRES-IR (`ir_mode = "fgmres"`, Krylov.jl extension) with the
# factorization as right preconditioner.

# KKT matrix with 5 duals whose static pivot perturbation (`pivot_epsilon = 1`, no 2×2 pairs) is far larger
# than the pivots: the factor is that of `A + E`, `E` of rank 5 and `ρ((A + E)⁻¹ E) ≈ 1`, so plain refinement
# stalls, while `(A + E)⁻¹ A` is the identity plus a rank-5 term and FGMRES converges in a few iterations
const FGMRES_STALL_PARAMS = (("pivot_pairs", "none"), ("pivot_epsilon", 1.0))
fgmres_stall_matrix() = kkt_matrix(Float64, 150, 5, 0.0)

RUN_SHARED && @testset "FGMRES-IR where plain IR stalls ($(backend_name(backend)), $INT)" for backend in BACKENDS,
                                                                                               INT in INTTYPES
    Random.seed!(666)
    K = fgmres_stall_matrix()
    b = K * rand(size(K, 1))
    solver = ir_solver(backend, K, "S", INT; params = FGMRES_STALL_PARAMS)
    @test getparam(solver, "npivots") > 0
    # plain IR: no step of the first 5 gains a digit, and 20 steps stay far from 1e-12
    r = [relres(K, ir_solve(backend, solver, b; steps), b) for steps in 0:5]
    @test r[1] > 1.0e-8
    @test minimum(r[2:end]) > r[1] / 10
    @test relres(K, ir_solve(backend, solver, b; steps = 20), b) > 1.0e-8
    # FGMRES-IR: relres ≤ 1e-12 within 20 iterations (early exit on ir_tol)
    setparam!(solver, "ir_mode", "fgmres")
    x = ir_solve(backend, solver, b; steps = 20, tol = 1.0e-12)
    iters = getparam(solver, "ir_n_steps")
    @test 1 <= iters <= 20
    @test relres(K, x, b) <= 1.0e-12
    # without ir_tol, all 20 iterations are run (or FGMRES ends on an exact residual)
    x = ir_solve(backend, solver, b; steps = 20, tol = 0.0)
    @test iters <= getparam(solver, "ir_n_steps") <= 20
    @test relres(K, x, b) <= 1.0e-12
    @test getparam(solver, "ir_mode") == "fgmres"
    # more iterations than the stored Krylov basis: the workspace is reallocated
    x = ir_solve(backend, solver, b; steps = 30, tol = 1.0e-14)
    @test relres(K, x, b) <= 1.0e-14
end

@testset "FGMRES-IR: element types, multiple right-hand sides, solve_mode ($(backend_name(backend)), $T)" for backend in BACKENDS,
                                                                                                               T in ELTYPES
    n = 200
    for (A, structure) in ((random_spd(T, n, 0.02), spd_structure(T)), (random_symindef(T, n, 0.02), sym_structure(T)))
        solver = ir_solver(backend, A, structure; params = (("ir_mode", "fgmres"), ("deterministic_mode", 1)))
        for nrhs in (1, 3)
            b = nrhs == 1 ? rand(T, n) : rand(T, n, nrhs)
            x = ir_solve(backend, solver, b; steps = 3, tol = 0.0)
            @test relres(A, x, b) <= tol(T)
            @test 1 <= getparam(solver, "ir_n_steps") <= 3
            # ir_tol: early exit, every column below the tolerance
            x = ir_solve(backend, solver, b; steps = 10, tol = 100 * eps(real(T)))
            @test getparam(solver, "ir_n_steps") < 10
            @test all(k -> relres(A, x[:, k], b[:, k]) <= tol(T), 1:nrhs)
            # X === B is allowed with "solve"; the six sub-phases compose to "solve"
            bd = to_device(backend, b)
            execute!("solve", solver, bd, bd; asynchronous = false)
            xs = to_host(bd)
            @test relres(A, xs, b) <= tol(T)
            bd = to_device(backend, b)
            xd = similar(bd)
            for phase in SOLVE_SUBPHASES
                execute!(phase, solver, xd, bd)
            end
            @test to_host(xd) == xs
        end
        # transposed and adjoint solves (complex "S": Aᴴ x = b goes through the conjugated operators)
        b = rand(T, n)
        for (mode, op) in ((1, transpose), (2, adjoint))
            setparam!(solver, "solve_mode", mode)
            x = ir_solve(backend, solver, b; steps = 3, tol = 0.0)
            @test relres(op(A), x, b) <= tol(T)
        end
        setparam!(solver, "solve_mode", 0)
    end
end

@testset "FGMRES-IR: uniform batch ($(backend_name(backend)), $T)" for backend in BACKENDS,
                                                                       T in filter(in((Float64, ComplexF32)), ELTYPES)
    n, nb, nrhs = 120, 3, 2
    members = batch_members(random_symindef(T, n, 0.03), nb)
    solver = DirectSolver(api_batch_matrix(backend, members, 'L'), sym_structure(T), 'L')
    setparam!(solver, "ir_mode", "fgmres")
    setparam!(solver, "ir_n_steps", 3)
    execute!("analysis", solver, nothing, nothing)
    execute!("factorization", solver, nothing, nothing)
    B = rand(T, n, nrhs, nb)
    Bd = to_device(backend, B)
    Xd = similar(Bd)
    execute!("solve", solver, Xd, Bd; asynchronous = false)
    @test all(batch_relres(members, to_host(Xd), B) .<= tol(T))
    @test 1 <= getparam(solver, "ir_n_steps") <= 3
    # ubatch_index: only member 2 is solved and refined, the others' columns are untouched
    setparam!(solver, "ubatch_index", 1)
    X2 = to_device(backend, zeros(T, n, nrhs, nb))
    execute!("solve", solver, X2, Bd; asynchronous = false)
    Xh = to_host(X2)
    @test relres(members[2], Xh[:, :, 2], B[:, :, 2]) <= tol(T)
    @test iszero(Xh[:, :, 1]) && iszero(Xh[:, :, 3])
end

RUN_SHARED && @testset "FGMRES-IR: interrupt and missing Krylov ($(backend_name(backend)))" for backend in BACKENDS
    Random.seed!(666)
    K = fgmres_stall_matrix()
    b = K * rand(size(K, 1))
    flag = Threads.Atomic{Bool}(false)
    solver = ir_solver(backend, K, "S"; params = (FGMRES_STALL_PARAMS..., ("deterministic_mode", 1),
                                                  ("user_host_interrupt", flag)))
    x0 = ir_solve(backend, solver, b; steps = 0)
    setparam!(solver, "ir_mode", "fgmres")
    setparam!(solver, "ir_n_steps", 10)
    # an interrupt during FGMRES leaves the unrefined solution in X and reports 0 iterations
    flag[] = true
    bd = to_device(backend, b)
    xd = similar(bd)
    @test thrown(() -> execute!("solve", solver, xd, bd; asynchronous = false)) isa InterruptedError
    @test to_host(xd) == x0
    @test getparam(solver, "ir_n_steps") == 0
    flag[] = false
    @test relres(K, ir_solve(backend, solver, b; tol = 1.0e-12), b) <= 1.0e-12
    # without Krylov.jl, ir_mode = "fgmres" raises NotSupportedError at the refinement (not at setparam!)
    provider = SDS.FGMRES_PROVIDER[]
    @test SDS.fgmres_available()
    try
        SDS.FGMRES_PROVIDER[] = nothing
        @test !SDS.fgmres_available()
        @test thrown(() -> execute!("solve", solver, xd, bd; asynchronous = false)) isa NotSupportedError
        # no refinement requested: nothing needs Krylov.jl
        @test ir_solve(backend, solver, b; steps = 0) == x0
    finally
        SDS.FGMRES_PROVIDER[] = provider
    end
end
