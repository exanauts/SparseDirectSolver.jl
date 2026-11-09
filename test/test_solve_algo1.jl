# solve_alg = "algo1" on DirectSolver (T36): the partitioned-inverse fused solve of src/fused.jl after the
# stock factorization. The dependency-counter kernels spin on resident workgroups, so the numeric tests run on
# the GPU backends of BACKENDS only (the backends with `forward_progress`); on CPU() the file checks that the
# analysis raises NotSupportedError, with the other unsupported combinations.

const ALGO1_BACKENDS = filter(SDS.forward_progress, BACKENDS)

# a DirectSolver of the lower triangle of A with solve_alg = "algo1" ("S": no pivoting), analyzed and factorized
function algo1_solver(backend, A::SparseMatrixCSC, structure; INT = Int32, kwargs...)
    s = DirectSolver(api_matrix(backend, tril(A), INT), structure, 'L')
    setparam!(s, "solve_alg", "algo1")
    structure in ("S", "H") && setparam!(s, "pivot_type", 'N')
    for (name, value) in kwargs
        setparam!(s, String(name), value)
    end
    analyze!(s)
    factorize!(s)
    return s
end

# the solution of one right-hand side on `backend` (host vector)
function algo1_solve(backend, s, b; phases = ("solve",))
    bd = to_device(backend, b)
    xd = similar(bd)
    for phase in phases
        execute!(phase, s, xd, bd)
    end
    KernelAbstractions.synchronize(backend)
    return to_host(xd)
end

# the well-conditioned SPD generators of test/matrices.jl (a 2-D and three 3-D Laplacians, the larger ones with a root
# front wider than 256: the vendor dense part of the fused solve; the 34³ one also has fronts of width <= 256
# between the wider ones near the root, a second dependency-counter segment) and the symmetric indefinite ones that factor
# without pivoting (diagonally dominant, a KKT matrix with SPD or indefinite Hessian and δ = 1, the grid KKT
# matrix whose root separators mix primal and dual rows, and a KKT matrix with a denser Hessian whose root front is
# wider than 256: the vendor dense sweeps of the fused solve on a factor of the stock regime-C LDLᵀ kernels, which
# rely on its unit diagonal and on pivot_type = 'N' applying no local pivot order)
algo1_spd_generators(::Type{T}) where {T} = (
    ("laplacian2d 30×30", () -> laplacian2d(T, 30, 30)),
    ("laplacian3d 10³", () -> laplacian3d(T, 10, 10, 10)),
    ("laplacian3d 18³", () -> laplacian3d(T, 18, 18, 18)),
    ("laplacian3d 34³", () -> laplacian3d(T, 34, 34, 34)),
    ("random_spd 300", () -> random_spd(T, 300, 0.02)),
    ("schur_example_cholesky 5×5", () -> schur_example_cholesky(T).A),
)
algo1_indefinite_generators(::Type{T}) where {T} = (
    ("random_symindef 400", () -> random_symindef(T, 400, 0.01)),
    ("kkt_matrix 300+150", () -> kkt_matrix(T, 300, 150, 1.0)),
    ("kkt_matrix 300+150, indefinite H", () -> kkt_matrix(T, 300, 150, 1.0; hessian = :indefinite)),
    ("grid_kkt_matrix 20×20", () -> grid_kkt_matrix(T, 20, 20, 1.0)),
    ("kkt_matrix 1000+500, denser H", () -> kkt_matrix(T, 1000, 500, 1.0; density_h = 0.01)),
)
# the generators whose weak pivots (no pivoting) leave the Float32 residual at about tol(T), for the fused solve as for
# the level sweeps: their solves are checked against weak_pivot_tol (test/utils.jl)
const ALGO1_WEAK_PIVOT_LABELS = ("kkt_matrix 300+150, indefinite H",)

@testset "solve_alg = \"algo1\" on $(backend_name(backend))" for backend in ALGO1_BACKENDS
    @testset "$structure ($label): $T" for T in eltypes_among((Float32, Float64)),
                                          (structure, (label, gen)) in
                                          vcat([("SPD", g) for g in algo1_spd_generators(T)],
                                               [("S", g) for g in algo1_indefinite_generators(T)])
        Random.seed!(666)
        A = gen()
        n = size(A, 1)
        rtol = label in ALGO1_WEAK_PIVOT_LABELS ? weak_pivot_tol(T) : tol(T)
        s = algo1_solver(backend, A, structure)
        @test getparam(s, "info") == 0
        @test s.fsolve isa SDS.Fused.FusedSolve
        label == "laplacian3d 34³" && @test length(s.fsolve.fz[1]) > 1         # several solve segments
        label == "kkt_matrix 1000+500, denser H" && @test maximum(s.symbolic.schedule.width) > 256   # vendor dense part
        b = rand(T, n)
        x = algo1_solve(backend, s, b)
        @test relres(A, x, b) <= rtol
        # the level sweeps on the same factorization
        setparam!(s, "solve_alg", "default")
        x0 = algo1_solve(backend, s, b)
        setparam!(s, "solve_alg", "algo1")
        @test norm(x - x0) / norm(x0) <= tol(T)
        # deterministic_mode = 1 (read at every solve): bitwise equal solves
        setparam!(s, "deterministic_mode", 1)
        x1 = algo1_solve(backend, s, b)
        @test algo1_solve(backend, s, b) == x1
        @test relres(A, x1, b) <= rtol
        # the sub-phases run the halves of the fused solve: bitwise equal to "solve" under deterministic_mode = 1
        @test algo1_solve(backend, s, b; phases = ("solve_fwd_perm", "solve_fwd", "solve_diag", "solve_bwd",
                                                   "solve_bwd_perm")) == x1
        setparam!(s, "deterministic_mode", 0)
        # iterative refinement composes with the fused solve (the corrections run it)
        setparam!(s, "ir_n_steps", 2)
        xr = algo1_solve(backend, s, b)
        @test getparam(s, "ir_n_steps") == 2
        @test relres(A, xr, b) <= tol(T)
        setparam!(s, "ir_n_steps", 0)
        # a refactorization with new values: the inverses follow the values
        A2 = 2 * A
        copyto!(s.A.nzval, to_device(backend, host_csr_arrays(tril(A2), Int32)[3]))
        refactorize!(s)
        @test getparam(s, "info") == 0
        x2 = algo1_solve(backend, s, b)
        @test relres(A2, x2, b) <= rtol
        # memory_estimates: slot 14 holds the fused solve's buffers, added to the device totals
        est = getparam(s, "memory_estimates")
        s0 = DirectSolver(api_matrix(backend, tril(A), Int32), structure, 'L')
        structure == "S" && setparam!(s0, "pivot_type", 'N')
        analyze!(s0)
        est0 = getparam(s0, "memory_estimates")
        @test est0[14] == 0 && est[14] > 0
        @test est[1] == est0[1] + est[14] && est[2] == est0[2] + est[14] && est[5] == est0[5] + est[14]
    end

    @testset "FGMRES refinement and matching: $T" for T in eltypes_among((Float32, Float64))
        Random.seed!(666)
        A = laplacian3d(T, 10, 10, 10)
        b = rand(T, size(A, 1))
        s = algo1_solver(backend, A, "SPD"; ir_n_steps = 3, ir_mode = "fgmres")
        if SDS.fgmres_available()
            x = algo1_solve(backend, s, b)
            @test relres(A, x, b) <= tol(T)
        else
            @test_throws NotSupportedError algo1_solve(backend, s, b)
        end
        # the matching permutation and scaling wrap the fused solve as the level sweeps
        for structure in ("SPD", "S")
            B = structure == "SPD" ? badly_scaled_spd(T, 200, 0.03; exponent = 1) : kkt_matrix(T, 200, 100, 1.0)
            s = algo1_solver(backend, B, structure; matching_alg = "algo5")
            c = rand(T, size(B, 1))
            @test relres(B, algo1_solve(backend, s, c), c) <= tol(T)
        end
    end

    @testset "refactorization and solve allocate no device memory: $T" for T in eltypes_among((Float32, Float64))
        Random.seed!(666)
        for (structure, A) in (("SPD", laplacian3d(T, 10, 10, 10)), ("SPD", laplacian3d(T, 18, 18, 18)),
                               ("S", random_symindef(T, 400, 0.01)))
            s = algo1_solver(backend, A, structure)
            bd = to_device(backend, rand(T, size(A, 1)))
            xd = similar(bd)
            cycle() = (refactorize!(s); solve!(s, xd, bd))
            cycle()
            KernelAbstractions.synchronize(backend)
            dev = device_allocated(backend, cycle)
            KernelAbstractions.synchronize(backend)
            ismissing(dev) ? (@test_skip "no device allocation counter on $(backend_name(backend))") :
                             (@test dev == 0)
            # host bytes: the fused part of the cycle (the inverses and the fused solve, against a level-sweep
            # cycle of the same solver) stays within the launch allowance of the fused pipeline
            host = @allocated cycle()
            setparam!(s, "solve_alg", "default")
            cycle()
            host0 = @allocated cycle()
            setparam!(s, "solve_alg", "algo1")
            refactorize!(s)
            @test host <= host0 + fused_cycle_alloc_budget(backend)
            host <= host0 + fused_cycle_alloc_budget(backend) ||
                @info "algo1 refactorization + solve on $(backend_name(backend)), $structure, $T: $host bytes " *
                      "(level sweeps: $host0)"
        end
    end
end

# the unsupported combinations raise NotSupportedError: at the analysis, the factorization or the solve
@testset "solve_alg = \"algo1\": unsupported combinations ($(backend_name(backend)), $T)" for backend in BACKENDS,
                                                                                            T in ELTYPES
    Random.seed!(666)
    A = laplacian2d(T, 12, 12)
    n = size(A, 1)
    b = to_device(backend, rand(T, n))
    x = similar(b)
    analysis(structure, M = A; kwargs...) = begin
        s = DirectSolver(api_matrix(backend, tril(M), Int32), structure, 'L')
        setparam!(s, "solve_alg", "algo1")
        for (name, value) in kwargs
            setparam!(s, String(name), value)
        end
        thrown(() -> analyze!(s))
    end
    if !SDS.forward_progress(backend)            # the KA CPU backend (and Metal): the counter kernels need it
        @test analysis(spd_structure(T)) isa NotSupportedError
        continue
    end
    if T <: Complex                               # no conjugation in the fused kernels
        for structure in ("HPD", "H", "S")
            @test analysis(structure; pivot_type = 'N') isa NotSupportedError
        end
        continue
    end
    @test analysis("G", A) isa NotSupportedError                              # LU
    @test analysis("S", A) isa NotSupportedError                              # Bunch–Kaufman (the default)
    @test analysis("S", A; pivot_type = 'D') isa NotSupportedError
    @test analysis("SPD", A; schur_mode = 1, user_schur_indices = [i > n - 3 ? 1 : 0 for i in 1:n]) isa
          NotSupportedError
    rowptr, colval, nzval = host_csr_arrays(tril(A), Int32)                 # a uniform batch
    sb = DirectSolver(to_device(backend, rowptr), to_device(backend, colval), to_device(backend, hcat(nzval, nzval)),
                      "SPD", 'L')
    setparam!(sb, "solve_alg", "algo1")
    @test thrown(() -> analyze!(sb)) isa NotSupportedError
    # at the solve: nrhs > 1, a transposed solve
    s = algo1_solver(backend, A, "SPD")
    B2 = to_device(backend, rand(T, n, 2))
    @test thrown(() -> solve!(s, similar(B2), B2)) isa NotSupportedError
    setparam!(s, "solve_mode", 1)
    @test thrown(() -> solve!(s, x, b)) isa NotSupportedError
    setparam!(s, "solve_mode", 0)
    solve!(s, x, b)                                                          # one right-hand side, n × 1 too
    X1 = to_device(backend, zeros(T, n, 1))
    solve!(s, X1, reshape(b, n, 1); asynchronous = false)
    @test relres(A, vec(to_host(X1)), to_host(b)) <= tol(T)
    # "algo1" set after an analysis or a factorization with "default": refused, never the level sweeps
    s = DirectSolver(api_matrix(backend, tril(A), Int32), "SPD", 'L')
    analyze!(s); factorize!(s)
    setparam!(s, "solve_alg", "algo1")
    @test thrown(() -> solve!(s, x, b)) isa NotSupportedError
    s = algo1_solver(backend, A, "SPD")
    setparam!(s, "solve_alg", "default")
    refactorize!(s)
    setparam!(s, "solve_alg", "algo1")
    @test thrown(() -> solve!(s, x, b)) isa NotSupportedError
    refactorize!(s)
    solve!(s, x, b; asynchronous = false)
    @test relres(A, to_host(x), to_host(b)) <= tol(T)
    # "S" with pivoting switched on after the analysis: the factorization refuses it
    s = algo1_solver(backend, random_symindef(T, 60, 0.1), "S")
    setparam!(s, "pivot_type", 'B')
    @test thrown(() -> refactorize!(s)) isa NotSupportedError
end
