# Port of `cudss_inertia_matching()` (CUDSS.jl test/test_cudss.jl, T21): the
# inertia of an SPD matrix factored as "S" with and without matching.
#
# Changes: cuDSS (through at least 0.8) reports `(0, 0)` once matching is enabled
# and CUDSS.jl marks that check `@test_broken`; here the symmetric scaling keeps
# the inertia (PLAN §1.4), so it is a plain `@test`, for every matching
# algorithm; every `T` (complex: structure "H", Hermitian) and `INT`; the SPD
# matrix is `random_spd(T, n, 0.4)` (CUDSS.jl: `B + Bᴴ + 2n I`).

@testset "inertia under matching ($(backend_name(backend)), $T, $INT)" for backend in BACKENDS, T in ELTYPES,
                                                                            INT in INTTYPES
    n = 16
    A_cpu = random_spd(T, n, 0.4)
    A_gpu = api_matrix(backend, A_cpu, INT)
    x_gpu = to_device(backend, zeros(T, n))
    b_cpu = rand(T, n)
    b_gpu = to_device(backend, b_cpu)
    expected = (INT(n), INT(0))
    structure = sym_structure(T)
    @testset "matching_alg = $alg" for alg in ("default", "algo1", "algo2", "algo3", "algo4", "algo5", "algo6")
        solver = DirectSolver(A_gpu, structure, 'F')
        setparam!(solver, "matching_alg", alg)
        execute!("analysis", solver, x_gpu, b_gpu)
        execute!("factorization", solver, x_gpu, b_gpu)
        @test getparam(solver, "inertia") == expected
        execute!("solve", solver, x_gpu, b_gpu)
        @test relres(A_cpu, to_host(x_gpu), b_cpu) <= tol(T)
    end
end
