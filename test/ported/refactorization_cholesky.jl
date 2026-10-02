# Port of `refactorization_cholesky()` (CUDSS.jl test/test_cudss.jl): a failed
# Cholesky factorization (indefinite matrix), then a refactorization of the
# shifted matrix through `update!`.
#
# Changes: `A A' - 20 I` is `random_spd(T, n, 0.01) - (n + 20) I`. CUDSS.jl
# expects `info == 1`, cuDSS's position of the first failed pivot; PLAN §1.4
# reports it in the original numbering, so the expected value is the original
# column of the first pivot, `perm_row[1]` (every diagonal entry of `B Bᴴ - 20 I`
# is negative for this sparse `B`).

function ported_refactorization_cholesky(backend, ::Type{T}, ::Type{INT}) where {T, INT}
    n, p = 100, 5
    A_cpu = random_spd(T, n, 0.01) - (n + 20) * I
    X_cpu = zeros(T, n, p)
    B_cpu = rand(T, n, p)

    A_gpu = api_matrix(backend, triu(A_cpu), INT)
    X_gpu = to_device(backend, X_cpu)
    B_gpu = to_device(backend, B_cpu)

    structure = T <: Real ? "SPD" : "HPD"
    solver = DirectSolver(A_gpu, structure, 'U')

    execute!("analysis", solver, X_gpu, B_gpu)
    execute!("factorization", solver, X_gpu, B_gpu)
    execute!("solve", solver, X_gpu, B_gpu)

    info = getparam(solver, "info")
    @test info == getparam(solver, "perm_row")[1]

    A_cpu2 = A_cpu + 21 * I
    update!(solver, api_matrix(backend, triu(A_cpu2), INT))

    execute!("refactorization", solver, X_gpu, B_gpu)
    execute!("solve", solver, X_gpu, B_gpu)

    info = getparam(solver, "info")
    @test info == 0
    @test relres(A_cpu2, to_host(X_gpu), B_cpu) <= tol(T)
    return nothing
end

@testset "refactorization ($(backend_name(backend)), $T, $INT)" for backend in BACKENDS, T in ELTYPES, INT in INTTYPES
    ported_refactorization_cholesky(backend, T, INT)
end
