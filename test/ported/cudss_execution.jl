# Port of `cudss_execution()`, testsets "SPD -- HPD" (T13) and "Symmetric -- Hermitian"
# (T15) (CUDSS.jl test/test_cudss.jl).
#
# Changes: global pivoting `pivot_type = 'C'/'R'` (with `reordering_alg =
# "algo2"` in CUDSS.jl) is not planned (PLAN §3.3): setting it raises
# `NotSupportedError`, and the case then runs with the default pivot type.
# "Symmetric -- Hermitian": the matrix is `random_symindef(T, n, 0.01)`
# (symmetric/Hermitian indefinite) instead of `sprand + I` symmetrized, and
# CUDSS.jl's shift `A + Diagonal(d)` with `d = rand(R, n)` is kept.

function ported_execution_spd(backend, ::Type{T}, ::Type{INT}, view, pivot) where {T, INT}
    n, p = 100, 5
    R = real(T)
    A_cpu = random_spd(T, n, 0.01)
    X_cpu = zeros(T, n, p)
    B_cpu = rand(T, n, p)

    A_gpu = api_matrix(backend, triangle_view(A_cpu, view), INT)
    X_gpu = to_device(backend, X_cpu)
    B_gpu = to_device(backend, B_cpu)

    structure = T <: Real ? "SPD" : "HPD"
    solver = DirectSolver(A_gpu, structure, view)
    if pivot in ('C', 'R')
        @test_throws NotSupportedError setparam!(solver, "pivot_type", pivot)
    else
        setparam!(solver, "pivot_type", pivot)
        @test getparam(solver, "pivot_type") == pivot
    end

    execute!("analysis", solver, X_gpu, B_gpu)
    execute!("factorization", solver, X_gpu, B_gpu)
    execute!("solve", solver, X_gpu, B_gpu)

    @test relres(A_cpu, to_host(X_gpu), B_cpu) <= tol(T)

    # In-place LLᵀ / LLᴴ
    d_cpu = rand(R, n)
    A_cpu2 = A_cpu + Diagonal(d_cpu)
    update!(solver, api_matrix(backend, triangle_view(A_cpu2, view), INT))

    C_cpu = rand(T, n, p)
    C_gpu = to_device(backend, C_cpu)

    execute!("refactorization", solver, X_gpu, C_gpu)
    execute!("solve", solver, X_gpu, C_gpu)

    @test relres(A_cpu2, to_host(X_gpu), C_cpu) <= tol(T)
    return nothing
end

function ported_execution_sym(backend, ::Type{T}, ::Type{INT}, view, pivot) where {T, INT}
    n, p = 100, 5
    R = real(T)
    A_cpu = random_symindef(T, n, 0.01)
    X_cpu = zeros(T, n, p)
    B_cpu = rand(T, n, p)

    A_gpu = api_matrix(backend, triangle_view(A_cpu, view), INT)
    X_gpu = to_device(backend, X_cpu)
    B_gpu = to_device(backend, B_cpu)

    structure = T <: Real ? "S" : "H"
    solver = DirectSolver(A_gpu, structure, view)
    if pivot in ('C', 'R')
        @test_throws NotSupportedError setparam!(solver, "pivot_type", pivot)
    else
        setparam!(solver, "pivot_type", pivot)
        @test getparam(solver, "pivot_type") == pivot
    end

    execute!("analysis", solver, X_gpu, B_gpu)
    execute!("factorization", solver, X_gpu, B_gpu)
    execute!("solve", solver, X_gpu, B_gpu)

    @test relres(A_cpu, to_host(X_gpu), B_cpu) <= tol(T)

    # In-place LDLᵀ / LDLᴴ
    d_cpu = rand(R, n)
    A_cpu2 = A_cpu + Diagonal(d_cpu)
    update!(solver, api_matrix(backend, triangle_view(A_cpu2, view), INT))

    C_cpu = rand(T, n, p)
    C_gpu = to_device(backend, C_cpu)

    execute!("refactorization", solver, X_gpu, C_gpu)
    execute!("solve", solver, X_gpu, C_gpu)

    @test relres(A_cpu2, to_host(X_gpu), C_cpu) <= tol(T)
    return nothing
end

@testset "Symmetric -- Hermitian ($(backend_name(backend)), $T, $INT)" for backend in BACKENDS, T in ELTYPES,
                                                                           INT in INTTYPES
    @testset "view = $view" for view in ('F', 'L', 'U')
        @testset "Pivoting = $pivot" for pivot in ('C', 'R', 'N')
            ported_execution_sym(backend, T, INT, view, pivot)
        end
    end
end

@testset "SPD -- HPD ($(backend_name(backend)), $T, $INT)" for backend in BACKENDS, T in ELTYPES, INT in INTTYPES
    @testset "view = $view" for view in ('F', 'L', 'U')
        @testset "Pivoting = $pivot" for pivot in ('C', 'R', 'N')
            ported_execution_spd(backend, T, INT, view, pivot)
        end
    end
end
