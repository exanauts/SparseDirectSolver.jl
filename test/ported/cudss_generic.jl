# Port of `cudss_generic()`, testsets "SPD -- HPD" (T13), "Symmetric --
# Hermitian" (`ldlt`, `ldlt!`, T15) and "Unsymmetric -- Non-Hermitian" (`lu`,
# `lu!`, T19) (CUDSS.jl test/test_cudss.jl).
#
# Changes: on the CPU backend the matrix is a `CSR` (`cholesky(::SparseMatrixCSC)`
# is CHOLMOD's); `CudssMatrix(x)` is `MatrixDescriptor(x)`. `rand(R)` scalings
# are shifted to `[1/2, 3/2)` so no scaled matrix is nearly zero (the complex
# `rand(T)` scalings of the LU part likewise, by `1/2` on the real part). The
# unsymmetric matrix is `random_general(T, n, 0.02)` instead of
# `sprand(T, n, n, 0.02) + I`.

function ported_generic_lu(backend, ::Type{T}, ::Type{INT}) where {T, INT}
    n = 100
    R = real(T)
    A_cpu = random_general(T, n, 0.02)
    b_cpu = rand(T, n)
    x_cpu = zeros(T, n)

    A_gpu = api_matrix(backend, A_cpu, INT)
    b_gpu = to_device(backend, b_cpu)
    scaled(c) = api_matrix(backend, c * A_cpu, INT)

    @testset "lu!" begin
        solver = DirectSolver(A_gpu, "G", 'F')
        x_gpu = to_device(backend, x_cpu)
        execute!("analysis", solver, x_gpu, b_gpu)
        solver = lu!(solver, A_gpu)
        @test !solver.fresh_factorization
    end

    @testset "ldiv!" begin
        solver = lu(A_gpu)
        x_gpu = to_device(backend, x_cpu)
        ldiv!(x_gpu, solver, b_gpu)
        @test relres(A_cpu, to_host(x_gpu), b_cpu) <= tol(T)

        c = rand(T) + R(0.5)
        lu!(solver, scaled(c))
        x_gpu .= b_gpu
        ldiv!(solver, x_gpu)
        @test relres(c * A_cpu, to_host(x_gpu), b_cpu) <= tol(T)

        c = rand(T) + R(0.5)
        lu!(solver, scaled(c))
        x_gpu .= b_gpu
        x_desc = MatrixDescriptor(x_gpu)
        ldiv!(solver, x_desc)
        @test relres(c * A_cpu, to_host(x_gpu), b_cpu) <= tol(T)

        c = rand(T) + R(0.5)
        lu!(solver, scaled(c))
        x_desc = MatrixDescriptor(x_gpu)
        b_desc = MatrixDescriptor(b_gpu)
        ldiv!(x_desc, solver, b_desc)
        @test relres(c * A_cpu, to_host(x_gpu), b_cpu) <= tol(T)
    end

    @testset "\\" begin
        solver = lu(A_gpu)
        x_gpu = solver \ b_gpu
        @test relres(A_cpu, to_host(x_gpu), b_cpu) <= tol(T)

        c = rand(T) + R(0.5)
        lu!(solver, scaled(c))
        x_gpu = solver \ b_gpu
        @test relres(c * A_cpu, to_host(x_gpu), b_cpu) <= tol(T)
    end
    return nothing
end

function ported_generic_spd(backend, ::Type{T}, ::Type{INT}, view) where {T, INT}
    n, p = 100, 5
    R = real(T)
    A_cpu = random_spd(T, n, 0.01)
    B_cpu = rand(T, n, p)
    X_cpu = zeros(T, n, p)

    A_gpu = api_matrix(backend, triangle_view(A_cpu, view), INT)
    B_gpu = to_device(backend, B_cpu)
    scaled(c) = api_matrix(backend, triangle_view(c * A_cpu, view), INT)

    @testset "cholesky!" begin
        structure = T <: Real ? "SPD" : "HPD"
        solver = DirectSolver(A_gpu, structure, view)
        X_gpu = to_device(backend, X_cpu)
        execute!("analysis", solver, X_gpu, B_gpu)
        solver = cholesky!(solver, A_gpu)
        @test !solver.fresh_factorization
    end

    @testset "ldiv!" begin
        solver = cholesky(A_gpu; view)
        X_gpu = to_device(backend, X_cpu)
        ldiv!(X_gpu, solver, B_gpu)
        @test relres(A_cpu, to_host(X_gpu), B_cpu) <= tol(T)

        c = rand(R) + R(0.5)
        cholesky!(solver, scaled(c))
        X_gpu .= B_gpu
        ldiv!(solver, X_gpu)
        @test relres(c * A_cpu, to_host(X_gpu), B_cpu) <= tol(T)

        c = rand(R) + R(0.5)
        cholesky!(solver, scaled(c))
        X_gpu .= B_gpu
        X_desc = MatrixDescriptor(X_gpu)
        ldiv!(solver, X_desc)
        @test relres(c * A_cpu, to_host(X_gpu), B_cpu) <= tol(T)

        c = rand(R) + R(0.5)
        cholesky!(solver, scaled(c))
        X_desc = MatrixDescriptor(X_gpu)
        B_desc = MatrixDescriptor(B_gpu)
        ldiv!(X_desc, solver, B_desc)
        @test relres(c * A_cpu, to_host(X_gpu), B_cpu) <= tol(T)
    end

    @testset "\\" begin
        solver = cholesky(A_gpu; view)
        X_gpu = solver \ B_gpu
        @test relres(A_cpu, to_host(X_gpu), B_cpu) <= tol(T)

        c = rand(R) + R(0.5)
        cholesky!(solver, scaled(c))
        X_gpu = solver \ B_gpu
        @test relres(c * A_cpu, to_host(X_gpu), B_cpu) <= tol(T)
    end
    return nothing
end

function ported_generic_sym(backend, ::Type{T}, ::Type{INT}, view) where {T, INT}
    n, p = 100, 5
    R = real(T)
    A_cpu = random_symindef(T, n, 0.01)
    B_cpu = rand(T, n, p)
    X_cpu = rand(T, n, p)

    A_gpu = api_matrix(backend, triangle_view(A_cpu, view), INT)
    B_gpu = to_device(backend, B_cpu)
    scaled(c) = api_matrix(backend, triangle_view(c * A_cpu, view), INT)

    @testset "ldlt!" begin
        structure = T <: Real ? "S" : "H"
        solver = DirectSolver(A_gpu, structure, view)
        X_gpu = to_device(backend, X_cpu)
        execute!("analysis", solver, X_gpu, B_gpu)
        solver = ldlt!(solver, A_gpu)
        @test !solver.fresh_factorization
    end

    @testset "ldiv!" begin
        solver = ldlt(A_gpu; view)
        X_gpu = to_device(backend, X_cpu)
        ldiv!(X_gpu, solver, B_gpu)
        @test relres(A_cpu, to_host(X_gpu), B_cpu) <= tol(T)

        c = rand(R) + R(0.5)
        ldlt!(solver, scaled(c))
        X_gpu .= B_gpu
        ldiv!(solver, X_gpu)
        @test relres(c * A_cpu, to_host(X_gpu), B_cpu) <= tol(T)

        c = rand(R) + R(0.5)
        ldlt!(solver, scaled(c))
        X_gpu .= B_gpu
        X_desc = MatrixDescriptor(X_gpu)
        ldiv!(solver, X_desc)
        @test relres(c * A_cpu, to_host(X_gpu), B_cpu) <= tol(T)

        c = rand(R) + R(0.5)
        ldlt!(solver, scaled(c))
        X_desc = MatrixDescriptor(X_gpu)
        B_desc = MatrixDescriptor(B_gpu)
        ldiv!(X_desc, solver, B_desc)
        @test relres(c * A_cpu, to_host(X_gpu), B_cpu) <= tol(T)
    end

    @testset "\\" begin
        solver = ldlt(A_gpu; view)
        X_gpu = solver \ B_gpu
        @test relres(A_cpu, to_host(X_gpu), B_cpu) <= tol(T)

        c = rand(R) + R(0.5)
        ldlt!(solver, scaled(c))
        X_gpu = solver \ B_gpu
        @test relres(c * A_cpu, to_host(X_gpu), B_cpu) <= tol(T)
    end
    return nothing
end

@testset "Unsymmetric -- Non-Hermitian ($(backend_name(backend)), $T, $INT)" for backend in BACKENDS, T in ELTYPES,
                                                                                 INT in INTTYPES
    ported_generic_lu(backend, T, INT)
end

@testset "Symmetric -- Hermitian ($(backend_name(backend)), $T, $INT)" for backend in BACKENDS, T in ELTYPES,
                                                                           INT in INTTYPES
    @testset "view = $view" for view in ('F', 'L', 'U')
        ported_generic_sym(backend, T, INT, view)
    end
end

@testset "SPD -- HPD ($(backend_name(backend)), $T, $INT)" for backend in BACKENDS, T in ELTYPES, INT in INTTYPES
    @testset "view = $view" for view in ('F', 'L', 'U')
        ported_generic_spd(backend, T, INT, view)
    end
end
