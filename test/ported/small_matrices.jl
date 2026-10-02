# Port of `small_matrices()`, testset "LLᵀ / LLᴴ" (CUDSS.jl test/test_cudss.jl):
# systems of size 1 to 16 with every view.

function ported_llt(backend, ::Type{T}, ::Type{INT}, A_cpu, x_cpu, b_cpu, uplo) where {T, INT}
    A_gpu = api_matrix(backend, triangle_view(A_cpu, uplo), INT)
    x_gpu = to_device(backend, x_cpu)
    b_gpu = to_device(backend, b_cpu)

    structure = T <: Real ? "SPD" : "HPD"
    solver = DirectSolver(A_gpu, structure, uplo)

    execute!("analysis", solver, x_gpu, b_gpu)
    execute!("factorization", solver, x_gpu, b_gpu)
    execute!("solve", solver, x_gpu, b_gpu)

    return relres(A_cpu, to_host(x_gpu), b_cpu)
end

@testset "LLᵀ / LLᴴ ($(backend_name(backend)), $T, $INT)" for backend in BACKENDS, T in ELTYPES, INT in INTTYPES
    @testset "Size of the linear system: $n" for n in 1:16
        A_cpu = random_spd(T, n, 0.01)
        x_cpu = zeros(T, n)
        b_cpu = rand(T, n)
        @testset "uplo = $uplo" for uplo in ('L', 'U', 'F')
            @test ported_llt(backend, T, INT, A_cpu, x_cpu, b_cpu, uplo) <= tol(T)
        end
    end
end
