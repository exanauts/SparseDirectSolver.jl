# Shared test infrastructure: backends, device transfers, matrix generators (T01).

@testset "backends" begin
    names = backend_name.(BACKENDS)
    @test BACKENDS[1] isa CPU
    @test count(==("CPU"), names) == 1
    cuda_expected = TEST_GPU && CUDA_LOADED && CUDA.functional()
    @test ("CUDA" in names) == cuda_expected
    rocm_expected = TEST_GPU && AMDGPU_LOADED && AMDGPU.functional()
    @test ("ROCm" in names) == rocm_expected
    @info "Active backends: $(join(names, ", "))"
    if CUDA_LOADED  # loading CUDA triggers the package extension
        @test Base.get_extension(SparseDirectSolver, :SparseDirectSolverCUDAExt) !== nothing
    end
    for backend in BACKENDS
        x = rand(Float32, 7)
        dx = to_device(backend, x)
        @test KernelAbstractions.get_backend(dx) == backend
        @test to_host(dx) == x
        A = sprand(ComplexF64, 9, 9, 0.3) + I
        @test to_host(to_device(backend, A)) == A
        for INT in INTTYPES
            dA = to_device(backend, A, INT)
            @test to_host(dA) == A
            @test eltype(dA) == ComplexF64
            @test index_eltype(dA) == INT   # issue #30: vendor constructors silently use Int32
        end
    end
end

@testset "generators: $T" for T in ELTYPES
    hermitian_ok(A) = T <: Real ? issymmetric(A) : ishermitian(A)

    for A in (laplacian2d(T, 4, 5), laplacian3d(T, 3, 3, 4), random_spd(T, 40, 0.1))
        @test A isa SparseMatrixCSC{T, Int}
        @test hermitian_ok(A)
        @test isposdef(Matrix(A))
    end
    @test size(laplacian2d(T, 4, 5)) == (20, 20)
    @test size(laplacian3d(T, 3, 3, 4)) == (36, 36)
    @test nnz(laplacian2d(T, 4, 5)) == 20 + 2 * (3 * 5 + 4 * 4)
    if T <: Complex
        A = random_hpd(T, 40, 0.1)
        @test A isa SparseMatrixCSC{T, Int}
        @test ishermitian(A) && !issymmetric(A)
        @test isposdef(Matrix(A))
    end

    A = random_symindef(T, 60, 0.08)
    @test A isa SparseMatrixCSC{T, Int}
    @test hermitian_ok(A)
    npos, nneg, nzero = eigen_inertia(A)
    @test npos > 0 && nneg > 0 && nzero == 0
    @test (npos, nneg) == (count(>(0), real(diag(A))), count(<(0), real(diag(A))))
    @test minimum(abs, eigvals(Hermitian(Matrix(A)))) >= 1 - 1.0e-4
    if T <: Complex
        As = random_symindef(T, 30, 0.1; hermitian = false)
        @test issymmetric(As) && !ishermitian(As)
        @test minimum(svdvals(Matrix(As))) > 0.1
    end

    for (δ, hessian) in ((1.0e-8, :spd), (0.0, :spd), (1.0e-2, :indefinite))
        K = kkt_matrix(T, 30, 10, δ; hessian)
        @test K isa SparseMatrixCSC{T, Int}
        @test size(K) == (40, 40)
        @test hermitian_ok(K)
        @test all(i -> K[i, i] == T(-δ), 31:40)
        rows = rowvals(K)
        @test all(i -> i in rows[nzrange(K, i)], 31:40)  # the (2,2) diagonal is stored, also for δ = 0
        npos, nneg, nzero = eigen_inertia(K; atol = δ / 10)
        if hessian === :spd
            @test (npos, nneg, nzero) == (30, 10, 0)
        else
            @test npos > 0 && nneg > 10 && nzero == 0
        end
    end

    G = random_general(T, 40, 0.1)
    @test G isa SparseMatrixCSC{T, Int}
    @test !issymmetric(G) && !ishermitian(G)
    @test opnorm(inv(Matrix(G)), Inf) <= 1 + 1.0e-3  # Varah bound for row diagonal dominance

    n, j = 12, 5
    for stored_zero in (false, true)
        Z = singular_block_matrix(T, n, j; stored_zero)
        @test Z isa SparseMatrixCSC{T, Int}
        @test hermitian_ok(Z)
        @test nnz(Z[:, j]) == (stored_zero ? 1 : 0) && nnz(Z[j, :]) == (stored_zero ? 1 : 0)
        @test Z[j, j] == 0
        @test rank(Matrix(Z)) == n - 1
        @test (stored_zero ? nnz(Z) - 1 : nnz(Z)) == nnz(singular_block_matrix(T, n, j))
        Zpos = copy(Z); Zpos[j, j] = 1
        @test isposdef(Matrix(Zpos))
        Zneg = copy(Z); Zneg[j, j] = -1
        @test !isposdef(Matrix(Zneg))
        @test cholesky(Hermitian(Matrix(Zneg)); check = false).info == j  # natural ordering fails at j
    end
end

@testset "default element type" begin
    @test eltype(laplacian2d(5, 5)) == Float64
    @test eltype(laplacian3d(2, 2, 2)) == Float64
    @test eltype(random_spd(20, 0.1)) == Float64
    @test eltype(random_hpd(20, 0.1)) == ComplexF64
    @test eltype(random_symindef(20, 0.1)) == Float64
    @test eltype(kkt_matrix(20, 5, 1.0e-8)) == Float64
    @test eltype(random_general(20, 0.1)) == Float64
    @test eltype(singular_block_matrix(10, 3)) == Float64
end

@testset "generators are reproducible from the seed" begin
    Random.seed!(1234)
    A1 = random_spd(Float64, 50, 0.05)
    Random.seed!(1234)
    A2 = random_spd(Float64, 50, 0.05)
    @test A1 == A2
    @test random_symindef(Float64, 30, 0.1; rng = Xoshiro(7)) == random_symindef(Float64, 30, 0.1; rng = Xoshiro(7))
end

@testset "CUDSS.jl doc examples: $T" for T in ELTYPES
    for ex in (schur_example_lu(T), schur_example_ldlt(T), schur_example_cholesky(T))
        A, S = Matrix(ex.A), ex.S
        @test ex.A isa SparseMatrixCSC{T, Int}
        @test ex.A * ex.x == ex.b
        s = findall(==(1), ex.schur_indices)
        r = findall(==(0), ex.schur_indices)
        @test S ≈ A[s, s] - A[s, r] * (A[r, r] \ A[r, s])
    end
    @test issymmetric(schur_example_ldlt(T).A)
    @test isposdef(Matrix(schur_example_cholesky(T).A))

    ex = ubatch_example(T)
    @test length(ex.A) == ex.nbatch == 3
    @test length(ex.nzval) == ex.nbatch * length(ex.colval)
    @test length(ex.b) == ex.n * ex.nbatch
    nnzA = length(ex.colval)
    for (k, Ak) in enumerate(ex.A)
        @test Ak isa SparseMatrixCSC{T, Int}
        vals = ex.nzval[((k - 1) * nnzA + 1):(k * nnzA)]
        B = zeros(T, ex.n, ex.n)
        for i in 1:ex.n, p in ex.rowptr[i]:(ex.rowptr[i + 1] - 1)
            B[i, ex.colval[p]] = vals[p]
        end
        @test B == Matrix(Ak)
        λ = ex.Λ[k]
        @test Matrix(Ak) == T[1+λ 0 3; 4 5+λ 0; 2 6 2+λ]
    end
end

@testset "tolerances and residuals" begin
    @test tol(Float32) == sqrt(eps(Float32))
    @test tol(ComplexF64) == sqrt(eps(Float64))
    A = laplacian2d(Float64, 5, 5)
    x = ones(25)
    @test relres(A, x, A * x) == 0
    @test relres(A, zeros(25), zeros(25)) == 0
    @test spd_structure(Float32) == "SPD" && spd_structure(ComplexF32) == "HPD"
    @test sym_structure(Float64) == "S" && sym_structure(ComplexF64) == "H"
end
