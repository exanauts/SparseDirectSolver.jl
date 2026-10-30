# FusedSolver (T33): the fused factorization and partitioned-inverse solve of src/fused.jl behind the
# handle-layer verbs. A split file (one part per element type).
#
# The first test file that leaves the CPU backend out: the fused kernels synchronize through dependency
# counters on which workgroups spin, which needs every workgroup resident on the device; the KernelAbstractions
# CPU backend runs the workgroups one after another and would deadlock. On CPU() the file asserts that the
# constructor raises NotSupportedError instead.

const FUSED_BACKENDS = filter(b -> !(b isa CPU), BACKENDS)

# the stored lower triangle of A on `backend`, as a user passes it (a vendor CSR on the GPUs)
fused_matrix(backend, A, ::Type{INT} = Int32) where {INT} = api_matrix(backend, tril(A), INT)

# the CSR values of the lower triangle of A, in the order of `fused_matrix`
fused_values(A, ::Type{INT} = Int32) where {INT} = host_csr_arrays(tril(A), INT)[3]

# analysis + factorization + solve of a fresh FusedSolver; returns (F, Ad, x, relres)
function fused_solve(backend, A::SparseMatrixCSC{T}, structure; INT = Int32, perm = nothing, kwargs...) where {T}
    Ad = fused_matrix(backend, A, INT)
    F = FusedSolver(Ad, structure, 'L'; kwargs...)
    perm === nothing || setparam!(F, "user_perm", perm)
    analyze!(F)
    factorize!(F)
    b = rand(T, size(A, 1))
    bd = to_device(backend, b)
    xd = similar(bd)
    solve!(F, xd, bd; asynchronous = false)
    return F, Ad, xd, relres(A, to_host(xd), b)
end

# the host bytes one refactorization and one solve allocate (the second call: the first may compile)
function fused_cycle_allocated(F, x, b)
    cycle() = (refactorize!(F); solve!(F, x, b))
    cycle()
    return @allocated cycle()
end

# The SPD generators of test/matrices.jl (the well-conditioned ones: `tol(T)` gates the residual), at sizes
# that give regime-A subtrees, regime-C fronts (wide, handled by the tuning's wide strategy) and, for the
# 18³ Laplacian, a root front wider than 256 (the vendor dense path of the solve).
fused_spd_generators(::Type{T}) where {T} = (
    ("laplacian2d 40×40", () -> laplacian2d(T, 40, 40)),
    ("laplacian3d 10³", () -> laplacian3d(T, 10, 10, 10)),
    ("laplacian3d 18³", () -> laplacian3d(T, 18, 18, 18)),
    ("random_spd 500", () -> random_spd(T, 500, 0.01)),
    ("schur_example_cholesky 5×5", () -> schur_example_cholesky(T).A),
)

# the symmetric indefinite generators, factored as "S" without pivoting: diagonally dominant (random_symindef)
# or with a regularized dual block (kkt_matrix with an indefinite Hessian, δ = 1)
fused_indefinite_generators(::Type{T}) where {T} = (
    ("random_symindef 400", () -> random_symindef(T, 400, 0.01)),
    ("kkt_matrix 300+150, indefinite H", () -> kkt_matrix(T, 300, 150, 1.0; hessian = :indefinite)),
)

RUN_SHARED && any(b -> b isa CPU, BACKENDS) && @testset "CPU backend: NotSupportedError" begin
    A = laplacian2d(Float64, 8, 8)
    for structure in ("SPD", "S")
        e = thrown(() -> FusedSolver(fused_matrix(CPU(), A), structure, 'L'))
        @test e isa NotSupportedError
        @test occursin("CPU backend", sprint(showerror, e))
    end
    # the raw-array constructor too
    rowptr, colval, nzval = host_csr_arrays(tril(A), Int32)
    @test_throws NotSupportedError FusedSolver(rowptr, colval, nzval, "SPD", 'L')
end

@testset "FusedSolver on $(backend_name(backend))" for backend in FUSED_BACKENDS
    @testset "\"SPD\" ($label): $T" for T in eltypes_among((Float32, Float64)),
                                         (label, gen) in fused_spd_generators(T)
        Random.seed!(666)
        A = gen()
        n = size(A, 1)
        F, Ad, xd, r = fused_solve(backend, A, "SPD")
        @test getparam(F, "info") == 0
        @test r <= tol(T)
        @test getparam(F, "inertia") == (n, 0)
        @test getparam(F, "npivots") == 0
        # the solution of the stock DirectSolver on the same ordering
        b = rand(T, n)
        bd = to_device(backend, b)
        x = F \ bd
        D = DirectSolver(fused_matrix(backend, A), "SPD", 'L')
        setparam!(D, "user_perm", getparam(F, "perm_reorder_row"))
        analyze!(D)
        factorize!(D)
        xs = to_host(D \ bd)
        @test norm(to_host(x) - xs) / norm(xs) <= tol(T)
        @test getparam(F, "lu_nnz") == getparam(D, "lu_nnz")
        # new values written into the aliased buffer (the interior-point loop), then refactorization
        A2 = T(2) * A + T(n) * I
        copyto!(Ad.nzVal, fused_values(A2))
        refactorize!(F)
        @test getparam(F, "info") == 0
        solve!(F, xd, bd; asynchronous = false)
        @test relres(A2, to_host(xd), b) <= tol(T)
        # update! with the same pattern points the solver at another buffer
        Ad3 = fused_matrix(backend, A)
        update!(F, Ad3)
        refactorize!(F)
        ldiv!(xd, F, bd)
        @test relres(A, to_host(xd), b) <= tol(T)
        @test getparam(F, "info") == 0
        @test_throws InvalidValueError update!(F, fused_matrix(backend, laplacian2d(T, 3, 3)))
        @test occursin("FusedSolver{$T, Int32}", sprint(show, F))
    end

    @testset "\"SPD\", Int64 indices: $T" for T in eltypes_among((Float32, Float64))
        Random.seed!(666)
        A = laplacian3d(T, 10, 10, 10)
        F, _, _, r = fused_solve(backend, A, "SPD"; INT = Int64)
        @test F isa FusedSolver{T, Int64}
        @test getparam(F, "info") == 0
        @test r <= tol(T)
    end

    @testset "\"SPD\": a non-positive pivot sets info: $T" for T in eltypes_among((Float32, Float64))
        Random.seed!(666)
        A = laplacian2d(T, 20, 20)
        j = 37
        A[j, j] = -1                                  # an indefinite matrix: Cholesky fails at the latest at j
        F, _, _, _ = fused_solve(backend, A, "SPD")
        @test getparam(F, "info") != 0
    end

    @testset "\"S\" ($label): $T" for T in eltypes_among((Float32, Float64)),
                                       (label, gen) in fused_indefinite_generators(T)
        Random.seed!(666)
        A = gen()
        n = size(A, 1)
        F, Ad, xd, r = fused_solve(backend, A, "S")
        @test getparam(F, "info") == 0
        @test r <= tol(T)
        @test getparam(F, "pivot_type") == 'N'
        # the inertia of the CPU reference LDLᵀ (the oracle)
        _, Nr, info_ref, _ = reference_ldlt(SparseMatrixCSC{Float64, Int}(A))
        @test info_ref == 0
        inertia = getparam(F, "inertia")
        @test inertia == SDS.inertia(Nr)
        @test inertia isa Tuple{Int32, Int32}
        @test getparam(F, "npivots") == 0
        # refactorization with new values in the aliased buffer (the same inertia: a congruence)
        Dg = Diagonal(T.(1 .+ rand(n)))
        A2 = Dg * A * Dg
        copyto!(Ad.nzVal, fused_values(A2))
        refactorize!(F)
        @test getparam(F, "info") == 0
        b = rand(T, n)
        bd = to_device(backend, b)
        solve!(F, xd, bd; asynchronous = false)
        @test relres(A2, to_host(xd), b) <= tol(T)
        @test getparam(F, "inertia") == inertia
    end

    @testset "\"S\": a zero pivot sets info without throwing: $T" for T in eltypes_among((Float32, Float64))
        n, j = 40, 17
        A = singular_block_matrix(T, n, j; stored_zero = true)
        F = FusedSolver(fused_matrix(backend, A), "S", 'L')
        analyze!(F)
        factorize!(F)
        @test getparam(F, "info") != 0
        refactorize!(F)
        @test getparam(F, "info") != 0
        # a regular matrix on the same pattern factors again (the per-front statuses are reset)
        A[j, j] = 3
        copyto!(F.inner.A.nzval, to_device(backend, fused_values(A)))
        refactorize!(F)
        @test getparam(F, "info") == 0
    end

    @testset "restrictions raise NotSupportedError: $T" for T in eltypes_among((Float32, Float64))
        Random.seed!(666)
        A = laplacian2d(T, 12, 12)
        n = size(A, 1)
        Ad = fused_matrix(backend, A)
        for structure in ("HPD", "H", "G")
            @test_throws NotSupportedError FusedSolver(structure == "G" ? api_matrix(backend, A, Int32) : Ad,
                                                       structure, structure == "G" ? 'F' : 'L')
        end
        # a uniform batch (nbatch = 2)
        rowptr, colval, nzval = host_csr_arrays(tril(A), Int32)
        @test_throws NotSupportedError FusedSolver(to_device(backend, rowptr), to_device(backend, colval),
                                                   to_device(backend, hcat(nzval, nzval)), "SPD", 'L')
        F = FusedSolver(Ad, "SPD", 'L')
        # every parameter outside the five of the analysis
        for (name, value) in (("schur_mode", 1), ("user_schur_indices", [i > n - 3 ? 1 : 0 for i in 1:n]),
                              ("matching_alg", "algo5"), ("ir_n_steps", 2), ("deterministic_mode", 1),
                              ("solve_mode", 1), ("pivot_type", 'B'), ("pivot_threshold", 0.1),
                              ("ubatch_size", 2), ("factor_precision", Float32), ("solve_alg", "algo1"),
                              ("info", 0))
            e = thrown(() -> setparam!(F, name, value))
            @test e isa NotSupportedError
        end
        @test_throws ArgumentError setparam!(F, "no_such_parameter", 1)     # unknown names as DirectSolver
        @test getparam(F, "reordering_alg") == "algo4"
        setparam!(F, "nd_seed", 5)
        @test getparam(F, "nd_seed") == 5
        # phases before their prerequisites
        @test_throws FactorizationError factorize!(F)
        @test_throws FactorizationError execute!("symbolic_factorization", F, nothing, nothing)
        analyze!(F)
        @test_throws FactorizationError refactorize!(F)
        @test_throws FactorizationError getparam(F, "inertia")
        factorize!(F)
        b = to_device(backend, rand(T, n))
        x = similar(b)
        # nrhs = 2, a transposed solve, the solve sub-phases, an unsupported data parameter
        B2 = to_device(backend, rand(T, n, 2))
        @test_throws NotSupportedError solve!(F, similar(B2), B2)
        @test_throws NotSupportedError F \ B2
        X1 = to_device(backend, zeros(T, n, 1))
        solve!(F, X1, reshape(b, n, 1); asynchronous = false)                    # an n × 1 matrix is one rhs
        @test relres(A, vec(to_host(X1)), to_host(b)) <= tol(T)
        for phase in ("solve_fwd_perm", "solve_fwd", "solve_diag", "solve_bwd", "solve_bwd_perm",
                      "solve_refinement", "solve_fwd_schur", "solve_bwd_schur")
            @test_throws NotSupportedError execute!(phase, F, x, b)
        end
        @test_throws NotSupportedError getparam(F, "diag")
        @test_throws NotSupportedError getparam(F, "pivot_stats")
        # options set on the inner solver behind the FusedSolver's back are refused at the phase
        F.inner.options.solve_mode = 1
        @test_throws NotSupportedError solve!(F, x, b)
        F.inner.options.solve_mode = 0
        for (field, value) in ((:deterministic_mode, 1), (:ir_n_steps, 2), (:schur_mode, 1),
                               (:matching_alg, SDS.MATCHING_MAX_DIAG_PRODUCT), (:factor_precision, T == Float64 ? Float32 : Float64))
            G = FusedSolver(Ad, "SPD", 'L')
            setfield!(G.inner.options, field, value)
            @test_throws NotSupportedError analyze!(G)
        end
        G = FusedSolver(fused_matrix(backend, random_symindef(T, 50, 0.1)), "S", 'L')
        G.inner.options.pivot_type = SDS.PIVOT_AUTO
        @test_throws NotSupportedError analyze!(G)
        @test_throws InvalidValueError solve!(F, to_device(backend, zeros(T, n + 1)), to_device(backend, zeros(T, n + 1)))
    end

    @testset "refactorization and solve allocate within the launch allowance: $T" for T in eltypes_among((Float32, Float64))
        Random.seed!(666)
        # two sizes: the allowance is per call, not per front
        for (structure, A) in (("SPD", laplacian3d(T, 10, 10, 10)), ("SPD", laplacian3d(T, 18, 18, 18)),
                               ("S", random_symindef(T, 400, 0.01)))
            F, _, xd, _ = fused_solve(backend, A, structure)
            bd = to_device(backend, rand(T, size(A, 1)))
            bytes = fused_cycle_allocated(F, xd, bd)
            @test bytes <= fused_cycle_alloc_budget(backend)
            bytes <= fused_cycle_alloc_budget(backend) ||
                @info "fused refactorization + solve on $(backend_name(backend)), $structure, $T: $bytes bytes"
        end
    end
end

for T in eltypes_among(Complex), backend in FUSED_BACKENDS
    @testset "complex element types raise NotSupportedError on $(backend_name(backend)): $T" begin
        for (structure, A) in (("HPD", random_hpd(T, 50, 0.1)), ("SPD", random_hpd(T, 50, 0.1)),
                               ("S", random_symindef(T, 50, 0.1; hermitian = false)),
                               ("H", random_symindef(T, 50, 0.1)))
            e = thrown(() -> FusedSolver(fused_matrix(backend, A), structure, 'L'))
            @test e isa NotSupportedError
        end
    end
end
