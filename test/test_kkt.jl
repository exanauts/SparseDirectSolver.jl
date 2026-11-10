# KKTSolver (src/kkt.jl): the front end for the KKT systems of an interior-point method, the successor of the MadNLP
# extension's SDSSolver (removed in 0.4.0). One symmetric matrix with a fixed pattern, factorized again with new
# values written into the aliased `nzval`, one right-hand side, the inertia read after every factorization.
#
# Real element types only: KKTSolver refuses complex ones. The pipeline is the fused one (a FusedSolver) where the
# backend has forward progress between workgroups and `pivoting = :none` (CUDA, ROCm, oneAPI), else the stock one (a
# DirectSolver): the CPU backend and Metal run `:cholesky` and `:ldl` + `:none` on the stock pipeline (the extension
# refused them on the CPU, #175), and `:bk` is stock everywhere. So every testset runs on every backend.

# the pipeline `fused = nothing` selects for `pivoting = :none` on `backend`
kkt_default_pipeline(backend) = SDS.forward_progress(backend) ? :fused : :stock

# The lower triangle of A in the layout of an interior-point code, as the arguments of KKTSolver and the value
# vector the solver aliases: a vendor CSC with view 'L' on a GPU backend with vendor sparse types (MadNLP's layout),
# else (the CPU backend, Metal) the raw arrays of the host CSC on `backend`, read as an upper CSR (view 'U').
function kkt_input(backend, A::SparseMatrixCSC{T}) where {T}
    L = SparseMatrixCSC{T, Int32}(tril(A))
    types = vendor_sparse_types(backend)
    if types === nothing
        colptr, rowval, nzval = to_device(backend, L.colptr), to_device(backend, L.rowval), to_device(backend, L.nzval)
        return (colptr, rowval, nzval, 'U'), nzval
    end
    _, SpCSC, DVec = types
    csc = SpCSC{T, Int32}(DVec{Int32}(L.colptr), DVec{Int32}(L.rowval), DVec{T}(L.nzval), size(L))
    return (csc, 'L'), csc.nzVal
end

# overwrite the aliased values with those of A2 (same pattern), as an interior-point method does between iterations
kkt_set_values!(nzval, A2::SparseMatrixCSC{T}) where {T} = copyto!(nzval, SparseMatrixCSC{T, Int32}(tril(A2)).nzval)

# the relative residuals of the in-place solve!(K, xb) and of the two-vector solve!(K, x, b) on random right-hand
# sides against the full matrix A, and whether the two-vector solve left b alone
function kkt_solves(backend, K, A::SparseMatrixCSC{T}) where {T}
    n = size(A, 1)
    b1 = rand(T, n)
    xb = to_device(backend, b1)
    same = solve!(K, xb) === xb
    r1 = relres(A, to_host(xb), b1)
    b2 = rand(T, n)
    b = to_device(backend, b2)
    x = to_device(backend, zeros(T, n))
    same &= solve!(K, x, b) === x
    r2 = relres(A, to_host(x), b2)
    return (; r1, r2, same, b_kept = to_host(b) == b2)
end

@testset "constructors on $(backend_name(backend))" for backend in BACKENDS
    @testset "laplacian2d 20×20: $T" for T in eltypes_among((Float32, Float64))
        A = laplacian2d(T, 20, 20)
        n = size(A, 1)
        L = SparseMatrixCSC{T, Int32}(tril(A))
        # the raw-array method: the lower-CSC arrays read as an upper CSR, on every backend
        Kr = KKTSolver(to_device(backend, L.colptr), to_device(backend, L.rowval), to_device(backend, L.nzval), 'U')
        @test Kr isa KKTSolver{T, Int32}
        @test size(Kr) == (n, n)
        @test size(Kr, 1) == n
        @test getparam(Kr, "info") == 0
        @test SDS.inertia(Kr) == (n, 0, 0)
        rr = kkt_solves(backend, Kr, A)
        @test rr.r1 <= tol(T)
        @test rr.r2 <= tol(T)
        @test rr.same
        @test rr.b_kept
        if vendor_sparse_types(backend) !== nothing
            # MadNLP's layout: the vendor CSC of the lower triangle with view 'L' is the same matrix as its arrays
            # read as an upper CSR
            (csc, uplo), _ = kkt_input(backend, A)
            @test uplo == 'L'
            K = KKTSolver(csc, uplo)
            @test K isa KKTSolver{T, Int32}
            @test getparam(K, "info") == 0
            @test SDS.inertia(K) == (n, 0, 0)
            b = rand(T, n)
            x, xr = to_device(backend, zeros(T, n)), to_device(backend, zeros(T, n))
            solve!(K, x, to_device(backend, b))
            solve!(Kr, xr, to_device(backend, b))
            @test relres(A, to_host(x), b) <= tol(T)
            @test isapprox(to_host(x), to_host(xr); rtol = tol(T))
        end
        if backend isa CPU
            # a SparseMatrixCSC on the CPU backend, each stored triangle
            for (uplo, M) in (('L', tril(A)), ('U', triu(A)), ('F', copy(A)))
                K = KKTSolver(M, uplo)
                @test K isa KKTSolver{T}
                @test SDS.pipeline(K) === :stock
                @test getparam(K, "info") == 0
                @test SDS.inertia(K) == (n, 0, 0)
                rs = kkt_solves(backend, K, A)
                @test rs.r1 <= tol(T)
                @test rs.r2 <= tol(T)
                # the values are read in place: new values in `nonzeros(M)` reach the next factorization
                nonzeros(M) .*= 2
                factorize!(K)
                @test getparam(K, "info") == 0
                rs = kkt_solves(backend, K, 2A)
                @test rs.r1 <= tol(T)
                @test rs.r2 <= tol(T)
            end
        end
    end
end

@testset "option validation on $(backend_name(backend))" for backend in BACKENDS
    @testset "$T" for T in eltypes_among((Float32, Float64))
        A = laplacian2d(T, 6, 6)
        args, _ = kkt_input(backend, A)
        @test_throws SDS.InvalidValueError KKTSolver(args...; factorization = :lu)
        @test_throws SDS.InvalidValueError KKTSolver(args...; factorization = :ldl, pivoting = :diag)
        # :bk needs the LDLᵀ factorization and the stock pipeline
        @test_throws SDS.InvalidValueError KKTSolver(args...; pivoting = :bk)
        @test_throws SDS.InvalidValueError KKTSolver(args...; factorization = :ldl, pivoting = :bk, fused = true)
        @test_throws SDS.InvalidValueError KKTSolver(args...; factorization = :ldl, pivot_threshold = 0.1)
        @test_throws SDS.InvalidValueError KKTSolver(args...; pivot_sign = ones(Int, size(A, 1)))
        if !SDS.forward_progress(backend)
            # the fused pipeline needs forward progress between workgroups (not the CPU backend, not Metal)
            @test_throws SDS.NotSupportedError KKTSolver(args...; fused = true)
            @test_throws SDS.NotSupportedError KKTSolver(args...; factorization = :ldl, fused = true)
        end
    end
    if backend isa CPU
        # a complex element type and a uniform batch are DirectSolver's
        A = laplacian2d(Float64, 6, 6)
        @test_throws SDS.NotSupportedError KKTSolver(SparseMatrixCSC{ComplexF64}(tril(A)), 'L')
        @test_throws SDS.NotSupportedError KKTSolver(SparseMatrixCSC{ComplexF64}(tril(A)), 'L'; factorization = :ldl)
        L = SparseMatrixCSC{Float64, Int32}(tril(A))
        @test_throws SDS.NotSupportedError KKTSolver(L.colptr, L.rowval, vcat(L.nzval, L.nzval), 'U')
    end
end

@testset "pipeline selection on $(backend_name(backend))" for backend in BACKENDS
    @testset "$T" for T in eltypes_among((Float32, Float64))
        A = laplacian2d(T, 12, 12)
        n = size(A, 1)
        args, _ = kkt_input(backend, A)
        expected = kkt_default_pipeline(backend)
        Kc = KKTSolver(args...)
        Kl = KKTSolver(args...; factorization = :ldl)
        Kb = KKTSolver(args...; factorization = :ldl, pivoting = :bk)
        @test SDS.pipeline(Kc) === expected
        @test SDS.pipeline(Kl) === expected
        @test SDS.pipeline(Kb) === :stock
        @test Kc.inner isa (expected === :fused ? SDS.FusedSolver : SDS.DirectSolver)
        @test Kb.inner isa SDS.DirectSolver
        backend isa CPU && @test expected === :stock
        if SDS.forward_progress(backend)
            # fused = false forces the stock pipeline, fused = true asks for the fused one
            for factorization in (:cholesky, :ldl)
                Ks = KKTSolver(args...; factorization, fused = false)
                @test SDS.pipeline(Ks) === :stock
                @test Ks.inner isa SDS.DirectSolver
                @test getparam(Ks, "info") == 0
                @test SDS.inertia(Ks) == (n, 0, 0)
                rs = kkt_solves(backend, Ks, A)
                @test rs.r1 <= tol(T)
                @test rs.r2 <= tol(T)
                @test SDS.pipeline(KKTSolver(args...; factorization, fused = true)) === :fused
            end
        end
        for K in (Kc, Kl, Kb)
            s = sprint(show, K)
            @test occursin("KKTSolver", s)
            @test occursin("pipeline = :$(SDS.pipeline(K))", s)
        end
    end
end

@testset "factorizations on $(backend_name(backend))" for backend in BACKENDS
    @testset ":cholesky, laplacian2d 30×30: $T" for T in eltypes_among((Float32, Float64))
        A = laplacian2d(T, 30, 30)
        n = size(A, 1)
        args, nzval = kkt_input(backend, A)
        K = KKTSolver(args...)
        @test SDS.pipeline(K) === kkt_default_pipeline(backend)
        @test getparam(K, "info") == 0
        @test SDS.inertia(K) == (n, 0, 0)
        rs = kkt_solves(backend, K, A)
        @test rs.r1 <= tol(T)
        @test rs.r2 <= tol(T)
        # the next interior-point iteration: new values in place, refactorization
        A2 = 2A
        kkt_set_values!(nzval, A2)
        @test factorize!(K) === K
        @test getparam(K, "info") == 0
        @test SDS.inertia(K) == (n, 0, 0)
        rs = kkt_solves(backend, K, A2)
        @test rs.r1 <= tol(T)
        @test rs.r2 <= tol(T)
    end

    # the KKT matrix of #141: under nested dissection (algo4) the root front is 80 wide with pivots of both signs
    @testset ":ldl, :none, grid_kkt_matrix 40×40, $reordering: $T" for T in eltypes_among((Float32, Float64)),
                                                                        reordering in ("algo4", "algo3")
        m = 40 * 40
        A = grid_kkt_matrix(T, 40, 40, 1.0)
        args, nzval = kkt_input(backend, A)
        K = KKTSolver(args...; factorization = :ldl, pivoting = :none, reordering)
        @test SDS.pipeline(K) === kkt_default_pipeline(backend)
        @test getparam(K, "info") == 0
        @test SDS.inertia(K) == (m, 0, m)
        rs = kkt_solves(backend, K, A)
        @test rs.r1 <= tol(T)
        @test rs.r2 <= tol(T)
        A2 = grid_kkt_matrix(T, 40, 40, 0.5)
        kkt_set_values!(nzval, A2)
        factorize!(K)
        @test getparam(K, "info") == 0
        @test SDS.inertia(K) == (m, 0, m)
        rs = kkt_solves(backend, K, A2)
        @test rs.r1 <= tol(T)
        @test rs.r2 <= tol(T)
    end

    # (A, A2): the matrix and the values of the next interior-point iteration on the same pattern. In both the
    # inertia is the sign count of the diagonal: random_symindef is strictly diagonally dominant, grid_kkt_matrix has
    # inertia (nx·ny, 0, nx·ny) with a positive primal and a negative dual diagonal.
    @testset ":ldl, :bk, $label: $T" for T in eltypes_among((Float32, Float64)),
                                         (label, gen) in (
                                             ("grid_kkt_matrix 40×40",
                                              () -> (grid_kkt_matrix(T, 40, 40, 1.0), grid_kkt_matrix(T, 40, 40, 0.5))),
                                             ("random_symindef 400",
                                              () -> (A = random_symindef(T, 400, 0.01); (A, 2A))))
        A, A2 = gen()
        npos, nneg = count(>(0), diag(A)), count(<(0), diag(A))
        args, nzval = kkt_input(backend, A)
        K = KKTSolver(args...; factorization = :ldl, pivoting = :bk, pivot_threshold = 0.01)
        @test SDS.pipeline(K) === :stock
        @test getparam(K, "info") == 0
        @test SDS.inertia(K) == (npos, 0, nneg)
        rs = kkt_solves(backend, K, A)
        @test rs.r1 <= tol(T)
        @test rs.r2 <= tol(T)
        kkt_set_values!(nzval, A2)
        factorize!(K)
        @test getparam(K, "info") == 0
        @test SDS.inertia(K) == (npos, 0, nneg)
        rs = kkt_solves(backend, K, A2)
        @test rs.r1 <= tol(T)
        @test rs.r2 <= tol(T)
    end

    # a failed factorization does not throw: "info" is nonzero; the regularized
    # matrix then factors
    @testset ":cholesky on an indefinite matrix: $T" for T in eltypes_among((Float32, Float64))
        n, j = 40, 17
        A = singular_block_matrix(T, n, j; stored_zero = true)
        A[j, j] = -1
        args, nzval = kkt_input(backend, A)
        K = KKTSolver(args...)
        @test getparam(K, "info") != 0
        factorize!(K)
        @test getparam(K, "info") != 0
        A[j, j] = 1
        kkt_set_values!(nzval, A)
        factorize!(K)
        @test getparam(K, "info") == 0
        @test SDS.inertia(K) == (n, 0, 0)
        rs = kkt_solves(backend, K, A)
        @test rs.r1 <= tol(T)
        @test rs.r2 <= tol(T)
    end
end

@testset "options on $(backend_name(backend))" for backend in BACKENDS
    # pivot_sign and pivot_epsilon on the stock no-pivot LDLᵀ: +1 for the primal, -1 for the dual rows, as an
    # interior-point method passes them; no pivot of grid_kkt_matrix is small, so the inertia is unchanged
    @testset "pivot_sign, pivot_epsilon (stock :ldl): $T" for T in eltypes_among((Float32, Float64))
        m = 30 * 30
        A = grid_kkt_matrix(T, 30, 30, 1.0)
        args, _ = kkt_input(backend, A)
        K = KKTSolver(args...; factorization = :ldl, fused = false, pivot_sign = vcat(fill(1, m), fill(-1, m)),
                      pivot_epsilon = 1.0e-10)
        @test SDS.pipeline(K) === :stock
        @test getparam(K, "pivot_epsilon") == 1.0e-10
        @test getparam(K, "info") == 0
        @test SDS.inertia(K) == (m, 0, m)
        rs = kkt_solves(backend, K, A)
        @test rs.r1 <= tol(T)
        @test rs.r2 <= tol(T)
        # a pivot_sign of the wrong length
        @test_throws SDS.InvalidValueError KKTSolver(args...; factorization = :ldl, fused = false,
                                                    pivot_sign = fill(1, m))
    end

    @testset "reordering = \"algo3\": $T" for T in eltypes_among((Float32, Float64))
        A = laplacian2d(T, 20, 20)
        n = size(A, 1)
        args, _ = kkt_input(backend, A)
        K = KKTSolver(args...; reordering = "algo3")
        @test getparam(K, "info") == 0
        @test SDS.inertia(K) == (n, 0, 0)
        rs = kkt_solves(backend, K, A)
        @test rs.r1 <= tol(T)
        @test rs.r2 <= tol(T)
    end

    # a random user_perm (n = 100: every front stays within the 256 columns of the fused solve)
    @testset "user_perm: $T" for T in eltypes_among((Float32, Float64))
        A = laplacian2d(T, 10, 10)
        n = size(A, 1)
        args, _ = kkt_input(backend, A)
        for (factorization, pivoting) in ((:cholesky, :none), (:ldl, :bk))
            K = KKTSolver(args...; factorization, pivoting, user_perm = randperm(n))
            @test getparam(K, "info") == 0
            @test SDS.inertia(K) == (n, 0, 0)
            rs = kkt_solves(backend, K, A)
            @test rs.r1 <= tol(T)
            @test rs.r2 <= tol(T)
        end
        @test_throws SDS.InvalidValueError KKTSolver(args...; user_perm = randperm(n - 1))
    end

    # both pipelines: `fused = false` where the default is the fused one
    @testset "tuning, fused = $fused: $T" for T in eltypes_among((Float32, Float64)),
                                              fused in (SDS.forward_progress(backend) ? (nothing, false) : (nothing,))
        A = laplacian2d(T, 20, 20)
        n = size(A, 1)
        args, _ = kkt_input(backend, A)
        K = KKTSolver(args...; fused, tuning = SDSTuning(; nd_nseps = 2))
        @test SDS.pipeline(K) === (fused === false ? :stock : kkt_default_pipeline(backend))
        @test getparam(K, "nd_nseps") == 2
        @test getparam(K, "info") == 0
        @test SDS.inertia(K) == (n, 0, 0)
        rs = kkt_solves(backend, K, A)
        @test rs.r1 <= tol(T)
        @test rs.r2 <= tol(T)
    end
end
