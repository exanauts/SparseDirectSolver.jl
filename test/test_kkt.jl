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
            # spare capacity in rowval/nzval beyond nnz (a valid SparseMatrixCSC) cannot be read in place
            M = tril(A)
            push!(M.rowval, 1)
            push!(M.nzval, one(T))
            @test nnz(M) < length(nonzeros(M))
            @test_throws SDS.InvalidValueError KKTSolver(M, 'L')
            @test_throws SDS.InvalidValueError KKTSolver(M, 'L'; factorization = :ldl, deferred_analysis = true)
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
        # matching (#193): `:ldl` only, a matching_alg string, on the stock pipeline
        @test_throws SDS.InvalidValueError KKTSolver(args...; matching = "algo5")
        @test_throws SDS.InvalidValueError KKTSolver(args...; matching = "algo6", factorization = :cholesky)
        @test_throws SDS.InvalidValueError KKTSolver(args...; factorization = :ldl, matching = "algo9")
        @test_throws SDS.InvalidValueError KKTSolver(args...; factorization = :ldl, matching = "mc64")
        if SDS.forward_progress(backend)
            # the fused LDLᵀ has the matching and the pivot replacement of the stock one (#194): fused = true takes them
            n = size(A, 1)
            for kw in ((; matching = "algo6"), (; pivot_epsilon = 1.0e-10), (; pivot_sign = ones(Int, n)))
                Kf = KKTSolver(args...; factorization = :ldl, fused = true, kw...)
                @test SDS.pipeline(Kf) === :fused
                @test getparam(Kf, "info") == 0
            end
            # the fused Cholesky reads pivot_epsilon
            Kc = KKTSolver(args...; pivot_epsilon = 1.0e-10, fused = true)
            @test SDS.pipeline(Kc) === :fused
            @test getparam(Kc, "pivot_epsilon") == 1.0e-10
        end
        # the options are checked at construction also when the analysis is deferred
        @test_throws SDS.InvalidValueError KKTSolver(args...; matching = "algo5", deferred_analysis = true)
        @test_throws SDS.InvalidValueError KKTSolver(args...; pivoting = :bk, deferred_analysis = true)
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
        # matching and pivot replacement of an LDLᵀ keep the default pipeline: both pipelines have them (#194)
        for kw in ((; matching = "algo5"), (; matching = "algo6"), (; pivot_epsilon = 1.0e-10),
                   (; pivot_sign = ones(Int, n)), (; matching = "algo6", pivot_epsilon = 1.0e-10))
            Ks = KKTSolver(args...; factorization = :ldl, kw...)
            @test SDS.pipeline(Ks) === expected
            @test Ks.inner isa (expected === :fused ? SDS.FusedSolver : SDS.DirectSolver)
            @test getparam(Ks, "info") == 0
            @test SDS.inertia(Ks) == (n, 0, 0)
        end
        # matching off ("none", "default") keeps the default pipeline, and :cholesky with pivot_epsilon too
        @test SDS.pipeline(KKTSolver(args...; factorization = :ldl, matching = "none")) === expected
        @test SDS.pipeline(KKTSolver(args...; factorization = :ldl, matching = "default")) === expected
        @test SDS.pipeline(KKTSolver(args...; pivot_epsilon = 1.0e-10)) === expected
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

    # matching (#193, #194): the LDLᵀ with the scaled symmetric matching of "algo5"/"algo6", whose cycles give the 2×2
    # pivot pairs of the ordering with either pivoting (`:none` on the default pipeline, `:bk` on the stock one); then
    # the next iteration's values, refactorized
    @testset ":ldl, matching = $matching, :$pivoting, grid_kkt_matrix 30×30: $T" for
            T in eltypes_among((Float32, Float64)), matching in ("algo5", "algo6"), pivoting in (:none, :bk)
        m = 30 * 30
        A = grid_kkt_matrix(T, 30, 30, 1.0)
        args, nzval = kkt_input(backend, A)
        K = KKTSolver(args...; factorization = :ldl, pivoting, matching)
        @test SDS.pipeline(K) === (pivoting === :none ? kkt_default_pipeline(backend) : :stock)
        @test getparam(K, "matching_alg") == matching
        @test getparam(K, "info") == 0
        @test SDS.inertia(K) == (m, 0, m)
        rs = kkt_solves(backend, K, A)
        @test rs.r1 <= tol(T)
        @test rs.r2 <= tol(T)
        A2 = grid_kkt_matrix(T, 30, 30, 0.5)
        kkt_set_values!(nzval, A2)
        factorize!(K)
        @test getparam(K, "info") == 0
        @test SDS.inertia(K) == (m, 0, m)
        rs = kkt_solves(backend, K, A2)
        @test rs.r1 <= tol(T)
        @test rs.r2 <= tol(T)
    end

    # matching jobs 1–4 have no scaling and give no 2×2 pairs: with pivoting = :none the analysis skips the
    # value-based pair detection as without matching; "algo5" keeps the pairs of its cycles
    @testset ":ldl, :none, matching = $matching, pivot_pairs: $T" for T in eltypes_among((Float32, Float64)),
                                                                     matching in ("algo1", "algo5")
        m = 30 * 30
        A = grid_kkt_matrix(T, 30, 30, 1.0)
        args, _ = kkt_input(backend, A)
        K = KKTSolver(args...; factorization = :ldl, matching)
        @test SDS.pipeline(K) === kkt_default_pipeline(backend)
        @test getparam(K, "pivot_pairs") == (matching == "algo1" ? "none" : "default")
        @test getparam(K, "info") == 0
        @test SDS.inertia(K) == (m, 0, m)
        rs = kkt_solves(backend, K, A)
        @test rs.r1 <= tol(T)
        @test rs.r2 <= tol(T)
        # :bk keeps the default pair detection with every job
        @test getparam(KKTSolver(args...; factorization = :ldl, pivoting = :bk, matching), "pivot_pairs") == "default"
    end

    # deferred analysis (#193): built on an all-zero nzval (no KKT values yet, as MadNLP builds its linear solver), the
    # first factorize! analyzes the values written since (matching and pivot pairs included), the next refactorizes
    @testset "deferred_analysis, $label: $T" for T in eltypes_among((Float32, Float64)),
                                                 (label, kw) in (("default pipeline", (;)),
                                                                 ("matching = algo6", (; matching = "algo6")),
                                                                 ("matching = algo5, :bk", (; matching = "algo5",
                                                                                            pivoting = :bk)))
        m = 30 * 30
        A = grid_kkt_matrix(T, 30, 30, 1.0)
        args, nzval = kkt_input(backend, A)
        fill!(nzval, zero(T))
        K = KKTSolver(args...; factorization = :ldl, deferred_analysis = true, kw...)
        @test SDS.pipeline(K) === (get(kw, :pivoting, :none) === :bk ? :stock : kkt_default_pipeline(backend))
        # before the first factorization: "info" is 0, solve! and inertia raise, show prints
        @test getparam(K, "info") == 0
        @test_throws SDS.FactorizationError SDS.inertia(K)
        @test_throws SDS.FactorizationError solve!(K, to_device(backend, ones(T, 2m)))
        @test occursin("not factorized", sprint(show, K))
        kkt_set_values!(nzval, A)
        @test factorize!(K) === K
        @test getparam(K, "info") == 0
        @test SDS.inertia(K) == (m, 0, m)
        @test occursin("info = 0", sprint(show, K))
        rs = kkt_solves(backend, K, A)
        @test rs.r1 <= tol(T)
        @test rs.r2 <= tol(T)
        A2 = grid_kkt_matrix(T, 30, 30, 0.5)
        kkt_set_values!(nzval, A2)
        factorize!(K)
        @test getparam(K, "info") == 0
        @test SDS.inertia(K) == (m, 0, m)
        rs = kkt_solves(backend, K, A2)
        @test rs.r1 <= tol(T)
        @test rs.r2 <= tol(T)
    end

    # analyze!(K): the analysis redone on the current values (as MadNLP's binding does when its refinement rejects a
    # solve). Built on grid_kkt_matrix with δ = 100, whose dual diagonal dominates, the matching of "algo6" is (close
    # to) the identity and gives no pairs; with δ = 0 that analysis meets zero dual pivots, replaced (npivots > 0).
    # After analyze! the matching pairs each dual with a primal, as a solver built on the δ = 0 values does: the same
    # inertia, no replaced pivot, the same accuracy; the next iteration refactorizes with the new analysis
    @testset "analyze!, matching = algo6, :$pivoting, fused = $fused: $T" for T in eltypes_among((Float32, Float64)),
                                                                             (pivoting, fused) in (
                                                                                 (:none, nothing), (:none, false),
                                                                                 (:bk, nothing))
        m = 30 * 30
        A1 = grid_kkt_matrix(T, 30, 30, 100.0)
        A2 = grid_kkt_matrix(T, 30, 30, 0.0)
        kw = (; factorization = :ldl, pivoting, fused, matching = "algo6", pivot_sign = vcat(fill(1, m), fill(-1, m)),
              pivot_epsilon = T == Float64 ? 1.0e-10 : 1.0e-5)
        args, nzval = kkt_input(backend, A1)
        K = KKTSolver(args...; kw...)
        @test SDS.pipeline(K) === (pivoting === :none && fused === nothing ? kkt_default_pipeline(backend) : :stock)
        kkt_set_values!(nzval, A2)
        factorize!(K)
        @test getparam(K, "info") == 0
        @test getparam(K, "npivots") > 0                                # the stale matching
        @test analyze!(K) === K
        @test getparam(K, "info") == 0
        @test SDS.pipeline(K) === (pivoting === :none && fused === nothing ? kkt_default_pipeline(backend) : :stock)
        # the parameters of the constructor are kept
        @test getparam(K, "matching_alg") == "algo6"
        @test getparam(K, "pivot_epsilon") == kw.pivot_epsilon
        # a solver built on the same values
        args2, _ = kkt_input(backend, A2)
        F = KKTSolver(args2...; kw...)
        @test getparam(F, "info") == 0
        @test SDS.inertia(K) == SDS.inertia(F) == (m, 0, m)
        @test getparam(K, "npivots") == getparam(F, "npivots") == 0
        rs, rf = kkt_solves(backend, K, A2), kkt_solves(backend, F, A2)
        @test rs.r1 <= weak_pivot_tol(T)
        @test rs.r2 <= weak_pivot_tol(T)
        @test rs.r1 <= 2 * max(rf.r1, rf.r2, tol(T))
        # the next iteration refactorizes with the new analysis
        A3 = grid_kkt_matrix(T, 30, 30, 1.0e-8)
        kkt_set_values!(nzval, A3)
        @test factorize!(K) === K
        @test getparam(K, "info") == 0
        @test SDS.inertia(K) == (m, 0, m)
        @test getparam(K, "npivots") == 0
        rs = kkt_solves(backend, K, A3)
        @test rs.r1 <= weak_pivot_tol(T)
        @test rs.r2 <= weak_pivot_tol(T)
    end

    # analyze!(K) before the first factorization (deferred_analysis = true) is the first factorization, and repeated
    # re-analyses keep ir_n_steps (the refinement workspace of each new analysis) and the Cholesky of a default solver
    @testset "analyze!, deferred_analysis, repeated, $label: $T" for T in eltypes_among((Float32, Float64)),
                                                                     (label, kw) in (
                                                                         ("cholesky", (;)),
                                                                         ("ldl, matching = algo5, ir_n_steps = 2",
                                                                          (; factorization = :ldl, matching = "algo5",
                                                                           ir_n_steps = 2)),
                                                                         ("ldl, matching = algo5, ir_n_steps = 2, stock",
                                                                          (; factorization = :ldl, matching = "algo5",
                                                                           ir_n_steps = 2, fused = false)))
        ldl = get(kw, :factorization, :cholesky) === :ldl
        A = ldl ? kkt_slack_matrix(T, 200, 100, 0.0) : laplacian2d(T, 20, 20)
        # the inertia of the factorization of a solver built on the values (replaced pivots count with their sign)
        args0, _ = kkt_input(backend, A)
        inertia = SDS.inertia(KKTSolver(args0...; kw...))
        args, nzval = kkt_input(backend, A)
        fill!(nzval, zero(T))
        K = KKTSolver(args...; deferred_analysis = true, kw...)
        @test SDS.pipeline(K) === (get(kw, :fused, nothing) === false ? :stock : kkt_default_pipeline(backend))
        @test_throws SDS.FactorizationError SDS.inertia(K)
        kkt_set_values!(nzval, A)
        @test analyze!(K) === K                                         # the first factorization
        @test getparam(K, "info") == 0
        @test occursin("info = 0", sprint(show, K))
        for _ in 1:3
            analyze!(K)
            @test getparam(K, "info") == 0
            @test SDS.inertia(K) == inertia
            rs = kkt_solves(backend, K, A)
            @test getparam(K, "ir_n_steps") == get(kw, :ir_n_steps, 0)
            @test rs.r1 <= tol(T)
            @test rs.r2 <= tol(T)
            @test rs.b_kept
        end
        factorize!(K)
        @test getparam(K, "info") == 0
        rs = kkt_solves(backend, K, A)
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
    # pivot_sign and pivot_epsilon on the no-pivot LDLᵀ: +1 for the primal, -1 for the dual rows, as an interior-point
    # method passes them; no pivot of grid_kkt_matrix is small, so the inertia is unchanged. Both pipelines replace
    # pivots (#194): `fused = nothing` keeps the default pipeline
    @testset "pivot_sign, pivot_epsilon (:ldl), fused = $fused: $T" for T in eltypes_among((Float32, Float64)),
                                                                       fused in (SDS.forward_progress(backend) ?
                                                                                 (false, nothing, true) : (false, nothing))
        m = 30 * 30
        A = grid_kkt_matrix(T, 30, 30, 1.0)
        args, _ = kkt_input(backend, A)
        K = KKTSolver(args...; factorization = :ldl, fused, pivot_sign = vcat(fill(1, m), fill(-1, m)),
                      pivot_epsilon = 1.0e-10)
        @test SDS.pipeline(K) === (fused === nothing ? kkt_default_pipeline(backend) : fused ? :fused : :stock)
        @test getparam(K, "pivot_epsilon") == 1.0e-10
        @test getparam(K, "info") == 0
        @test SDS.inertia(K) == (m, 0, m)
        @test getparam(K, "npivots") == 0
        rs = kkt_solves(backend, K, A)
        @test rs.r1 <= tol(T)
        @test rs.r2 <= tol(T)
        # a pivot_sign of the wrong length
        @test_throws SDS.InvalidValueError KKTSolver(args...; factorization = :ldl, fused, pivot_sign = fill(1, m))
    end

    # #194: the static LDLᵀ of an interior-point method with matching, pivot_epsilon and pivot_sign on every pipeline
    # the backend runs. grid_kkt_matrix with δ = 0 (no regularization: zero dual diagonal) has the inertia
    # (m, 0, m); the matching pairs each dual with a primal, so no pivot is replaced and the inertia is exact. Then
    # the next iteration's values; the deferred analysis as an interior-point method builds the solver
    @testset ":ldl, matching = algo5, pivot_epsilon, pivot_sign, fused = $fused, deferred = $deferred: $T" for
            T in eltypes_among((Float32, Float64)), deferred in (false, true),
            fused in (SDS.forward_progress(backend) ? (nothing, false, true) : (nothing, false))
        m = 30 * 30
        A = grid_kkt_matrix(T, 30, 30, 0.0)
        args, nzval = kkt_input(backend, A)
        deferred && fill!(nzval, zero(T))
        K = KKTSolver(args...; factorization = :ldl, matching = "algo5", fused, deferred_analysis = deferred,
                      pivot_epsilon = T == Float64 ? 1.0e-10 : 1.0e-5, pivot_sign = vcat(fill(1, m), fill(-1, m)))
        @test SDS.pipeline(K) === (fused === nothing ? kkt_default_pipeline(backend) : fused ? :fused : :stock)
        deferred && (kkt_set_values!(nzval, A); factorize!(K))
        @test getparam(K, "info") == 0
        @test SDS.inertia(K) == (m, 0, m)
        rs = kkt_solves(backend, K, A)
        @test rs.r1 <= weak_pivot_tol(T)
        @test rs.r2 <= weak_pivot_tol(T)
        A2 = grid_kkt_matrix(T, 30, 30, 1.0e-8)
        kkt_set_values!(nzval, A2)
        factorize!(K)
        @test getparam(K, "info") == 0
        @test SDS.inertia(K) == (m, 0, m)
        rs = kkt_solves(backend, K, A2)
        @test rs.r1 <= weak_pivot_tol(T)
        @test rs.r2 <= weak_pivot_tol(T)
    end

    # #194: ir_n_steps on every pipeline the backend runs. kkt_slack_matrix with δ = 0 under algo5 meets zero pivots,
    # replaced by ±pivot_epsilon (the default), which costs the unrefined solve its accuracy; refinement against the
    # caller's values restores it, on the fused pipeline as on the stock one, and both agree
    @testset ":ldl, matching = algo5, pivot replacement, ir_n_steps: $T" for T in eltypes_among((Float32, Float64))
        Random.seed!(666)
        A = kkt_slack_matrix(T, 200, 100, 0.0)
        n = size(A, 1)
        args, _ = kkt_input(backend, A)
        b = rand(T, n)
        bd = to_device(backend, b)
        xs = Dict{Any, Vector{T}}()
        for fused in (SDS.forward_progress(backend) ? (false, true) : (false,)), k in (nothing, 0, 2)
            K = KKTSolver(args...; factorization = :ldl, matching = "algo5", fused, ir_n_steps = k)
            @test SDS.pipeline(K) === (fused ? :fused : :stock)
            @test getparam(K, "info") == 0
            @test getparam(K, "npivots") > 0
            x = to_device(backend, zeros(T, n))
            solve!(K, x, bd)
            @test getparam(K, "ir_n_steps") == something(k, 0)
            xs[(fused, k)] = to_host(x)
            r = relres(A, xs[(fused, k)], b)
            something(k, 0) == 0 ? (@test r > tol(T)) : (@test r <= tol(T))
            rs = kkt_solves(backend, K, A)                              # in place and with two vectors
            something(k, 0) == 0 || @test rs.r1 <= tol(T) && rs.r2 <= tol(T) && rs.b_kept
        end
        xr = xs[(false, 2)]
        if SDS.forward_progress(backend)
            @test norm(xs[(true, 2)] - xr) / norm(xr) <= tol(T)
        end
        @test_throws SDS.InvalidValueError KKTSolver(args...; factorization = :ldl, ir_n_steps = -1)
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
