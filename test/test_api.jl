# T13: the public API v0.1, handle layer (`DirectSolver`, `execute!` phases,
# `update!`, parameters) and LinearAlgebra layer, on SPD/HPD matrices.

api_rhs(::Type{T}, n, nrhs) where {T} = nrhs == 1 ? rand(T, n) : rand(T, n, nrhs)

# a solver on `backend` after "analysis" and "factorization" of the `view` triangle of `A`
function api_solver(backend, A::SparseMatrixCSC{T}, ::Type{INT} = Int32; view = 'L') where {T, INT}
    solver = DirectSolver(api_matrix(backend, triangle_view(A, view), INT), spd_structure(T), view)
    execute!("analysis", solver, nothing, nothing)
    execute!("factorization", solver, nothing, nothing)
    return solver
end

function api_solve(backend, solver, b; phase = "solve")
    bd = to_device(backend, b)
    xd = similar(bd)
    execute!(phase, solver, xd, bd; asynchronous = false)
    return to_host(xd)
end

@testset "constructors ($(backend_name(backend)), $T, $INT)" for backend in BACKENDS, T in ELTYPES, INT in INTTYPES
    A = random_spd(T, 60, 0.05)
    L = tril(A)
    s = spd_structure(T)
    M = api_matrix(backend, L, INT)
    C = M isa CSR ? M : CSR(M)
    for solver in (DirectSolver(M, s, 'L'), DirectSolver(C, s, 'L'), DirectSolver(C.rowptr, C.colval, C.nzval, s, 'L'))
        @test solver isa DirectSolver{T, INT}
        @test solver isa AbstractDirectSolver{T, INT}
        @test solver isa LinearAlgebra.Factorization{T}
        @test size(solver) == (60, 60) && size(solver, 1) == 60
        @test solver.fresh_factorization
        @test solver.backend == backend
        @test getparam(solver, "info") == 0
        @test occursin("DirectSolver{$T, $INT}", sprint(show, solver))
        b = rand(T, 60)
        execute!("analysis", solver, nothing, nothing)
        execute!("factorization", solver, nothing, nothing)
        @test relres(A, api_solve(backend, solver, b), b) <= tol(T)
        @test occursin("factorized", sprint(show, solver))
    end
    # zero-based arrays: the index keyword relabels the arrays of the CSR
    Cz = CSR(SparseMatrixCSC{T, INT}(L); index = 'Z')
    Cz = CSR(to_device(backend, Cz.rowptr), to_device(backend, Cz.colval), to_device(backend, Cz.nzval); index = 'Z')
    solver = DirectSolver(Cz.rowptr, Cz.colval, Cz.nzval, s, 'L'; index = 'Z')
    @test solver.A.index == SDS.INDEX_ZERO
    b = rand(T, 60)
    execute!("analysis", solver, nothing, nothing)
    execute!("factorization", solver, nothing, nothing)
    @test relres(A, api_solve(backend, solver, b), b) <= tol(T)
    # invalid arguments
    @test thrown(() -> DirectSolver(C, "SPX", 'L')) isa InvalidValueError
    @test thrown(() -> DirectSolver(C, s, 'X')) isa InvalidValueError
    @test thrown(() -> DirectSolver(C, s, 'L'; index = 'Q')) isa InvalidValueError
    @test thrown(() -> DirectSolver(CSR(C.rowptr, C.colval, C.nzval, 60, 61), s, 'L')) isa InvalidValueError
    batch = CSR(C.rowptr, C.colval, to_device(backend, repeat(Array(C.nzval), 2)), 60, 60)
    @test thrown(() -> DirectSolver(batch, s, 'L')) isa NotSupportedError
end

@testset "phases ($(backend_name(backend)), $T, $INT)" for backend in BACKENDS, T in ELTYPES, INT in INTTYPES
    A = laplacian2d(T, 15, 12) + spdiagm(0 => rand(real(T), 180))
    n = size(A, 1)
    for view in ('L', 'U', 'F')
        solver = DirectSolver(api_matrix(backend, triangle_view(A, view), INT), spd_structure(T), view)
        # deterministic_mode = 1: the bitwise comparisons below need the atomic-free forward sweep (the default atomic
        # sweep sums regime-B updates in a run-dependent order on a GPU)
        setparam!(solver, "deterministic_mode", 1)
        b = rand(T, n, 3)
        bd = to_device(backend, b)
        xd = similar(bd)
        # "reordering" + "symbolic_factorization" = "analysis"
        execute!("reordering", solver, xd, bd)
        @test solver.stage == SDS.STAGE_REORDERED
        @test isperm(getparam(solver, "perm_reorder_row"))
        execute!("symbolic_factorization", solver, xd, bd)
        execute!("factorization", solver, xd, bd)
        @test !solver.fresh_factorization
        @test getparam(solver, "info") == 0
        execute!("solve", solver, xd, bd; asynchronous = false)
        x = to_host(xd)
        @test relres(A, x, b) <= tol(T)
        # the four solve sub-phases give the same solution
        yd = similar(bd)
        for phase in ("solve_fwd_perm", "solve_fwd", "solve_bwd", "solve_bwd_perm")
            execute!(phase, solver, yd, bd)
        end
        @test to_host(yd) == x
        # in place, X === B
        cd = copy(bd)
        execute!("solve", solver, cd, cd)
        @test to_host(cd) == x
        # the named wrappers
        solver2 = DirectSolver(api_matrix(backend, triangle_view(A, view), INT), spd_structure(T), view)
        setparam!(solver2, "deterministic_mode", 1)
        @test analyze!(solver2) === solver2
        @test factorize!(solver2) === solver2
        @test refactorize!(solver2; asynchronous = false) === solver2
        @test solve!(solver2, yd, bd; asynchronous = false) === yd
        @test to_host(yd) == x
    end
end

@testset "phase order and unsupported phases ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    A = random_spd(T, 40, 0.1)
    b = to_device(backend, rand(T, 40))
    x = similar(b)
    solver = DirectSolver(api_matrix(backend, tril(A)), spd_structure(T), 'L')
    for phase in ("symbolic_factorization", "factorization", "refactorization", "solve", "solve_fwd_perm", "solve_fwd",
                  "solve_bwd", "solve_bwd_perm")
        @test thrown(() -> execute!(phase, solver, x, b)) isa FactorizationError
    end
    for name in ("lu_nnz", "perm_reorder_row", "diag", "memory_estimates")
        @test thrown(() -> getparam(solver, name)) isa FactorizationError
    end
    execute!("reordering", solver, x, b)
    @test thrown(() -> execute!("factorization", solver, x, b)) isa FactorizationError
    @test thrown(() -> getparam(solver, "lu_nnz")) isa FactorizationError
    execute!("symbolic_factorization", solver, x, b)
    @test thrown(() -> execute!("solve", solver, x, b)) isa FactorizationError
    @test thrown(() -> execute!("refactorization", solver, x, b)) isa FactorizationError
    @test thrown(() -> getparam(solver, "diag")) isa FactorizationError
    execute!("factorization", solver, x, b)
    execute!("refactorization", solver, x, b)
    execute!("solve", solver, x, b)
    @test relres(A, to_host(x), to_host(b)) <= tol(T)
    # unknown phase strings: ArgumentError; phases of later tasks: NotSupportedError
    @test thrown(() -> execute!("solving", solver, x, b)) isa ArgumentError
    @test thrown(() -> execute!("Solve", solver, x, b)) isa ArgumentError
    @test execute!("solve_refinement", solver, x, b) === nothing     # T16; ir_n_steps = 0: no-op
    for phase in ("solve_fwd_schur", "solve_bwd_schur")
        @test thrown(() -> execute!(phase, solver, x, b)) isa NotSupportedError
    end
    # "solve_diag" (T15) is the identity for Cholesky
    @test execute!("solve_diag", solver, x, b) === nothing
    # a new analysis resets the factorization
    execute!("analysis", solver, x, b)
    @test solver.fresh_factorization
    @test thrown(() -> execute!("solve", solver, x, b)) isa FactorizationError
    # bad right-hand sides
    execute!("factorization", solver, x, b)
    @test thrown(() -> execute!("solve", solver, nothing, b)) isa InvalidValueError
    @test thrown(() -> execute!("solve", solver, x, to_device(backend, rand(T, 41)))) isa DimensionMismatch
    @test thrown(() -> execute!("solve", solver, x, to_device(backend, rand(T, 40, 2)))) isa DimensionMismatch
    @test thrown(() -> execute!("solve", solver, x, to_device(backend, rand(T, 40, 2, 2)))) isa NotSupportedError
    other = T <: Complex ? ComplexF64 === T ? ComplexF32 : ComplexF64 : Float64 === T ? Float32 : Float64
    bo = to_device(backend, rand(other, 40))
    @test thrown(() -> execute!("solve", solver, bo, bo)) isa InvalidValueError
    backend isa CPU ||
        @test thrown(() -> execute!("solve", solver, rand(T, 40), rand(T, 40))) isa InvalidValueError
    # options and structures that are not implemented yet
    for (name, value) in (("matching_alg", "algo1"), ("schur_mode", 1), ("ubatch_size", 2), ("schedule", "syncfree"))
        s2 = DirectSolver(api_matrix(backend, tril(A)), spd_structure(T), 'L')
        setparam!(s2, name, value)
        @test thrown(() -> execute!("analysis", s2, x, b)) isa NotSupportedError
    end
    setparam!(solver, "solve_alg", "algo1")
    @test thrown(() -> execute!("solve", solver, x, b)) isa NotSupportedError
    setparam!(solver, "solve_alg", "default")
    s3 = DirectSolver(api_matrix(backend, A), "G", 'F')      # LU: T19
    @test thrown(() -> execute!("analysis", s3, x, b)) isa NotSupportedError
    if T <: Complex
        s4 = DirectSolver(api_matrix(backend, tril(A)), "SPD", 'L')
        @test thrown(() -> execute!("analysis", s4, x, b)) isa InvalidValueError
    end
end

@testset "data parameters ($(backend_name(backend)), $T, $INT)" for backend in BACKENDS, T in ELTYPES, INT in INTTYPES
    A = random_spd(T, 80, 0.04)
    n = size(A, 1)
    solver = api_solver(backend, A, INT)
    S = solver.host_symbolic
    # permutations: valid, 1-based, the permutation of the factor
    for name in ("perm_reorder_row", "perm_reorder_col", "perm_row", "perm_col")
        p = getparam(solver, name)
        @test p isa Vector{Int} && isperm(p)
        @test p == S.partition.perm
    end
    perm = getparam(solver, "perm_reorder_row")
    # lu_nnz = nnz(L) of the brute-force symbolic factorization of A[perm, perm]
    _, counts, _ = brute_force_symbolic(A, perm)
    @test getparam(solver, "lu_nnz") isa Int64
    @test getparam(solver, "lu_nnz") == sum(counts) == S.partition.nnz_L == nnz(solver)
    @test getparam(solver, "nsuperpanels") == SDS.nsupernodes(S)
    @test getparam(solver, "flops") == S.partition.flops > 0
    est = getparam(solver, "memory_estimates")
    @test est isa Vector{Int64} && length(est) == 16 && est == SDS.memory_estimates(S, T, INT)
    @test est[1] > 0 && est[2] >= est[1]
    # diag: the diagonal of the Cholesky factor of A[perm, perm]
    dref = diag(cholesky(Hermitian(Matrix(A[perm, perm]), :L)).L)
    d = getparam(solver, "diag")
    @test typeof(KernelAbstractions.get_backend(d)) == typeof(backend)
    @test eltype(d) == T
    @test to_host(d) ≈ dref rtol = tol(T)
    @test to_host(diag(solver)) == to_host(d)
    # getparam! into host and device buffers, with element type conversion
    buf = zeros(INT, n)
    @test getparam!(buf, solver, "perm_row") === buf && buf == perm
    dbuf = to_device(backend, zeros(INT, n))
    getparam!(dbuf, solver, "perm_reorder_col")
    @test to_host(dbuf) == perm
    hd = zeros(T, n)
    getparam!(hd, solver, "diag")
    @test hd == to_host(d)
    mest = zeros(Int64, 16)
    getparam!(mest, solver, "memory_estimates")
    @test mest == est
    @test thrown(() -> getparam!(zeros(INT, n - 1), solver, "perm_row")) isa DimensionMismatch
    @test thrown(() -> getparam!(zeros(INT, n), solver, "lu_nnz")) isa InvalidValueError
    # info: get and set
    @test getparam(solver, "info") == 0
    setparam!(solver, "info", 3)
    @test getparam(solver, "info") == 3
    @test thrown(() -> setparam!(solver, "info", 1.5)) isa InvalidValueError
    execute!("refactorization", solver, nothing, nothing)   # resets info
    @test getparam(solver, "info") == 0
    # computed data parameters cannot be set; later ones are not implemented yet
    for name in ("lu_nnz", "perm_row", "diag", "nsuperpanels", "memory_estimates")
        @test thrown(() -> setparam!(solver, name, 1)) isa ArgumentError
    end
    for name in ("perm_matching", "scale_row", "scale_col", "schur_shape", "schur_matrix", "nd_partition_tree",
                 "hybrid_device_memory_min")
        @test thrown(() -> getparam(solver, name)) isa NotSupportedError
    end
    # pivot statistics of a successful Cholesky factorization (T15)
    @test getparam(solver, "inertia") == (n, 0) && getparam(solver, "inertia") isa Tuple{INT, INT}
    @test getparam(solver, "npivots") == 0 && getparam(solver, "npivots") isa INT
    @test getparam(solver, "pivot_stats") == (npos = n, nneg = 0, nzero = 0, nperturbed = 0, n2x2 = 0)
    @test thrown(() -> getparam(solver, "no_such_parameter")) isa ArgumentError
    @test thrown(() -> setparam!(solver, "no_such_parameter", 1)) isa ArgumentError
    # configuration parameters go to the solver's options
    setparam!(solver, "reordering_alg", "algo3")
    @test getparam(solver, "reordering_alg") == "algo3" == getparam(solver.options, "reordering_alg")
    @test getparam(solver, "ir_n_steps") == 0
    # user_perm (host or device, 0- or 1-based) drives the next analysis
    up = collect(n:-1:1)
    setparam!(solver, "user_perm", to_device(backend, INT.(up .- 1)))
    @test getparam(solver, "user_perm") == up .- 1
    execute!("reordering", solver, nothing, nothing)
    @test getparam(solver, "perm_reorder_row") == up
    execute!("symbolic_factorization", solver, nothing, nothing)
    execute!("factorization", solver, nothing, nothing)
    b = rand(T, n)
    @test relres(A, api_solve(backend, solver, b), b) <= tol(T)
end

@testset "update!, refactorization and info ($(backend_name(backend)), $T, $INT)" for backend in BACKENDS, T in ELTYPES,
                                                                                      INT in INTTYPES
    B = random_spd(T, 100, 0.03) - 101 * I        # B Bᴴ - I: not positive definite (no diagonal cancels)
    n = size(B, 1)
    solver = DirectSolver(api_matrix(backend, triu(B), INT), spd_structure(T), 'U')
    b = rand(T, n)
    execute!("analysis", solver, nothing, nothing)
    execute!("factorization", solver, nothing, nothing)
    info = getparam(solver, "info")
    @test 1 <= info <= n
    @test thrown(() -> logabsdet(solver)) isa FactorizationError
    # the reference factorization fails at the same original column
    S = solver.host_symbolic
    @test SDS.ref_factorize!(SDS.allocate_numeric(S, T), S, CSR(SparseMatrixCSC{T, INT}(triu(B))).nzval) == info
    # new values: a CSR / vendor matrix, then raw arrays
    A = B + (n + 2) * I
    update!(solver, api_matrix(backend, triu(A), INT))
    execute!("refactorization", solver, nothing, nothing)
    @test getparam(solver, "info") == 0
    @test relres(A, api_solve(backend, solver, b), b) <= tol(T)
    A2 = B + 2 * (n + 2) * I
    M2 = api_matrix(backend, triu(A2), INT)
    C2 = M2 isa CSR ? M2 : CSR(M2)
    update!(solver, C2.rowptr, C2.colval, C2.nzval)
    execute!("refactorization", solver, nothing, nothing)
    @test relres(A2, api_solve(backend, solver, b), b) <= tol(T)
    # in-place value changes in the solver's own buffer
    copyto!(solver.A.nzval, to_device(backend, CSR(SparseMatrixCSC{T, INT}(triu(A))).nzval))
    execute!("refactorization", solver, nothing, nothing)
    @test relres(A, api_solve(backend, solver, b), b) <= tol(T)
    # mismatched data
    @test thrown(() -> update!(solver, api_matrix(backend, triu(random_spd(T, n + 1, 0.03)), INT))) isa InvalidValueError
    @test thrown(() -> update!(solver, api_matrix(backend, A, INT))) isa InvalidValueError   # other nnz
    Cz = CSR(SparseMatrixCSC{T, INT}(triu(A)); index = 'Z')
    @test thrown(() -> update!(solver, CSR(to_device(backend, Cz.rowptr), to_device(backend, Cz.colval),
                                           to_device(backend, Cz.nzval); index = 'Z'))) isa InvalidValueError
end

@testset "right-hand side layouts ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    A = laplacian3d(T, 6, 6, 5)
    n = size(A, 1)
    solver = api_solver(backend, A)
    # deterministic_mode = 1 for the bitwise comparisons of the layouts (atomic sweep: run-dependent sums on a GPU)
    setparam!(solver, "deterministic_mode", 1)
    for nrhs in (1, 2, 5, 3)      # the workspace grows to 5 right-hand sides and stays
        b = api_rhs(T, n, nrhs)
        x = api_solve(backend, solver, b)
        @test relres(A, x, b) <= tol(T)
        @test SDS.max_rhs(solver.workspace) == max(nrhs, nrhs == 3 ? 5 : nrhs)
        if nrhs > 1
            # MatrixDescriptor, column-major and row-major, and strided vectors
            bd = to_device(backend, b)
            X = MatrixDescriptor(similar(bd))
            execute!("solve", solver, X, MatrixDescriptor(bd))
            @test to_host(X.data) == x
            bt = to_device(backend, permutedims(b))
            Xt = MatrixDescriptor(similar(bt); transposed = true)
            execute!("solve", solver, Xt, MatrixDescriptor(bt; transposed = true))
            @test permutedims(to_host(Xt.data)) == x
            bs = to_device(backend, vec(b))
            xs = similar(bs)
            execute!("solve", solver, xs, bs)
            @test reshape(to_host(xs), n, nrhs) == x
            @test thrown(() -> execute!("solve", solver, X, MatrixDescriptor(bt; transposed = true))) isa
                  InvalidValueError
        end
    end
    @test thrown(() -> execute!("solve", solver, MatrixDescriptor(T, n), MatrixDescriptor(T, n))) isa InvalidValueError
    # deterministic_mode: the atomic-free forward sweep, bitwise reproducible
    setparam!(solver, "deterministic_mode", 1)
    b = rand(T, n, 2)
    x1 = api_solve(backend, solver, b)
    @test x1 == api_solve(backend, solver, b)
    @test relres(A, x1, b) <= tol(T)
    setparam!(solver, "deterministic_mode", 0)
    # solve_mode: A = Aᴴ, and Aᵀ = conj(A) (T16)
    setparam!(solver, "solve_mode", 2)
    @test relres(A, api_solve(backend, solver, b), b) <= tol(T)
    setparam!(solver, "solve_mode", 1)
    @test relres(sparse(transpose(A)), api_solve(backend, solver, b), b) <= tol(T)
    setparam!(solver, "solve_mode", 0)
    # iterative refinement (T16)
    setparam!(solver, "ir_n_steps", 2)
    @test relres(A, api_solve(backend, solver, b), b) <= tol(T)
    @test getparam(solver, "ir_n_steps") == 2
    setparam!(solver, "ir_n_steps", 0)
end

@testset "CSC input (transposed CSR) ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    A = random_spd(T, 70, 0.05)
    Lh = tril(A)
    # CSC arrays of the lower triangle = CSR arrays of the upper triangle of Aᵀ (MadNLP's layout)
    Ct = csr_of_transpose(Lh)
    C = CSR(to_device(backend, Ct.rowptr), to_device(backend, Ct.colval), to_device(backend, Ct.nzval), 70, 70;
            transposed = true)
    # complex Hermitian: the stored matrix is conj(A), the solve conjugates (T16)
    solver = DirectSolver(C, spd_structure(T), 'L')
    execute!("analysis", solver, nothing, nothing)
    execute!("factorization", solver, nothing, nothing)
    b = rand(T, 70)
    @test relres(A, api_solve(backend, solver, b), b) <= tol(T)
end

@testset "LinearAlgebra layer ($(backend_name(backend)), $T, $INT)" for backend in BACKENDS, T in ELTYPES,
                                                                       INT in INTTYPES
    A = random_spd(T, 120, 0.03)
    n = size(A, 1)
    b = rand(T, n, 2)
    bd = to_device(backend, b)
    for view in ('L', 'U', 'F')
        F = cholesky(api_matrix(backend, triangle_view(A, view), INT); view)
        @test F isa DirectSolver{T, INT}
        @test !F.fresh_factorization && getparam(F, "info") == 0
        # deterministic_mode = 1 for the bitwise comparisons below (atomic sweep: run-dependent sums on a GPU)
        setparam!(F, "deterministic_mode", 1)
        @test relres(A, to_host(F \ bd), b) <= tol(T)
        @test relres(A, to_host(F \ bd[:, 1]), b[:, 1]) <= tol(T)
        xd = similar(bd)
        @test ldiv!(xd, F, bd) === xd
        @test relres(A, to_host(xd), b) <= tol(T)
        cd = copy(bd)
        @test ldiv!(F, cd) === cd
        @test to_host(cd) == to_host(xd)
        X = MatrixDescriptor(similar(bd))
        ldiv!(X, F, MatrixDescriptor(bd))
        @test to_host(X.data) == to_host(xd)
        Y = MatrixDescriptor(copy(bd))
        ldiv!(F, Y)
        @test to_host(Y.data) == to_host(xd)
        # logabsdet, logdet, diag, nnz
        la, sg = logabsdet(F)
        @test la ≈ logabsdet(Matrix(A))[1] rtol = tol(T)
        @test sg == one(T)
        @test logdet(F) == la
        @test nnz(F) == getparam(F, "lu_nnz")
        @test to_host(diag(F)) == to_host(getparam(F, "diag"))
        # cholesky! reuses the analysis: refactorization from now on
        c = rand(real(T)) + 1
        F2 = cholesky!(F, api_matrix(backend, triangle_view(c * A, view), INT))
        @test F2 === F
        @test relres(c * A, to_host(F \ bd), b) <= tol(T)
    end
    # cholesky! after a bare analysis: "factorization", then "refactorization"
    solver = DirectSolver(api_matrix(backend, tril(A), INT), spd_structure(T), 'L')
    analyze!(solver)
    @test solver.fresh_factorization
    cholesky!(solver, api_matrix(backend, tril(A), INT))
    @test !solver.fresh_factorization
    @test relres(A, to_host(solver \ bd), b) <= tol(T)
    # check = true throws on a failed factorization; the default does not (CUDSS.jl)
    B = random_spd(T, n, 0.03) - n * I
    @test thrown(() -> cholesky(api_matrix(backend, tril(B), INT); view = 'L', check = true)) isa FactorizationError
    Fb = cholesky(api_matrix(backend, tril(B), INT); view = 'L')
    @test getparam(Fb, "info") > 0
    @test thrown(() -> cholesky!(Fb, api_matrix(backend, tril(B), INT); check = true)) isa FactorizationError
    # the Symmetric/Hermitian wrappers of the vendor matrix types (GPU extensions)
    if !(backend isa CPU)
        M = api_matrix(backend, tril(A), INT)
        Fh = cholesky(Hermitian(M, :L))
        @test relres(A, to_host(Fh \ bd), b) <= tol(T)
        if T <: Real
            Fs = cholesky(Symmetric(M, :L))
            @test relres(A, to_host(Fs \ bd), b) <= tol(T)
        end
    end
end

@testset "MadNLP-like refactorization loop ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    nh, nj = 300, 100
    K = kkt_matrix(T, nh, nj, 1.0e-8)
    H = K[1:nh, 1:nh]
    INT = Int32
    solver = DirectSolver(api_matrix(backend, tril(H), INT), spd_structure(T), 'L')
    b = rand(T, nh)
    bd = to_device(backend, b)
    xd = similar(bd)
    execute!("analysis", solver, xd, bd)
    # 20 iterations with a changing diagonal, the values written into the solver's buffer as MadNLP does
    σs = [real(T)((-0.5)^k * k) for k in 1:20]
    vals = [CSR(SparseMatrixCSC{T, INT}(tril(H + σ * I))).nzval for σ in σs]
    allocs = Any[]
    for (k, σ) in enumerate(σs)
        copyto!(solver.A.nzval, vals[k])
        phase = k == 1 ? "factorization" : "refactorization"
        a = device_allocated(backend, () -> (execute!(phase, solver, xd, bd); execute!("solve", solver, xd, bd)))
        KernelAbstractions.synchronize(backend)
        @test getparam(solver, "info") == 0
        @test relres(H + σ * I, to_host(xd), b) <= tol(T)
        k > 1 && push!(allocs, a)
    end
    if any(ismissing, allocs)
        @test_skip "no device allocation counter on $(backend_name(backend))"
    elseif backend isa CPU
        budget = numeric_alloc_budget(solver.symbolic, T) + solve_alloc_budget(solver.symbolic, solver.workspace)
        @test maximum(allocs) <= budget
    else
        @test maximum(allocs) == 0
    end
end
