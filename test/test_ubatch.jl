# T17: uniform batch (PLAN §1.6, §3.5): one analysis for `nbatch` matrices with
# one sparsity pattern; factors, `info`, `inertia` and pivot statistics per
# member; the right-hand side layouts of CUDSS.jl (strided vector,
# `n × (nrhs nbatch)` matrix, `n × nrhs × nbatch` array,
# `MatrixDescriptor(T, n, nrhs; nbatch)`); `ubatch_index` and `ubatch_mask`.
# Members come from `batch_members` (test/matrices.jl), device matrices from
# `api_batch_matrix` (test/backends.jl).

ubatch_spd(::Type{T}) where {T} = random_spd(T, 120, 0.03)
# "S": real symmetric or complex symmetric (not Hermitian) indefinite, diagonally dominant
ubatch_sym(::Type{T}) where {T} = random_symindef(T, 120, 0.03; hermitian = T <: Real)

# the three paths of the numeric phase: regimes A + B (default), B only, C only (strided-batched dense calls)
ubatch_regimes() = (("default", Options()), ("regime B", Options(subtree_budgets = Int[])),
                    ("regime C", Options(subtree_budgets = Int[], factorization_alg = "algo2")))

function ubatch_solver(backend, members, structure, ::Type{INT} = Int32; view = 'L', opts = Options(),
                       params = ()) where {INT}
    solver = DirectSolver(api_batch_matrix(backend, members, view, INT), structure, view)
    solver.options = opts
    for (name, value) in params
        setparam!(solver, name, value)
    end
    execute!("analysis", solver, nothing, nothing)
    execute!("factorization", solver, nothing, nothing)
    return solver
end

# solve on the device with the host right-hand side `B` (any batch layout) into a copy of `X0`
function ubatch_solve(backend, solver, B; X0 = zero(B))
    Bd = to_device(backend, B)
    Xd = to_device(backend, X0)
    execute!("solve", solver, Xd, Bd; asynchronous = false)
    return to_host(Xd)
end

@testset "batches of SPD and symmetric systems ($(backend_name(backend)), $T)" for backend in BACKENDS,
                                                                                  T in ELTYPES
    for (structure, A) in ((spd_structure(T), ubatch_spd(T)), ("S", ubatch_sym(T))), nb in (1, 2, 3, 16, 64)
        @testset "$structure nb = $nb" begin
            members = batch_members(A, nb)
            solver = ubatch_solver(backend, members, structure)
            @test solver.nbatch == nb == nbatch(solver.A)
            @test getparam(solver, "info") == (nb == 1 ? 0 : zeros(Int, nb))
            # the cuDSS multi-RHS batch bug (CUDSS.jl "Known issues"): every column of every member is right
            for nrhs in (1, 2, 4)
                B = rand(T, size(A, 1), nrhs, nb)
                @test maximum(batch_relres(members, ubatch_solve(backend, solver, B), B)) <= tol(T)
            end
        end
    end
end

@testset "regimes A, B, C and refinement ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    nb, nrhs = 5, 3
    for (structure, A) in ((spd_structure(T), ubatch_spd(T)), ("S", ubatch_sym(T))), (rname, opts) in ubatch_regimes()
        @testset "$structure $rname" begin
            members = batch_members(A, nb)
            solver = ubatch_solver(backend, members, structure; opts)
            B = rand(T, size(A, 1), nrhs, nb)
            @test maximum(batch_relres(members, ubatch_solve(backend, solver, B), B)) <= tol(T)
            # refinement uses each member's values; the steps are counted once for the batch
            setparam!(solver, "ir_n_steps", 2)
            @test maximum(batch_relres(members, ubatch_solve(backend, solver, B), B)) <= tol(T)
            @test getparam(solver, "ir_n_steps") == 2
            # the diagonal of every member, member after member
            d = to_host(getparam(solver, "diag"))
            @test length(d) == nb * size(A, 1)
            setparam!(solver, "ubatch_index", 3)
            execute!("refactorization", solver, nothing, nothing)   # member 4 again: same factor
            @test to_host(getparam(solver, "diag")) == d
        end
    end
end

@testset "per-member inertia and pivot statistics ($(backend_name(backend)), $T)" for backend in BACKENDS,
                                                                                      T in ELTYPES
    nb = 3
    A = random_symindef(T, 60, 0.05)                         # "S" (real) / "H" (complex)
    members = batch_members(A, nb)
    members[2] = -members[2]                                  # the opposite inertia
    solver = ubatch_solver(backend, members, sym_structure(T))
    inertia = getparam(solver, "inertia")
    stats = getparam(solver, "pivot_stats")
    npiv = getparam(solver, "npivots")
    @test length(inertia) == length(stats) == length(npiv) == nb
    for k in 1:nb
        @test npiv[k] == 0
        @test inertia[k] == eigen_npos_nneg(members[k])
        @test (stats[k].npos, stats[k].nneg) == inertia[k]
        single = DirectSolver(api_matrix(backend, tril(members[k]), Int32), sym_structure(T), 'L')
        execute!("analysis", single, nothing, nothing)
        execute!("factorization", single, nothing, nothing)
        @test getparam(single, "pivot_stats") == stats[k]
    end
    @test inertia[1] == reverse(inertia[2])
    # Cholesky: npos = n for every member
    spd = batch_members(ubatch_spd(T), nb)
    s2 = ubatch_solver(backend, spd, spd_structure(T))
    @test getparam(s2, "inertia") == fill((Int32(120), Int32(0)), nb)
    @test all(st -> st.nperturbed == 0 && st.npos == 120, getparam(s2, "pivot_stats"))
end

@testset "ubatch_index and ubatch_mask ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    nb, nrhs = 4, 2
    for (structure, A) in ((spd_structure(T), ubatch_spd(T)), ("S", ubatch_sym(T))), (rname, opts) in ubatch_regimes()
        @testset "$structure $rname" begin
            n = size(A, 1)
            members = batch_members(A, nb)
            solver = ubatch_solver(backend, members, structure; opts)
            S = solver.symbolic
            snapshot() = [SDS.member_numeric(solver.numeric, S, k) for k in 1:nb]
            same(a, b) = a.factor == b.factor && a.d == b.d && a.piv == b.piv && a.pivot_kind == b.pivot_kind &&
                         a.stats == b.stats && a.info == b.info
            before = snapshot()
            fresh = batch_members(A, nb + 1)[2:end]           # new values for every member (member 1 of a draw is A)
            update!(solver, api_batch_matrix(backend, fresh, 'L'))
            # ubatch_index = 2 (0-based): only member 3 is refactorized and solved
            setparam!(solver, "ubatch_index", 2)
            execute!("refactorization", solver, nothing, nothing)
            after = snapshot()
            @test all(k -> same(after[k], before[k]), (1, 2, 4))
            @test after[3].factor != before[3].factor
            B = rand(T, n, nrhs, nb)
            X0 = fill(T(7), n, nrhs, nb)
            X = ubatch_solve(backend, solver, B; X0)
            @test relres(fresh[3], X[:, :, 3], B[:, :, 3]) <= tol(T)
            @test X[:, :, [1, 2, 4]] == X0[:, :, [1, 2, 4]]
            # ubatch_mask: members 1 and 4
            setparam!(solver, "ubatch_index", -1)
            setparam!(solver, "ubatch_mask", [1, 0, 0, 1])
            @test getparam(solver, "ubatch_mask") == [1, 0, 0, 1]
            execute!("refactorization", solver, nothing, nothing)
            masked = snapshot()
            @test same(masked[2], after[2]) && same(masked[3], after[3])
            @test masked[1].factor != after[1].factor && masked[4].factor != after[4].factor
            X = ubatch_solve(backend, solver, B; X0)
            @test relres(fresh[1], X[:, :, 1], B[:, :, 1]) <= tol(T)
            @test relres(fresh[4], X[:, :, 4], B[:, :, 4]) <= tol(T)
            @test X[:, :, [2, 3]] == X0[:, :, [2, 3]]
            # both: their intersection (member 4)
            setparam!(solver, "ubatch_index", 3)
            X = ubatch_solve(backend, solver, B; X0)
            @test relres(fresh[4], X[:, :, 4], B[:, :, 4]) <= tol(T)
            @test X[:, :, 1:3] == X0[:, :, 1:3]
            # all members again: member 2 still has its old factor, the others the new one
            setparam!(solver, "ubatch_index", -1)
            setparam!(solver, "ubatch_mask", nothing)
            X = ubatch_solve(backend, solver, B; X0)
            @test relres(members[2], X[:, :, 2], B[:, :, 2]) <= tol(T)
            @test all(k -> relres(fresh[k], X[:, :, k], B[:, :, k]) <= tol(T), (1, 3, 4))
            # an index or a mask outside the batch
            setparam!(solver, "ubatch_index", nb)
            @test thrown(() -> execute!("refactorization", solver, nothing, nothing)) isa InvalidValueError
            setparam!(solver, "ubatch_index", -1)
            setparam!(solver, "ubatch_mask", [1, 0])
            @test thrown(() -> execute!("solve", solver, to_device(backend, X), to_device(backend, B))) isa
                  InvalidValueError
            setparam!(solver, "ubatch_mask", zeros(Int, nb))
            @test thrown(() -> execute!("refactorization", solver, nothing, nothing)) isa InvalidValueError
        end
    end
end

@testset "right-hand side layouts ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    nb = 3
    for (structure, A) in ((spd_structure(T), ubatch_spd(T)), ("S", ubatch_sym(T))), nrhs in (1, 2)
        n = size(A, 1)
        members = batch_members(A, nb)
        # the atomic forward sweep sums in a run-dependent order on GPUs (T16): compare bitwise deterministically
        solver = ubatch_solver(backend, members, structure; params = (("deterministic_mode", 1),))
        B3 = rand(T, n, nrhs, nb)
        X3 = ubatch_solve(backend, solver, B3)                          # n × nrhs × nbatch
        @test maximum(batch_relres(members, X3, B3)) <= tol(T)
        @test ubatch_solve(backend, solver, vec(B3)) == vec(X3)          # strided (n nrhs nbatch,)
        @test ubatch_solve(backend, solver, reshape(B3, n, nrhs * nb)) == reshape(X3, n, nrhs * nb)
        # MatrixDescriptor(T, n, nrhs; nbatch) (≅ CudssMatrix(T, n, nrhs; nbatch)) on 3-D, matrix and strided data
        for shape in ((n, nrhs, nb), (n, nrhs * nb), (n * nrhs * nb,))
            Bd = to_device(backend, reshape(B3, shape))
            Xd = to_device(backend, zeros(T, shape))
            bdesc = nrhs == 1 ? MatrixDescriptor(T, n; nbatch = nb) : MatrixDescriptor(T, n, nrhs; nbatch = nb)
            xdesc = nrhs == 1 ? MatrixDescriptor(T, n; nbatch = nb) : MatrixDescriptor(T, n, nrhs; nbatch = nb)
            update!(bdesc, Bd)
            update!(xdesc, Xd)
            execute!("solve", solver, xdesc, bdesc; asynchronous = false)
            @test vec(to_host(Xd)) == vec(X3)
        end
        wrong = to_device(backend, zeros(T, n * nb, nrhs))
        @test thrown(() -> update!(MatrixDescriptor(T, n, nrhs; nbatch = nb), wrong)) isa InvalidValueError
        # in place (X === B), the LinearAlgebra layer
        Bd = to_device(backend, B3)
        execute!("solve", solver, Bd, Bd)
        @test to_host(Bd) == X3
        # the number of columns must split over the members; row-major batches are not supported
        @test thrown(() -> ubatch_solve(backend, solver, rand(T, n, nb + 1))) isa DimensionMismatch
        Bt = to_device(backend, rand(T, nrhs * nb, n))
        bt = MatrixDescriptor(Bt; transposed = true)
        @test thrown(() -> execute!("solve", solver, bt, bt)) isa NotSupportedError
    end
end

@testset "a failed member ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    nb, j = 4, 37
    A = ubatch_spd(T)
    n = size(A, 1)
    for (r, (rname, opts)) in enumerate(ubatch_regimes())
        @testset "$rname" begin
            members = batch_members(A, nb)
            M = copy(members[3])
            M[j, j] = -M[j, j]                                # not positive definite: fails at column j
            members[3] = M
            solver = ubatch_solver(backend, members, spd_structure(T); opts)
            @test getparam(solver, "info") == [0, 0, j, 0]
            single = DirectSolver(api_matrix(backend, tril(M), Int32), spd_structure(T), 'L')
            single.options = ubatch_regimes()[r][2]
            execute!("analysis", single, nothing, nothing)
            execute!("factorization", single, nothing, nothing)
            @test getparam(single, "info") == j
            # the other members solve, and only member 3 is refactorized after a fix
            B = rand(T, n, 2, nb)
            X = ubatch_solve(backend, solver, B)
            @test all(k -> relres(members[k], X[:, :, k], B[:, :, k]) <= tol(T), (1, 2, 4))
            members[3] = batch_members(A, 2)[2]
            update!(solver, api_batch_matrix(backend, members, 'L'))
            setparam!(solver, "ubatch_mask", [0, 0, 1, 0])
            execute!("refactorization", solver, nothing, nothing)
            @test getparam(solver, "info") == zeros(Int, nb)
            setparam!(solver, "ubatch_mask", nothing)
            X = ubatch_solve(backend, solver, B)
            @test maximum(batch_relres(members, X, B)) <= tol(T)
        end
    end
    # the LinearAlgebra layer reports every member's info
    members = batch_members(A, 2)
    M = copy(members[2])
    M[j, j] = -M[j, j]
    members[2] = M
    e = thrown(() -> cholesky(api_batch_matrix(backend, members, 'L'); view = 'L', check = true))
    @test e isa FactorizationError && e.info == [0, j]
end

@testset "LinearAlgebra layer and errors ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    nb = 3
    A = ubatch_spd(T)
    n = size(A, 1)
    members = batch_members(A, nb)
    Ad = api_batch_matrix(backend, members, 'L')
    # auto-detection from length(nzval) ÷ length(colval)
    F = cholesky(Ad; view = 'L')
    @test F.nbatch == nb && getparam(F, "info") == zeros(Int, nb)
    B = rand(T, n, nb)
    Xd = to_device(backend, zeros(T, n, nb))
    ldiv!(Xd, F, to_device(backend, B))
    @test maximum(batch_relres(members, to_host(Xd), B)) <= tol(T)
    @test maximum(batch_relres(members, to_host(F \ to_device(backend, B)), B)) <= tol(T)
    B3 = rand(T, n, 2, nb)
    @test maximum(batch_relres(members, to_host(F \ to_device(backend, B3)), B3)) <= tol(T)
    @test thrown(() -> logabsdet(F)) isa NotSupportedError
    fresh = batch_members(A, nb)
    cholesky!(F, api_batch_matrix(backend, fresh, 'L'))
    @test maximum(batch_relres(fresh, to_host(F \ to_device(backend, B)), B)) <= tol(T)
    G = ldlt(api_batch_matrix(backend, batch_members(ubatch_sym(T), nb), 'L'); view = 'L')
    @test G.nbatch == nb
    # values given as an nnz × nbatch matrix
    rowptr, colval, nzval = batch_csr(members, 'L', Int32)
    Cm = CSR(to_device(backend, rowptr), to_device(backend, colval), to_device(backend, reshape(nzval, :, nb)))
    @test nbatch(Cm) == nb
    sm = DirectSolver(Cm, spd_structure(T), 'L')
    execute!("analysis", sm, nothing, nothing)
    execute!("factorization", sm, nothing, nothing)
    @test maximum(batch_relres(members, ubatch_solve(backend, sm, B), B)) <= tol(T)
    # ubatch_size: 0 (deduced) or the batch size
    s = DirectSolver(Ad, spd_structure(T), 'L')
    setparam!(s, "ubatch_size", nb + 1)
    @test thrown(() -> execute!("analysis", s, nothing, nothing)) isa InvalidValueError
    setparam!(s, "ubatch_size", nb)
    execute!("analysis", s, nothing, nothing)
    execute!("factorization", s, nothing, nothing)
    @test getparam(s, "info") == zeros(Int, nb)
    setparam!(s, "info", [1, 2, 3])
    @test getparam(s, "info") == [1, 2, 3]
    setparam!(s, "info", 0)
    @test getparam(s, "info") == zeros(Int, nb)
    @test thrown(() -> setparam!(s, "info", [1, 2])) isa InvalidValueError
    # update! keeps the batch size
    @test thrown(() -> update!(s, api_batch_matrix(backend, members[1:2], 'L'))) isa InvalidValueError
    @test occursin("nbatch = $nb", sprint(show, s))
end
