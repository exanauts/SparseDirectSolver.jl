# T15: device LDLᵀ/LDLᴴ in regimes A/B/C (fused kernels, in-front Bunch–Kaufman, perturbation,
# pivot_sign), the diagonal solve and the pivot statistics, against the T14 reference.
# Structure "S" for real T, "H" for complex T (sym_structure), plus complex symmetric "S".

# regime mixes of the analysis (merged into each case's options)
const LDLT_REGIMES = (("default", (;)),
                      ("A+B+C, small budgets", (subtree_budgets = [8192, 16384], regime_c_width = 16,
                                                regime_c_rows = 128)),
                      ("C only", (factorization_alg = "algo2", subtree_budgets = Int[])))

# the T14 matrices: (name, matrix, option keywords, structure, compare the inertia with the eigenvalues)
function ldlt_cases(::Type{T}) where {T}
    nh, nj = 300, 100
    cases = Any[("random_symindef(400,0.01)", random_symindef(T, 400, 0.01), (;), sym_structure(T), false),
                ("kkt(300,100,1e-2)", kkt_matrix(T, nh, nj, 1.0e-2), (;), sym_structure(T), false),
                ("kkt(300,100,1e-8)", kkt_matrix(T, nh, nj, 1.0e-8), (;), sym_structure(T), false),
                ("kkt(300,100,1e-8) interleaved", kkt_matrix(T, nh, nj, 1.0e-8),
                 (user_perm = kkt_interleaved_perm(nh, nj),), sym_structure(T), false),
                ("random_symindef(300,0.02)", random_symindef(T, 300, 0.02), (;), sym_structure(T), true),
                ("kkt indefinite H, δ = 0", kkt_matrix(T, 200, 100, 0.0; hessian = :indefinite), (;),
                 sym_structure(T), true),
                ("kkt 1e-3 H, δ = 0, interleaved (2×2)",
                 kkt_matrix(T, 200, 100, 0.0; hessian = :indefinite, hessian_scale = 1.0e-3),
                 (user_perm = kkt_interleaved_perm(200, 100),), sym_structure(T), true)]
    if T <: Complex
        push!(cases, ("complex symmetric", random_symindef(T, 300, 0.02; hermitian = false), (;), "S", false))
    end
    return cases
end

@testset "device factor equals the reference ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    Random.seed!(666)
    for (name, A, kw, structure, eig) in ldlt_cases(T), (rname, rkw) in LDLT_REGIMES
        @testset "$name, $rname" begin
            opts = Options(; kw..., rkw...)
            S, Nr, Sd, Nd, nz = ldlt_setup(backend, A; opts, structure)
            @test SDS.factorize!(Nd, Sd, nz; opts) == 0
            Nh = SDS.host_numeric(Nd)
            # the same pivot sequence: local orders and pivot kinds are equal, D and L to rounding
            @test Nh.piv == Nr.piv
            @test Nh.pivot_kind == Nr.pivot_kind
            @test d_error(Nh, Nr) <= panel_tol(T)
            @test panel_error(Nh, Nr) <= panel_tol(T)
            @test Nh.stats == Nr.stats
            st = SDS.pivot_totals(Nd)
            @test st == SDS.pivot_stats(Nr) && Nh.totals == Nr.totals
            herm = structure == "H" || T <: Real
            herm && eig && @test (st.npos, st.nneg) == eigen_npos_nneg(A)
            herm && @test st.npos + st.nneg == size(A, 1)
            # device solve: forward (local pivot orders), diagonal, backward sweeps
            ws = SDS.allocate_solve(Sd, T, backend, 5)
            for nrhs in (1, 5), det in (false, true)
                b = nrhs == 1 ? rand(T, size(A, 1)) : rand(T, size(A, 1), nrhs)
                x = device_solve(backend, ws, Sd, Nd, b; deterministic = det)
                if name == "kkt(300,100,1e-8)" && T in (Float32, ComplexF32)
                    # T14 / issue #66: this draw needs one refinement step in ComplexF32 (the reference too); in
                    # Float32 the reference factor itself exceeds tol(T) on 1 of 20 right-hand sides (5.3e-4,
                    # #75), and the regime-C GEMMs (rounding only) hit such a draw
                    x = x + device_solve(backend, ws, Sd, Nd, b - A * x; deterministic = det)
                end
                @test relres(A, x, b) <= tol(T)
            end
        end
    end
end

@testset "regime-A groups and Int64 maps ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    Random.seed!(666)
    A = kkt_matrix(T, 200, 100, 0.0; hessian = :indefinite, hessian_scale = 1.0e-3)
    opts = Options(user_perm = kkt_interleaved_perm(200, 100), subtree_budgets = [8192, 16384, 32768, 49152])
    S, Nr, Sd, Nd, nz = ldlt_setup(backend, A, Int64; opts)
    @test !isempty(Nd.plan.sub_first) && SDS.pivot_stats(Nr).n2x2 > 0
    SDS.factorize!(Nd, Sd, nz; opts)
    Nh = SDS.host_numeric(Nd)
    @test Nh.piv == Nr.piv && Nh.pivot_kind == Nr.pivot_kind
    @test d_error(Nh, Nr) <= panel_tol(T) && panel_error(Nh, Nr) <= panel_tol(T)
    # views of the matrix give the same factor
    for view in ('U', 'F')
        _, _, Sv, Nv, nzv = ldlt_setup(backend, A, Int64; view, opts)
        SDS.factorize!(Nv, Sv, nzv; opts)
        @test to_host(Nv.factor) == Nh.factor && to_host(Nv.d) == Nh.d && to_host(Nv.piv) == Nh.piv
    end
end

@testset "pivot_epsilon = 0: a zero pivot fails ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    Random.seed!(666)
    n, j = 60, 23
    F = SDS.FRONT_STATS_FIELDS
    # an exactly zero pivot is tiny whatever ε; with ε = 0 it stays zero and info is its original column (#65)
    for (rname, rkw) in (("default", (;)), ("regime-C root", (factorization_alg = "algo2", subtree_budgets = Int[]))),
        stored_zero in (false, true)
        A = singular_block_matrix(T, n, j; stored_zero)
        opts = Options(; pivot_epsilon = 0.0, rkw...)
        S, Nr, info_ref, C = reference_ldlt(A; opts)
        @test info_ref == j
        Sd = SDS.adapt(backend, S, Int32)
        Nd = SDS.allocate_numeric(Sd, T, backend)
        @test SDS.factorize!(Nd, Sd, to_device(backend, C.nzval); opts) == j
        Nh = SDS.host_numeric(Nd)
        @test Nh.pivot_kind == Nr.pivot_kind && Nh.piv == Nr.piv
        @test d_error(Nh, Nr) <= panel_tol(T)
        k = findfirst(==(SDS.PIVOT_KIND_PERTURBED), Nh.pivot_kind)
        @test k !== nothing && iszero(Nh.d[k]) && iszero(Nr.d[k])
        @test to_host(Nd.stats)[F:F:end] == Nr.stats[F:F:end]
        @test count(!iszero, Nr.stats[F:F:end]) == 1
        st = SDS.pivot_totals(Nd)
        @test st.nzero == 1 && st.nperturbed == 1
    end
    # the public API reports it in "info"; ε > 0 again completes
    solver = DirectSolver(api_matrix(backend, tril(singular_block_matrix(T, n, j))), sym_structure(T), 'L')
    setparam!(solver, "pivot_epsilon", 0.0)
    execute!("analysis", solver, nothing, nothing)
    execute!("factorization", solver, nothing, nothing)
    @test getparam(solver, "info") == j
    setparam!(solver, "pivot_epsilon", nothing)
    execute!("refactorization", solver, nothing, nothing)
    @test getparam(solver, "info") == 0
end

@testset "perturbation and pivot_sign ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    n, j = 60, 23
    R = real(T)
    # the zero pivot in its own root front, regime C ("algo2"), or as the analysis schedules it
    for (rname, rkw) in (("default", (;)), ("regime-C root", (factorization_alg = "algo2", subtree_budgets = Int[]))),
        stored_zero in (false, true), sgn in (1, -1)
        A = singular_block_matrix(T, n, j; stored_zero)
        psign = zeros(Int8, n)
        psign[j] = sgn
        opts = Options(; pivot_sign = psign, rkw...)
        S, Nr, Sd, Nd, nz = ldlt_setup(backend, A; opts)
        @test SDS.factorize!(Nd, Sd, nz; opts) == 0
        Nh = SDS.host_numeric(Nd)
        # the same pivot sequence; D to rounding (the device may contract to FMA), perturbed pivots exactly
        @test Nh.pivot_kind == Nr.pivot_kind && Nh.piv == Nr.piv
        @test d_error(Nh, Nr) <= panel_tol(T)
        pert = Nr.pivot_kind .== SDS.PIVOT_KIND_PERTURBED
        @test Nh.d[1:n][pert] == Nr.d[1:n][pert]
        st = SDS.pivot_totals(Nd)
        @test st.nperturbed >= 1 && st.nzero >= 1
        k = findfirst(==(SDS.PIVOT_KIND_PERTURBED), Nh.pivot_kind)
        _, _, p = SDS.extract_ldlt(S, Nh)
        @test k !== nothing && p[k] == j
        @test real(Nh.d[k]) == sgn * R(SDS.default_pivot_epsilon(R)) && iszero(imag(Nh.d[k]))
        @test (st.npos, st.nneg) == (sgn > 0 ? (n, 0) : (n - 1, 1))
        if rname == "regime-C root"
            s = findfirst(s -> k in SDS.sncols(S.partition, s), 1:SDS.nsupernodes(S))
            @test S.partition.snparent[s] == 0 && SDS.takes_c_path(S.schedule, s)
        end
        x0 = rand(T, n)
        x0[j] = 0
        b = A * x0
        ws = SDS.allocate_solve(Sd, T, backend, 1)
        @test relres(A, device_solve(backend, ws, Sd, Nd, b), b) <= 1.0e-6
    end
    # pivot_epsilon, the scaled variant (ε max|aᵢⱼ| on the device), 'N' and 'D'
    A = singular_block_matrix(T, n, j)
    for opts in (Options(pivot_epsilon = 1.0e-3, pivot_sign = Int8.(-(1:n .== j))),
                 Options(pivot_epsilon = 1.0e-3, pivot_epsilon_alg = "algo1"),
                 Options(pivot_type = 'N'), Options(pivot_type = 'D'))
        S, Nr, Sd, Nd, nz = ldlt_setup(backend, A; opts)
        SDS.factorize!(Nd, Sd, nz; opts)
        Nh = SDS.host_numeric(Nd)
        @test Nh.pivot_kind == Nr.pivot_kind && Nh.piv == Nr.piv
        @test d_error(Nh, Nr) <= panel_tol(T)
        pert = Nr.pivot_kind .== SDS.PIVOT_KIND_PERTURBED
        @test Nh.d[1:n][pert] == Nr.d[1:n][pert]
        @test SDS.pivot_totals(Nd).nperturbed == 1
    end
    # pivot_sign of the wrong length
    S, Nr, Sd, Nd, nz = ldlt_setup(backend, A)
    @test thrown(() -> SDS.factorize!(Nd, Sd, nz; opts = Options(pivot_sign = ones(Int8, n - 1)))) isa
          InvalidValueError
end

@testset "regime C: blocked pivot steps and GEMMs ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    # every front on the regime-C path (`panel_ldlt_kernel!` per block of nb columns, GEMMs on the trailing
    # columns and the contribution block); block sizes below the root's width, so that 2×2 pivots straddle
    # block boundaries; every dense implementation of the GEMMs
    Random.seed!(666)
    conly = (factorization_alg = "algo2", subtree_budgets = Int[])
    cases = (("kkt 1e-3 H, δ = 0, interleaved (2×2)",
              kkt_matrix(T, 200, 100, 0.0; hessian = :indefinite, hessian_scale = 1.0e-3),
              (user_perm = kkt_interleaved_perm(200, 100),)),
             ("random_symindef(200,0.5)", random_symindef(T, 200, 0.5), (;)))
    for (name, A, kw) in cases
        opts = Options(; kw..., conly...)
        S, Nr, Sd, Nd, nz = ldlt_setup(backend, A; opts)
        sp = S.partition
        roots = findall(==(0), sp.snparent)
        s = roots[argmax([SDS.snwidth(sp, r) for r in roots])]
        w = SDS.snwidth(sp, s)
        # a small block size at which a 2×2 pivot starts at the last column of a block (every front is on
        # the regime-C path)
        straddle(nb) = any(1:SDS.nsupernodes(sp)) do v
            c0, wv = sp.super_ptr[v], SDS.snwidth(sp, v)
            any(k -> k % nb == 0 && Nr.pivot_kind[c0 + k - 1] == SDS.PIVOT_KIND_2X2_FIRST, 1:(wv - 1))
        end
        small = findfirst(straddle, 2:16)
        name == first(cases[1]) && @test SDS.pivot_stats(Nr).n2x2 > 0 && small !== nothing
        for nb in (something(small, 2) + 1, SDS.LDLT_C_NB)
            @test SDS.takes_c_path(S.schedule, s) && w > nb
            for impl in SDS.dense_impls(:gemm, backend, T)
                @test SDS.factorize_ldlt!(Nd, Sd, nz; impl, opts, nb) == 0
                Nh = SDS.host_numeric(Nd)
                @test Nh.piv == Nr.piv
                @test Nh.pivot_kind == Nr.pivot_kind
                @test d_error(Nh, Nr) <= panel_tol(T)
                @test panel_error(Nh, Nr) <= panel_tol(T)
                @test Nh.stats == Nr.stats
                @test SDS.pivot_totals(Nd) == SDS.pivot_stats(Nr)
                ws = SDS.allocate_solve(Sd, T, backend, 1)
                b = rand(T, size(A, 1))
                @test relres(A, device_solve(backend, ws, Sd, Nd, b; deterministic = true), b) <= tol(T)
            end
        end
    end
    # the block size is checked
    S, Nr, Sd, Nd, nz = ldlt_setup(backend, cases[2][2]; opts = Options(; conly...))
    @test thrown(() -> SDS.factorize_ldlt!(Nd, Sd, nz; nb = 0)) isa InvalidValueError
    @test thrown(() -> SDS.factorize_ldlt!(Nd, Sd, nz; nb = SDS.LDLT_C_NB + 1)) isa InvalidValueError
end

@testset "pivot_type 'D' and 'N' on quasi-definite KKT ($(backend_name(backend)), $T)" for backend in BACKENDS,
                                                                                         T in ELTYPES
    nh, nj = 300, 100
    A = kkt_matrix(T, nh, nj, 1.0e-2)
    for pt in ('D', 'N')
        opts = Options(pivot_type = pt)
        S, Nr, Sd, Nd, nz = ldlt_setup(backend, A; opts)
        SDS.factorize!(Nd, Sd, nz; opts)
        Nh = SDS.host_numeric(Nd)
        @test Nh.piv == Nr.piv && Nh.pivot_kind == Nr.pivot_kind && d_error(Nh, Nr) <= panel_tol(T)
        st = SDS.pivot_totals(Nd)
        @test st.n2x2 == 0 && st.nperturbed == 0 && (st.npos, st.nneg) == (nh, nj)
        pt == 'N' && @test Nh.piv == 1:(nh + nj)
    end
end

@testset "determinism and refactorization ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    A = kkt_matrix(T, 200, 100, 0.0; hessian = :indefinite, hessian_scale = 1.0e-3)
    opts = Options(user_perm = kkt_interleaved_perm(200, 100), subtree_budgets = [8192, 16384], regime_c_width = 16,
                   regime_c_rows = 128)
    S, Nr, Sd, Nd, nz = ldlt_setup(backend, A; opts)
    SDS.factorize!(Nd, Sd, nz; opts)
    factor, d, piv = to_host(Nd.factor), to_host(Nd.d), to_host(Nd.piv)
    # new values, then the old ones again: bitwise the same factor
    nz2 = to_device(backend, 2 .* to_host(nz))
    SDS.factorize!(Nd, Sd, nz2; opts)
    @test to_host(Nd.d) ≈ 2 .* d
    SDS.factorize!(Nd, Sd, nz; opts)
    @test to_host(Nd.factor) == factor && to_host(Nd.d) == d && to_host(Nd.piv) == piv
    # solves: the deterministic sweep is bitwise reproducible
    ws = SDS.allocate_solve(Sd, T, backend, 2)
    b = rand(T, 300, 2)
    @test device_solve(backend, ws, Sd, Nd, b; deterministic = true) ==
          device_solve(backend, ws, Sd, Nd, b; deterministic = true)
    # no allocation in the numeric phase and the solve (CPU backend)
    if backend isa CPU
        SDS.factorize!(Nd, Sd, nz; opts)
        @test (@allocated SDS.factorize!(Nd, Sd, nz; opts)) <= ldlt_alloc_budget(S)
        x = similar(b)
        SDS.sweep_solve!(x, ws, Sd, Nd, b)
        @test (@allocated SDS.sweep_solve!(x, ws, Sd, Nd, b)) <= solve_alloc_budget(S, ws; ldlt = true)
    end
end

@testset "solve sweeps: diagonal step and dense path ($(backend_name(backend)), $T)" for backend in BACKENDS,
                                                                                      T in ELTYPES
    A = kkt_matrix(T, 200, 100, 0.0; hessian = :indefinite, hessian_scale = 1.0e-3)
    for (rname, rkw) in LDLT_REGIMES
        opts = Options(; user_perm = kkt_interleaved_perm(200, 100), rkw...)
        S, Nr, Sd, Nd, nz = ldlt_setup(backend, A; opts)
        SDS.factorize!(Nd, Sd, nz; opts)
        ws = SDS.allocate_solve(Sd, T, backend, 3)
        b = rand(T, 300, 3)
        bd = to_device(backend, b)
        # the sweeps one at a time: Y = P b, L Z = Y (local pivot orders), D W = Z, Lᴴ V = W
        SDS.permute_rhs!(ws.Y, bd, Sd.perm)
        Y = to_host(ws.Y)
        SDS.forward_sweep!(ws, Sd, Nd; deterministic = true)
        Z = to_host(ws.Y)
        SDS.diagonal_sweep!(ws, Sd, Nd)
        W = to_host(ws.Y)
        SDS.backward_sweep!(ws, Sd, Nd)
        xd = similar(bd)
        SDS.unpermute_solution!(xd, ws.Y, Sd.perm)
        @test to_host(xd) == device_solve(backend, ws, Sd, Nd, b; deterministic = true)
        @test relres(A, to_host(xd), b) <= tol(T)
        # against the extracted factors (A[q, q] = L D Lᴴ, q = perm ∘ piv): Y[piv] = L Z, Z = D W
        L, D, _ = SDS.extract_ldlt(S, SDS.host_numeric(Nd))
        piv = Int.(to_host(Nd.piv))
        @test norm(L * Z - Y[piv, :]) <= tol(T) * norm(Y)
        @test norm(D * W - Z) <= tol(T) * norm(Z)
        # every dense implementation of the regime-C path
        if rname == "C only"
            for impl in SDS.dense_impls(:trsm, backend, T)
                impl in SDS.dense_impls(:gemm, backend, T) || continue
                @test relres(A, device_solve(backend, ws, Sd, Nd, b; impl), b) <= tol(T)
            end
        end
    end
end

@testset "MadNLP-style inertia correction on the device ($(backend_name(backend)), $T)" for backend in BACKENDS,
                                                                                         T in ELTYPES
    nh, nj = 200, 100
    A = kkt_matrix(T, nh, nj, 0.0; hessian = :indefinite)
    for (label, opts) in MADNLP_ORDERINGS
        @testset "$label" begin
            done, iterations, solver, δw = madnlp_inertia_loop_device(backend, A, nh, nj, opts)
            @test done && iterations > 1
            @test getparam(solver, "npivots") == 0
            # the same answer as the reference loop
            done_r, iterations_r, _, _, δw_r = madnlp_inertia_loop(A, nh, nj, opts)
            @test done_r && iterations == iterations_r && δw == δw_r
            Aw = A + spdiagm(0 => [fill(T(δw), nh); zeros(T, nj)])
            b = rand(T, nh + nj)
            x = to_device(backend, zeros(T, nh + nj))
            execute!("solve", solver, x, to_device(backend, b))
            @test relres(Aw, to_host(x), b) <= tol(T)
        end
    end
end

@testset "public API ($(backend_name(backend)), $T, $INT)" for backend in BACKENDS, T in ELTYPES, INT in INTTYPES
    nh, nj = 200, 100
    n = nh + nj
    A = kkt_matrix(T, nh, nj, 0.0; hessian = :indefinite, hessian_scale = 1.0e-3)
    solver = DirectSolver(api_matrix(backend, tril(A), INT), sym_structure(T), 'L')
    setparam!(solver, "user_perm", kkt_interleaved_perm(nh, nj))
    setparam!(solver, "deterministic_mode", 1)
    for name in ("npivots", "inertia", "pivot_stats")
        @test thrown(() -> getparam(solver, name)) isa FactorizationError
    end
    b = to_device(backend, rand(T, n, 2))
    x, y = similar(b), similar(b)
    execute!("analysis", solver, x, b)
    execute!("factorization", solver, x, b)
    @test getparam(solver, "info") == 0
    # the pivot statistics against the reference factorization of the same analysis
    S = solver.host_symbolic
    Nr = SDS.allocate_numeric(S, T)
    SDS.ref_ldlt!(Nr, S, SDS.CSR(tril(A)).nzval; opts = solver.options)
    st = SDS.pivot_stats(Nr)
    @test st.n2x2 > 0
    @test getparam(solver, "pivot_stats") == st
    @test getparam(solver, "inertia") == (st.npos, st.nneg) == eigen_npos_nneg(A)
    @test getparam(solver, "inertia") isa Tuple{INT, INT}
    @test getparam(solver, "npivots") == 0 && getparam(solver, "npivots") isa INT
    d = getparam(solver, "diag")
    @test typeof(KernelAbstractions.get_backend(d)) == typeof(backend)
    @test norm(to_host(d) - Nr.d[1:n]) <= panel_tol(T) * norm(Nr.d)
    hd = zeros(T, n)
    getparam!(hd, solver, "diag")
    @test hd == to_host(d)
    # the sub-phases, "solve_diag" included, compose to "solve"
    execute!("solve", solver, x, b)
    @test relres(A, to_host(x), to_host(b)) <= tol(T)
    for phase in ("solve_fwd_perm", "solve_fwd", "solve_diag", "solve_bwd", "solve_bwd_perm")
        execute!(phase, solver, y, b)
    end
    @test to_host(y) == to_host(x)
    # logabsdet = log|det A| with the sign of det A
    la, sg = logabsdet(solver)
    la_ref, sg_ref = logabsdet(Matrix(A))
    @test la ≈ la_ref rtol = sqrt(eps(real(T)))
    @test sg ≈ sg_ref
    # pivot_sign: wrong length rejected; host or device vector used by the next factorization
    @test thrown(() -> setparam!(solver, "pivot_sign", ones(Int8, n - 1))) isa InvalidValueError
    setparam!(solver, "pivot_sign", to_device(backend, [ones(Int8, nh); -ones(Int8, nj)]))
    @test getparam(solver, "pivot_sign") == [ones(Int8, nh); -ones(Int8, nj)]
    execute!("refactorization", solver, x, b)
    @test getparam(solver, "inertia") == (st.npos, st.nneg)
    # a zero pivot is perturbed with the requested sign
    j = 23
    Z = singular_block_matrix(T, 60, j)
    for sgn in (1, -1)
        zs = DirectSolver(api_matrix(backend, tril(Z), INT), sym_structure(T), 'L')
        setparam!(zs, "pivot_sign", Int8.(sgn .* (1:60 .== j)))
        execute!("analysis", zs, nothing, nothing)
        execute!("factorization", zs, nothing, nothing)
        @test getparam(zs, "npivots") == 1
        @test getparam(zs, "inertia") == (sgn > 0 ? (60, 0) : (59, 1))
        ps = getparam(zs, "pivot_stats")
        @test ps.nperturbed == 1 && ps.nzero == 1
    end
    # the LinearAlgebra layer
    F = ldlt(SDS.CSR(SparseMatrixCSC{T, INT}(A)); view = 'F')
    backend isa CPU && @test relres(A, F \ to_host(b), to_host(b)) <= tol(T)
    @test F.structure == SDS.convert(SDS.Structure, sym_structure(T))
end

@testset "complex symmetric through the API ($(backend_name(backend)), $T)" for backend in BACKENDS,
                                                                              T in COMPLEX_ELTYPES
    A = random_symindef(T, 300, 0.02; hermitian = false)
    solver = DirectSolver(api_matrix(backend, tril(A), Int32), "S", 'L')
    execute!("analysis", solver, nothing, nothing)
    execute!("factorization", solver, nothing, nothing)
    @test getparam(solver, "inertia") == (0, 0)
    b = rand(T, 300, 3)
    x = to_device(backend, zeros(T, 300, 3))
    execute!("solve", solver, x, to_device(backend, b))
    @test relres(A, to_host(x), b) <= tol(T)
    la, sg = logabsdet(solver)
    la_ref, sg_ref = logabsdet(Matrix(A))
    @test la ≈ la_ref rtol = sqrt(eps(real(T)))
    @test sg ≈ sg_ref rtol = sqrt(eps(real(T)))
end
