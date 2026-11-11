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

# #194: the kernel site that factors each column of an "S" FusedSolver: :A (regime-A subtree), :P (narrow panel
# with the in-panel TRSM, at most SYRK_IN rows below the panel), :PT (narrow panel with the TRSM and TILE roles), :W
# (in-kernel wide panel)
function fused_column_sites(F)
    S = F.inner.symbolic
    sp = to_host(S.super_ptr)
    site = fill(:none, length(sp) - 1)
    site[to_host(S.subtree_nodes)] .= :A
    role, ra = to_host(F.fp.role), to_host(F.fp.ra)
    for k in eachindex(role)
        v = Int(ra[k])
        if role[k] == SDS.Fused.ROLE_PANEL_B || role[k] == SDS.Fused.ROLE_PANEL_C
            site[v] = S.schedule.rows[v] - S.schedule.width[v] <= SDS.Fused.SYRK_IN ? :P : :PT
        elseif role[k] == SDS.Fused.ROLE_WPANEL
            site[v] = :W
        end
    end
    return [site[v] for v in 1:(length(sp) - 1) for _ in sp[v]:(sp[v + 1] - 1)]
end

# #194: refactorize the inner DirectSolver of an analyzed "S" FusedSolver with the stock static LDLᵀ (same analysis,
# values, scaling, pivot_epsilon and pivot_sign), then the FusedSolver; returns the host copies of both factors (Nh
# fused, Nr stock) and the stock "info", "npivots" and "inertia"
function fused_stock_oracle(F)
    execute!("refactorization", F.inner, nothing, nothing; asynchronous = false)
    Nr = SDS.host_numeric(F.inner.numeric)
    stock = (; info = getparam(F.inner, "info"), npivots = getparam(F.inner, "npivots"),
             inertia = getparam(F.inner, "inertia"))
    refactorize!(F)
    return SDS.host_numeric(F.inner.numeric), Nr, stock
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

    # issue #141: under nested dissection the root front of a saddle-point matrix is wider than 64 columns and
    # its pivots take both signs. Both wide-front strategies factor it (`wides = :host` once handed it to the
    # potrf-derived LDLᵀ, whose Cholesky fails on the first pivot of the other sign).
    @testset "\"S\": a wide front with pivots of both signs, wides = :$wides: $T" for T in eltypes_among((Float32, Float64)),
                                                                                    wides in (:host, :merged)
        nx = 40
        m = nx * nx
        A = grid_kkt_matrix(T, nx, nx, 1.0)
        F, _, _, r = fused_solve(backend, A, "S"; tuning = SDS.Fused.SDSTuning(; wides))
        @test maximum(F.inner.symbolic.schedule.width) > 64     # the wide front exists (the mechanism is exercised)
        @test isempty(F.fp.wides)                               # and no LDLᵀ front goes to the host path
        @test getparam(F, "info") == 0
        @test getparam(F, "inertia") == (m, m)
        @test r <= tol(T)
        refactorize!(F)
        @test getparam(F, "info") == 0
        @test getparam(F, "inertia") == (m, m)
    end

    @testset "\"S\": a zero pivot sets info without throwing: $T" for T in eltypes_among((Float32, Float64))
        n, j = 40, 17
        A = singular_block_matrix(T, n, j; stored_zero = true)
        F = FusedSolver(fused_matrix(backend, A), "S", 'L')
        setparam!(F, "pivot_epsilon", 0.0)            # epsilon 0 replaces no pivot (#194): the failure path
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

    # #194: static pivot replacement, the rule of the stock no-pivot LDLᵀ. The zero pivot of singular_block_matrix
    # (any order meets it) becomes pivot_sign[j]·ε exactly, with the default ε and a larger one; the fused factor
    # matches the stock factor of the inner solver on the same analysis (pivot kinds exactly, D within growth_tol)
    @testset "\"S\": a zero pivot replaced with pivot_sign $sgn, pivot_epsilon = $eps: $T" for T in eltypes_among((Float32, Float64)),
                                                                                         sgn in (1, -1),
                                                                                         eps in (nothing, 1.0e-3)
        n, j = 40, 17
        R = real(T)
        A = singular_block_matrix(T, n, j; stored_zero = true)
        psign = zeros(Int8, n)
        psign[j] = sgn
        F = FusedSolver(fused_matrix(backend, A), "S", 'L')
        setparam!(F, "pivot_sign", psign)
        eps === nothing || setparam!(F, "pivot_epsilon", eps)
        analyze!(F)
        factorize!(F)
        @test getparam(F, "info") == 0
        @test getparam(F, "npivots") == 1
        @test getparam(F, "inertia") == (sgn > 0 ? (n, 0) : (n - 1, 1))
        Nh, Nr, stock = fused_stock_oracle(F)
        @test stock.info == 0 && stock.npivots == 1
        @test Nh.pivot_kind == Nr.pivot_kind
        @test d_error(Nh, Nr) <= growth_tol(T, Nr)
        k = findfirst(==(SDS.PIVOT_KIND_PERTURBED), Nh.pivot_kind)
        @test k !== nothing && to_host(F.inner.symbolic.perm)[k] == j     # factor column k is row j
        @test Nh.d[k] == sgn * R(something(eps, SDS.default_pivot_epsilon(R)))
        x0 = rand(T, n)
        x0[j] = 0
        b = A * x0                                    # consistent right-hand side
        x = to_host(F \ to_device(backend, b))
        @test relres(A, x, b) <= tol(T)
    end

    # #194: every kernel site replaces pivots. grid_kkt_matrix 40×40 with δ = 1 has no pivot of modulus in
    # (1.57, 2.98) (Float32 and Float64), so ε = 2.5 replaces the same 1545 pivots in both factorizations, among them
    # pivots of regime-A subtrees, narrow panels on both TRSM paths and in-kernel wide panels
    @testset "\"S\": pivot replacement at every kernel site: $T" for T in eltypes_among((Float32, Float64))
        nx = 40
        m = nx * nx
        A = grid_kkt_matrix(T, nx, nx, 1.0)
        F = FusedSolver(fused_matrix(backend, A), "S", 'L')
        setparam!(F, "pivot_epsilon", 2.5)
        setparam!(F, "pivot_sign", vcat(fill(1, m), fill(-1, m)))
        analyze!(F)
        factorize!(F)
        @test getparam(F, "info") == 0
        Nh, Nr, stock = fused_stock_oracle(F)
        @test getparam(F, "info") == 0
        @test Nh.pivot_kind == Nr.pivot_kind
        @test getparam(F, "npivots") == stock.npivots > 0
        @test getparam(F, "inertia") == stock.inertia == (m, m)
        @test d_error(Nh, Nr) <= growth_tol(T, Nr)
        sites = fused_column_sites(F)
        replaced = sites[Nh.pivot_kind[1:(2m)] .== SDS.PIVOT_KIND_PERTURBED]
        for site in (:A, :P, :PT, :W)
            @test count(==(site), replaced) > 0
        end
    end

    # #194: matching on "S" (the scaled values, the pivot pairs of its cycles in the ordering, the scaling undone in
    # the solve), with pivot replacement on kkt_slack_matrix (δ = 0: the odd slacks and duals meet zero pivots in
    # the static order, which bounds its accuracy as on the stock pipeline: no residual gate there). Against the
    # inner stock factor, a refactorization with new values, bitwise reproducible factor and D, the graph replay
    # (CUDA) equal to the eager factorization, and the allocation budget
    @testset "\"S\", matching_alg = $alg, $label: $T" for T in eltypes_among((Float32, Float64)),
                                                         alg in ("algo5", "algo6"),
                                                         (label, slack) in (("grid_kkt_matrix 30×30", false),
                                                                            ("kkt_slack_matrix 200+2·100, δ = 0", true))
        Random.seed!(666)
        A = slack ? kkt_slack_matrix(T, 200, 100, 0.0) : grid_kkt_matrix(T, 30, 30, 1.0)
        n = size(A, 1)
        Ad = fused_matrix(backend, A)
        F = FusedSolver(Ad, "S", 'L')
        setparam!(F, "matching_alg", alg)
        @test getparam(F, "matching_alg") == alg
        analyze!(F)
        @test getparam(F, "pivot_pairs") == "default"            # the pairs of the matching shape the ordering
        slack && @test !isempty(F.inner.ordering.pairs)
        @test length(getparam(F, "scale_row")) == n
        @test length(getparam(F, "perm_matching")) == n
        factorize!(F)
        @test getparam(F, "info") == 0
        b = rand(T, n)
        bd = to_device(backend, b)
        xd = similar(bd)
        solve!(F, xd, bd; asynchronous = false)
        slack || @test relres(A, to_host(xd), b) <= tol(T)
        Nh, Nr, stock = fused_stock_oracle(F)
        @test Nh.pivot_kind == Nr.pivot_kind
        @test getparam(F, "npivots") == stock.npivots
        @test getparam(F, "inertia") == stock.inertia
        @test d_error(Nh, Nr) <= growth_tol(T, Nr)
        slack && @test stock.npivots > 0
        # the next interior-point values: refactorization with the matching and the scaling of the analysis
        A2 = slack ? (Dg = Diagonal(T.(1 .+ rand(n))); sparse(Dg * A * Dg)) : grid_kkt_matrix(T, 30, 30, 0.5)
        copyto!(Ad.nzVal, fused_values(A2))
        refactorize!(F)
        @test getparam(F, "info") == 0
        solve!(F, xd, bd; asynchronous = false)
        slack || @test relres(A2, to_host(xd), b) <= tol(T)
        N = F.inner.numeric
        f1, d1 = to_host(N.factor), to_host(N.d)
        same = 0
        for _ in 1:3
            refactorize!(F)
            same += to_host(N.factor) == f1 && to_host(N.d) == d1
        end
        @test same == 3
        bytes = fused_cycle_allocated(F, xd, bd)
        @test bytes <= fused_cycle_alloc_budget(backend)
        # the graph replay of the refactorization (the scale launch runs eagerly before it) against the first,
        # eager factorization
        G = FusedSolver(Ad, "S", 'L'; tuning = SDS.Fused.SDSTuning(; graph_max_n = typemax(Int)))
        setparam!(G, "matching_alg", alg)
        analyze!(G)
        factorize!(G)
        fg, dg = to_host(G.inner.numeric.factor), to_host(G.inner.numeric.d)
        sameg = 0
        for _ in 1:3
            refactorize!(G)
            sameg += getparam(G, "info") == 0 && to_host(G.inner.numeric.factor) == fg &&
                     to_host(G.inner.numeric.d) == dg
        end
        @test sameg == 3
        backend_name(backend) == "CUDA" && @test !(G.gfact in (nothing, false))
    end

    # #194: iterative refinement on the fused solve (ir_n_steps, read at every solve) with matching and pivot
    # replacement. kkt_slack_matrix with δ = 0 under algo5 meets zero pivots, replaced by ±ε (the default ε), which
    # leaves the unrefined solve inaccurate; the residual of each step is the one of the caller's values (neither
    # scaled nor with the replaced pivots), so one step restores the accuracy. In place (X === B: the copy of B for the
    # residual) equal to out of place under deterministic_mode = 1; against a stock DirectSolver with the same
    # analysis options and refinement (the same ordering); the graph replay of the corrections (CUDA) equal to the
    # eager solve; the allocation budget of a refining cycle
    @testset "\"S\", ir_n_steps with matching and pivot replacement: $T" for T in eltypes_among((Float32, Float64))
        Random.seed!(666)
        A = kkt_slack_matrix(T, 200, 100, 0.0)
        n = size(A, 1)
        Ad = fused_matrix(backend, A)
        F = FusedSolver(Ad, "S", 'L')
        for (name, value) in (("matching_alg", "algo5"), ("deterministic_mode", 1))
            setparam!(F, name, value)
        end
        analyze!(F)
        factorize!(F)
        @test getparam(F, "info") == 0
        @test getparam(F, "npivots") > 0
        @test getparam(F, "ir_n_steps") == 0
        b = rand(T, n)
        bd = to_device(backend, b)
        xd = similar(bd)
        res = zeros(real(T), 3)
        xk = Vector{Vector{T}}(undef, 3)
        for k in 0:2
            setparam!(F, "ir_n_steps", k)
            @test F.inner.refinement !== nothing || k == 0             # the workspace exists before the solve
            solve!(F, xd, bd; asynchronous = false)
            @test getparam(F, "ir_n_steps") == k                       # the steps of the last solve
            xk[k + 1] = to_host(xd)
            res[k + 1] = relres(A, xk[k + 1], b)
        end
        @test to_host(bd) == b                                         # B is not touched
        @test res[1] > tol(T)                                          # the replaced pivots cost accuracy
        @test res[2] <= tol(T) && res[2] < res[1] / 10
        @test res[3] <= tol(T)
        xb = copy(bd)
        solve!(F, xb, xb; asynchronous = false)                        # X === B: B kept for the residual
        @test to_host(xb) == xk[3]
        # the stock pipeline with the same options and the same refinement
        D = DirectSolver(Ad, "S", 'L')
        SDS.Fused._apply_tuning!(D, F.tuning)
        for (name, value) in (("pivot_type", 'N'), ("matching_alg", "algo5"), ("ir_n_steps", 2))
            setparam!(D, name, value)
        end
        analyze!(D)
        factorize!(D)
        @test getparam(D, "perm_reorder_row") == getparam(F, "perm_reorder_row")
        @test getparam(D, "npivots") == getparam(F, "npivots")
        xs = to_host(D \ bd)
        @test relres(A, xs, b) <= tol(T)
        @test norm(xk[3] - xs) / norm(xs) <= tol(T)
        # a refactorization with new values keeps refining
        A2 = (Dg = Diagonal(T.(1 .+ rand(n))); sparse(Dg * A * Dg))
        copyto!(Ad.nzVal, fused_values(A2))
        refactorize!(F)
        @test getparam(F, "info") == 0
        solve!(F, xd, bd; asynchronous = false)
        @test relres(A2, to_host(xd), b) <= tol(T)
        bytes = fused_cycle_allocated(F, xd, bd)
        @test bytes <= fused_cycle_alloc_budget(backend)
        # the solve graph replays the corrections too: bitwise equal to the eager solve (deterministic_mode = 1) of
        # the same solver (the same analysis, factor and tuning but graph_max_n; the tuning is read at every solve,
        # as in the #174 testset). A fresh solver would not do: its analysis of A2 (the scaling, hence the pivot
        # pairs and the ordering) differs from F's analysis of A, and its tuning from the platform's (solve_workgroup)
        x2 = to_host(xd)
        eager = F.tuning
        F.tuning = SDS.Fused.SDSTuning((f === :graph_max_n ? typemax(Int) : getfield(eager, f)
                                        for f in fieldnames(SDS.Fused.SDSTuning))...)
        same = 0
        for _ in 1:3                                    # eager sweeps and capture at the first solve, then replays
            solve!(F, xb, bd; asynchronous = false)
            same += to_host(xb) == x2
        end
        @test same == 3
        backend_name(backend) == "CUDA" && @test !(F.gsolve in (nothing, false))
        F.tuning = eager
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
                              ("matching_alg", "algo5"), ("ir_tol", 1.0e-10), ("ir_mode", "fgmres"),
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
        # refinement (#194): plain steps on either structure; the early exit (ir_tol, a host synchronization per step)
        # and FGMRES-IR, set behind the back after the analysis, are refused at the solve
        setparam!(F, "ir_n_steps", 2)
        solve!(F, x, b; asynchronous = false)
        @test getparam(F, "ir_n_steps") == 2
        @test relres(A, to_host(x), to_host(b)) <= tol(T)
        F.inner.options.ir_tol = 1.0e-10
        @test_throws NotSupportedError solve!(F, x, b)
        F.inner.options.ir_tol = 0.0
        F.inner.options.ir_mode = SDS.IR_FGMRES
        @test_throws NotSupportedError solve!(F, x, b)
        F.inner.options.ir_mode = SDS.IR_PLAIN
        setparam!(F, "ir_n_steps", 0)
        for (field, value) in ((:ir_tol, 1.0e-10), (:ir_mode, SDS.IR_FGMRES), (:schur_mode, 1),
                               (:matching_alg, SDS.MATCHING_MAX_DIAG_PRODUCT), (:factor_precision, T == Float64 ? Float32 : Float64))
            G = FusedSolver(Ad, "SPD", 'L')
            setfield!(G.inner.options, field, value)
            @test_throws NotSupportedError analyze!(G)
        end
        G = FusedSolver(fused_matrix(backend, random_symindef(T, 50, 0.1)), "S", 'L')
        G.inner.options.pivot_type = SDS.PIVOT_AUTO
        @test_throws NotSupportedError analyze!(G)
        # #194: pivot_sign and matching apply to "S" only; pivot_epsilon_alg = "algo1" is not supported on "S"; a
        # pivot_sign of the wrong length is refused (by the inner solver's setparam!)
        @test_throws NotSupportedError setparam!(F, "pivot_sign", ones(Int, n))
        @test_throws NotSupportedError setparam!(F, "matching_alg", "algo6")
        @test getparam(F, "matching_alg") == "default"
        setparam!(F, "matching_alg", "none")                                    # matching off is accepted
        G = FusedSolver(fused_matrix(backend, random_symindef(T, 50, 0.1)), "S", 'L')
        G.inner.options.pivot_epsilon_alg = SDS.PIVOT_EPSILON_SCALED
        @test_throws NotSupportedError analyze!(G)
        G = FusedSolver(fused_matrix(backend, random_symindef(T, 50, 0.1)), "S", 'L')
        @test_throws InvalidValueError setparam!(G, "pivot_sign", ones(Int, 49))
        @test_throws InvalidValueError solve!(F, to_device(backend, zeros(T, n + 1)), to_device(backend, zeros(T, n + 1)))
    end

    # the T34 knobs off their defaults (which never reach these paths): the per-front wide strategy of a
    # merged Cholesky plan (`wide_host_min`, the segment split), the widest fused-solve workgroup and the
    # graph replay of refactorization and solve (`graph_max_n`; CUDA only, `capture_graph` returns `nothing`
    # elsewhere and the phases launch eagerly). Two refactorizations and solves: the first capture, then a
    # replay with new values in the aliased buffer.
    @testset "SDSTuning T34 knobs ($label): $T" for T in eltypes_among((Float32, Float64)),
                                                    (label, gen) in (("laplacian2d 40×40", () -> laplacian2d(T, 40, 40)),
                                                                     ("laplacian3d 10³", () -> laplacian3d(T, 10, 10, 10)))
        Random.seed!(666)
        A = gen()
        n = size(A, 1)
        tuning = SDS.Fused.SDSTuning(; wides = :merged, wide_host_min = 64, solve_workgroup = 256,
                                     graph_max_n = typemax(Int))
        F, Ad, xd, r = fused_solve(backend, A, "SPD"; tuning)
        @test F.tuning.solve_workgroup == 256
        @test getparam(F, "info") == 0
        @test r <= tol(T)
        label == "laplacian3d 10³" && @test !isempty(F.fp.wides)     # a front wider than 64 on the host path
        b = rand(T, n)
        bd = to_device(backend, b)
        D = DirectSolver(fused_matrix(backend, A), "SPD", 'L')
        setparam!(D, "user_perm", getparam(F, "perm_reorder_row"))
        analyze!(D)
        factorize!(D)
        xs = to_host(D \ bd)
        A2 = T(2) * A + T(n) * I
        for (k, Ak) in enumerate((A, A2))
            k == 2 && copyto!(Ad.nzVal, fused_values(A2))
            refactorize!(F)
            @test getparam(F, "info") == 0
            for _ in 1:2
                solve!(F, xd, bd; asynchronous = false)
                @test relres(Ak, to_host(xd), b) <= tol(T)
            end
            k == 1 && @test norm(to_host(xd) - xs) / norm(xs) <= tol(T)
        end
        if backend_name(backend) == "CUDA"
            # the graphs exist (vendor calls in the captured range included, #174)
            @test (F.gfact !== nothing && F.gfact !== false) == SDS.Fused._graph_fact_ok(F)
            @test (F.gsolve !== nothing && F.gsolve !== false) == SDS.Fused._graph_solve_ok(F)
            @test SDS.Fused._graph_fact_ok(F) && SDS.Fused._graph_solve_ok(F)
        else
            @test F.gfact in (nothing, false) && F.gsolve in (nothing, false)
        end
        # deterministic_mode = 1 (read at the solve): the pull forward sweep, captured as its own graph
        setparam!(F, "deterministic_mode", 1)
        solve!(F, xd, bd; asynchronous = false)
        x1 = to_host(xd)
        @test relres(A2, x1, b) <= tol(T)
        solve!(F, xd, bd; asynchronous = false)
        @test to_host(xd) == x1
        backend_name(backend) == "CUDA" && SDS.Fused._graph_solve_ok(F) &&
            @test F.gdet && F.gsolve !== nothing && F.gsolve !== false
        bytes = fused_cycle_allocated(F, xd, bd)
        @test bytes <= fused_cycle_alloc_budget(backend)
    end

    # #174: graph replay with vendor dense calls in the captured ranges. `wides = :host` puts every wide front
    # of the factorization on the host vendor path (cuSOLVER potrf, cuBLAS trsm/syrk on CUDA), and the 18³
    # Laplacian has a front wider than MW, which the solve sweeps through vendor dense calls. The capture binds
    # the vendor handles to the capture stream first; the replayed factor is bitwise equal to the eager one
    # (the fused factorization is deterministic, T35), the solution too under deterministic_mode = 1.
    @testset "graph replay with host-wide fronts and wide solve fronts (#174): $T" for T in eltypes_among((Float32, Float64))
        Random.seed!(666)
        A = laplacian3d(T, 18, 18, 18)
        n = size(A, 1)
        A2 = T(2) * A + T(n) * I
        eager = SDS.Fused.SDSTuning(; wides = :host)
        graph = SDS.Fused.SDSTuning(; wides = :host, graph_max_n = typemax(Int))
        F, Ad, xd, r = fused_solve(backend, A, "SPD"; tuning = eager)
        @test r <= tol(T)
        @test !isempty(F.fp.wides)                                                # host-wide fronts
        @test maximum(F.inner.symbolic.schedule.width) > SDS.Fused.MW             # vendor fronts in the solve
        setparam!(F, "deterministic_mode", 1)
        b = rand(T, n)
        bd = to_device(backend, b)
        for (k, Ak) in enumerate((A, A2))
            k == 2 && copyto!(Ad.nzVal, fused_values(A2))
            F.tuning = eager
            refactorize!(F)
            @test getparam(F, "info") == 0
            factor = to_host(F.inner.numeric.factor)
            solve!(F, xd, bd; asynchronous = false)
            x = to_host(xd)
            @test relres(Ak, x, b) <= tol(T)
            F.tuning = graph
            for _ in 1:3                                    # capture (or eager at the first solve), then replays
                refactorize!(F)
                @test getparam(F, "info") == 0
                @test to_host(F.inner.numeric.factor) == factor
                solve!(F, xd, bd; asynchronous = false)
                @test to_host(xd) == x
            end
        end
        if backend_name(backend) == "CUDA"
            @test !(F.gfact in (nothing, false)) && !(F.gsolve in (nothing, false))
        else
            @test F.gfact in (nothing, false) && F.gsolve in (nothing, false)
        end
    end

    # T35: the fused factorization is deterministic (the parent-pull extend-add in child_list order, the
    # in-kernel wide-front tiles applying the panels in order, plain adds), and deterministic_mode = 1 (read
    # at every solve) makes the forward solve deterministic too (the pull sweep through the update buffers).
    # On matrices whose plan has extend-add roles (C-group parents with several contribution-block children)
    # and in-kernel wide-front tiles (every "S" wide front; "SPD" with wides = :merged): factor and D bitwise
    # equal over 10 refactorizations under both modes, the solution too under deterministic_mode = 1, the
    # fused factor within growth_tol of the stock factor on the same analysis.
    @testset "deterministic_mode = $mode, \"$structure\" ($label): $T" for T in eltypes_among((Float32, Float64)),
                                       mode in (1, 0),
                                       (structure, label, gen, tuning) in (
            ("SPD", "laplacian3d 18³, wides = :merged", () -> laplacian3d(T, 18, 18, 18), SDS.Fused.SDSTuning(; wides = :merged)),
            ("S", "laplacian3d 18³", () -> laplacian3d(T, 18, 18, 18), nothing),
            ("S", "grid_kkt_matrix 40×40", () -> grid_kkt_matrix(T, 40, 40, 1.0), nothing))
        Random.seed!(666)
        A = gen()
        n = size(A, 1)
        Ad = fused_matrix(backend, A)
        F = FusedSolver(Ad, structure, 'L'; tuning)
        mode == 1 && setparam!(F, "deterministic_mode", 1)
        @test getparam(F, "deterministic_mode") == mode
        analyze!(F)
        roles = Array(F.fp.role)
        @test count(==(SDS.Fused.ROLE_EA), roles) > 0          # the ordered extend-add is exercised
        @test count(==(SDS.Fused.ROLE_WTILE), roles) > 0       # and the ordered wide-front tiles
        factorize!(F)
        @test getparam(F, "info") == 0
        b = rand(T, n)
        bd = to_device(backend, b)
        xd = similar(bd)
        solve!(F, xd, bd; asynchronous = false)
        @test relres(A, to_host(xd), b) <= tol(T)
        N = F.inner.numeric
        f1, d1, x1 = to_host(N.factor), to_host(N.d), to_host(xd)
        inertia = getparam(F, "inertia")
        same = 0
        samex = 0
        for _ in 1:10
            refactorize!(F)
            solve!(F, xd, bd; asynchronous = false)
            same += getparam(F, "info") == 0 && getparam(F, "inertia") == inertia &&
                    to_host(N.factor) == f1 && to_host(N.d) == d1
            samex += to_host(xd) == x1
            @test relres(A, to_host(xd), b) <= tol(T)
        end
        @test same == 10                                        # factor and D bitwise, in both modes
        mode == 1 && @test samex == 10                          # the solution bitwise (deterministic_mode = 1)
        # the host bytes of a refactorization plus a solve
        bytes = fused_cycle_allocated(F, xd, bd)
        @test bytes <= fused_cycle_alloc_budget(backend)
        # deterministic_mode is read at every solve: the other forward sweep on the same factor
        setparam!(F, "deterministic_mode", 1 - mode)
        solve!(F, xd, bd; asynchronous = false)
        @test norm(to_host(xd) - x1) / norm(x1) <= tol(T)
        setparam!(F, "deterministic_mode", mode)
        solve!(F, xd, bd; asynchronous = false)
        mode == 1 && @test to_host(xd) == x1
        # the stock factorization on the same analysis (it overwrites the shared factor storage)
        execute!("refactorization", F.inner, xd, bd; asynchronous = false)
        @test getparam(F.inner, "info") == 0
        _, Nr, info_ref, _ = reference_ldlt(SparseMatrixCSC{Float64, Int}(A))
        @test info_ref == 0
        fs = to_host(N.factor)
        @test maximum(abs, f1 - fs) / maximum(abs, fs) <= growth_tol(T, Nr)
        structure == "S" && @test maximum(abs, d1 - to_host(N.d)) / maximum(abs, to_host(N.d)) <= growth_tol(T, Nr)
        # memory_estimates: the inner solver's plus the fused plans' buffers (slot 13: the private
        # contribution blocks, slot 14: the rest), added to the device totals
        est = getparam(F, "memory_estimates")
        est0 = getparam(F.inner, "memory_estimates")
        @test est[13] == length(F.fp.pcb) * sizeof(T) > 0
        @test est[14] > 0
        @test est[1] == est0[1] + est[13] + est[14] && est[2] == est0[2] + est[13] + est[14]
        @test est[5] == est0[5] + est[13] + est[14]
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
