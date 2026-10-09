# T24: ND partition tree in the cuDSS encoding ("nd_partition_tree" / "user_nd_partition_tree") and the ordering
# cache (an analysis under a stored "user_perm" reproduces the stored analysis).

# a solver on `backend` for the `view` triangle of `A` (all of it for "G"), with `params` set before the analysis
function ndtree_solver(backend, A::SparseMatrixCSC, structure, ::Type{INT} = Int32; params = ()) where {INT}
    view = structure == "G" ? 'F' : 'L'
    solver = DirectSolver(api_matrix(backend, triangle_view(A, view), INT), structure, view)
    for (name, value) in params
        setparam!(solver, name, value)
    end
    execute!("analysis", solver, nothing, nothing)
    return solver
end

# the analysis outputs that a cached ordering must reproduce
function ndtree_analysis(solver)
    sp = solver.host_symbolic.partition
    sc = solver.host_symbolic.schedule
    return (perm = getparam(solver, "perm_reorder_row"), super_ptr = sp.super_ptr, snparent = sp.snparent,
            lu_nnz = getparam(solver, "lu_nnz"), nsuperpanels = getparam(solver, "nsuperpanels"),
            flops = getparam(solver, "flops"), tree = getparam(solver, "nd_partition_tree"),
            memory = getparam(solver, "memory_estimates"), nlevels = sc.nlevels)
end

@testset "encoding: flat layout and node ranges" begin
    # k = 2: leaves (left, right), then the root; columns: left, right, root
    @test SDS.nd_tree_nodes([2, 3, 1], 6) == [2, 2, 3, 3, 3, 1]
    # k = 3: leaves 4..7 first, then depth-1 nodes 2, 3, the root last; postorder column layout
    tree = [1, 2, 0, 3, 1, 1, 2]           # sizes of heap nodes 4, 5, 6, 7, 2, 3, 1
    @test SDS.nd_tree_nodes(tree, 10) == [4, 5, 5, 2, 7, 7, 7, 3, 1, 1]
    @test [SDS._nd_flat(3, h) for h in 1:7] == [7, 5, 6, 1, 2, 3, 4]
    @test SDS.nd_tree_nodes([5], 5) == fill(1, 5)
    @test SDS.nd_tree_nodes(Int[0], 0) == Int[]
    for (bad, n) in (([1, 2], 3), ([1, 2, 3, 4], 10), ([2, -1, 5], 6), ([2, 3, 2], 6), (Int[], 0))
        @test thrown(() -> SDS.nd_tree_nodes(bad, n)) isa InvalidValueError
    end
    # dependency rule: a chain 1 → 2 → 3 (etree) fits [1, 1, 1] only with the root last
    chain = [2, 3, 0]
    @test SDS.check_nd_partition_tree([2, 0, 1], 2, chain) === nothing
    @test SDS.check_nd_partition_tree([1, 1, 1], 2, [3, 3, 0]) === nothing   # two leaves under a separator
    @test thrown(() -> SDS.check_nd_partition_tree([1, 1, 1], 2, chain)) isa InvalidValueError   # left → right
    @test thrown(() -> SDS.check_nd_partition_tree([2, 0, 1], 3, chain)) isa InvalidValueError   # nd_nlevels
    @test thrown(() -> SDS.check_nd_partition_tree([3], 0, chain)) isa InvalidValueError
end

@testset "export: shape and dependency rule ($alg, $name)" for alg in ("algo3", "algo4", "algo5", "default"),
        (name, A, structure) in (("lap2d", laplacian2d(Float64, 24, 24), "SPD"),
                                 ("lap3d", laplacian3d(Float64, 8, 8, 8), "SPD"),
                                 ("kkt", kkt_matrix(Float64, 120, 40, 1.0e-8), "S"),
                                 ("general", random_general(Float64, 150, 0.03), "G"))
    n = size(A, 1)
    solver = ndtree_solver(CPU(), A, structure; params = (("reordering_alg", alg),))
    sp = solver.host_symbolic.partition
    for k in (1, 2, 5, 10)
        setparam!(solver, "nd_nlevels", k)
        tree = getparam(solver, "nd_partition_tree")
        @test tree == SDS.nd_partition_tree(sp, k)
        @test length(tree) == 2^k - 1 && all(>=(0), tree) && sum(tree) == n
        @test SDS.check_nd_partition_tree(tree, k, sp.parent) === nothing
        k == 1 && @test tree == [n]
        buf = zeros(Int32, 2^k - 1)
        @test getparam!(buf, solver, "nd_partition_tree") == tree
    end
    for k in (0, SDS.ND_TREE_MAX_LEVELS + 1)
        setparam!(solver, "nd_nlevels", k)
        @test thrown(() -> getparam(solver, "nd_partition_tree")) isa InvalidValueError
    end
    if alg == "algo4" && name == "lap2d"
        # nested dissection of a 24 × 24 grid: a grid-line root separator and two balanced halves
        setparam!(solver, "nd_nlevels", 2)
        left, right, root = getparam(solver, "nd_partition_tree")
        @test root <= 2 * 24
        @test min(left, right) >= n ÷ 4
    end
end

@testset "export needs the analysis" begin
    A = laplacian2d(Float64, 8, 8)
    solver = DirectSolver(api_matrix(CPU(), tril(A), Int32), "SPD", 'L')
    @test thrown(() -> getparam(solver, "nd_partition_tree")) isa FactorizationError
    execute!("reordering", solver, nothing, nothing)
    @test thrown(() -> getparam(solver, "nd_partition_tree")) isa FactorizationError
    execute!("symbolic_factorization", solver, nothing, nothing)
    @test sum(getparam(solver, "nd_partition_tree")) == 64
end

@testset "supernode_partition depends only on the etree" begin
    for A in (laplacian2d(Float64, 20, 20), random_spd(Float64, 200, 0.02), kkt_matrix(Float64, 100, 30, 1.0e-8)),
            opts in (Options(), Options(use_superpanels = 0))
        P = SDS.SymmetricPattern(SparseMatrixCSC(A))
        for perm in (collect(1:size(A, 1)), SDS.compute_ordering(P, opts; alg = :amd).perm,
                     SDS.compute_ordering(P, opts; alg = :nd).perm)
            sp = SDS.supernode_partition(P, perm, opts)
            sp2 = SDS.supernode_partition(P, sp.perm, opts)
            @test sp2.perm == sp.perm && sp2.super_ptr == sp.super_ptr && sp2.snparent == sp.snparent
            @test sp2.rowval == sp.rowval && sp2.nnz_L == sp.nnz_L && sp2.nnz_stored == sp.nnz_stored
        end
    end
end

@testset "ordering cache: round trip ($(backend_name(backend)), $T, $structure)" for backend in BACKENDS,
        T in ELTYPES, structure in ("SPD", "S", "G")
    s = structure == "SPD" ? spd_structure(T) : structure == "S" ? sym_structure(T) : "G"
    A = structure == "SPD" ? laplacian3d(T, 7, 7, 6) :
        structure == "S" ? kkt_matrix(T, 150, 50, 1.0e-8) : random_general(T, 200, 0.02)
    n = size(A, 1)
    # the 2×2 pivot pairs are not part of the encoding and not applied under a user_perm
    base = structure == "S" ? (("pivot_pairs", "none"),) : ()
    for alg in ("algo3", "algo4", "default")
        params = (base..., ("reordering_alg", alg))
        ref = ndtree_solver(backend, A, s; params)
        a = ndtree_analysis(ref)
        # import: the permutation (0-based, as cuDSS returns it) and the tree; the ordering algorithm is not run
        cached = ndtree_solver(backend, A, s; params = (base..., ("reordering_alg", "algo5"),
                                                         ("user_perm", Int32.(a.perm .- 1)),
                                                         ("user_nd_partition_tree", a.tree)))
        @test cached.ordering.alg_used === :user
        @test ndtree_analysis(cached) == a
        # user_perm alone gives the same analysis
        @test ndtree_analysis(ndtree_solver(backend, A, s; params = (base..., ("user_perm", a.perm)))) == a
        # and the same factors: solve with both
        b = rand(T, n)
        x = map((ref, cached)) do solver
            execute!("factorization", solver, nothing, nothing)
            api_solve(backend, solver, b)
        end
        @test relres(A, x[1], b) <= tol(T) && relres(A, x[2], b) <= tol(T)
        @test x[2] ≈ x[1] rtol = tol(T)
    end
end

@testset "ordering cache: Schur mode and matching ($T)" for T in eltypes_among((Float64, ComplexF32))
    backend = first(BACKENDS)
    # Schur complement mode: the Schur block stays last under the cached permutation
    A = laplacian2d(T, 12, 12)
    flags = zeros(Int32, 144)
    flags[[5, 40, 77, 100, 141]] .= 1
    params = (("schur_mode", 1), ("user_schur_indices", flags), ("reordering_alg", "algo3"))
    ref = ndtree_solver(backend, A, spd_structure(T); params)
    a = ndtree_analysis(ref)
    cached = ndtree_solver(backend, A, spd_structure(T); params = (params..., ("user_perm", a.perm),
                                                                   ("user_nd_partition_tree", a.tree)))
    @test ndtree_analysis(cached) == a
    # matching ("G"): the permutation refers to the matched matrix of the analysis
    G = random_general(T, 120, 0.04)
    params = (("matching_alg", "algo5"), ("reordering_alg", "algo4"))
    ref = ndtree_solver(backend, G, "G"; params)
    a = ndtree_analysis(ref)
    cached = ndtree_solver(backend, G, "G"; params = (params..., ("user_perm", a.perm),
                                                      ("user_nd_partition_tree", a.tree)))
    @test ndtree_analysis(cached) == a
end

@testset "import: validation" begin
    A = laplacian2d(Float64, 10, 10)
    ref = ndtree_solver(CPU(), A, "SPD"; params = (("reordering_alg", "algo4"),))
    perm = getparam(ref, "perm_reorder_row")
    tree = getparam(ref, "nd_partition_tree")
    fresh(params) = (s = DirectSolver(api_matrix(CPU(), tril(A), Int32), "SPD", 'L');
                     foreach(((k, v),) -> setparam!(s, k, v), params); s)
    analysis_error(params) = thrown(() -> execute!("analysis", fresh(params), nothing, nothing))
    @test analysis_error((("user_nd_partition_tree", tree),)) isa InvalidValueError            # no user_perm
    @test analysis_error((("user_perm", perm), ("user_nd_partition_tree", tree[1:511]))) isa InvalidValueError
    @test analysis_error((("user_perm", perm), ("user_nd_partition_tree", tree),
                          ("nd_nlevels", 9))) isa InvalidValueError                              # 2^9 - 1 ≠ 1023
    bad = copy(tree)
    bad[end] += 1                                                                                  # sizes ≠ n
    @test analysis_error((("user_perm", perm), ("user_nd_partition_tree", bad))) isa InvalidValueError
    # a tree that does not describe the permutation: two sibling leaves, while the ND separator comes last
    wrong = zeros(Int, 1023)
    wrong[1] = wrong[2] = 50
    @test analysis_error((("user_perm", perm), ("user_nd_partition_tree", wrong))) isa InvalidValueError
    # the tree of another permutation
    @test analysis_error((("user_perm", reverse(perm)), ("user_nd_partition_tree", tree))) isa InvalidValueError
    # "reordering" alone validates too, and a valid import analyses
    s = fresh((("user_perm", perm), ("user_nd_partition_tree", bad)))
    @test thrown(() -> execute!("reordering", s, nothing, nothing)) isa InvalidValueError
    s = fresh((("user_perm", perm), ("user_nd_partition_tree", tree), ("nd_nlevels", 10)))
    execute!("analysis", s, nothing, nothing)
    @test getparam(s, "lu_nnz") == getparam(ref, "lu_nnz")
    # a smaller tree with nd_nlevels set accordingly
    setparam!(ref, "nd_nlevels", 3)
    t3 = getparam(ref, "nd_partition_tree")
    s = fresh((("user_perm", perm), ("user_nd_partition_tree", t3), ("nd_nlevels", 3)))
    execute!("analysis", s, nothing, nothing)
    @test getparam(s, "nd_partition_tree") == t3
end
