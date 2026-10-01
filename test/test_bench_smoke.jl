# Smoke test of the benchmark harness (CPU only, no network, no GPU package):
# the generated benchmark matrices, the MatrixMarket/dump-name helpers, and the
# phase timer with a dummy solver closure and with the CHOLMOD reference.

include(joinpath(@__DIR__, "..", "bench", "matrices.jl"))
include(joinpath(@__DIR__, "..", "bench", "harness.jl"))

@testset "bench/matrices.jl: generated matrices" begin
    mats = BenchMatrices.generated_matrices()
    @test [M.name for M in mats] == ["lap2d_300", "lap3d_40"]
    @test [M.name for M in mats] == [g.name for g in BenchMatrices.GENERATED]
    for (M, g) in zip(mats, BenchMatrices.GENERATED)
        @test M isa BenchMatrices.BenchMatrix
        @test M.source === :generated
        @test M.A isa SparseMatrixCSC{Float64,Int}
        @test size(M.A) == (g.n, g.n)
        @test nnz(M.A) == g.nnz
        @test M.structures == ["SPD", "S"]
        @test issymmetric(M.A)
    end
    # Same matrices as the test-suite generators.
    @test mats[1].A == laplacian2d(Float64, 300, 300)
    @test mats[2].A == laplacian3d(Float64, 40, 40, 40)
    # Without network access only the generated matrices are requested.
    @test length(BenchMatrices.bench_matrices(; suitesparse = false, dumps = false)) == 2
    @test [s.name for s in BenchMatrices.SUITESPARSE] ⊇
          ["HB/bcsstk17", "Boeing/bcsstk38", "GHS_psdef/apache2", "Rajat/rajat21", "TSOPF/TSOPF_RS_b39_c7"]
end

@testset "bench/matrices.jl: KKT dumps and MatrixMarket I/O" begin
    @test BenchMatrices.dump_name("pglib_opf_case118_ieee", "k2", 3) == "kkt_pglib_opf_case118_ieee_k2_3.mtx"
    @test BenchMatrices.parse_dump_name("kkt_pglib_opf_case118_ieee_condensed_12.mtx") ==
          (case = "pglib_opf_case118_ieee", kind = "condensed", iter = 12, ext = "mtx")
    @test BenchMatrices.parse_dump_name("/some/dir/kkt_case5_k2_1.jld2").ext == "jld2"
    @test BenchMatrices.parse_dump_name("kkt_case5_lifted_1.mtx") === nothing
    @test BenchMatrices.parse_dump_name("README.md") === nothing
    @test_throws ArgumentError BenchMatrices.dump_name("case5", "k3", 1)

    mktempdir() do dir
        K = kkt_matrix(Float64, 6, 3, 1e-8)  # symmetric indefinite
        C = random_spd(Float64, 9, 0.3)
        G = random_general(Float64, 7, 0.4)
        BenchMatrices.write_mtx(joinpath(dir, BenchMatrices.dump_name("case5", "k2", 1)), K; symmetric = true)
        BenchMatrices.write_mtx(joinpath(dir, BenchMatrices.dump_name("case5", "condensed", 1)), C)
        BenchMatrices.write_mtx(joinpath(dir, "general.mtx"), G)
        write(joinpath(dir, "notes.txt"), "ignored")
        @test BenchMatrices.read_mtx(joinpath(dir, "general.mtx")) == G
        dumps = BenchMatrices.dump_matrices(dir)
        @test [M.name for M in dumps] == ["kkt_case5_condensed_1", "kkt_case5_k2_1"]
        @test dumps[1].A == C && dumps[1].structures == ["SPD", "S"]
        @test dumps[2].A == K && dumps[2].structures == ["S"]
        @test all(M -> M.source === :dump, dumps)
    end
    @test BenchMatrices.dump_matrices(joinpath(tempdir(), "sds-no-such-dir")) == []

    # Explicitly stored zeros of a KKT triangle survive symmetrization and MatrixMarket I/O.
    L = sparse([1, 2, 3, 3], [1, 1, 2, 3], [4.0, 0.0, -1.0, 2.0], 3, 3)
    S = BenchMatrices.symmetrize_triangle(L)
    @test S == [4.0 0.0 0.0; 0.0 0.0 -1.0; 0.0 -1.0 2.0]
    @test nnz(S) == 6
    mktempdir() do dir
        path = BenchMatrices.write_mtx(joinpath(dir, "kkt_z_k2_1.mtx"), S; symmetric = true)
        @test nnz(BenchMatrices.read_mtx(path)) == 6
        @test nnz(only(BenchMatrices.dump_matrices(dir)).A) == 6
    end
end

@testset "bench/harness.jl: time_phases" begin
    A = BenchMatrices.laplacian2d(10, 10)
    b = collect(1.0:size(A, 1))
    calls = Symbol[]
    dummy = (A, b) -> (analysis = () -> push!(calls, :analysis),
                       factorization = () -> push!(calls, :factorization),
                       refactorization = () -> push!(calls, :refactorization),
                       solve = () -> (push!(calls, :solve); Matrix(A) \ b),
                       stats = () -> (lu_nnz = 42, nsuperpanels = 3))
    nsync = Ref(0)
    r = BenchHarness.time_phases(dummy, A, b; nruns = 3, nwarmup = 1, synchronize = () -> nsync[] += 1)
    @test r isa NamedTuple
    @test propertynames(r) == (:analysis, :factorization, :refactorization, :solve, :samples, :nruns,
                               :lu_nnz, :flops, :nsuperpanels, :relres)
    @test calls == repeat([:analysis, :factorization, :refactorization, :solve], 4)
    @test nsync[] == 2 * 4 * 4
    @test r.nruns == 3
    @test propertynames(r.samples) == BenchHarness.PHASES
    for p in BenchHarness.PHASES
        @test length(r.samples[p]) == 3
        @test all(≥(0), r.samples[p])
        @test r[p] == sort(r.samples[p])[2]  # median of 3
    end
    @test r.lu_nnz == 42 && r.nsuperpanels == 3 && r.flops === missing
    @test r.relres ≤ tol(Float64)
    @test_throws ArgumentError BenchHarness.time_phases(dummy, A, b; nruns = 0)

    # Without `stats` every statistic is missing.
    nostats = (A, b) -> (analysis = () -> nothing, factorization = () -> nothing,
                         refactorization = () -> nothing, solve = () -> b)
    r0 = BenchHarness.time_phases(nostats, A, b; nruns = 1, nwarmup = 0)
    @test r0.lu_nnz === missing && r0.flops === missing && r0.nsuperpanels === missing

    # The CPU reference solver used by `cudss_baseline.jl --solver=cholmod`.
    for (structure, M) in (("SPD", A), ("S", A), ("G", random_general(Float64, 50, 0.1)))
        rc = BenchHarness.time_phases(BenchHarness.cholmod_solver(structure), M, ones(size(M, 1)); nruns = 2)
        @test rc.relres ≤ tol(Float64)
        @test rc.lu_nnz ≥ nnz(structure == "G" ? M : tril(M))
    end
    @test_throws ArgumentError BenchHarness.cholmod_solver("H")

    # CSV rows: one per (matrix, structure), failures keep the row with a status.
    row = BenchHarness.csv_row("dummy", "lap2d_10", A, "SPD", r)
    @test length(row) == length(BenchHarness.CSV_COLUMNS)
    @test row[end] == "ok" && row[3] == "100" && row[10] == "42" && row[11] == ""
    bad = BenchHarness.csv_row("dummy", "lap2d_10", A, "S", ErrorException("cuDSS info = 7, \"x\""))
    @test length(bad) == length(BenchHarness.CSV_COLUMNS) && !occursin(',', bad[end])
    mktempdir() do dir
        path = BenchHarness.write_csv(joinpath(dir, "sub", "out.csv"), [row, bad])
        lines = readlines(path)
        @test lines[1] == join(BenchHarness.CSV_COLUMNS, ',')
        @test length(lines) == 3
        @test all(l -> count(==(','), l) == length(BenchHarness.CSV_COLUMNS) - 1, lines)
    end
end
