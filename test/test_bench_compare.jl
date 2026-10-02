# The cuDSS comparison scripts (bench/features.jl, bench/compare_report.jl): the
# feature table, the TASKS.md status gate and the Markdown rendering on synthetic
# CSVs. CPU only, no solver runs, no plotting (bench/compare.jl needs a GPU or the
# bench environment and is run by hand).

using DelimitedFiles

include(joinpath(@__DIR__, "..", "bench", "matrices.jl"))
include(joinpath(@__DIR__, "..", "bench", "compare_report.jl"))
const BF = BenchCompareReport.BenchFeatures

@testset "bench/features.jl: feature table" begin
    feats = BF.COMPARE_FEATURES
    @test allunique(f.id for f in feats)
    @test all(f.structure in ("SPD", "S", "G") for f in feats)
    @test all(f.kind in (:single, :ubatch, :nubatch, :schur) for f in feats)
    @test all(BF.task_status(f.task) in (' ', '~', 'x', '!') for f in feats)  # every task has a header
    @test BF.feature("ldlt").task == "T15"
    @test_throws ErrorException BF.feature("nope")
    @test !feats[findfirst(f -> f.id == "mixed_precision", feats)].cudss_supported

    # every feature selects matrices from the harness (generated, SuiteSparse names, dump names)
    fake(name, source, structures) = (; name, source, structures)
    harness = vcat([fake(M.name, M.source, M.structures) for M in BenchMatrices.generated_matrices()],
                   [fake("GHS_psdef/apache2", :suitesparse, ["SPD", "S"]), fake("Rajat/rajat21", :suitesparse, ["G"]),
                    fake("kkt_pglib_opf_case1354_pegase_condensed_1", :dump, ["SPD", "S"]),
                    fake("kkt_pglib_opf_case1354_pegase_k2_1", :dump, ["S"])])
    for f in feats
        @test any(M -> f.structure in M.structures && f.matrices(M), harness)
    end
    k2 = BF.feature("ldlt_ir2")
    @test [M.name for M in harness if k2.matrices(M)] == ["kkt_pglib_opf_case1354_pegase_k2_1"]
end

@testset "bench/features.jl: TASKS.md status gate" begin
    mktempdir() do dir
        path = joinpath(dir, "TASKS.md")
        write(path, """
            ## T13 — Public API v0.1   `[!]`
            text mentioning T15 — something `[x]`
            ## T15 — GPU LDLᵀ   `[ ]`
            ### T25 — Performance pass   `[x]`
            """)
        @test BF.task_status("T13"; tasks_md = path) == '!'
        @test BF.task_status("T15"; tasks_md = path) == ' '
        @test BF.task_status("T25"; tasks_md = path) == 'x'
        @test_throws ErrorException BF.task_status("T99"; tasks_md = path)
        @test BF.feature_implemented(BF.feature("cholesky_f64"); tasks_md = path)
        @test !BF.feature_implemented(BF.feature("ldlt"); tasks_md = path)
    end
end

@testset "bench/compare_report.jl: ratios and Markdown" begin
    cols = ["solver", "feature", "task", "matrix", "structure", "n", "nnz", "T", "nrhs", "nbatch",
            "analysis_s", "factorization_s", "refactorization_s", "solve_s",
            "analysis_min_s", "factorization_min_s", "refactorization_min_s", "solve_min_s",
            "lu_nnz", "flops", "nsuperpanels", "relres", "status", "device", "version", "git_sha", "date"]
    row(solver, feature, matrix, t, status = "ok") =
        [solver, feature, "T13", matrix, "SPD", "100", "500", "Float64", "1", "1",
         (status == "ok" ? fill(string(t), 8) : fill("", 8))..., (status == "ok" ? ["1000", "2.0e6", "0", "1.0e-12"] : fill("", 4))...,
         status, "GPU", "0.8.0", "abc123", "2026-10-02"]
    mktempdir() do dir
        write_rows(path, rows) = open(io -> writedlm(io, [permutedims(cols); permutedims(reduce(hcat, rows))], ','), path, "w")
        write_rows(joinpath(dir, "cudss.csv"), [row("cudss", "cholesky_f64", "a", 0.001), row("cudss", "cholesky_f64", "b", 0.002),
                                               row("cudss", "ldlt", "a", 0.001)])
        write_rows(joinpath(dir, "sds.csv"), [row("sds", "cholesky_f64", "a", 0.002), row("sds", "cholesky_f64", "b", 0.008),
                                             row("sds", "cholesky_nrhs16", "a", 0.0, "NotSupportedError: not implemented yet (T99)")])
        c = read_rows(joinpath(dir, "cudss.csv"))
        s = read_rows(joinpath(dir, "sds.csv"))
        @test isempty(read_rows(joinpath(dir, "missing.csv")))
        @test length(c) == 3 && c[1]["matrix"] == "a"

        ratios, count = ratio_summary(BF.feature("cholesky_f64"), c, s)
        @test count == 2
        @test ratios["factorization"] ≈ sqrt(2 * 4)   # geometric mean of 2× and 4×
        ratios, count = ratio_summary(BF.feature("ldlt"), c, s)
        @test count == 0 && ratios["solve"] === nothing

        implemented = f -> f.task in ("T12", "T13")
        md = sprint(io -> render_markdown(io, c, s; implemented, plot = false))
        @test occursin("| Cholesky, Float64 | T13 | done | 2 | 2.83× | 2.83× | 2.83× | 2.83× |", md)
        @test occursin("| LDLᵀ, static pivoting | T15 | pending |  |  |  |  |  |", md)
        @test occursin("| a | 100 | 1 | 2 | 2.00× |", md)            # analysis cuDSS 1 ms, SDS 2 ms
        @test occursin("| a | 100 | 1 |  |  |", md)                  # ldlt: SDS cell blank
        @test occursin("fail", md) && occursin("* SDS on a: NotSupportedError", md)
        @test occursin("Not run yet.", md)                             # features without rows
        @test occursin("cuDSS 0.8.0 on GPU, repository abc123, 2026-10-02.", md)
        @test !occursin("comparison.png", md)
        for f in BF.COMPARE_FEATURES
            @test occursin("## " * f.title, md)
        end
    end
end
