# Kernel-level gate of the fused pipeline against the stock path and cuDSS, per matrix:
# steady-state refactorization and solve medians, the launch composition (CUDA.@profile)
# and the two wide-front strategies. One solver per process (CUDSS.jl and the CUDA
# extension define the same `cholesky` methods), as bench/compare.jl.
#
#   julia --project=bench bench/e2e/fused_probe.jl [--solver=sds|cudss] [--dump=<path.mtx>] [--only-dump]
#
# Default matrices: the case118/case1354 condensed dumps of bench/data, HB/bcsstk17 and
# Boeing/bcsstk38 (MatrixDepot, cached after the first use), lap2d_300, lap3d_40. `--dump`
# adds an .mtx (the 78k-bus condensed KKT produced by bench/dump_madnlp_kkt.jl, 148 MB, kept
# out of the repository); `--only-dump` runs that matrix alone. Always run on an idle GPU.
#
# Reading the sds output: the first numbers of a block are taken while the device clock
# ramps; the "repeated at the end" refactorization and the pre-T33 recipe line are the
# steady-state values. The profile tables give the launch count and the per-kernel split.
using SparseArrays, LinearAlgebra, Random, Statistics, Printf
using CUDA, CUDA.CUSPARSE

function parse_args(args)
    o = Dict("solver" => "sds", "dump" => "", "only-dump" => false)
    for a in args
        if a == "--only-dump"
            o["only-dump"] = true
        elseif (m = match(r"^--(solver|dump)=(.*)$", a)) !== nothing
            o[m[1]] = m[2]
        else
            error("unknown argument \"$a\"; see the header of bench/e2e/fused_probe.jl")
        end
    end
    o["solver"] in ("sds", "cudss") || error("--solver=sds or --solver=cudss")
    return o
end
const OPTS = parse_args(ARGS)
const SOLVER = OPTS["solver"]

if SOLVER == "cudss"
    using CUDSS
else
    using SparseDirectSolver, Metis
    const SDS = SparseDirectSolver
    const Fused = SDS.Fused
end

const BD = joinpath(@__DIR__, "..")
include(joinpath(BD, "matrices.jl"))
using .BenchMatrices

say(x...) = (println(x...); flush(stdout))
say("device: ", CUDA.name(CUDA.device()), "  solver: ", SOLVER,
    SOLVER == "cudss" ? "  cuDSS $(CUDSS.version())" : "  SDS $(pkgversion(SparseDirectSolver))")

mats = Pair{String, SparseMatrixCSC{Float64, Int}}[]
if !OPTS["only-dump"]
    for M in BenchMatrices.dump_matrices()
        M.name in ("kkt_pglib_opf_case118_ieee_condensed_10", "kkt_pglib_opf_case1354_pegase_condensed_10") || continue
        push!(mats, M.name => SparseMatrixCSC{Float64, Int}(M.A))
    end
    try
        for M in BenchMatrices.suitesparse_matrices(; names = ["HB/bcsstk17", "Boeing/bcsstk38"])
            push!(mats, M.name => SparseMatrixCSC{Float64, Int}(M.A))
        end
    catch err
        say("suitesparse skipped: ", sprint(showerror, err))
    end
    push!(mats, "lap2d_300" => SparseMatrixCSC{Float64, Int}(BenchMatrices.laplacian2d(300, 300)))
    push!(mats, "lap3d_40" => SparseMatrixCSC{Float64, Int}(BenchMatrices.laplacian3d(40, 40, 40)))
end
isempty(OPTS["dump"]) ||
    push!(mats, basename(OPTS["dump"]) => SparseMatrixCSC{Float64, Int}(sparse(BenchMatrices.read_mtx(OPTS["dump"]))))
say("matrices: ", join(first.(mats), ", "))

# keep the device at full clock before a timed block (bench/compare.jl: spin_device)
const SPIN = CUDA.zeros(Float32, 2^24)
function spin_device(seconds = 0.2)
    t0 = time()
    while time() - t0 < seconds
        SPIN .= sin.(SPIN) .+ 1f0
        CUDA.synchronize()
    end
end
wall(f; n = 11, warm = 3) = begin
    spin_device()
    for _ in 1:warm
        f(); CUDA.synchronize()
    end
    median([(@elapsed (f(); CUDA.synchronize())) for _ in 1:n])
end

function profile_show(label, f)
    f(); CUDA.synchronize()
    res = CUDA.@profile f()
    say("---- profile: ", label)
    show(IOContext(stdout, :limit => false, :displaysize => (2000, 300)), res)
    println(); flush(stdout)
end

for (name, A) in mats
    n = size(A, 1)
    Random.seed!(666)
    bh = rand(n)
    Ad = CuSparseMatrixCSR(tril(A))
    bd = CuArray(bh); xd = similar(bd)
    say("\n================ ", name, "  n = ", n, "  nnz(A) = ", nnz(A))
    if SOLVER == "cudss"
        s = CudssSolver(Ad, "SPD", 'L')
        cudss("analysis", s, xd, bd); cudss("factorization", s, xd, bd); cudss("solve", s, xd, bd)
        CUDA.synchronize()
        rel = norm(bh - A * Array(xd)) / norm(bh)
        tr = wall(() -> cudss("refactorization", s, xd, bd))
        ts = wall(() -> cudss("solve", s, xd, bd))
        @printf("%-45s cuDSS refact %7.3f ms  solve %7.3f ms  relres %.2e\n", name, 1e3 * tr, 1e3 * ts, rel)
        flush(stdout)
        continue
    end
    # the FusedSolver with the default tuning of this backend
    F = FusedSolver(Ad, "SPD", 'L')
    say("tuning: ", F.tuning)
    ta = @elapsed analyze!(F; asynchronous = false)
    factorize!(F; asynchronous = false)
    solve!(F, xd, bd; asynchronous = false)
    s = F.inner
    sc = s.symbolic.schedule
    say(@sprintf("analysis %.2f s; supernodes %d, max width %d, nnz(L) %.3e, levels %d; roles %d, segments %d, host wides %d, subtree launches %d, C-group scatter launches %d",
                 ta, length(sc.width), maximum(sc.width), Float64(getparam(F, "lu_nnz")), maximum(sc.level),
                 length(F.fp.role), length(F.fp.segments), length(F.fp.wides), length(F.fp.ssub),
                 count(==(0), s.numeric.plan.group_width)))
    rel = norm(bh - A * Array(xd)) / norm(bh)
    say(@sprintf("fused info %d  relres %.2e", getparam(F, "info"), rel))
    t_f = wall(() -> refactorize!(F))
    t_s = wall(() -> solve!(F, xd, bd))
    # the same objects through the module functions (the pre-T33 extension recipe)
    nz = F.nzval
    t_pre = wall(() -> (Fused.refact_fused!(s, nz, F.fp); F.invert!()))
    t_pre_s = wall(() -> Fused.algo2_solve!(xd, s, bd, F.inv11, F.inv_ptr, F.fz))
    t_noinv = wall(() -> Fused.refact_fused!(s, nz, F.fp))
    # the stock path on the same analysis (same ordering and amalgamation)
    SDS.execute!("refactorization", s, xd, bd; asynchronous = false)
    SDS.execute!("solve", s, xd, bd; asynchronous = false)
    t_stock = wall(() -> SDS.execute!("refactorization", s, xd, bd))
    t_stock_s = wall(() -> SDS.execute!("solve", s, xd, bd))
    @printf("%-45s refact: FusedSolver %7.3f ms | module recipe %7.3f | without the inverse %7.3f | stock %7.3f\n",
            name, 1e3 * t_f, 1e3 * t_pre, 1e3 * t_noinv, 1e3 * t_stock)
    @printf("%-45s solve : FusedSolver %7.3f ms | module recipe %7.3f | stock %7.3f\n",
            name, 1e3 * t_s, 1e3 * t_pre_s, 1e3 * t_stock_s)
    flush(stdout)
    refactorize!(F); CUDA.synchronize()
    profile_show("fused refactorization (wides = :$(F.tuning.wides))", () -> refactorize!(F))
    profile_show("fused solve", () -> solve!(F, xd, bd))
    profile_show("stock refactorization (same analysis)", () -> SDS.execute!("refactorization", s, xd, bd))
    # the other wide-front strategy
    other = F.tuning.wides === :merged ? :host : :merged
    Fo = FusedSolver(Ad, "SPD", 'L'; tuning = Fused.SDSTuning(; wides = other, subtree_budgets = F.tuning.subtree_budgets))
    analyze!(Fo; asynchronous = false)
    factorize!(Fo; asynchronous = false)
    say(@sprintf("wides = :%s: roles %d, segments %d, host wides %d, info %d", other, length(Fo.fp.role),
                 length(Fo.fp.segments), length(Fo.fp.wides), getparam(Fo, "info")))
    t_fo = wall(() -> refactorize!(Fo))
    t_so = wall(() -> solve!(Fo, xd, bd))
    @printf("%-45s wides = :%-6s refact %7.3f ms  solve %7.3f ms\n", name, other, 1e3 * t_fo, 1e3 * t_so)
    t_f2 = wall(() -> refactorize!(F))
    @printf("%-45s refact (FusedSolver, repeated at the end): %7.3f ms\n", name, 1e3 * t_f2)
    flush(stdout)
    GC.gc(); CUDA.reclaim()
end
say("done")
