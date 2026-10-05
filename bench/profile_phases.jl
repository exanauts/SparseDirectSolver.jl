# Experiment 0 of PERFORMANCE.md: where SDS spends refactorization and solve
# time on CUDA. For every harness matrix and structure, one warm
# refactorization and one warm solve run under `CUDA.@profile` (CUPTI); the
# script reports the launch plan of the schedule (regime-A subtrees, regime-B
# and regime-C fronts, launch groups), the wall time, the number of kernels and
# copies, the GPU busy time (sum of kernel durations), the share of the longest
# kernel and its grid, and writes `bench/profile/phase_split.{md,csv}`.
#
#   julia --project=bench bench/profile_phases.jl [--only=m1,m2] [--structures=SPD,S]
#                                                 [--no-suitesparse] [--no-dumps]
#
# The SDS phases are run with `asynchronous = false`, as in bench/compare.jl. The
# wall time is the best of 5 warm runs, each after 0.2 s of device load so that
# the GPU is at full clock; the profiled run is a sixth one.
# LDLᵀ is skipped on matrices whose LDLᵀ factorization took more than 10 s in
# bench/comparison/sds.csv (GHS_psdef/apache2, lap3d_40): issue #75.

using LinearAlgebra
using SparseArrays
using Random
using Printf
using Dates
using SparseDirectSolver, Metis
using CUDA, CUDA.cuSPARSE

include(joinpath(@__DIR__, "matrices.jl"))
using .BenchMatrices

function parse_args(args)
    o = Dict("only" => "", "structures" => "SPD,S", "suitesparse" => "true", "dumps" => "true")
    for a in args
        if a in ("--no-suitesparse", "--no-dumps")
            o[a[6:end]] = "false"
        elseif (m = match(r"^--(only|structures)=(.*)$", a)) !== nothing
            o[m[1]] = m[2]
        else
            error("unknown argument \"$a\"; see the header of bench/profile_phases.jl")
        end
    end
    return o
end

const OPTS = parse_args(ARGS)
const SLOW_LDLT = ("GHS_psdef/apache2",)
const OUTDIR = joinpath(@__DIR__, "profile")

# kernel name without the KernelAbstractions/CUDA template arguments
function short_name(name::AbstractString)
    m = match(r"^(?:void\s+)?(?:gpu_)?([A-Za-z0-9_]+?)_?(?:\(|<|$)", name)
    return m === nothing ? first(name, 40) : m[1]
end

# keep the GPU at full clock before each timed phase (see `spin_device` in bench/compare.jl)
const SPIN_BUFFER = Ref{Any}(nothing)
function spin_device(seconds = 0.2)
    SPIN_BUFFER[] === nothing && (SPIN_BUFFER[] = CUDA.zeros(Float32, 2^24))
    buf, t0 = SPIN_BUFFER[], time()
    while time() - t0 < seconds
        buf .= sin.(buf) .+ 1f0
        CUDA.synchronize()
    end
end

"""
    profile_phase(f) -> NamedTuple

Run `f()` (one phase, ending in a device synchronization) under `CUDA.@profile`
and summarize the device trace: wall time, kernels, copies, GPU busy time and
the longest kernel.
"""
function profile_phase(f)
    wall = Inf
    for _ in 1:5                               # wall time: best of 5 warm runs at full clock
        spin_device()
        t0 = time_ns()
        f()
        wall = min(wall, (time_ns() - t0) / 1e6)
    end
    spin_device()
    r = CUDA.@profile trace = true f()
    d = r.device
    iskernel = [d.grid[i] !== missing for i in eachindex(d.id)]
    dur = [(d.stop[i] - d.start[i]) * 1e3 for i in eachindex(d.id)]
    kidx = findall(iskernel)
    busy = sum(dur[kidx]; init = 0.0)
    span = isempty(d.id) ? 0.0 : (maximum(d.stop) - minimum(d.start)) * 1e3
    top = isempty(kidx) ? 0 : kidx[argmax(dur[kidx])]
    byname = Dict{String, Tuple{Int, Float64}}()
    for i in kidx
        k = short_name(d.name[i])
        c, t = get(byname, k, (0, 0.0))
        byname[k] = (c + 1, t + dur[i])
    end
    return (; wall, kernels = length(kidx), copies = count(!, iskernel), busy, span,
            top_ms = top == 0 ? 0.0 : dur[top], top_name = top == 0 ? "" : short_name(d.name[top]),
            top_blocks = top == 0 ? 0 : Int(d.grid[top].x * d.grid[top].y * d.grid[top].z),
            top_threads = top == 0 ? 0 : Int(d.block[top].x * d.block[top].y * d.block[top].z),
            byname)
end

function plan_summary(s)
    sc = s.symbolic.schedule
    P = s.numeric.plan
    reg = Array(sc.regime)
    A, B, C = SparseDirectSolver.REGIME_A, SparseDirectSolver.REGIME_B, SparseDirectSolver.REGIME_C
    nsub = length(sc.subtree_root)
    sizes = nsub == 0 ? Int[] : diff(Array(sc.subtree_ptr))
    return (; nsuper = length(reg), a_fronts = count(==(A), reg), subtrees = nsub,
            largest_subtree = isempty(sizes) ? 0 : maximum(sizes),
            b_fronts = count(==(B), reg), c_fronts = count(==(C), reg),
            groups = length(P.sub_first) + length(P.group_first), nlevels = maximum(Array(sc.level); init = 0))
end

function run_case(M::BenchMatrix, structure::String)
    A = M.A
    n = size(A, 1)
    Random.seed!(666)
    Ad = CuSparseMatrixCSR(tril(A))
    b = CuArray(rand(n))
    x = similar(b)
    s = SparseDirectSolver.DirectSolver(Ad, structure, 'L')
    ex(p) = SparseDirectSolver.execute!(p, s, x, b; asynchronous = false)
    ta = @elapsed ex("analysis")
    ex("factorization")
    info = SparseDirectSolver.getparam(s, "info")
    for _ in 1:2
        ex("refactorization")
        ex("solve")
    end
    CUDA.synchronize()
    refac = profile_phase(() -> (ex("refactorization"); CUDA.synchronize()))
    solve = profile_phase(() -> (ex("solve"); CUDA.synchronize()))
    relres = norm(Array(b) - A * Array(x)) / norm(Array(b))
    return (; matrix = M.name, structure, n, nnz = nnz(A), info, analysis_ms = ta * 1e3,
            lnnz = SparseDirectSolver.getparam(s, "lu_nnz"), relres, plan_summary(s)..., refac, solve)
end

function main()
    only = isempty(OPTS["only"]) ? nothing : Set(split(OPTS["only"], ','))
    structures = split(OPTS["structures"], ',')
    mats = bench_matrices(; suitesparse = OPTS["suitesparse"] == "true", dumps = OPTS["dumps"] == "true")
    only === nothing || filter!(M -> M.name in only, mats)
    rows = []
    for M in mats, st in structures
        st in M.structures || continue
        st == "S" && M.name in SLOW_LDLT && continue
        print(rpad("$(M.name) $st", 52))
        r = try
            run_case(M, String(st))
        catch err
            println("ERROR ", sprint(showerror, err)[1:min(end, 200)])
            continue
        end
        @printf("refac %8.2f ms (%4d kernels, top %5.1f%% %s on %d blocks)  solve %8.2f ms (%4d kernels)\n",
                r.refac.wall, r.refac.kernels, 100 * r.refac.top_ms / max(r.refac.busy, eps()),
                r.refac.top_name, r.refac.top_blocks, r.solve.wall, r.solve.kernels)
        push!(rows, r)
    end
    mkpath(OUTDIR)
    write_csv(joinpath(OUTDIR, "phase_split.csv"), rows)
    write_md(joinpath(OUTDIR, "phase_split.md"), rows)
    println("wrote ", joinpath(OUTDIR, "phase_split.{md,csv}"))
end

f1(x) = @sprintf("%.3g", x)

function write_csv(path, rows)
    cols = ["matrix", "structure", "n", "nnz", "lnnz", "nsuper", "a_fronts", "subtrees", "largest_subtree",
            "b_fronts", "c_fronts", "groups", "nlevels", "refac_wall_ms", "refac_kernels", "refac_copies",
            "refac_busy_ms", "refac_top_ms", "refac_top_name", "refac_top_blocks", "solve_wall_ms",
            "solve_kernels", "solve_copies", "solve_busy_ms", "solve_top_ms", "solve_top_name", "relres"]
    open(path, "w") do io
        println(io, join(cols, ','))
        for r in rows
            v = [r.matrix, r.structure, r.n, r.nnz, r.lnnz, r.nsuper, r.a_fronts, r.subtrees, r.largest_subtree,
                 r.b_fronts, r.c_fronts, r.groups, r.nlevels, r.refac.wall, r.refac.kernels, r.refac.copies,
                 r.refac.busy, r.refac.top_ms, r.refac.top_name, r.refac.top_blocks, r.solve.wall,
                 r.solve.kernels, r.solve.copies, r.solve.busy, r.solve.top_ms, r.solve.top_name, r.relres]
            println(io, join(string.(v), ','))
        end
    end
end

function write_md(path, rows)
    open(path, "w") do io
        println(io, "# SDS phase split on CUDA (PERFORMANCE.md, experiment 0)\n")
        println(io, "Generated by `bench/profile_phases.jl` on ", CUDA.name(CUDA.device()), ", ", Dates.today(),
                ". Wall time: best of 5 warm runs at full clock; a sixth run per phase under `CUDA.@profile` gives the kernels. *busy* is the sum of kernel durations; ",
                "*top* is the longest kernel, its share of *busy* and its number of thread blocks.\n")
        println(io, "## Schedule\n")
        println(io, "| matrix | structure | n | nnz(L) | supernodes | A fronts | A subtrees | largest subtree | B fronts | C fronts | launch groups | levels |")
        println(io, "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for r in rows
            println(io, "| ", join([r.matrix, r.structure, r.n, f1(r.lnnz), r.nsuper, r.a_fronts, r.subtrees,
                                    r.largest_subtree, r.b_fronts, r.c_fronts, r.groups, r.nlevels], " | "), " |")
        end
        for (title, key) in (("Refactorization", :refac), ("Solve", :solve))
            println(io, "\n## ", title, "\n")
            println(io, "| matrix | structure | wall ms | kernels | copies | busy ms | top kernel | top ms | top share | top blocks |")
            println(io, "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
            for r in rows
                p = getfield(r, key)
                println(io, "| ", join([r.matrix, r.structure, f1(p.wall), p.kernels, p.copies, f1(p.busy),
                                        "`" * p.top_name * "`", f1(p.top_ms),
                                        @sprintf("%.0f%%", 100 * p.top_ms / max(p.busy, eps())),
                                        p.top_blocks], " | "), " |")
            end
        end
    end
end

main()
