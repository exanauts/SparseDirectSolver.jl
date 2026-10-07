# cuDSS vs SparseDirectSolver.jl: per-phase timings for every feature of
# `features.jl` on the harness matrices, measured with BenchmarkTools. One solver
# per process (CUDSS.jl and the CUDA extension define the same `cholesky`
# methods), each run merged into `bench/comparison/<solver>.csv`;
# `compare_report.jl` renders the side-by-side table and plot.
#
#   julia --project=bench bench/compare.jl --solver=cudss [options]
#   julia --project=bench bench/compare.jl --solver=sds [--backend=cuda|cpu] [options]
#
# Options:
#   --features=a,b      feature ids (default: all; see bench/features.jl)
#   --only=m1,m2        matrix names (default: all harness matrices)
#   --samples=5         BenchmarkTools samples per phase (after its warm-up sample)
#   --seconds=60        time budget per phase; sampling stops at whichever limit comes first
#   --single-run-above=5  if one factorization takes longer (seconds), record the
#                       times of a single run instead of a BenchmarkTools trial
#   --no-suitesparse    skip the MatrixDepot matrices
#   --no-dumps          skip the KKT dumps under bench/data/
#   --force             run SDS features whose task is not done in TASKS.md (rows record the error)
#
# Each phase is timed on a fresh solver: the `setup` of the benchmark creates the
# solver and runs the phases it depends on, the timed expression runs the phase
# and synchronizes the device. On a GPU the setup ends by keeping the device busy
# for 0.2 s so that it is at full clock (see `spin_device`). A first run of all
# phases compiles and gives `info`, the statistics and the residual; when the
# factorization of a second run exceeds --single-run-above, the times of that
# run are recorded instead of a trial.

using LinearAlgebra
using SparseArrays
using Random
using Dates
using BenchmarkTools
using DelimitedFiles

include(joinpath(@__DIR__, "matrices.jl"))
include(joinpath(@__DIR__, "features.jl"))
using .BenchMatrices
using .BenchFeatures

function parse_args(args)
    o = Dict{String, String}("solver" => "", "backend" => "cuda", "features" => "", "only" => "",
                             "samples" => "5", "seconds" => "60", "single-run-above" => "5", "suitesparse" => "true", "dumps" => "true",
                             "force" => "false")
    for a in args
        if a in ("--no-suitesparse", "--no-dumps", "--force")
            o[a == "--force" ? "force" : a[6:end]] = a == "--force" ? "true" : "false"
        elseif (m = match(r"^--(solver|backend|features|only|samples|seconds|single-run-above)=(.*)$", a)) !== nothing
            o[m[1]] = m[2]
        else
            error("unknown argument \"$a\"; see the header of bench/compare.jl")
        end
    end
    o["solver"] in ("cudss", "sds") || error("--solver=cudss or --solver=sds is required")
    o["backend"] in ("cuda", "cpu") || error("--backend must be cuda or cpu")
    return o
end

const OPTS = parse_args(ARGS)
const SOLVER = OPTS["solver"]
const ON_GPU = SOLVER == "cudss" || OPTS["backend"] == "cuda"

if SOLVER == "cudss"
    using CUDA, CUDA.cuSPARSE, CUDSS
else
    using SparseDirectSolver, Metis   # Metis: nested dissection in the default ordering
    ON_GPU && using CUDA, CUDA.cuSPARSE
end

dev(x) = ON_GPU ? CUDA.CuArray(x) : x
dev_csr(A) = ON_GPU ? CUDA.cuSPARSE.CuSparseMatrixCSR(A) : A
sync() = ON_GPU ? CUDA.synchronize() : nothing

# The GPU drops to a low-clock power state while the host runs the (untimed)
# setup, and the timed phase then starts slow: on the RTX 4080, lap2d_300
# factorization varied between 6 and 19 ms. Keeping the device busy for 0.2 s
# right before each timed phase gives a stable 5.9 ms (T04 baseline: 6.05 ms).
const SPIN_SECONDS = 0.2
const SPIN_BUFFER = Ref{Any}(nothing)
function spin_device()
    ON_GPU || return
    SPIN_BUFFER[] === nothing && (SPIN_BUFFER[] = CUDA.zeros(Float32, 2^24))
    buf, t0 = SPIN_BUFFER[], time()
    while time() - t0 < SPIN_SECONDS
        buf .= sin.(buf) .+ 1f0
        CUDA.synchronize()
    end
end

# --- the two solvers behind one set of names (the parameter strings mirror cuDSS) ---

# CUDSS.jl has no getter for CUDSS_DATA_FLOPS (an Int64 in cuDSS 0.8, see cudss_baseline.jl)
function cudss_flops(s)
    ref, nw = Ref{Int64}(0), Ref{Csize_t}(0)
    CUDSS.cudssDataGet(s.data.handle, s.data, CUDSS.CUDSS_DATA_FLOPS, ref, 8, nw)
    return nw[] == 8 && ref[] > 0 ? ref[] : missing
end

const API = if SOLVER == "cudss"
    (solver = CUDSS.CudssSolver, batched = CUDSS.CudssBatchedSolver, set = CUDSS.cudss_set,
     get = CUDSS.cudss_get, exec = CUDSS.cudss, flops = cudss_flops,
     # a uniform batch passes x and b as descriptors with nbatch columns
     ubatch_rhs = (X, nb) -> (D = CUDSS.CudssMatrix(eltype(X), size(X, 1); nbatch = nb); CUDSS.cudss_update(D, X); D),
     version = string(CUDSS.version()))
else
    (solver = SparseDirectSolver.DirectSolver,
     batched = (a...) -> getproperty(SparseDirectSolver, :BatchedDirectSolver)(a...),  # T22
     set = SparseDirectSolver.setparam!, get = SparseDirectSolver.getparam,
     exec = (p, s, x, b) -> SparseDirectSolver.execute!(p, s, x, b; asynchronous = false),
     flops = s -> SparseDirectSolver.getparam(s, "flops"),
     ubatch_rhs = (X, nb) -> X,  # assumed (n, nbatch) arrays; revisit when T17 lands
     version = string(pkgversion(SparseDirectSolver)))
end

side_params(f) = merge(f.params, SOLVER == "cudss" ? f.cudss_params : f.sds_params)

relres(A, x, b) = norm(b - A * x) / norm(b)
colmax_relres(A, X, B) = maximum(relres(A, X[:, j], B[:, j]) for j in axes(B, 2))

# --- systems: `fresh()` builds a configured solver with its x and b, untimed ---

"""
    build_system(f, Ms) -> (; fresh, solve_phase, check)

`fresh()` returns `(s, x, b)`: a new solver for feature `f` on the matrices `Ms`
(one matrix, or the batch members for `:nubatch`) with its device right-hand
side and solution. `check(x)` returns the relative residual on the host
(`missing` for the Schur feature).
"""
function build_system(f::Feature, Ms)
    T, view = f.T, f.structure == "G" ? 'F' : 'L'
    stored(A) = f.structure == "G" ? A : tril(A)
    Random.seed!(666)
    if f.kind == :nubatch
        As = [SparseMatrixCSC{T, Int}(M.A) for M in Ms]
        Ads = [dev_csr(stored(A)) for A in As]
        bs = [rand(T, size(A, 1)) for A in As]
        bds = dev.(bs)
        fresh = () -> (s = API.batched(Ads, f.structure, view); configure!(s, f); (s, similar.(bds), bds))
        check = xs -> maximum(relres(As[k], Array(xs[k]), bs[k]) for k in eachindex(As))
        return (; fresh, solve_phase = "solve", check)
    end
    A = SparseMatrixCSC{T, Int}(only(Ms).A)
    n = size(A, 1)
    if f.kind == :ubatch
        # nbatch value sets on one pattern: member k scales the diagonal by 1 + 0.01(k-1) (stays SPD)
        Lt = sparse(transpose(stored(A)))  # CSC of the transpose = CSR arrays of the stored triangle
        ondiag = [Lt.rowval[p] == j for j in 1:n for p in nzrange(Lt, j)]
        scale(k) = 1 .+ T(0.01) * (k - 1) .* ondiag
        nzval = reduce(hcat, Lt.nzval .* scale(k) for k in 1:f.nbatch)
        rowptr, colval, nzd = dev(Cint.(Lt.colptr)), dev(Cint.(Lt.rowval)), dev(nzval)
        B = rand(T, n, f.nbatch)
        Bd = dev(B)
        fresh = () -> begin
            s = API.solver(rowptr, colval, nzd, f.structure, view)
            API.set(s, "ubatch_size", f.nbatch)
            configure!(s, f)
            Xd = similar(Bd)
            (s, API.ubatch_rhs(Xd, f.nbatch), API.ubatch_rhs(Bd, f.nbatch), Xd)
        end
        Ak(k) = A + Diagonal(T(0.01) * (k - 1) .* diag(A))
        check = X -> maximum(relres(Ak(k), Array(X)[:, k], B[:, k]) for k in 1:f.nbatch)
        return (; fresh, solve_phase = "solve", check)
    end
    Ad = dev_csr(stored(A))
    B = f.nrhs == 1 ? rand(T, n) : rand(T, n, f.nrhs)
    Bd = dev(B)
    fresh = () -> begin
        s = API.solver(Ad, f.structure, view)
        if f.kind == :schur  # Schur block: the last min(64, n ÷ 10) rows and columns
            API.set(s, "user_schur_indices", Cint[i > n - min(64, n ÷ 10) for i in 1:n])
        end
        configure!(s, f)
        (s, similar(Bd), Bd)
    end
    check = f.kind == :schur ? (X -> missing) : (X -> colmax_relres(A, reshape(Array(X), n, :), reshape(B, n, :)))
    return (; fresh, solve_phase = f.kind == :schur ? "solve_fwd_schur" : "solve", check)
end

configure!(s, f) = foreach(((k, v),) -> API.set(s, k, v), side_params(f))

# the solution array `check` reads: the plain device array (index 4) for a uniform batch
solution(st) = length(st) == 4 ? st[4] : st[2]

# --- timing ------------------------------------------------------------------

const PHASES = ("analysis", "factorization", "refactorization", "solve")
const BEFORE = Dict("analysis" => (), "factorization" => ("analysis",),
                    "refactorization" => ("analysis", "factorization"), "solve" => ("analysis", "factorization"))

phase_name(sys, p) = p == "solve" ? sys.solve_phase : p
run_phase(sys, st, p) = API.exec(phase_name(sys, p), st[1], st[2], st[3])

"""
    time_phase(sys, phase; samples, seconds) -> (median_s, min_s)

BenchmarkTools trial of one phase; the setup builds a fresh solver and runs the
phases `phase` depends on.
"""
function time_phase(sys, phase; samples, seconds)
    prep = () -> (st = sys.fresh(); foreach(p -> run_phase(sys, st, p), BEFORE[phase]); sync(); spin_device(); st)
    core = st -> (run_phase(sys, st, phase); sync())
    b = @benchmarkable $core(st) setup = (st = $prep()) evals = 1
    t = run(b; samples, seconds, gcsample = true)
    return median(t).time / 1e9, minimum(t).time / 1e9
end

function stat(f, s)
    try
        f(s)
    catch
        missing
    end
end

"""
    measure(sys; samples, seconds, single_run_above) -> NamedTuple

A first run of all four phases compiles and gives `info`, the statistics and
the residual. A second run is timed phase by phase; if its factorization took
longer than `single_run_above` seconds, its times are recorded (`samples = 1`),
otherwise each phase gets a BenchmarkTools trial (median and minimum).
"""
function measure(sys; samples, seconds, single_run_above)
    st = sys.fresh()
    foreach(p -> run_phase(sys, st, p), PHASES)
    sync()
    info = API.get(st[1], "info")
    all(iszero, info) || error("info = $info after the factorization")
    stats = (relres = sys.check(solution(st)), lu_nnz = stat(s -> API.get(s, "lu_nnz"), st[1]),
             flops = stat(API.flops, st[1]), nsuperpanels = stat(s -> API.get(s, "nsuperpanels"), st[1]))
    st = sys.fresh()
    once = Dict{String, Float64}()
    for p in PHASES
        p == "analysis" || spin_device()
        t0 = time_ns()
        run_phase(sys, st, p)
        sync()
        once[p] = (time_ns() - t0) / 1e9
    end
    single = once["factorization"] > single_run_above
    times = single ? Dict(p => (once[p], once[p]) for p in PHASES) :
            Dict(p => time_phase(sys, p; samples, seconds) for p in PHASES)
    return (; times, samples = single ? 1 : samples, stats...)
end

# --- CSV ---------------------------------------------------------------------

const COLUMNS = ["solver", "feature", "task", "matrix", "structure", "n", "nnz", "T", "nrhs", "nbatch",
                 "analysis_s", "factorization_s", "refactorization_s", "solve_s",
                 "analysis_min_s", "factorization_min_s", "refactorization_min_s", "solve_min_s",
                 "lu_nnz", "flops", "nsuperpanels", "relres", "samples", "status", "device", "version", "git_sha", "date"]

cell(x) = x === missing || x === nothing ? "" : x isa AbstractFloat ? repr(Float64(x)) : string(x)
clean(msg) = first(strip(replace(msg, r"[,\"\s]+" => " ")), 200)

function device_name()
    ON_GPU && return replace(CUDA.name(CUDA.device()), "," => " ")
    return "CPU (KernelAbstractions, $(Threads.nthreads()) threads)"
end

function git_sha()
    try
        sha = readchomp(`git -C $(@__DIR__) rev-parse --short HEAD`)
        dirty = !isempty(readchomp(`git -C $(@__DIR__) status --porcelain --untracked-files=no`))
        return dirty ? sha * "+dirty" : sha
    catch
        return ""
    end
end

function csv_row(f, name, Ms, result, meta)
    n, nz = sum(M -> size(M.A, 1), Ms), sum(M -> nnz(M.A), Ms)
    head = [SOLVER, f.id, f.task, name, f.structure, string(n), string(nz), string(f.T), string(f.nrhs),
            string(f.kind == :nubatch ? length(Ms) : f.nbatch)]
    if result isa Exception
        body = vcat(fill("", 13), clean(sprint(showerror, result)))
    else
        body = vcat([cell(result.times[p][1]) for p in PHASES], [cell(result.times[p][2]) for p in PHASES],
                    [cell(x) for x in (result.lu_nnz, result.flops, result.nsuperpanels, result.relres, result.samples)], "ok")
    end
    return vcat(head, body, meta)
end

"""
    merge_csv(path, rows)

Writes `rows` to `path`, keeping the rows of an existing file whose
(feature, matrix) is not among `rows`; sorted by feature order, then matrix.
"""
function merge_csv(path, rows)
    old = Vector{Vector{String}}()
    if isfile(path)
        data, header = readdlm(path, ',', String; header = true, quotes = false)
        if vec(header) == COLUMNS
            rerun = Set((r[2], r[4]) for r in rows)
            old = [data[i, :] for i in axes(data, 1) if (data[i, 2], data[i, 4]) ∉ rerun]
        else
            @warn "$path has other columns; replacing it"
        end
    end
    order = Dict(f.id => i for (i, f) in enumerate(COMPARE_FEATURES))
    all_rows = sort!(vcat(old, rows); by = r -> (get(order, r[2], typemax(Int)), r[4]))
    mkpath(dirname(path))
    open(path, "w") do io
        writedlm(io, [permutedims(COLUMNS); permutedims(reduce(hcat, all_rows))], ',')
    end
end

# --- driver ------------------------------------------------------------------

function main()
    ON_GPU && !CUDA.functional() && error("CUDA is not functional on this machine")
    samples, seconds = parse(Int, OPTS["samples"]), parse(Float64, OPTS["seconds"])
    single_run_above = parse(Float64, OPTS["single-run-above"])
    ids = filter(!isempty, split(OPTS["features"], ','))
    features = isempty(ids) ? COMPARE_FEATURES : feature.(ids)
    mats = bench_matrices(; suitesparse = OPTS["suitesparse"] == "true", dumps = OPTS["dumps"] == "true")
    only_names = filter(!isempty, split(OPTS["only"], ','))
    isempty(only_names) || filter!(M -> M.name in only_names, mats)
    meta = [device_name(), API.version, git_sha(), string(today())]
    out = joinpath(@__DIR__, "comparison", "$SOLVER.csv")
    println(SOLVER, " ", API.version, " on ", meta[1], ", samples = ", samples)

    rows = Vector{Vector{String}}()
    for f in features
        if SOLVER == "cudss" && !f.cudss_supported
            println(rpad(f.id, 18), "no cuDSS counterpart, skipped")
            continue
        elseif SOLVER == "sds" && !feature_implemented(f) && OPTS["force"] != "true"
            println(rpad(f.id, 18), "pending ", f.task, ", skipped")
            continue
        end
        selected = filter(M -> f.structure in M.structures && f.matrices(M), mats)
        groups = f.kind == :nubatch ? (isempty(selected) ? [] : ["batch_of_$(length(selected))" => selected]) :
                 [M.name => [M] for M in selected]
        for (name, Ms) in groups
            print(rpad(f.id, 18), rpad(name, 42))
            flush(stdout)
            result = try
                Base.invokelatest(measure, build_system(f, Ms); samples, seconds, single_run_above)
            catch err
                err isa InterruptException && rethrow()
                err
            end
            if result isa Exception
                println("FAILED: ", first(sprint(showerror, result), 160))
            else
                println(join(("$p $(round(result.times[p][1] * 1e3; sigdigits = 3)) ms" for p in PHASES), ", "),
                        ", relres ", result.relres === missing ? "-" : round(result.relres; sigdigits = 2),
                        result.samples == 1 ? " (single run)" : "")
            end
            push!(rows, csv_row(f, name, Ms, result, meta))
            merge_csv(out, rows)  # after every row, so an interrupted run keeps what it measured
        end
    end
    println(isempty(rows) ? "nothing to run" : "wrote $out")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
