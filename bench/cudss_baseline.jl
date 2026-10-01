# cuDSS baseline: analysis / factorization / refactorization / solve (nrhs = 1)
# median times over 5 runs after a warm-up, plus lu_nnz, flops, nsuperpanels,
# for every benchmark matrix and applicable structure.
#
#   julia --project=bench bench/cudss_baseline.jl [options]
#
# Options:
#   --solver=cudss|cholmod   cudss (default, needs a functional CUDA GPU) or the
#                            CPU reference (CHOLMOD/UMFPACK from SparseArrays;
#                            skips "S" where "SPD" applies, CHOLMOD ldlt is simplicial)
#   --nruns=N                timed runs per phase (default 5), after one warm-up
#   --only=a,b               restrict to these matrix names
#   --no-suitesparse         skip the MatrixDepot downloads
#   --no-dumps               skip the KKT dumps under bench/data/
#   --out=PATH               output CSV (default bench/results/<solver>_baseline.csv)

using LinearAlgebra
using SparseArrays
using Random

include(joinpath(@__DIR__, "matrices.jl"))
include(joinpath(@__DIR__, "harness.jl"))
using .BenchMatrices
using .BenchHarness

function parse_args(args)
    opts = Dict{String,String}("solver" => "cudss", "nruns" => "5", "only" => "",
                               "suitesparse" => "true", "dumps" => "true", "out" => "")
    for a in args
        if a == "--no-suitesparse"
            opts["suitesparse"] = "false"
        elseif a == "--no-dumps"
            opts["dumps"] = "false"
        elseif (m = match(r"^--(solver|nruns|only|out)=(.*)$", a)) !== nothing
            opts[m[1]] = m[2]
        else
            error("unknown argument \"$a\"; see the header of bench/cudss_baseline.jl")
        end
    end
    return opts
end

# --- cuDSS through CUDSS.jl --------------------------------------------------

# Loaded at top level (not from inside `main`) so that the bindings are visible
# to the functions below without world-age workarounds; skipped for the CPU reference.
if parse_args(ARGS)["solver"] == "cudss"
    using CUDA, CUDA.cuSPARSE, CUDSS
end

# CUDSS.jl has no getter for CUDSS_DATA_FLOPS; read it through the C API. The
# header does not document its type; cuDSS 0.8.0 writes an int64 (checked on an
# RTX 4080: lap2d_300 SPD gives 3.18e8 with lu_nnz = 2.43e6, consistent with
# Σ colcount² = 5.18e8 of CHOLMOD's 4.12e6-entry factor; read as a double the
# same bits are a denormal). `missing` if this cuDSS does not provide it or the
# value is not positive; the raw bits are printed once in that case.
const FLOPS_WARNED = Ref(false)
function cudss_flops(solver)
    try
        ref = Ref{Int64}(0)
        nw = Ref{Csize_t}(0)
        CUDSS.cudssDataGet(solver.data.handle, solver.data, CUDSS.CUDSS_DATA_FLOPS, ref, 8, nw)
        nw[] == 8 && ref[] > 0 && return ref[]
        if !FLOPS_WARNED[]
            FLOPS_WARNED[] = true
            @warn "CUDSS_DATA_FLOPS is not a positive Int64; recording missing" written = Int(nw[]) as_int64 = ref[] as_float64 = reinterpret(Float64, ref[])
        end
        return missing
    catch
        return missing
    end
end

function cudss_solver(structure)
    # Symmetric structures pass the lower triangle (view 'L'), as in the CUDSS.jl docs.
    view = structure == "G" ? 'F' : 'L'
    return function (A, b)
        Ad = CUDA.cuSPARSE.CuSparseMatrixCSR(structure == "G" ? A : tril(A))
        bd = CUDA.CuVector(b)
        xd = similar(bd)
        solver = CUDSS.CudssSolver(Ad, structure, view)
        check() = (info = CUDSS.cudss_get(solver, "info"); info == 0 || error("cuDSS info = $info"))
        return (analysis = () -> CUDSS.cudss("analysis", solver, xd, bd),
                factorization = () -> CUDSS.cudss("factorization", solver, xd, bd),
                refactorization = () -> CUDSS.cudss("refactorization", solver, xd, bd),
                solve = () -> (CUDSS.cudss("solve", solver, xd, bd); xd),
                stats = () -> (check();
                               (lu_nnz = CUDSS.cudss_get(solver, "lu_nnz"),
                                flops = cudss_flops(solver),
                                nsuperpanels = CUDSS.cudss_get(solver, "nsuperpanels"))))
    end
end

# --- driver ------------------------------------------------------------------

function main(args = ARGS)
    opts = parse_args(args)
    solvername = opts["solver"]
    nruns = parse(Int, opts["nruns"])
    if solvername == "cudss"
        CUDA.functional() ||
            error("CUDA is not functional on this machine; use --solver=cholmod for the CPU reference")
        make = cudss_solver
        sync = () -> CUDA.synchronize()
        println("cuDSS baseline on ", CUDA.name(CUDA.device()))
    elseif solvername == "cholmod"
        make = cholmod_solver
        sync = () -> nothing
        println("CPU reference baseline (CHOLMOD/UMFPACK), BLAS threads = ", BLAS.get_num_threads())
    else
        error("--solver must be cudss or cholmod, got \"$solvername\"")
    end
    out = isempty(opts["out"]) ? joinpath(@__DIR__, "results", "$(solvername)_baseline.csv") : opts["out"]

    mats = bench_matrices(; suitesparse = opts["suitesparse"] == "true", dumps = opts["dumps"] == "true")
    only = filter(!isempty, split(opts["only"], ','))
    isempty(only) || filter!(M -> M.name in only, mats)

    rows = Vector{Vector{String}}()
    for M in mats, structure in M.structures
        # CHOLMOD's ldlt is simplicial (no supernodes): on SPD matrices it only measures
        # that, and takes minutes to hours on the large ones. The CPU reference runs "S"
        # only where it is the sole symmetric structure (K2 dumps).
        solvername == "cholmod" && structure == "S" && "SPD" in M.structures && continue
        Random.seed!(666)
        b = rand(size(M.A, 1))
        print(rpad(M.name, 44), rpad(structure, 4), " n = ", rpad(size(M.A, 1), 8), " ")
        flush(stdout)
        result = try
            Base.invokelatest(time_phases, make(structure), M.A, b; nruns = nruns, synchronize = sync)
        catch err
            err isa InterruptException && rethrow()
            err
        end
        if result isa Exception
            println("FAILED: ", sprint(showerror, result))
        else
            println("analysis ", round(result.analysis; sigdigits = 3), " s, factorization ",
                    round(result.factorization; sigdigits = 3), " s, refactorization ",
                    round(result.refactorization; sigdigits = 3), " s, solve ",
                    round(result.solve; sigdigits = 3), " s, relres ", round(result.relres; sigdigits = 2))
        end
        flush(stdout)
        push!(rows, csv_row(solvername, M.name, M.A, structure, result))
    end
    write_csv(out, rows)
    println("wrote ", out)
    return out
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
