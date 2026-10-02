# Numeric factorization time per regime mix (TASKS.md T11): `factorize!` on the
# generated T04 matrices with regimes A+B+C (default analysis), B+C
# (`subtree_budgets = []`) and C only (`subtree_budgets = []`,
# `factorization_alg = "algo2"`). Prints a Markdown table: fronts per regime,
# launches, median factorization time.
#
# Usage (package environment, like bench/front_bins.jl):
#
#   julia --project=. bench/regimes.jl [options]
#
#   --backend=cpu|cuda   KA CPU backend (default) or CUDA; CUDA must be installed in an environment on
#                        the load path (e.g. `julia --project=bench`, or a stacked environment)
#   --T=Float64          element type (Float32, Float64, ComplexF32, ComplexF64)
#   --nruns=5            timed runs per measurement (median), after one warm-up
#   --impl=auto          dense implementation of the regime-C path (auto, vendor, generic, ka)
#   --only=a,b           matrices to run (default: lap2d_300,lap3d_40); also lap2d_<k>, lap3d_<k> for a k-grid

using LinearAlgebra
using SparseArrays
using Statistics: median
using KernelAbstractions
using SparseDirectSolver
const SDS = SparseDirectSolver

include(joinpath(@__DIR__, "matrices.jl"))

function parse_args(args)
    o = Dict{String, String}("backend" => "cpu", "T" => "Float64", "nruns" => "5", "impl" => "auto",
                             "only" => "lap2d_300,lap3d_40")
    for a in args
        m = match(r"^--(backend|T|nruns|impl|only)=(.*)$", a)
        m === nothing && error("unknown argument \"$a\"; see the header of bench/regimes.jl")
        o[m[1]] = m[2]
    end
    return o
end

const OPTS = parse_args(ARGS)

if OPTS["backend"] == "cuda"
    @eval using CUDA
    const BACKEND = CUDABackend()
    device_sync() = CUDA.synchronize()
    to_dev(x) = CuArray(x)
elseif OPTS["backend"] == "cpu"
    const BACKEND = CPU()
    device_sync() = nothing
    to_dev(x) = copy(x)
else
    error("unknown backend $(OPTS["backend"])")
end

function bench_matrix(name)
    m = match(r"^lap([23])d_(\d+)$", name)
    m === nothing && error("unknown matrix $name (expected lap2d_<k> or lap3d_<k>)")
    k = parse(Int, m[2])
    return m[1] == "2" ? BenchMatrices.laplacian2d(k, k) : BenchMatrices.laplacian3d(k, k, k)
end

const MIXES = (("A+B+C", Options()), ("B+C", Options(subtree_budgets = Int[])),
               ("C only", Options(subtree_budgets = Int[], factorization_alg = "algo2")))

function main()
    T = eval(Symbol(OPTS["T"]))
    nruns = parse(Int, OPTS["nruns"])
    impl = Symbol(OPTS["impl"])
    println("| matrix | regimes | fronts A/B/C | subtrees | launches | factorize! (ms) | max rel. diff to A+B+C |")
    println("| --- | --- | --- | --- | --- | --- | --- |")
    for name in split(OPTS["only"], ',')
        A = SparseMatrixCSC{T, Int}(bench_matrix(String(name)))
        C = SDS.CSR(tril(A))
        nz = to_dev(C.nzval)
        Fref = nothing
        for (mix, opts) in MIXES
            S = SDS.symbolic_analysis(C, T <: Complex ? "HPD" : "SPD", 'L'; opts, T)
            Sd = SDS.adapt(BACKEND, S, Int32)
            N = SDS.allocate_numeric(Sd, T, BACKEND)
            SDS.factorize!(N, Sd, nz; impl) == 0 || error("factorization failed on $name ($mix)")
            device_sync()
            t = Float64[]
            for _ in 1:nruns
                t0 = time_ns()
                SDS.factorize!(N, Sd, nz; impl)            # ends with the info read: synchronized
                push!(t, (time_ns() - t0) / 1e6)
            end
            F = Array(N.factor)
            Fref === nothing && (Fref = F)
            sc = S.schedule
            nr = map(r -> count(==(r), sc.regime), (SDS.REGIME_A, SDS.REGIME_B, SDS.REGIME_C))
            println("| $name | $mix | $(join(nr, "/")) | $(SDS.nsubtrees(sc)) | $(SDS.nlaunches(sc)) | ",
                    round(median(t); digits = 2), " | ", round(maximum(abs, F - Fref) / maximum(abs, Fref); sigdigits = 2),
                    " |")
        end
    end
end

main()
