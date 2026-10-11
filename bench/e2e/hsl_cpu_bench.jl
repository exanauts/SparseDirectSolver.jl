# CPU state-of-the-art baseline on the 78k KKT: MA27, MA57 (serial), MA97 (OpenMP),
# via the registered MadNLPHSL wrappers (default options = each solver's own ordering).
#   julia +1 --startup-file=no --project=~/.julia/environments/hslbench bench/e2e/hsl_cpu_bench.jl
using SparseArrays, LinearAlgebra, Random, Statistics, Printf
using MadNLPHSL, HSL
import MadNLP: factorize!, solve!
const BD = joinpath(@__DIR__, "..")
include(joinpath(BD, "matrices.jl"))
using .BenchMatrices: read_mtx

A = SparseMatrixCSC{Float64, Int}(sparse(read_mtx(joinpath(BD, "data",
    "kkt_pglib_opf_case78484_epigrids_condensed_10.mtx"))))
n = size(A, 1)
L = tril(A)
csc32 = SparseMatrixCSC{Float64, Int32}(Int32(n), Int32(n), Int32.(L.colptr), Int32.(L.rowval), L.nzval)
Random.seed!(666)
bh = rand(n)
say(x...) = (println(x...); flush(stdout))

function bench(label, M)
    tf = @elapsed factorize!(M)
    tr = median([(@elapsed factorize!(M)) for _ in 1:5])
    x = copy(bh); solve!(M, x)
    ts = median([begin x .= bh; @elapsed solve!(M, x) end for _ in 1:5])
    x .= bh; solve!(M, x)
    rel = norm(bh - A * x) / norm(bh)
    @printf "%-12s first %8.1f ms  refact %8.1f ms  solve %6.1f ms  relres %.2e\n" label tf*1e3 tr*1e3 ts*1e3 rel
    flush(stdout)
end

try
say("MA27 (serial)…")
ta = @elapsed M27 = MadNLPHSL.Ma27Solver(csc32; opt = MadNLPHSL.Ma27Options(ma27_liw_init_factor = 15.0, ma27_la_init_factor = 30.0, ma27_meminc_factor = 3.0))
say("  init ", round(ta; digits = 2), " s")
bench("MA27", M27)
catch e
    say("MA27 FAILED: ", typeof(e))
end

try
say("MA57 (serial + BLAS)…")
ta = @elapsed M57 = MadNLPHSL.Ma57Solver(csc32)
say("  init ", round(ta; digits = 2), " s")
bench("MA57", M57)
catch e
    say("MA57 FAILED: ", typeof(e))
end

say("MA97 (OpenMP)…")
ta = @elapsed M97 = MadNLPHSL.Ma97Solver(csc32)
say("  init ", round(ta; digits = 2), " s")
for nt in (1, 8, 24)
    HSL.omp_set_num_threads(Cint(nt))
    bench("MA97 nt=$nt", M97)
end
say("done")
