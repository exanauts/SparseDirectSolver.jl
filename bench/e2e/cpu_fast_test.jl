# CPU fast-path prototype: gate vs stock refactorization + timing, 78k KKT.
#   julia +1.13 -t <N> --project=bench bench/e2e/cpu_fast_test.jl
using SparseDirectSolver, SparseArrays, LinearAlgebra, Random, Statistics, Printf
const SDS = SparseDirectSolver
include(joinpath(@__DIR__, "SDSCPU.jl"))
using .SDSCPU
const BD = joinpath(@__DIR__, "..")
include(joinpath(BD, "matrices.jl"))
using .BenchMatrices: read_mtx

A = SparseMatrixCSC{Float64, Int}(sparse(read_mtx(joinpath(BD, "data",
    "kkt_pglib_opf_case78484_epigrids_condensed_10.mtx"))))
n = size(A, 1)
Lt = sparse(transpose(tril(A)))
Random.seed!(666)
bh = rand(n)
S3 = (p = Vector{Int32}(undef, n); read!(joinpath(BD, "perm_cand3.bin"), p); Int.(p))

s = DirectSolver(Vector{Int}(Lt.colptr), Vector{Int}(Lt.rowval), Vector{Float64}(Lt.nzval), "SPD", 'L')
s.options.subtree_parallelism = 16384
s.options.subtree_budgets = [16384]
# CPU wants wider supernodes than the GPU champion (48): amal 96 = -17% at 24t,
# interleaved (fewer, bigger fronts = fewer memory passes + BLAS3-friendlier)
s.options.amalgamation = (max_width = parse(Int, get(ENV, "SDSCPU_AMAL", "96")),
                          zero_fraction = 0.25, min_width = 8)
SDS.setparam!(s, "user_perm", S3)
x = zeros(n); b = copy(bh)
ex(ph) = SDS.execute!(ph, s, x, b; asynchronous = false)
ex("analysis"); ex("factorization"); ex("refactorization")
tr0 = median([(@elapsed ex("refactorization")) for _ in 1:3])
fac0 = copy(s.numeric.factor)
nz = SDS._factor_values(s)

ctx = SDSCPU.CPUCtx(s)
info = SDSCPU.refact_cpu!(s, nz, ctx)
df = maximum(abs, s.numeric.factor - fac0) / maximum(abs, fac0)
@printf "threads %d  gate: info %d  Δfactor %.2e\n" Threads.nthreads() info df
(info == 0 && df < 1e-6) || error("CPU fast path gate FAILED")
tr1 = median([(@elapsed SDSCPU.refact_cpu!(s, nz, ctx)) for _ in 1:5])
ex("solve")
rel = norm(bh - A * x) / norm(bh)
@printf "threads %d  stock refact %8.1f ms   fast refact %8.1f ms   speedup %.2fx   relres(stock solve) %.2e\n" Threads.nthreads() tr0*1e3 tr1*1e3 tr0/tr1 rel
println("done")
