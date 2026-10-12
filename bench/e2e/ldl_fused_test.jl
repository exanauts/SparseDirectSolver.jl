# Fused no-pivot LDL vs stock LDL (pivot 'N') on the 78k ACOPF condensed KKT, plus
# the fused-Cholesky control in the same process.
#   CUDA_VISIBLE_DEVICES=1 julia +1.13 --project=bench bench/e2e/ldl_fused_test.jl
using SparseDirectSolver, SparseArrays, LinearAlgebra, Random, Statistics, Printf
using CUDA, CUDA.CUSPARSE
const SDS = SparseDirectSolver
const SDSFused = SparseDirectSolver.Fused

const BD = joinpath(@__DIR__, "..")
include(joinpath(BD, "matrices.jl"))
using .BenchMatrices: read_mtx

A = SparseMatrixCSC{Float64, Int}(sparse(read_mtx(joinpath(BD, "data",
    "kkt_pglib_opf_case78484_epigrids_condensed_10.mtx"))))
n = size(A, 1)
Lh = tril(A)
Random.seed!(666)
bh = rand(n)
S3 = (p = Vector{Int32}(undef, n); read!(joinpath(BD, "perm_cand3.bin"), p); Int.(p))

function setup(structure)
    bd = CuArray(bh); xd = similar(bd)
    s = DirectSolver(CuSparseMatrixCSR(Lh), structure, 'L')
    SDS.setparam!(s, "user_perm", S3)
    structure == "S" && SDS.setparam!(s, "pivot_type", 'N')
    ex(ph) = SDS.execute!(ph, s, xd, bd; asynchronous = false)
    ex("analysis"); ex("factorization"); ex("refactorization"); ex("solve")
    CUDA.synchronize()
    return s, ex, bd, xd
end

say(x...) = (println(x...); flush(stdout))

# --- stock LDL pivot-N: reference factor/d + timing ---
s, ex, bd, xd = setup("S")
tr_stock = median([CUDA.@elapsed ex("refactorization") for _ in 1:7])
fac0 = Array(s.numeric.factor)
d0 = Array(s.numeric.d)
@printf "stock  S/N   refact %8.2f ms  info %d\n" tr_stock*1e3 SDS.getparam(s, "info")
flush(stdout)

# --- fused LDL on the same solver ---
nzval = SDS._factor_values(s)
fp = SDSFused.build_fused_plan(s; ldl = true, wgf = 128, scap = 128)
info = SDSFused.refact_fused!(s, nzval, fp)
CUDA.synchronize()
fac1 = Array(s.numeric.factor)
d1 = Array(s.numeric.d)
df = maximum(abs, fac1 - fac0) / maximum(abs, fac0)
dd = maximum(abs, d1 - d0) / maximum(abs, d0)
@printf "fused  S/N   info %d  Δfactor %.2e  Δd %.2e\n" info df dd
(info == 0 && df < 1e-6 && dd < 1e-6) || error("fused LDL gate FAILED")
tr_fused = median([CUDA.@elapsed SDSFused.refact_fused!(s, nzval, fp) for _ in 1:7])
@printf "fused  S/N   refact %8.2f ms\n" tr_fused*1e3
flush(stdout)

# --- fused solve (inverse path + D pass) ---
inv11, inv_ptr, invertf, _ = SDSFused.build_inverse(s)
fz1 = SDSFused.build_fused(s; kstart = 1, wcap = 256)
fz = ([(fz1, 64)], true)
invertf()
x2 = similar(bd)
SDSFused.algo2_solve!(x2, s, bd, inv11, inv_ptr, fz)
CUDA.synchronize()
rel = norm(bh - A * Array(x2)) / norm(bh)
@printf "fused  S/N   solve relres %.2e\n" rel
rel < 1.0 || error("fused LDL solve gate FAILED")
ts_fused = median([CUDA.@elapsed SDSFused.algo2_solve!(x2, s, bd, inv11, inv_ptr, fz) for _ in 1:7])
@printf "fused  S/N   solve %8.2f ms\n" ts_fused*1e3
flush(stdout)

# --- fused Cholesky control (fresh solver, same matrix as SPD) ---
s2, ex2, bd2, xd2 = setup("SPD")
nz2 = SDS._factor_values(s2)
fp2 = SDSFused.build_fused_plan(s2; wgf = 128, scap = 128)
info2 = SDSFused.refact_fused!(s2, nz2, fp2)
tr_chol = median([CUDA.@elapsed SDSFused.refact_fused!(s2, nz2, fp2) for _ in 1:7])
@printf "fused  SPD   refact %8.2f ms  info %d  (control)\n" tr_chol*1e3 info2
@printf "RATIO fused LDL / fused Cholesky = %.3f\n" tr_fused / tr_chol
say("done")
