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

# in-kernel wides (merged segments), workgroup 128, spin cap 128
const TUNING = SDSFused.SDSTuning(; wides = :merged, workgroup = 128, spin_cap = 128)

# a FusedSolver whose analysis also ran the stock factorization and solve on its inner DirectSolver
function setup(structure)
    bd = CuArray(bh); xd = similar(bd)
    F = FusedSolver(CuSparseMatrixCSR(Lh), structure, 'L'; tuning = TUNING)
    setparam!(F, "user_perm", S3)
    analyze!(F; asynchronous = false)
    ex(ph) = SDS.execute!(ph, F.inner, xd, bd; asynchronous = false)
    ex("refactorization"); ex("solve")
    CUDA.synchronize()
    return F, ex, bd, xd
end

say(x...) = (println(x...); flush(stdout))

# --- stock LDL pivot-N on the inner solver: reference factor/d + timing ---
F, ex, bd, xd = setup("S")
s = F.inner
tr_stock = median([CUDA.@elapsed ex("refactorization") for _ in 1:7])
fac0 = Array(s.numeric.factor)
d0 = Array(s.numeric.d)
@printf "stock  S/N   refact %8.2f ms  info %d\n" tr_stock*1e3 SDS.getparam(s, "info")
flush(stdout)

# --- fused LDL on the same analysis ---
factorize!(F; asynchronous = false)
info = getparam(F, "info")
fac1 = Array(s.numeric.factor)
d1 = Array(s.numeric.d)
df = maximum(abs, fac1 - fac0) / maximum(abs, fac0)
dd = maximum(abs, d1 - d0) / maximum(abs, d0)
@printf "fused  S/N   info %d  Δfactor %.2e  Δd %.2e\n" info df dd
(info == 0 && df < 1e-6 && dd < 1e-6) || error("fused LDL gate FAILED")
# refactorize! = the fused factorization + the per-front inverses of the solve
tr_fused = median([CUDA.@elapsed refactorize!(F) for _ in 1:7])
@printf "fused  S/N   refact %8.2f ms\n" tr_fused*1e3
flush(stdout)

# --- fused solve (inverse path + D pass) ---
x2 = similar(bd)
solve!(F, x2, bd; asynchronous = false)
rel = norm(bh - A * Array(x2)) / norm(bh)
@printf "fused  S/N   solve relres %.2e\n" rel
# gate against the stock solve's residual on the same system (xd holds the stock
# solution at this point), not an absolute threshold a zero vector nearly meets
rel_stock = norm(bh - A * Array(xd)) / norm(bh)
rel <= 10 * rel_stock + 1e-12 ||
    error("fused LDL solve gate FAILED: relres $rel vs stock $rel_stock")
ts_fused = median([CUDA.@elapsed solve!(F, x2, bd) for _ in 1:7])
@printf "fused  S/N   solve %8.2f ms\n" ts_fused*1e3
flush(stdout)

# --- fused Cholesky control (fresh solver, same matrix as SPD) ---
F2, _, _, _ = setup("SPD")
factorize!(F2)
info2 = getparam(F2, "info")
tr_chol = median([CUDA.@elapsed refactorize!(F2) for _ in 1:7])
@printf "fused  SPD   refact %8.2f ms  info %d  (control)\n" tr_chol*1e3 info2
@printf "RATIO fused LDL / fused Cholesky = %.3f\n" tr_fused / tr_chol
say("done")
