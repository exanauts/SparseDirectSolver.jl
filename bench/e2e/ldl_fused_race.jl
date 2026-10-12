# 20 gated fused-LDL refactorizations (race validation) + the in-kernel-wides
# Cholesky control that isolates the wide-strategy cost from the LDL cost.
#   CUDA_VISIBLE_DEVICES=1 julia +1.13 --project=bench bench/e2e/ldl_fused_race.jl
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
    return s
end

s = setup("S")
fac0 = Array(s.numeric.factor)
d0 = Array(s.numeric.d)
mref = maximum(abs, fac0)
nz = SDS._factor_values(s)
fp = SDSFused.build_fused_plan(s; ldl = true, wides_potrf = true, wgf = 128, scap = 128)
nbad = 0
for t in 1:20
    info = SDSFused.refact_fused!(s, nz, fp); CUDA.synchronize()
    df = maximum(abs, Array(s.numeric.factor) - fac0) / mref
    dd = maximum(abs, Array(s.numeric.d) - d0) / maximum(abs, d0)
    ok = info == 0 && df < 1e-6 && dd < 1e-6
    ok || (global nbad += 1)
    @printf "run %2d  %s  info %d  Δfactor %.2e  Δd %.2e\n" t (ok ? "OK " : "BAD") info df dd
    flush(stdout)
end
println("bad: ", nbad, " of 20")

# Cholesky with the SAME wide strategy (in-kernel), isolating the LDL-math cost
s2 = setup("SPD")
nz2 = SDS._factor_values(s2)
fp2i = SDSFused.build_fused_plan(s2; inkernel_wides = true, wgf = 128, scap = 128)
SDSFused.refact_fused!(s2, nz2, fp2i)
tr_chol_ik = median([CUDA.@elapsed SDSFused.refact_fused!(s2, nz2, fp2i) for _ in 1:7])
tr_ldl = median([CUDA.@elapsed SDSFused.refact_fused!(s, nz, fp) for _ in 1:7])
@printf "fused SPD (in-kernel wides) refact %8.2f ms\n" tr_chol_ik*1e3
@printf "fused S/N (in-kernel wides) refact %8.2f ms   ratio same-strategy %.3f\n" tr_ldl*1e3 tr_ldl/tr_chol_ik
println("done")
