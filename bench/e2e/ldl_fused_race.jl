# 20 gated fused-LDL refactorizations (race validation) + the fused Cholesky control
# with the same wide-front strategy, which isolates the wide-strategy cost from the LDL cost.
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

# host wides (the LDL wide fronts on the Cholesky vendor path, wides_potrf), workgroup 128,
# spin cap 128: the plan this gate was written for
const TUNING = SDSFused.SDSTuning(; wides = :host, workgroup = 128, spin_cap = 128)

# a FusedSolver whose analysis also ran the stock factorization and solve on its inner
# DirectSolver; the stock factor of the same matrix is the gate reference
function setup(structure)
    bd = CuArray(bh); xd = similar(bd)
    F = FusedSolver(CuSparseMatrixCSR(Lh), structure, 'L'; tuning = TUNING)
    setparam!(F, "user_perm", S3)
    analyze!(F; asynchronous = false)
    ex(ph) = SDS.execute!(ph, F.inner, xd, bd; asynchronous = false)
    ex("refactorization"); ex("solve")
    CUDA.synchronize()
    return F
end

F = setup("S")
s = F.inner
fac0 = Array(s.numeric.factor)
d0 = Array(s.numeric.d)
mref = maximum(abs, fac0)
nbad = 0
for t in 1:20
    factorize!(F; asynchronous = false)
    info = getparam(F, "info")
    df = maximum(abs, Array(s.numeric.factor) - fac0) / mref
    dd = maximum(abs, Array(s.numeric.d) - d0) / maximum(abs, d0)
    ok = info == 0 && df < 1e-6 && dd < 1e-6
    ok || (global nbad += 1)
    @printf "run %2d  %s  info %d  Δfactor %.2e  Δd %.2e\n" t (ok ? "OK " : "BAD") info df dd
    flush(stdout)
end
println("bad: ", nbad, " of 20")
nbad == 0 || error("fused LDL race gate FAILED: $nbad of 20 runs bad")   # a gate must exit nonzero

# Cholesky with the SAME wide strategy (host wides) as the LDL solver above
F2 = setup("SPD")
refactorize!(F2)
# refactorize! = the fused factorization + the per-front inverses of the solve
tr_chol_host = median([CUDA.@elapsed refactorize!(F2) for _ in 1:7])
tr_ldl = median([CUDA.@elapsed refactorize!(F) for _ in 1:7])
@printf "fused SPD (host wides) refact %8.2f ms\n" tr_chol_host*1e3
@printf "fused S/N (host wides) refact %8.2f ms   ratio same-strategy %.3f\n" tr_ldl*1e3 tr_ldl/tr_chol_host
println("done")
