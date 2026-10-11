# Radeon VII (gfx906) baseline: fused Cholesky (campaign champion knobs) and the
# first fused-LDL run on AMD, gated, interleaved.
#   ~/.juliaup/bin/julia +1.13 --startup-file=no --project=~/.julia/environments/sdsamd \
#       bench/e2e/amd_baseline.jl
using SparseDirectSolver, SparseArrays, LinearAlgebra, Random, Statistics, Printf
using AMDGPU
const SDS = SparseDirectSolver
include(joinpath(@__DIR__, "SDSFused.jl"))
using .SDSFused
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
SDSFused.BACKEND[] = AMDGPU.ROCBackend()
sync() = AMDGPU.synchronize()

# build via plain device arrays (AMDGPU has no CuSparseMatrixCSR; DirectSolver takes raw triplets)
function mks(structure)
    Lt = sparse(transpose(Lh))                    # CSC of upper = CSR of lower
    colptr = ROCArray(Vector{Int}(Lt.colptr))
    rowval = ROCArray(Vector{Int}(Lt.rowval))
    nzval = ROCArray(Vector{Float64}(Lt.nzval))
    s = DirectSolver(colptr, rowval, nzval, structure, 'L')
    s.options.subtree_parallelism = 16384
    s.options.subtree_budgets = [8192]            # campaign AMD champion
    s.options.regime_c_width = 32
    s.options.regime_c_rows = 64
    s.options.amalgamation = (max_width = 48, zero_fraction = 0.25, min_width = 8)
    SDS.setparam!(s, "user_perm", S3)
    structure == "S" && SDS.setparam!(s, "pivot_type", 'N')
    bd = ROCArray(bh); xd = similar(bd)
    ex(ph) = SDS.execute!(ph, s, xd, bd; asynchronous = false)
    ex("analysis"); ex("factorization"); ex("refactorization"); ex("solve"); sync()
    return s, bd, xd
end

say(x...) = (println(x...); flush(stdout))
say("building solvers…")
sc, bdc, xdc = mks("SPD")
sl, bdl, xdl = mks("S")
fac0 = Array(sl.numeric.factor); d0 = Array(sl.numeric.d)
nzc = SDS._factor_values(sc); nzl = SDS._factor_values(sl)
# campaign AMD champion: merged in-kernel wides, wgf 256, spin cap 16
fpc = SDSFused.build_fused_plan(sc; inkernel_wides = true, merge_segments = true, wgf = 256, scap = 16)
fpl = SDSFused.build_fused_plan(sl; ldl = true, merge_segments = true, wgf = 256, scap = 16)
infoc = SDSFused.refact_fused!(sc, nzc, fpc); sync()
infol = SDSFused.refact_fused!(sl, nzl, fpl); sync()
df = maximum(abs, Array(sl.numeric.factor) - fac0) / maximum(abs, fac0)
dd = maximum(abs, Array(sl.numeric.d) - d0) / maximum(abs, d0)
@printf "gates: chol info %d   ldl info %d  Δfactor %.2e  Δd %.2e\n" infoc infol df dd
(infoc == 0 && infol == 0 && df < 1e-6 && dd < 1e-6) || error("AMD gate FAILED")
tc = Float64[]; tl = Float64[]
for r in 1:7
    push!(tc, @elapsed (SDSFused.refact_fused!(sc, nzc, fpc); sync()))
    push!(tl, @elapsed (SDSFused.refact_fused!(sl, nzl, fpl); sync()))
end
@printf "gfx906 interleaved  chol %7.2f ms   ldl %7.2f ms   ratio %.3f\n" median(tc)*1e3 median(tl)*1e3 median(tl)/median(tc)
say("done")
