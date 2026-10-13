# gfx906 knob sweep around the champion config (chol fused refact, gated, each
# config interleaved against the champion in the same process).
#   julia +1.13 --project=~/.julia/environments/sdsamd bench/e2e/amd_sweep.jl
using SparseDirectSolver, SparseArrays, LinearAlgebra, Random, Statistics, Printf
using AMDGPU
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
sync() = AMDGPU.synchronize()
say(x...) = (println(x...); flush(stdout))

function mks(; budgets = [8192], amal = (max_width = 48, zero_fraction = 0.25, min_width = 8))
    Lt = sparse(transpose(Lh))
    s = DirectSolver(ROCArray(Vector{Int}(Lt.colptr)), ROCArray(Vector{Int}(Lt.rowval)),
                     ROCArray(Vector{Float64}(Lt.nzval)), "SPD", 'L')
    s.options.subtree_parallelism = 16384
    s.options.subtree_budgets = budgets
    s.options.regime_c_width = 32
    s.options.regime_c_rows = 64
    s.options.amalgamation = amal
    SDS.setparam!(s, "user_perm", S3)
    bd = ROCArray(bh); xd = similar(bd)
    ex(ph) = SDS.execute!(ph, s, xd, bd; asynchronous = false)
    ex("analysis"); ex("factorization"); ex("refactorization"); ex("solve"); sync()
    return s, Array(s.numeric.factor)                 # stock factor = per-solver gate reference
end

# champion baseline solver (fixed) for interleaving
sb, facb = mks()
nzb = SDS._factor_values(sb)
fpb = SDSFused.build_fused_plan(sb; inkernel_wides = true, merge_segments = true, wgf = 256, scap = 16)
SDSFused.refact_fused!(sb, nzb, fpb); sync()

function trial(label, s, fac0, fp)
    nz = SDS._factor_values(s)
    info = SDSFused.refact_fused!(s, nz, fp); sync()
    df = maximum(abs, Array(s.numeric.factor) - fac0) / maximum(abs, fac0)
    ok = info == 0 && df < 1e-6
    tb = Float64[]; tt = Float64[]
    for r in 1:5
        push!(tb, @elapsed (SDSFused.refact_fused!(sb, nzb, fpb); sync()))
        push!(tt, @elapsed (SDSFused.refact_fused!(s, nz, fp); sync()))
    end
    @printf "%-34s %s  trial %7.2f ms   champion %7.2f ms   delta %+6.1f%%\n" label (ok ? "OK " : "BAD") median(tt)*1e3 median(tb)*1e3 (median(tt)/median(tb) - 1)*100
    flush(stdout)
end

# plan-level knobs on the SAME solver/symbolic (cheap)
for (wgf, scap) in ((256, 8), (256, 32), (256, 64), (512, 16), (128, 16))
    fp = SDSFused.build_fused_plan(sb; inkernel_wides = true, merge_segments = true, wgf = wgf, scap = scap)
    trial("wgf=$wgf scap=$scap", sb, facb, fp)
end

# symbolic-level knobs (re-analysis each)
for budgets in ([4096], [16384])
    s, f0 = mks(; budgets)
    fp = SDSFused.build_fused_plan(s; inkernel_wides = true, merge_segments = true, wgf = 256, scap = 16)
    trial("budgets=$budgets", s, f0, fp)
end
for mw in (64, 96)
    s, f0 = mks(; amal = (max_width = mw, zero_fraction = 0.25, min_width = 8))
    fp = SDSFused.build_fused_plan(s; inkernel_wides = true, merge_segments = true, wgf = 256, scap = 16)
    trial("amal max_width=$mw", s, f0, fp)
end
say("done")

# EA chunk size (plan-level)
for eac in (8192, 32768, 65536)
    fp = SDSFused.build_fused_plan(sb; inkernel_wides = true, merge_segments = true, wgf = 256,
                                   scap = 16, ea_chunk = eac)
    trial("ea_chunk=$eac", sb, facb, fp)
end
say("done2")
