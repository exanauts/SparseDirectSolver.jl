# Locate NaNs in the fused LDL factor by front class.
#   CUDA_VISIBLE_DEVICES=1 julia +1.13 --project=bench bench/e2e/ldl_fused_debug.jl
using SparseDirectSolver, SparseArrays, LinearAlgebra, Random, Statistics, Printf
using CUDA, CUDA.CUSPARSE
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
SDSFused.BACKEND[] = CUDA.CUDABackend()

bd = CuArray(bh); xd = similar(bd)
s = DirectSolver(CuSparseMatrixCSR(Lh), "S", 'L')
SDS.setparam!(s, "user_perm", S3)
SDS.setparam!(s, "pivot_type", 'N')
ex(ph) = SDS.execute!(ph, s, xd, bd; asynchronous = false)
ex("analysis"); ex("factorization"); ex("refactorization"); ex("solve"); CUDA.synchronize()
fac0 = Array(s.numeric.factor)
d0 = Array(s.numeric.d)

nzval = SDS._factor_values(s)
fp = SDSFused.build_fused_plan(s; ldl = true, wgf = 128, scap = 128)
info = SDSFused.refact_fused!(s, nzval, fp)
CUDA.synchronize()
fac1 = Array(s.numeric.factor)
d1 = Array(s.numeric.d)

S = s.symbolic
fptr = Array(S.front_ptr); fr = Array(S.front_nrows); fc = Array(S.front_ncols)
sptr = Array(S.super_ptr)
stn = Array(S.subtree_nodes)
ns = length(fr)
covered = falses(ns); for v in stn; covered[v] = true; end
dbad = findall(isnan, @view d1[1:n])
println("NaN in d[1:n]: ", length(dbad), "   first cols: ", first(dbad, 5))
# classify fronts
nbad = Dict{String,Int}(); tot = Dict{String,Int}()
firstbad = Int[]
for v in 1:ns
    f, w = fr[v], fc[v]
    m = f - w
    cls = covered[v] ? "subtree" : (w > 64 ? "wide" : (m <= 64 ? "panel-inline" : "panel-split"))
    tot[cls] = get(tot, cls, 0) + 1
    rng = fptr[v]:(fptr[v] + f * w - 1)
    bad = any(isnan, @view fac1[rng])
    dbadv = any(isnan, @view d1[sptr[v]:(sptr[v] + w - 1)])
    if bad || dbadv
        nbad[cls] = get(nbad, cls, 0) + 1
        length(firstbad) < 6 && push!(firstbad, v)
    end
end
println("front classes: ", tot)
println("NaN fronts:    ", nbad)
for v in firstbad
    f, w = fr[v], fc[v]
    dfrag = d1[sptr[v]:min(sptr[v] + w - 1, sptr[v] + 4)]
    @printf "  front %d  f %d w %d  covered %s  d[1:5] %s  d0 %s\n" v f w string(covered[v]) string(round.(dfrag; sigdigits=3)) string(round.(d0[sptr[v]:min(sptr[v]+4, sptr[v]+w-1)]; sigdigits=3))
end
# also: max |Δ| over non-NaN fronts
ok = .!isnan.(fac1)
println("Δfactor over finite entries: ", maximum(abs.(fac1[ok] .- fac0[ok])) / maximum(abs, fac0))
println("done")
