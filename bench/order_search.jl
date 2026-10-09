# Proposal E: native ordering search. Sweep METIS NodeND knobs (seed, ufactor,
# nseps, compression, ccorder) and score every candidate on the host with
# SDS's evaluate_ordering (etree depth, fill, flops) against the cuDSS
# reordering (depth 30, nnz(L) 14.8M) and the SDS default. Winners are written
# to bench/perm_<name>.bin for the full-pipeline scripts.
#
#   julia +1.13 --project=bench bench/order_search.jl

using SparseDirectSolver, SparseArrays, LinearAlgebra, Random, Printf
using CUDA, CUDA.CUSPARSE
using Metis
const SDS = SparseDirectSolver
const LM = Metis.LibMetis

include(joinpath(@__DIR__, "matrices.jl"))
using .BenchMatrices: read_mtx

A = SparseMatrixCSC{Float64, Int}(sparse(read_mtx(joinpath(@__DIR__, "data",
    "kkt_pglib_opf_case78484_epigrids_condensed_10.mtx"))))
n = size(A, 1)
Lt = sparse(transpose(tril(A)))                 # CSR arrays of the lower triangle
rowptr = Vector{Int}(Lt.colptr)
colval = Vector{Int}(Lt.rowval)
P = SDS.SymmetricPattern(rowptr, colval, n, SDS.STRUCTURE_SPD;
                         view = SDS.VIEW_LOWER, index = SDS.INDEX_ONE)

up = Vector{Int32}(undef, n)
read!(joinpath(@__DIR__, "cudss_perm.bin"), up)

const LHD = tril(A)
results = Tuple{String, Int, Int, Float64, Vector{Int}}[]
function score(name, perm)
    r = SDS.evaluate_ordering(P, perm)
    # the metric that matters is the SUPERNODAL schedule depth, not the column etree
    sv = DirectSolver(CuSparseMatrixCSR(LHD), "SPD", 'L')
    sv.options.regime_c_width = 32; sv.options.regime_c_rows = 64; sv.options.subtree_parallelism = 16384
    SDS.setparam!(sv, "user_perm", Vector{Int}(perm))
    b = CUDA.rand(Float64, n); x = similar(b)
    SDS.execute!("analysis", sv, x, b; asynchronous = false)
    sc = sv.symbolic.schedule
    slv = maximum(Array(sc.level); init = 0)
    nsup = length(sc.width)
    push!(results, (name, slv, r.nnz_L, r.flops, Vector{Int}(perm)))
    @printf "%-28s SCHED levels %4d  supernodes %6d  col-levels %4d  nnz(L) %6.2fM  flops %7.3fG\n" name slv nsup r.nlevels r.nnz_L/1e6 r.flops/1e9
    flush(stdout)
    sv = nothing; GC.gc(); CUDA.reclaim()
end

score("cuDSS reordering", Int.(up))

# graph of A (pattern, no diagonal) for METIS
G = Metis.graph(A; check_hermitian = false)
setopt(o, v) = (Metis.options[Int(o) + 1] = Cint(v))
reset_opts() = (fill!(Metis.options, Cint(-1)); setopt(LM.METIS_OPTION_NUMBERING, 1))

reset_opts()
perm, _ = Metis.permutation(G)
score("metis default", Int.(perm))

for seed in (1, 2, 3, 4, 5, 6, 7)
    reset_opts(); setopt(LM.METIS_OPTION_SEED, seed)
    perm, _ = Metis.permutation(G)
    score("metis seed=$seed", Int.(perm))
end
for uf in (30, 60, 100, 200, 500)
    reset_opts(); setopt(LM.METIS_OPTION_UFACTOR, uf)
    perm, _ = Metis.permutation(G)
    score("metis ufactor=$uf", Int.(perm))
end
for ns in (2, 4)
    reset_opts(); setopt(LM.METIS_OPTION_NSEPS, ns)
    perm, _ = Metis.permutation(G)
    score("metis nseps=$ns", Int.(perm))
end
reset_opts(); setopt(LM.METIS_OPTION_COMPRESS, 0)
perm, _ = Metis.permutation(G)
score("metis compress=0", Int.(perm))
reset_opts(); setopt(LM.METIS_OPTION_CCORDER, 1)
perm, _ = Metis.permutation(G)
score("metis ccorder=1", Int.(perm))
reset_opts(); setopt(LM.METIS_OPTION_PFACTOR, 100)
perm, _ = Metis.permutation(G)
score("metis pfactor=100", Int.(perm))
reset_opts(); setopt(LM.METIS_OPTION_NSEPS, 4); setopt(LM.METIS_OPTION_UFACTOR, 100)
perm, _ = Metis.permutation(G)
score("metis nseps=4 uf=100", Int.(perm))

sort!(results; by = r -> (r[2], r[3]))
println("\nbest by depth:")
for r in results[1:min(5, end)]
    @printf "  %-28s levels %4d  nnz(L) %6.2fM\n" r[1] r[2] r[3]/1e6
end
best = results[1]
open(joinpath(@__DIR__, "perm_best.bin"), "w") do io
    write(io, Int32.(best[5]))
end
println("wrote perm_best.bin = ", best[1])
println("done")
