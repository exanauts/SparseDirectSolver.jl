# T22 (issue #108): the automatic ordering choice on the harness matrices, host only.
# For AMD and ND: supernodal schedule depth (scored, amalgamated as in the analysis),
# fundamental-supernode depth, column-etree depth, nnz(L), flops; the choice of the
# T22 cost model (`flops + level_flops × sdepth`) and of the T05 one
# (`flops × (1 + nlevels / n)`). Generated matrices, SuiteSparse (MatrixDepot) and
# every dump under bench/data/ (e.g. the 78k-bus condensed KKT of `dump_madnlp_kkt.jl`).
#
#   julia --project=bench bench/ordering_chooser.jl            # all
#   SDS_BENCH_SUITESPARSE=0 julia --project=bench bench/ordering_chooser.jl

using SparseDirectSolver, SparseArrays, LinearAlgebra, Printf
using Metis
const SDS = SparseDirectSolver

include(joinpath(@__DIR__, "matrices.jl"))
using .BenchMatrices: generated_matrices, suitesparse_matrices, dump_matrices

function chooser_rows(name, A)
    P = SDS.SymmetricPattern(SDS.CSR(SparseMatrixCSC{Float64, Int}(A + A')), "S"; view = 'F')
    t = @elapsed auto = SDS.compute_ordering(P, Options())
    n = size(A, 1)
    old = argmin(c -> c.flops * (1 + c.nlevels / max(n, 1)), auto.stats.candidates).alg
    for c in auto.stats.candidates
        perm = SDS.compute_ordering(P, Options(); alg = c.alg).perm
        parent = SDS.etree(P, perm)
        post = SDS.postorder(parent)
        fdepth, _ = SDS.schedule_depth(parent, post, SDS.colcounts(P, perm, parent, post), nothing)
        @printf("| %s | %d | %s | %d | %d | %d | %.3g | %.3g | %s | %s | %.2f |\n", name, n, uppercase(String(c.alg)),
                c.sdepth, fdepth, c.nlevels, c.nnz_L, c.flops, c.alg === auto.alg_used ? "**T22**" : "",
                c.alg === old ? "T05" : "", t)
    end
    flush(stdout)
end

mats = [(M.name, M.A) for M in generated_matrices()]
get(ENV, "SDS_BENCH_SUITESPARSE", "1") == "1" && append!(mats, [(M.name, M.A) for M in suitesparse_matrices()])
append!(mats, [(M.name, M.A) for M in dump_matrices()])

println("| matrix | n | ordering | schedule depth | fundamental depth | column-etree depth | nnz(L) | flops | ",
        "chosen (T22) | chosen (T05 model) | auto ordering s |")
println("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
for (name, A) in mats
    chooser_rows(name, A)
end
