# Smoke + quality gate for the GPU-hybrid NodeND prototype: a 2-D Laplacian
# ordered by the exact port and by the hybrid, scored by nnz_L through SDS's
# own symbolic analysis. Passes = both perms valid and the hybrid within 5%
# fill of the exact port (measured 1.003–1.015 on the benchmark set; 5% is
# the alarm threshold, not the expectation).
#   julia +1.13 --project=bench bench/ordering/nodend_smoke.jl [grid=60]
using SparseDirectSolver, SparseArrays, LinearAlgebra, Printf
const SDS = SparseDirectSolver
const KA = SDS.KernelAbstractions

include(joinpath(@__DIR__, "gpu_nodend.jl"))

laplacian2d(nx, ny) = begin
    Tx = spdiagm(-1 => -ones(nx - 1), 0 => 4ones(nx), 1 => -ones(nx - 1))
    Ty = spdiagm(-1 => -ones(ny - 1), 1 => -ones(ny - 1))
    Ix = sparse(1.0I, nx, nx); Iy = sparse(1.0I, ny, ny)
    kron(Iy, Tx) + kron(Ty, Ix)
end

function nnzL(A, perm)
    n = size(A, 1)
    Lt = sparse(transpose(tril(A)))
    P = SDS.SymmetricPattern(Vector{Int}(Lt.colptr), Vector{Int}(Lt.rowval), n,
                             SDS.STRUCTURE_SPD; view = SDS.VIEW_LOWER, index = SDS.INDEX_ONE)
    Int(SDS.evaluate_ordering(P, perm).nnz_L)
end

g = isempty(ARGS) ? 60 : parse(Int, ARGS[1])
A = laplacian2d(g, g)
n = size(A, 1)
println("lap2d $(g)x$(g)  n = $n")

pe, _ = MetisExact.nodend_exact(A)
@assert isperm(pe)
ph, _ = Hybrid.nodend_hybrid(KA.CPU(), A)
@assert isperm(ph)

le = nnzL(A, pe)
lh = nnzL(A, ph)
ratio = lh / le
@printf "nnz_L exact %d  hybrid %d  ratio %.4f\n" le lh ratio
@assert ratio < 1.05 "hybrid fill regressed beyond the 5% alarm threshold"
println("SMOKE OK")
