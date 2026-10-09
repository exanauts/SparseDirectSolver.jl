# dump cuDSS's reordering permutation for the 78k condensed dump
using SparseArrays, LinearAlgebra, Random
using CUDA, CUDA.cuSPARSE, CUDSS
include(joinpath(@__DIR__, "matrices.jl"))
using .BenchMatrices: read_mtx
A = SparseMatrixCSC{Float64, Int}(sparse(read_mtx(joinpath(@__DIR__, "data",
    "kkt_pglib_opf_case78484_epigrids_condensed_10.mtx"))))
n = size(A, 1)
Ad = CuSparseMatrixCSR(tril(A))
b = CuArray(rand(n)); x = similar(b)
solver = CUDSS.CudssSolver(Ad, "SPD", 'L')
CUDSS.cudss("analysis", solver, x, b)
CUDA.synchronize()
buf = Vector{Cint}(undef, n)
nw = Ref{Csize_t}(0)
CUDSS.cudssDataGet(solver.data.handle, solver.data, CUDSS.CUDSS_DATA_PERM_REORDER_ROW, buf, sizeof(buf), nw)
println("bytes written: ", nw[])
p = buf
println("perm length ", length(p), " extrema ", extrema(p))
open(joinpath(@__DIR__, "cudss_perm.bin"), "w") do io
    write(io, Int32.(p))
end
println("written")
