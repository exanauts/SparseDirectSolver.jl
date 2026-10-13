# MA27 (serial multifrontal, 1983) on the 78k KKT, via libcoinhsl (the libHSL
# 2025 build STUBS the Archive codes: its ma27*_ symbols return instantly with
# zeroed output, and the subset build returns -26 "not available").
#   julia +1 --startup-file=no --project=bench bench/e2e/ma27_bench.jl
using SparseArrays, LinearAlgebra, Random, Statistics, Printf
const LIB = "/usr/local/lib/libcoinhsl.so"
const BD = joinpath(@__DIR__, "..")
include(joinpath(BD, "matrices.jl"))
using .BenchMatrices: read_mtx

MAT = get(ENV, "MA27_MAT", "kkt_pglib_opf_case78484_epigrids_condensed_10.mtx")
A = SparseMatrixCSC{Float64, Int}(sparse(read_mtx(joinpath(BD, "data", MAT))))
println("matrix: ", MAT, "  n ", size(A, 1), "  nnz(tril) ", nnz(tril(A)))
n = Int32(size(A, 1))
L = tril(A)
nz = Int32(nnz(L))
I32 = Vector{Int32}(undef, nz); J32 = Vector{Int32}(undef, nz)
let k = 0
    for j in 1:size(L, 2), p in L.colptr[j]:(L.colptr[j + 1] - 1)
        k += 1
        I32[k] = Int32(L.rowval[p])
        J32[k] = Int32(j)
    end
end
vals = L.nzval
Random.seed!(666)
bh = rand(Float64, Int(n))
say(x...) = (println(x...); flush(stdout))

icntl = zeros(Int32, 30); cntl = zeros(Float64, 5)
@ccall LIB.ma27id_(icntl::Ptr{Int32}, cntl::Ptr{Float64})::Cvoid
icntl[1:2] .= 0
cntl[1] = 1e-8                                   # pivtol as MadNLP's IPM default
info = zeros(Int32, 20)

liw = Int32(8 * (2 * Int(nz) + 3 * Int(n) + 1))
iw = Vector{Int32}(undef, liw)
ikeep = zeros(Int32, 3n)
iw1 = zeros(Int32, 2n)
nsteps = Int32[0]; iflag = Int32[0]; ops = Float64[0.0]
ta = @elapsed @ccall LIB.ma27ad_(n::Ref{Int32}, nz::Ref{Int32}, I32::Ptr{Int32}, J32::Ptr{Int32},
                                 iw::Ptr{Int32}, liw::Ref{Int32}, ikeep::Ptr{Int32}, iw1::Ptr{Int32},
                                 nsteps::Ptr{Int32}, iflag::Ptr{Int32}, icntl::Ptr{Int32},
                                 cntl::Ptr{Float64}, info::Ptr{Int32}, ops::Ptr{Float64})::Cvoid
say("analysis ", round(ta; digits = 2), " s  info[1] ", info[1], "  forecast la ", info[5],
    "  forecast liw ", info[6], "  nsteps ", nsteps[1])
info[1] >= 0 || error("ma27ad failed: info = $(info[1:4])")

a = Vector{Float64}(undef, max(5 * Int(info[5]), 2 * Int(nz)))
resize!(iw, max(5 * Int(info[6]), 2 * Int(nz)))
maxfrt = Int32[0]
function factor!()
    copyto!(a, 1, vals, 1, Int(nz))
    @ccall LIB.ma27bd_(n::Ref{Int32}, nz::Ref{Int32}, I32::Ptr{Int32}, J32::Ptr{Int32},
                       a::Ptr{Float64}, Int32(length(a))::Ref{Int32}, iw::Ptr{Int32},
                       Int32(length(iw))::Ref{Int32}, ikeep::Ptr{Int32}, nsteps::Ptr{Int32},
                       maxfrt::Ptr{Int32}, iw1::Ptr{Int32}, icntl::Ptr{Int32}, cntl::Ptr{Float64},
                       info::Ptr{Int32})::Cvoid
    return Int(info[1])
end
tf = 0.0; st = 0
for attempt in 1:4
    global tf = @elapsed global st = factor!()
    say("factor attempt ", attempt, ": info[1] ", st, "  info[2] ", info[2], "  maxfrt ", maxfrt[1])
    if st == -3
        resize!(iw, 2 * max(length(iw), Int(info[2])))
    elseif st == -4
        resize!(a, 2 * max(length(a), Int(info[2])))
    else
        break
    end
end
st >= 0 || error("ma27bd failed: info = $(info[1:4])")
say("first factor ", round(tf * 1e3; digits = 1), " ms  factor-real info[9] ", info[9],
    "  factor-int info[10] ", info[10], "  ncmpa info[11] ", info[11], "  twos info[14] ", info[14])
tr = median([(@elapsed factor!()) for _ in 1:5])

w = Vector{Float64}(undef, Int(maxfrt[1]))
x = copy(bh)
sv!() = @ccall LIB.ma27cd_(n::Ref{Int32}, a::Ptr{Float64}, Int32(length(a))::Ref{Int32},
                           iw::Ptr{Int32}, Int32(length(iw))::Ref{Int32}, w::Ptr{Float64},
                           maxfrt::Ptr{Int32}, x::Ptr{Float64}, iw1::Ptr{Int32}, nsteps::Ptr{Int32},
                           icntl::Ptr{Int32}, info::Ptr{Int32})::Cvoid
sv!()
rel = norm(bh - A * x) / norm(bh)
ts = median([begin x .= bh; @elapsed sv!() end for _ in 1:5])
@printf "MA27  refact %8.1f ms  solve %6.1f ms  relres %.2e\n" tr*1e3 ts*1e3 rel
say("done")
