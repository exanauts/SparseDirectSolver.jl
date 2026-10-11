# MA57 (serial multifrontal with BLAS3, MA27's successor) via libcoinhsl.
#   julia +1 --startup-file=no bench/e2e/ma57_bench.jl
using SparseArrays, LinearAlgebra, Random, Statistics, Printf
const LIB = "/usr/local/lib/libcoinhsl.so"
const BD = joinpath(@__DIR__, "..")
include(joinpath(BD, "matrices.jl"))
using .BenchMatrices: read_mtx

MAT = get(ENV, "MA57_MAT", "kkt_pglib_opf_case78484_epigrids_condensed_10.mtx")
A = SparseMatrixCSC{Float64, Int}(sparse(read_mtx(joinpath(BD, "data", MAT))))
println("matrix: ", MAT, "  n ", size(A, 1), "  nnz(tril) ", nnz(tril(A)))
n = Int32(size(A, 1))
L = tril(A)
ne = Int32(nnz(L))
I32 = Vector{Int32}(undef, ne); J32 = Vector{Int32}(undef, ne)
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

cntl = zeros(Float64, 5); icntl = zeros(Int32, 20)
@ccall LIB.ma57id_(cntl::Ptr{Float64}, icntl::Ptr{Int32})::Cvoid
icntl[5] = 0                                      # print level
cntl[1] = 1e-8                                    # pivtol, as MadNLP
info = zeros(Int32, 40); rinfo = zeros(Float64, 20)
lkeep = Int32(7 * Int(n) + Int(ne) + max(Int(n), Int(ne)) + 42)
keep = zeros(Int32, lkeep)
iwork = zeros(Int32, 5 * Int(n))
ta = @elapsed @ccall LIB.ma57ad_(n::Ref{Int32}, ne::Ref{Int32}, I32::Ptr{Int32}, J32::Ptr{Int32},
                                 lkeep::Ref{Int32}, keep::Ptr{Int32}, iwork::Ptr{Int32},
                                 icntl::Ptr{Int32}, info::Ptr{Int32}, rinfo::Ptr{Float64})::Cvoid
say("analysis ", round(ta; digits = 2), " s  info[1] ", info[1], "  forecast fact ", info[9],
    "  forecast ifact ", info[10])
info[1] >= 0 || error("ma57ad failed: info = $(info[1:4])")

lfact = Int32(max(3 * Int(info[9]), 2 * Int(ne)))
fact = Vector{Float64}(undef, lfact)
lifact = Int32(max(3 * Int(info[10]), 2 * Int(ne)))
ifact = Vector{Int32}(undef, lifact)
iwn = zeros(Int32, n)
function factor!()
    @ccall LIB.ma57bd_(n::Ref{Int32}, ne::Ref{Int32}, vals::Ptr{Float64}, fact::Ptr{Float64},
                       Int32(length(fact))::Ref{Int32}, ifact::Ptr{Int32},
                       Int32(length(ifact))::Ref{Int32}, lkeep::Ref{Int32}, keep::Ptr{Int32},
                       iwn::Ptr{Int32}, icntl::Ptr{Int32}, cntl::Ptr{Float64}, info::Ptr{Int32},
                       rinfo::Ptr{Float64})::Cvoid
    return Int(info[1])
end
tf = 0.0; st = 0
for attempt in 1:4
    global tf = @elapsed global st = factor!()
    say("factor attempt ", attempt, ": info[1] ", st, "  info[2] ", info[2])
    if st == -3
        resize!(fact, 2 * max(length(fact), Int(info[17])))
    elseif st == -4
        resize!(ifact, 2 * max(length(ifact), Int(info[18])))
    else
        break
    end
end
st >= 0 || error("ma57bd failed: info = $(info[1:4])")
say("first factor ", round(tf * 1e3; digits = 1), " ms  factor-real info[14] ", info[14],
    "  negpiv info[24] ", info[24])
tr = median([(@elapsed factor!()) for _ in 1:5])

x = copy(bh)
wrk = zeros(Float64, n)
sv!() = @ccall LIB.ma57cd_(Int32(1)::Ref{Int32}, n::Ref{Int32}, fact::Ptr{Float64},
                           Int32(length(fact))::Ref{Int32}, ifact::Ptr{Int32},
                           Int32(length(ifact))::Ref{Int32}, Int32(1)::Ref{Int32}, x::Ptr{Float64},
                           n::Ref{Int32}, wrk::Ptr{Float64}, n::Ref{Int32}, iwn::Ptr{Int32},
                           icntl::Ptr{Int32}, info::Ptr{Int32})::Cvoid
sv!()
rel = norm(bh - A * x) / norm(bh)
ts = median([begin x .= bh; @elapsed sv!() end for _ in 1:5])
@printf "MA57  refact %8.1f ms  solve %6.1f ms  relres %.2e\n" tr*1e3 ts*1e3 rel
say("done")
