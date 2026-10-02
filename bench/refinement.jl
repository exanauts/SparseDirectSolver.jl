# Iterative refinement on the MadNLP K2 dumps (T16, issue #71): relres =
# ‖b − A x‖ / ‖b‖ after `ir_n_steps` ∈ {0, 2, 5} with the handle-layer solver
# (structure "S", view 'L', default pivoting), b = A * ones. Prints a Markdown
# table; dumps come from bench/dump_madnlp_kkt.jl (bench/data/kkt_*_k2_*.mtx).
# Runs on the KA CPU backend, or on CUDA with --backend=cuda (needs CUDA).
#
#   julia --project=. bench/refinement.jl [--only=substring,...] [--steps=0,2,5] [--backend=cpu|cuda]

using LinearAlgebra
using SparseArrays
using SparseDirectSolver

include(joinpath(@__DIR__, "matrices.jl"))

function parse_args(args)
    o = Dict{String, String}("only" => "", "steps" => "0,2,5", "backend" => "cpu")
    for a in args
        m = match(r"^--(only|steps|backend)=(.*)$", a)
        m === nothing && error("unknown argument \"$a\"; see the header of bench/refinement.jl")
        o[m[1]] = m[2]
    end
    return o
end

const OPTS = parse_args(ARGS)
const STEPS = parse.(Int, split(OPTS["steps"], ','))

if OPTS["backend"] == "cuda"
    @eval using CUDA
    to_dev(x::AbstractVector) = CuArray(x)
    to_dev(A::SparseMatrixCSC) = CUDA.CUSPARSE.CuSparseMatrixCSR(A)
else
    to_dev(x) = x
end

function relres_table(name, A::SparseMatrixCSC{T}) where {T}
    L = tril(A)
    solver = DirectSolver(to_dev(L), "S", 'L')
    analyze!(solver)
    factorize!(solver; asynchronous = false)
    b = A * ones(T, size(A, 1))
    bd = to_dev(b)
    xd = similar(bd)
    res = Float64[]
    for k in STEPS
        setparam!(solver, "ir_n_steps", k)
        solve!(solver, xd, bd; asynchronous = false)
        x = Array(xd)
        push!(res, norm(b - A * x) / norm(b))
    end
    println("| ", name, " | ", size(A, 1), " | ", getparam(solver, "npivots"), " | ",
            join((string(round(r; sigdigits = 2)) for r in res), " | "), " |")
    return res
end

selected(name) = isempty(OPTS["only"]) || any(s -> occursin(s, name), split(OPTS["only"], ','))

dumps = isdir(BenchMatrices.DATA_DIR) ?
        sort([f for f in readdir(BenchMatrices.DATA_DIR) if (d = BenchMatrices.parse_dump_name(f)) !== nothing &&
              d.kind == "k2" && d.ext == "mtx"]) : String[]
isempty(dumps) && println("no K2 dumps in ", BenchMatrices.DATA_DIR, " (run bench/dump_madnlp_kkt.jl)")
println("| K2 dump | n | npivots | ", join(("relres, ir_n_steps = $k" for k in STEPS), " | "), " |")
println("| --- | ---: | ---: | ", join(("---:" for _ in STEPS), " | "), " |")
for f in dumps
    selected(f) || continue
    A = BenchMatrices.symmetrize_triangle(tril(BenchMatrices.read_mtx(joinpath(BenchMatrices.DATA_DIR, f))))
    relres_table(f, A)
end
