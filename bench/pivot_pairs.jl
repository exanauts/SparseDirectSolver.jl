# Fill and pivot quality of the 2×2 pivot pairs of the "S" analysis (issue #66):
# `pivot_pairs` ∈ {"none", "default", "all"} on the MadNLP K2 dumps in bench/data
# (bench/dump_madnlp_kkt.jl) and on the KKT generators of test/matrices.jl.
# Host only: analysis plus the CPU reference LDLᵀ (`ref_ldlt!`). Prints a Markdown
# table per matrix: pairs, nnz(L) and its ratio to "none", supernodes, zero and
# perturbed pivots, 2×2 pivots, max|L| and the factor error ‖A[p,p] − LDLᵀ‖_F/‖A‖_F.
#
# Usage (package environment):
#
#   julia --project=. bench/pivot_pairs.jl [--only=substring,...] [--generators=true|false]
#
# The pivot statistics of the dumps say little about accuracy without scaling: the
# K2 systems are not scaled, so max|L| reaches 1e15 in every mode (issue #71). The
# rows "algo5/<pairs>" (T21) run the public solver (KA CPU backend) with
# `matching_alg = "algo5"`: symmetric MC64 scaling `D A D` and the 2×2 pairs of the
# matching cycles; max|L| and the error are those of the scaled matrix
# `‖(DAD)[p,p] − LDLᵀ‖_F/‖DAD‖_F`, `relres` is `‖b − A x‖/‖b‖` of one solve
# without refinement (`b = A·1`), and `relres+5` after 5 refinement steps.

using LinearAlgebra
using Random
using SparseArrays
using SparseDirectSolver
const SDS = SparseDirectSolver

include(joinpath(@__DIR__, "matrices.jl"))
include(joinpath(@__DIR__, "..", "test", "matrices.jl"))

function parse_args(args)
    o = Dict{String, String}("only" => "", "generators" => "true")
    for a in args
        m = match(r"^--(only|generators)=(.*)$", a)
        m === nothing && error("unknown argument \"$a\"; see the header of bench/pivot_pairs.jl")
        o[m[1]] = m[2]
    end
    return o
end

const OPTS = parse_args(ARGS)
const MODES = ("none", "default", "all")

function measure(A::SparseMatrixCSC{T}, mode) where {T}
    C = SDS.CSR(tril(A))
    opts = Options(pivot_pairs = mode)
    structure = T <: Complex ? "H" : "S"
    t = @elapsed S = SDS.symbolic_analysis(C, structure, 'L'; opts)
    N = SDS.allocate_numeric(S, T)
    SDS.ref_ldlt!(N, S, C.nzval; opts)
    st = SDS.pivot_stats(N)
    L, D, p = SDS.extract_ldlt(S, N)
    err = norm(A[p, p] - L * D * L') / norm(A)
    P = SDS.SymmetricPattern(C, structure; view = 'L')
    pp = SDS.analysis_pairs(P, C.rowptr, C.colval, C.nzval, C.nrows, structure, opts; view = 'L', index = C.index)
    ncand = pp.candidates === nothing ? count(SDS.pivot_candidates(P, C, structure; view = 'L').candidate) :
            count(pp.candidates.candidate)
    npairs = length(SDS.compute_ordering(P, opts; T, pp.pairs, pp.candidates).pairs)
    return (; npairs, ncand, nnz_L = S.partition.nnz_L, nsn = SDS.nsupernodes(S.partition), nzero = st.nzero,
            nperturbed = st.nperturbed, n2x2 = st.n2x2, maxL = Float64(maximum(abs, L)), err, t)
end

# the public solver with matching (T21): scaled factor quality and the residual of A
function measure_matching(A::SparseMatrixCSC{T}, mode) where {T}
    structure = T <: Complex ? "H" : "S"
    solver = DirectSolver(SDS.CSR(tril(A)), structure, 'L')
    setparam!(solver, "matching_alg", "algo5")
    setparam!(solver, "pivot_pairs", mode)
    t = @elapsed analyze!(solver)
    factorize!(solver)
    st = getparam(solver, "pivot_stats")
    L, D, p = SDS.extract_ldlt(solver.host_symbolic, solver.numeric)
    d = getparam(solver, "scale_row")
    As = Diagonal(d) * A * Diagonal(d)
    err = norm(As[p, p] - L * D * L') / norm(As)
    b = A * ones(T, size(A, 1))
    x = zeros(T, size(A, 1))
    solve!(solver, x, b)
    rel0 = norm(b - A * x) / norm(b)
    setparam!(solver, "ir_n_steps", 5)
    solve!(solver, x, b)
    rel5 = norm(b - A * x) / norm(b)
    return (; npairs = length(solver.ordering.pairs), nnz_L = solver.host_symbolic.partition.nnz_L,
            nsn = SDS.nsupernodes(solver.host_symbolic), nzero = st.nzero, nperturbed = st.nperturbed, n2x2 = st.n2x2,
            maxL = Float64(maximum(abs, L)), err, rel0, rel5, t)
end

function report(name, A)
    rows = [mode => measure(A, mode) for mode in MODES]
    base = last(rows[1]).nnz_L
    println("\n### ", name, "  (n = ", size(A, 1), ", candidates = ", last(rows[1]).ncand, ")\n")
    println("| pivot_pairs | pairs | nnz(L) | ratio | supernodes | zero | perturbed | 2×2 | max abs L | error | analysis s |")
    println("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for (mode, r) in rows
        println("| ", mode, " | ", r.npairs, " | ", r.nnz_L, " | ", round(r.nnz_L / base; digits = 3), " | ", r.nsn, " | ",
                r.nzero, " | ", r.nperturbed, " | ", r.n2x2, " | ", round(r.maxL; sigdigits = 2), " | ",
                round(r.err; sigdigits = 2), " | ", round(r.t; digits = 3), " |")
    end
    println("\n| matching/pivot_pairs | pairs | nnz(L) | ratio | supernodes | zero | perturbed | 2×2 | max abs L | error | ",
            "relres | relres+5 | analysis s |")
    println("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for mode in ("none", "default", "all")
        r = measure_matching(A, mode)
        println("| algo5/", mode, " | ", r.npairs, " | ", r.nnz_L, " | ", round(r.nnz_L / base; digits = 3), " | ", r.nsn,
                " | ", r.nzero, " | ", r.nperturbed, " | ", r.n2x2, " | ", round(r.maxL; sigdigits = 2), " | ",
                round(r.err; sigdigits = 2), " | ", round(r.rel0; sigdigits = 2), " | ", round(r.rel5; sigdigits = 2),
                " | ", round(r.t; digits = 3), " |")
    end
end

selected(name) = isempty(OPTS["only"]) || any(s -> occursin(s, name), split(OPTS["only"], ','))

if OPTS["generators"] == "true"
    for (name, make) in (("kkt_matrix(300, 100, 1e-8)", () -> kkt_matrix(Float64, 300, 100, 1.0e-8)),
                         ("kkt_matrix(200, 100, 0; indefinite)",
                          () -> kkt_matrix(Float64, 200, 100, 0.0; hessian = :indefinite)),
                         ("kkt_slack_matrix(200, 100, 0)", () -> kkt_slack_matrix(Float64, 200, 100, 0.0)))
        selected(name) || continue
        Random.seed!(666)
        report(name, make())
    end
end

dumps = isdir(BenchMatrices.DATA_DIR) ?
        sort([f for f in readdir(BenchMatrices.DATA_DIR) if (d = BenchMatrices.parse_dump_name(f)) !== nothing &&
              d.kind == "k2" && d.ext == "mtx"]) : String[]
isempty(dumps) && println("\nno K2 dumps in ", BenchMatrices.DATA_DIR, " (run bench/dump_madnlp_kkt.jl)")
for f in dumps
    selected(f) || continue
    A = BenchMatrices.symmetrize_triangle(tril(BenchMatrices.read_mtx(joinpath(BenchMatrices.DATA_DIR, f))))
    report(f, A)
end
