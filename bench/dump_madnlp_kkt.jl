# Dumps MadNLP KKT matrices of pglib-opf AC-OPF cases to bench/data/ as
# kkt_<case>_<kind>_<iter>.mtx, kind ∈ {k2, condensed}:
#
#   k2         MadNLP.SparseKKTSystem (augmented K2 system, symmetric indefinite)
#   condensed  MadNLP.SparseCondensedKKTSystem (condensed system, SPD by construction)
#
# at three interior-point iterations each (default 1, 10, 20). The matrices are
# those of the last factorization MadNLP performed when stopped by `max_iter`.
#
#   julia --project=<env with MadNLP, ExaModels, ExaModelsPower> bench/dump_madnlp_kkt.jl \
#         [case ...] [--iters=1,10,20]
#
# Default case: pglib_opf_case118_ieee. The pglib-opf data files are fetched by
# ExaModelsPower (through ExaPowerIO) on first use. Without those packages the
# script prints how to install them and exits with status 0.

const REQUIRED = ("MadNLP", "ExaModels", "ExaModelsPower")

missing_pkgs = [p for p in REQUIRED if Base.find_package(p) === nothing]
if !isempty(missing_pkgs)
    println("""
    dump_madnlp_kkt.jl needs $(join(REQUIRED, ", ")); missing in this environment: $(join(missing_pkgs, ", ")).
    Install them in a separate environment (they are not part of bench/Project.toml), e.g.

        julia --project=bench/kkt -e 'using Pkg; Pkg.add(["MadNLP", "ExaModels", "ExaModelsPower"])'
        julia --project=bench/kkt bench/dump_madnlp_kkt.jl pglib_opf_case118_ieee

    and then run bench/cudss_baseline.jl, which picks up bench/data/kkt_*.mtx.""")
    exit(0)
end

using LinearAlgebra
using SparseArrays
using MadNLP
using ExaModels
using ExaModelsPower

include(joinpath(@__DIR__, "matrices.jl"))
using .BenchMatrices: dump_name, write_mtx, symmetrize_triangle, DATA_DIR

const KINDS = ("k2" => MadNLP.SparseKKTSystem, "condensed" => MadNLP.SparseCondensedKKTSystem)

"""
    kkt_matrix_at(model, kkt_system, iter) -> (A, k)

Runs MadNLP for at most `iter` iterations with the given KKT system type and
returns the full symmetric KKT matrix `A` of the last factorization (MadNLP
stores the lower triangle in `kkt.aug_com`; explicit zeros are kept so the
pattern is the same at every iteration) and the iteration `k` it belongs to
(`k < iter` when MadNLP converged earlier).
"""
function kkt_matrix_at(model, kkt_system, iter::Integer)
    solver = MadNLP.MadNLPSolver(model; kkt_system = kkt_system, max_iter = iter,
                                 print_level = MadNLP.ERROR)
    MadNLP.solve!(solver)
    L = SparseMatrixCSC{Float64,Int}(solver.kkt.aug_com)
    istril(L) || error("expected a lower-triangular aug_com, got a general matrix")
    return symmetrize_triangle(L), solver.cnt.k
end

function main(args = ARGS)
    cases = String[]
    iters = [1, 10, 20]
    for a in args
        if startswith(a, "--iters=")
            iters = parse.(Int, split(a[9:end], ','))
        else
            push!(cases, a)
        end
    end
    isempty(cases) && push!(cases, "pglib_opf_case118_ieee")
    mkpath(DATA_DIR)
    for case in cases
        case = first(splitext(basename(case)))  # accept pglib_opf_x and pglib_opf_x.m
        model, = ExaModelsPower.ac_opf_model(case * ".m")
        for (kind, kkt_system) in KINDS, it in iters
            A, k = kkt_matrix_at(model, kkt_system, it)
            k == it || @warn "$case ($kind): MadNLP stopped at iteration $k < $it"
            path = joinpath(DATA_DIR, dump_name(case, kind, k))
            write_mtx(path, A; symmetric = true)
            println(rpad(basename(path), 50), " n = ", size(A, 1), ", nnz = ", nnz(A))
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
