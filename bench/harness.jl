# Solver-agnostic timing harness shared by the baselines (cuDSS now, the
# SparseDirectSolver phases from T13 on). No GPU package is loaded here: the
# GPU-specific parts (synchronization, statistics) come in through closures.

module BenchHarness

using LinearAlgebra
using SparseArrays
using Statistics: median

export PHASES, time_phases, cholmod_solver, CSV_COLUMNS, csv_row, write_csv

"""
    PHASES

Timed phases, in execution order: `(:analysis, :factorization, :refactorization, :solve)`.
"""
const PHASES = (:analysis, :factorization, :refactorization, :solve)

"""
    time_phases(make_solver, A, b; nruns = 5, nwarmup = 1, synchronize = () -> nothing) -> NamedTuple

Times the phases of a sparse direct solver. `make_solver(A, b)` (not timed) is
called once per run and returns a `NamedTuple` of closures

* `analysis()`, `factorization()`, `refactorization()`: run the phase;
* `solve()`: runs the solve and returns the solution `x` (any array type, it is
  copied to the host with `Array` after timing);
* `stats()` (optional): returns a `NamedTuple` with any of `lu_nnz`, `flops`,
  `nsuperpanels`; missing entries are reported as `missing`.

Each phase is timed as `synchronize(); t = time_ns(); phase(); synchronize()`
so asynchronous GPU work is included. `nwarmup` runs are discarded, then the
median over `nruns` runs is reported.

Returned fields:

* `analysis`, `factorization`, `refactorization`, `solve`: median seconds;
* `samples`: `NamedTuple` of the `nruns` raw times per phase (seconds);
* `nruns`: number of timed runs;
* `lu_nnz`, `flops`, `nsuperpanels`: from `stats()` after the last run, or `missing`;
* `relres`: `‖b - A x‖ / ‖b‖` for the solution of the last run.
"""
function time_phases(make_solver, A::AbstractMatrix, b::AbstractVector;
                     nruns::Integer = 5, nwarmup::Integer = 1, synchronize = () -> nothing)
    nruns ≥ 1 || throw(ArgumentError("nruns must be ≥ 1, got $nruns"))
    nwarmup ≥ 0 || throw(ArgumentError("nwarmup must be ≥ 0, got $nwarmup"))
    samples = Dict(p => Float64[] for p in PHASES)
    x = nothing
    solver = nothing
    for run in 1:(nwarmup + nruns)
        solver = make_solver(A, b)
        for p in PHASES
            f = getproperty(solver, p)
            synchronize()
            t0 = time_ns()
            y = f()
            synchronize()
            t = (time_ns() - t0) / 1e9
            p === :solve && (x = y)
            run > nwarmup && push!(samples[p], t)
        end
    end
    xh = Array(x)
    relres = norm(b - A * xh) / max(norm(b), floatmin(Float64))
    stats = hasproperty(solver, :stats) ? solver.stats() : NamedTuple()
    getstat(k) = haskey(stats, k) ? getproperty(stats, k) : missing
    return (analysis = median(samples[:analysis]),
            factorization = median(samples[:factorization]),
            refactorization = median(samples[:refactorization]),
            solve = median(samples[:solve]),
            samples = NamedTuple{PHASES}(Tuple(samples[p] for p in PHASES)),
            nruns = Int(nruns),
            lu_nnz = getstat(:lu_nnz),
            flops = getstat(:flops),
            nsuperpanels = getstat(:nsuperpanels),
            relres = relres)
end

"""
    cholmod_solver(structure) -> make_solver

CPU reference solver for [`time_phases`](@ref) built on SparseArrays:
CHOLMOD `cholesky` for `"SPD"`, CHOLMOD `ldlt` for `"S"` (no pivoting, so only
for matrices whose LDLᵀ exists), UMFPACK `lu` for `"G"`. CHOLMOD analysis is
the symbolic factorization alone; UMFPACK has no separate analysis entry point
in SparseArrays, so its `analysis` is a full `lu` and `factorization` an `lu!`.
`stats()` reports `lu_nnz` (stored entries of the factor); `flops` and
`nsuperpanels` are not available.
"""
function cholmod_solver(structure::AbstractString)
    structure in ("SPD", "S", "G") ||
        throw(ArgumentError("structure must be \"SPD\", \"S\" or \"G\", got \"$structure\""))
    return function (A, b)
        F = Ref{Any}(nothing)
        if structure == "G"
            analysis = () -> (F[] = lu(A); nothing)
            factorize = () -> (lu!(F[], A); nothing)
            lu_nnz = () -> nnz(F[].L) + nnz(F[].U)
        else
            S = Symmetric(A, :L)
            analysis = () -> (F[] = SparseArrays.CHOLMOD.symbolic(SparseArrays.CHOLMOD.Sparse(S)); nothing)
            factorize = structure == "SPD" ? () -> (cholesky!(F[], S); nothing) :
                                             () -> (ldlt!(F[], S); nothing)
            lu_nnz = () -> nnz(F[])
        end
        return (analysis = analysis, factorization = factorize, refactorization = factorize,
                solve = () -> F[] \ b, stats = () -> (lu_nnz = lu_nnz(),))
    end
end

"""
    CSV_COLUMNS

Columns of the baseline CSV files, one row per `(matrix, structure)`: timings
in seconds (medians), `lu_nnz`, `flops`, `nsuperpanels` (empty when the solver
does not report them), `relres` of the last solve, `status` (`ok` or the error).
"""
const CSV_COLUMNS = ("solver", "matrix", "n", "nnz", "structure", "analysis_s", "factorization_s",
                     "refactorization_s", "solve_s", "lu_nnz", "flops", "nsuperpanels", "relres",
                     "status")

_csv(x) = x === missing || x === nothing ? "" : x isa AbstractFloat ? repr(Float64(x)) : string(x)

"""
    csv_row(solver, matrix, A, structure, result) -> Vector{String}

One CSV row (see [`CSV_COLUMNS`](@ref)) from a [`time_phases`](@ref) result,
or from an exception (`result isa Exception`: empty timings, `status` = error).
"""
function csv_row(solver::AbstractString, matrix::AbstractString, A::AbstractMatrix,
                 structure::AbstractString, result)
    head = [solver, matrix, string(size(A, 1)), string(nnz(A)), structure]
    if result isa Exception
        msg = replace(sprint(showerror, result), r"[\s,\"]+" => " ")
        return vcat(head, fill("", 8), first(msg, 200))
    end
    # A tuple, not a vector: a vector literal would promote the integer statistics to Float64.
    fields = (result.analysis, result.factorization, result.refactorization, result.solve,
              result.lu_nnz, result.flops, result.nsuperpanels, result.relres)
    return vcat(head, collect(String, map(_csv, fields)), "ok")
end

"""
    write_csv(path, rows)

Writes `CSV_COLUMNS` and `rows` (vectors of strings, from [`csv_row`](@ref)).
"""
function write_csv(path::AbstractString, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(CSV_COLUMNS, ','))
        for r in rows
            length(r) == length(CSV_COLUMNS) || error("CSV row has $(length(r)) fields")
            println(io, join(r, ','))
        end
    end
    return path
end

end # module BenchHarness
