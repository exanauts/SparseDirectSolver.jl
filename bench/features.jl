# Feature table of the cuDSS vs SparseDirectSolver.jl comparison (`compare.jl`,
# `compare_report.jl`): one entry per planned feature, in TASKS.md order. A
# feature counts as implemented when its task is marked done in TASKS.md; until
# then the SparseDirectSolver run skips it and the report leaves its cells blank.
# Plain Float32 factorizations are left out: cuDSS fails on every condensed KKT
# dump in Float32, and single precision is not accurate enough for these
# systems. Float32 factors appear only with Float64 refinement (T27).
#
# Declarative only: no GPU package, no solver package, so the test suite can
# include it. Matrix selectors take anything with `name`, `source` and
# `structures` fields (a `BenchMatrices.BenchMatrix`).

module BenchFeatures

export Feature, COMPARE_FEATURES, feature, task_status, feature_implemented

"""
    Feature(; id, title, task, structure, matrices, kwargs...)

One comparison feature. `structure` is the cuDSS structure string, `T` the
element type, `nrhs` the right-hand sides per system. `kind` selects how the
systems are built: `:single` (one matrix per row), `:ubatch` (uniform batch of
`nbatch` value sets on one pattern), `:nubatch` (all selected matrices as one
non-uniform batch), `:schur` (partial factorization with a Schur block).
`matrices(M)` decides whether harness matrix `M` belongs to the feature.
`params` are set on both solvers (the names mirror cuDSS), `cudss_params` and
`sds_params` on one side only. `cudss_supported = false` marks a feature
without cuDSS counterpart.
"""
Base.@kwdef struct Feature
    id::String
    title::String
    task::String
    structure::String
    matrices::Function
    T::DataType = Float64
    nrhs::Int = 1
    nbatch::Int = 1
    kind::Symbol = :single
    params::Dict{String, Any} = Dict{String, Any}()
    cudss_params::Dict{String, Any} = Dict{String, Any}()
    sds_params::Dict{String, Any} = Dict{String, Any}()
    cudss_supported::Bool = true
end

# matrix selectors
is_condensed(M) = M.source == :dump && occursin("_condensed_", M.name)
is_k2(M) = M.source == :dump && occursin("_k2_", M.name)
spd(M) = "SPD" in M.structures
sym(M) = "S" in M.structures
unsym(M) = "G" in M.structures
case1354_condensed(M) = is_condensed(M) && occursin("case1354", M.name)

const COMPARE_FEATURES = Feature[
    Feature(id = "cholesky_f64", title = "Cholesky, Float64", task = "T13", structure = "SPD", matrices = spd),
    Feature(id = "cholesky_nrhs16", title = "Cholesky, 16 right-hand sides", task = "T12", structure = "SPD",
            matrices = spd, nrhs = 16),
    Feature(id = "ldlt", title = "LDLᵀ, static pivoting", task = "T15", structure = "S", matrices = sym),
    Feature(id = "ldlt_ir2", title = "LDLᵀ + 2 refinement steps, K2 dumps", task = "T16", structure = "S",
            matrices = is_k2, params = Dict{String, Any}("ir_n_steps" => 2)),
    Feature(id = "ubatch8", title = "Uniform batch of 8, Cholesky", task = "T17", structure = "SPD",
            matrices = is_condensed, nbatch = 8, kind = :ubatch),
    Feature(id = "lu", title = "LU, unsymmetric", task = "T19", structure = "G", matrices = unsym),
    Feature(id = "schur", title = "Schur complement, Cholesky", task = "T20", structure = "SPD",
            matrices = is_condensed, kind = :schur, params = Dict{String, Any}("schur_mode" => 1)),
    Feature(id = "ldlt_matching", title = "LDLᵀ + matching (algo5), K2 dumps", task = "T21", structure = "S",
            matrices = is_k2, params = Dict{String, Any}("matching_alg" => "algo5")),
    Feature(id = "nubatch", title = "Non-uniform batch, condensed dumps", task = "T22", structure = "SPD",
            matrices = is_condensed, kind = :nubatch),
    Feature(id = "solve_algo1", title = "Partitioned-inverse solve (SDS algo1, cuDSS default)", task = "T25",
            structure = "SPD", matrices = spd, sds_params = Dict{String, Any}("solve_alg" => "algo1")),
    Feature(id = "hybrid_memory", title = "Hybrid memory mode", task = "T26", structure = "SPD",
            matrices = M -> M.name == "GHS_psdef/apache2" || case1354_condensed(M),
            params = Dict{String, Any}("hybrid_memory_mode" => 1)),
    Feature(id = "mixed_precision", title = "Float32 factors, Float64 refinement", task = "T27",
            structure = "SPD", matrices = spd, sds_params = Dict{String, Any}("factor_precision" => Float32),
            cudss_supported = false),
]

"""
    feature(id) -> Feature

The entry of [`COMPARE_FEATURES`](@ref) with this `id`.
"""
function feature(id::AbstractString)
    i = findfirst(f -> f.id == id, COMPARE_FEATURES)
    i === nothing && error("unknown feature \"$id\"; available: $(join((f.id for f in COMPARE_FEATURES), ", "))")
    return COMPARE_FEATURES[i]
end

const TASKS_MD = joinpath(@__DIR__, "..", "TASKS.md")
const TASK_HEADER = r"^#{2,3} (T\d\d) — .*`\[(.)\]`\s*$"

"""
    task_status(task; tasks_md = TASKS.md) -> Char

Status marker of `task` (`"T13"`) in TASKS.md: `' '` open, `'~'` in progress,
`'x'` or `'!'` done. Errors if the task has no header.
"""
function task_status(task::AbstractString; tasks_md::AbstractString = TASKS_MD)
    for line in eachline(tasks_md)
        m = match(TASK_HEADER, line)
        m !== nothing && m[1] == task && return only(m[2])
    end
    error("$task has no header with a status marker in $tasks_md")
end

"""
    feature_implemented(f; tasks_md = TASKS.md) -> Bool

Whether the task of feature `f` is marked done in TASKS.md.
"""
feature_implemented(f::Feature; tasks_md::AbstractString = TASKS_MD) = task_status(f.task; tasks_md) in ('x', '!')

end # module BenchFeatures
