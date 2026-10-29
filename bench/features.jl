# Feature table of the cuDSS vs SparseDirectSolver.jl comparison (`compare.jl`,
# `compare_report.jl`): one entry per implemented feature, in TASKS.md order.
# A feature is listed only once the package implements it; the TASKS.md status
# gate (`feature_implemented`) is a safety check that keeps an entry whose task
# is not done out of the SparseDirectSolver run, not a way to list pending ones.
# Plain Float32 factorizations are left out: cuDSS fails on every condensed KKT
# dump in Float32, and single precision is not accurate enough for these
# systems.
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
`nbatch` value sets on one pattern), `:schur` (partial factorization with a
Schur block), `:fused` (a `FusedSolver`: the fused factorization and
partitioned-inverse solve, the inner solver of the MadNLP extension's
`SDSSolver`; cuDSS runs it as a plain `:single` system). `matrices(M)` decides
whether harness matrix `M` belongs to the feature. `params` are set on both
solvers (the names mirror cuDSS), `cudss_params` and `sds_params` on one side
only. `task` is the TASKS.md task id (`"T13"`) or the pull request (`"#126"`)
that implemented the feature.
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
end

# matrix selectors
is_condensed(M) = M.source == :dump && occursin("_condensed_", M.name)
is_k2(M) = M.source == :dump && occursin("_k2_", M.name)
spd(M) = "SPD" in M.structures
sym(M) = "S" in M.structures
unsym(M) = "G" in M.structures

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
    Feature(id = "cholesky_fused", title = "Cholesky, fused pipeline (the `SDSSolver` path)", task = "#126",
            structure = "SPD", matrices = spd, kind = :fused),
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
const TASK_ID = r"^T\d\d$"

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

Whether the task of feature `f` is marked done in TASKS.md. A feature whose
`task` is not a task id (a pull request, `"#126"`) is implemented.
"""
feature_implemented(f::Feature; tasks_md::AbstractString = TASKS_MD) =
    occursin(TASK_ID, f.task) ? task_status(f.task; tasks_md) in ('x', '!') : true

end # module BenchFeatures
