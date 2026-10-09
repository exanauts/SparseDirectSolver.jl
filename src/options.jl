# Parameter tables and the `Options` container behind `setparam!`/`getparam`
# (PLAN §1.3, §1.4, §1.7, §3.1).
#
# `setparam!` validates the *type and range* of a value and stores it in a typed
# field. Whether a phase implements a stored value is checked by that phase, so
# the tables below only distinguish what can be stored, what is permanently
# unsupported, and what is computed by a solver.

"""
    CONFIG_PARAMETERS

The configuration parameter names of CUDSS.jl (`CUDSS_CONFIG_PARAMETERS`), verbatim.
"""
const CONFIG_PARAMETERS = ("reordering_alg", "factorization_alg", "solve_alg",
                           "matching_alg", "solve_mode", "ir_n_steps", "ir_tol", "pivot_type",
                           "pivot_threshold", "pivot_epsilon", "max_lu_nnz", "hybrid_memory_mode",
                           "hybrid_device_memory_limit", "use_cuda_register_memory", "host_nthreads",
                           "hybrid_execute_mode", "pivot_epsilon_alg", "nd_nlevels", "ubatch_size",
                           "ubatch_index", "use_superpanels", "device_count", "device_indices",
                           "schur_mode", "deterministic_mode", "nd_ubfactor")

"""
    DATA_PARAMETERS

The data parameter names of CUDSS.jl (`CUDSS_DATA_PARAMETERS`), verbatim.
"""
const DATA_PARAMETERS = ("info", "lu_nnz", "npivots", "inertia", "perm_reorder_row",
                         "perm_reorder_col", "perm_row", "perm_col", "diag", "user_perm",
                         "hybrid_device_memory_min", "comm_device", "comm_host", "memory_estimates",
                         "perm_matching", "scale_row", "scale_col", "nsuperpanels",
                         "user_schur_indices", "schur_shape", "schur_matrix",
                         "user_nd_partition_tree", "nd_partition_tree", "user_host_interrupt")

"""
    CUDSS08_DATA_PARAMETERS

cuDSS 0.8 data parameters that CUDSS.jl does not list in `DATA_PARAMETERS`
(PLAN §1.4): `"ir_n_steps"` (refinement steps performed; the configuration
parameter of the same name is the number requested), `"ubatch_mask"` and `"flops"`.
"""
const CUDSS08_DATA_PARAMETERS = ("ir_n_steps", "ubatch_mask", "flops")

"""
    EXTRA_PARAMETERS

Parameters beyond cuDSS (PLAN §1.7): the data parameters `"pivot_sign"` (input)
and `"pivot_stats"` (output), and the configuration parameters `"ir_mode"`,
`"factor_precision"`, `"amalgamation"`, `"schedule"`, `"pivot_pairs"` and
`"pivot_pair_tolerance"`.
"""
const EXTRA_PARAMETERS = ("pivot_sign", "pivot_stats", "ir_mode", "factor_precision",
                          "amalgamation", "schedule", "pivot_pairs", "pivot_pair_tolerance")

"""
    default_pivot_epsilon(T) -> Float64

Default `pivot_epsilon` for element type `T`, as in cuDSS: `1e-5` for
`Float32`/`ComplexF32`, `1e-13` for `Float64`/`ComplexF64`.
"""
default_pivot_epsilon(::Type{Float32}) = 1.0e-5
default_pivot_epsilon(::Type{Float64}) = 1.0e-13
default_pivot_epsilon(::Type{Complex{R}}) where {R <: Union{Float32, Float64}} = default_pivot_epsilon(R)

"""
    AmalgamationParams

Type of the `"amalgamation"` parameter: `(max_width, zero_fraction, min_width)`,
the relaxed-amalgamation limits of PLAN §2.3 step 4 (maximum width of a
*merged* panel; a fundamental supernode wider than that stays whole, maximum
fraction of explicit zeros added to the factor, target minimum width).
"""
const AmalgamationParams = @NamedTuple{max_width::Int, zero_fraction::Float64, min_width::Int}

const DEFAULT_AMALGAMATION = AmalgamationParams((32, 0.25, 8))

# regime A local-memory budgets (bytes): 16, 32 and 48 KiB (PLAN §2.3 step 5), clamped to the backend's
# `max_local_bytes` by the analysis (`resolve_subtree_budgets`)
const DEFAULT_SUBTREE_BUDGETS = [16 * 1024, 32 * 1024, 48 * 1024]

# Regime A runs a subtree on one workgroup. A tree of small fronts fits the
# local-memory budgets as a whole and then ran on one workgroup of one SM: the
# pglib KKT dumps were factored 10-20x slower than with regime A off
# (PERFORMANCE.md, experiment 0; bench/profile/phase_split.md). A subtree may do
# at most `1/SUBTREE_PARALLELISM` of the factorization flops. 4096 was the best
# or near-best value of a sweep over 128..4096 on the harness (RTX 4080); it
# leaves the SuiteSparse and Laplacian subtrees as they were.
"Default `subtree_parallelism`: a regime-A subtree does at most this fraction (inverse) of the factorization flops."
const SUBTREE_PARALLELISM = 4096

# `Options` fields that are analysis tuning knobs, not parameter strings (T07)
const TUNING_OPTIONS = (:regime_c_width, :regime_c_rows, :subtree_budgets, :subtree_parallelism,
                        :subtree_max_fronts,
                        :memory_budget)

"""
    Options(; kwargs...)

Every configuration parameter of PLAN §1.3 and §1.7 plus the user-provided data
parameters (`user_perm`, `user_schur_indices`, `user_nd_partition_tree`,
`user_host_interrupt`, `ubatch_mask`, `pivot_sign`), in typed fields with the
defaults below. Keyword arguments are applied through [`setparam!`](@ref), so
`Options(ir_n_steps = 2)` validates like `setparam!(opts, "ir_n_steps", 2)`.

| Field | Default | Meaning of the default |
| --- | --- | --- |
| `reordering_alg` | `REORDERING_DEFAULT` | automatic AMD/ND choice |
| `factorization_alg` | `FACTORIZATION_DEFAULT` | automatic regime choice |
| `solve_alg` | `SOLVE_DEFAULT` | level-batched sweeps |
| `matching_alg` | `MATCHING_NONE` | no matching |
| `solve_mode` | `0` | solve with `A` (1: `Aᵀ`, 2: `Aᴴ`) |
| `ir_n_steps` | `0` | no refinement (cuDSS parity) |
| `ir_tol` | `0.0` | no early exit |
| `pivot_type` | `PIVOT_AUTO` | Bunch–Kaufman for `S`/`H`, none for `SPD`/`HPD` |
| `pivot_threshold` | `0.01` | pivot acceptance threshold; with `pivot_pairs = "default"` also the partner threshold of the 2×2 pairs chosen at analysis (a value set after `"analysis"` changes the in-front test, not the pairs already chosen) |
| `pivot_epsilon` | `nothing` | [`default_pivot_epsilon`](@ref)`(T)` |
| `pivot_epsilon_alg` | `PIVOT_EPSILON_DEFAULT` | |
| `max_lu_nnz` | `-1` | no limit (any negative value) |
| `hybrid_memory_mode` | `0` | device memory only |
| `hybrid_device_memory_limit` | `0` | automatic |
| `use_cuda_register_memory` | `1` | pinned host memory allowed |
| `host_nthreads` | `0` | `Threads.nthreads()` |
| `hybrid_execute_mode` | `0` | device execution only |
| `nd_nlevels` | `10` | minimum nested-dissection levels |
| `ubatch_size` | `0` | deduced from the matrix values (1 for a single matrix) |
| `ubatch_index` | `-1` | all batch members |
| `use_superpanels` | `1` | supernode amalgamation on |
| `schur_mode` | `0` | off |
| `deterministic_mode` | `0` | atomic forward solve allowed |
| `nd_ubfactor` | `-1` | ordering library default |
| `ir_mode` | `IR_PLAIN` | plain iterative refinement |
| `factor_precision` | `nothing` | factors in the input precision |
| `amalgamation` | `(max_width = 32, zero_fraction = 0.25, min_width = 8)` | |
| `schedule` | `SCHEDULE_AUTO` | |
| `pivot_pairs` | `PIVOT_PAIRS_DEFAULT` | 2×2 pivot pairs in the analysis of `"S"`/`"H"` (structurally zero pivots; `"all"`: every candidate; with matching `"algo5"`/`"algo6"`: from the cycles of the matching); decided from the values present at analysis: an all-zero `nzval` gives no pairs, an undefined one arbitrary pairs |
| `pivot_pair_tolerance` | `1e-6` ([`PIVOT_PAIR_TOLERANCE`](@ref)) | relative diagonal size below which a row is a 2×2 candidate |
| `regime_c_width` | `64` | fronts wider than this go to regime C (vendor dense calls) |
| `regime_c_rows` | `512` | fronts with more rows than this go to regime C |
| `subtree_budgets` | `[16384, 32768, 49152]` | regime A local-memory budgets in bytes (kernel classes of 8 to 64 KiB); the defaults are clamped to the backend's local memory per workgroup, an explicit budget above it is an `InvalidValueError`; empty disables regime A |
| `subtree_parallelism` | `4096` | a regime-A subtree does at most `1/subtree_parallelism` of the factorization flops (`0`: no limit) |
| `subtree_max_fronts` | `0` | a regime-A subtree has at most this many fronts (`0`: no limit); the serial walk of a subtree is its workgroup's critical path |
| `memory_budget` | `-1` | update-stack bytes per level chunk (negative: no limit, no chunking) |
| `user_perm`, `user_schur_indices`, `user_nd_partition_tree`, `ubatch_mask`, `pivot_sign` | `nothing` | not provided |
| `user_host_interrupt` | `nothing` | not provided |

Vectors are stored as host copies; `user_host_interrupt` is stored by reference
because it is polled while a phase runs.

The last five rows are analysis tuning knobs of the schedule (PLAN §2.3 step 5,
TASKS T07). They are not cuDSS parameter strings, so [`setparam!`](@ref) does not
know them; set them with `Options(; regime_c_width = 128)` (validated) or by
assigning the field.
"""
mutable struct Options
    # configuration parameters of CUDSS.jl (PLAN §1.3)
    reordering_alg::ReorderingAlg
    factorization_alg::FactorizationAlg
    solve_alg::SolveAlg
    matching_alg::MatchingAlg
    solve_mode::Int
    ir_n_steps::Int
    ir_tol::Float64
    pivot_type::PivotType
    pivot_threshold::Float64
    pivot_epsilon::Union{Nothing, Float64}
    pivot_epsilon_alg::PivotEpsilonAlg
    max_lu_nnz::Int64
    hybrid_memory_mode::Int
    hybrid_device_memory_limit::Int64
    use_cuda_register_memory::Int
    host_nthreads::Int
    hybrid_execute_mode::Int
    nd_nlevels::Int
    ubatch_size::Int
    ubatch_index::Int
    use_superpanels::Int
    schur_mode::Int
    deterministic_mode::Int
    nd_ubfactor::Int
    # configuration parameters beyond cuDSS (PLAN §1.7)
    ir_mode::IRMode
    factor_precision::Union{Nothing, DataType}
    amalgamation::AmalgamationParams
    schedule::ScheduleKind
    pivot_pairs::PivotPairsMode
    pivot_pair_tolerance::Float64
    # analysis tuning knobs of the schedule (T07; not parameter strings)
    regime_c_width::Int
    regime_c_rows::Int
    subtree_budgets::Vector{Int}
    subtree_parallelism::Int
    subtree_max_fronts::Int
    memory_budget::Int64
    # user-provided data parameters (inputs of the phases)
    user_perm::Union{Nothing, Vector{Int}}
    user_schur_indices::Union{Nothing, Vector{Int}}
    user_nd_partition_tree::Union{Nothing, Vector{Int}}
    user_host_interrupt::Union{Nothing, Threads.Atomic{Bool}}
    ubatch_mask::Union{Nothing, Vector{Int}}
    pivot_sign::Union{Nothing, Vector{Int8}}

    function Options(; kwargs...)
        opts = new(
            REORDERING_DEFAULT, FACTORIZATION_DEFAULT, SOLVE_DEFAULT, MATCHING_NONE,
            0, 0, 0.0, PIVOT_AUTO, 0.01, nothing, PIVOT_EPSILON_DEFAULT, -1,
            0, 0, 1, 0, 0, 10, 0, -1, 1, 0, 0, -1,
            IR_PLAIN, nothing, DEFAULT_AMALGAMATION, SCHEDULE_AUTO, PIVOT_PAIRS_DEFAULT, PIVOT_PAIR_TOLERANCE,
            64, 512, copy(DEFAULT_SUBTREE_BUDGETS), SUBTREE_PARALLELISM, 0, -1,
            nothing, nothing, nothing, nothing, nothing, nothing,
        )
        for (name, value) in kwargs
            if name in TUNING_OPTIONS
                setfield!(opts, name, _parse_tuning(Val(name), value))
            else
                setparam!(opts, String(name), value)
            end
        end
        return opts
    end
end

function Base.copy(opts::Options)
    c = Options()
    for f in fieldnames(Options)
        x = getfield(opts, f)
        setfield!(c, f, x isa Vector ? copy(x) : x)
    end
    return c
end

function Base.show(io::IO, opts::Options)
    default = Options()
    changed = (f for f in fieldnames(Options) if !isequal(getfield(opts, f), getfield(default, f)))
    print(io, "Options(")
    join(io, ("$f = $(repr(_format_option(getfield(opts, f))))" for f in changed), ", ")
    print(io, ")")
    return nothing
end

"""
    resolved_pivot_epsilon(opts::Options, T) -> Float64

`opts.pivot_epsilon`, or [`default_pivot_epsilon`](@ref)`(T)` when it was not set.
"""
resolved_pivot_epsilon(opts::Options, ::Type{T}) where {T} =
    something(opts.pivot_epsilon, default_pivot_epsilon(T))

# --- parameter table ---------------------------------------------------------

# Disposition of a parameter name (PLAN §1.3/§1.4/§1.7):
#   :port, :reinterpret  stored in `field`
#   :deferred            stored in `field`; a non-default value warns once (feature not built yet)
#   :not_planned         NotSupportedError (PLAN "not planned")
#   :output              computed by a solver, read-only
#   :solver              data that lives in a solver (`info`, `schur_matrix`)
struct ParameterSpec
    kind::Symbol      # :config or :data
    status::Symbol
    field::Symbol     # Options field, :none if not stored in Options
end

const PARAMETER_SPECS = Dict{String, ParameterSpec}(
    # CUDSS.jl configuration parameters
    "reordering_alg" => ParameterSpec(:config, :reinterpret, :reordering_alg),
    "factorization_alg" => ParameterSpec(:config, :reinterpret, :factorization_alg),
    "solve_alg" => ParameterSpec(:config, :reinterpret, :solve_alg),
    "matching_alg" => ParameterSpec(:config, :port, :matching_alg),
    "solve_mode" => ParameterSpec(:config, :port, :solve_mode),
    "ir_n_steps" => ParameterSpec(:config, :port, :ir_n_steps),
    "ir_tol" => ParameterSpec(:config, :port, :ir_tol),
    "pivot_type" => ParameterSpec(:config, :port, :pivot_type),
    "pivot_threshold" => ParameterSpec(:config, :port, :pivot_threshold),
    "pivot_epsilon" => ParameterSpec(:config, :port, :pivot_epsilon),
    "max_lu_nnz" => ParameterSpec(:config, :port, :max_lu_nnz),
    "hybrid_memory_mode" => ParameterSpec(:config, :deferred, :hybrid_memory_mode),
    "hybrid_device_memory_limit" => ParameterSpec(:config, :deferred, :hybrid_device_memory_limit),
    "use_cuda_register_memory" => ParameterSpec(:config, :reinterpret, :use_cuda_register_memory),
    "host_nthreads" => ParameterSpec(:config, :reinterpret, :host_nthreads),
    "hybrid_execute_mode" => ParameterSpec(:config, :deferred, :hybrid_execute_mode),
    "pivot_epsilon_alg" => ParameterSpec(:config, :port, :pivot_epsilon_alg),
    "nd_nlevels" => ParameterSpec(:config, :port, :nd_nlevels),
    "ubatch_size" => ParameterSpec(:config, :port, :ubatch_size),
    "ubatch_index" => ParameterSpec(:config, :port, :ubatch_index),
    "use_superpanels" => ParameterSpec(:config, :reinterpret, :use_superpanels),
    "device_count" => ParameterSpec(:config, :not_planned, :none),
    "device_indices" => ParameterSpec(:config, :not_planned, :none),
    "schur_mode" => ParameterSpec(:config, :port, :schur_mode),
    "deterministic_mode" => ParameterSpec(:config, :port, :deterministic_mode),
    "nd_ubfactor" => ParameterSpec(:config, :port, :nd_ubfactor),
    # CUDSS.jl data parameters
    "info" => ParameterSpec(:data, :solver, :none),
    "lu_nnz" => ParameterSpec(:data, :output, :none),
    "npivots" => ParameterSpec(:data, :output, :none),
    "inertia" => ParameterSpec(:data, :output, :none),
    "perm_reorder_row" => ParameterSpec(:data, :output, :none),
    "perm_reorder_col" => ParameterSpec(:data, :output, :none),
    "perm_row" => ParameterSpec(:data, :output, :none),
    "perm_col" => ParameterSpec(:data, :output, :none),
    "diag" => ParameterSpec(:data, :output, :none),
    "user_perm" => ParameterSpec(:data, :port, :user_perm),
    "hybrid_device_memory_min" => ParameterSpec(:data, :output, :none),
    "comm_device" => ParameterSpec(:data, :not_planned, :none),
    "comm_host" => ParameterSpec(:data, :not_planned, :none),
    "memory_estimates" => ParameterSpec(:data, :output, :none),
    "perm_matching" => ParameterSpec(:data, :output, :none),
    "scale_row" => ParameterSpec(:data, :output, :none),
    "scale_col" => ParameterSpec(:data, :output, :none),
    "nsuperpanels" => ParameterSpec(:data, :output, :none),
    "user_schur_indices" => ParameterSpec(:data, :port, :user_schur_indices),
    "schur_shape" => ParameterSpec(:data, :output, :none),
    "schur_matrix" => ParameterSpec(:data, :solver, :none),
    "user_nd_partition_tree" => ParameterSpec(:data, :port, :user_nd_partition_tree),
    "nd_partition_tree" => ParameterSpec(:data, :output, :none),
    "user_host_interrupt" => ParameterSpec(:data, :port, :user_host_interrupt),
    # cuDSS 0.8 data parameters missing from CUDSS.jl ("ir_n_steps" is listed above as config)
    "ubatch_mask" => ParameterSpec(:data, :port, :ubatch_mask),
    "flops" => ParameterSpec(:data, :output, :none),
    # PLAN §1.7
    "pivot_sign" => ParameterSpec(:data, :port, :pivot_sign),
    "pivot_stats" => ParameterSpec(:data, :output, :none),
    "ir_mode" => ParameterSpec(:config, :port, :ir_mode),
    "factor_precision" => ParameterSpec(:config, :port, :factor_precision),
    "amalgamation" => ParameterSpec(:config, :port, :amalgamation),
    "schedule" => ParameterSpec(:config, :port, :schedule),
    "pivot_pairs" => ParameterSpec(:config, :port, :pivot_pairs),
    "pivot_pair_tolerance" => ParameterSpec(:config, :port, :pivot_pair_tolerance),
)

"""
    parameter_spec(name) -> ParameterSpec

Look up a parameter name; unknown names raise `ArgumentError`.
"""
function parameter_spec(name::AbstractString)
    spec = get(PARAMETER_SPECS, name, nothing)
    spec === nothing && throw(ArgumentError("unknown data or config parameter \"$name\""))
    return spec
end

# --- value parsing (external value -> stored value) ----------------------------

_invalid(name, value, expected) =
    InvalidValueError("invalid value $(repr(value)) for parameter \"$name\"; expected $expected")

function _parse_int(name, value, lo, hi, expected)
    value isa Integer || throw(_invalid(name, value, expected))
    lo <= value <= hi || throw(_invalid(name, value, expected))
    return Int(value)
end

function _parse_float(name, value)
    expected = "a finite, nonnegative real number"
    (value isa Real && !(value isa Bool)) || throw(_invalid(name, value, expected))
    (isfinite(value) && value >= 0) || throw(_invalid(name, value, expected))
    return Float64(value)
end

function _parse_algorithm(::Type{E}, name, value) where {E <: AlgorithmEnum}
    value isa E && return value
    value isa AbstractString && return convert(E, value)
    if value isa Integer && !(value isa Bool)
        for (_, instance) in enum_spellings(E)
            Int(instance) == value && return instance
        end
    end
    expected = join((repr(first(p)) for p in enum_spellings(E)), ", ")
    throw(_invalid(name, value, "one of $expected"))
end

function _parse_string_enum(::Type{E}, name, value) where {E <: StringEnum}
    value isa E && return value
    value isa AbstractString && return convert(E, value)
    expected = join((repr(first(p)) for p in enum_spellings(E)), ", ")
    throw(_invalid(name, value, "one of $expected"))
end

function _parse_int_vector(name, value; allowed = nothing)
    value === nothing && return nothing
    value isa AbstractVector{<:Integer} || throw(_invalid(name, value, "an integer vector or nothing"))
    v = convert(Vector{Int}, Array(value))
    if allowed !== nothing && !all(in(allowed), v)
        throw(InvalidValueError("parameter \"$name\" only accepts the entries $(Tuple(allowed))"))
    end
    return v
end

# Generic fallback: every stored field has a method below.
function _parse_option end

for (field, E) in ((:reordering_alg, ReorderingAlg), (:factorization_alg, FactorizationAlg),
                   (:solve_alg, SolveAlg), (:matching_alg, MatchingAlg),
                   (:pivot_epsilon_alg, PivotEpsilonAlg))
    @eval _parse_option(::Val{$(QuoteNode(field))}, value, _) =
        _parse_algorithm($E, $(String(field)), value)
end

for (field, E) in ((:schedule, ScheduleKind), (:ir_mode, IRMode), (:pivot_pairs, PivotPairsMode))
    @eval _parse_option(::Val{$(QuoteNode(field))}, value, _) =
        _parse_string_enum($E, $(String(field)), value)
end

# (field, lowest, highest, description of the admissible values)
for (field, lo, hi, expected) in (
        (:solve_mode, 0, 2, "0 (A), 1 (Aᵀ) or 2 (Aᴴ)"),
        (:ir_n_steps, 0, typemax(Int), "an integer ≥ 0"),
        (:max_lu_nnz, typemin(Int64), typemax(Int64), "an integer (negative: no limit)"),
        (:hybrid_memory_mode, 0, 1, "0 or 1"),
        (:hybrid_device_memory_limit, 0, typemax(Int64), "an integer ≥ 0 (0: automatic)"),
        (:use_cuda_register_memory, 0, 1, "0 or 1"),
        (:host_nthreads, 0, typemax(Int), "an integer ≥ 0 (0: Threads.nthreads())"),
        (:hybrid_execute_mode, 0, 1, "0 or 1"),
        (:nd_nlevels, 0, typemax(Int), "an integer ≥ 0"),
        (:ubatch_size, 0, typemax(Int), "an integer ≥ 0 (0: deduced from the matrix)"),
        (:ubatch_index, -1, typemax(Int), "-1 (all members) or a 0-based member index"),
        (:use_superpanels, 0, 1, "0 or 1"),
        (:schur_mode, 0, 1, "0 or 1"),
        (:deterministic_mode, 0, 1, "0 or 1"),
        (:nd_ubfactor, -1, typemax(Int), "-1 (library default) or an integer ≥ 0"),
    )
    @eval _parse_option(::Val{$(QuoteNode(field))}, value, _) =
        _parse_int($(String(field)), value, $lo, $hi, $expected)
end

for field in (:ir_tol, :pivot_threshold, :pivot_pair_tolerance)
    @eval _parse_option(::Val{$(QuoteNode(field))}, value, _) = _parse_float($(String(field)), value)
end

_parse_option(::Val{:pivot_epsilon}, value, _) =
    value === nothing ? nothing : _parse_float("pivot_epsilon", value)

function _parse_option(::Val{:pivot_type}, value, _)
    pivot = if value isa PivotType
        value
    elseif value isa AbstractChar
        convert(PivotType, value)
    else
        throw(_invalid("pivot_type", value, "one of 'A', 'N', 'D', 'L', 'B'"))
    end
    if pivot == PIVOT_GLOBAL_COL || pivot == PIVOT_GLOBAL_ROW
        throw(NotSupportedError("global pivoting (pivot_type = $(repr(convert(Char, pivot)))) is not " *
                                "supported (PLAN §3.3); use 'A', 'N', 'D', 'L' or 'B'"))
    end
    return pivot
end

function _parse_option(::Val{:factor_precision}, value, _)
    (value === nothing || value === Float32 || value === Float64) && return value
    if value isa Type && value <: AbstractFloat
        throw(NotSupportedError("factor_precision = $value is not supported; use Float32, Float64 or nothing"))
    end
    throw(_invalid("factor_precision", value, "Float32, Float64 or nothing"))
end

function _parse_option(::Val{:amalgamation}, value, current)
    expected = "a NamedTuple with keys among (:max_width, :zero_fraction, :min_width)"
    value isa NamedTuple || throw(_invalid("amalgamation", value, expected))
    all(in(keys(DEFAULT_AMALGAMATION)), keys(value)) || throw(_invalid("amalgamation", value, expected))
    p = merge(current, value)
    (p.max_width isa Integer && p.min_width isa Integer && p.zero_fraction isa Real) ||
        throw(_invalid("amalgamation", value, "integer widths and a real zero_fraction"))
    (1 <= p.min_width <= p.max_width) ||
        throw(InvalidValueError("amalgamation: need 1 ≤ min_width ≤ max_width, got $(p.min_width) and $(p.max_width)"))
    (isfinite(p.zero_fraction) && p.zero_fraction >= 0) ||
        throw(InvalidValueError("amalgamation: zero_fraction must be finite and ≥ 0, got $(p.zero_fraction)"))
    return AmalgamationParams((p.max_width, p.zero_fraction, p.min_width))
end

_parse_option(::Val{:user_perm}, value, _) = _parse_int_vector("user_perm", value)
_parse_option(::Val{:user_nd_partition_tree}, value, _) = _parse_int_vector("user_nd_partition_tree", value)
_parse_option(::Val{:user_schur_indices}, value, _) =
    _parse_int_vector("user_schur_indices", value; allowed = (0, 1))
_parse_option(::Val{:ubatch_mask}, value, _) = _parse_int_vector("ubatch_mask", value; allowed = (0, 1))

function _parse_option(::Val{:pivot_sign}, value, _)
    v = _parse_int_vector("pivot_sign", value; allowed = (-1, 0, 1))
    return v === nothing ? nothing : convert(Vector{Int8}, v)
end

function _parse_option(::Val{:user_host_interrupt}, value, _)
    (value === nothing || value isa Threads.Atomic{Bool}) && return value
    throw(_invalid("user_host_interrupt", value, "a Threads.Atomic{Bool} or nothing"))
end

# tuning knobs (keywords of `Options`, not parameter strings)
for field in (:regime_c_width, :regime_c_rows)
    @eval _parse_tuning(::Val{$(QuoteNode(field))}, value) =
        _parse_int($(String(field)), value, 1, typemax(Int), "an integer ≥ 1")
end

_parse_tuning(::Val{:subtree_parallelism}, value) =
    _parse_int("subtree_parallelism", value, 0, typemax(Int), "an integer ≥ 0 (0: no limit)")
_parse_tuning(::Val{:subtree_max_fronts}, value) =
    _parse_int("subtree_max_fronts", value, 0, typemax(Int), "an integer ≥ 0 (0: no limit)")

_parse_tuning(::Val{:memory_budget}, value) =
    Int64(_parse_int("memory_budget", value, typemin(Int64), typemax(Int64), "an integer (negative: no limit)"))

function _parse_tuning(::Val{:subtree_budgets}, value)
    expected = "a vector of positive byte counts (empty: no regime A)"
    (value isa AbstractVector || value isa Tuple) || throw(_invalid("subtree_budgets", value, expected))
    all(x -> x isa Integer && x > 0, value) || throw(_invalid("subtree_budgets", value, expected))
    return sort!(unique!(Vector{Int}(collect(value))))
end

# --- value formatting (stored value -> value returned by getparam) -------------

_format_option(x::StringEnum) = convert(String, x)
_format_option(x::CharEnum) = convert(Char, x)
_format_option(x::Vector) = copy(x)
_format_option(x) = x

# --- setparam! / getparam -------------------------------------------------------

function _warn_deferred(name, value)
    @warn "parameter \"$name\" = $(repr(value)) is accepted but has no effect yet " *
          "(hybrid modes are planned for milestone M12)" maxlog = 1 _id = Symbol("deferred_", name)
    return nothing
end

function _set_not_planned(name, value)
    # A single device is what the package always uses, so asking for it is fine.
    name == "device_count" && isequal(value, 1) && return nothing
    throw(NotSupportedError("parameter \"$name\" is not supported (multi-GPU and MGMN are not planned)"))
end

"""
    setparam!(opts::Options, name::String, value)

Set the configuration or user-input data parameter `name` (a CUDSS.jl name from
[`CONFIG_PARAMETERS`](@ref)/[`DATA_PARAMETERS`](@ref) or a name from
[`EXTRA_PARAMETERS`](@ref)) after validating `value`.

Accepted values:

* algorithms (`"reordering_alg"`, `"factorization_alg"`, `"solve_alg"`,
  `"matching_alg"`, `"pivot_epsilon_alg"`): `"default"`, `"algoN"` or the integer `N`;
* `"pivot_type"`: `'A'`, `'N'`, `'D'`, `'L'`, `'B'` (`'C'`/`'R'` raise `NotSupportedError`);
  LDLᵀ/LDLᴴ: `'A'`, `'B'`, `'L'` Bunch–Kaufman, `'D'` 1×1 only, `'N'` none; LU (`"G"`, [`lu_pivoting`](@ref)):
  `'A'`, `'B'`, `'L'` in-block threshold row pivoting, `'N'`, `'D'` diagonal pivots;
* integer and flag parameters: an `Integer` in the documented range;
* `"ir_tol"`, `"pivot_threshold"`, `"pivot_epsilon"`: a finite real `≥ 0`
  (`nothing` resets `"pivot_epsilon"` to [`default_pivot_epsilon`](@ref));
* `"ir_mode"`: `"ir"` or `"fgmres"`; `"schedule"`: `"auto"`, `"subtree+level"`, `"syncfree"`;
* `"pivot_pairs"` (beyond cuDSS): `"default"` (for `"S"`/`"H"`, a row with a zero
  or negligible diagonal whose pivot would be structurally zero is ordered
  together with a 2×2 pivot partner, see [`zero_pivot_pairs!`](@ref)), `"all"`
  (every such row, see [`pivot_pairs`](@ref); about 2× `nnz(L)` on KKT systems)
  or `"none"`;
  ignored for the other structures, with `user_perm` and with the natural ordering;
  with `matching_alg = "algo5"`/`"algo6"` the pairs come from the cycles of the
  scaled symmetric matching instead ([`matching_pairs`](@ref)).
  Pairs are decided from the values present at `"analysis"` (the first batch
  member): an all-zero `nzval` gives no pairs and an undefined one arbitrary pairs,
  so run `"analysis"` after the first assembly of the matrix (MadNLP: after the
  first KKT assembly); `"all"` reads the values as well (its candidates and
  partners come from them);
* `"pivot_threshold"` also shapes the ordering with `pivot_pairs = "default"`:
  partners are accepted at analysis down to `pivot_threshold · maxₖ |aᵢₖ|`
  ([`zero_pivot_pairs!`](@ref)); a value set after `"analysis"` changes the
  in-front pivot test but not the pairs already chosen;
* `"pivot_pair_tolerance"` (beyond cuDSS): a finite real `≥ 0`, the relative
  diagonal size `τ` below which a row is a 2×2 candidate (`|aᵢᵢ| ≤ τ maxⱼ≠ᵢ |aᵢⱼ|`,
  on the scaled matrix with matching; default [`PIVOT_PAIR_TOLERANCE`](@ref));
* `"factor_precision"`: `Float32`, `Float64` or `nothing`;
* `"amalgamation"`: a `NamedTuple` with any of `max_width`, `zero_fraction`, `min_width`;
* `"user_perm"`, `"user_nd_partition_tree"`: an integer vector (host or device), or `nothing`;
* `"user_schur_indices"`, `"ubatch_mask"`: a vector of 0/1 flags, or `nothing`;
* `"pivot_sign"`: a vector with entries in `(-1, 0, 1)`, or `nothing`;
* `"user_host_interrupt"`: a `Threads.Atomic{Bool}`, or `nothing`.

Unknown names raise `ArgumentError`; values of the wrong type or range raise
[`InvalidValueError`](@ref); `"device_count" ≠ 1`, `"device_indices"`,
`"comm_device"`, `"comm_host"` and global pivoting raise
[`NotSupportedError`](@ref). Parameters computed by a solver (`"lu_nnz"`,
`"inertia"`, …) cannot be set (`ArgumentError`). The hybrid-mode parameters are
stored but have no effect yet; a non-default value warns once.
`reordering_alg = "algo1"`/`"algo2"` (COLAMD-based orderings with global
pivoting) is accepted and warns once that the symmetric-pattern path is used.
"""
function setparam!(opts::Options, name::AbstractString, value)
    spec = parameter_spec(name)
    if spec.status === :not_planned
        _set_not_planned(name, value)
    elseif spec.status === :output
        throw(ArgumentError("the data parameter \"$name\" is computed by the solver and can't be set"))
    elseif spec.status === :solver
        throw(ArgumentError("the data parameter \"$name\" belongs to a solver; use setparam!(solver, \"$name\", value)"))
    else
        x = _parse_option(Val(spec.field), value, getfield(opts, spec.field))
        setfield!(opts, spec.field, x)
        if spec.status === :deferred && !isequal(x, getfield(Options(), spec.field))
            _warn_deferred(name, value)
        end
        if spec.field === :reordering_alg && (x == REORDERING_BTF_COLAMD || x == REORDERING_COLAMD)
            @warn "reordering_alg = \"$(convert(String, x))\" (COLAMD with global pivoting) is not " *
                  "available; the symmetric-pattern ordering and in-front pivoting are used instead " *
                  "(PLAN §3.3)" maxlog = 1
        end
    end
    return nothing
end

"""
    getparam(opts::Options, name::String)

Value of the configuration or user-input data parameter `name`, in the form
[`setparam!`](@ref) accepts: algorithm and mode strings (`"algo3"`, `"fgmres"`),
`Char` for `"pivot_type"`, `Int`/`Float64` for numbers, a copy of stored
vectors, `nothing` for unset optional values (`"pivot_epsilon"`,
`"factor_precision"`, user vectors). `"device_count"` is always `1`.

Unknown names raise `ArgumentError`, as do data parameters computed by a solver
(query those on the solver). `"device_indices"`, `"comm_device"` and
`"comm_host"` raise [`NotSupportedError`](@ref).
"""
function getparam(opts::Options, name::AbstractString)
    spec = parameter_spec(name)
    if spec.status === :not_planned
        name == "device_count" && return 1
        throw(NotSupportedError("parameter \"$name\" is not supported (multi-GPU and MGMN are not planned)"))
    elseif spec.status === :output || spec.status === :solver
        throw(ArgumentError("the data parameter \"$name\" is computed by a solver; query it with getparam(solver, \"$name\")"))
    end
    return _format_option(getfield(opts, spec.field))
end
