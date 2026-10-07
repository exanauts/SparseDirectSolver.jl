# Option tables, setparam!/getparam on Options, enums and their spellings,
# error types (T01).

# Literal copies of ../CUDSS.jl/src/types.jl (CUDSS.jl v0.8.1).
const CUDSS_JL_DATA_PARAMETERS = ("info", "lu_nnz", "npivots", "inertia", "perm_reorder_row",
                                  "perm_reorder_col", "perm_row", "perm_col", "diag", "user_perm",
                                  "hybrid_device_memory_min", "comm_device", "comm_host", "memory_estimates",
                                  "perm_matching", "scale_row", "scale_col", "nsuperpanels",
                                  "user_schur_indices", "schur_shape", "schur_matrix",
                                  "user_nd_partition_tree", "nd_partition_tree", "user_host_interrupt")

const CUDSS_JL_CONFIG_PARAMETERS = ("reordering_alg", "factorization_alg", "solve_alg",
                                    "matching_alg", "solve_mode", "ir_n_steps", "ir_tol", "pivot_type",
                                    "pivot_threshold", "pivot_epsilon", "max_lu_nnz", "hybrid_memory_mode",
                                    "hybrid_device_memory_limit", "use_cuda_register_memory", "host_nthreads",
                                    "hybrid_execute_mode", "pivot_epsilon_alg", "nd_nlevels", "ubatch_size",
                                    "ubatch_index", "use_superpanels", "device_count", "device_indices",
                                    "schur_mode", "deterministic_mode", "nd_ubfactor")

# Configuration parameters PLAN §1.3/§1.7 mark port or reinterpret:
# name => (valid value, value of the wrong type).
const PORTED_CONFIG_VALUES = Dict{String, Tuple{Any, Any}}(
    "reordering_alg" => ("algo3", 3.0),
    "factorization_alg" => ("algo1", :algo1),
    "solve_alg" => ("algo1", 1.0),
    "matching_alg" => ("algo5", 'x'),
    "solve_mode" => (2, "2"),
    "ir_n_steps" => (3, 1.5),
    "ir_tol" => (1.0e-10, "small"),
    "pivot_type" => ('B', "B"),
    "pivot_threshold" => (0.1, 'x'),
    "pivot_epsilon" => (1.0e-8, "tiny"),
    "max_lu_nnz" => (10, 10.0),
    "use_cuda_register_memory" => (0, 0.5),
    "host_nthreads" => (4, "4"),
    "pivot_epsilon_alg" => ("algo2", 2.0),
    "nd_nlevels" => (5, 5.0),
    "ubatch_size" => (4, "4"),
    "ubatch_index" => (2, 2.5),
    "use_superpanels" => (0, "off"),
    "schur_mode" => (1, "on"),
    "deterministic_mode" => (1, 1.0),
    "nd_ubfactor" => (30, 0.3),
    # PLAN §1.7
    "pivot_sign" => (Int8[1, -1, 0, 1], [1.0, -1.0]),
    "ir_mode" => ("fgmres", :fgmres),
    "factor_precision" => (Float32, "Float32"),
    "amalgamation" => ((max_width = 16, zero_fraction = 0.1, min_width = 4), 0.25),
    "schedule" => ("subtree+level", :auto),
    "pivot_pairs" => ("none", :none),
    "pivot_pair_tolerance" => (1.0e-4, "small"),
)

const DEFERRED_CONFIG = ("hybrid_memory_mode", "hybrid_device_memory_limit", "hybrid_execute_mode")
const NOT_PLANNED = ("device_count", "device_indices", "comm_device", "comm_host")
const USER_DATA = ("user_perm", "user_schur_indices", "user_nd_partition_tree", "user_host_interrupt",
                   "ubatch_mask", "pivot_sign")

@testset "parameter tables" begin
    @test CONFIG_PARAMETERS == CUDSS_JL_CONFIG_PARAMETERS
    @test DATA_PARAMETERS == CUDSS_JL_DATA_PARAMETERS
    @test EXTRA_PARAMETERS == ("pivot_sign", "pivot_stats", "ir_mode", "factor_precision",
                               "amalgamation", "schedule", "pivot_pairs", "pivot_pair_tolerance")
    @test CUDSS08_DATA_PARAMETERS == ("ir_n_steps", "ubatch_mask", "flops")
    # every listed name is known to setparam!/getparam
    for name in (CONFIG_PARAMETERS..., DATA_PARAMETERS..., CUDSS08_DATA_PARAMETERS..., EXTRA_PARAMETERS...)
        @test haskey(SDS.PARAMETER_SPECS, name)
    end
    # the round-trip table covers exactly the port/reinterpret configuration names
    config_like = setdiff([CONFIG_PARAMETERS..., EXTRA_PARAMETERS...],
                          [DEFERRED_CONFIG..., NOT_PLANNED..., "pivot_stats"])
    @test Set(keys(PORTED_CONFIG_VALUES)) == Set(config_like)
end

@testset "defaults" begin
    opts = Options()
    @test opts.ir_n_steps == 0
    @test opts.pivot_type == SDS.PIVOT_AUTO
    @test opts.use_superpanels == 1
    @test opts.deterministic_mode == 0
    @test opts.schedule == SDS.SCHEDULE_AUTO
    @test opts.factor_precision === nothing
    @test getparam(opts, "schedule") == "auto"
    @test getparam(opts, "pivot_type") == 'A'
    @test getparam(opts, "reordering_alg") == "default"
    @test getparam(opts, "matching_alg") == "default"
    @test getparam(opts, "ir_mode") == "ir"
    @test getparam(opts, "pivot_epsilon") === nothing
    @test getparam(opts, "amalgamation") == (max_width = 32, zero_fraction = 0.25, min_width = 8)
    @test getparam(opts, "device_count") == 1
    for name in USER_DATA
        @test getparam(opts, name) === nothing
    end
    @test default_pivot_epsilon(Float32) == 1.0e-5
    @test default_pivot_epsilon(Float64) == 1.0e-13
    @test default_pivot_epsilon(ComplexF32) == 1.0e-5
    @test default_pivot_epsilon(ComplexF64) == 1.0e-13
    @test SDS.resolved_pivot_epsilon(opts, Float64) == 1.0e-13
    setparam!(opts, "pivot_epsilon", 1.0e-9)
    @test SDS.resolved_pivot_epsilon(opts, Float32) == 1.0e-9
    setparam!(opts, "pivot_epsilon", nothing)
    @test SDS.resolved_pivot_epsilon(opts, Float32) == 1.0e-5
end

@testset "round trip and wrong types: $name" for name in sort!(collect(keys(PORTED_CONFIG_VALUES)))
    valid, wrong = PORTED_CONFIG_VALUES[name]
    opts = Options()
    @test setparam!(opts, name, valid) === nothing
    @test getparam(opts, name) == valid
    @test thrown(() -> setparam!(opts, name, wrong)) isa InvalidValueError
    @test getparam(opts, name) == valid  # a rejected value leaves the parameter unchanged
end

@testset "values out of range" begin
    opts = Options()
    for (name, value) in (("solve_mode", 3), ("ir_n_steps", -1), ("ir_tol", -1.0), ("ir_tol", NaN),
                          ("pivot_threshold", Inf), ("pivot_epsilon", -1.0e-8), ("pivot_type", 'X'),
                          ("reordering_alg", "algo6"), ("reordering_alg", 6), ("factorization_alg", "algo3"),
                          ("solve_alg", "algo2"), ("matching_alg", "algo7"), ("pivot_epsilon_alg", "algo3"),
                          ("use_superpanels", 2), ("ubatch_index", -2), ("ubatch_size", -1),
                          ("nd_ubfactor", -2), ("schedule", "fast"), ("ir_mode", "gmres"),
                          ("amalgamation", (min_width = 64,)), ("amalgamation", (zero_fraction = -0.1,)),
                          ("amalgamation", (foo = 1,)), ("pivot_sign", [2, 0]),
                          ("user_schur_indices", [0, 2]), ("ubatch_mask", [1, -1]))
        @test thrown(() -> setparam!(opts, name, value)) isa InvalidValueError
    end
    @test thrown(() -> setparam!(opts, "factor_precision", Float16)) isa NotSupportedError
    @test opts.amalgamation == SDS.DEFAULT_AMALGAMATION
end

@testset "alternative spellings" begin
    opts = Options()
    setparam!(opts, "reordering_alg", 4)
    @test getparam(opts, "reordering_alg") == "algo4"
    setparam!(opts, "matching_alg", SDS.MATCHING_AUTO)
    @test getparam(opts, "matching_alg") == "algo6"
    setparam!(opts, "pivot_type", SDS.PIVOT_DIAGONAL)
    @test getparam(opts, "pivot_type") == 'D'
    for c in ('A', 'N', 'D', 'L', 'B')
        setparam!(opts, "pivot_type", c)
        @test getparam(opts, "pivot_type") == c
    end
    setparam!(opts, "schedule", "syncfree")
    @test opts.schedule == SDS.SCHEDULE_SYNCFREE
    setparam!(opts, "factor_precision", Float64)
    @test getparam(opts, "factor_precision") === Float64
    setparam!(opts, "factor_precision", nothing)
    @test getparam(opts, "factor_precision") === nothing
    # amalgamation merges partial updates
    setparam!(opts, "amalgamation", (zero_fraction = 0.5,))
    setparam!(opts, "amalgamation", (max_width = Int32(16),))
    @test getparam(opts, "amalgamation") == (max_width = 16, zero_fraction = 0.5, min_width = 8)
    @test opts.amalgamation isa SDS.AmalgamationParams
end

@testset "user-provided data parameters" begin
    opts = Options()
    for backend in BACKENDS
        setparam!(opts, "user_perm", to_device(backend, Int32[3, 1, 2]))
        @test getparam(opts, "user_perm") == [3, 1, 2]
        @test getparam(opts, "user_perm") isa Vector{Int}
    end
    setparam!(opts, "user_perm", [2, 0, 1])  # 0-based permutations are stored as given
    @test getparam(opts, "user_perm") == [2, 0, 1]
    getparam(opts, "user_perm")[1] = 99      # getparam returns a copy
    @test getparam(opts, "user_perm") == [2, 0, 1]
    setparam!(opts, "user_perm", nothing)
    @test getparam(opts, "user_perm") === nothing
    setparam!(opts, "user_schur_indices", Int32[0, 0, 1, 1, 1])
    @test getparam(opts, "user_schur_indices") == [0, 0, 1, 1, 1]
    setparam!(opts, "user_nd_partition_tree", [1, 2, 3])
    @test getparam(opts, "user_nd_partition_tree") == [1, 2, 3]
    setparam!(opts, "ubatch_mask", [1, 0, 1])
    @test getparam(opts, "ubatch_mask") == [1, 0, 1]
    flag = Threads.Atomic{Bool}(false)
    setparam!(opts, "user_host_interrupt", flag)
    @test getparam(opts, "user_host_interrupt") === flag
    setparam!(opts, "pivot_sign", [1, -1, 0])
    @test getparam(opts, "pivot_sign") == Int8[1, -1, 0]
    @test getparam(opts, "pivot_sign") isa Vector{Int8}
    for (name, wrong) in (("user_perm", [1.0, 2.0]), ("user_perm", 3), ("user_schur_indices", "1"),
                          ("user_nd_partition_tree", (1, 2)), ("user_host_interrupt", Ref(true)),
                          ("user_host_interrupt", true), ("ubatch_mask", [true, 1.5]))
        @test thrown(() -> setparam!(opts, name, wrong)) isa InvalidValueError
    end
end

@testset "deferred parameters warn once and are stored" begin
    for (name, value) in (("hybrid_memory_mode", 1), ("hybrid_device_memory_limit", 2048),
                          ("hybrid_execute_mode", 1))
        opts = Options()
        @test_logs (:warn, r"no effect yet") setparam!(opts, name, value)
        @test getparam(opts, name) == value
        @test_logs setparam!(opts, name, 0)  # the default value is silent
        @test thrown(() -> setparam!(opts, name, "1")) isa InvalidValueError
    end
    for algo in ("algo1", "algo2")
        opts = Options()
        @test_logs (:warn, r"COLAMD") setparam!(opts, "reordering_alg", algo)
        @test getparam(opts, "reordering_alg") == algo
    end
end

@testset "not supported" begin
    opts = Options()
    @test thrown(() -> setparam!(opts, "device_count", 2)) isa NotSupportedError
    @test thrown(() -> setparam!(opts, "device_indices", Cint[0, 1])) isa NotSupportedError
    @test thrown(() -> getparam(opts, "device_indices")) isa NotSupportedError
    @test setparam!(opts, "device_count", 1) === nothing  # a single device is what the package does
    @test getparam(opts, "device_count") == 1
    for name in ("comm_device", "comm_host")
        @test thrown(() -> setparam!(opts, name, C_NULL)) isa NotSupportedError
        @test thrown(() -> getparam(opts, name)) isa NotSupportedError
    end
    for c in ('C', 'R')
        @test thrown(() -> setparam!(opts, "pivot_type", c)) isa NotSupportedError
    end
    @test getparam(opts, "pivot_type") == 'A'
end

@testset "unknown names and solver data" begin
    opts = Options()
    for f in (() -> setparam!(opts, "not_a_parameter", 1), () -> getparam(opts, "not_a_parameter"),
              () -> Options(not_a_parameter = 1))
        err = thrown(f)
        @test err isa ArgumentError
        @test occursin("not_a_parameter", err.msg)
    end
    solver_data = setdiff([DATA_PARAMETERS..., CUDSS08_DATA_PARAMETERS..., EXTRA_PARAMETERS...],
                          [USER_DATA..., NOT_PLANNED..., CONFIG_PARAMETERS..., EXTRA_PARAMETERS[3:end]...])
    @test "info" in solver_data && "pivot_stats" in solver_data && "flops" in solver_data
    for name in solver_data
        @test thrown(() -> setparam!(opts, name, 1)) isa ArgumentError
        @test thrown(() -> getparam(opts, name)) isa ArgumentError
    end
end

@testset "Options constructor, copy, show" begin
    opts = Options(ir_n_steps = 2, schedule = "syncfree", pivot_sign = [1, -1])
    @test getparam(opts, "ir_n_steps") == 2
    @test opts.schedule == SDS.SCHEDULE_SYNCFREE
    @test thrown(() -> Options(ir_n_steps = -1)) isa InvalidValueError
    c = copy(opts)
    @test c.pivot_sign == opts.pivot_sign && c.pivot_sign !== opts.pivot_sign
    setparam!(c, "ir_n_steps", 5)
    @test getparam(opts, "ir_n_steps") == 2
    @test sprint(show, Options()) == "Options()"
    @test sprint(show, opts) == "Options(ir_n_steps = 2, schedule = \"syncfree\", pivot_sign = Int8[1, -1])"
end

# spelling => cuDSS integer value, per enum (../CUDSS.jl/src/libcudss.jl and PLAN §1.7)
const ENUM_VALUES = (
    SDS.Structure => ("G" => 0, "S" => 1, "H" => 2, "SPD" => 3, "HPD" => 4),
    SDS.MatrixView => ('F' => 0, 'L' => 1, 'U' => 2),
    SDS.IndexBase => ('Z' => 0, 'O' => 1),
    SDS.Phase => ("reordering" => 1, "symbolic_factorization" => 2, "analysis" => 3,
                  "factorization" => 4, "refactorization" => 8, "solve_fwd_perm" => 16,
                  "solve_fwd" => 32, "solve_diag" => 64, "solve_bwd" => 128, "solve_bwd_perm" => 256,
                  "solve_refinement" => 512, "solve" => 1008, "solve_fwd_schur" => 48,
                  "solve_bwd_schur" => 384),
    SDS.PivotType => ('A' => 0, 'N' => 1, 'C' => 2, 'R' => 3, 'D' => 4, 'L' => 5, 'B' => 6),
    SDS.ReorderingAlg => ("default" => 0, "algo1" => 1, "algo2" => 2, "algo3" => 3, "algo4" => 4,
                          "algo5" => 5),
    SDS.FactorizationAlg => ("default" => 0, "algo1" => 1, "algo2" => 2),
    SDS.SolveAlg => ("default" => 0, "algo1" => 1),
    SDS.MatchingAlg => ("default" => 0, "algo1" => 1, "algo2" => 2, "algo3" => 3, "algo4" => 4,
                        "algo5" => 5, "algo6" => 6),
    SDS.PivotEpsilonAlg => ("default" => 0, "algo1" => 1, "algo2" => 2),
    SDS.ScheduleKind => ("auto" => 0, "subtree+level" => 1, "syncfree" => 2),
    SDS.IRMode => ("ir" => 0, "fgmres" => 1),
    SDS.PivotPairsMode => ("default" => 0, "none" => 1, "all" => 2),
)

@testset "enum spellings: $(nameof(E))" for (E, table) in ENUM_VALUES
    S = first(first(table)) isa Char ? Char : String
    @test length(table) == length(instances(E))
    for (spelling, value) in table
        x = convert(E, spelling)
        @test x isa E
        @test Int(x) == value
        @test convert(S, x) == spelling
    end
    bad = S === Char ? 'X' : "XYZ"
    @test thrown(() -> convert(E, bad)) isa InvalidValueError
end

@testset "phase bits" begin
    for part in (SDS.PHASE_SOLVE_FWD_PERM, SDS.PHASE_SOLVE_FWD, SDS.PHASE_SOLVE_DIAG,
                 SDS.PHASE_SOLVE_BWD, SDS.PHASE_SOLVE_BWD_PERM, SDS.PHASE_SOLVE_REFINEMENT)
        @test SDS.phase_includes(SDS.PHASE_SOLVE, part)
        @test !SDS.phase_includes(SDS.PHASE_ANALYSIS, part)
    end
    @test SDS.phase_includes(SDS.PHASE_ANALYSIS, SDS.PHASE_REORDERING)
    @test SDS.phase_includes(SDS.PHASE_ANALYSIS, SDS.PHASE_SYMBOLIC_FACTORIZATION)
    @test SDS.phase_includes(SDS.PHASE_SOLVE_FWD_SCHUR, SDS.PHASE_SOLVE_FWD)
    @test !SDS.phase_includes(SDS.PHASE_SOLVE_FWD_SCHUR, SDS.PHASE_SOLVE_DIAG)
    @test SDS.phase_includes(SDS.PHASE_SOLVE_BWD_SCHUR, SDS.PHASE_SOLVE_BWD_PERM)
end

@testset "error types" begin
    for E in (NotSupportedError, InvalidValueError, InterruptedError)
        @test E <: SparseDirectSolverError
        @test occursin("boom", sprint(showerror, E("boom")))
    end
    @test FactorizationError <: SparseDirectSolverError
    e = FactorizationError(3)
    @test e.info == 3
    @test sprint(showerror, e) == "FactorizationError: info = 3"
    @test occursin("not factorized", sprint(showerror, FactorizationError(0, "not factorized")))
    @test FactorizationError([0, 2, 0]).info == [0, 2, 0]
    @test occursin("user_host_interrupt", sprint(showerror, InterruptedError()))
end
