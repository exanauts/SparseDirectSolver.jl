# Port of `cudss_solver()` with `cudss_solver_data_parameters` and
# `cudss_solver_config_parameters` (CUDSS.jl test/test_cudss.jl): the get/set
# loop over every data and configuration parameter.
#
# Changes:
# * structures "G", "S", "H", "SPD" and "HPD" (complex `T`: "SPD" is checked to
#   be rejected at analysis); the matrix is SPD/HPD, so `"inertia"` is `(n, 0)`
#   for real "S" and "H" (complex symmetric "S": `(0, 0)`, no inertia) and
#   `"diag"` is real positive except for complex "S" (and has a positive real
#   part for "G", whose complex `D` is not exactly real); "G" reads the full
#   matrix: views 'L'/'U' are rejected at analysis (`InvalidValueError`; cuDSS
#   ignores the view of a general matrix);
# * `matching_alg = "algo6"` is set before the analysis, as in CUDSS.jl (T21):
#   `"perm_matching"` is checked to be a permutation and `"scale_row"`/`"scale_col"`
#   to be positive; the inertia stays `(n, 0)` under matching (cuDSS 0.8 reports
#   `(0, 0)`, see `cudss_inertia_matching.jl`); a second solver without matching
#   checks that the matching outputs raise `InvalidValueError` there;
# * cuDSS's set-buffer-then-get protocol for vector data is
#   `getparam!(buffer, solver, name)`; `"hybrid_device_memory_min"` is not
#   implemented yet (`NotSupportedError`);
# * configuration values that the package does not accept raise errors instead
#   of being passed to the library: `factorization_alg`/`pivot_epsilon_alg`
#   `"algo3"`–`"algo5"` and `solve_alg` `"algo2"`–`"algo5"` (`InvalidValueError`,
#   no such algorithm, PLAN §1.3), `pivot_type` `'C'`/`'R'` (`NotSupportedError`,
#   PLAN §3.3); accepted values are read back with `getparam`.

const PORTED_ACCEPTED_ALGOS = Dict(
    "reordering_alg" => ("default", "algo1", "algo2", "algo3", "algo4", "algo5"),
    "matching_alg" => ("default", "algo1", "algo2", "algo3", "algo4", "algo5"),
    "factorization_alg" => ("default", "algo1", "algo2"),
    "solve_alg" => ("default", "algo1"),
    "pivot_epsilon_alg" => ("default", "algo1", "algo2"),
)

const PORTED_NOT_IMPLEMENTED_DATA = ("hybrid_device_memory_min",)

function ported_solver_data_parameters(backend, solver, structure, n, memory_estimates, buffer_int::Vector{INT},
                                       buffer_R, buffer_T) where {INT}
    @testset "data parameter = $parameter" for parameter in DATA_PARAMETERS
        parameter ∈ ("comm_device", "comm_host", "user_schur_indices", "schur_shape", "schur_matrix",
                     "user_nd_partition_tree", "nd_partition_tree", "user_host_interrupt") && continue
        @testset "setparam!" begin
            (parameter == "nsuperpanels") && continue
            if parameter == "user_perm"
                perm_cpu = INT[i for i in n:-1:1]
                setparam!(solver, parameter, perm_cpu)
                @test getparam(solver, parameter) == perm_cpu
                perm_gpu = to_device(backend, perm_cpu)
                setparam!(solver, parameter, perm_gpu)
                @test getparam(solver, parameter) == perm_cpu
            end
            if parameter ∈ ("perm_row", "perm_col", "perm_reorder_row", "perm_reorder_col")
                getparam!(buffer_int, solver, parameter)
                @test isperm(buffer_int)
            end
            if parameter == "perm_matching"
                getparam!(buffer_int, solver, parameter)
                @test isperm(buffer_int)
            end
            if parameter ∈ ("scale_row", "scale_col")
                getparam!(buffer_R, solver, parameter)
                @test all(x -> isfinite(x) && x > 0, buffer_R)
            end
            if parameter == "diag"
                getparam!(buffer_T, solver, parameter)
                csym = structure == "S" && eltype(buffer_T) <: Complex      # D of a complex symmetric matrix
                if structure == "G"
                    @test all(x -> real(x) > 0, buffer_T)
                else
                    csym || @test all(x -> real(x) > 0 && imag(x) == 0, buffer_T)
                end
            end
            if parameter == "memory_estimates"
                getparam!(memory_estimates, solver, parameter)
                @test memory_estimates[1] > 0
            end
            if parameter == "info"
                setparam!(solver, parameter, 1)
                @test getparam(solver, parameter) == 1
            end
            # computed data cannot be set
            parameter ∈ ("lu_nnz", "perm_row", "diag") && @test_throws ArgumentError setparam!(solver, parameter, 1)
        end
        @testset "getparam" begin
            parameter ∈ ("comm_device", "comm_host", "user_perm", "perm_row", "perm_col") && continue
            (parameter == "inertia") && !(structure ∈ ("S", "H")) && continue
            if parameter ∈ PORTED_NOT_IMPLEMENTED_DATA
                @test_throws NotSupportedError getparam(solver, parameter)
            else
                val = getparam(solver, parameter)
                parameter == "info" && @test val == 1
                parameter == "lu_nnz" && @test val >= n
                parameter == "nsuperpanels" && @test 1 <= val <= n
                parameter ∈ ("perm_reorder_row", "perm_reorder_col") && @test isperm(val)
                parameter == "diag" && @test length(val) == n
                parameter == "memory_estimates" && @test length(val) == 16
                parameter == "npivots" && @test val == 0
                csym = structure == "S" && eltype(buffer_T) <: Complex
                parameter == "inertia" && @test val == (csym ? (0, 0) : (n, 0))
            end
        end
    end
    return nothing
end

function ported_solver_config_parameters(solver)
    @testset "config parameter = $parameter" for parameter in CONFIG_PARAMETERS
        parameter ∈ ("device_indices", "nd_nlevels", "ubatch_size", "ubatch_index") && continue
        @testset "getparam" begin
            if parameter != "host_nthreads"
                val = getparam(solver, parameter)
                parameter == "device_count" && @test val == 1
            end
        end
        @testset "setparam!" begin
            for (name, value) in (("device_count", 1), ("solve_mode", 0), ("ir_n_steps", 1), ("ir_tol", 1.0e-8),
                                  ("pivot_threshold", 2.0), ("pivot_epsilon", 1.0e-12), ("max_lu_nnz", 10),
                                  ("hybrid_device_memory_limit", 2048), ("host_nthreads", 0))
                if parameter == name
                    setparam!(solver, parameter, value)
                    @test getparam(solver, parameter) == value
                end
            end
            for algo in ("default", "algo1", "algo2", "algo3", "algo4", "algo5")
                haskey(PORTED_ACCEPTED_ALGOS, parameter) || continue
                if algo in PORTED_ACCEPTED_ALGOS[parameter]
                    setparam!(solver, parameter, algo)
                    @test getparam(solver, parameter) == algo
                else
                    @test_throws InvalidValueError setparam!(solver, parameter, algo)
                end
            end
            for flag in (0, 1)
                if parameter ∈ ("schur_mode", "deterministic_mode", "hybrid_memory_mode", "hybrid_execute_mode",
                                "use_superpanels", "use_cuda_register_memory")
                    setparam!(solver, parameter, flag)
                    @test getparam(solver, parameter) == flag
                end
            end
            for pivoting in ('C', 'R', 'N')
                parameter == "pivot_type" || continue
                if pivoting == 'N'
                    setparam!(solver, parameter, pivoting)
                    @test getparam(solver, parameter) == pivoting
                else
                    @test_throws NotSupportedError setparam!(solver, parameter, pivoting)
                end
            end
        end
    end
    return nothing
end

function ported_solver(backend, ::Type{T}, ::Type{INT}, A_gpu, structure, view, n) where {T, INT}
    R = real(T)
    solver = DirectSolver(A_gpu, structure, view)

    x_cpu = zeros(T, n)
    x_gpu = to_device(backend, x_cpu)
    b_cpu = rand(T, n)
    b_gpu = to_device(backend, b_cpu)

    # without matching, the matching outputs are refused
    plain = DirectSolver(A_gpu, structure, view)
    execute!("analysis", plain, x_gpu, b_gpu)
    @test_throws InvalidValueError getparam(plain, "perm_matching")

    setparam!(solver, "matching_alg", "algo6")  # enable matching (AUTO) for "perm_matching" / "scale_row" / "scale_col"
    execute!("analysis", solver, x_gpu, b_gpu)
    execute!("factorization", solver, x_gpu, b_gpu)

    memory_estimates = Vector{Int64}(undef, 16)
    buffer_int = Vector{INT}(undef, n)
    buffer_R = Vector{R}(undef, n)
    buffer_T = Vector{T}(undef, n)

    ported_solver_data_parameters(backend, solver, structure, n, memory_estimates, buffer_int, buffer_R, buffer_T)
    ported_solver_config_parameters(solver)
    return nothing
end

@testset "solver parameters ($(backend_name(backend)), $T, $INT)" for backend in BACKENDS, T in ELTYPES,
                                                                      INT in INTTYPES
    n = 20
    A_cpu = random_spd(T, n, 1.0)
    @testset "structure = $structure" for structure in (T <: Real ? ("G", "S", "H", "SPD", "HPD") :
                                                       ("G", "S", "H", "HPD"))
        @testset "view = $view" for view in ('L', 'U', 'F')
            A_gpu = api_matrix(backend, triangle_view(A_cpu, view), INT)
            if structure == "G" && view != 'F'
                @test_throws InvalidValueError execute!("analysis", DirectSolver(A_gpu, structure, view), nothing,
                                                        nothing)
                continue
            end
            ported_solver(backend, T, INT, A_gpu, structure, view, n)
        end
    end
    if T <: Complex
        solver = DirectSolver(api_matrix(backend, A_cpu, INT), "SPD", 'F')
        @test_throws InvalidValueError execute!("analysis", solver, nothing, nothing)
    end
end
