# Test driver on ParallelTestRunner: every `test_*.jl` file in this directory is a test file and runs in its own
# sandbox module on a pool of worker processes (one compilation of the solver per worker, the files in parallel); the
# long files run once per element type (`SPLIT_FILES`).
#
# Run a subset with SDS_TEST_ONLY="test_options,test_aqua" or with test arguments (prefix match, `!name` excludes:
# `Pkg.test(; test_args = ["test_symbolic", "!test_symbolic_etree"])`), leave files out with SDS_TEST_SKIP="test_aqua",
# skip the GPU backends with SDS_TEST_GPU=0 and the CPU backend with SDS_TEST_CPU=0 (see backends.jl). The number of
# workers is ParallelTestRunner's default (CPU threads and free memory); set it with `--jobs=N` or PTR_NUM_JOBS.

using ParallelTestRunner
using SparseDirectSolver

const TEST_DIR = @__DIR__

const TEST_FILES = sort!([first(splitext(f)) for f in readdir(TEST_DIR) if startswith(f, "test_") && endswith(f, ".jl")])

function test_file_list(var::String)
    names = [first(splitext(strip(s))) for s in split(get(ENV, var, ""), ',') if !isempty(strip(s))]
    unknown = setdiff(names, TEST_FILES)
    isempty(unknown) ||
        error("$var names unknown test files $(unknown); available: $(join(TEST_FILES, ", "))")
    return names
end

# what every test file sees: the packages (the shared helpers come with the test, see `test_expr`)
const init_code = quote
    using Test
    using Random
    using LinearAlgebra
    using SparseArrays
    using Aqua
    using KernelAbstractions
    using SparseDirectSolver
    using Metis  # activates SparseDirectSolverMetisExt (nested-dissection orderings, T05)
    using Krylov  # activates SparseDirectSolverKrylovExt (ir_mode = "fgmres", T18)
    const SDS = SparseDirectSolver
end

# the element types of the running test (`test/utils.jl`), set by the test before the helpers are included
const init_worker_code = quote
    const SDS_TEST_PART = Ref{Any}(nothing)
end

# Long test files run once per element type, "test_api[Float64]" etc., each part in a worker of its own: a
# worker compiles the solver for every element type it tests, and that compilation, not the tests, sets the wall
# time. The testsets that do not loop over the element types run in the Float64 part (`RUN_SHARED`).
const SPLIT_FILES = ["test_api", "test_dense", "test_fgmres", "test_numeric_cholesky_a", "test_numeric_cholesky_b",
                     "test_numeric_cholesky_c", "test_numeric_ldlt", "test_numeric_lu", "test_ported", "test_refinement",
                     "test_solve", "test_ubatch"]
const SPLIT_ELTYPES = (Float32, Float64, ComplexF32, ComplexF64)

# the seed goes with the test, not into `init_code`: ParallelTestRunner seeds with 1 after `init_code`
test_expr(name, part) = quote
    Main.SDS_TEST_PART[] = $part
    include($(joinpath(TEST_DIR, "utils.jl")))
    include($(joinpath(TEST_DIR, "matrices.jl")))
    include($(joinpath(TEST_DIR, "backends.jl")))
    Random.seed!(666)
    include($(joinpath(TEST_DIR, "$name.jl")))
end

testsuite = Dict{String, Expr}()
for name in TEST_FILES
    if name in SPLIT_FILES
        for T in SPLIT_ELTYPES
            testsuite["$name[$T]"] = test_expr(name, ((T,), T == Float64))
        end
    else
        testsuite[name] = test_expr(name, nothing)
    end
end

# SDS_TEST_ONLY/SDS_TEST_SKIP name test files: a split file stands for all its parts
names_test(names, test) = any(n -> test == n || startswith(test, n * "["), names)

args = parse_args(ARGS)
if filter_tests!(testsuite, args)
    only, skip = test_file_list("SDS_TEST_ONLY"), test_file_list("SDS_TEST_SKIP")
    isempty(only) || filter!(t -> names_test(only, first(t)), testsuite)
    filter!(t -> !names_test(skip, first(t)), testsuite)
end

# the backend banner once, from the main process (also fails early when no backend is left to test)
module BackendBanner end
Core.eval(BackendBanner, init_code)
Core.eval(BackendBanner, :(include($(joinpath(TEST_DIR, "backends.jl")))))
BackendBanner.print_backends()

runtests(SparseDirectSolver, args; testsuite, init_code, init_worker_code,
         history_key = get(ENV, "SDS_TEST_CPU", "1") == "0" ? "gpu" : nothing)
