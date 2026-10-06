# Test driver on ParallelTestRunner: every `test_*.jl` file in this directory is a test file and runs in its own
# sandbox module on a pool of worker processes (one compilation of the solver per worker, the files in parallel).
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

# what every test file sees: the packages and the shared helpers (TASKS.md "Shared test conventions")
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
    include($(joinpath(TEST_DIR, "utils.jl")))
    include($(joinpath(TEST_DIR, "matrices.jl")))
    include($(joinpath(TEST_DIR, "backends.jl")))
end

# the seed goes with the file, not into `init_code`: ParallelTestRunner seeds with 1 after `init_code`
testsuite = Dict(name => quote
                     Random.seed!(666)
                     include($(joinpath(TEST_DIR, "$name.jl")))
                 end for name in TEST_FILES)

args = parse_args(ARGS)
if filter_tests!(testsuite, args)
    only, skip = test_file_list("SDS_TEST_ONLY"), test_file_list("SDS_TEST_SKIP")
    isempty(only) || filter!(t -> first(t) in only, testsuite)
    filter!(t -> !(first(t) in skip), testsuite)
end

# the backend banner once, from the main process (also fails early when no backend is left to test)
module BackendBanner end
Core.eval(BackendBanner, init_code)
BackendBanner.print_backends()

runtests(SparseDirectSolver, args; testsuite, init_code,
         history_key = get(ENV, "SDS_TEST_CPU", "1") == "0" ? "gpu" : nothing)
