# Test driver. Every `test_*.jl` file in this directory is a test file; run a
# subset with SDS_TEST_ONLY="test_options,test_aqua", leave files out with
# SDS_TEST_SKIP="test_aqua", skip the GPU backends with SDS_TEST_GPU=0 and the
# CPU backend with SDS_TEST_CPU=0 (see backends.jl).

using Test
using Random
using LinearAlgebra
using SparseArrays
using Aqua
using KernelAbstractions
using SparseDirectSolver
using Metis  # activates SparseDirectSolverMetisExt (nested-dissection orderings, T05)

const SDS = SparseDirectSolver

include("utils.jl")
include("matrices.jl")
include("backends.jl")

const TEST_FILES = sort!([first(splitext(f)) for f in readdir(@__DIR__)
                          if startswith(f, "test_") && endswith(f, ".jl")])

function test_file_list(var::String)
    names = [first(splitext(strip(s))) for s in split(get(ENV, var, ""), ',') if !isempty(strip(s))]
    unknown = setdiff(names, TEST_FILES)
    isempty(unknown) ||
        error("$var names unknown test files $(unknown); available: $(join(TEST_FILES, ", "))")
    return names
end

function selected_test_files()
    only = test_file_list("SDS_TEST_ONLY")
    skip = test_file_list("SDS_TEST_SKIP")
    return setdiff(isempty(only) ? TEST_FILES : only, skip)
end

@testset "SparseDirectSolver.jl" begin
    for name in selected_test_files()
        @testset "$name" begin
            Random.seed!(666)
            tic = time()
            include("$name.jl")
            println("$name: $(round(time() - tic; digits = 1)) s")
        end
    end
end
