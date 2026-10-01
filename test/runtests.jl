# Test driver. Every `test_*.jl` file in this directory is a test file; run a
# subset with SDS_TEST_ONLY="test_options,test_aqua" and skip the GPU backends
# with SDS_TEST_GPU=0 (see backends.jl).

using Test
using Random
using LinearAlgebra
using SparseArrays
using Aqua
using KernelAbstractions
using SparseDirectSolver

const SDS = SparseDirectSolver

include("utils.jl")
include("matrices.jl")
include("backends.jl")

const TEST_FILES = sort!([first(splitext(f)) for f in readdir(@__DIR__)
                          if startswith(f, "test_") && endswith(f, ".jl")])

function selected_test_files()
    only = strip(get(ENV, "SDS_TEST_ONLY", ""))
    isempty(only) && return TEST_FILES
    names = [first(splitext(strip(s))) for s in split(only, ',') if !isempty(strip(s))]
    unknown = setdiff(names, TEST_FILES)
    isempty(unknown) ||
        error("SDS_TEST_ONLY names unknown test files $(unknown); available: $(join(TEST_FILES, ", "))")
    return names
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
