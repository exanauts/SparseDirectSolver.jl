# The CPU allocation checks of the numeric and solve phases (PLAN §3.9, `ka_cpu_alloc_budget`, #145), measured
# in a child process (#188). `@allocated` counts the allocations of the whole process, and KernelAbstractions
# >= 0.10.6 `yield`s after every CPU launch of 50 µs or more, so in this worker ParallelTestRunner's
# communication tasks would be charged to the measured call. `allocations_child.jl` runs the checks, with the
# setups of the test files they were taken from, in a fresh `julia` without other tasks and writes
# `(name, bytes, budget)` per check; this file asserts `bytes <= budget` for each. One child per element type
# (a split file). GPU allocation checks (`device_allocated`) stay in their test files.

TEST_CPU && @testset "CPU allocations in a child process ($T)" for T in ELTYPES
    script = joinpath(@__DIR__, "allocations_child.jl")
    out = tempname()
    # `julia_cmd` forwards --check-bounds, --code-coverage, --depwarn, -O, -g, ... but neither the project nor
    # the thread count; the environment (the Pkg.test load path, SDS_TEST_ELTYPES) is inherited, the child
    # measures on the CPU backend only
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -t $(Threads.nthreads())
           $script $T $RUN_SHARED $out`
    cmd = addenv(cmd, "SDS_TEST_CPU" => "1", "SDS_TEST_GPU" => "0")
    output = IOBuffer()
    proc = run(pipeline(ignorestatus(cmd); stdout = output, stderr = output))
    results = isfile(out) ? [(String(n), parse(Int, b), parse(Int, l)) for (n, b, l) in split.(readlines(out), '\t')] :
              Tuple{String, Int, Int}[]
    rm(out; force = true)
    @test success(proc)
    @test !isempty(results)
    for (name, bytes, budget) in results
        @testset "$name" begin
            @test bytes <= budget
        end
    end
    if !success(proc) || any(r -> r[2] > r[3], results)
        println("allocations_child.jl ($T) output:\n", String(take!(output)))
        for (name, bytes, budget) in results
            bytes > budget && println("  $name: $bytes bytes, budget $budget")
        end
    end
end
