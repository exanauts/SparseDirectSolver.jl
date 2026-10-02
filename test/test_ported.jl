# T13: the SPD/HPD parts of CUDSS.jl's test suite (`../CUDSS.jl/test/test_cudss.jl`),
# ported to the public API: `CudssSolver → DirectSolver`, `cudss(…) → execute!(…)`,
# `cudss_set/cudss_get → setparam!/getparam`, `cudss_update → update!`,
# `CudssMatrix → MatrixDescriptor`. Every file in `test/ported/` runs for every
# backend, `T ∈ ELTYPES` and `INT ∈ INTTYPES` (CUDSS.jl tests `Cint` only).
#
# Common changes, besides the names: matrices come from `test/matrices.jl`
# (`random_spd(T, n, density)` = `B Bᴴ + n I` instead of `A A' + I`), device
# copies from `api_matrix`/`to_device`, and residuals are checked on the host
# with `relres(A, x, b) ≤ tol(T)` (CUDSS.jl: `norm(b - A x) ≤ √eps(R)`).

for file in sort!(filter(endswith(".jl"), readdir(joinpath(@__DIR__, "ported"))))
    @testset "$(first(splitext(file)))" begin
        include(joinpath(@__DIR__, "ported", file))
    end
end
