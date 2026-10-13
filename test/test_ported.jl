# T13: the SPD/HPD parts of CUDSS.jl's test suite (`../CUDSS.jl/test/test_cudss.jl`),
# ported to the public API: `CudssSolver → DirectSolver`, `cudss(…) → execute!(…)`,
# `cudss_set/cudss_get → setparam!/getparam`, `cudss_update → update!`,
# `CudssMatrix → MatrixDescriptor`. Every file in `test/ported/` runs for every
# backend, `T ∈ ELTYPES` and `INT ∈ PORTED_INTTYPES`.
#
# Common changes, besides the names: matrices come from `test/matrices.jl`
# (`random_spd(T, n, density)` = `B Bᴴ + n I` instead of `A A' + I`), device
# copies from `api_matrix`/`to_device`, and residuals are checked on the host
# with `relres(A, x, b) ≤ tol(T)` (CUDSS.jl: `norm(b - A x) ≤ √eps(R)`).

# Int32 only, as CUDSS.jl itself tests (`Cint`): running the ported suite a second time with Int64 maps
# recompiled every kernel family for Int64 in this worker and roughly doubled the file's cost, while the
# Int64 coverage of each feature lives in its owning file (test_api and the numeric public-API testsets
# through int_types(T); test_solve, test_schur and test_refinement's static-pivot testset directly).
const PORTED_INTTYPES = (Int32,)

for file in sort!(filter(endswith(".jl"), readdir(joinpath(@__DIR__, "ported"))))
    @testset "$(first(splitext(file)))" begin
        include(joinpath(@__DIR__, "ported", file))
    end
end
