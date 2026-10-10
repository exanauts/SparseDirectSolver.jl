# End-to-end MadNLP on the 78k-bus ACOPF (AMD Radeon VII): SDS as the linear
# solver, same device, same model, SparseCondensedKKTSystem.
#
#   julia +1.13 --project=bench/e2e bench/e2e/run_amd.jl   (on shin-compute-002)

using Printf
using AMDGPU
using MadNLP, MadNLPGPU
using ExaModels, ExaModelsPower

include(joinpath(@__DIR__, "MadNLPSDS.jl"))
using .MadNLPSDS

const CASE = "pglib_opf_case78484_epigrids.m"

function run_one(label, model, linear_solver; warm = true)
    kwargs = (; kkt_system = MadNLP.SparseCondensedKKTSystem,
              equality_treatment = MadNLP.RelaxEquality,
              fixed_variable_treatment = MadNLP.RelaxBound,
              linear_solver, tol = 1e-6, print_level = MadNLP.ERROR, max_iter = 200)
    warm && madnlp(model; kwargs..., max_iter = 2)      # JIT warm-up
    GC.gc()
    sol = madnlp(model; kwargs...)
    c = sol.counters
    @printf "%-16s status %-22s iters %3d  obj %.8e\n" label string(sol.status) c.k sol.objective
    @printf "    wall %7.2f s   solver %7.2f s   linsolve %7.2f s   eval %7.2f s   init %7.2f s\n" c.total_time c.solver_time c.linear_solver_time c.eval_function_time c.init_time
    @printf "    factorizations %d   backsolves %d\n" c.factorization_cnt c.backsolve_cnt
    flush(stdout)
    return sol
end

println("building model on ROCm…"); flush(stdout)
model, _ = ac_opf_model(CASE; backend = ROCBackend())
run_one("SDS/AMD", model, SDSSolver)
println("done")
