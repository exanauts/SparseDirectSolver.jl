# MadNLP + SparseDirectSolver workflow on a fresh NVIDIA machine, one file:
#
#   julia +1.13 fresh_machine.jl
#
# Installs everything into its own environment (~/.sds-fresh), downloads the
# pglib case on first use, then solves the 78k-bus ACOPF with cuDSS and with
# SparseDirectSolver as MadNLP's linear solver on the same GPU. The workflow
# passes when both solvers converge in the same iterations to the same objective.
#
# AMD machines: replace CUDA/CUDSS with AMDGPU below, CUDABackend with
# ROCBackend, and compare SDSSolver against SDSProtoSolver (there is no cuDSS);
# a system ROCm installation is required.

import Pkg
Pkg.activate(joinpath(homedir(), ".sds-fresh"))
Pkg.add(["MadNLP", "MadNLPGPU", "ExaModels", "ExaModelsPower", "CUDA", "CUDSS", "Metis", "Printf"])
Pkg.add(url = "https://github.com/exanauts/SparseDirectSolver.jl", rev = "divfree-chol")
Pkg.instantiate()

using SparseDirectSolver, MadNLP, MadNLPGPU, ExaModels, ExaModelsPower
using CUDA, CUDSS, Metis, Printf

# the MadNLP linear-solver wrappers (SDSSolver = stock, SDSProtoSolver = PR #107
# prototype kernels), shipped with the package under bench/e2e
include(joinpath(pkgdir(SparseDirectSolver), "bench", "e2e", "MadNLPSDS.jl"))
using .MadNLPSDS

function solve(label, model, linear_solver)
    kwargs = (; kkt_system = MadNLP.SparseCondensedKKTSystem,
              equality_treatment = MadNLP.RelaxEquality,
              fixed_variable_treatment = MadNLP.RelaxBound,
              linear_solver, tol = 1e-6, print_level = MadNLP.ERROR, max_iter = 200)
    madnlp(model; kwargs..., max_iter = 2)                       # warm-up (JIT)
    GC.gc(); CUDA.reclaim()
    sol = madnlp(model; kwargs...)
    c = sol.counters
    @printf "%-10s %-20s iters %3d  obj %.8e  wall %6.2f s  linsolve %6.2f s\n" label string(sol.status) c.k sol.objective c.total_time c.linear_solver_time
    return sol
end

println("device: ", CUDA.name(CUDA.device()))
model, _ = ac_opf_model("pglib_opf_case78484_epigrids.m"; backend = CUDABackend())

solve("cuDSS", model, MadNLPGPU.CUDSSSolver)
solve("SDS", model, SDSProtoSolver)
println("done")
