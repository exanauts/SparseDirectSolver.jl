# MadNLP + SparseDirectSolver workflow on a fresh GPU machine, one file:
#
#   julia +1.13 fresh_machine.jl          (or:  julia +1.13 -i fresh_machine.jl)
#
# Installs everything into its own environment (~/.sds-fresh-<backend>), downloads
# the pglib case on first use, then solves the 78k-bus ACOPF with MadNLP on the
# GPU using SparseDirectSolver (the SDSProtoSolver wrapper) as the linear solver.
# The vendor is autodetected; force it with SDS_BACKEND=cuda|amdgpu. (AMD needs a
# system ROCm installation.)

import Pkg
const BACKEND = get(ENV, "SDS_BACKEND", Sys.which("nvidia-smi") === nothing ? "amdgpu" : "cuda")
Pkg.activate(joinpath(homedir(), ".sds-fresh-" * BACKEND))
Pkg.add(["MadNLP", "MadNLPGPU", "ExaModels", "ExaModelsPower", "Metis", "Printf"])
Pkg.add(BACKEND == "cuda" ? ["CUDA"] : ["AMDGPU"])
# SDS as a proper dev checkout (editable, canonical path), not a frozen Pkg.add
const SDS_DEV = joinpath(homedir(), ".julia", "dev", "SparseDirectSolver")
isdir(SDS_DEV) ||
    run(`git clone --branch divfree-chol https://github.com/exanauts/SparseDirectSolver.jl $SDS_DEV`)
Pkg.develop(path = SDS_DEV)
Pkg.instantiate()

using SparseDirectSolver, MadNLP, MadNLPGPU, ExaModels, ExaModelsPower
using Metis, Printf
if BACKEND == "cuda"
    using CUDA
else
    using AMDGPU
end

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
    GC.gc()
    BACKEND == "cuda" && CUDA.reclaim()
    sol = madnlp(model; kwargs...)
    c = sol.counters
    @printf "%-10s %-20s iters %3d  obj %.8e  wall %6.2f s  linsolve %6.2f s\n" label string(sol.status) c.k sol.objective c.total_time c.linear_solver_time
    return sol
end

println("device: ", BACKEND == "cuda" ? CUDA.name(CUDA.device()) : string(AMDGPU.device()))
model, _ = ac_opf_model("pglib_opf_case78484_epigrids.m";
                        backend = BACKEND == "cuda" ? CUDABackend() : ROCBackend())

solve("SDS", model, SDSProtoSolver)
println("done")
