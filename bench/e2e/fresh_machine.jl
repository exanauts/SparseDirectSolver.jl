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
Pkg.add(["MadNLP", "MadNLPGPU", "ExaModels", "ExaModelsPower", "Metis"])
Pkg.add(BACKEND == "cuda" ? ["CUDA", "CUDSS"] : ["AMDGPU"])
# SDS as a proper dev checkout (editable, canonical path), not a frozen Pkg.add;
# an existing checkout is updated to the branch head (local edits block the pull loudly)
const SDS_DEV = joinpath(homedir(), ".julia", "dev", "SparseDirectSolver")
const SDS_REF = "divfree-chol"
if isdir(SDS_DEV)
    run(`git -C $SDS_DEV fetch origin $SDS_REF`)
    run(`git -C $SDS_DEV checkout $SDS_REF`)
    run(`git -C $SDS_DEV merge --ff-only FETCH_HEAD`)
else
    run(`git clone --branch $SDS_REF https://github.com/exanauts/SparseDirectSolver.jl $SDS_DEV`)
end
Pkg.develop(path = SDS_DEV)
Pkg.instantiate()

using SparseDirectSolver, MadNLP, MadNLPGPU, ExaModels, ExaModelsPower
using Metis
if BACKEND == "cuda"
    using CUDA, CUDSS      # CUDSS must be LOADED: MadNLPGPU's CUDA extension (the GPU
else                       # condensed-KKT constructors) activates only with it present
    using AMDGPU
end

# the MadNLP linear-solver wrappers (SDSSolver = stock, SDSProtoSolver = PR #107
# prototype kernels), shipped with the package under bench/e2e
include(joinpath(pkgdir(SparseDirectSolver), "bench", "e2e", "MadNLPSDS.jl"))
using .MadNLPSDS

println("device: ", BACKEND == "cuda" ? CUDA.name(CUDA.device()) : string(AMDGPU.device()))

m, _ = ac_opf_model("pglib_opf_case78484_epigrids.m";
                    backend = BACKEND == "cuda" ? CUDABackend() : ROCBackend())

# SparseCondensedKKTSystem is required (the SPD Cholesky path); MadNLP defaults
# equality/fixed-variable treatment correctly for it. tol: condensed default is 1e-4.
sol = madnlp(m; linear_solver = SDSProtoSolver,
             kkt_system = MadNLP.SparseCondensedKKTSystem, tol = 1e-6)
