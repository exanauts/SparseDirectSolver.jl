# MadNLP + SparseDirectSolver workflow on a fresh GPU machine, one file:
#
#   julia +1.13 --startup-file=no fresh_machine.jl
#   julia +1.13 --startup-file=no -i fresh_machine.jl          (drop into a REPL after)
#
# Installs everything into its own environment (~/.sds-fresh-<backend>), downloads
# the pglib case on first use, then solves the 78k-bus ACOPF with MadNLP on the
# GPU using SparseDirectSolver (the SDSSolver wrapper) as the linear solver.
# The vendor is autodetected; force it with SDS_BACKEND=cuda|amdgpu. (AMD needs a
# system ROCm installation.)

import Pkg
# vendor detection must RUN the tool, not just find it: cluster nodes often have
# nvidia-smi on PATH without an NVIDIA driver (and vice versa)
_works(cmd) = try
    success(pipeline(cmd; stdout = devnull, stderr = devnull))
catch
    false
end
# device files first: they answer correctly even when NVML/nvidia-smi is wedged
# (driver updated under a loaded module — a real failure mode); tools as fallback
_has_nvidia_gpu() = !isempty(filter(f -> occursin(r"^nvidia\d+$", f), readdir("/dev"))) ||
    try occursin("GPU", read(`nvidia-smi -L`, String)) catch; false end
_has_amd_gpu() = ispath("/dev/kfd") || _works(`rocm-smi`) || _works(`rocminfo`)
const BACKEND = get(ENV, "SDS_BACKEND") do
    _has_nvidia_gpu() ? "cuda" :
    _has_amd_gpu() ? "amdgpu" :
    error("no GPU found (/dev/nvidia*, /dev/kfd, nvidia-smi, rocm-smi all absent); set SDS_BACKEND=cuda|amdgpu")
end
Pkg.activate(joinpath(homedir(), ".sds-fresh-" * BACKEND))
Pkg.add(["MadNLP", "MadNLPGPU", "ExaModels", "ExaModelsPower", "Metis"])
Pkg.add(BACKEND == "cuda" ? ["CUDA", "CUDSS"] : ["AMDGPU"])
# SDS as a proper dev checkout (editable, canonical path), not a frozen Pkg.add;
# an existing checkout is updated to the branch head (local edits block the pull loudly)
const SDS_DEV = joinpath(homedir(), ".julia", "dev", "SparseDirectSolver")
const SDS_REF = "consolidated-prs"   # flip to main when PR 126 merges
if isdir(SDS_DEV)
    # the checkout may be shared (NFS home) with a concurrent run: retry once on
    # git lock contention, then proceed on the existing tree with a warning
    ok = success(`git -C $SDS_DEV fetch origin $SDS_REF`) ||
         (sleep(5); success(`git -C $SDS_DEV fetch origin $SDS_REF`))
    if ok
        run(`git -C $SDS_DEV checkout $SDS_REF`)
        run(`git -C $SDS_DEV merge --ff-only FETCH_HEAD`)
    else
        @warn "could not update $SDS_DEV (concurrent run?); continuing on the existing checkout"
    end
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

# the MadNLP linear-solver wrapper (SDSSolver: the fused kernels with per-arch
# tuning), shipped with the package under bench/e2e
# the MadNLP interface is the SparseDirectSolverMadNLPExt package extension
const SDSSolver = SparseDirectSolver.madnlp_solver()

println("device: ", BACKEND == "cuda" ? CUDA.name(CUDA.device()) : string(AMDGPU.device()))

m, _ = ac_opf_model("pglib_opf_case78484_epigrids.m";
                    backend = BACKEND == "cuda" ? CUDABackend() : ROCBackend())

# SparseCondensedKKTSystem is required (the SPD Cholesky path); MadNLP defaults
# equality/fixed-variable treatment correctly for it. tol: condensed default is 1e-4.
#
# All performance knobs live in one bundle: pass sds_tuning = SparseDirectSolver.Fused.SDSTuning(; ...)
# to override the measured per-architecture defaults (see bench/ARCH-TUNING.md);
# sds_user_perm = perm (1-based, length n) bypasses the built-in reordering entirely.
sol = madnlp(m; linear_solver = SDSSolver,
             kkt_system = MadNLP.SparseCondensedKKTSystem, tol = 1e-6)
