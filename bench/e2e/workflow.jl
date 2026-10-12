# The MadNLP + SparseDirectSolver 78k-bus ACOPF workflow, run on a project
# environment — the @sds named environment that fresh_machine.sh generates, or the
# in-repo bench/e2e one (its [sources] entry wires SparseDirectSolver either way,
# so `Pkg.instantiate()` is the only setup):
#
#   julia +1.13 --startup-file=no --project=@sds        bench/e2e/workflow.jl
#   julia +1.13 --startup-file=no --project=bench/e2e   bench/e2e/workflow.jl
#
# fresh_machine.jl inlines the same flow for a machine without a checkout.
# Backend is autodetected (SDS_BACKEND=cuda|amdgpu
# overrides); device files are consulted first, since nvidia-smi can be wedged by
# a driver updated under a loaded kernel module.
const BACKEND = get(ENV, "SDS_BACKEND") do
    !isempty(filter(f -> occursin(r"^nvidia\d+$", f), readdir("/dev"))) ? "cuda" :
    ispath("/dev/kfd") ? "amdgpu" :
    error("no GPU found (/dev/nvidia*, /dev/kfd absent); set SDS_BACKEND=cuda|amdgpu")
end

using SparseDirectSolver, MadNLP, MadNLPGPU, ExaModels, ExaModelsPower, Metis
if BACKEND == "cuda"
    using CUDA, CUDSS      # CUDSS must be LOADED: MadNLPGPU's CUDA extension (the GPU
else                       # condensed-KKT constructors) activates only with it present
    using AMDGPU
end

const SDSSolver = SparseDirectSolver.madnlp_solver()
println("device: ", BACKEND == "cuda" ? CUDA.name(CUDA.device()) : string(AMDGPU.device()))

m, _ = ac_opf_model("pglib_opf_case78484_epigrids.m";
                    backend = BACKEND == "cuda" ? CUDABackend() : ROCBackend())

# SparseCondensedKKTSystem is required (the SPD Cholesky path). All performance
# knobs live in one bundle: sds_tuning = SparseDirectSolver.Fused.SDSTuning(; ...)
# overrides the per-architecture defaults (bench/ARCH-TUNING.md).
# SDS_PERM_CACHE=0 disables the on-disk permutation cache, so every solve pays the
# full METIS reordering (for timing it); default: cached under ~/.julia/sds_perm_cache
solve() = madnlp(m; linear_solver = SDSSolver,
                 kkt_system = MadNLP.SparseCondensedKKTSystem, tol = 1e-6,
                 (get(ENV, "SDS_PERM_CACHE", "1") == "0" ? (; sds_perm_cache = "") : (;))...)

t1 = @elapsed sol = solve()
println("\n== first solve (pays Julia compilation): ", round(t1; digits = 1), " s")
println("   status = ", sol.status, "  iters = ", sol.counters.k, "  obj = ", sol.objective)
for f in propertynames(sol.counters)
    v = getfield(sol.counters, f)
    v isa Real && endswith(String(f), "_time") && f !== :start_time &&
        println("   ", f, " = ", round(v; digits = 2), " s")
end
# the honest speed is the warm number: same process, nothing cached between the
# two solves except compiled code (and the on-disk permutation of this pattern)
t2 = @elapsed sol2 = solve()
println("== second solve (warm): ", round(t2; digits = 1), " s",
        "  iters = ", sol2.counters.k, "  obj = ", sol2.objective)
