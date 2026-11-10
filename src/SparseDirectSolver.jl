"""
    SparseDirectSolver

Portable sparse direct solver (LLᵀ/LLᴴ, LDLᵀ/LDLᴴ, LDU) for GPUs on
KernelAbstractions.jl, with the parameter names and phases of CUDSS.jl.
See `PLAN.md` for the design.
"""
module SparseDirectSolver

using Adapt
using Atomix
using CliqueTrees
using KernelAbstractions
using LinearAlgebra
using SparseArrays

# AMD.jl and Metis.jl activate CliqueTrees' AMD and METIS extensions
# (`CliqueTrees.AMD()` and `CliqueTrees.METIS()` orderings)
import AMD
import Metis

export SparseDirectSolverError, NotSupportedError, InvalidValueError, FactorizationError, InterruptedError
export CONFIG_PARAMETERS, DATA_PARAMETERS, CUDSS08_DATA_PARAMETERS, EXTRA_PARAMETERS
export Options, setparam!, getparam, default_pivot_epsilon
# `update!` is public but not exported (issue #144): CUDA.jl >= 6.5 exports an `update!`
# of its own (CuGraph), so `using CUDA, SparseDirectSolver` would leave both unusable.
# Write `SparseDirectSolver.update!(solver, A)` or `using SparseDirectSolver: update!`.
export CSR, csr_of_transpose, to_backend, nbatch, MatrixDescriptor
public update!
export AbstractDirectSolver, DirectSolver, execute!, analyze!, factorize!, refactorize!, solve!, getparam!

include("errors.jl")
include("logging.jl")
include("types.jl")
include("options.jl")
include("matrix.jl")

# dense layer (PLAN §2.6): vendor bindings, KA fallbacks, interface, capability audit
include("dense/vendor.jl")
include("dense/fallback/common.jl")
include("dense/fallback/gemm.jl")
include("dense/fallback/trsm.jl")
include("dense/fallback/potrf.jl")
include("dense/fallback/getrf.jl")
include("dense/interface.jl")
include("dense/capabilities.jl")

# host symbolic engine (PLAN §2.3)
include("symbolic/pattern.jl")
include("symbolic/etree.jl")
include("symbolic/ordering.jl")
include("symbolic/pairs.jl")
include("symbolic/supernodes.jl")
include("symbolic/schedule.jl")
include("symbolic/layout.jl")
include("symbolic/device.jl")
include("symbolic/maps.jl")

# matching and scaling (PLAN §1.3, milestone M9), host
include("matching/mc64.jl")

# numeric storage, CPU reference multifrontal factorization (PLAN §7, the oracle)
include("numeric/batch.jl")
include("numeric/storage.jl")
include("reference/cholesky.jl")
include("reference/ldlt.jl")
include("reference/lu.jl")

# numeric phase on the device (PLAN §2.4)
include("numeric/assembly.jl")
include("numeric/front.jl")
include("numeric/subtree.jl")
include("numeric/factorize.jl")
include("numeric/ldlt.jl")
include("numeric/ldlt_c.jl")
include("numeric/lu.jl")
include("numeric/schur.jl")
include("numeric/extract.jl")

# solve phase on the device (PLAN §2.5)
include("solve/permute.jl")
include("solve/sweeps.jl")
include("solve/refinement.jl")

# public API (PLAN §3.1): handle-style layer and LinearAlgebra layer
include("solver.jl")
include("generic.jl")

# the fused pipeline (factorization mega-kernel + partitioned-inverse solve),
# its per-architecture tuning bundle and `FusedSolver`, the pipeline behind the
# handle-layer verbs
include("fused.jl")
using .Fused: FusedSolver, SDSTuning, default_tuning
export FusedSolver, SDSTuning, default_tuning

# the front end for the KKT systems of an interior-point method: pipeline choice,
# factorization parameters, refactorization, in-place solve and inertia. MadNLP's
# binding (MadCoreGPU's MadCoreGPUSDSExt) is built on it. `inertia` and `pipeline`
# are public but not exported: MadNLP has an `inertia` of its own.
include("kkt.jl")
export KKTSolver
public inertia, pipeline

__init__() = _init_log_level()

end # module SparseDirectSolver
