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
using GPUArrays
using GPUArraysCore
using KernelAbstractions
using LinearAlgebra
using SparseArrays

# AMD.jl activates CliqueTrees' AMD extension (`CliqueTrees.AMD()` orderings)
import AMD

export SparseDirectSolverError, NotSupportedError, InvalidValueError, FactorizationError, InterruptedError
export CONFIG_PARAMETERS, DATA_PARAMETERS, CUDSS08_DATA_PARAMETERS, EXTRA_PARAMETERS
export Options, setparam!, getparam, default_pivot_epsilon
export CSR, csr_of_transpose, to_backend, nbatch, MatrixDescriptor, update!

include("errors.jl")
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
include("symbolic/supernodes.jl")
include("symbolic/schedule.jl")
include("symbolic/layout.jl")
include("symbolic/maps.jl")

# numeric storage, CPU reference multifrontal factorization (PLAN §7, the oracle)
include("numeric/storage.jl")
include("reference/cholesky.jl")

# numeric phase on the device (PLAN §2.4)
include("numeric/assembly.jl")
include("numeric/front.jl")
include("numeric/factorize.jl")
include("numeric/extract.jl")

end # module SparseDirectSolver
