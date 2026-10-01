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

export SparseDirectSolverError, NotSupportedError, InvalidValueError, FactorizationError, InterruptedError
export CONFIG_PARAMETERS, DATA_PARAMETERS, CUDSS08_DATA_PARAMETERS, EXTRA_PARAMETERS
export Options, setparam!, getparam, default_pivot_epsilon
export CSR, csr_of_transpose, to_backend, nbatch, MatrixDescriptor, update!

include("errors.jl")
include("types.jl")
include("options.jl")
include("matrix.jl")

end # module SparseDirectSolver
