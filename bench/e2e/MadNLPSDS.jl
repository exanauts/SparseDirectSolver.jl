# MadNLP linear-solver interface to SparseDirectSolver.jl (prototype).
# Backend-agnostic: accepts any device CSC (CUSPARSE.CuSparseMatrixCSC,
# rocSPARSE.ROCSparseMatrixCSC) whose colPtr/rowVal/nzVal are device vectors.
# The CSC arrays of MadNLP's lower-triangular KKT matrix are read as the CSR
# of the upper triangle, exactly as MadNLPGPU's CUDSSSolver does (view 'U').
module MadNLPSDS

using LinearAlgebra, SparseArrays
using MadNLP
using SparseDirectSolver
const SDS = SparseDirectSolver

Base.@kwdef mutable struct SDSSolverOptions <: MadNLP.AbstractOptions
    sds_reordering::String = "algo4"          # native ND (see PR #107: the auto chooser picks AMD here)
    sds_regime_c_width::Int = 32
    sds_regime_c_rows::Int = 256
    sds_subtree_parallelism::Int = 16384
    sds_subtree_budgets::Vector{Int} = [16384]
end

mutable struct SDSSolver{T} <: MadNLP.AbstractLinearSolver{T}
    inner::Any                                 # SDS.DirectSolver (type elided: backend-parametric)
    tril::Any                                  # the device CSC whose nzVal MadNLP updates in place
    x::Any
    b::Any
    factorized_once::Bool
    last_info::Int
    opt::SDSSolverOptions
    logger::MadNLP.MadNLPLogger
end

function SDSSolver(
        csc;
        opt = SDSSolverOptions(),
        logger = MadNLP.MadNLPLogger(),
    )
    T = eltype(csc.nzVal)
    n = size(csc, 1)
    s = SDS.DirectSolver(csc.colPtr, csc.rowVal, csc.nzVal, "SPD", 'U')
    s.options.regime_c_width = opt.sds_regime_c_width
    s.options.regime_c_rows = opt.sds_regime_c_rows
    s.options.subtree_parallelism = opt.sds_subtree_parallelism
    s.options.subtree_budgets = copy(opt.sds_subtree_budgets)
    SDS.setparam!(s, "reordering_alg", opt.sds_reordering)
    x = similar(csc.nzVal, n)
    b = similar(csc.nzVal, n)
    SDS.execute!("analysis", s, x, b; asynchronous = false)
    return SDSSolver{T}(s, csc, x, b, false, 0, opt, logger)
end

function MadNLP.factorize!(M::SDSSolver)
    M.inner.A.nzval === M.tril.nzVal ||
        SDS.update!(M.inner, M.tril.colPtr, M.tril.rowVal, M.tril.nzVal)
    phase = M.factorized_once ? "refactorization" : "factorization"
    SDS.execute!(phase, M.inner, M.x, M.b; asynchronous = false)
    M.factorized_once = true
    M.last_info = Int(SDS.getparam(M.inner, "info"))
    return M
end

function MadNLP.solve_linear_system!(M::SDSSolver{T}, xb) where {T}
    copyto!(M.b, xb)
    SDS.execute!("solve", M.inner, M.x, M.b; asynchronous = false)
    copyto!(xb, M.x)
    return xb
end

# older MadNLP spellings route to the same implementation
MadNLP.solve!(M::SDSSolver, xb::AbstractVector) = MadNLP.solve_linear_system!(M, xb)

MadNLP.input_type(::Type{SDSSolver}) = :csc
MadNLP.default_options(::Type{SDSSolver}) = SDSSolverOptions()
MadNLP.is_inertia(::SDSSolver) = true
function MadNLP.inertia(M::SDSSolver)
    n = size(M.tril, 1)
    return M.last_info == 0 ? (n, 0, 0) : (n - 2, 1, 1)
end
MadNLP.improve!(::SDSSolver) = false
MadNLP.introduce(::SDSSolver) = "SparseDirectSolver.jl"
MadNLP.is_supported(::Type{SDSSolver}, ::Type{Float32}) = true
MadNLP.is_supported(::Type{SDSSolver}, ::Type{Float64}) = true

export SDSSolver, SDSSolverOptions

end # module
