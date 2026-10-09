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
const KA = SDS.KernelAbstractions

include("SDSProto.jl")
using .SDSProto

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

# ---------------------------------------------------------------------------
# prototype-kernel wrapper: fused factorization + algo1/fused solve (PR #107)
mutable struct SDSProtoSolver{T} <: MadNLP.AbstractLinearSolver{T}
    inner::Any
    tril::Any
    x::Any
    b::Any
    nzval::Any
    fp::Any                                    # fused factorization plan
    fz::Any                                    # fused solve segments
    inv11::Any
    inv_ptr::Any
    invert!::Any
    last_info::Int
    opt::SDSSolverOptions
    logger::MadNLP.MadNLPLogger
end

function SDSProtoSolver(
        csc;
        opt = SDSSolverOptions(),
        logger = MadNLP.MadNLPLogger(),
    )
    T = eltype(csc.nzVal)
    n = size(csc, 1)
    backend = KA.get_backend(csc.nzVal)
    SDSProto.BACKEND[] = backend
    s = SDS.DirectSolver(csc.colPtr, csc.rowVal, csc.nzVal, "SPD", 'U')
    s.options.regime_c_width = 32
    s.options.regime_c_rows = 64               # prototype kernels own BOTH phases: no coupling conflict
    s.options.subtree_parallelism = 16384
    s.options.subtree_budgets = [16384]
    SDS.setparam!(s, "reordering_alg", opt.sds_reordering)
    x = similar(csc.nzVal, n)
    b = similar(csc.nzVal, n)
    SDS.execute!("analysis", s, x, b; asynchronous = false)
    SDS.execute!("factorization", s, x, b; asynchronous = false)
    SDS.execute!("solve", s, x, b; asynchronous = false)   # allocates the solve workspace
    nzval = SDS._factor_values(s)
    fp = SDSProto.build_fused_plan(s)
    inv11, inv_ptr, invertf, _ = SDSProto.build_inverse(s)
    # solve-side plan state (module globals, single solver instance)
    plan0 = s.workspace.plan
    nodes0 = s.symbolic.schedule.group_nodes
    stp = Array(s.symbolic.subtree_ptr)
    sub_ids = Int32[]
    for k in eachindex(plan0.kind)
        plan0.kind[k] == SDS.SOLVE_SUBTREES || continue
        append!(sub_ids, Int32.(nodes0[plan0.first[k]:plan0.last[k]]))
    end
    sort!(sub_ids; by = e -> stp[e + 1] - stp[e], rev = true)
    SDSProto.SSUB[] = SDSProto.DVEC(sub_ids)
    SDSProto.NSUB[] = length(sub_ids)
    SDSProto.INV11[] = inv11
    SDSProto.INVPTR[] = inv_ptr
    fz1 = SDSProto.build_fused(s; kstart = 1, wcap = 256)
    segs = [(fz1, 64)]
    fz = (segs, true)
    M = SDSProtoSolver{T}(s, csc, x, b, nzval, fp, fz, inv11, inv_ptr, invertf, 0, opt, logger)
    MadNLP.factorize!(M)                       # prime the fused path once
    return M
end

function MadNLP.factorize!(M::SDSProtoSolver)
    M.last_info = Int(SDSProto.refact_fused!(M.inner, M.nzval, M.fp))
    M.invert!()
    return M
end

function MadNLP.solve_linear_system!(M::SDSProtoSolver{T}, xb) where {T}
    copyto!(M.b, xb)
    SDSProto.algo2_solve!(M.x, M.inner, M.b, M.inv11, M.inv_ptr, M.fz)
    copyto!(xb, M.x)
    KA.synchronize(KA.get_backend(xb))
    return xb
end

MadNLP.solve!(M::SDSProtoSolver, xb::AbstractVector) = MadNLP.solve_linear_system!(M, xb)
MadNLP.input_type(::Type{SDSProtoSolver}) = :csc
MadNLP.default_options(::Type{SDSProtoSolver}) = SDSSolverOptions()
MadNLP.is_inertia(::SDSProtoSolver) = true
function MadNLP.inertia(M::SDSProtoSolver)
    n = size(M.tril, 1)
    return M.last_info == 0 ? (n, 0, 0) : (n - 2, 1, 1)
end
MadNLP.improve!(::SDSProtoSolver) = false
MadNLP.introduce(::SDSProtoSolver) = "SparseDirectSolver.jl (PR #107 prototype kernels)"
MadNLP.is_supported(::Type{SDSProtoSolver}, ::Type{Float32}) = true
MadNLP.is_supported(::Type{SDSProtoSolver}, ::Type{Float64}) = true

export SDSSolver, SDSSolverOptions, SDSProtoSolver

end # module
