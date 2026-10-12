# MadNLP linear-solver interface to SparseDirectSolver.jl, as a package
# extension (loads when MadNLP and SparseDirectSolver are both present).
# Backend-agnostic: accepts any device CSC (CUSPARSE.CuSparseMatrixCSC,
# rocSPARSE.ROCSparseMatrixCSC) whose colPtr/rowVal/nzVal are device vectors.
# The CSC arrays of MadNLP's lower-triangular KKT matrix are read as the CSR
# of the upper triangle, exactly as MadNLPGPU's CUDSSSolver does (view 'U').
# Access from user code: SparseDirectSolver.SDSSolver (a declared global of the
# parent package this extension assigns in its __init__, the MadNLPGPU idiom).
module SparseDirectSolverMadNLPExt

using LinearAlgebra, SparseArrays
using MadNLP
using SparseDirectSolver
const SDS = SparseDirectSolver
const KA = SDS.KernelAbstractions
const Fused = SDS.Fused
const SDSTuning = Fused.SDSTuning

Base.@kwdef mutable struct SDSSolverOptions <: MadNLP.AbstractOptions
    # :cholesky (SPD; the condensed-KKT default) or :ldl (no-pivot LDL^T: factors
    # indefinite matrices and reports TRUE inertia from the signs of D)
    sds_factorization::Symbol = :cholesky
    sds_reordering::String = "algo4"          # native ND
    # ALL performance knobs live in the tuning bundle; nothing (the default)
    # selects the measured per-architecture profile from Fused.default_tuning
    sds_tuning::Union{Nothing, SDSTuning} = nothing
    # a user permutation bypasses the reordering entirely (1-based, length n)
    sds_user_perm::Union{Nothing, Vector{Int}} = nothing
end

# shared solver construction knobs: the custom reordering (nd knobs or a user
# permutation) and the amalgamation, applied before the analysis. No caching of
# any kind: every solver instance recomputes its ordering (owner call, 2026-10-12);
# pin one explicitly with sds_user_perm.
function _apply_reordering!(s, opt::SDSSolverOptions, t::SDSTuning)
    s.options.amalgamation = t.amalgamation
    if opt.sds_user_perm !== nothing
        SDS.setparam!(s, "user_perm", opt.sds_user_perm)
        return nothing
    end
    SDS.setparam!(s, "reordering_alg", opt.sds_reordering)
    SDS.setparam!(s, "nd_nseps", t.nd_nseps)
    SDS.setparam!(s, "nd_seed", t.nd_seed)
    return nothing
end

"""
    SDSSolver(csc; opt = SDSSolverOptions(), logger = MadNLP.MadNLPLogger())

The MadNLP linear solver backed by the fused SparseDirectSolver pipeline
(factorization mega-kernel + partitioned-inverse solve), with per-architecture
tuning from `Fused.default_tuning`. Available as `SparseDirectSolver.SDSSolver`
(exported) once MadNLP is loaded.
"""
mutable struct SDSSolver{T} <: MadNLP.AbstractLinearSolver{T}
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

function SDSSolver(
        csc;
        opt = SDSSolverOptions(),
        logger = MadNLP.MadNLPLogger(),
    )
    T = eltype(csc.nzVal)
    n = size(csc, 1)
    backend = KA.get_backend(csc.nzVal)
    t = Fused.resolve_tuning(something(opt.sds_tuning, Fused.default_tuning(backend)), backend)
    ldl = opt.sds_factorization === :ldl
    s = SDS.DirectSolver(csc.colPtr, csc.rowVal, csc.nzVal, ldl ? "S" : "SPD", 'U')
    ldl && SDS.setparam!(s, "pivot_type", 'N')
    s.options.regime_c_width = t.regime_c_width
    s.options.regime_c_rows = t.regime_c_rows  # the fused kernels own BOTH phases: no coupling conflict
    s.options.subtree_parallelism = t.subtree_parallelism
    s.options.subtree_budgets = copy(t.subtree_budgets)
    _apply_reordering!(s, opt, t)
    x = similar(csc.nzVal, n)
    b = similar(csc.nzVal, n)
    SDS.execute!("analysis", s, x, b; asynchronous = false)
    SDS.execute!("factorization", s, x, b; asynchronous = false)
    SDS.execute!("solve", s, x, b; asynchronous = false)   # allocates the solve workspace
    nzval = SDS._factor_values(s)
    merged = t.wides === :merged
    # LDL wide fronts run the verbatim Cholesky vendor path (wides_potrf): valid while a
    # front's pivots share a sign; a sign flip fails potrf -> info != 0 -> MadNLP regularizes
    fp = Fused.build_fused_plan(s; ldl, inkernel_wides = merged, merge_segments = merged,
                                wides_potrf = !merged, wgf = t.workgroup, scap = t.spin_cap)
    inv11, inv_ptr, invertf, _ = Fused.build_inverse(s)
    # the solve plan: one merged dependency-counter sweep over the whole tree
    fz1 = Fused.build_fused(s; kstart = 1, wcap = 256)
    fz1 === nothing && throw(SDS.NotSupportedError("fused solve: no plan entry with fronts of width <= 256"))
    fz = ([(fz1, 64)], true)
    M = SDSSolver{T}(s, csc, x, b, nzval, fp, fz, inv11, inv_ptr, invertf, 0, opt, logger)
    MadNLP.factorize!(M)                       # prime the fused path once
    return M
end

function MadNLP.factorize!(M::SDSSolver)
    M.last_info = Int(Fused.refact_fused!(M.inner, M.nzval, M.fp))
    M.invert!()
    return M
end

function _solve!(M::SDSSolver, xb)
    copyto!(M.b, xb)
    Fused.algo2_solve!(M.x, M.inner, M.b, M.inv11, M.inv_ptr, M.fz)
    copyto!(xb, M.x)
    KA.synchronize(KA.get_backend(xb))
    return xb
end

MadNLP.solve!(M::SDSSolver, xb::AbstractVector) = _solve!(M, xb)
# the dev MadNLP routes through solve_linear_system!; the registered 0.8 does not have it
isdefined(MadNLP, :solve_linear_system!) &&
    @eval MadNLP.solve_linear_system!(M::SDSSolver, xb) = _solve!(M, xb)
MadNLP.input_type(::Type{SDSSolver}) = :csc
MadNLP.default_options(::Type{SDSSolver}) = SDSSolverOptions()
MadNLP.is_inertia(::SDSSolver) = true
# a failed factorization is reported as an indefinite inertia, so that MadNLP
# regularizes and factorizes again
function MadNLP.inertia(M::SDSSolver)
    n = size(M.tril, 1)
    if M.opt.sds_factorization === :ldl
        # failure tuple must sum to n (the inertia contract), same as the Cholesky branch
        M.last_info == 0 || return (n - 2, 1, 1)
        l = count(x -> x < zero(x), view(M.inner.numeric.d, 1:n))   # true inertia from D
        return (n - l, 0, l)
    end
    return M.last_info == 0 ? (n, 0, 0) : (n - 2, 1, 1)
end
MadNLP.improve!(::SDSSolver) = false
MadNLP.introduce(::SDSSolver) = "SparseDirectSolver.jl (fused pipeline)"
MadNLP.is_supported(::Type{SDSSolver}, ::Type{Float32}) = true
MadNLP.is_supported(::Type{SDSSolver}, ::Type{Float64}) = true

function __init__()
    setglobal!(SparseDirectSolver, :SDSSolver, SDSSolver)
    setglobal!(SparseDirectSolver, :SDSSolverOptions, SDSSolverOptions)
    return nothing
end

end # module
