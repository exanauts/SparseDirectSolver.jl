# MadNLP linear-solver interface to SparseDirectSolver.jl, as a package
# extension (loads when MadNLP and SparseDirectSolver are both present).
# Backend-agnostic: accepts any device CSC (CUSPARSE.CuSparseMatrixCSC,
# rocSPARSE.ROCSparseMatrixCSC) whose colPtr/rowVal/nzVal are device vectors,
# and the SparseMatrixCSC of MadNLP's CPU KKT systems (the `:bk` path on the
# KernelAbstractions CPU backend; the fused pipeline needs a GPU, #175).
# The CSC arrays of MadNLP's lower-triangular KKT matrix are read as the CSR
# of the upper triangle, exactly as MadNLPGPU's CUDSSSolver does (view 'U').
# Access from user code: SparseDirectSolver.SDSSolver (a declared global of the
# parent package this extension assigns in its __init__, the MadNLPGPU idiom).
module SparseDirectSolverMadNLPExt

using MadNLP
using SparseDirectSolver
using SparseArrays: SparseMatrixCSC
const SDS = SparseDirectSolver
const KA = SDS.KernelAbstractions
const Fused = SDS.Fused
const SDSTuning = Fused.SDSTuning

"""
    SDSSolverOptions(; sds_factorization = :cholesky, sds_pivoting = :none,
                       sds_pivot_threshold = nothing, sds_reordering = "algo4",
                       sds_tuning = nothing, sds_user_perm = nothing)

The MadNLP options of [`SDSSolver`](@ref), passed as keywords of `madnlp` or
in an `SDSSolverOptions`. Available as `SparseDirectSolver.SDSSolverOptions`
once MadNLP is loaded.

* `sds_factorization::Symbol = :cholesky`: `:cholesky` factors the KKT matrix
  as `"SPD"` (the condensed-KKT default); `:ldl` as `"S"`, `L D Lᵀ`, and
  reports the true inertia from the signs of `D`.
* `sds_pivoting::Symbol = :none`: `:ldl` only. `:none` factors in identity
  pivot order through the fused pipeline (a [`FusedSolver`](@ref SparseDirectSolver.FusedSolver));
  `:bk` runs the stock in-front Bunch–Kaufman path of a
  [`DirectSolver`](@ref SparseDirectSolver.DirectSolver): more robust on
  ill-conditioned KKT systems, at the level-batched speed of the handle API.
* `sds_pivot_threshold::Union{Nothing, Float64} = nothing`: `:bk` only, the
  `pivot_threshold` of the in-front acceptance test (`nothing` keeps the
  package default `0.01`; cuDSS's default is `1.0`).
* `sds_reordering::String = "algo4"`: the `reordering_alg` of the analysis
  (`"algo4"` nested dissection, `"algo3"` AMD, `"default"` the cost-model
  choice).
* `sds_tuning::Union{Nothing, SDSTuning} = nothing`: a
  [`SDSTuning`](@ref SparseDirectSolver.Fused.SDSTuning) bundle of performance
  knobs; `nothing` selects the per-architecture profile of
  [`default_tuning`](@ref SparseDirectSolver.Fused.default_tuning).
* `sds_user_perm::Union{Nothing, Vector{Int}} = nothing`: a 1-based
  permutation of length `n` that replaces the reordering (`user_perm`); every
  solver instance otherwise recomputes its ordering.
"""
Base.@kwdef mutable struct SDSSolverOptions <: MadNLP.AbstractOptions
    # :cholesky (SPD; the condensed-KKT default) or :ldl (no-pivot LDL^T: factors
    # indefinite matrices and reports TRUE inertia from the signs of D)
    sds_factorization::Symbol = :cholesky
    # :ldl only. :none factors in identity pivot order through the fused pipeline
    # (a FusedSolver); :bk routes factorization and solve through the stock in-front
    # Bunch-Kaufman path (a DirectSolver: src/numeric/ldlt.jl and the pivot-aware
    # sweep solve): more robust on ill-conditioned KKT systems, at stock
    # (launch-per-level) speed
    sds_pivoting::Symbol = :none
    # in-front pivot acceptance threshold of :bk (nothing keeps the SDS default
    # 0.01; cuDSS's default is 1.0)
    sds_pivot_threshold::Union{Nothing, Float64} = nothing
    sds_reordering::String = "algo4"          # native ND
    # ALL performance knobs live in the tuning bundle; nothing (the default)
    # selects the measured per-architecture profile from Fused.default_tuning
    sds_tuning::Union{Nothing, SDSTuning} = nothing
    # a user permutation bypasses the reordering entirely (1-based, length n)
    sds_user_perm::Union{Nothing, Vector{Int}} = nothing
end

# the reordering of the options, applied before the analysis on top of the tuning's
# (native ND with the tuning's nd_nseps/nd_seed): a user permutation bypasses the
# reordering entirely, else `sds_reordering`. No caching of any kind: every solver
# instance recomputes its ordering (owner call, 2026-10-12); pin one explicitly with
# sds_user_perm.
function _apply_reordering!(s, opt::SDSSolverOptions)
    if opt.sds_user_perm !== nothing
        SDS.setparam!(s, "user_perm", opt.sds_user_perm)
    else
        SDS.setparam!(s, "reordering_alg", opt.sds_reordering)
    end
    return nothing
end

"""
    SDSSolver(csc; opt = SDSSolverOptions(), logger = MadNLP.MadNLPLogger())

`csc` is the lower triangle of the KKT matrix as MadNLP passes it: a vendor CSC
(`CuSparseMatrixCSC`, `ROCSparseMatrixCSC`, `oneSparseMatrixCSC`) on a GPU backend,
or a `SparseMatrixCSC` on the KernelAbstractions CPU backend, where only
`sds_pivoting = :bk` runs (`:none` raises `NotSupportedError`: the fused pipeline
needs a GPU).

The MadNLP linear solver backed by SparseDirectSolver.jl: a shell over an inner
`AbstractDirectSolver`, a [`FusedSolver`](@ref SparseDirectSolver.FusedSolver)
(the fused factorization mega-kernel and partitioned-inverse solve, with the
per-architecture tuning of `Fused.default_tuning`) for `sds_pivoting = :none`,
a `DirectSolver` (stock Bunch-Kaufman LDLᵀ, same tuning) for `:bk`; both are
driven by `refactorize!`, `solve!` and `getparam(inner, "inertia")`. Available
as `SparseDirectSolver.SDSSolver` (exported) once MadNLP is loaded.
"""
# the CSC arrays of MadNLP's matrix (#175): a vendor CSC's colPtr/rowVal/nzVal on a GPU
# backend, a SparseMatrixCSC's colptr/rowval/nzval on the CPU backend
_csc_arrays(csc::SparseMatrixCSC) = (csc.colptr, csc.rowval, csc.nzval)
_csc_arrays(csc) = (csc.colPtr, csc.rowVal, csc.nzVal)

mutable struct SDSSolver{T} <: MadNLP.AbstractLinearSolver{T}
    inner::SDS.AbstractDirectSolver
    tril::Any
    x::Any
    b::Any
    last_info::Int
    opt::SDSSolverOptions
    logger::MadNLP.MadNLPLogger
end

function SDSSolver(
        csc;
        opt = SDSSolverOptions(),
        logger = MadNLP.MadNLPLogger(),
    )
    colptr, rowval, nzval = _csc_arrays(csc)
    T = eltype(nzval)
    n = size(csc, 1)
    backend = KA.get_backend(nzval)
    ldl = opt.sds_factorization === :ldl
    opt.sds_pivoting in (:none, :bk) ||
        throw(ArgumentError("sds_pivoting must be :none or :bk, got $(repr(opt.sds_pivoting))"))
    bk = opt.sds_pivoting === :bk
    bk && !ldl && throw(ArgumentError("sds_pivoting = :bk needs sds_factorization = :ldl"))
    bk || !(backend isa KA.CPU) ||
        throw(SDS.NotSupportedError("SDSSolver: sds_pivoting = :none runs the fused pipeline, which the " *
                                    "KernelAbstractions CPU backend cannot run; use sds_factorization = :ldl, " *
                                    "sds_pivoting = :bk on the CPU"))
    s = if bk
        # the stock path is the whole pipeline: pivot_type AUTO (Bunch-Kaufman for "S") and
        # the default 2x2 pivot pairs; a user threshold reaches the in-front acceptance test
        t = Fused._resolved_tuning(opt.sds_tuning, backend)
        d = SDS.DirectSolver(colptr, rowval, nzval, "S", 'U')
        opt.sds_pivot_threshold !== nothing && SDS.setparam!(d, "pivot_threshold", opt.sds_pivot_threshold)
        Fused._apply_tuning!(d, t)
    else
        SDS.FusedSolver(colptr, rowval, nzval, ldl ? "S" : "SPD", 'U'; tuning = opt.sds_tuning)
    end
    _apply_reordering!(s, opt)
    x = similar(nzval, n)
    b = similar(nzval, n)
    SDS.analyze!(s, x, b; asynchronous = false)
    SDS.factorize!(s, x, b; asynchronous = false)   # primes the factorization path once
    return SDSSolver{T}(s, csc, x, b, Int(SDS.getparam(s, "info")), opt, logger)
end

function MadNLP.factorize!(M::SDSSolver)
    SDS.refactorize!(M.inner, M.x, M.b; asynchronous = false)
    M.last_info = Int(SDS.getparam(M.inner, "info"))
    return M
end

function _solve!(M::SDSSolver, xb)
    copyto!(M.b, xb)
    SDS.solve!(M.inner, M.x, M.b)
    copyto!(xb, M.x)
    KA.synchronize(KA.get_backend(xb))
    return xb
end

MadNLP.solve!(M::SDSSolver, xb::AbstractVector) = _solve!(M, xb)
MadNLP.solve_linear_system!(M::SDSSolver, xb) = _solve!(M, xb)   # MadNLP >= 0.9 routes through it
MadNLP.input_type(::Type{SDSSolver}) = :csc
MadNLP.default_options(::Type{SDSSolver}) = SDSSolverOptions()
MadNLP.is_inertia(::SDSSolver) = true
# a failed factorization is reported as an indefinite inertia, so that MadNLP
# regularizes and factorizes again
function MadNLP.inertia(M::SDSSolver)
    n = size(M.tril, 1)
    # failure tuple must sum to n (the inertia contract)
    M.last_info == 0 || return (n - 2, 1, 1)
    M.opt.sds_factorization === :ldl || return (n, 0, 0)
    npos, nneg = SDS.getparam(M.inner, "inertia")   # of D (2x2 blocks included for :bk)
    return (Int(npos), n - Int(npos) - Int(nneg), Int(nneg))
end
MadNLP.improve!(::SDSSolver) = false
MadNLP.introduce(M::SDSSolver) = M.inner isa SDS.FusedSolver ? "SparseDirectSolver.jl (fused pipeline)" :
                                 "SparseDirectSolver.jl (Bunch-Kaufman LDLᵀ, DirectSolver)"
MadNLP.is_supported(::Type{SDSSolver}, ::Type{Float32}) = true
MadNLP.is_supported(::Type{SDSSolver}, ::Type{Float64}) = true

function __init__()
    setglobal!(SparseDirectSolver, :SDSSolver, SDSSolver)
    setglobal!(SparseDirectSolver, :SDSSolverOptions, SDSSolverOptions)
    return nothing
end

end # module
