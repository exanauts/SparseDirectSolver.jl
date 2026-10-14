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
    # :ldl only. :diag is simpler partial in-front pivoting: per column take the largest
    # remaining diagonal of the fully-summed block and permute it in symmetrically, never a
    # 2x2. Only 1x1 pivots, so the factor keeps the no-pivot shape plus a permutation and
    # every no-pivot role applies: 1.04x of no-pivot on the 78k condensed KKT (GV100),
    # against :bk's 1.84x. It cannot rescue a front whose remaining diagonals are ALL zero
    # (an unregularized saddle block) where :bk's 2x2 can, so it sits between :none and :bk
    # in robustness.
    # :none factors in identity pivot order; :bk runs in-front Bunch-Kaufman
    # through the fused pipeline (stock pivot machinery in the subtree and PANEL kernels,
    # vendor-sytrf wide fronts) with the pivot-aware fused solve; :bk_hybrid pivots the
    # subtrees and narrow fronts only, wide fronts take the potrf-derived path (a sign
    # flip there fails -> regularization, the :none behavior) at near-:none speed
    sds_pivoting::Symbol = :none
    # in-front pivot acceptance threshold of the BK modes (nothing keeps the SDS
    # default 0.01; cuDSS's default is 1.0)
    sds_pivot_threshold::Union{Nothing, Float64} = nothing
    # 2x2 pivot pairs of the "S" analysis (nothing = the mode's default: "default" for
    # the BK modes, "none" for :none). Pairs are LOAD-BEARING for in-front BK on saddle
    # structures: they place zero-diagonal columns adjacently in one front so the kernels
    # can pair them (measured: without pairs the n=60 saddle perturbs 13 pivots and the
    # solve degrades to 5e-2, stock and fused alike). On mostly-positive condensed KKT
    # the detection finds nothing and costs ~42 s of analysis on the 78k: set "none" there.
    sds_pivot_pairs::Union{Nothing, String} = nothing
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
    fp_slow::Any                               # :bk fallback plan (real pivoting on the wides)
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
    opt.sds_pivoting in (:none, :diag, :bk, :bk_hybrid) ||
        throw(ArgumentError("sds_pivoting must be :none, :diag, :bk or :bk_hybrid, got " *
                            "$(repr(opt.sds_pivoting))"))
    diagp = opt.sds_pivoting === :diag
    bk = opt.sds_pivoting in (:bk, :bk_hybrid)
    (bk || diagp) && !ldl &&
        throw(ArgumentError("sds_pivoting = $(repr(opt.sds_pivoting)) needs sds_factorization = :ldl"))
    s = SDS.DirectSolver(csc.colPtr, csc.rowVal, csc.nzVal, ldl ? "S" : "SPD", 'U')
    # BK modes keep pivot_type AUTO (the in-front machinery reads it); a user threshold
    # reaches the acceptance test of the narrow fronts and subtrees
    ldl && !bk && !diagp && SDS.setparam!(s, "pivot_type", 'N')
    diagp && SDS.setparam!(s, "pivot_type", 'D')
    bk && opt.sds_pivot_threshold !== nothing &&
        SDS.setparam!(s, "pivot_threshold", opt.sds_pivot_threshold)
    # pairs: the no-pivot LDL never consumes them ("none" skips the costly "S" detection);
    # the BK modes keep them by default (see sds_pivot_pairs above)
    if ldl
        pairs = something(opt.sds_pivot_pairs, bk ? "default" : "none")   # :diag takes no pairs
        SDS.setparam!(s, "pivot_pairs", pairs)
    end
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
    fp_slow = nothing
    if diagp
        fp = Fused.build_fused_plan(s; ldl, diag = true, wgf = t.workgroup, scap = t.spin_cap)
    elseif bk
        # NOTE an optimistic fast-path-with-certified-retry was measured and rejected:
        # in-front BK cannot honor |l| <= 1/u on wide fronts either (the fallback accepts
        # sub-threshold pivots when nothing in the fully-summed block passes; the stock
        # factor's wide multipliers reach 4.6e4 on the 78k, LARGER than the unpivoted
        # 3.9e4), so no sound certification criterion separates the two.
        # the BK panel roles want a wider workgroup than the no-pivot path the tuning
        # bundle was measured for: a pivoted front is factored by ONE workgroup over all
        # its rows, where no-pivot spreads the rows below the block across TRSM/TILE roles
        # (78k condensed, GV100: 0.0874s at 256 vs 0.0906 at 128 and 0.0902 at 512)
        wgf_bk = max(t.workgroup, 256)
        if opt.sds_pivoting === :bk            # full BK: vendor-sytrf (or blocked) wides
            fp = Fused.build_fused_plan(s; ldl, bk = true, wgf = wgf_bk, scap = t.spin_cap)
        else                                   # :bk_hybrid: BK narrow, in-kernel no-pivot wides
            fp = Fused.build_fused_plan(s; ldl, bk = true, inkernel_wides = true,
                                        merge_segments = true, wgf = wgf_bk, scap = t.spin_cap)
        end
    else
        merged = t.wides === :merged
        # LDL wide fronts run the verbatim Cholesky vendor path (wides_potrf): valid while a
        # front's pivots share a sign; a sign flip fails potrf -> info != 0 -> MadNLP regularizes
        fp = Fused.build_fused_plan(s; ldl, inkernel_wides = merged, merge_segments = merged,
                                    wides_potrf = !merged, wgf = t.workgroup, scap = t.spin_cap)
    end
    inv11, inv_ptr, invertf, _ = Fused.build_inverse(s)
    # the solve plan: one merged dependency-counter sweep over the whole tree
    fz1 = Fused.build_fused(s; kstart = 1, wcap = 256)
    fz1 === nothing && throw(SDS.NotSupportedError("fused solve: no plan entry with fronts of width <= 256"))
    fz = ([(fz1, 64)], true)
    M = SDSSolver{T}(s, csc, x, b, nzval, fp, fp_slow, fz, inv11, inv_ptr, invertf, 0, opt, logger)
    MadNLP.factorize!(M)                       # prime the factorization path once
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

# true inertia of D with 2x2 pivot blocks, reduced on the device (the per-front stats are
# Cholesky-interpretation garbage after cholesky_stats!, so they are never read here)
function _bk_nneg(N, n)
    d1 = view(N.d, 1:n)
    doff = view(N.d, (n + 1):(2n))
    d2 = view(N.d, 2:(n + 1))
    kv = view(N.pivot_kind, 1:n)
    return mapreduce(+, d1, doff, d2, kv; init = 0) do a, b, c, k
        if k == SDS.PIVOT_KIND_2X2_FIRST
            a * c - b * b < 0 ? 1 : (a + c > 0 ? 0 : 2)
        elseif k == SDS.PIVOT_KIND_2X2_SECOND
            0
        else
            Int(a < zero(a))
        end
    end
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
        if M.opt.sds_pivoting !== :none
            l = _bk_nneg(M.inner.numeric, n)                # true inertia, 2x2 blocks included
            return (n - l, 0, l)
        end
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
