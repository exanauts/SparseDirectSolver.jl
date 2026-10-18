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
    # saddle-point partition, filled in by MadNLP's block_structure API before the
    # analysis: rows 1:n_primal are primal, the rest dual. 0 means undeclared.
    n_primal::Int = 0
    n_dual::Int = 0
    # use the declared partition as the static pivot sign (+1 primal, -1 dual), so a
    # perturbed pivot takes the sign the saddle structure demands and the inertia MadNLP
    # regularizes against comes out right. Opt-in (upstreamed from the benchmark's
    # experimental branch); it only bites where a pivot is actually perturbed, i.e. under
    # :bk or with sds_stabilize = true.
    # false | true/:block (+1 primal, -1 dual) | :negative (-1 everywhere).
    # :negative reproduces the CPU lazy stabilization that worked on hard ACOPF
    # (ordered_ldl.jl pivot_mode=:subtract with n_d = 0: a near-zero pivot takes
    # d -= eps in the NEGATIVE direction for BOTH blocks, which collapsed the
    # always-on-delta iteration blowup, 2868_rte 1269 -> 184 iterations).
    sds_pivot_signs::Union{Bool, Symbol} = false
    # lazy stabilization for the 1x1 paths (:none and :diag): a pivot below pivot_epsilon
    # takes +/-epsilon with the sign policy instead of failing the front, which is what the
    # :bk path has always done in _lt_select!. Off by default, so the failure contract the
    # existing modes were measured under is unchanged. With it, the three pivoting modes
    # share one stabilization policy and differ only in pivoting.
    sds_stabilize::Bool = false
    # pivot_epsilon: the magnitude of the lazy/static perturbation, and the threshold
    # below which a pivot is considered tiny. nothing keeps SDS's default, which for
    # Float64 is 1e-13 -- five orders below the fixed 1e-8 the CPU lazy stabilization
    # was measured with, so it must be set explicitly to reproduce that configuration.
    sds_pivot_epsilon::Union{Nothing, Float64} = nothing
    # "default" (fixed epsilon) or "algo1" (epsilon scaled by max |a_ij|). A real SDS
    # parameter that was never exposed here, which is why condensed_sds_pe silently
    # measured the default -- see validate_cells in src/gpu_configs.jl.
    sds_pivot_epsilon_alg::Union{Nothing, String} = nothing
    sds_reordering::String = "algo4"          # native ND
    # ALL performance knobs live in the tuning bundle; nothing (the default)
    # selects the measured per-architecture profile from Fused.default_tuning
    sds_tuning::Union{Nothing, SDSTuning} = nothing
    # a user permutation bypasses the reordering entirely (1-based, length n)
    sds_user_perm::Union{Nothing, Vector{Int}} = nothing
    # ---- shifted-KKT specialization (the ShiftedKKT method on SDS's own factorization) ----
    # Symmetric Ruiz equilibration before every factorization (:none | :ruiz): sds_ruiz_sweeps
    # sweeps of d_i <- d_i / sqrt(max_j |d_i a_ij d_j|); the factorization is of D A D and the
    # solve applies D on both sides. The user's values are never modified: the solver then
    # factors an owned copy.
    sds_equilibrate::Symbol = :none
    sds_ruiz_sweeps::Int = 4
    # Constant shift of the dual block of the (equilibrated) matrix: rows n_primal+1:n get
    # a_ii -= sds_dual_shift, which makes a saddle-point matrix quasidefinite so the static
    # factorization is stable. Needs n_primal from MadNLP's block structure. 0 disables.
    sds_dual_shift::Float64 = 0.0
    # Krylov recovery of each solve (:none | :gmres): restarted GMRES on the UNSHIFTED,
    # unscaled matrix, left-preconditioned by D M^-1 D with M the factorization above, so the
    # bias of the shift leaves the Newton step. A solution is accepted when its componentwise
    # backward error is at most sds_ir_omega_tol or its relative residual at most sds_ir_rtol;
    # at most sds_ir_cycles cycles of sds_ir_restart steps.
    sds_ir_mode::Symbol = :none
    sds_ir_cycles::Int = 2
    sds_ir_restart::Int = 50
    sds_ir_omega_tol::Float64 = 1e-6
    sds_ir_rtol::Float64 = 1e-14
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
    shift::Any                                 # ShiftState when a shifted option is on, else nothing
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
    opt.sds_equilibrate in (:none, :ruiz) ||
        throw(ArgumentError("sds_equilibrate must be :none or :ruiz, got $(repr(opt.sds_equilibrate))"))
    opt.sds_ir_mode in (:none, :gmres) ||
        throw(ArgumentError("sds_ir_mode must be :none or :gmres, got $(repr(opt.sds_ir_mode))"))
    opt.sds_dual_shift >= 0 || throw(ArgumentError("sds_dual_shift must be >= 0"))
    opt.sds_dual_shift == 0 || opt.n_primal > 0 ||
        throw(ArgumentError("sds_dual_shift needs the KKT block structure (n_primal is 0)"))
    useshift = opt.sds_equilibrate !== :none || opt.sds_dual_shift != 0 || opt.sds_ir_mode !== :none
    # with a shifted option the factorization reads an owned copy of the values (D A D - shift);
    # csc.nzVal stays the user's matrix, which the recovery multiplies with
    vals = useshift ? copy(csc.nzVal) : csc.nzVal
    s = SDS.DirectSolver(csc.colPtr, csc.rowVal, vals, ldl ? "S" : "SPD", 'U')
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
    opt.sds_pivot_epsilon === nothing ||
        SDS.setparam!(s, "pivot_epsilon", opt.sds_pivot_epsilon)
    opt.sds_pivot_epsilon_alg === nothing ||
        SDS.setparam!(s, "pivot_epsilon_alg", opt.sds_pivot_epsilon_alg)
    let sg = opt.sds_pivot_signs === true ? :block : opt.sds_pivot_signs
        sg === false || sg in (:block, :negative) ||
            throw(ArgumentError("sds_pivot_signs must be false, true/:block or :negative, got " *
                                string(sg)))
        if sg === :negative
            SDS.setparam!(s, "pivot_sign", fill(Int8(-1), n))
        elseif sg === :block && opt.n_primal > 0
            npr = opt.n_primal
            0 < npr <= n || throw(ArgumentError("n_primal = $npr outside 1:$n"))
            psign = Vector{Int8}(undef, n)
            @inbounds for i in 1:n
                psign[i] = i <= npr ? Int8(1) : Int8(-1)
            end
            SDS.setparam!(s, "pivot_sign", psign)
        end
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
        fp = Fused.build_fused_plan(s; ldl, diag = true, stabilize = opt.sds_stabilize,
                                    wgf = t.workgroup, scap = t.spin_cap)
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
                                    wides_potrf = !merged, stabilize = opt.sds_stabilize,
                                    wgf = t.workgroup, scap = t.spin_cap)
    end
    inv11, inv_ptr, invertf, _ = Fused.build_inverse(s)
    # the solve plan: one merged dependency-counter sweep over the whole tree
    fz1 = Fused.build_fused(s; kstart = 1, wcap = 256)
    fz1 === nothing && throw(SDS.NotSupportedError("fused solve: no plan entry with fronts of width <= 256"))
    fz = ([(fz1, 64)], true)
    shift = useshift ? _shift_state(csc, vals, opt, backend) : nothing
    M = SDSSolver{T}(s, csc, x, b, nzval, fp, fp_slow, fz, inv11, inv_ptr, invertf, 0, opt, logger, shift)
    MadNLP.factorize!(M)                       # prime the factorization path once
    return M
end

# ---------------------------------------------------------------------------------------------
# shifted-KKT specialization: the factorization reads an owned copy of the values, equilibrated
# (D A D) and dual-shifted; the solve applies D on both sides and, with recovery, refines the
# step on the user's unshifted matrix by left-preconditioned restarted GMRES.

mutable struct ShiftState
    vals::Any          # owned values the factorization reads: D A D with the dual shift
    truevals::Any      # the user's values (csc.nzVal), read only
    n::Int
    nprimal::Int
    d::Any             # Ruiz scaling, nothing without equilibration
    r::Any             # row-max work vector
    frowptr::Any       # full symmetric pattern, CSR over both triangles (device)
    fcolval::Any
    fsrc::Any          # stored position of each full entry
    trow::Any          # row / column of each stored entry (the scaling D A D)
    tcol::Any
    dualdiag::Any      # stored positions of the dual-block diagonal (the shift)
    rowsum::Any        # |A| 1 of the user's matrix (backward error of the recovery)
    gm::Any            # GMRES workspace, nothing without recovery
    nsolves::Int
    nunaccepted::Int
end

# full symmetric pattern and value maps from the stored triangle (host, once)
function _full_pattern(colptr, rowval, n)
    cnt = zeros(Int, n)
    for j in 1:n, k in colptr[j]:(colptr[j + 1] - 1)
        i = rowval[k]
        cnt[i] += 1
        i != j && (cnt[j] += 1)
    end
    rowptr = Vector{Int}(undef, n + 1)
    rowptr[1] = 1
    for i in 1:n
        rowptr[i + 1] = rowptr[i] + cnt[i]
    end
    pos = rowptr[1:n]
    nnzf = rowptr[n + 1] - 1
    colval = Vector{Int}(undef, nnzf)
    src = Vector{Int}(undef, nnzf)
    trow = Vector{Int}(undef, length(rowval))
    tcol = Vector{Int}(undef, length(rowval))
    diag = zeros(Int, n)
    for j in 1:n, k in colptr[j]:(colptr[j + 1] - 1)
        i = rowval[k]
        trow[k] = i
        tcol[k] = j
        colval[pos[i]] = j; src[pos[i]] = k; pos[i] += 1
        if i != j
            colval[pos[j]] = i; src[pos[j]] = k; pos[j] += 1
        else
            diag[i] = k
        end
    end
    any(iszero, diag) && throw(ArgumentError("the shifted options need every diagonal entry stored"))
    return rowptr, colval, src, trow, tcol, diag
end

_dev(like, v::AbstractVector{Int}) = (o = similar(like, eltype(like), length(v)); copyto!(o, eltype(like).(v)); o)

function _shift_state(csc, vals, opt::SDSSolverOptions, backend)
    T = eltype(csc.nzVal)
    n = size(csc, 1)
    rowptr, colval, src, trow, tcol, diag = _full_pattern(Array(csc.colPtr), Array(csc.rowVal), n)
    ruiz = opt.sds_equilibrate === :ruiz
    d = ruiz ? similar(csc.nzVal, T, n) : nothing
    r = ruiz ? similar(csc.nzVal, T, n) : nothing
    dual = opt.sds_dual_shift == 0 ? nothing : _dev(csc.rowVal, diag[(opt.n_primal + 1):n])
    gm = nothing
    rowsum = nothing
    if opt.sds_ir_mode === :gmres
        m = opt.sds_ir_restart
        vec() = similar(csc.nzVal, T, n)
        ones1 = vec(); fill!(ones1, one(T))
        gm = (; x = vec(), r = vec(), z = vec(), tmp = vec(), b0 = vec(), ones = ones1,
                Vm = similar(csc.nzVal, T, n, m + 1), hd = similar(csc.nzVal, T, 2 * (m + 1)),
                H = zeros(T, m + 1, m))
        rowsum = vec()
    end
    return ShiftState(vals, csc.nzVal, n, opt.n_primal, d, r,
                      _dev(csc.rowVal, rowptr), _dev(csc.rowVal, colval), _dev(csc.rowVal, src),
                      _dev(csc.rowVal, trow), _dev(csc.rowVal, tcol), dual, rowsum, gm, 0, 0)
end

KA.@kernel function _rowmax_kernel!(r, @Const(rowptr), @Const(colval), @Const(src), @Const(vals), @Const(d))
    i = KA.@index(Global)
    m = zero(eltype(r))
    for k in rowptr[i]:(rowptr[i + 1] - 1)
        m = max(m, abs(vals[src[k]]) * d[colval[k]])
    end
    r[i] = m * d[i]
end

KA.@kernel function _scale_kernel!(vals, @Const(trow), @Const(tcol), @Const(d))
    k = KA.@index(Global)
    vals[k] *= d[trow[k]] * d[tcol[k]]
end

KA.@kernel function _shift_kernel!(vals, @Const(pos), delta)
    i = KA.@index(Global)
    vals[pos[i]] -= delta
end

# y = A x (ABS = false) or |A| x (ABS = true) over the full pattern with the given stored values
KA.@kernel function _spmv_kernel!(y, @Const(rowptr), @Const(colval), @Const(src), @Const(vals), @Const(x), ::Val{ABS}) where {ABS}
    i = KA.@index(Global)
    acc = zero(eltype(y))
    for k in rowptr[i]:(rowptr[i + 1] - 1)
        a = vals[src[k]]
        acc += (ABS ? abs(a) : a) * x[colval[k]]
    end
    y[i] = acc
end

function _ruiz!(S::ShiftState, be, sweeps)
    fill!(S.d, one(eltype(S.d)))
    kernel! = _rowmax_kernel!(be, 256)
    for _ in 1:sweeps
        kernel!(S.r, S.frowptr, S.fcolval, S.fsrc, S.vals, S.d; ndrange = S.n)
        S.d .= ifelse.(S.r .> 0, S.d ./ sqrt.(S.r), S.d)
    end
    return S.d
end

# refresh the owned values from the user's matrix: equilibrate, then shift the dual diagonal
function _prepare_values!(M::SDSSolver)
    S = M.shift
    S === nothing && return nothing
    be = KA.get_backend(S.vals)
    copyto!(S.vals, S.truevals)
    if S.d !== nothing
        _ruiz!(S, be, M.opt.sds_ruiz_sweeps)
        _scale_kernel!(be, 256)(S.vals, S.trow, S.tcol, S.d; ndrange = length(S.vals))
    end
    S.dualdiag === nothing ||
        _shift_kernel!(be, 256)(S.vals, S.dualdiag, eltype(S.vals)(M.opt.sds_dual_shift); ndrange = length(S.dualdiag))
    S.gm === nothing ||
        _spmv_kernel!(be, 256)(S.rowsum, S.frowptr, S.fcolval, S.fsrc, S.truevals, S.gm.ones, Val(true); ndrange = S.n)
    KA.synchronize(be)
    return nothing
end

# out = D M^-1 D rhs (D = I without equilibration); out may alias rhs
function _apply!(M::SDSSolver, out, rhs)
    S = M.shift
    scaled = S !== nothing && S.d !== nothing
    scaled ? (M.b .= S.d .* rhs) : copyto!(M.b, rhs)
    Fused.algo2_solve!(M.x, M.inner, M.b, M.inv11, M.inv_ptr, M.fz)
    scaled ? (out .= S.d .* M.x) : copyto!(out, M.x)
    return out
end

# r = b - A x on the user's (unshifted, unscaled) matrix
function _residual!(M::SDSSolver, r, x, b)
    S = M.shift
    _spmv_kernel!(KA.get_backend(r), 256)(r, S.frowptr, S.fcolval, S.fsrc, S.truevals, x, Val(false); ndrange = S.n)
    r .= b .- r
    return r
end

# componentwise backward error max_i |r_i| / (|A| |x| + |b|)_i; rows whose denominator is below
# n u max(denominator) use their natural scale ||A_i||_1 ||x||_inf instead (LAPACK's dgerfs rule)
function _omega!(M::SDSSolver, G, x, r, b)
    S = M.shift
    T = eltype(x)
    G.tmp .= abs.(x)
    _spmv_kernel!(KA.get_backend(x), 256)(G.z, S.frowptr, S.fcolval, S.fsrc, S.truevals, G.tmp, Val(true); ndrange = S.n)
    xinf = maximum(abs, x)
    G.tmp .= G.z .+ abs.(b)
    thr = T(S.n) * eps(T) * maximum(G.tmp)
    G.z .= ifelse.(G.tmp .> thr, G.tmp, G.z .+ S.rowsum .* xinf)
    G.z .= abs.(r) ./ (G.z .+ floatmin(T))
    return maximum(G.z)
end

function _givens_col!(H, cs, sn, g, j)
    for i in 1:(j - 1)
        t = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
        H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
        H[i, j] = t
    end
    dd = hypot(H[j, j], H[j + 1, j])
    if dd > 0
        cs[j] = H[j, j] / dd; sn[j] = H[j + 1, j] / dd
    else
        cs[j] = one(dd); sn[j] = zero(dd)
    end
    H[j, j] = dd; H[j + 1, j] = zero(dd)
    g[j + 1] = -sn[j] * g[j]
    g[j] = cs[j] * g[j]
    return nothing
end

# one left-preconditioned GMRES cycle on A dx = r from dx = 0 (CGS2 orthogonalization, Givens
# residual estimate); stops at 1e-4 of the initial estimate or on breakdown. Adds dx to x.
function _gmres_cycle!(M::SDSSolver, G, x, r)
    T = eltype(x)
    mres = M.opt.sds_ir_restart
    Vm, hd, z = G.Vm, G.hd, G.z
    _apply!(M, z, r)
    beta = norm(z)
    (isfinite(beta) && beta > 0) || return 0
    view(Vm, :, 1) .= z ./ beta
    H = G.H
    fill!(H, zero(T))
    g = zeros(T, mres + 1); g[1] = beta
    cs = zeros(T, mres); sn = zeros(T, mres)
    jused = 0
    for j in 1:mres
        _spmv_kernel!(KA.get_backend(x), 256)(G.tmp, M.shift.frowptr, M.shift.fcolval, M.shift.fsrc, M.shift.truevals,
                                              view(Vm, :, j), Val(false); ndrange = M.shift.n)
        _apply!(M, z, G.tmp)
        nz0 = norm(z)
        Vj = view(Vm, :, 1:j)
        h1 = view(hd, 1:j)
        h2 = view(hd, (mres + 1) .+ (1:j))
        mul!(h1, Vj', z); mul!(z, Vj, h1, -one(T), one(T))
        mul!(h2, Vj', z); mul!(z, Vj, h2, -one(T), one(T))
        h1 .+= h2
        hh = Array(h1)
        nrm = norm(z)
        for i in 1:j
            H[i, j] = hh[i]
        end
        H[j + 1, j] = isfinite(nrm) ? nrm : zero(T)
        brk = !isfinite(nrm) || nrm <= sqrt(eps(T)) * nz0
        brk || (view(Vm, :, j + 1) .= z ./ nrm)
        _givens_col!(H, cs, sn, g, j)
        jused = j
        (brk || abs(g[j + 1]) <= T(1.0e-4) * beta) && break
    end
    jused == 0 && return 0
    y = zeros(T, jused)
    for i in jused:-1:1
        acc = g[i]
        for k in (i + 1):jused
            acc -= H[i, k] * y[k]
        end
        y[i] = acc / H[i, i]
    end
    yd = similar(x, T, jused); copyto!(yd, y)
    mul!(x, view(Vm, :, 1:jused), yd, one(T), one(T))
    return jused
end

"""
    shift_stats(M::SDSSolver) -> (; nsolves, nunaccepted)

Recovery counters of the shifted specialization (zeros without it).
"""
shift_stats(M::SDSSolver) = M.shift === nothing ? (; nsolves = 0, nunaccepted = 0) :
    (; nsolves = M.shift.nsolves, nunaccepted = M.shift.nunaccepted)


function MadNLP.factorize!(M::SDSSolver)
    _prepare_values!(M)
    M.last_info = Int(Fused.refact_fused!(M.inner, M.nzval, M.fp))
    M.invert!()
    return M
end

function _solve!(M::SDSSolver, xb)
    S = M.shift
    be = KA.get_backend(xb)
    if S === nothing || S.gm === nothing
        _apply!(M, xb, xb)
        KA.synchronize(be)
        return xb
    end
    G = S.gm
    copyto!(G.b0, xb)
    nb = norm(G.b0)
    (isfinite(nb) && nb != 0) || return xb
    S.nsolves += 1
    x, r = G.x, G.r
    _apply!(M, x, G.b0)
    accepted = false
    for _ in 1:M.opt.sds_ir_cycles
        _residual!(M, r, x, G.b0)
        relres = norm(r) / nb
        omega = _omega!(M, G, x, r, G.b0)
        if (isfinite(omega) && omega <= M.opt.sds_ir_omega_tol) || relres <= M.opt.sds_ir_rtol
            accepted = true
            break
        end
        _gmres_cycle!(M, G, x, r)
    end
    if !accepted
        _residual!(M, r, x, G.b0)
        omega = _omega!(M, G, x, r, G.b0)
        ((isfinite(omega) && omega <= M.opt.sds_ir_omega_tol) || norm(r) / nb <= M.opt.sds_ir_rtol) ||
            (S.nunaccepted += 1)
    end
    copyto!(xb, x)
    KA.synchronize(be)
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
MadNLP.introduce(M::SDSSolver) = M.shift === nothing ? "SparseDirectSolver.jl (fused pipeline)" :
    "SparseDirectSolver.jl (fused pipeline, shifted: equilibrate=$(M.opt.sds_equilibrate) " *
    "dual_shift=$(M.opt.sds_dual_shift) recovery=$(M.opt.sds_ir_mode))"
MadNLP.is_supported(::Type{SDSSolver}, ::Type{Float32}) = true
MadNLP.is_supported(::Type{SDSSolver}, ::Type{Float64}) = true

function __init__()
    setglobal!(SparseDirectSolver, :SDSSolver, SDSSolver)
    setglobal!(SparseDirectSolver, :SDSSolverOptions, SDSSolverOptions)
    return nothing
end

end # module
