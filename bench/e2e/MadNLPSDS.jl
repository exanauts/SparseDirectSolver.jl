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

"""
    SDSTuning

Every performance knob of the fused pipeline in one immutable bundle, so a
platform's tuning is a single value (see bench/ARCH-TUNING.md for how each
field was measured and the per-architecture champions). All fields are
runtime parameters — no environment variables, no recompilation.
"""
Base.@kwdef struct SDSTuning
    wides::Symbol = :auto                     # :auto | :host | :merged
    subtree_budgets::Vector{Int} = [16384]    # regime-A local-memory class(es)
    subtree_parallelism::Int = 16384
    regime_c_width::Int = 32
    regime_c_rows::Int = 64
    amalgamation::NamedTuple = (max_width = 48, zero_fraction = 0.25, min_width = 8)
    workgroup::Int = 128                      # fused-kernel workgroup size
    spin_cap::Int = 128                       # dependency-counter backoff cap
    nd_nseps::Int = 4                         # METIS knobs (78k-ACOPF champions)
    nd_seed::Int = 3
end

"""
    default_tuning(backend) -> SDSTuning

The measured per-architecture defaults — the single place they live.
2048-threads/SM NVIDIA parts (sm_70/80/90): host wides, 16K budgets.
1536-thread parts (sm_86/89, Blackwell workstation): merged wides, 8K budgets.
AMD (gfx906-measured): merged, 8K, workgroup 256, spin cap 16.
"""
function default_tuning(backend)
    if occursin("CUDABackend", string(typeof(backend)))
        M = Base.moduleroot(parentmodule(typeof(backend)))   # CUDACore under the CUDA 6 split
        small = M.attribute(M.device(), M.DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR) < 2048
        return small ? SDSTuning(; wides = :merged, subtree_budgets = [8192]) :
                       SDSTuning(; wides = :host)
    end
    return SDSTuning(; wides = :merged, subtree_budgets = [8192], workgroup = 256, spin_cap = 16)
end

_resolve(t::SDSTuning, backend) = t.wides === :auto ?
    SDSTuning(t; wides = default_tuning(backend).wides) : t
SDSTuning(t::SDSTuning; wides) = SDSTuning(wides, t.subtree_budgets, t.subtree_parallelism,
    t.regime_c_width, t.regime_c_rows, t.amalgamation, t.workgroup, t.spin_cap, t.nd_nseps, t.nd_seed)

Base.@kwdef mutable struct SDSSolverOptions <: MadNLP.AbstractOptions
    sds_reordering::String = "algo4"          # native ND
    # ALL performance knobs live in the tuning bundle; nothing (the default)
    # selects the measured per-architecture profile from default_tuning(backend)
    sds_tuning::Union{Nothing, SDSTuning} = nothing
    # a user permutation bypasses the reordering entirely (1-based, length n)
    sds_user_perm::Union{Nothing, Vector{Int}} = nothing
    # on-disk permutation cache keyed by pattern + ordering knobs ("" disables)
    sds_perm_cache::String = joinpath(homedir(), ".julia", "sds_perm_cache")
end

# merged in-kernel wides win on parts with < 2048 threads/SM (and on AMD); the CUDA
# module is reached through the backend's parent module, so this file stays backend-free
function _perm_cache_file(opt::SDSSolverOptions, t::SDSTuning, csc)
    (opt.sds_perm_cache == "" || opt.sds_user_perm !== nothing) && return nothing
    n = size(csc, 1)
    h = hash((n, length(csc.rowVal), hash(Array(csc.colPtr)), hash(Array(csc.rowVal)),
              opt.sds_reordering, t.nd_nseps, t.nd_seed))
    return joinpath(opt.sds_perm_cache, "perm-" * string(h, base = 16) * ".bin")
end

# shared solver construction knobs: the custom reordering (nd knobs or a user
# permutation) and the amalgamation, applied before the analysis
# returns the cache file to WRITE after the analysis (nothing: no write needed)
function _apply_reordering!(s, opt::SDSSolverOptions, t::SDSTuning, csc)
    s.options.amalgamation = t.amalgamation
    if opt.sds_user_perm !== nothing
        SDS.setparam!(s, "user_perm", opt.sds_user_perm)
        return nothing
    end
    cf = _perm_cache_file(opt, t, csc)
    if cf !== nothing && isfile(cf)
        n = size(csc, 1)
        perm = Vector{Int32}(undef, n)
        read!(cf, perm)
        SDS.setparam!(s, "user_perm", Int.(perm))
        return nothing
    end
    SDS.setparam!(s, "reordering_alg", opt.sds_reordering)
    SDS.setparam!(s, "nd_nseps", t.nd_nseps)
    SDS.setparam!(s, "nd_seed", t.nd_seed)
    return cf
end

function _save_perm(cf, s)
    cf === nothing && return nothing
    perm = SDS.getparam(s, "perm_reorder_row")
    mkpath(dirname(cf))
    open(cf, "w") do io
        write(io, Int32.(perm))
    end
    return nothing
end

# ---------------------------------------------------------------------------
# THE solver: fused factorization + partitioned-inverse/fused solve (PR #107 kernels,
# October 2026 per-architecture tuning; the stock-library wrapper was retired after
# the fused path won on every platform — see bench/PERFORMANCE-2026-10.md)
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
    SDSProto.BACKEND[] = backend
    t = _resolve(something(opt.sds_tuning, default_tuning(backend)), backend)
    s = SDS.DirectSolver(csc.colPtr, csc.rowVal, csc.nzVal, "SPD", 'U')
    s.options.regime_c_width = t.regime_c_width
    s.options.regime_c_rows = t.regime_c_rows  # prototype kernels own BOTH phases: no coupling conflict
    s.options.subtree_parallelism = t.subtree_parallelism
    s.options.subtree_budgets = copy(t.subtree_budgets)
    cf = _apply_reordering!(s, opt, t, csc)
    x = similar(csc.nzVal, n)
    b = similar(csc.nzVal, n)
    SDS.execute!("analysis", s, x, b; asynchronous = false)
    _save_perm(cf, s)
    SDS.execute!("factorization", s, x, b; asynchronous = false)
    SDS.execute!("solve", s, x, b; asynchronous = false)   # allocates the solve workspace
    nzval = SDS._factor_values(s)
    merged = t.wides === :merged
    fp = SDSProto.build_fused_plan(s; inkernel_wides = merged, merge_segments = merged,
                                   wgf = t.workgroup, scap = t.spin_cap)
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
    M = SDSSolver{T}(s, csc, x, b, nzval, fp, fz, inv11, inv_ptr, invertf, 0, opt, logger)
    MadNLP.factorize!(M)                       # prime the fused path once
    return M
end

function MadNLP.factorize!(M::SDSSolver)
    M.last_info = Int(SDSProto.refact_fused!(M.inner, M.nzval, M.fp))
    M.invert!()
    return M
end

function MadNLP.solve_linear_system!(M::SDSSolver{T}, xb) where {T}
    copyto!(M.b, xb)
    SDSProto.algo2_solve!(M.x, M.inner, M.b, M.inv11, M.inv_ptr, M.fz)
    copyto!(xb, M.x)
    KA.synchronize(KA.get_backend(xb))
    return xb
end

MadNLP.solve!(M::SDSSolver, xb::AbstractVector) = MadNLP.solve_linear_system!(M, xb)
MadNLP.input_type(::Type{SDSSolver}) = :csc
MadNLP.default_options(::Type{SDSSolver}) = SDSSolverOptions()
MadNLP.is_inertia(::SDSSolver) = true
function MadNLP.inertia(M::SDSSolver)
    n = size(M.tril, 1)
    return M.last_info == 0 ? (n, 0, 0) : (n - 2, 1, 1)
end
MadNLP.improve!(::SDSSolver) = false
MadNLP.introduce(::SDSSolver) = "SparseDirectSolver.jl (PR #107 prototype kernels)"
MadNLP.is_supported(::Type{SDSSolver}, ::Type{Float32}) = true
MadNLP.is_supported(::Type{SDSSolver}, ::Type{Float64}) = true

const SDSProtoSolver = SDSSolver     # the former name; the prototype is the solver now

export SDSSolver, SDSSolverOptions, SDSProtoSolver

end # module
