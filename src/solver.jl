# Handle-style public API (PLAN §1.2–§1.4, §3.1, §3.2): `DirectSolver`, the
# phase strings of `execute!` (≅ `cudss(phase, solver, x, b)`), the named
# phase wrappers, `update!` (≅ `cudss_update`) and `setparam!`/`getparam`/
# `getparam!` on a solver (≅ `cudss_set`/`cudss_get`).
#
# A solver moves through the stages none → reordered → analyzed → factorized.
# "reordering" computes the fill-reducing ordering on the host;
# "symbolic_factorization" builds supernodes, schedule, layout and maps, moves
# the maps to the device and allocates the numeric storage and a solve
# workspace (the only allocations; growing the number of right-hand sides
# reallocates the workspace once); "factorization"/"refactorization" and the
# solve phases then run without allocation (PLAN §3.9).

"""
    AbstractDirectSolver{T, INT} <: LinearAlgebra.Factorization{T}

Supertype of the solvers of SparseDirectSolver.jl (≅ `AbstractCudssSolver`).
"""
abstract type AbstractDirectSolver{T, INT} <: LinearAlgebra.Factorization{T} end

# stages of a solver, in phase order
const STAGE_NONE = 0
const STAGE_REORDERED = 1
const STAGE_ANALYZED = 2
const STAGE_FACTORIZED = 3

# device array types of a backend (concrete field types of `DirectSolver`)
_symbolic_type(backend, ::Type{INT}) where {INT} =
    Symbolic{INT, typeof(KernelAbstractions.allocate(backend, INT, 0))}
_numeric_type(backend, ::Type{T}) where {T} =
    Numeric{T, typeof(KernelAbstractions.allocate(backend, T, 0)), typeof(KernelAbstractions.allocate(backend, Int64, 0)),
            typeof(KernelAbstractions.allocate(backend, Int32, 0)), typeof(KernelAbstractions.allocate(backend, Int8, 0))}
_workspace_type(backend, ::Type{T}) where {T} = SolveWorkspace{T, typeof(KernelAbstractions.allocate(backend, T, 0, 0))}
_refinement_type(backend, ::Type{T}, ::Type{INT}) where {T, INT} =
    RefinementWorkspace{T, real(T), typeof(KernelAbstractions.allocate(backend, INT, 0)),
                        typeof(KernelAbstractions.allocate(backend, T, 0, 0)),
                        typeof(KernelAbstractions.allocate(backend, real(T), 0))}

"""
    DirectSolver{T, INT, M, B, SY, NU, WS, RF} <: AbstractDirectSolver{T, INT}

Sparse direct solver handle (≅ `CudssSolver`), PLAN §3.1–§3.2.

    DirectSolver(A::CSR, structure::String, view::Char; index = A.index)
    DirectSolver(rowptr, colval, nzval, structure::String, view::Char; index = 'O')
    DirectSolver(A::SparseMatrixCSC, structure::String, view::Char; index = 'O')
    DirectSolver(A::CuSparseMatrixCSR, structure, view; index = 'O')   # CUDA extension
    DirectSolver(A::CuSparseMatrixCSC, structure, view; index = 'O')   # CUDA extension

`structure` is `"G"`, `"S"`, `"H"`, `"SPD"` or `"HPD"` and `view` is `'L'`,
`'U'` or `'F'` (the triangle of the matrix that is read), as in CUDSS.jl;
`index` (`'O'`/`'Z'`) is the base of the row pointers and column indices. The
CSR arrays are wrapped without copies and live on a KernelAbstractions backend
(the solver's `backend`), except for a `SparseMatrixCSC`, which is converted to
host CSR arrays once. A CSC matrix (`CuSparseMatrixCSC`, or a [`CSR`](@ref) with
`transposed = true`, e.g. from [`csr_of_transpose`](@ref)) is read as the CSR
of its transpose with the view flipped (`'L'` ↔ `'U'`); for complex Hermitian
matrices that transpose is the conjugate, and the solve phases conjugate the
right-hand side and the solution (as for `solve_mode`, see
[`solve_conjugated`](@ref)).

Implemented at this point: structures `"SPD"` (real `T`), `"HPD"`, `"S"`
(LDLᵀ; complex symmetric for complex `T`) and `"H"` (LDLᴴ), single matrices
and uniform batches (see below), the phases `"reordering"`, `"symbolic_factorization"`,
`"analysis"`, `"factorization"`, `"refactorization"`, `"solve"`,
`"solve_fwd_perm"`, `"solve_fwd"`, `"solve_diag"`, `"solve_bwd"`,
`"solve_bwd_perm"` and `"solve_refinement"` ([`execute!`](@ref)). Structure
`"G"` and the Schur phases raise [`NotSupportedError`](@ref) when they are
executed.

Uniform batch (PLAN §1.6, §3.5, ≅ CUDSS.jl's uniform batch): `nbatch`
matrices with the pattern of `rowptr`/`colval` and values `nzval`, either a
vector of `nbatch · nnz` entries (member after member) or an `nnz × nbatch`
matrix; `nbatch` is deduced from the values (`length(nzval) ÷ length(colval)`),
and `"ubatch_size"` must be `0` (deduced) or equal to it. One analysis serves
every member; each member has its own factor, `"info"`, `"inertia"`,
`"npivots"` and `"pivot_stats"` (vectors over the members). Right-hand sides
and solutions hold `nrhs` columns per member, member after member: a strided
vector of `n · nrhs · nbatch` entries, an `n × (nrhs · nbatch)` matrix, an
`n × nrhs × nbatch` array or a [`MatrixDescriptor`](@ref) with `nbatch`.
`"ubatch_index"` (0-based member, `-1` = all) and `"ubatch_mask"` (one 0/1
flag per member) restrict the factorization and solve phases to a subset of
the members: the other members' factors and solutions are left untouched.
`"pivot_sign"` and the other options apply to every member; the 2×2 pivot
pairs of `"S"`/`"H"` (`"pivot_pairs"`) come from the first member's values.

Fields: `A` (the current [`CSR`](@ref), re-pointed by [`update!`](@ref)),
`structure`, `view` (as given), `options` ([`Options`](@ref), set through
[`setparam!`](@ref)), `backend`, `nbatch`, `fresh_factorization` (`true` until
the first `"factorization"` after an analysis; the `LinearAlgebra` layer then
switches `cholesky!` to `"refactorization"`), `info` (the `"info"` data
parameter), the analysis (`ordering`, host `host_symbolic`, device `symbolic`),
the numeric storage `numeric`, the solve `workspace`, the refinement storage
`refinement` (allocated by the first solve that refines) and `ir_steps` (the
`"ir_n_steps"` data parameter: refinement steps of the last solve, `-1` before
one).
"""
mutable struct DirectSolver{T, INT, M <: CSR{T, INT}, B <: KernelAbstractions.Backend, SY, NU, WS, RF} <:
               AbstractDirectSolver{T, INT}
    A::M
    structure::Structure
    view::MatrixView
    options::Options
    backend::B
    nbatch::Int
    fresh_factorization::Bool
    info::Vector{Int}
    stage::Int
    host_rowptr::Vector{INT}
    host_colval::Vector{INT}
    ordering::Union{Nothing, Ordering}
    host_symbolic::Union{Nothing, Symbolic{Int, Vector{Int}}}
    symbolic::Union{Nothing, SY}
    numeric::Union{Nothing, NU}
    workspace::Union{Nothing, WS}
    refinement::Union{Nothing, RF}
    ir_steps::Int
    # explicit parameters only: the default outer constructor would leave SY, NU, WS, RF unbound
    # (they occur only in `Union{Nothing, …}` fields; Aqua on Julia 1.10)
    function DirectSolver{T, INT, M, B, SY, NU, WS, RF}(A, structure, view, options, backend, nbatch,
                                                        fresh_factorization, info, stage, host_rowptr,
                                                        host_colval, ordering, host_symbolic, symbolic,
                                                        numeric, workspace, refinement,
                                                        ir_steps) where {T, INT, M, B, SY, NU, WS, RF}
        return new{T, INT, M, B, SY, NU, WS, RF}(A, structure, view, options, backend, nbatch, fresh_factorization,
                                                 info, stage, host_rowptr, host_colval, ordering, host_symbolic,
                                                 symbolic, numeric, workspace, refinement, ir_steps)
    end
end

const SOLVER_ELTYPES = Union{Float32, Float64, ComplexF32, ComplexF64}

function DirectSolver(A::CSR{T, INT}, structure, view; index = A.index) where {T, INT}
    T <: SOLVER_ELTYPES ||
        throw(NotSupportedError("element type $T is not supported; use Float32, Float64, ComplexF32 or ComplexF64"))
    INT <: Union{Int32, Int64} || throw(NotSupportedError("index type $INT is not supported; use Int32 or Int64"))
    s = _structure(structure)
    v = _matrix_view(view)
    base = _index_base(index)
    A.nrows == A.ncols || throw(InvalidValueError("the matrix must be square, got $(A.nrows) × $(A.ncols)"))
    if base != A.index
        A = CSR(A.rowptr, A.colval, A.nzval, A.nrows, A.ncols; index = base, transposed = A.transposed)
    end
    nb = nbatch(A)
    backend = KernelAbstractions.get_backend(A)
    SY, NU, WS = _symbolic_type(backend, INT), _numeric_type(backend, T), _workspace_type(backend, T)
    RF = _refinement_type(backend, T, INT)
    return DirectSolver{T, INT, typeof(A), typeof(backend), SY, NU, WS, RF}(
        A, s, v, Options(), backend, nb, true, zeros(Int, nb), STAGE_NONE, INT[], INT[], nothing, nothing, nothing,
        nothing, nothing, nothing, -1)
end

function DirectSolver(rowptr::AbstractVector{<:Integer}, colval::AbstractVector{<:Integer}, nzval::AbstractVecOrMat,
                      structure, view; index = INDEX_ONE)
    return DirectSolver(CSR(rowptr, colval, nzval; index), structure, view)
end

DirectSolver(A::SparseMatrixCSC, structure, view; index = INDEX_ONE) =
    DirectSolver(CSR(A; index), structure, view)

Base.size(solver::DirectSolver) = (solver.A.nrows, solver.A.nrows)
Base.size(solver::DirectSolver, d::Integer) = d <= 2 ? solver.A.nrows : 1

const _STAGE_NAMES = ("created", "reordered", "analyzed", "factorized")

function Base.show(io::IO, solver::DirectSolver{T, INT}) where {T, INT}
    print(io, "DirectSolver{", T, ", ", INT, "}(n = ", size(solver, 1), ", nnz = ", nnz(solver.A), ", structure = \"",
          convert(String, solver.structure), "\", view = '", convert(Char, solver.view), "', backend = ",
          nameof(typeof(solver.backend)), ", ", _STAGE_NAMES[solver.stage + 1])
    solver.nbatch > 1 && print(io, ", nbatch = ", solver.nbatch)
    if solver.host_symbolic !== nothing
        print(io, ", ", nsupernodes(solver.host_symbolic), " supernodes, nnz(L) = ",
              solver.host_symbolic.partition.nnz_L)
    end
    solver.stage == STAGE_FACTORIZED && print(io, ", info = ", _info_value(solver))
    print(io, ")")
    return nothing
end

# the view of the stored CSR arrays (a transposed CSR stores the other triangle)
function _stored_view(solver::DirectSolver)
    solver.A.transposed || return solver.view
    solver.view == VIEW_LOWER && return VIEW_UPPER
    solver.view == VIEW_UPPER && return VIEW_LOWER
    return solver.view
end

# ---------------------------------------------------------------------------
# update!

"""
    update!(solver::DirectSolver, A)
    update!(solver::DirectSolver, rowptr, colval, nzval)

Point `solver` at new matrix data without copying (≅ `cudss_update`): a
[`CSR`](@ref) of the same type, a `SparseMatrixCSC` (converted on the host), a
`CuSparseMatrixCSR`/`CuSparseMatrixCSC` (CUDA extension), or raw CSR arrays
(with the index base and orientation of the solver's matrix). The sparsity
pattern must be the one of the analysis (only the sizes are checked, as in
cuDSS); new values are used by the next `"factorization"` or
`"refactorization"`. Raises [`InvalidValueError`](@ref) for a different size,
number of stored entries, index base, orientation or array type.
"""
function update!(solver::DirectSolver{T, INT, M}, A::CSR) where {T, INT, M}
    old = solver.A
    (A.nrows == old.nrows && A.ncols == old.ncols && nnz(A) == nnz(old)) ||
        throw(InvalidValueError("update!: the new matrix is $(A.nrows) × $(A.ncols) with $(nnz(A)) stored entries, " *
                                "the solver has $(old.nrows) × $(old.ncols) with $(nnz(old))"))
    (A.index == old.index && A.transposed == old.transposed) ||
        throw(InvalidValueError("update!: the index base and orientation must match the solver's matrix"))
    nbatch(A) == solver.nbatch ||
        throw(InvalidValueError("update!: $(nbatch(A)) batch members, the solver has $(solver.nbatch)"))
    A isa M || throw(InvalidValueError("update!: the solver holds a $M, got a $(typeof(A))"))
    solver.A = A
    return solver
end

function update!(solver::DirectSolver, rowptr::AbstractVector{<:Integer}, colval::AbstractVector{<:Integer},
                 nzval::AbstractVecOrMat)
    n = solver.A.nrows
    return update!(solver, CSR(rowptr, colval, nzval, n, n; index = solver.A.index,
                               transposed = solver.A.transposed))
end

update!(solver::DirectSolver, A::SparseMatrixCSC) = update!(solver, CSR(A; index = solver.A.index))

# ---------------------------------------------------------------------------
# execute!

_phase_error(phase, msg) = FactorizationError(0, "phase \"$phase\": $msg")

function _parse_phase(phase::AbstractString)
    for (spelling, p) in enum_spellings(Phase)
        spelling == phase && return p
    end
    expected = join((repr(first(p)) for p in enum_spellings(Phase)), ", ")
    throw(ArgumentError("unknown phase \"$phase\"; expected one of $expected"))
end

"""
    execute!(phase::String, solver::DirectSolver, X, B; asynchronous = true) -> nothing

Execute `phase` on `solver` (≅ `cudss(phase, solver, x, b)`), PLAN §1.2:

* `"reordering"`: fill-reducing ordering on the host (`reordering_alg`, `user_perm`, …);
* `"symbolic_factorization"`: supernodes, schedule, layout and device maps;
  allocates the numeric storage and the solve workspace (needs `"reordering"`);
* `"analysis"`: both;
* `"factorization"`: numeric factorization with the current values of the
  matrix (needs the analysis); sets `"info"` (`0` or, for `"SPD"`/`"HPD"`, the
  original column of the first non-positive pivot; LDLᵀ/LDLᴴ perturbs tiny
  pivots instead and always reports `0`, see `"npivots"`) and clears
  `fresh_factorization`;
* `"refactorization"`: the same with the analysis and storage reused, `"info"`
  reset first (needs a previous `"factorization"`);
* `"solve"`: `X = op(A)⁻¹ B` (`solve_mode` 0: `A`, 1: `Aᵀ`, 2: `Aᴴ`), followed by
  `ir_n_steps` steps of iterative refinement; the sub-phases `"solve_fwd_perm"`
  (permute `B` into the workspace), `"solve_fwd"` (forward sweep),
  `"solve_diag"` (`D⁻¹` of LDLᵀ/LDLᴴ, the identity for Cholesky), `"solve_bwd"`
  (backward sweep), `"solve_bwd_perm"` (inverse permutation into `X`) and
  `"solve_refinement"` (refine `X` in place, [`refine!`](@ref)) do the same in
  six calls with the same `X`, `B` (bitwise equal to `"solve"`).

`X` and `B` are only read by the solve phases (pass anything, e.g. `nothing`,
to the others): vectors, `n × nrhs` matrices, strided vectors or
[`MatrixDescriptor`](@ref)s (row-major when `transposed`) on the solver's
backend; `X === B` is allowed in every phase but `"solve_refinement"` (it needs
the original `B`; `"solve"` keeps a copy of `B` when they alias). A failed
factorization does not throw (as in cuDSS): check `getparam(solver, "info")`;
solving with it gives garbage.

Refinement (PLAN §2.5): each step computes the residual `R = B - op(A) X` with
the solver's current matrix values (KA SpMV over the full pattern), and, when
`ir_tol > 0`, stops once `‖Rₖ‖₂ ≤ ir_tol ‖Bₖ‖₂` for every right-hand side `k`
(this costs one host synchronization per step; `ir_tol = 0`, the default, never
stops early and never synchronizes), then adds the correction `op(A)⁻¹ R`. The
steps performed are the data parameter `"ir_n_steps"` ([`getparam`](@ref)).
With `ir_mode = "fgmres"` (needs `using Krylov`, else [`NotSupportedError`](@ref))
the refinement is FGMRES on the correction system, right-preconditioned by the
factorization ([`fgmres_refine!`](@ref)): at most `ir_n_steps` iterations (one
SpMV and one solve each, as a plain step), stopping once every relative
residual is below `ir_tol` (`ir_tol = 0`: all `ir_n_steps` iterations);
`"ir_n_steps"` then reports the FGMRES iterations.

The flag `"user_host_interrupt"` (a `Threads.Atomic{Bool}`, [`setparam!`](@ref))
is polled at the start of the analysis phases, between the launch groups of the
factorization and between refinement steps: when set, the phase raises
[`InterruptedError`](@ref). An interrupted factorization leaves the solver
analyzed (the next phase must be `"factorization"`); an interrupted refinement
leaves the last completed iterate in `X` and reports the steps completed in
`"ir_n_steps"`.

`"solve_fwd_schur"` and `"solve_bwd_schur"` raise [`NotSupportedError`](@ref)
until their task (T20); unknown phase strings raise `ArgumentError`. Executing
a phase before the phases it depends on raises [`FactorizationError`](@ref).
With `asynchronous = false` the backend is synchronized before returning
(`KernelAbstractions.synchronize`). Phase summaries are logged (see
[`SparseDirectSolver.set_log_level!`](@ref)).
"""
function execute!(phase::AbstractString, solver::DirectSolver, X, B; asynchronous::Bool = true)
    p = _parse_phase(phase)
    if p == PHASE_REORDERING
        _reorder!(solver)
    elseif p == PHASE_SYMBOLIC_FACTORIZATION
        _symbolic!(solver)
    elseif p == PHASE_ANALYSIS
        _reorder!(solver)
        _symbolic!(solver)
    elseif p == PHASE_FACTORIZATION || p == PHASE_REFACTORIZATION
        _factorize!(solver, p)
    elseif p == PHASE_SOLVE || p == PHASE_SOLVE_FWD_PERM || p == PHASE_SOLVE_FWD || p == PHASE_SOLVE_DIAG ||
           p == PHASE_SOLVE_BWD || p == PHASE_SOLVE_BWD_PERM || p == PHASE_SOLVE_REFINEMENT
        _solve!(solver, p, X, B)
    else
        throw(NotSupportedError("phase \"$phase\" is not implemented yet (T20)"))
    end
    asynchronous || KernelAbstractions.synchronize(solver.backend)
    return nothing
end

# options that no phase implements yet: refuse them instead of silently ignoring them
function _check_analysis_supported(solver::DirectSolver{T}) where {T}
    s = solver.structure
    s == STRUCTURE_GENERAL &&
        throw(NotSupportedError("structure \"G\" (LU) is not implemented yet (T19); use \"S\", \"H\", \"SPD\" or " *
                                "\"HPD\""))
    T <: Complex && s == STRUCTURE_SPD &&
        throw(InvalidValueError("a complex positive definite matrix needs structure \"HPD\""))
    opts = solver.options
    opts.ubatch_size == 0 || opts.ubatch_size == solver.nbatch ||
        throw(InvalidValueError("ubatch_size = $(opts.ubatch_size), but the matrix values hold $(solver.nbatch) " *
                                "batch member(s) (length(nzval) ÷ nnz)"))
    opts.matching_alg == MATCHING_NONE ||
        throw(NotSupportedError("matching_alg = \"$(convert(String, opts.matching_alg))\" is not implemented yet (T21)"))
    opts.schur_mode == 0 || throw(NotSupportedError("schur_mode = 1 is not implemented yet (T20)"))
    opts.user_nd_partition_tree === nothing ||
        throw(NotSupportedError("user_nd_partition_tree is not implemented yet (T24)"))
    opts.schedule == SCHEDULE_SYNCFREE && throw(NotSupportedError("schedule = \"syncfree\" is not implemented yet"))
    opts.factor_precision === nothing || opts.factor_precision === real(T) ||
        throw(NotSupportedError("factor_precision = $(opts.factor_precision) for $T input is not implemented yet"))
    return nothing
end

function _reorder!(solver::DirectSolver{T}) where {T}
    _check_analysis_supported(solver)
    _poll_interrupt(solver.options.user_host_interrupt)
    tic = time_ns()
    A = solver.A
    solver.host_rowptr = Array(A.rowptr)
    solver.host_colval = Array(A.colval)
    P = SymmetricPattern(solver.host_rowptr, solver.host_colval, A.nrows, solver.structure;
                         view = _stored_view(solver), index = A.index)
    # 2×2 pivot pairs ("S"/"H"): the only host copy of the values, at analysis (of the first batch member)
    pp = analysis_pairs(P, solver.host_rowptr, solver.host_colval, _first_member(A), A.nrows, solver.structure,
                        solver.options; view = _stored_view(solver), index = A.index)
    solver.ordering = compute_ordering(P, solver.options; T, pp.pairs, pp.candidates)
    solver.host_symbolic = solver.symbolic = solver.numeric = solver.workspace = solver.refinement = nothing
    solver.stage = STAGE_REORDERED
    _log(LOG_INFO, () -> "reordering: n = $(A.nrows), nnz = $(nnz(A)), $(_elapsed(tic))")
    return solver
end

# the values of the first batch member (all of them for a single matrix)
_first_member(A::CSR) = nbatch(A) == 1 ? A.nzval : A.nzval isa AbstractMatrix ? view(A.nzval, :, 1) :
                        view(A.nzval, 1:nnz(A))

function _symbolic!(solver::DirectSolver{T, INT}) where {T, INT}
    solver.stage >= STAGE_REORDERED ||
        throw(_phase_error("symbolic_factorization", "needs \"reordering\" (or run \"analysis\")"))
    _check_analysis_supported(solver)
    _poll_interrupt(solver.options.user_host_interrupt)
    tic = time_ns()
    opts = solver.options
    A = solver.A
    ord = solver.ordering
    P = SymmetricPattern(solver.host_rowptr, solver.host_colval, A.nrows, solver.structure;
                         view = _stored_view(solver), index = A.index)
    sp = supernode_partition(factor_pattern(P, ord), ord.perm, opts)
    sc = build_schedule(sp, opts, T; reserve = subtree_local_reserve(solver.structure))
    layout = build_layout(sp, sc; ldlt = _is_ldlt_structure(solver.structure))
    Sh = Symbolic(sp, sc, layout, solver.host_rowptr, solver.host_colval, A.nrows, solver.structure;
                  view = _stored_view(solver), index = A.index)
    nrhs = solver.workspace === nothing ? solver.nbatch : max_rhs(solver.workspace)
    Sd = adapt(solver.backend, Sh, INT)
    solver.host_symbolic = Sh
    solver.symbolic = Sd
    solver.numeric = allocate_numeric(Sd, T, solver.backend; nbatch = solver.nbatch)
    solver.workspace = allocate_solve(Sd, T, solver.backend, nrhs)
    solver.refinement = nothing
    solver.stage = STAGE_ANALYZED
    solver.fresh_factorization = true
    fill!(solver.info, 0)
    _log(LOG_INFO, () -> "symbolic_factorization: $(nsupernodes(Sh)) supernodes, nnz(L) = $(sp.nnz_L), " *
                         "flops = $(sp.flops), $(_elapsed(tic))")
    return solver
end

function _factorize!(solver::DirectSolver, p::Phase)
    name = convert(String, p)
    solver.stage >= STAGE_ANALYZED || throw(_phase_error(name, "needs \"analysis\""))
    p == PHASE_REFACTORIZATION && solver.stage < STAGE_FACTORIZED &&
        throw(_phase_error(name, "needs a previous \"factorization\""))
    N, S = solver.numeric, solver.symbolic
    _set_members!(solver)
    if p == PHASE_REFACTORIZATION
        for j in 1:N.plan.nact[]
            solver.info[N.plan.members_host[j]] = 0
        end
    end
    tic = time_ns()
    try
        _numeric_phase!(N, S, solver.A.nzval, solver.options)
    catch err
        if err isa InterruptedError
            # the panels are partly overwritten: back to "analyzed", a "factorization" must follow
            solver.stage = STAGE_ANALYZED
            solver.fresh_factorization = true
            fill!(solver.info, 0)
            _log(LOG_INFO, () -> "$name: interrupted")
        end
        rethrow()
    end
    member_info!(solver.info, N, S)
    solver.stage = STAGE_FACTORIZED
    p == PHASE_FACTORIZATION && (solver.fresh_factorization = false)
    _log(LOG_INFO, () -> "$name: info = $(_info_value(solver)), $(_elapsed(tic)) (launches, asynchronous)")
    return solver
end

# the members the next numeric or solve phase processes (`ubatch_index`, `ubatch_mask`)
function _set_members!(solver::DirectSolver)
    opts = solver.options
    nb = solver.nbatch
    if opts.ubatch_index == -1 && opts.ubatch_mask === nothing
        N = solver.numeric
        N.plan.nact[] == nb || set_members!(N, 1:nb)
        return N
    end
    members = active_members(nb, opts.ubatch_index, opts.ubatch_mask)
    isempty(members) && throw(InvalidValueError("ubatch_index and ubatch_mask select no batch member"))
    return set_members!(solver.numeric, members)
end

# the "info" data parameter: an `Int` for a single matrix, a vector over the members of a batch
_info_value(solver::DirectSolver) = solver.nbatch == 1 ? solver.info[1] : copy(solver.info)

_elapsed(tic) = string(round((time_ns() - tic) / 1.0e6; digits = 3), " ms")

# function barrier: concrete storage types for the numeric phase
_numeric_phase!(N::Numeric, S::Symbolic, nzval::AbstractVector, opts::Options) = factorize!(N, S, nzval; opts)
_numeric_phase!(N::Numeric, S::Symbolic, nzval::AbstractMatrix, opts::Options) = factorize!(N, S, vec(nzval); opts)

# user data of a right-hand side / solution: (array, transposed); an `n × nrhs × nbatch` array is the
# `n × (nrhs nbatch)` matrix with the same memory (uniform batch)
_rhs_data(X::AbstractVecOrMat) = (X, false)
function _rhs_data(X::MatrixDescriptor)
    X.data === nothing && throw(InvalidValueError("MatrixDescriptor has no data; call update! first"))
    X.nbatch == 1 || !X.transposed ||
        throw(NotSupportedError("row-major (transposed) right-hand sides of a uniform batch are not supported"))
    return (_rhs_matrix(X.data), X.transposed)
end
_rhs_data(X::AbstractArray{<:Any, 3}) = (_rhs_matrix(X), false)
_rhs_data(X::AbstractArray) = throw(InvalidValueError("$(ndims(X))-D right-hand sides are not supported"))
_rhs_data(X) = throw(InvalidValueError("the solve phases need a vector, a matrix, an n × nrhs × nbatch array or a " *
                                       "MatrixDescriptor, got $(typeof(X))"))

_rhs_matrix(X::AbstractVecOrMat) = X
_rhs_matrix(X::AbstractArray{<:Any, 3}) = reshape(X, size(X, 1), size(X, 2) * size(X, 3))

function _check_rhs_backend(solver::DirectSolver, X, name)
    typeof(KernelAbstractions.get_backend(X)) == typeof(solver.backend) ||
        throw(InvalidValueError("$name lives on $(typeof(KernelAbstractions.get_backend(X))), the solver on " *
                                "$(typeof(solver.backend))"))
    return nothing
end

# the workspace for `nrhs` right-hand sides (reallocated only when nrhs grows)
function _workspace!(solver::DirectSolver{T}, nrhs::Int) where {T}
    ws = solver.workspace
    max_rhs(ws) >= nrhs && return ws
    ws = allocate_solve(solver.symbolic, T, solver.backend, nrhs)
    solver.workspace = ws
    return ws
end

# the refinement workspace for `nrhs` right-hand sides (the device map is built at the first use after an
# analysis, the residual storage reallocated only when nrhs grows)
function _refinement!(solver::DirectSolver{T, INT}, nrhs::Int) where {T, INT}
    W = solver.refinement
    if W === nothing
        F = full_pattern_map(solver.host_rowptr, solver.host_colval, solver.A.nrows, solver.structure;
                             view = _stored_view(solver), index = solver.A.index)
        W = allocate_refinement(refinement_map(F), T, INT, solver.backend, max(nrhs, max_rhs(solver.workspace)))
    elseif max_rhs(W) < nrhs
        W = _grow_refinement(W, nrhs)
    end
    solver.refinement = W
    return W
end

function _solve!(solver::DirectSolver{T}, p::Phase, X, B) where {T}
    name = convert(String, p)
    solver.stage >= STAGE_FACTORIZED || throw(_phase_error(name, "needs a \"factorization\""))
    opts = solver.options
    opts.solve_alg == SOLVE_DEFAULT ||
        throw(NotSupportedError("solve_alg = \"$(convert(String, opts.solve_alg))\" is not implemented yet (M11)"))
    Bd, bt = _rhs_data(B)
    Xd, xt = _rhs_data(X)
    xt == bt || throw(InvalidValueError("X and B must have the same layout (transposed or not)"))
    eltype(Bd) == T && eltype(Xd) == T ||
        throw(InvalidValueError("X ($(eltype(Xd))) and B ($(eltype(Bd))) must have the solver's element type $T"))
    _check_rhs_backend(solver, Bd, "B")
    _check_rhs_backend(solver, Xd, "X")
    _solve_phase!(solver, p, Xd, Bd, bt)
    return solver
end

function _solve_phase!(solver::DirectSolver, p::Phase, X::AbstractVecOrMat, B::AbstractVecOrMat, transposed::Bool)
    S, N = solver.symbolic, solver.numeric
    opts = solver.options
    n = S.n
    nu = rhs_count(B, n; transposed)                   # user columns: nrhs per member × nbatch
    rhs_count(X, n; transposed) == nu ||
        throw(DimensionMismatch("X has $(rhs_count(X, n; transposed)) right-hand sides, B has $nu"))
    nb = solver.nbatch
    nb == 1 || !transposed ||
        throw(NotSupportedError("row-major (transposed) right-hand sides of a uniform batch are not supported"))
    nu % nb == 0 ||
        throw(DimensionMismatch("$nu right-hand side columns do not split over the $nb members of the batch"))
    nrhs = nu ÷ nb
    _set_members!(solver)
    ws = _workspace!(solver, nu)
    bm = batch_map(N; nrhs)
    det = opts.deterministic_mode == 1
    cj = solve_conjugated(solver.structure, eltype(ws), solver.A.transposed, opts.solve_mode)
    refine = opts.ir_n_steps > 0 && (p == PHASE_SOLVE || p == PHASE_SOLVE_REFINEMENT)
    if p == PHASE_SOLVE_REFINEMENT && refine && Base.mightalias(X, B)
        throw(InvalidValueError("\"solve_refinement\" needs the original right-hand side: X and B must not alias"))
    end
    W = refine ? _refinement!(solver, nu) : nothing
    if p == PHASE_SOLVE
        Bs, bt = B, transposed
        if refine && Base.mightalias(X, B)     # the solve overwrites B: keep a copy for the residual
            copy_rhs!(W.Bc, B; nrhs = nu, transposed)
            Bs, bt = W.Bc, false
        end
        permute_rhs!(ws.Y, B, S.perm; transposed, conjugate = cj, bm)
        forward_sweep!(ws, S, N; nrhs, deterministic = det)
        diagonal_sweep!(ws, S, N; nrhs)
        backward_sweep!(ws, S, N; nrhs)
        unpermute_solution!(X, ws.Y, S.perm; transposed, conjugate = cj, bm)
        refine ? _refine_phase!(solver, W, ws, X, Bs, transposed, bt, cj, det) : (solver.ir_steps = 0)
    elseif p == PHASE_SOLVE_FWD_PERM
        permute_rhs!(ws.Y, B, S.perm; transposed, conjugate = cj, bm)
    elseif p == PHASE_SOLVE_FWD
        forward_sweep!(ws, S, N; nrhs, deterministic = det)
    elseif p == PHASE_SOLVE_DIAG
        diagonal_sweep!(ws, S, N; nrhs)
    elseif p == PHASE_SOLVE_BWD
        backward_sweep!(ws, S, N; nrhs)
    elseif p == PHASE_SOLVE_BWD_PERM
        unpermute_solution!(X, ws.Y, S.perm; transposed, conjugate = cj, bm)
    else  # PHASE_SOLVE_REFINEMENT
        refine ? _refine_phase!(solver, W, ws, X, B, transposed, transposed, cj, det) : (solver.ir_steps = 0)
    end
    return nothing
end

function _refine_phase!(solver::DirectSolver, W::RefinementWorkspace, ws::SolveWorkspace, X, B, xt::Bool, bt::Bool,
                        cj::Bool, det::Bool)
    opts = solver.options
    done = Ref(0)   # the corrections applied, also when the refinement is interrupted
    try
        driver = opts.ir_mode == IR_FGMRES ? fgmres_refine! : refine!
        driver(X, B, W, ws, solver.symbolic, solver.numeric, vec(solver.A.nzval); nsteps = opts.ir_n_steps,
               tol = opts.ir_tol, transposed = xt, b_transposed = bt, conjugate = cj, deterministic = det,
               interrupt = opts.user_host_interrupt, progress = done)
    finally
        solver.ir_steps = done[]
    end
    steps = done[]
    unit = opts.ir_mode == IR_FGMRES ? "FGMRES iterations" : "steps"
    _log(LOG_INFO, () -> "solve_refinement: $steps of $(opts.ir_n_steps) $unit")
    return nothing
end

"""
    analyze!(solver, X = nothing, B = nothing; asynchronous = true) -> solver

`execute!("analysis", solver, X, B; asynchronous)`.
"""
function analyze!(solver::DirectSolver, X = nothing, B = nothing; asynchronous::Bool = true)
    execute!("analysis", solver, X, B; asynchronous)
    return solver
end

"""
    factorize!(solver, X = nothing, B = nothing; asynchronous = true) -> solver

`execute!("factorization", solver, X, B; asynchronous)`.
"""
function factorize!(solver::DirectSolver, X = nothing, B = nothing; asynchronous::Bool = true)
    execute!("factorization", solver, X, B; asynchronous)
    return solver
end

"""
    refactorize!(solver, X = nothing, B = nothing; asynchronous = true) -> solver

`execute!("refactorization", solver, X, B; asynchronous)`.
"""
function refactorize!(solver::DirectSolver, X = nothing, B = nothing; asynchronous::Bool = true)
    execute!("refactorization", solver, X, B; asynchronous)
    return solver
end

"""
    solve!(solver, X, B; asynchronous = true) -> X

`execute!("solve", solver, X, B; asynchronous)`.
"""
function solve!(solver::DirectSolver, X, B; asynchronous::Bool = true)
    execute!("solve", solver, X, B; asynchronous)
    return X
end

# ---------------------------------------------------------------------------
# parameters

# data parameters the solver computes (PLAN §1.4, §1.7)
const SOLVER_OUTPUTS = ("lu_nnz", "flops", "nsuperpanels", "memory_estimates", "perm_reorder_row",
                        "perm_reorder_col", "perm_row", "perm_col", "diag", "npivots", "inertia", "pivot_stats")

# task that provides the other computed data parameters
function _output_task(name)
    name in ("perm_matching", "scale_row", "scale_col") && return "T21"
    name in ("schur_shape", "schur_matrix") && return "T20"
    name == "nd_partition_tree" && return "T24"
    return "M12"   # hybrid_device_memory_min
end

"""
    setparam!(solver::DirectSolver, name::String, value)

Set a configuration or user-input data parameter of `solver` (≅ `cudss_set`):
everything [`setparam!`](@ref)`(::Options, …)` accepts, stored in
`solver.options` and used by the next phase that reads it, plus the data
parameter `"info"` (an integer, or for a uniform batch an integer for every
member or a vector of `nbatch` integers; CUDSS.jl resets it before a
refactorization). Uniform batch: `"ubatch_size"` (`0` or the batch size),
`"ubatch_index"` (0-based member, `-1` = all) and `"ubatch_mask"` (`nbatch`
flags 0/1, or `nothing`) select the members of the next phases; the index and
the mask are checked against the batch size when a phase runs.
`"pivot_sign"` (PLAN §1.7: a vector of `n` entries in `(-1, 0, 1)`, host or
device, or `nothing`) is checked against the size of the matrix; the next
`"factorization"`/`"refactorization"` copies it to the device. Setting
`"ir_n_steps"` (the number of refinement steps requested) makes
[`getparam`](@ref) report that value again until the next solve.
Computed data parameters (`"lu_nnz"`, `"diag"`, `"perm_row"`, …) cannot be set
(`ArgumentError`): read them with [`getparam`](@ref) or [`getparam!`](@ref),
which replaces cuDSS's set-buffer-then-get protocol.
"""
function setparam!(solver::DirectSolver, name::AbstractString, value)
    if name == "info"
        if value isa AbstractVector && solver.nbatch > 1
            length(value) == solver.nbatch && all(x -> x isa Integer && !(x isa Bool), value) ||
                throw(InvalidValueError("invalid value $(repr(value)) for parameter \"info\"; expected " *
                                        "$(solver.nbatch) integers"))
            solver.info .= value
        else
            (value isa Integer && !(value isa Bool)) ||
                throw(InvalidValueError("invalid value $(repr(value)) for parameter \"info\"; expected an integer"))
            fill!(solver.info, Int(value))
        end
    elseif name == "schur_matrix"
        throw(NotSupportedError("the data parameter \"schur_matrix\" is not implemented yet (T20)"))
    elseif name == "pivot_sign" && value !== nothing && length(value) != size(solver, 1)
        throw(InvalidValueError("pivot_sign has $(length(value)) entries, the matrix has $(size(solver, 1)) rows"))
    else
        setparam!(solver.options, name, value)
        name == "ir_n_steps" && (solver.ir_steps = -1)
    end
    return nothing
end

function _need_stage(solver::DirectSolver, stage::Int, name)
    solver.stage >= stage ||
        throw(FactorizationError(0, "the data parameter \"$name\" needs the " *
                                    "$(stage == STAGE_FACTORIZED ? "\"factorization\"" : "\"analysis\"") phase"))
    return nothing
end

"""
    getparam(solver::DirectSolver, name::String)

Value of a parameter of `solver` (≅ `cudss_get`). Configuration and user-input
parameters come from `solver.options` ([`getparam`](@ref)`(::Options, …)`).
The data parameters computed by the solver:

| name | value | available after |
| --- | --- | --- |
| `"info"` | `Int`: `0`, or the original column (1-based) of the first non-positive pivot | always |
| `"ubatch_mask"` | the `"ubatch_mask"` set (or `nothing`) | always |
| `"ir_n_steps"` | `Int`: refinement steps performed by the last `"solve"`/`"solve_refinement"` (≤ the configured `ir_n_steps`, fewer when `ir_tol` stopped early); the configured value before the first solve and after `setparam!(solver, "ir_n_steps", k)` | always |
| `"lu_nnz"` | `Int64`: nonzeros of `L` (diagonal included, amalgamation zeros excluded) | analysis |
| `"flops"` | `Float64`: factorization flops of the stored panels | analysis |
| `"nsuperpanels"` | `Int`: supernodes after amalgamation | analysis |
| `"memory_estimates"` | `Vector{Int64}` (16 entries, see [`memory_estimates`](@ref)) | analysis |
| `"perm_reorder_row"`, `"perm_reorder_col"` | `Vector{Int}`: the fill-reducing permutation, 1-based (`perm[k]` = original index of the `k`-th pivot) | reordering |
| `"perm_row"`, `"perm_col"` | `Vector{Int}`: the final permutation of the factor (= the reordering for Cholesky) | analysis |
| `"diag"` | vector of `T` on the solver's backend: the diagonal of `L` (Cholesky) or of `D` (LDLᵀ/LDLᴴ; for a 2×2 block its two diagonal entries) in factor order | factorization |
| `"npivots"` | `INT`: perturbed pivots (LDLᵀ/LDLᴴ; `0` for Cholesky) | factorization |
| `"inertia"` | `Tuple{INT, INT}`: `(npos, nneg)` of D (after perturbation, so the inertia of `A + E`; read it with `"npivots"`); `(0, 0)` for complex symmetric `"S"`; Cholesky: `(number of positive pivots, 0)` | factorization |
| `"pivot_stats"` | `NamedTuple` `(npos, nneg, nzero, nperturbed, n2x2)` of `Int64` (PLAN §1.7) | factorization |

Uniform batch (`nbatch > 1`): `"info"`, `"npivots"`, `"inertia"` and
`"pivot_stats"` are vectors with one entry per batch member, `"diag"` is the
`n · nbatch` vector of the members' diagonals one after the other; the
analysis outputs are shared by the members.

The pivot statistics are reduced on the device ([`reduce_stats!`](@ref)) and
copied to the host when read (one synchronization). The reordering
permutation after `"reordering"` alone is the ordering algorithm's; `"symbolic_factorization"` composes it with the supernodal
renumbering, and from then on both permutations are the one the factor uses.
Data parameters of later tasks (`"perm_matching"`, `"scale_row"`,
`"schur_shape"`, …) raise [`NotSupportedError`](@ref); reading
one before the phase that computes it raises [`FactorizationError`](@ref).
"""
function getparam(solver::DirectSolver, name::AbstractString)
    spec = parameter_spec(name)
    name == "info" && return _info_value(solver)
    name == "ir_n_steps" && solver.ir_steps >= 0 && return solver.ir_steps
    spec.status === :output || spec.status === :solver || return getparam(solver.options, name)
    name in SOLVER_OUTPUTS ||
        throw(NotSupportedError("the data parameter \"$name\" is not implemented yet ($(_output_task(name)))"))
    if name == "perm_reorder_row" || name == "perm_reorder_col"
        _need_stage(solver, STAGE_REORDERED, name)
        Sh = solver.host_symbolic
        return Sh === nothing ? copy(solver.ordering.perm) : copy(Sh.partition.perm)
    end
    if name == "diag"
        _need_stage(solver, STAGE_FACTORIZED, name)
        return _factor_diag(solver)
    end
    if name in ("npivots", "inertia", "pivot_stats")
        _need_stage(solver, STAGE_FACTORIZED, name)
        return _pivot_output(solver, name)
    end
    _need_stage(solver, STAGE_ANALYZED, name)
    Sh = solver.host_symbolic
    sp = Sh.partition
    name == "lu_nnz" && return Int64(sp.nnz_L)
    name == "flops" && return sp.flops
    name == "nsuperpanels" && return nsuperpanels(sp)
    name == "memory_estimates" && return _memory_estimates(solver)
    return copy(sp.perm)   # perm_row, perm_col
end

_memory_estimates(solver::DirectSolver{T, INT}) where {T, INT} = memory_estimates(solver.host_symbolic, T, INT)

"""
    getparam!(buffer, solver::DirectSolver, name::String) -> buffer

Write the vector-valued data parameter `name` (`"perm_reorder_row"`,
`"perm_reorder_col"`, `"perm_row"`, `"perm_col"`, `"diag"`,
`"memory_estimates"`, `"user_perm"`) of [`getparam`](@ref) into `buffer`, a host
or device vector of the right length (converted to its element type), without
the C-style set-buffer-then-get protocol of CUDSS.jl.
"""
function getparam!(buffer::AbstractVector, solver::DirectSolver, name::AbstractString)
    value = getparam(solver, name)
    value isa AbstractVector ||
        throw(InvalidValueError("getparam!: the parameter \"$name\" is not a vector; use getparam"))
    length(buffer) == length(value) ||
        throw(DimensionMismatch("getparam!: \"$name\" has $(length(value)) entries, the buffer $(length(buffer))"))
    if eltype(value) == eltype(buffer)
        copyto!(buffer, value)
    else
        copyto!(buffer, convert(Vector{eltype(buffer)}, Array(value)))
    end
    return buffer
end

# work item (s, member k of nb): the diagonal of the panel of supernode s of member k into d[(k - 1) n + …]
@kernel function _factor_diag_kernel!(d, factor, super_ptr, front_ptr, front_nrows, ns, nb)
    q = @index(Global, Linear)
    s = (q - 1) % ns + 1
    k = (q - 1) ÷ ns + 1
    @inbounds if k <= nb
        dk = _mview(d, k, nb)
        c0 = Int(super_ptr[s])
        w = Int(super_ptr[s + 1]) - c0
        f = Int(front_nrows[s])
        p0 = Int(member_panels(front_ptr, k, nb)[s])
        for j in 0:(w - 1)
            dk[c0 + j] = factor[p0 + j * f + j]
        end
    end
end

# npivots / inertia / pivot_stats from the statistics reduced on the device (Cholesky: reduced on demand),
# per member for a batch
function _pivot_output(solver::DirectSolver{T, INT}, name) where {T, INT}
    S, N = solver.symbolic, solver.numeric
    if !_is_ldlt_structure(S.structure)
        _set_members!(solver)
        reduce_stats!(N, S)
    end
    totals = Array(N.totals)
    out(st) = name == "npivots" ? INT(st.nperturbed) : name == "inertia" ? (INT(st.npos), INT(st.nneg)) : st
    solver.nbatch == 1 && return out(pivot_totals(totals, 1, 1))
    return [out(pivot_totals(totals, solver.nbatch, k)) for k in 1:solver.nbatch]
end

# diagonal of L (Cholesky) or D (LDLᵀ/LDLᴴ) in factor order, on the solver's backend, the members of a batch one
# after the other (Cholesky: one launch, one work item per (supernode, member))
function _factor_diag(solver::DirectSolver{T}) where {T}
    S, N = solver.symbolic, solver.numeric
    n, nb = S.n, solver.nbatch
    d = KernelAbstractions.zeros(solver.backend, T, n * nb)
    if _is_ldlt_structure(S.structure)
        for k in 1:nb
            copyto!(d, (k - 1) * n + 1, N.d, (k - 1) * 2n + 1, n)
        end
        return d
    end
    ns = nsupernodes(S)
    ns > 0 || return d
    _factor_diag_kernel!(solver.backend, 64)(d, N.factor, S.super_ptr, S.front_ptr, S.front_nrows, ns, nb;
                                             ndrange = ns * nb)
    return d
end
