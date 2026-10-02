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

"""
    DirectSolver{T, INT, M, B, SY, NU, WS} <: AbstractDirectSolver{T, INT}

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
matrices this needs the conjugated solve of `solve_mode` (T16) and raises
[`NotSupportedError`](@ref) for now.

Implemented at this point (v0.1): structures `"SPD"` (real `T`) and `"HPD"`, a
single matrix (no uniform batch), the phases `"reordering"`,
`"symbolic_factorization"`, `"analysis"`, `"factorization"`,
`"refactorization"`, `"solve"`, `"solve_fwd_perm"`, `"solve_fwd"`,
`"solve_bwd"` and `"solve_bwd_perm"` ([`execute!`](@ref)). Other structures and
phases raise [`NotSupportedError`](@ref) when they are executed.

Fields: `A` (the current [`CSR`](@ref), re-pointed by [`update!`](@ref)),
`structure`, `view` (as given), `options` ([`Options`](@ref), set through
[`setparam!`](@ref)), `backend`, `nbatch`, `fresh_factorization` (`true` until
the first `"factorization"` after an analysis; the `LinearAlgebra` layer then
switches `cholesky!` to `"refactorization"`), `info` (the `"info"` data
parameter), the analysis (`ordering`, host `host_symbolic`, device `symbolic`),
the numeric storage `numeric` and the solve `workspace`.
"""
mutable struct DirectSolver{T, INT, M <: CSR{T, INT}, B <: KernelAbstractions.Backend, SY, NU, WS} <:
               AbstractDirectSolver{T, INT}
    A::M
    structure::Structure
    view::MatrixView
    options::Options
    backend::B
    nbatch::Int
    fresh_factorization::Bool
    info::Int
    stage::Int
    host_rowptr::Vector{INT}
    host_colval::Vector{INT}
    ordering::Union{Nothing, Ordering}
    host_symbolic::Union{Nothing, Symbolic{Int, Vector{Int}}}
    symbolic::Union{Nothing, SY}
    numeric::Union{Nothing, NU}
    workspace::Union{Nothing, WS}
    # explicit parameters only: the default outer constructor would leave SY, NU, WS unbound
    # (they occur only in `Union{Nothing, …}` fields; Aqua on Julia 1.10)
    function DirectSolver{T, INT, M, B, SY, NU, WS}(A, structure, view, options, backend, nbatch,
                                                    fresh_factorization, info, stage, host_rowptr,
                                                    host_colval, ordering, host_symbolic, symbolic,
                                                    numeric, workspace) where {T, INT, M, B, SY, NU, WS}
        return new{T, INT, M, B, SY, NU, WS}(A, structure, view, options, backend, nbatch, fresh_factorization,
                                             info, stage, host_rowptr, host_colval, ordering, host_symbolic,
                                             symbolic, numeric, workspace)
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
    if A.transposed && T <: Complex && _is_hermitian(s)
        throw(NotSupportedError("a complex Hermitian matrix given as CSC (the CSR of its transpose) needs the " *
                                "conjugated solve of solve_mode (T16); pass the CSR arrays instead"))
    end
    nb = nbatch(A)
    nb == 1 || throw(NotSupportedError("uniform batches (nbatch = $nb) are not implemented yet (T17)"))
    backend = KernelAbstractions.get_backend(A)
    SY, NU, WS = _symbolic_type(backend, INT), _numeric_type(backend, T), _workspace_type(backend, T)
    return DirectSolver{T, INT, typeof(A), typeof(backend), SY, NU, WS}(
        A, s, v, Options(), backend, nb, true, 0, STAGE_NONE, INT[], INT[], nothing, nothing, nothing, nothing,
        nothing)
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
    if solver.host_symbolic !== nothing
        print(io, ", ", nsupernodes(solver.host_symbolic), " supernodes, nnz(L) = ",
              solver.host_symbolic.partition.nnz_L)
    end
    solver.stage == STAGE_FACTORIZED && print(io, ", info = ", solver.info)
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
  matrix (needs the analysis); sets `"info"` (`0` or the original column of the
  first non-positive pivot) and clears `fresh_factorization`;
* `"refactorization"`: the same with the analysis and storage reused, `"info"`
  reset first (needs a previous `"factorization"`);
* `"solve"`: `X = A⁻¹ B`; the sub-phases `"solve_fwd_perm"` (permute `B` into the
  workspace), `"solve_fwd"` (forward sweep), `"solve_bwd"` (backward sweep) and
  `"solve_bwd_perm"` (inverse permutation into `X`) do the same in four calls
  with the same `X`, `B`.

`X` and `B` are only read by the solve phases (pass anything, e.g. `nothing`,
to the others): vectors, `n × nrhs` matrices, strided vectors or
[`MatrixDescriptor`](@ref)s (row-major when `transposed`) on the solver's
backend; `X === B` is allowed. A failed factorization does not throw (as in
cuDSS): check `getparam(solver, "info")`; solving with it gives garbage.

`"solve_diag"`, `"solve_refinement"`, `"solve_fwd_schur"` and
`"solve_bwd_schur"` raise [`NotSupportedError`](@ref) until their task (T16,
T20); unknown phase strings raise `ArgumentError`. Executing a phase before
the phases it depends on raises [`FactorizationError`](@ref).
With `asynchronous = false` the backend is synchronized before returning
(`KernelAbstractions.synchronize`).
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
    elseif p == PHASE_SOLVE || p == PHASE_SOLVE_FWD_PERM || p == PHASE_SOLVE_FWD || p == PHASE_SOLVE_BWD ||
           p == PHASE_SOLVE_BWD_PERM
        _solve!(solver, p, X, B)
    else
        throw(NotSupportedError("phase \"$phase\" is not implemented yet " *
                                "($(p == PHASE_SOLVE_DIAG || p == PHASE_SOLVE_REFINEMENT ? "T16" : "T20"))"))
    end
    asynchronous || KernelAbstractions.synchronize(solver.backend)
    return nothing
end

# options that no phase implements yet: refuse them instead of silently ignoring them
function _check_analysis_supported(solver::DirectSolver{T}) where {T}
    s = solver.structure
    (s == STRUCTURE_SPD || s == STRUCTURE_HPD) ||
        throw(NotSupportedError("structure \"$(convert(String, s))\" is not implemented yet (LDLᵀ/LDLᴴ: T14/T15, " *
                                "LU: T19); use \"SPD\" or \"HPD\""))
    T <: Complex && s != STRUCTURE_HPD &&
        throw(InvalidValueError("a complex positive definite matrix needs structure \"HPD\""))
    opts = solver.options
    opts.ubatch_size > 1 && throw(NotSupportedError("uniform batches (ubatch_size > 1) are not implemented yet (T17)"))
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
    A = solver.A
    solver.host_rowptr = Array(A.rowptr)
    solver.host_colval = Array(A.colval)
    P = SymmetricPattern(solver.host_rowptr, solver.host_colval, A.nrows, solver.structure;
                         view = _stored_view(solver), index = A.index)
    solver.ordering = compute_ordering(P, solver.options; T)
    solver.host_symbolic = solver.symbolic = solver.numeric = solver.workspace = nothing
    solver.stage = STAGE_REORDERED
    return solver
end

function _symbolic!(solver::DirectSolver{T, INT}) where {T, INT}
    solver.stage >= STAGE_REORDERED ||
        throw(_phase_error("symbolic_factorization", "needs \"reordering\" (or run \"analysis\")"))
    _check_analysis_supported(solver)
    opts = solver.options
    A = solver.A
    ord = solver.ordering
    P = SymmetricPattern(solver.host_rowptr, solver.host_colval, A.nrows, solver.structure;
                         view = _stored_view(solver), index = A.index)
    sp = supernode_partition(P, ord.perm, opts)
    sc = build_schedule(sp, opts, T)
    layout = build_layout(sp, sc)
    Sh = Symbolic(sp, sc, layout, solver.host_rowptr, solver.host_colval, A.nrows, solver.structure;
                  view = _stored_view(solver), index = A.index)
    nrhs = solver.workspace === nothing ? 1 : max_rhs(solver.workspace)
    Sd = adapt(solver.backend, Sh, INT)
    solver.host_symbolic = Sh
    solver.symbolic = Sd
    solver.numeric = allocate_numeric(Sd, T, solver.backend)
    solver.workspace = allocate_solve(Sd, T, solver.backend, nrhs)
    solver.stage = STAGE_ANALYZED
    solver.fresh_factorization = true
    solver.info = 0
    return solver
end

function _factorize!(solver::DirectSolver, p::Phase)
    name = convert(String, p)
    solver.stage >= STAGE_ANALYZED || throw(_phase_error(name, "needs \"analysis\""))
    p == PHASE_REFACTORIZATION && solver.stage < STAGE_FACTORIZED &&
        throw(_phase_error(name, "needs a previous \"factorization\""))
    p == PHASE_REFACTORIZATION && (solver.info = 0)
    solver.info = _numeric_phase!(solver.numeric, solver.symbolic, solver.A.nzval)
    solver.stage = STAGE_FACTORIZED
    p == PHASE_FACTORIZATION && (solver.fresh_factorization = false)
    return solver
end

# function barrier: concrete storage types for the numeric phase
_numeric_phase!(N::Numeric, S::Symbolic, nzval::AbstractVector) = factorize!(N, S, nzval)
_numeric_phase!(N::Numeric, S::Symbolic, nzval::AbstractMatrix) = factorize!(N, S, vec(nzval))

# user data of a right-hand side / solution: (array, transposed)
_rhs_data(X::AbstractVecOrMat) = (X, false)
function _rhs_data(X::MatrixDescriptor)
    X.data === nothing && throw(InvalidValueError("MatrixDescriptor has no data; call update! first"))
    X.nbatch == 1 || throw(NotSupportedError("batched right-hand sides are not implemented yet (T17)"))
    X.data isa AbstractVecOrMat || throw(NotSupportedError("3-D right-hand sides are not implemented yet (T17)"))
    return (X.data, X.transposed)
end
_rhs_data(X::AbstractArray) = throw(NotSupportedError("$(ndims(X))-D right-hand sides are not implemented yet (T17)"))
_rhs_data(X) = throw(InvalidValueError("the solve phases need a vector, a matrix or a MatrixDescriptor, got $(typeof(X))"))

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

function _warn_ir_ignored(n)
    @warn "ir_n_steps = $n: iterative refinement is not implemented yet (T16); the solve runs without it" maxlog = 1
    return nothing
end

function _solve!(solver::DirectSolver{T}, p::Phase, X, B) where {T}
    name = convert(String, p)
    solver.stage >= STAGE_FACTORIZED || throw(_phase_error(name, "needs a \"factorization\""))
    opts = solver.options
    opts.solve_alg == SOLVE_DEFAULT ||
        throw(NotSupportedError("solve_alg = \"$(convert(String, opts.solve_alg))\" is not implemented yet (M11)"))
    # SPD/HPD: A = Aᴴ, and A = Aᵀ for real T; the transposed solve of a complex matrix needs T16
    opts.solve_mode == 1 && T <: Complex &&
        throw(NotSupportedError("solve_mode = 1 (Aᵀ) for complex matrices is not implemented yet (T16)"))
    opts.ir_n_steps > 0 && _warn_ir_ignored(opts.ir_n_steps)
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
    n = S.n
    nrhs = rhs_count(B, n; transposed)
    rhs_count(X, n; transposed) == nrhs ||
        throw(DimensionMismatch("X has $(rhs_count(X, n; transposed)) right-hand sides, B has $nrhs"))
    ws = _workspace!(solver, nrhs)
    det = solver.options.deterministic_mode == 1
    if p == PHASE_SOLVE
        sweep_solve!(X, ws, S, N, B; transposed, deterministic = det)
    elseif p == PHASE_SOLVE_FWD_PERM
        permute_rhs!(ws.Y, B, S.perm; transposed)
    elseif p == PHASE_SOLVE_FWD
        forward_sweep!(ws, S, N; nrhs, deterministic = det)
    elseif p == PHASE_SOLVE_BWD
        backward_sweep!(ws, S, N; nrhs)
    else  # PHASE_SOLVE_BWD_PERM
        unpermute_solution!(X, ws.Y, S.perm; transposed)
    end
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

# data parameters the v0.1 solver computes (PLAN §1.4)
const SOLVER_OUTPUTS = ("lu_nnz", "flops", "nsuperpanels", "memory_estimates", "perm_reorder_row",
                        "perm_reorder_col", "perm_row", "perm_col", "diag")

# task that provides the other computed data parameters
function _output_task(name)
    name in ("npivots", "inertia", "pivot_stats") && return "T14/T15"
    name in ("perm_matching", "scale_row", "scale_col") && return "T21"
    name in ("schur_shape", "schur_matrix") && return "T20"
    name == "nd_partition_tree" && return "T24"
    name == "ubatch_mask" && return "T17"
    return "M12"   # hybrid_device_memory_min
end

"""
    setparam!(solver::DirectSolver, name::String, value)

Set a configuration or user-input data parameter of `solver` (≅ `cudss_set`):
everything [`setparam!`](@ref)`(::Options, …)` accepts, stored in
`solver.options` and used by the next phase that reads it, plus the data
parameter `"info"` (an integer; CUDSS.jl resets it before a refactorization).
Computed data parameters (`"lu_nnz"`, `"diag"`, `"perm_row"`, …) cannot be set
(`ArgumentError`): read them with [`getparam`](@ref) or [`getparam!`](@ref),
which replaces cuDSS's set-buffer-then-get protocol.
"""
function setparam!(solver::DirectSolver, name::AbstractString, value)
    if name == "info"
        (value isa Integer && !(value isa Bool)) ||
            throw(InvalidValueError("invalid value $(repr(value)) for parameter \"info\"; expected an integer"))
        solver.info = Int(value)
    elseif name == "schur_matrix"
        throw(NotSupportedError("the data parameter \"schur_matrix\" is not implemented yet (T20)"))
    else
        setparam!(solver.options, name, value)
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
| `"lu_nnz"` | `Int64`: nonzeros of `L` (diagonal included, amalgamation zeros excluded) | analysis |
| `"flops"` | `Float64`: factorization flops of the stored panels | analysis |
| `"nsuperpanels"` | `Int`: supernodes after amalgamation | analysis |
| `"memory_estimates"` | `Vector{Int64}` (16 entries, see [`memory_estimates`](@ref)) | analysis |
| `"perm_reorder_row"`, `"perm_reorder_col"` | `Vector{Int}`: the fill-reducing permutation, 1-based (`perm[k]` = original index of the `k`-th pivot) | reordering |
| `"perm_row"`, `"perm_col"` | `Vector{Int}`: the final permutation of the factor (= the reordering for Cholesky) | analysis |
| `"diag"` | vector of `T` on the solver's backend: the diagonal of `L` in factor order | factorization |

The reordering permutation after `"reordering"` alone is the ordering
algorithm's; `"symbolic_factorization"` composes it with the supernodal
renumbering, and from then on both permutations are the one the factor uses.
Data parameters of later tasks (`"npivots"`, `"inertia"`, `"perm_matching"`,
`"scale_row"`, `"schur_shape"`, …) raise [`NotSupportedError`](@ref); reading
one before the phase that computes it raises [`FactorizationError`](@ref).
"""
function getparam(solver::DirectSolver, name::AbstractString)
    spec = parameter_spec(name)
    name == "info" && return solver.info
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

@kernel function _factor_diag_kernel!(d, factor, super_ptr, front_ptr, front_nrows, ns)
    s = @index(Global, Linear)
    @inbounds if s <= ns
        c0 = Int(super_ptr[s])
        w = Int(super_ptr[s + 1]) - c0
        f = Int(front_nrows[s])
        p0 = Int(front_ptr[s])
        for j in 0:(w - 1)
            d[c0 + j] = factor[p0 + j * f + j]
        end
    end
end

# diagonal of L in factor order, on the solver's backend (one launch, one work item per supernode)
function _factor_diag(solver::DirectSolver{T}) where {T}
    S, N = solver.symbolic, solver.numeric
    d = KernelAbstractions.zeros(solver.backend, T, S.n)
    ns = nsupernodes(S)
    ns > 0 || return d
    _factor_diag_kernel!(solver.backend, 64)(d, N.factor, S.super_ptr, S.front_ptr, S.front_nrows, ns; ndrange = ns)
    return d
end
