# KKTSolver: the front end for the linear systems of an interior-point method (one
# symmetric matrix with a fixed pattern, factorized again with new values at every
# iteration, one right-hand side, the inertia read after each factorization). It picks
# the pipeline (the fused one of src/fused.jl where the backend runs it, else the stock
# DirectSolver), sets the parameters of the chosen factorization and drives the phases.
# MadNLP's SparseDirectSolver binding (MadCoreGPU's MadCoreGPUSDSExt) is a shell over it.

"""
    KKTSolver(A, view::Char = 'L'; kwargs...)
    KKTSolver(rowptr, colval, nzval, view::Char = 'L'; kwargs...)

Solver for a real symmetric matrix that is factorized many times with the same
sparsity pattern and solved for one right-hand side at a time, as the KKT
systems of an interior-point method are. `A` is anything
[`DirectSolver`](@ref) accepts (a [`CSR`](@ref), a `SparseMatrixCSC`, a vendor
CSR or CSC matrix such as `CuSparseMatrixCSC`, `ROCSparseMatrixCSC` or
`oneSparseMatrixCSC`) and `view` the stored triangle (`'L'`, `'U'` or `'F'`); a
CSC matrix, `SparseMatrixCSC` included, is read without a copy, as the CSR of
its transpose (a `SparseMatrixCSC` with spare capacity in `rowval`/`nzval`
raises [`InvalidValueError`](@ref), see [`csr_of_transpose`](@ref)). The
backend is the one of the arrays.

The constructor chooses the pipeline, sets the parameters, runs the analysis
and factorizes the values `A` holds; with `deferred_analysis = true` it stops
after the parameters, and the first [`factorize!`](@ref) runs the analysis on
the values present then. The values alias the caller's `nzval`: write the new
values into it and call [`factorize!`](@ref), then [`solve!`](@ref) and
[`inertia`](@ref SparseDirectSolver.inertia).

```julia
K = KKTSolver(A, 'L'; factorization = :ldl)   # analysis and first factorization
nonzeros(A) .= new_values                      # same pattern
factorize!(K)
getparam(K, "info") == 0 || error("factorization failed")
npos, nzero, nneg = SparseDirectSolver.inertia(K)
solve!(K, xb)                                  # xb holds b, then x
```

Keywords:

* `factorization::Symbol = :cholesky`: `:cholesky` factors `A = L Lᵀ`
  (structure `"SPD"`), `:ldl` factors `A = L D Lᵀ` (structure `"S"`) and gives
  the inertia from `D`.
* `pivoting::Symbol = :none`: `:ldl` only. `:none` factors in the pivot order
  of the analysis (static pivoting: a pivot below `pivot_epsilon` is
  replaced); `:bk` uses Bunch–Kaufman pivoting inside the fronts, which the
  stock pipeline provides.
* `matching::AbstractString = "none"`: `:ldl` only, the `"matching_alg"` of the
  analysis (`"none"` or `"default"` off, `"algo1"`–`"algo6"`; `"algo5"` and
  `"algo6"` also scale the matrix and choose the 2×2 pivot pairs of the
  ordering from the cycles of the matching, with either `pivoting`; with
  `"algo1"`–`"algo4"` and `pivoting = :none` the stock analysis looks for no
  pairs, as without matching), on both pipelines. It reads the values present
  at the analysis, so pass
  `deferred_analysis = true` when the matrix holds no real values yet.
* `fused::Union{Nothing, Bool} = nothing`: `true` runs the fused pipeline (a
  [`FusedSolver`](@ref)), `false` the stock one (a [`DirectSolver`](@ref)).
  `nothing` selects the fused pipeline where it runs: `pivoting = :none` on a
  backend with forward progress between workgroups (CUDA, AMDGPU, oneAPI).
  The CPU backend, Metal and `pivoting = :bk` get the stock pipeline.
* `pivot_threshold = nothing`: `:bk` only, the `"pivot_threshold"` of the
  acceptance test.
* `pivot_epsilon = nothing`: the `"pivot_epsilon"` below which a pivot is
  replaced (`nothing`: `1e-13` for `Float64`, `1e-5` for `Float32`, as in
  cuDSS), on both pipelines. With `:ldl` and `pivoting = :none` a pivot `d`
  with `|d| < pivot_epsilon` becomes `±pivot_epsilon`; `0` replaces none, and
  a zero pivot then fails the factorization. On the fused pipeline a NaN or
  infinite pivot always fails the factorization (`info ≠ 0`).
* `pivot_sign = nothing`: `:ldl` only, a vector of `n` entries in `(-1, 0, 1)`,
  the `"pivot_sign"` a replaced pivot takes (an interior-point method passes
  `1` for the primal and `-1` for the dual rows; `0` or `nothing`: the sign of
  the pivot), on both pipelines.
* `reordering::String = "algo4"`: the `"reordering_alg"` of the analysis
  (`"algo4"` nested dissection, `"algo3"` AMD, `"default"` the choice of the
  cost model).
* `user_perm = nothing`: a 1-based permutation of length `n` that replaces the
  reordering.
* `tuning::Union{Nothing, SDSTuning} = nothing`: the performance settings, see
  [`SDSTuning`](@ref SparseDirectSolver.Fused.SDSTuning). The fused pipeline
  takes [`default_tuning`](@ref SparseDirectSolver.Fused.default_tuning) of the
  backend for `nothing`; the stock pipeline keeps the defaults of
  [`DirectSolver`](@ref) for `nothing`.
* `deferred_analysis::Bool = false`: `true` leaves the analysis and the first
  factorization to the first [`factorize!`](@ref), for a matrix whose values
  are not assembled yet when the solver is built (the matching and the 2×2
  pivot pairs are decided from the values of the analysis). Until then
  `getparam(K, "info")` is `0`, and [`solve!`](@ref) and
  [`inertia`](@ref SparseDirectSolver.inertia) raise
  [`FactorizationError`](@ref).

`nothing` keeps the default of the parameter. The inner solver is the field
`inner`; [`getparam`](@ref) is forwarded to it, and
[`pipeline`](@ref SparseDirectSolver.pipeline) tells which one runs.

Raises [`InvalidValueError`](@ref) for an unknown `factorization`, `pivoting`
or `matching`, for `:bk` without `:ldl` or with `fused = true`, for matching
without `:ldl`, for `pivot_threshold` without `:bk` and for `pivot_sign`
without `:ldl`; [`NotSupportedError`](@ref) for a complex element type, a
uniform batch and `fused = true` on a backend that cannot run the fused
pipeline. Every option is checked by the constructor, also with
`deferred_analysis = true`.
"""
mutable struct KKTSolver{T, INT, S <: AbstractDirectSolver{T, INT}, V <: AbstractVector{T}}
    inner::S
    x::V
    b::V
    info::Int
    factorized::Bool          # false until the first factorization (deferred_analysis = true)
    factorization::Symbol
    pivoting::Symbol
end

KKTSolver(A, view::Char = 'L'; kwargs...) = _kkt_solver(structure -> DirectSolver(A, structure, view); kwargs...)

# a SparseMatrixCSC is read in place as the CSR of its transpose (`DirectSolver(::SparseMatrixCSC, …)` converts to
# CSR, a copy the caller's new values would not reach); the solver flips the view of a transposed CSR, and
# csr_of_transpose refuses spare capacity in rowval/nzval
KKTSolver(A::SparseMatrixCSC, view::Char = 'L'; kwargs...) =
    _kkt_solver(structure -> DirectSolver(csr_of_transpose(A), structure, view); kwargs...)

function KKTSolver(rowptr::AbstractVector{<:Integer}, colval::AbstractVector{<:Integer}, nzval::AbstractVector,
                   view::Char = 'L'; kwargs...)
    return _kkt_solver(structure -> DirectSolver(rowptr, colval, nzval, structure, view); kwargs...)
end

function _kkt_solver(make; factorization::Symbol = :cholesky, pivoting::Symbol = :none,
                     matching::AbstractString = "none", fused::Union{Nothing, Bool} = nothing,
                     pivot_threshold = nothing, pivot_epsilon = nothing, pivot_sign = nothing,
                     reordering::AbstractString = "algo4", user_perm = nothing,
                     tuning::Union{Nothing, Fused.SDSTuning} = nothing, deferred_analysis::Bool = false)
    factorization in (:cholesky, :ldl) ||
        throw(InvalidValueError("factorization must be :cholesky or :ldl, got $(repr(factorization))"))
    pivoting in (:none, :bk) || throw(InvalidValueError("pivoting must be :none or :bk, got $(repr(pivoting))"))
    ldl = factorization === :ldl
    bk = pivoting === :bk
    bk && !ldl && throw(InvalidValueError("pivoting = :bk needs factorization = :ldl"))
    bk && fused === true &&
        throw(InvalidValueError("fused = true needs pivoting = :none (the fused pipeline does not pivot)"))
    bk || pivot_threshold === nothing || throw(InvalidValueError("pivot_threshold needs pivoting = :bk"))
    ldl || pivot_sign === nothing || throw(InvalidValueError("pivot_sign needs factorization = :ldl"))

    d = make(ldl ? "S" : "SPD")
    T = eltype(d.A.nzval)
    T <: Real || throw(NotSupportedError("KKTSolver: complex element type $T is not supported; use DirectSolver"))
    d.nbatch == 1 ||
        throw(NotSupportedError("KKTSolver: a uniform batch ($(d.nbatch) members) is not supported; use DirectSolver"))
    # the parsed option decides whether matching is on ("none" and "default" are off); bad strings raise here
    setparam!(d, "matching_alg", matching)
    matched = d.options.matching_alg != MATCHING_NONE
    matched && !ldl &&
        throw(InvalidValueError("matching = $(repr(matching)) needs factorization = :ldl"))
    backend = d.backend
    # the fused pipeline factors without pivoting; it has the matching and the static pivot replacement of the
    # stock no-pivot LDLᵀ (#194), so only Bunch–Kaufman and a backend without forward progress need the stock one
    use_fused = something(fused, !bk && forward_progress(backend))
    inner = if use_fused
        # "S": pivot_type = 'N'; the pivot pairs follow the matching at every analysis (`Fused._fused_pairs!`)
        Fused._fused_solver(d, tuning)
    else
        tuning === nothing || Fused._apply_tuning!(d, Fused._resolved_tuning(tuning, backend))
        if bk
            pivot_threshold === nothing || setparam!(d, "pivot_threshold", pivot_threshold)
        elseif ldl
            setparam!(d, "pivot_type", 'N')
            # no-pivot LDLᵀ consumes no 2×2 pair, but the pairs of the cycles of a job-5 matching ("algo5", "algo6")
            # shape the ordering; jobs 1–4 would fall back to the value-based detection, skipped as without matching
            matched && matching_job(d.options.matching_alg) == 5 || setparam!(d, "pivot_pairs", "none")
        end
        d
    end
    ldl && pivot_sign !== nothing && setparam!(inner, "pivot_sign", pivot_sign)
    pivot_epsilon === nothing || setparam!(inner, "pivot_epsilon", pivot_epsilon)
    if user_perm !== nothing
        setparam!(inner, "user_perm", user_perm)
    else
        setparam!(inner, "reordering_alg", reordering)
    end
    n = size(d, 1)
    x = similar(d.A.nzval, n)
    b = similar(d.A.nzval, n)
    INT = eltype(d.A.rowptr)
    K = KKTSolver{T, INT, typeof(inner), typeof(x)}(inner, x, b, 0, false, factorization, pivoting)
    deferred_analysis || _first_factorization!(K)
    return K
end

# the analysis and the first factorization (a refactorization needs one), on the values the matrix holds now
function _first_factorization!(K::KKTSolver)
    analyze!(K.inner, K.x, K.b; asynchronous = false)
    factorize!(K.inner, K.x, K.b; asynchronous = false)
    K.factorized = true
    K.info = Int(getparam(K.inner, "info"))
    return K
end

_check_factorized(K::KKTSolver, what) =
    K.factorized || throw(FactorizationError(0, "KKTSolver: $what needs a factorization; with " *
                                                "deferred_analysis = true call factorize!(K) first"))

Base.size(K::KKTSolver) = size(K.inner)
Base.size(K::KKTSolver, d::Integer) = size(K.inner, d)

function Base.show(io::IO, K::KKTSolver{T, INT}) where {T, INT}
    print(io, "KKTSolver{", T, ", ", INT, "}(n = ", size(K, 1), ", factorization = :", K.factorization,
          ", pivoting = :", K.pivoting, ", pipeline = :", pipeline(K))
    K.factorized ? print(io, ", info = ", K.info, ")") : print(io, ", not factorized)")
    return nothing
end

"""
    pipeline(K::KKTSolver) -> Symbol

`:fused` when `K` runs the fused pipeline (`K.inner` is a
[`FusedSolver`](@ref)), `:stock` when it runs the stock one (a
[`DirectSolver`](@ref)).
"""
pipeline(K::KKTSolver) = K.inner isa Fused.FusedSolver ? :fused : :stock

"""
    factorize!(K::KKTSolver) -> K

Factorize the values the matrix of `K` holds now, with the analysis of the
constructor. With `deferred_analysis = true` the first call runs the analysis
(ordering, matching, 2×2 pivot pairs) on these values and then the first
factorization; the later calls refactorize with that analysis. The call returns
when the factorization is complete. A failed factorization does not throw:
`getparam(K, "info")` is then the column of the first failed pivot, and `0`
otherwise.
"""
function factorize!(K::KKTSolver)
    K.factorized || return _first_factorization!(K)
    refactorize!(K.inner, K.x, K.b; asynchronous = false)
    K.info = Int(getparam(K.inner, "info"))
    return K
end

"""
    solve!(K::KKTSolver, xb::AbstractVector) -> xb
    solve!(K::KKTSolver, x::AbstractVector, b::AbstractVector) -> x

Solve `A x = b` with the last factorization; with one vector, `xb` holds `b` on
entry and `x` on return. The vectors are on the backend of `K`. The call
returns when the solution is complete. Before the first factorization
(`deferred_analysis = true`) it raises [`FactorizationError`](@ref).
"""
function solve!(K::KKTSolver, x::AbstractVector, b::AbstractVector)
    _check_factorized(K, "solve!")
    copyto!(K.b, b)
    solve!(K.inner, K.x, K.b)
    copyto!(x, K.x)
    KernelAbstractions.synchronize(KernelAbstractions.get_backend(K.x))
    return x
end

solve!(K::KKTSolver, xb::AbstractVector) = solve!(K, xb, xb)

"""
    inertia(K::KKTSolver) -> (npos, nzero, nneg)

The numbers of positive, zero and negative eigenvalues of the matrix `K`
factorized last, which sum to `n`: for `factorization = :ldl` from the signs of
`D` (2×2 blocks included), for `:cholesky` the number of positive pivots and
`0` negative ones. It is the inertia of the factorized matrix, so a replaced
pivot counts with the sign it was given. Valid when `getparam(K, "info")` is
`0`; before the first factorization (`deferred_analysis = true`) it raises
[`FactorizationError`](@ref).
"""
function inertia(K::KKTSolver)
    _check_factorized(K, "inertia")
    n = size(K, 1)
    npos, nneg = getparam(K.inner, "inertia")
    return (Int(npos), n - Int(npos) - Int(nneg), Int(nneg))
end

# "info" is the one of the last factorization of `K`, read once (a scalar read of device memory)
getparam(K::KKTSolver, name::AbstractString) = name == "info" ? K.info : getparam(K.inner, name)
