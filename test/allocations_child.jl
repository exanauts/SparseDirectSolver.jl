# The CPU allocation checks of the test suite, run by `test_allocations.jl` in a child process (#188):
#
#     julia --project=<test env> -t <threads> allocations_child.jl <eltype> <shared> <output file>
#
# `@allocated` counts the allocations of the whole process, and KernelAbstractions >= 0.10.6 `yield`s after
# every CPU launch of 50 µs or more. In a ParallelTestRunner worker the runner's communication tasks then run
# inside the measured call and are charged to it; this process has no other task, so the yield is a no-op and
# `@allocated` sees the solver only. Not a test file (no `test_` prefix): it measures and writes one line
# `name<TAB>bytes<TAB>budget` per check to <output file>; `test_allocations.jl` asserts `bytes <= budget`.
# Each check reproduces the setup of the test it was taken from (named in the check) and calls the measured
# function once to compile before measuring (`measured`). The budgets are computed here,
# with this process's `--code-coverage` setting (`ka_cpu_alloc_budget`).

using Test
using Random
using LinearAlgebra
using SparseArrays
using Aqua
using KernelAbstractions
using SparseDirectSolver
using SparseDirectSolver: update!
using Krylov
const SDS = SparseDirectSolver

length(ARGS) == 3 || error("usage: allocations_child.jl <eltype> <shared> <output file>")
const CHILD_ELTYPE = only(T for T in (Float32, Float64, ComplexF32, ComplexF64) if string(T) == ARGS[1])
const CHILD_SHARED = parse(Bool, ARGS[2])
const CHILD_OUTPUT = ARGS[3]

# as `test_expr` in runtests.jl: the part of a split test file for one element type
const SDS_TEST_PART = Ref{Any}(((CHILD_ELTYPE,), CHILD_SHARED))
include(joinpath(@__DIR__, "backends.jl"))
include(joinpath(@__DIR__, "utils.jl"))
include(joinpath(@__DIR__, "matrices.jl"))
Random.seed!(666)

TEST_CPU || error("allocations_child.jl measures on the CPU backend: SDS_TEST_CPU must not be 0")

const RESULTS = Tuple{String, Int, Int}[]
record!(name, bytes, budget) = (push!(RESULTS, (name, Int(bytes), Int(budget))); nothing)

# `f()` returns the bytes of one measured call: the first call compiles, the second is measured. The warm-up is
# the measuring call itself, not only the solver call it wraps: in this script's functions a first measuring
# call still compiled after a direct warm-up call of the solver (4–9 MB and ~100 ms with KA 0.10.7, Julia
# 1.13), while the test files made both calls from top-level testset scope.
measured(f) = (f(); f())

# test_helpers.jl, "KA CPU backend launches allocation-free" (#145): the CPU allocation budgets grant nothing
# per launch, so a KA release that allocates per launch again would fail the numeric and solve budgets all
# over the suite; this guard names the cause
function check_ka_launch()
    y = zeros(Int, 256)
    launch!() = (ka_launch_probe_kernel!(CPU(), 64)(y; ndrange = length(y)); KernelAbstractions.synchronize(CPU()))
    launch!(); launch!(); launch!()                       # compile and warm up
    bytes = @allocated launch!()
    y == 1:256 || error("ka_launch_probe_kernel! computed a wrong result")
    record!("test_helpers: KA CPU backend launches allocation-free", bytes,
            Base.JLOptions().code_coverage != 0 ? 64 : 0)
end

# test_numeric_cholesky_{a,b,c}.jl, "views, index bases, refactorization (CPU, T)": one `factorize!` of
# `random_spd(300, 0.02)` with each option set of the testset
function check_refactorization(::Type{T}, file, allocated, opts_list) where {T}
    for (oname, opts) in opts_list
        Random.seed!(666)
        A = random_spd(T, 300, 0.02)
        S, _, _, Sd, Nd, nz = numeric_setup(CPU(), A; opts)
        SDS.factorize!(Nd, Sd, nz) == 0 || error("$file: the factorization failed")
        record!("$file: views, index bases, refactorization, $oname (CPU, $T)",
                measured(() -> allocated(Nd, Sd, nz)), numeric_alloc_budget(S, T))
    end
end

# test_numeric_cholesky_c.jl, "dense implementations (CPU, T)": the allocation-free dense paths (T32, #53),
# `:ka` like `:auto`; `:generic` is the allocating reference (PLAN §3.9)
function check_dense_implementations(::Type{T}) where {T}
    Random.seed!(666)
    A = laplacian2d(T, 25, 25)
    S, _, _, Sd, Nd, nz = numeric_setup(CPU(), A; opts = NUMERIC_C_OPTS)
    for impl in SDS.dense_impls(:potrf, CPU(), T)
        impl === :generic && continue
        SDS.factorize!(Nd, Sd, nz; impl) == 0 || error("test_numeric_cholesky_c: the factorization failed")
        record!("test_numeric_cholesky_c: dense implementations, impl = :$impl (CPU, $T)",
                measured(() -> numeric_c_allocated(Nd, Sd, nz; impl)), numeric_alloc_budget(S, T; impl))
    end
end

# the calls "determinism and refactorization" in test_numeric_ldlt.jl and test_numeric_lu.jl make before their
# allocation checks: factorizations with the values, doubled values and the values again, then two
# deterministic solves (with A and Aᵀ for LU). They are repeated here: without them the second `sweep_solve!`
# after the factorization still compiles (8 MB and ~80 ms with KA 0.10.7, Julia 1.13), which a single warm-up
# call would leave inside the measurement.
function ldlt_lu_preamble(Sd, Nd, nz, opts)
    SDS.factorize!(Nd, Sd, nz; opts)
    SDS.factorize!(Nd, Sd, to_device(CPU(), 2 .* to_host(nz)); opts)
    SDS.factorize!(Nd, Sd, nz; opts)
    return nothing
end
function ldlt_lu_solves(ws, Sd, Nd, b, transposes)
    for transpose in transposes, _ in 1:2
        transpose === nothing ? device_solve(CPU(), ws, Sd, Nd, b; deterministic = true) :
        device_solve(CPU(), ws, Sd, Nd, b; deterministic = true, transpose)
    end
    return nothing
end

# test_numeric_ldlt.jl, "determinism and refactorization (CPU, T)": the numeric phase and the solve of the
# 2×2-pivot KKT matrix, `impl = :ka` the allocation-free KA dense fallbacks on the regime-C path (T32, #53)
ldlt_factorize_allocated(Nd, Sd, nz, opts, impl) = @allocated SDS.factorize!(Nd, Sd, nz; opts, impl)
function check_ldlt(::Type{T}) where {T}
    Random.seed!(666)
    A = kkt_matrix(T, 200, 100, 0.0; hessian = :indefinite, hessian_scale = 1.0e-3)
    opts = Options(user_perm = kkt_interleaved_perm(200, 100), subtree_budgets = [8192, 16384], regime_c_width = 16,
                   regime_c_rows = 128)
    S, _, Sd, Nd, nz = ldlt_setup(CPU(), A; opts)
    ldlt_lu_preamble(Sd, Nd, nz, opts)
    ws = SDS.allocate_solve(Sd, T, CPU(), 2)
    b = rand(T, 300, 2)
    ldlt_lu_solves(ws, Sd, Nd, b, (nothing,))
    x = similar(b)
    for impl in (:auto, :ka)
        SDS.factorize!(Nd, Sd, nz; opts, impl) == 0 || error("test_numeric_ldlt: the factorization failed")
        record!("test_numeric_ldlt: determinism and refactorization, factorize!, impl = :$impl (CPU, $T)",
                measured(() -> ldlt_factorize_allocated(Nd, Sd, nz, opts, impl)), ldlt_alloc_budget(S, Nd; impl))
        record!("test_numeric_ldlt: determinism and refactorization, sweep_solve!, impl = :$impl (CPU, $T)",
                measured(() -> solve_allocated(x, ws, Sd, Nd, b; impl)), solve_alloc_budget(S, ws; ldlt = true, impl))
    end
end

# test_numeric_lu.jl, "determinism and refactorization (CPU, T)": the numeric phase and the solves with A and
# Aᵀ of the row-interchanging weak-diagonal matrix
lu_factorize_allocated(Nd, Sd, nz, opts) = @allocated SDS.factorize!(Nd, Sd, nz; opts)
function check_lu(::Type{T}) where {T}
    Random.seed!(666)
    A = weak_diagonal_general(T, 300, 0.02)
    opts = Options(pivot_threshold = 1.0, subtree_budgets = [8192, 16384], regime_c_width = 16, regime_c_rows = 128)
    S, _, Sd, Nd, nz = lu_setup(CPU(), A; opts)
    ldlt_lu_preamble(Sd, Nd, nz, opts)
    ws = SDS.allocate_solve(Sd, T, CPU(), 2)
    b = rand(T, 300, 2)
    ldlt_lu_solves(ws, Sd, Nd, b, (false, true))
    SDS.factorize!(Nd, Sd, nz; opts) == 0 || error("test_numeric_lu: the factorization failed")
    record!("test_numeric_lu: determinism and refactorization, factorize! (CPU, $T)",
            measured(() -> lu_factorize_allocated(Nd, Sd, nz, opts)), ldlt_alloc_budget(S, Nd; lu = true))
    x = similar(b)
    for transpose in (false, true)
        record!("test_numeric_lu: determinism and refactorization, sweep_solve!, transpose = $transpose (CPU, $T)",
                measured(() -> solve_allocated(x, ws, Sd, Nd, b; transpose)), solve_alloc_budget(S, ws; ldlt = true))
    end
end

# test_solve.jl, the former "no allocations (CPU, T)": both forward-sweep variants, `impl = :ka` the
# allocation-free KA dense fallbacks on the regime-C path (T32, #53)
function check_solve(::Type{T}) where {T}
    Random.seed!(666)
    A = laplacian2d(T, 40, 40)
    S, Sd, Nd, ws = solve_setup(CPU(), A; opts = SOLVE_OPTS)
    b = rand(T, size(A, 1), 5)
    x = similar(b)
    for deterministic in (false, true), impl in (:auto, :ka)
        bytes = measured(() -> solve_allocated(x, ws, Sd, Nd, b; deterministic, impl))
        relres(A, x, b) <= tol(T) || error("test_solve: residual above tolerance")
        record!("test_solve: no allocations, deterministic = $deterministic, impl = :$impl (CPU, $T)", bytes,
                solve_alloc_budget(Sd, ws; impl))
    end
end

# test_api.jl, "MadNLP-like refactorization loop (CPU, T)": 20 refactorizations and solves through the handle
# layer with a changing diagonal, the values written into the solver's buffer as MadNLP does; the largest
# allocation of iterations 2–20 against the budget of one factorization and one solve. The loop runs twice
# and the second pass is measured (`measured`): in test_api.jl the earlier testsets of the worker had compiled
# the "refactorization" phase, here iteration 2 of the first pass compiles it.
function api_loop_allocated(backend, solver, H, b, bd, xd, σs, vals)
    allocs = Int[]
    for (k, σ) in enumerate(σs)
        copyto!(solver.A.nzval, vals[k])
        phase = k == 1 ? "factorization" : "refactorization"
        a = device_allocated(backend, () -> (execute!(phase, solver, xd, bd); execute!("solve", solver, xd, bd)))
        KernelAbstractions.synchronize(backend)
        getparam(solver, "info") == 0 || error("test_api: the factorization failed in iteration $k")
        relres(H + σ * I, to_host(xd), b) <= tol(eltype(b)) ||
            error("test_api: residual above tolerance in iteration $k")
        k > 1 && push!(allocs, a)
    end
    return maximum(allocs)
end
function check_api_loop(::Type{T}) where {T}
    Random.seed!(666)
    backend = CPU()
    nh, nj = 300, 100
    K = kkt_matrix(T, nh, nj, 1.0e-8)
    H = K[1:nh, 1:nh]
    INT = Int32
    solver = DirectSolver(api_matrix(backend, tril(H), INT), spd_structure(T), 'L')
    b = rand(T, nh)
    bd = to_device(backend, b)
    xd = similar(bd)
    execute!("analysis", solver, xd, bd)
    σs = [real(T)((-0.5)^k * k) for k in 1:20]
    vals = [CSR(SparseMatrixCSC{T, INT}(tril(H + σ * I))).nzval for σ in σs]
    record!("test_api: MadNLP-like refactorization loop (CPU, $T)",
            measured(() -> api_loop_allocated(backend, solver, H, b, bd, xd, σs, vals)),
            numeric_alloc_budget(solver.symbolic, T) + solve_alloc_budget(solver.symbolic, solver.workspace))
end

# test_dense.jl, the former ":ka dense fallbacks allocation-free (CPU, T)" (T32, #53): 8 rounds of one call of
# every `:ka` dense fallback (`ka_dense_round!`, 15 launches)
function check_ka_dense(::Type{T}) where {T}
    Random.seed!(666)
    n, f = 40, 70
    A, B, C = randn(T, n, n), randn(T, n, n), zeros(T, n, n)
    S = dense_hpd(T, n)
    buf = randn(T, f * n + 3)
    P = reshape(view(buf, 4:(f * n + 3)), f, n)                       # a panel view, as in a front
    F = view(P, 1:n, 1:n)
    Pr = view(P, (n + 1):f, :)
    info, ipiv = zeros(Int32, 4), zeros(Int32, n)
    A3, C3 = randn(T, n, n, 2), zeros(T, n, n, 2)
    F3 = cat(LowerTriangular(S) + n * I, LowerTriangular(S) + n * I; dims = 3)
    X3 = cat(S, S; dims = 3)
    flags = [('N', 'N'), ('T', 'C'), ('C', 'T')]
    args = (C, A, B, S, F, Pr, info, ipiv, C3, A3, F3, X3, flags)
    rounds = 8
    bytes = measured(() -> ka_dense_rounds_allocated(args, rounds))
    launches = 15 * rounds
    record!("test_dense: :ka dense fallbacks allocation-free (CPU, $T)", bytes,
            ka_cpu_alloc_budget(launches, launches * ka_dense_localmem(T)))
end

RUN_SHARED && check_ka_launch()
for T in ELTYPES
    check_refactorization(T, "test_numeric_cholesky_a", numeric_a_allocated,
                          (("NUMERIC_A_OPTS", NUMERIC_A_OPTS), ("NUMERIC_ABC_OPTS", numeric_abc_opts(CPU()))))
    check_refactorization(T, "test_numeric_cholesky_b", numeric_b_allocated,
                          (("NUMERIC_B_OPTS", NUMERIC_B_OPTS), ("NUMERIC_BC_OPTS", NUMERIC_BC_OPTS)))
    check_refactorization(T, "test_numeric_cholesky_c", numeric_c_allocated, (("NUMERIC_C_OPTS", NUMERIC_C_OPTS),))
    check_dense_implementations(T)
    check_ldlt(T)
    check_lu(T)
    check_solve(T)
    check_api_loop(T)
    check_ka_dense(T)
end

open(CHILD_OUTPUT, "w") do io
    for (name, bytes, budget) in RESULTS
        println(io, name, '\t', bytes, '\t', budget)
    end
end
