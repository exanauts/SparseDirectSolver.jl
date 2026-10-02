# TASKS.md — consecutive implementation tasks for SparseDirectSolver.jl

Each task is sized for one working session of a capable coding model and ends
with an explicit list of tests that must pass. Tasks are ordered; do not start
the next one. After every task the owner re-evaluates `PLAN.md` and this file,
so the **Report** block at the end of each task must be filled in honestly.

## Session protocol

1. Read `PLAN.md` (the sections the task names), the task text, and the Report
   blocks of the previous tasks. `RESEARCH.md` is background.
2. Work only inside the task's scope. If something in `PLAN.md` turns out to be
   wrong or impractical, do the task the best way you can and write the
   deviation into the Report; do not silently redesign.
3. Test environment on this machine: Julia 1.13 (package compat `julia = "1.10"`),
   Linux (WSL2), one NVIDIA RTX 4080 (16 GB). **Locally, tests use only the KA
   CPU backend and CUDA.** The exanauts CI runners also run the suite on an AMD
   GPU (`amdgpu` label); oneAPI and Metal are never tested. Their extension
   files (T23) are written by analogy and only checked for precompilation.
4. CUDA.jl 6.x is split into packages. Use `CUDACore` (arrays, `CUDABackend`),
   `cuSPARSE` (`CuSparseMatrixCSR`, `CuSparseMatrixCSC`), `cuBLAS` and
   `cuSOLVER` (dense routines) as weak dependencies; the umbrella `CUDA` package
   re-exports them and is what the tests load.
5. Run the full suite with
   `julia --project=. -e 'using Pkg; Pkg.test()'`.
   `SDS_TEST_GPU=0` skips CUDA; `SDS_TEST_ONLY="name1,name2"` restricts to the
   named test files (both implemented in T01).
6. Every public function gets a docstring. Tests are deterministic
   (`Random.seed!(666)`). Tolerances and matrix generators come from
   `test/utils.jl` and `test/matrices.jl` (T01); do not invent new ones per file.
7. Done = every listed test passes on CPU and on CUDA, the file-level status
   marker is set, and the Report block is filled in. Status markers:
   `[ ]` not started, `[~]` in progress, `[x]` done, `[!]` done with deviations.

## Report block template (copy under each task when done)

```text
### Report
- Status: [x] / [!]
- What was built: (files, public functions)
- Tests: (command used, counts: pass/fail/broken on CPU, on CUDA)
- Measurements: (timings, sizes, anything numeric that was asked for)
- Deviations from PLAN.md / this task: (what and why)
- Open issues / follow-ups: (bugs found, limits hit, suggestions for the next task)
- Suggested plan changes: (one line each, or "none")
```

## Shared test conventions (created in T01, used everywhere)

* `test/backends.jl`: `BACKENDS = Any[CPU()]`, plus `CUDABackend()` when
  `CUDA.functional()` and `SDS_TEST_GPU != "0"`. `to_device(backend, x)` maps
  `Array`/`SparseMatrixCSC` to the backend (`CuArray`, `CuSparseMatrixCSR`),
  `to_host(x)` maps back. `backend_name(backend)` for testset names.
* `test/matrices.jl` (all return `SparseMatrixCSC{T,Int}`):
  `laplacian2d(T, nx, ny)`, `laplacian3d(T, nx, ny, nz)` (SPD),
  `random_spd(T, n, density)` (`A*A' + n*I`), `random_hpd` (complex),
  `random_symindef(T, n, density)` (symmetric, indefinite, nonsingular),
  `kkt_matrix(T, nh, nj, δ)` (`[H Jᵀ; J -δI]`, H SPD),
  `random_general(T, n, density)` (diagonally dominant),
  `singular_block_matrix(T, n, j)` (symmetric with a structurally zero pivot at
  column `j` after natural ordering, for perturbation tests),
  the two 5×5 Schur examples and the 3×3 uniform-batch example from
  `../CUDSS.jl/docs/src/`.
* `test/utils.jl`: `tol(T) = sqrt(eps(real(T)))`,
  `relres(A, x, b) = norm(b - A*x) / max(norm(b), one(real(eltype(b))))`,
  `ELTYPES = (Float32, Float64, ComplexF32, ComplexF64)`,
  `REAL_ELTYPES`, `INTTYPES = (Int32, Int64)`.
* Every numeric test loops `for backend in BACKENDS, T in ELTYPES` unless the
  task says otherwise, and asserts `relres(A, x, b) ≤ tol(T)` on
  well-conditioned generators.

---

## T01 — Package scaffolding, option tables, test infrastructure   `[!]`

**Reads**: PLAN §1.3, §1.4, §1.7, §3.1, §3.2, §4.

#### Deliverables

* The git repository, `.gitignore` and the CI workflows under `.github/workflows/`
  already exist (remote `exanauts/SparseDirectSolver.jl`); keep them working.
* `Project.toml`: name `SparseDirectSolver`, fresh UUID, version `0.1.0`,
  deps `Adapt, Atomix, CliqueTrees, GPUArrays, GPUArraysCore, KernelAbstractions,
  LinearAlgebra, SparseArrays`; weakdeps `CUDACore, cuSPARSE` with extension
  `SparseDirectSolverCUDAExt = ["CUDACore", "cuSPARSE"]` (the file is created
  in T02; declare it now or in T02, but the package must load in both tasks);
  compat for everything incl. `julia = "1.10"`, `KernelAbstractions = "0.9"`.
* `LICENSE` (MIT, same form as `../CUDSS.jl/LICENSE`, owner's name), `README.md`
  stub pointing to `PLAN.md`.
* `src/SparseDirectSolver.jl` module with includes; `src/errors.jl`
  (`SparseDirectSolverError`, `NotSupportedError`, `InvalidValueError`,
  `FactorizationError(info)`, `InterruptedError`); `src/types.jl` (enums
  `Structure`, `MatrixView`, `IndexBase`, `Phase` with the cuDSS bit values
  1,2,3,4,8,16,32,64,128,256,512,1008, `PivotType`, `ReorderingAlg`,
  `MatchingAlg`, `ScheduleKind`; string↔enum conversions with exactly the
  CUDSS.jl spellings from `../CUDSS.jl/src/types.jl`, plus the new pivot chars
  `'D' 'L' 'B'`); `src/options.jl` (`Options` struct with every config
  parameter of PLAN §1.3 and §1.7, typed, with defaults; `CONFIG_PARAMETERS`,
  `DATA_PARAMETERS` tuples copied verbatim from `../CUDSS.jl/src/types.jl`;
  `EXTRA_PARAMETERS` for §1.7; `setparam!(opts, name::String, value)`,
  `getparam(opts, name)`; `default_pivot_epsilon(::Type{Float32}) = 1e-5`,
  `(::Type{Float64}) = 1e-13`).
* `test/Project.toml` (Test, Random, LinearAlgebra, SparseArrays, Aqua,
  KernelAbstractions, Adapt; `CUDA`/`AMDGPU` are added by CI or the developer
  and loaded with `Base.find_package` guards in `test/backends.jl`), `test/runtests.jl` (honors `SDS_TEST_GPU`,
  `SDS_TEST_ONLY`), `test/backends.jl`, `test/matrices.jl`, `test/utils.jl`,
  `test/test_options.jl`.
* CI already in place (modelled on `../ExaPF.jl`): `ci.yml` runs the CPU-only
  suite on GitHub runners and the full suite on the exanauts self-hosted
  runners labelled `cuda` and `amdgpu` (those add `CUDA`/`AMDGPU` to the test
  environment themselves, so `test/Project.toml` must not list them as hard
  deps that fail to load without a GPU). Make the suite skip a backend
  cleanly when its package is absent or not functional.

#### Tests that must pass (`test/test_options.jl`, `test/test_aqua.jl`)

* `using SparseDirectSolver` loads; `Aqua.test_all(SparseDirectSolver; ambiguities=false)`.
* `CONFIG_PARAMETERS` and `DATA_PARAMETERS` equal the tuples in
  `../CUDSS.jl/src/types.jl` (embed the literal tuples in the test).
* For every name in `CONFIG_PARAMETERS ∪ EXTRA_PARAMETERS` that PLAN §1.3/§1.7
  marks port/reinterpret: `setparam!` with a valid value then `getparam`
  returns it; a value of the wrong type throws `InvalidValueError`.
  `"device_count"`, `"device_indices"` throw `NotSupportedError`. An unknown
  name throws `ArgumentError` whose message contains the name.
* Defaults: `ir_n_steps == 0`, `pivot_type == PIVOT_AUTO`, `use_superpanels == 1`,
  `deterministic_mode == 0`, `schedule == :auto`, `factor_precision === nothing`.
* Every string of every enum converts and converts back; `"XYZ"` throws.
* `BACKENDS` contains `CPU()` and contains `CUDABackend()` iff `CUDA.functional()`
  (print which are active).
* Generators in `test/matrices.jl`: `issymmetric`, `isposdef` (small sizes),
  `eigvals` sign counts for `random_symindef` (both signs present), each
  generator returns the requested `T` and `Int` indices.

### Report

- Status: [!] (done; deviations listed below are small and mostly additive)
- What was built:
  - `Project.toml` (UUID `dc8e0fd9-41f1-440b-9624-22817132f700`, v0.1.0, deps/weakdeps/compat
    exactly as specified; resolves to KA 0.9.43, GPUArrays 11.5, CliqueTrees 1.19, CUDA 6.4.1
    in the test env), `LICENSE` (MIT, Michel Schanen), `README.md` stub.
  - `src/SparseDirectSolver.jl`; `src/errors.jl` (`SparseDirectSolverError`, `NotSupportedError`,
    `InvalidValueError`, `FactorizationError(info[, msg])` (parametric so `info` can be a batch
    vector), `InterruptedError`).
  - `src/types.jl`: enums `Structure`, `MatrixView`, `IndexBase`, `Phase`, `PivotType`,
    `ReorderingAlg`, `MatchingAlg`, `ScheduleKind`, plus `FactorizationAlg`, `SolveAlg`,
    `PivotEpsilonAlg`, `IRMode`, all with the cuDSS 0.8 integer values. `enum_spellings(E)` is the
    single table behind `convert(E, "SPD")`/`convert(E, 'L')` and `convert(String, x)`/`convert(Char, x)`;
    `phase_includes(phase, part)` tests phase bits.
  - `src/options.jl`: `CONFIG_PARAMETERS`, `DATA_PARAMETERS` (verbatim), `CUDSS08_DATA_PARAMETERS`,
    `EXTRA_PARAMETERS`; `Options` (typed fields for every §1.3/§1.7 config parameter and the
    user-input data parameters `user_perm`, `user_schur_indices`, `user_nd_partition_tree`,
    `user_host_interrupt`, `ubatch_mask`, `pivot_sign`; keyword constructor validates through
    `setparam!`; `copy`; `show` prints non-default values); `setparam!`/`getparam` driven by one
    `PARAMETER_SPECS` table (port / reinterpret / deferred / not_planned / output / solver);
    `default_pivot_epsilon`, `resolved_pivot_epsilon(opts, T)`.
  - `ext/SparseDirectSolverCUDAExt.jl`: empty stub (declared now, filled in T02), loads with CUDA.
  - `test/Project.toml`, `test/runtests.jl` (auto-discovers `test_*.jl`; `SDS_TEST_GPU`,
    `SDS_TEST_ONLY`, clear error for unknown names; seeds 666 per file), `test/backends.jl`
    (`BACKENDS`, `to_device(backend, x)`, `to_device(backend, A, INT)`, `to_host`, `backend_name`;
    CUDA and AMDGPU guarded by `Base.find_package` + `functional()`), `test/utils.jl` (`ELTYPES`,
    `REAL_ELTYPES`, `COMPLEX_ELTYPES`, `INTTYPES`, `tol`, `relres`, `spd_structure`, `sym_structure`,
    `eigen_inertia`, `thrown`), `test/matrices.jl` (all generators of the conventions, each with a
    `Float64` method without `T` and an `rng` keyword; `kkt_matrix(...; hessian = :spd | :indefinite)`;
    `random_symindef` is diagonally dominant with known inertia; `singular_block_matrix(...;
    stored_zero)`; `schur_example_lu/_ldlt/_cholesky`, `ubatch_example`), `test/test_options.jl`,
    `test/test_aqua.jl`, `test/test_helpers.jl`.
- Tests: `julia --project=. -e 'using Pkg; Pkg.test()'` with CUDA added to the test env:
  932 pass / 0 fail / 0 broken (CPU + CUDA, RTX 4080; test_aqua 8.8 s, test_helpers 27 s,
  test_options 5 s). `SDS_TEST_GPU=0`: 922 pass. Julia 1.10.12 (lts), CPU only, CUDA absent from
  the test env: 922 pass.
- Measurements: none asked.
- Deviations from PLAN.md / this task:
  - Defaults test checks `opts.schedule == SCHEDULE_AUTO`, not `== :auto`: the task asks both for a
    `ScheduleKind` enum and for a Symbol default; I kept the enum (same representation as
    `pivot_type == PIVOT_AUTO`). `getparam(opts, "schedule") == "auto"`.
  - Generator/backend checks live in a third file, `test/test_helpers.jl`, not in `test_options.jl`.
  - Extra enums `FactorizationAlg`, `SolveAlg`, `PivotEpsilonAlg` (mirroring cuDSS 0.8's
    per-parameter enums) and `IRMode`; `Phase` also has `PHASE_SOLVE_FWD_SCHUR = 48` and
    `PHASE_SOLVE_BWD_SCHUR = 384` for the CUDSS.jl shorthands.
  - `CUDSS08_DATA_PARAMETERS = ("ir_n_steps", "ubatch_mask", "flops")`: PLAN §1.4 ports these but
    CUDSS.jl's tuple does not list them.
  - `getparam(opts, …)` returns the spelling `setparam!` accepts (`"algo3"`, `'B'`), not the cuDSS
    integer that `cudss_get` returns; enum conversion errors are `InvalidValueError` (CUDSS.jl:
    `ArgumentError`). Unknown *names* are `ArgumentError` as required.
  - `setparam!(opts, "device_count", 1)` is accepted as a no-op (single device is what we do);
    any other value, `"device_indices"`, `"comm_device"`, `"comm_host"` and `pivot_type` `'C'`/`'R'`
    raise `NotSupportedError`.
  - Deferred hybrid parameters (`hybrid_memory_mode`, `hybrid_device_memory_limit`,
    `hybrid_execute_mode`) are stored and warn once per name on a non-default value;
    `reordering_alg = "algo1"/"algo2"` warns once (PLAN §1.3).
  - Defaults chosen where PLAN is silent (please confirm): `pivot_threshold = 0.01` (from T14;
    no upper bound, CUDSS.jl's own test sets 2.0), `ir_tol = 0` (no early exit),
    `max_lu_nnz = -1` (negative = no limit), `nd_nlevels = 10`, `nd_ubfactor = -1` (library
    default), `host_nthreads = 0` (= `Threads.nthreads()`), `ubatch_size = 0` (deduced from
    `nzVal`, as PLAN §3.1 auto-detects batches), `use_cuda_register_memory = 1`.
  - The CUDSS.jl docs have three 5×5 Schur examples (LU, LDLᵀ, LLᵀ), not two; all three are in
    `test/matrices.jl`. Symmetric examples return the full matrix (the docs pass `tril`/`triu`).
- Open issues / follow-ups:
  - Local GPU runs need CUDA in the test env: `julia --project=test -e 'using Pkg; Pkg.add("CUDA")'`,
    then do not commit `test/Project.toml` (Pkg.test's sandbox hides the global environment).
    I restored the clean file before committing.
  - The AMDGPU branch of `test/backends.jl` is written by analogy with ExaPF and untested here.
    Once the `amdgpu` runner sees `ROCBackend()` in `BACKENDS`, numeric tests from T02 on will run
    on ROCm before the AMDGPU extension exists (T23).
  - T13: the ported `cudss_solver` loop sets `"algo1"`–`"algo5"` for every algorithm parameter,
    `pivot_type` `'C'`/`'R'`; our validation rejects algorithms beyond each cuDSS 0.8 enum and
    global pivoting, so that loop must be restricted (T13 already says "implemented parameters").
  - T16: `"ir_n_steps"` is both a config name (requested steps) and a cuDSS 0.8 data name (steps
    performed); `getparam(opts, "ir_n_steps")` is the config value, the solver-level getter must
    decide how to expose both.
  - All test files are included into `Main`; shared helpers belong in `utils.jl`/`matrices.jl`/
    `backends.jl` to avoid top-level name clashes between test files.
- Suggested plan changes:
  - TASKS T01: write the default as `schedule == SCHEDULE_AUTO` (or drop `ScheduleKind`).
  - PLAN §1.4: note the `ir_n_steps` config/data name collision and how the flat getter resolves it.
  - AGENTS.md "Commands": add the local `Pkg.add("CUDA")` step for the test env.

---

## T02 — CSR container, backend adapters, matrix descriptors   `[!]`

**Reads**: PLAN §1.1, §2.3 step 1 (only the input side), §3.1.

#### Deliverables

* `src/matrix.jl`: `CSR{T,INT,VI,VT}` (fields `rowptr, colval, nzval, nrows,
  ncols, index::IndexBase, transposed::Bool`); constructors from raw arrays
  (`index` keyword), from `SparseMatrixCSC` (host conversion, then `adapt` to a
  backend via `to_backend(A, backend)`), `csr_of_transpose(A::SparseMatrixCSC)`
  (zero-copy: `colptr` as `rowptr`, `transposed = true`); `SparseMatrixCSC(::CSR)`;
  `size`, `nnz`, `nbatch(A) = length(nzval) ÷ length(colval)` for vector
  `nzval`, `size(nzval, 2)` for matrix `nzval`; `KernelAbstractions.get_backend(::CSR)`.
* `MatrixDescriptor{T,A}`: wraps a dense vector/matrix/3-D array with
  `nrows, ncols, nbatch, transposed`; `MatrixDescriptor(T, n; nbatch)`,
  `MatrixDescriptor(T, m, n; nbatch, transposed)`, `MatrixDescriptor(x::AbstractArray)`,
  `update!(desc, x)` (re-points, no copy, checks shape).
* `ext/SparseDirectSolverCUDAExt.jl`: `CSR(A::CuSparseMatrixCSR)` sharing
  arrays, `CSR(A::CuSparseMatrixCSC)` sharing arrays with `transposed = true`,
  `CuSparseMatrixCSR(A::CSR)`, `to_backend(A::SparseMatrixCSC, ::CUDABackend)`.

#### Tests that must pass (`test/test_matrix.jl`, CPU + CUDA, `T ∈ ELTYPES`, `INT ∈ INTTYPES`)

* `SparseMatrixCSC(CSR(A)) == A` for random sparse `A`, 1-based and 0-based.
* `csr_of_transpose(A)` shares memory with `A` (`pointer(...)` equality on
  host) and `SparseMatrixCSC(csr_of_transpose(A)) == transpose(A)` after
  honoring the flag.
* CUDA: `CSR(CuSparseMatrixCSR(A))` shares device memory (compare `pointer`);
  CSC path sets `transposed`.
* `nbatch` is 1 for a plain matrix, 3 for `nzval` of length `3nnz`, and 3 for
  an `nnz×3` matrix.
* `MatrixDescriptor` shape checks: `update!` with a wrong length throws
  `InvalidValueError`; 3-D array `(n, p, nb)` sets `nbatch = nb`.

### Report

- Status: [!] (done; small deviations below)
- What was built:
  - `src/matrix.jl`: `CSR{T,INT,VI,VT}` (fields as specified; inner constructor validates lengths
    and that `nrows`, `ncols`, `nnz` fit in `INT`, never reads device memory); `CSR(rowptr, colval, nzval[, nrows, ncols]; index, transposed)`
    (zero-copy, `index` as `'O'`/`'Z'` or `IndexBase`), `CSR(A::SparseMatrixCSC; index)` (host
    conversion), `csr_of_transpose(A)` (zero-copy, `transposed = true`), `to_backend(A, backend)` for
    `SparseMatrixCSC` and `CSR` (generic through `KernelAbstractions.allocate` + `copyto!`, no copy if
    already on `backend`), `SparseMatrixCSC(A::CSR[, k])` (host copy of batch member `k`, rebased to 1),
    `size`, `eltype`, `nnz`, `nbatch`, `get_backend`, `Adapt.adapt_structure`, `show`.
  - `MatrixDescriptor{T,A}` (mutable; `data::Union{Nothing,A}`, `nrows, ncols, nbatch, transposed`;
    explicit inner constructor so Aqua `unbound_args` passes on Julia 1.10),
    `MatrixDescriptor(T, n; nbatch)`, `MatrixDescriptor(T, m, n; nbatch, transposed)`,
    `MatrixDescriptor(x; transposed)`, `update!(desc, x)` (re-points, no copy; checks eltype, array
    type, length and shape, `InvalidValueError` otherwise), `size`, `nbatch`, `get_backend`.
  - `ext/SparseDirectSolverCUDAExt.jl`: `CSR(::CuSparseMatrixCSR)` and `CSR(::CuSparseMatrixCSC)`
    (shared arrays, CSC → `transposed = true`), `CuSparseMatrixCSR(::CSR)` (shares one-based arrays,
    rebases zero-based ones into a copy, rejects batches), `to_backend(::SparseMatrixCSC, ::CUDABackend)`.
  - Exports: `CSR, csr_of_transpose, to_backend, nbatch, MatrixDescriptor, update!`.
  - `test/test_matrix.jl`.
- Tests: `SDS_TEST_GPU=0 julia --project=. -e 'using Pkg; Pkg.test()'` (Julia 1.13.1, ubuntu-latest):
  1885 pass / 0 fail / 0 broken (test_matrix alone: 963); same counts with Julia 1.10.10 (Aqua
  included). The CUDA extension was checked to load
  and define its methods in a scratch environment with CUDA.jl 6 (no GPU, `CUDA.functional() == false`).
  CUDA/AMDGPU: pending CI on the PR.
- Measurements: none asked.
- Deviations from PLAN.md / this task:
  - `size`, `nnz` and `SparseMatrixCSC(::CSR)` describe the *stored* CSR matrix; the `transposed`
    flag is left to consumers (the test applies it, as the task text says "after honoring the flag").
    For complex `T` the flag means plain transpose, not adjoint.
  - `rowptr` and `colval` share the type `VI`, so PLAN §1.1's `offsetType` (Int64 `rowPtr` with Int32
    `colVal`) is not representable yet; mixed index types raise `InvalidValueError`. Adding a fifth
    type parameter later is mechanical.
  - `MatrixDescriptor` follows CUDSS.jl for `transposed`: `MatrixDescriptor(T, m, n; transposed = true)`
    has logical size `n × m` and expects an `(m, n)` column-major buffer (row-major `n × m`).
    Descriptors created from sizes only have `A = AbstractArray{T}` (accept any array of `T`);
    descriptors created from an array accept only that array type. `update!` accepts either a
    strided vector of the right length or an array with exactly the descriptor's shape.
  - The CUDA test builds `CuSparseMatrixCSR{T,INT}`/`CuSparseMatrixCSC{T,INT}` from host arrays
    instead of `to_device(backend, A, INT)`: cuSPARSE 6 ignores `Ti` in
    `CuSparseMatrixCSR{Tv,Ti}(::SparseMatrixCSC)` and always uses `Cint` (issue #30). Likewise
    cuSPARSE's own `SparseMatrixCSC(::CuSparseMatrixCSR{T,Int64})` goes through `Int32` buffers and
    fails, so tests convert device matrices back through `SparseMatrixCSC(CSR(dA))` (noted on #30).
- Open issues / follow-ups:
  - #30: `test/backends.jl` `to_device(::CUDABackend, A, Int64)` returns Int32 indices.
  - No ROCm adapters (`CSR(::ROCSparseMatrixCSR)`) until T23; on the `amdgpu` runner the generic
    `to_backend`/`MatrixDescriptor` tests run on ROCm through `KernelAbstractions.allocate`.
  - `SparseMatrixCSC(::CSR)` on a device copies arrays to the host; it is a test/debug helper only.
- Suggested plan changes:
  - PLAN §1.1 / T02: decide whether `offsetType` (separate `rowptr` eltype) is needed for v1; if so,
    give `CSR` separate `VP`/`VI` parameters.

---

## T03 — Dense-op interface, capability audit, KA fallbacks   `[!]`

**Reads**: PLAN §2.6, §2.7, §2.4 (kernel design rules).

#### Deliverables

* `src/dense/interface.jl`: `gemm!(C, A, B, α, β; transA=:N, transB=:N)`,
  `syrk!(C, A, α, β; uplo=:L)` (`herk!` for complex), `trsm!(side, uplo, trans,
  diag, α, A, B)`, `potrf!(uplo, A) -> info::Int`, `getrf!(A, ipiv) -> info`,
  `laswp!(A, ipiv)`, strided batched `gemm_strided_batched!`,
  `trsm_strided_batched!` (uniform sizes: data pointer + stride + count).
  Each op takes `impl::Symbol = :auto` with values `:generic`, `:vendor`, `:ka`.
  Dispatch: `:auto` picks vendor if the capability table says so, else generic
  if available, else KA.
* `src/dense/capabilities.jl`: `capabilities(backend, T)` probes with small
  try/catch calls (generic `mul!`, triangular `ldiv!`, `cholesky!`, `lu!`;
  vendor syrk/herk, gemm_strided_batched, trsm_batched, potrfBatched,
  getrf_batched, sytrf; whether `reshape(view(buf, r), f, w)` is accepted by
  `mul!`; whether `Atomix.@atomic` add works on `T`), caches per
  `(backend, T)`, `print_capabilities(io, backend)`.
* `src/dense/fallback/`: KA kernels, correctness first, written with 1-D
  workgroups and `@localmem` only: `ka_gemm!` (tiled, `Val(16)`/`Val(32)`),
  `ka_syrk!`, `ka_trsm!` (per RHS column, forward/back substitution),
  `ka_potrf!` (one workgroup, unblocked, returns info through a device scalar),
  `ka_getrf!` (one workgroup, partial pivoting), `ka_laswp!`, strided batched
  variants of gemm/trsm (batch index = extra ndrange dimension).
* CUDA extension additions: vendor bindings (`cuBLAS.syrk!/herk!`,
  `gemm_strided_batched!`, `trsm_batched!`, `cuSOLVER.potrf!`, `potrfBatched!`,
  `getrf!`, `sytrf!` probe only). Add `cuBLAS`, `cuSOLVER` to the extension's
  weakdeps.

#### Tests that must pass (`test/test_dense.jl`, CPU + CUDA, `T ∈ ELTYPES`)

* For every op, for every `impl` the capability table marks available on that
  backend, and for sizes `(m,n,k) ∈ {(1,1,1), (7,5,3), (32,32,32), (100,64,33), (257,17,9)}`:
  result equals the `LinearAlgebra` reference within `50·eps(real(T))·norm`.
  Inputs include panel views `reshape(view(buf, a:b), f, w)` with `f > w`.
* `potrf!` returns `info == j` (1-based) for a matrix made non-SPD at column `j`
  for every impl; `getrf!` satisfies `P*L*U ≈ A` (`P` from `ipiv`) for every impl.
* Strided batched gemm/trsm with `count ∈ {1, 4, 33}` equals per-matrix results.
* The audit prints a table; asserts `generic_mul == true` on both backends and
  on CUDA `vendor_syrk == vendor_gemm_strided_batched == vendor_potrf == true`.
  Record the full CUDA table in the Report.

### Report

- Status: [!] (done on the CPU backend; CUDA results and the CUDA capability table come from CI)
- What was built:
  - `src/dense/interface.jl`: `gemm!(C, A, B, α, β; transA, transB, impl)`, `syrk!(C, A, α, β; uplo, impl)`,
    `herk!` (real `α`, `β`; real `T` → `syrk!`), `trsm!(side, uplo, trans, diag, α, A, B; impl)`,
    `potrf!(uplo, A; impl) -> info::Int`, `getrf!(A, ipiv; impl) -> info::Int` (m × n, LAPACK pivots),
    `laswp!(A, ipiv; reverse, impl)`, `gemm_strided_batched!`, `trsm_strided_batched!` (3-D arrays, batch =
    3rd dim), `strided_batch(buf, offset, m, n, stride, count)` (zero-copy 3-D view: data pointer + stride +
    count), `dense_impls(op, backend, T)`, `select_impl(op, X, impl)`, table `DENSE_OPS`. Flags accept `'N'` or `:N`.
    `:auto` = vendor, else generic, else KA; an explicit unavailable impl raises `NotSupportedError`.
  - `src/dense/capabilities.jl`: `DenseCapabilities` (17 Bool probes), `capabilities(backend, T)` (cached per
    `(backend, T)` under a lock; each probe is a 4×4 call in try/catch *and* checks the result against host
    LinearAlgebra), `print_capabilities([io,] backend)`.
  - `src/dense/vendor.jl`: `vendor_gemm!`, `vendor_syrk!`, `vendor_herk!`, `vendor_trsm!`, `vendor_potrf!`,
    `vendor_getrf!`, `vendor_sytrf!`, `vendor_gemm_strided_batched!`, `vendor_trsm_batched!`,
    `vendor_potrf_batched!`, `vendor_getrf_batched!`; fallback methods throw `NotSupportedError`; host
    BLAS/LAPACK methods for host arrays (`Matrix` and views/reshapes with an `Array` parent; GPU arrays are
    `DenseArray`s too and must never reach host BLAS).
  - `src/dense/fallback/`: `common.jl` (`_get`/`_set!` serve matrices and 3-D batches with one kernel),
    `gemm.jl` (`ka_gemm!` tiled with `Val(16)`/`Val(32)` (32 when both dims of C ≥ 64), `ka_syrk!` (same
    kernel, triangle mask, `conjugate` for herk), `ka_gemm_strided_batched!`), `trsm.jl` (`ka_trsm!`,
    `ka_trsm_strided_batched!`: one workgroup per RHS line, both sides reduced to one substitution kernel),
    `potrf.jl` (`ka_potrf!(uplo, A, info)`), `getrf.jl` (`ka_getrf!(A, ipiv, info)` with a `@localmem`
    argmax tree reduction, `ka_laswp!`). All kernels: 1-D workgroups, `@localmem` sized from `Val`, no atomics,
    no subgroup ops, batch index = second ndrange dimension; `ka_potrf!`/`ka_getrf!` also take 3-D batches and
    write `info` to a device `Int32` vector without host synchronization.
  - `ext/SparseDirectSolverCUDAExt.jl`: methods of every `vendor_*` function for `StridedCuArray`s (cuBLAS
    gemm/syrk/herk/trsm/gemm_strided_batched, `cublas?trsmBatched` on a pointer array built from the strides,
    cuSOLVER potrf/getrf/sytrf/`cusolverDn?potrfBatched`, cuBLAS `getrf_batched!`); non-`Cint` pivot vectors go
    through a `Cint` temporary. `Project.toml`: `cuBLAS`, `cuSOLVER` added to `[weakdeps]`, the extension
    trigger list and `[compat]` (`"6"`).
  - Tests: `test/test_dense.jl`; shared helpers `dense_tol`, `DENSE_SIZES`, `ipiv_permutation` (`utils.jl`),
    `dense_hpd`, `dense_triangular` (`matrices.jl`), `to_panel` and `to_host(::SubArray)` (`backends.jl`).
- Tests: `SDS_TEST_GPU=0 julia --project=. -e 'using Pkg; Pkg.test()'` (Julia 1.13.1, ubuntu-latest, CPU):
  19115 pass / 0 fail / 0 broken (test_dense: 17228, 172 s). Julia 1.10.10, `test_dense` + `test_aqua`:
  17232 pass. The CUDA extension was loaded in a scratch environment with CUDA.jl 6.4.1 (no GPU): all
  `vendor_*` methods are defined. CUDA/AMDGPU: pending CI on the PR.
  Review round 1: 19942 pass / 0 fail / 0 broken (CPU, Julia 1.13; test_dense: 18055) after looping the
  batched-factorization and `:auto` testsets over `ELTYPES` and adding the oversized-`ipiv` and
  non-strided-view tests.
  CI fix round 1: the first CUDA run failed 4 tests (`potrf 32`, ComplexF64, `j = 32`, panel and plain):
  the cuSOLVER-backed `:vendor`/`:generic` paths returned `info = 0` for a matrix whose last pivot is
  negative. `potrf!` now validates `info = 0` from vendor/generic with `ka_chol_diag_info!` (first
  diagonal entry of the factor that is not a finite positive real). After the fix, CPU: 19117 pass.
- Measurements: capability table of the KA CPU backend (printed by `test_dense`):

  ```text
  capability                  | Float32 | Float64 | ComplexF32 | ComplexF64
  generic_mul/_trsm/_cholesky/_lu | yes | yes | yes | yes
  vendor_gemm/_syrk/_trsm/_potrf/_getrf/_sytrf | yes | yes | yes | yes   (host BLAS/LAPACK)
  vendor_herk                 | no      | no      | yes        | yes
  vendor_*_batched (4 probes) | no      | no      | no         | no
  reshape_view_mul            | yes     | yes     | yes        | yes
  atomic_add                  | yes     | yes     | no         | no
  ```

  The full CUDA table is printed by the `capability audit (CUDA)` testset in the CI log of the `cuda` job;
  it could not be recorded here (no GPU on the implementer runner).
- Deviations from PLAN.md / this task:
  - The KA CPU backend gets "vendor" bindings: host BLAS/LAPACK through LinearAlgebra (in the core, no new
    dependency). On CPU, `:vendor` and `:generic` therefore both reach LAPACK; batched vendor probes are `false`.
  - Batched ops have no `:generic` path (a per-member loop of generic calls would be `count` launches);
    `:auto` picks vendor, else KA. `laswp!` has only `:ka` (no LinearAlgebra entry point, no cuSOLVER binding
    in CUDA.jl); both are listed in `DENSE_OPS`.
  - `syrk!` with `impl = :generic` uses `mul!` and overwrites the opposite triangle with the same symmetric
    update; `:vendor`/`:ka` leave it untouched (documented, tested).
  - `sytrf` and the batched potrf/getrf are vendor bindings + probes only, as asked; `ka_potrf!`/`ka_getrf!`
    accept 3-D batches anyway (one workgroup per member), which the batched tests exercise on the KA path.
  - `strided_batch` needs `stride` to be a multiple of `m` when `stride > m n` (the padded batch is a view
    of a reshaped contiguous range; Julia cannot express arbitrary member strides otherwise).
  - Test tolerances (`dense_tol = 50·eps`) multiply Frobenius norms of the operands: `|α|‖A‖‖B‖ + |β|‖C‖`
    for gemm/syrk, `‖ref‖` for trsm (well-conditioned `dense_triangular`), `‖L‖²` for potrf and `‖L‖‖U‖`
    for getrf (backward-error form).
  - Nothing is exported: the dense layer is internal (`SparseDirectSolver.gemm!`, …), PLAN §3.1 lists no
    dense names in the public API.
- Open issues / follow-ups:
  - The interface allocates small device scalars (`info`, `Cint` pivot temporaries, cuBLAS pointer arrays)
    and `potrf!`/`getrf!` read `info` on the host. T09 should call `ka_potrf!`/`ka_getrf!` with
    preallocated `info` and keep the reads at phase boundaries (PLAN §3.9).
  - cuBLAS' own `trsm_batched!` accepts only `Vector{<:CuArray}` (its `unsafe_batch` has no method for
    views), so the extension calls `cublas?trsmBatched[_64]` directly; check on CI that the pointer-array
    path works for padded `strided_batch` views. Issue #37 (found-by-agent).
  - `laswp!`/`ka_laswp!` take `npiv` (LAPACK `k2`, default `length(ipiv)`); T09/T10 must pass
    `npiv = min(m, n)` when the pivot buffer is preallocated and oversized (review round 1).
  - The KA kernels are correctness-first (unblocked potrf/getrf, one workgroup per matrix; one workgroup per
    RHS column in trsm). Blocked/tiled large-front kernels belong to T09–T11.
  - cuSOLVER `zpotrf` (n = 32) did not report a non-positive last pivot (CUDA CI of T03); `potrf!` checks
    the factor diagonal after vendor/generic. The batched vendor potrf (`cusolverDn?potrfBatched`) may need
    the same check (`ka_chol_diag_info!` accepts 3-D batches) when T09 uses it.
  - `atomic_add` is `false` for complex types on CPU (Atomix has no complex atomics); the default forward
    solve (T12) must split complex accumulation into real/imag parts or use the atomic-free variant.
    Issue #36 (found-by-agent).
  - `ka_gemm!` with `Val(32)` launches 1024-item workgroups (32 KiB local memory for `ComplexF64`); T23
    should make `_default_tile` consult the backend's maximum workgroup size (oneAPI may cap at 512).
- Suggested plan changes:
  - PLAN §2.6: state that the KA CPU backend uses host BLAS/LAPACK as its "vendor" library.
  - PLAN §2.6: list `laswp` as KA-only on CUDA (CUDA.jl has no `laswp` binding) unless a raw
    `cusolverDn?laswp` binding is added.

---

## T04 — Benchmark harness and cuDSS baselines   `[!]`

**Reads**: PLAN §5 (M0), §7; RESEARCH "Phase 0" and section 2.

#### Deliverables

* `bench/Project.toml` (CUDA, CUDSS, MatrixDepot, SparseArrays, LinearAlgebra,
  Statistics, DelimitedFiles; MadNLP/ExaModels/PGLib optional),
  `bench/matrices.jl`: generated Laplacians (2D 300×300, 3D 40³), a list of
  SuiteSparse names via MatrixDepot (at least `HB/bcsstk17`, `Boeing/bcsstk38`,
  `GHS_psdef/apache2`, `Rajat/rajat21`, `TSOPF/TSOPF_RS_b39_c7`), and a loader
  for `.mtx` or `.jld2` dumps under `bench/data/` (gitignored) following a
  naming convention `kkt_<case>_<kind>_<iter>.mtx` with `kind ∈ {k2, condensed}`.
* `bench/dump_madnlp_kkt.jl`: script that, when MadNLP + ExaModels + PGLib are
  available, dumps K2 and condensed KKT matrices of a pglib case at three
  iterations. If those packages are not installed, the script prints how to
  install them and exits 0.
* `bench/cudss_baseline.jl`: for each matrix and `structure ∈ {"SPD","S"}`
  (as applicable): cuDSS analysis / factorization / refactorization / solve
  (`nrhs = 1`) median times over 5 runs after a warm-up, plus `lu_nnz`,
  `flops`, `nsuperpanels`; writes `bench/results/cudss_baseline.csv`.
  `bench/report.jl` prints the CSV as a table.
* `bench/README.md` with the commands.

#### Tests that must pass (`test/test_bench_smoke.jl`, CPU only)

* `include("../bench/matrices.jl")` (without MatrixDepot network access: the
  generated matrices only) returns matrices with the documented names/sizes.
* The timing helper used by the baseline runs on a 10×10 Laplacian with a dummy
  solver closure and returns a `NamedTuple` with the documented fields.

**Report must include** the baseline table for every matrix that could be
obtained on this machine (CUDSS.jl is in `../CUDSS.jl`).

### Report

- Status: [!] (harness done and exercised end to end on the CPU; the cuDSS table was measured
  afterwards on the owner's RTX 4080, issue #40)
- What was built:
  - `bench/Project.toml` (CUDA 6, CUDSS 0.8, MatrixDepot 1, DelimitedFiles, LinearAlgebra, Random,
    SparseArrays, Statistics; resolves and loads on ubuntu-latest, `CUDA.functional() == false`).
    MadNLP/ExaModels/ExaModelsPower are not in it (separate `bench/kkt` env, gitignored).
  - `bench/matrices.jl`, module `BenchMatrices`: `BenchMatrix(name, source, structures, A)`;
    `generated_matrices()` (`lap2d_300`: n = 90 000, nnz = 448 800; `lap3d_40`: n = 64 000,
    nnz = 438 400; table `GENERATED`); `suitesparse_matrices()` through MatrixDepot (optional, guarded by
    `Base.find_package`) for `SUITESPARSE` = `HB/bcsstk17`, `Boeing/bcsstk38`, `GHS_psdef/apache2`
    (`SPD`,`S`), `Rajat/rajat21`, `TSOPF/TSOPF_RS_b39_c7` (`G`); `dump_matrices(dir)` for
    `bench/data/kkt_<case>_<kind>_<iter>.{mtx,jld2}` (`k2` → `S`, `condensed` → `SPD`,`S`; `.jld2` needs
    JLD2, key `"A"`); `dump_name`, `parse_dump_name`, dependency-free `read_mtx`/`write_mtx`,
    `symmetrize_triangle` (keeps explicit zeros), `bench_matrices(; generated, suitesparse, dumps)`.
  - `bench/harness.jl`, module `BenchHarness`: `time_phases(make_solver, A, b; nruns = 5, nwarmup = 1,
    synchronize)` → `(analysis, factorization, refactorization, solve, samples, nruns, lu_nnz, flops,
    nsuperpanels, relres)` (medians in seconds; `make_solver(A, b)` returns closures `analysis`,
    `factorization`, `refactorization`, `solve` (returns `x`), optional `stats`); `cholmod_solver(structure)`
    CPU reference; `CSV_COLUMNS`, `csv_row`, `write_csv`.
  - `bench/cudss_baseline.jl` (`--solver=cudss|cholmod`, `--nruns`, `--only`, `--no-suitesparse`,
    `--no-dumps`, `--out`): every matrix × applicable structure, symmetric structures pass the lower
    triangle with view `'L'`; `lu_nnz`, `nsuperpanels` via `cudss_get`, `flops` via the C API
    (`cudssDataGet(..., CUDSS_DATA_FLOPS, Ref{Int64})`, CUDSS.jl has no getter; first read as a
    `Float64`, corrected to `Int64` in #40); a failing matrix
    becomes a CSV row with its error as status. Writes `bench/results/<solver>_baseline.csv`.
  - `bench/report.jl` (Markdown table of the CSVs, ms), `bench/dump_madnlp_kkt.jl`, `bench/README.md`.
  - `test/test_bench_smoke.jl`; `Statistics` added to `test/Project.toml` (the harness uses `median`).
- Tests: `SDS_TEST_GPU=0 julia --project=. -e 'using Pkg; Pkg.test()'` (Julia 1.13.1, ubuntu-latest, CPU):
  20013 pass / 0 fail / 0 broken (test_bench_smoke: 71, 7 s). CUDA/AMDGPU: pending CI on the PR
  (the smoke test is CPU-only and does not touch the GPU).
  Scripts run by hand: `dump_madnlp_kkt.jl` without the packages prints the install hint and exits 0;
  with MadNLP 0.10.1 / ExaModels 0.12.1 / ExaModelsPower 0.3.1 it wrote six case118 dumps (23 s);
  `cudss_baseline.jl` (cudss) stops with "CUDA is not functional … use --solver=cholmod";
  `cudss_baseline.jl --solver=cholmod` ran on all 13 matrices (MatrixDepot downloads worked).
- Measurements:
  - cuDSS baseline (issue #40; `julia --project=bench bench/cudss_baseline.jl`, all 39 matrix ×
    structure rows `ok`): NVIDIA GeForce RTX 4080, WSL2, CUDA driver/runtime 13.4, cuDSS 0.8.0
    (CUDSS.jl 0.8.1, CUDA.jl 6.4.1), Julia 1.13.1; default cuDSS configuration; median of 5 after 1
    warm-up, times in ms; `flops` = `CUDSS_DATA_FLOPS`, `nsp` = `nsuperpanels`. KKT dumps of
    pglib_opf_case14_ieee / case118_ieee / case1354_pegase written by `bench/dump_madnlp_kkt.jl
    --iters=1,10,20` with MadNLP 0.9.2, ExaModels 0.9.7, ExaModelsPower 0.3.1 (case14 stops at 11
    (condensed) / 15 (K2)).

    ```text
    matrix                            struct       n      nnz  analysis  factor  refactor  solve     lu_nnz    flops nsp   relres
    lap2d_300                         SPD      90000   448800     271.0    6.05      4.15  0.655    2428175   3.18e8   0  1.8e-12
    lap2d_300                         S        90000   448800     300.0    16.5      14.2   1.38    2428175   3.18e8   0  1.6e-12
    lap3d_40                          SPD      64000   438400     342.0    50.3      45.6   1.38   17527585  2.19e10   2  8.3e-14
    lap3d_40                          S        64000   438400     345.0    63.7      58.7   1.46   17527585  2.19e10   2  7.7e-14
    HB/bcsstk17                       SPD      10974   428650      42.0    3.53      2.34  0.419    1077096   1.67e8   0  3.7e-11
    HB/bcsstk17                       S        10974   428650      41.5    4.53      3.31   0.39    1077096   1.67e8   0  4.3e-11
    Boeing/bcsstk38                   SPD       8032   355460      35.8    3.34      2.31  0.464     798965    1.3e8   0  2.2e-10
    Boeing/bcsstk38                   S         8032   355460      36.1    4.21      3.23   0.48     798965    1.3e8   0  8.2e-11
    GHS_psdef/apache2                 SPD     715176  4817870    4350.0   447.0     418.0   6.57  144624555  2.29e11  21  1.8e-10
    GHS_psdef/apache2                 S       715176  4817870    4380.0   520.0     491.0   6.77  144624555  2.29e11  21  1.6e-10
    Rajat/rajat21                     G       411676  1893370    1780.0    31.8      29.4   2.03    6891584   5.21e9   2    3.9e7
    TSOPF/TSOPF_RS_b39_c7             G        14098   252446      85.4    2.32       1.0  0.301    1470000   9.55e7   0   3.8e29
    kkt_case118_ieee_condensed_1      SPD       1088    12860      11.5   0.908     0.281  0.183      15200 272000.0   0   4.5e-5
    kkt_case118_ieee_condensed_1      S         1088    12860      12.2    1.04     0.329  0.212      15200 272000.0   0   1.9e-5
    kkt_case118_ieee_condensed_10     SPD       1088    12860      12.2    1.02     0.319  0.242      15200 272000.0   0   1.2e-5
    kkt_case118_ieee_condensed_10     S         1088    12860      12.0   0.814     0.324  0.202      15200 272000.0   0   1.0e-5
    kkt_case118_ieee_condensed_20     SPD       1088    12860      11.8   0.846     0.296  0.193      15200 272000.0   0      0.3
    kkt_case118_ieee_condensed_20     S         1088    12860      11.5   0.774     0.311  0.348      15200 272000.0   0     0.12
    kkt_case118_ieee_k2_1             S         3150    17714      16.5   0.938     0.315  0.211      20211 197000.0   0      1.9
    kkt_case118_ieee_k2_10            S         3150    17714      15.9   0.792     0.336  0.307      20211 197000.0   0      7.3
    kkt_case118_ieee_k2_20            S         3150    17714      15.9   0.749     0.323  0.329      20211 197000.0   0      8.5
    kkt_case1354_pegase_condensed_1   SPD      11192   136724      52.1     1.1     0.582  0.342     155201   3.33e6   0  0.00034
    kkt_case1354_pegase_condensed_1   S        11192   136724      53.2    1.31     0.616  0.261     155201   3.33e6   0  0.00031
    kkt_case1354_pegase_condensed_10  SPD      11192   136724      51.8    1.09     0.652  0.319     155201   3.33e6   0  0.00014
    kkt_case1354_pegase_condensed_10  S        11192   136724      52.0    1.32     0.616  0.351     155201   3.33e6   0  0.00014
    kkt_case1354_pegase_condensed_20  SPD      11192   136724      57.1    1.46     0.796   0.36     155201   3.33e6   0   6.8e-5
    kkt_case1354_pegase_condensed_20  S        11192   136724      58.1    1.73     0.953  0.387     155201   3.33e6   0   8.6e-5
    kkt_case1354_pegase_k2_1          S        33811   188063     108.0     2.2      1.38  0.527     202300   2.33e6   0     85.0
    kkt_case1354_pegase_k2_10         S        33811   188063     113.0    2.44      1.58  0.619     202300   2.33e6   0     57.0
    kkt_case1354_pegase_k2_20         S        33811   188063     115.0    2.69      1.82  0.641     202300   2.33e6   0    170.0
    kkt_case14_ieee_condensed_1       SPD        118     1282      17.8   0.326     0.282  0.204       1268  15300.0   0   7.1e-7
    kkt_case14_ieee_condensed_1       S          118     1282      17.4   0.291     0.313  0.281       1268  15300.0   0   6.1e-7
    kkt_case14_ieee_condensed_10      SPD        118     1282      7.26   0.135     0.131  0.122       1268  15300.0   0   8.5e-5
    kkt_case14_ieee_condensed_10      S          118     1282      7.24   0.135     0.138  0.144       1268  15300.0   0   0.0001
    kkt_case14_ieee_condensed_11      SPD        118     1282      7.27   0.169     0.142  0.139       1268  15300.0   0   2.1e-5
    kkt_case14_ieee_condensed_11      S          118     1282      7.34   0.136      0.13  0.139       1268  15300.0   0   1.0e-5
    kkt_case14_ieee_k2_1              S          344     1924      7.85   0.643     0.337  0.236       2024  15100.0   0     0.26
    kkt_case14_ieee_k2_10             S          344     1924      7.44   0.516     0.155  0.151       2024  15100.0   0     0.28
    kkt_case14_ieee_k2_15             S          344     1924      7.48   0.508      0.13  0.145       2024  15100.0   0     0.31
    ```

    Notes on the cuDSS run:
    - `CUDSS_DATA_FLOPS` is an `int64` in cuDSS 0.8.0, not a double (read as `Float64` it gave the
      denormal 1.57e-315, i.e. the `Int64` 317 918 802 for lap2d_300). The value is plausible: with
      cuDSS's 2.43e6-entry factor it compares to Σ colcount² = 5.18e8 of CHOLMOD's 4.12e6-entry
      factor. `bench/cudss_baseline.jl` now reads an `Int64`.
    - `analysis` is never re-run on the same `CudssSolver`: `time_phases` builds a fresh solver per
      run, so each solver sees analysis → factorization → refactorization → solve once.
    - The large residuals are cuDSS's default configuration (static pivoting with perturbation, no
      matching, no refinement), not the harness: UMFPACK solves the same systems to 3e-6 (rajat21),
      1e-12 (TSOPF_RS_b39_c7), 1e-11 (case118 K2) and 1e-10 (case1354 K2). cuDSS reports `npivots` =
      1517 (rajat21), 9 (TSOPF), 1036 (case118 K2 10), 8059 (case1354 K2 10). With `"matching_alg" =
      "algo5"` they become 2.7e-5, 2.2e-12, 5.4e-13 and 7.7e-3 (`npivots` 0, 0, 0, 14); with
      `"ir_n_steps" = 5` on top, 4.4e-7, 9.8e-13, 3.7e-13 and 4.6e-11 (one-off check by hand, not in
      the CSV). The condensed case118 KKT at iteration 20 is close to singular (`cond₁` ≈ 9e19,
      UMFPACK relres 0.13), so its 0.3 / 0.12 are not a cuDSS defect.
  - CPU reference (`--solver=cholmod`, CHOLMOD supernodal Cholesky for `SPD`, UMFPACK for `G`;
    ubuntu-latest, 4 cores, 2 BLAS threads; median of 5 after 1 warm-up, ms):

    ```text
    matrix                         struct       n       nnz  analysis  factor  refactor   solve      lu_nnz  relres
    lap2d_300                      SPD      90000    448800      19.5    45.0      34.1   7.56     4117190  1.6e-12
    lap3d_40                       SPD      64000    438400      35.0   383.3     320.8  16.3     22073203  5.9e-14
    HB/bcsstk17                    SPD      10974    428650      10.3    10.9      10.7   1.08     1124822  3.2e-11
    Boeing/bcsstk38                SPD       8032    355460      11.0    11.2      11.5   0.73      805686  1.2e-10
    GHS_psdef/apache2              SPD     715176   4817870     694.6  4666.0    4215.0 194.5    191800254  1.5e-10
    Rajat/rajat21                  G       411676   1893370   20910.0 11440.0   11210.0  10.3      3176760  3.4e-6
    TSOPF/TSOPF_RS_b39_c7          G        14098    252446      17.7    11.6      11.6   0.38      298985  1.1e-12
    kkt_..._case118_ieee_condensed_1  SPD    1088     12860      0.68    0.38      0.32  0.025       11634  3.4e-5
    kkt_..._case118_ieee_condensed_10 SPD    1088     12860      0.65    0.37      0.30  0.020       11634  1.4e-5
    kkt_..._case118_ieee_condensed_20 SPD    1088     12860      0.69    0.37      0.30  0.019       11634  4.0e-2
    kkt_..._case118_ieee_k2_{1,10,20} S      3150     17714   FAILED: ZeroPivotException (CHOLMOD ldlt, no pivoting)
    ```

    (`S` rows of the SPD matrices were measured once before the skip was added: CHOLMOD `ldlt` is
    simplicial, lap2d_300 219 ms, lap3d_40 18.8 s factorization; on apache2 it did not finish in 30 min.)
- Deviations from PLAN.md / this task:
  - The dump script builds the pglib-opf AC-OPF models with **ExaModelsPower** (`ac_opf_model`, fetches
    pglib-opf 23.07 through ExaPowerIO) instead of PGLib.jl: PGLib.jl only returns PowerModels data and
    would need a hand-written ExaModels OPF formulation. Default iterations are 1, 10, 20 (case118
    converges at 22 (condensed) / 29 (K2), so 30 did not exist); when MadNLP stops earlier the file is
    named after the actual iteration. The dumped matrix is `kkt.aug_com` after `max_iter = k`, i.e. the
    last factorized KKT (lower triangle, explicit zeros kept so every iteration has the same pattern).
  - Unsymmetric SuiteSparse matrices (rajat21, TSOPF_RS_b39_c7) run with structure `G` (neither `SPD`
    nor `S` applies); `structures` per matrix is in the `SUITESPARSE` table. The SuiteSparse
    TSOPF_RS_b39_c7 has n = 14 098 (checked on load).
  - Added a CPU reference solver (`--solver=cholmod`) to the same harness so the harness is exercised
    without a GPU and PLAN §7's CHOLMOD comparison has a starting point. It skips `S` where `SPD`
    applies (simplicial `ldlt`), and UMFPACK has no separate analysis in SparseArrays, so its
    "analysis" is a full `lu` and factorization/refactorization are `lu!`.
  - A fresh solver object is created (untimed) for every run, so each run times analysis →
    factorization → refactorization (same values) → solve on a new solver.
  - The 10×10 Laplacian of the smoke test is the 2-D Laplacian on a 10 × 10 grid (n = 100).
  - `bench/matrices.jl` has its own Laplacian generators (the bench env cannot include the test helpers);
    the smoke test checks they equal `test/matrices.jl`'s `laplacian2d`/`laplacian3d`.
- Open issues / follow-ups:
  - Owner: run the cuDSS baseline on the RTX 4080 and record it (tracked in #40). Done, see
    Measurements; `CUDSS_DATA_FLOPS` turned out to be an `Int64`.
  - Accuracy comparisons against cuDSS (M1/M3) should use the same configuration on both sides: with
    the default configuration cuDSS does not solve the unsymmetric SuiteSparse matrices or the K2 KKTs
    accurately (see the notes under Measurements), matching (`matching_alg`) does.
  - The condensed case118 KKTs are badly conditioned at late iterations (CHOLMOD relres 4e-2 at
    iteration 20 without refinement); cuDSS comparisons on condensed systems should report the residual
    after refinement (T16) as well.
  - NREL opf_matrices and a CUTEst subset (PLAN M0) are not in the harness yet; the dump loader accepts
    any `kkt_<case>_<kind>_<iter>.mtx`, other `.mtx` sources need a small loader. MA57 refinement counts
    (M0) are not measured (no HSL).
  - Julia 1.13 quirk: `abspath(PROGRAM_FILE) == @__FILE__ && main()` failed to parse as a script in an
    environment without a `julia` compat entry; the scripts use an `if … end` block instead.
- Suggested plan changes:
  - PLAN §5 M0 / T04: name ExaModelsPower (not PGLib.jl) as the source of pglib-opf models for the dumps.
  - PLAN §7: list the CHOLMOD/UMFPACK CPU reference as part of the harness (`--solver=cholmod`), and
    note that CHOLMOD `ldlt` is simplicial, so it is not a meaningful LDLᵀ performance reference.

---

## T05 — Symbolic I: pattern, ordering, elimination tree, column counts   `[!]`

**Reads**: PLAN §2.3 steps 1–3; RESEARCH section 4 (ordering).

#### Deliverables

* `src/symbolic/pattern.jl`: `SymmetricPattern(n, colptr, rowval)` (host `Int`,
  1-based, no diagonal, both triangles) built from a `CSR` pattern + `Structure`
  plus `MatrixView` + `IndexBase`: expand the given triangle (`'L'`/`'U'`), use
  only the lower triangle for `'F'` on symmetric types, symmetrize for `"G"`,
  drop duplicates, reject non-square. Also `full_pattern_map` (for IR SpMV,
  T16): the CSR of the full symmetric matrix on the host and, for each entry,
  the index into the user's `nzval` (with a conjugate flag for `"H"`).
* `src/symbolic/ordering.jl`: `compute_ordering(pattern, opts; T) -> Ordering`
  (`perm`, `iperm`, `alg_used`, `stats`): `"algo5"` natural; `user_perm`
  (validated with `isperm`, 0/1-based accepted); AMD via
  `CliqueTrees.permutation(graph; alg=AMD())`; MMD; ND via `METIS()` only when
  `ext/SparseDirectSolverMetisExt.jl` is loaded (weakdep `Metis`; the extension
  just flips a flag and provides the alg object); `"default"`: compute AMD and,
  if available, ND, evaluate `(nnzL, flops, nlevels)` for both with the etree
  code below and pick by `flops × (1 + nlevels / n)`; expose the choice in
  `stats`. Honor `nd_nlevels`/`nd_ubfactor` when ND is used.
* `src/symbolic/etree.jl`: `etree(pattern, perm)` (Liu, path compression),
  `postorder(parent)`, `colcounts(pattern, perm, parent, post)`
  (Gilbert–Ng–Peyton), `nnz_L(counts)`, `cholesky_flops(counts)`,
  `tree_levels(parent)` (height of each node, number of levels).

#### Tests that must pass (`test/test_symbolic_etree.jl`, host only)

* On 200 random symmetric patterns with `n ∈ 5:60`: `etree` equals a
  brute-force reference written in the test (dense symbolic elimination, parent
  = first off-diagonal nonzero of column `j` of the filled matrix) and
  `colcounts` equals the column counts of that filled matrix.
* On `laplacian2d(50, 50)`, `laplacian3d(12,12,12)` and `random_spd(2000, 0.002)`
  with AMD and natural orderings: `nnz_L(counts)` equals the nnz of the CHOLMOD
  factor computed with the same permutation (`cholesky(A; perm=perm)`,
  converted to `SparseMatrixCSC`, numerical zeros dropped).
* Every ordering returns a valid permutation; `user_perm` is returned
  unchanged (1-based) whether given 0- or 1-based; AMD gives `nnz_L` at most
  that of natural ordering on the Laplacians; `"default"` chooses an ordering
  with `flops` ≤ 1.1× the better of AMD/ND and reports it in `stats`.
* `SymmetricPattern` from `'L'`, `'U'`, `'F'` inputs of the same symmetric
  matrix are identical; `"G"` on an unsymmetric matrix gives the pattern of
  `A + Aᵀ`; 0-based input gives the same pattern as 1-based.
* `full_pattern_map` reconstructs the full matrix values exactly from `nzval`
  for `'L'`, `'U'`, `'F'` and for `"H"` with conjugation.

### Report

- Status: [!] (done; one new dependency and the `nd_nlevels` interpretation below)
- What was built:
  - `src/symbolic/pattern.jl`: `SymmetricPattern` (`n`, `colptr`, `rowval`: host `Int`, 1-based, sorted,
    no diagonal, no duplicates, both triangles; the raw constructor validates these invariants),
    `SymmetricPattern(A::CSR, structure; view)` and `SymmetricPattern(rowptr, colval, n, structure; view, index)`
    (structure/view/index as strings, chars or enums; `'L'`/`'U'` select a triangle, `'F'` on symmetric
    structures reads the lower triangle, `"G"` gives A + Aᵀ and requires `'F'`; non-square, out-of-range
    columns, bad `rowptr` → `InvalidValueError`), `neighbors(P, j)`, `nnz`, `==`, `SparseMatrixCSC(P)`.
    `FullPatternMap` + `full_pattern_map(A::CSR | rowptr, colval, n, structure; view, index)`: 1-based CSR of
    the full matrix plus a source list per entry (`srcptr`, `src` = position in the user's `nzval`,
    `conjflag` for mirrored `"H"`/`"HPD"` entries); duplicated user entries become several sources and are
    summed, so the T16 SpMV can gather without atomics. `full_values(F, nzval)` and
    `SparseMatrixCSC(F, nzval)` evaluate it on the host.
  - `src/symbolic/etree.jl`: `etree(P, perm)` (Liu, path compression, permuted numbering, roots = 0),
    `postorder(parent)` (non-recursive, children in increasing order, detects cycles),
    `colcounts(P, perm, parent, post)` (Gilbert–Ng–Peyton, CSparse `cs_counts` skeleton/leaf scheme,
    diagonal included), `nnz_L(counts)`, `cholesky_flops(counts) = Σ cⱼ²` (CHOLMOD convention),
    `tree_levels(parent) -> (height, nlevels)` (leaves height 1).
  - `src/symbolic/ordering.jl`: `compute_ordering(P, opts; T, alg) -> Ordering` (`perm`, `iperm`,
    `alg_used ∈ (:natural, :amd, :mmd, :nd, :user)`, `stats = (nnz_L, flops, nlevels, cost, candidates,
    auto, nd_available)`), `evaluate_ordering(P, perm; T)`, `ordering_cost(flops, nlevels, n) =
    flops × (1 + nlevels/n)`, `nd_available()`, `ND_PROVIDER`. `user_perm` wins over `reordering_alg`
    and is accepted 0- or 1-based (returned 1-based); `"algo5"` natural, `"algo3"` AMD
    (`CliqueTrees.AMD()`), `"algo4"` ND (Metis extension, `NotSupportedError` without it),
    `"algo1"/"algo2"` AMD on the symmetric pattern (setparam! already warns), `"default"` evaluates AMD
    and ND (if loaded) and keeps the lower cost. Keyword `alg = :mmd` reaches `CliqueTrees.MMD()` (no cuDSS
    spelling). `T` complex multiplies the reported flops by 4.
  - `ext/SparseDirectSolverMetisExt.jl` (weakdep `Metis`): sets `ND_PROVIDER` to a function returning
    `CliqueTrees.METIS(ufactor = nd_ubfactor)`.
  - Nothing new is exported (the symbolic layer is internal, like the dense layer).
  - Tests: `test/test_symbolic_etree.jl`; `test/runtests.jl` loads `Metis` (added to `test/Project.toml`).
- New dependencies:
  - `AMD` (hard dep, compat `0.5`): `CliqueTrees.AMD()` throws "`import AMD` to use algorithm AMD" unless
    AMD.jl is loaded (it is CliqueTrees' `AMDExt`); AMD.jl only wraps SuiteSparse_jll, already a dependency
    of SparseArrays.
  - `Metis` (weakdep + extension, compat `1`; test dependency): ND orderings as planned.
- Tests: `SDS_TEST_GPU=0 julia --project=. -e 'using Pkg; Pkg.test()'` (Julia 1.13.1, ubuntu-latest, CPU):
  21060 pass / 0 fail / 0 broken (test_symbolic_etree: 1047, 6–13 s). Julia 1.10.10,
  `test_symbolic_etree` + `test_aqua`: 1057 pass. CUDA/AMDGPU: pending CI on the PR (the new tests are
  host-only and do not touch a GPU).
- Measurements (ubuntu-latest, Julia 1.13, `"default"` after a warm-up; times include both candidates and
  their etree/colcount evaluation; pattern = `SymmetricPattern` from a `CSR`):

  ```text
  matrix            n      pattern  default  chosen | AMD nnz(L)  flops   levels | ND nnz(L)  flops   levels
  lap2d 300×300   90000  14–94 ms 338 ms   nd     |  2 928 059  4.67e8  1997  |  2 465 905  3.49e8   863
  lap3d 40³       64000  10–96 ms 381 ms   nd     | 20 614 676  3.27e10 6178  | 14 387 160  1.62e10 3311
  lap2d 50×50      2500   0.3 ms   5.9 ms   amd    |     35 913  1.04e6   249  |     40 203  1.32e6   139
  lap3d 12³        1728   0.2 ms   6.2 ms   nd     |     76 038  8.54e6   376  |     62 653  5.19e6   241
  random_spd 2000  2000   0.6 ms    12 ms   amd    |    489 460  2.96e8   966  |    573 264  3.60e8  1058
  kkt 3000+1000    4000   1.8 ms    29 ms   amd    |  2 111 983  2.77e9  2028  |  2 625 611  3.80e9  2298
  ```

  (pattern times varied between two runs, GC.) AMD alone: 20 ms (lap2d 300²), 46 ms (lap3d 40³); METIS: 280 ms / 420 ms; etree + colcounts: 10–16 ms.
  `levels` here is the height of the column elimination tree (before supernodes, T06).
- Deviations from PLAN.md / this task:
  - `nd_nlevels`: ND is `METIS_NodeND` (`CliqueTrees.METIS(ufactor = nd_ubfactor)`), which has no level
    parameter; `nd_nlevels` is read as cuDSS documents it, a *minimum* number of dissection levels, which
    NodeND's full recursion meets whenever the graph is large enough. I first used CliqueTrees'
    level-capped `ND{3}(AMD(), METISND(); level = nd_nlevels)`; it was 4–13× slower than NodeND and gave
    more fill than AMD (lap2d 300²: 4.50 M vs NodeND 2.47 M; lap3d 40³: 23.8 M vs 14.4 M), so I dropped it.
    `nd_nlevels` becomes meaningful with the partition-tree export (T24).
  - `view`/`index` refer to the *stored* CSR arrays (cuDSS semantics; MadNLP passes CSC as CSR with `'U'`).
    `SymmetricPattern` ignores `CSR.transposed` (same pattern either way); `full_pattern_map` describes
    the stored matrix and leaves `transposed` to the solver (`solve_mode`, T16).
  - `"G"` with view `'L'`/`'U'` raises `InvalidValueError` (no silent fallback).
  - The random symmetric patterns of the etree test come from the existing `random_symindef(n, density)`
    with densities 0.02–0.4; odd trials use a random permutation, even ones the natural order.
  - `compute_ordering` takes an extra `alg` keyword (MMD has no `reordering_alg` spelling).
- Open issues / follow-ups:
  - `max_lu_nnz` is not checked yet: it needs the post-amalgamation `lu_nnz` (T06/T07).
  - Analysis copies `rowptr`/`colval` to the host (`Array(...)`) once; that is the planned phase boundary.
  - The cost model keeps AMD on the KKT and random matrices (lower flops and fewer levels there) and ND on
    the big Laplacians; the `nlevels/n` term is small for these sizes, so the choice is flop-driven. T06
    should revisit it with supernodal levels (the number of launches) instead of column-etree height.
  - `FullPatternMap.src` indexes the first batch member; T16 adds `(k - 1) * nnz` per member.
- Suggested plan changes:
  - PLAN §2.3 step 2: name AMD.jl as a dependency (CliqueTrees' AMD needs it) and state that ND is
    `METIS_NodeND` with `nd_nlevels` as a minimum.

---

## T06 — Symbolic II: supernodes and GPU-tuned amalgamation   `[!]`

**Reads**: PLAN §2.3 steps 3–4; RESEARCH section 4 (amalgamation).

#### Deliverables

* `src/symbolic/supernodes.jl`: `fundamental_supernodes(parent, post, counts)`;
  `amalgamate(sn, parent, counts, params)` with
  `params = (max_width = 32, zero_fraction = 0.25, min_width = 8)` from
  `opts.amalgamation` (`use_superpanels = 0` disables amalgamation);
  `SupernodePartition`: `super_ptr` (column ranges), `snparent`, `snpost`,
  `rows` per supernode (sorted, the supernode's own columns first), `nnz_stored`
  (including explicit zeros), `nnz_L` (true), `flops`; supernodal symbolic
  factorization computing `rows` by unions over children. `nsuperpanels`.

#### Tests that must pass (`test/test_symbolic_supernodes.jl`, host only)

* Partition validity: ranges contiguous and covering `1:n`; `snparent` is a
  forest consistent with the column etree; `snpost` is a valid postorder.
* Structure: for the small brute-force cases of T05, every structural nonzero
  of `L` lies inside a stored panel, and with amalgamation disabled
  `nnz_stored == nnz_L`.
* With amalgamation enabled, `nnz_stored ≤ (1 + zero_fraction) · nnz_L`, every
  supernode width `≤ max_width`, and on `laplacian2d(100,100)` with AMD the
  number of supernodes is at least 2× smaller than with amalgamation disabled.
* `flops` equals `cholesky_flops` of T05 when amalgamation is disabled and is
  ≥ it when enabled.

### Report

- Status: [!] (done; wide supernodes are split to `max_width`, an extra renumbering, see deviations)
- What was built (`src/symbolic/supernodes.jl`, internal like the rest of the symbolic layer, nothing exported):
  - `ColumnPartition(order, super_ptr, snparent)`: supernodes as contiguous column ranges after a renumbering;
    `order` is a topological order of the column etree (column `k` of the supernodal numbering is etree
    column `order[k]`). `nsupernodes`.
  - `fundamental_supernodes(parent, post, counts)`: renumbers by `post` and joins `k+1` to `k` when `k` is
    its only child and `counts[k] == counts[k+1] + 1` (written here, not CliqueTrees' `supernodetree`).
  - `amalgamate(sn, parent, counts, params)` with `params = opts.amalgamation`: (1) supernodes wider than
    `max_width` are split into a chain of near-equal panels; (2) bottom-up over the tree, the children of
    each supernode are tried in decreasing order of their off-diagonal row count (= increasing explicit
    zeros per column, an order merging does not change) and merged when the width stays `≤ max_width`,
    the factor-wide explicit zeros stay `≤ zero_fraction · nnz(L)` (a global budget), and either the merged
    panel has `≤ zero_fraction` explicit zeros per true nonzero or its width is `≤ min_width` (tiny panels
    merge more eagerly, still inside the global budget). Merging child `c` into `p` adds
    `w_c (w_c + f_p − f_c)` zeros and gives `f = w_c + f_p` rows. (3) The merged tree is postordered and
    columns are renumbered so every merged supernode is contiguous.
  - `SupernodePartition` (`n`, `perm`/`iperm` = T05 ordering composed with the supernodal renumbering,
    `parent`/`counts` = column etree and true counts in that numbering, `super_ptr`, `col2sn`, `snparent`,
    `snpost`, `rowptr`/`rowval` = sorted rows per supernode, own columns first, `nnz_stored` (lower-trapezoidal
    panel entries incl. explicit zeros), `nnz_L` (true), `flops` (`Σ` over panel columns of rows², the
    `cholesky_flops` convention), `amalgamated`); constructor
    `SupernodePartition(P, perm, parent, counts, cp; amalgamated)` = supernodal symbolic factorization by
    unions of the pattern columns and the children's rows; accessors `sncols`, `snrows`, `snwidth`,
    `nsuperpanels` (the `"nsuperpanels"` data parameter, wired to `getparam` in T13).
  - `supernode_partition(P, perm, opts)`: etree → postorder → colcounts → fundamental → amalgamate (unless
    `use_superpanels == 0`) → symbolic.
  - Tests: `test/test_symbolic_supernodes.jl`. `brute_force_symbolic` and `offdiag_pattern` moved from
    `test/test_symbolic_etree.jl` to `test/utils.jl` (shared; `brute_force_symbolic` now also returns the
    filled pattern); no T05 assertion changed.
- Tests: `SDS_TEST_GPU=0 julia --project=. -e 'using Pkg; Pkg.test()'` (Julia 1.13.1, ubuntu-latest, CPU):
  31062 pass / 0 fail / 0 broken (test_symbolic_supernodes: 10002, 3–11 s). Besides the listed checks the
  tests verify `parent == etree(P, perm)` and `counts == colcounts(...)` for the composed permutation, that a
  child's below-diagonal rows are a subset of its parent's rows (needed by T07 `relind`), `nnz_L` against
  CHOLMOD for the composed permutation, the stored panels' exact structure without amalgamation, and the
  amalgamation bounds on the 200 brute-force cases with four parameter sets. Julia 1.10 was not available
  on the runner (not run). CUDA/AMDGPU: pending CI on the PR (the new tests are host-only).
- Measurements (ubuntu-latest; `fund` = `use_superpanels = 0`, `amal` = defaults `(32, 0.25, 8)`;
  `ns` supernodes, `w̄` mean width, `<8` fraction of supernodes narrower than 8, `lev` supernodal tree
  height, `stored/L` = `nnz_stored / nnz_L`, `fl` = flops / `cholesky_flops`):

  ```text
  matrix          ord |  fund ns   w̄   <8   lev |  amal ns   w̄   <8   lev  stored/L   fl
  lap2d 100²      amd |    7510  1.33 0.99  42  |    3407  2.94 0.93   45   1.250   1.18
  lap2d 100²      nd  |    7654  1.31 0.99  27  |    3616  2.77 0.85   32   1.250   1.18
  lap2d 300²      amd |   67510  1.33 0.99  54  |   13734  6.55 0.89   83   1.236   1.06
  lap2d 300²      nd  |   69186  1.30 0.99  34  |   20781  4.33 0.69   48   1.250   1.07
  lap3d 12³       amd |    1199  1.44 0.98  33  |     304  5.68 0.59   21   1.250   1.13
  lap3d 12³       nd  |    1206  1.43 0.99  21  |     541  3.19 0.81   20   1.250   1.18
  lap3d 30³       amd |   18240  1.48 0.99  38  |    4609  5.86 0.63   96   1.079   1.02
  lap3d 30³       nd  |   17977  1.50 0.98  46  |    4569  5.91 0.57   70   1.115   1.03
  kkt 3000+1000   amd |    2238  1.79 0.99 262  |     534  7.49 0.63   89   1.223   1.22
  kkt 3000+1000   nd  |    2369  1.69 0.97 666  |     724  5.52 0.75  132   1.250   1.28
  random_spd 2000 amd |    1026  1.95 0.98  35  |     253  7.91 0.57   47   1.155   1.10
  random_spd 2000 nd  |    1007  1.99 0.96 144  |     257  7.78 0.60   53   1.176   1.15
  ```

  `supernode_partition` (etree + colcounts + supernodes + rows) takes 1–5 ms for n ≤ 4000, 3–70 ms for
  n = 10⁴–9·10⁴ (GC noise between runs). The T06 target holds: lap2d 100² AMD 7510 → 3407 supernodes.
- Deviations from PLAN.md / this task:
  - The supernodal numbering differs from the T05 ordering: `SupernodePartition.perm = perm[order]`, a
    topological reordering of the etree (same fill), because merged supernodes are only contiguous after
    renumbering. Every later phase must use `SupernodePartition.perm`, not `Ordering.perm`.
  - With amalgamation on, fundamental supernodes wider than `max_width` are split into chains (the test
    requires every width `≤ max_width`). This lengthens the tree under big separators: lap3d 30³ AMD has
    96 supernodal levels with the split and 15 without it (`max_width = 10⁶`, widest panel 2306 columns);
    KKT levels still drop 262 → 89.
  - `fundamental_supernodes` is our code (a dozen lines), not CliqueTrees' `supernodetree`, so the
    partition is built directly on our etree/colcounts.
  - `zero_fraction` is enforced as a global bound on `nnz_stored − nnz_L` (≤ `zero_fraction · nnz_L`), plus a
    local per-panel ratio unless the panel is `≤ min_width` wide; `min_width` is a "merge eagerly below"
    threshold, not a guarantee.
- Open issues / follow-ups:
  - The 25 % budget is exhausted on the 2D Laplacians while most panels stay narrower than 8 (lap2d 100²:
    93 %); bottom-up greedy spends the budget in tree order. A global cheapest-first merge (priority queue)
    or a larger budget for small fronts is a T25 tuning item; the KKT/random matrices reach `w̄ ≈ 6–8`.
  - `max_lu_nnz` can now be checked against `nnz_stored` (`× 2 − n` for `"G"`); left to the phase code
    (T07/T13), which owns the error reporting.
  - T05's ordering cost model still uses column-etree height; `tree_levels(sp.snparent)` now gives the
    supernodal level count if the owner wants the cost model to use it.
- Suggested plan changes:
  - PLAN §2.3 step 4: let `max_width` cap *merging* only and leave wide fundamental supernodes whole (they go
    to regime C anyway), or split only above a separate regime-C width; the chain split adds levels.
  - PLAN §2.3: state that the supernodal renumbering composes with the ordering (`perm[order]`).

---

## T07 — Symbolic III: schedule, static layout, device maps   `[!]`

**Reads**: PLAN §2.3 steps 5–8, §2.2 (regimes), §3.4.

#### Deliverables

* `src/symbolic/schedule.jl`: assembly-tree levels (height above leaves);
  regime assignment: regime A subtrees by greedy postorder merge while the
  subtree's peak live front+CB bytes `≤ budget` (budgets `(16, 32, 48) KiB`
  for `T`, chosen per subtree, stored as a class), regime B bins keyed by
  `(wclass ∈ {8,16,32,64}, fclass ∈ {64,128,256,512})`, regime C when
  `w > 64` or `f > 512` (both thresholds are options); per level: lists per
  bin and per class; level chunking when the level's update-stack bytes exceed
  `opts.memory_budget` (default: no limit); `nlaunches(schedule)`.
* `src/symbolic/layout.jl`: panel offsets (contiguous col-major `f×w`),
  D offsets, update-stack offsets per level with the high-water mark,
  `memory_estimates(symbolic, T) :: Vector{Int64}` of length 16 with a
  documented layout (device factor bytes, device stack bytes, device map
  bytes, host bytes, …; unused slots 0).
* `src/symbolic/maps.jl`: `amap` (for each `nzval` index: destination offset
  in its panel, or 0 if ignored by the view), `relind` per child (positions of
  the child's rows in the parent's rows), per-front descriptors (offset, f, w,
  parent, CB offset), subtree descriptors for regime A, solve gather lists;
  `Symbolic` struct holding everything; `adapt(backend, symbolic, INT)`
  moving the index arrays to the device as `INT` vectors.

#### Tests that must pass (`test/test_symbolic_schedule.jl`, host + adaptation on CUDA)

* Levels: `level(parent) > level(child)` for all edges; every supernode belongs
  to exactly one of {a subtree, a (level, bin)}, or is regime C; subtree peak
  bytes ≤ its budget.
* Reconstruction: allocate host panels, scatter `nzval` through `amap`, read
  the panels back into a sparse lower-triangular matrix, and compare `==`
  with `tril(P A Pᵀ)` for views `'L'`, `'U'`, `'F'`, both index bases, `"S"`
  and `"H"` (conjugation applied for `'U'` input of `"H"`).
* `relind`: for every child/parent pair, `parent_rows[relind] == child_rows[(w_child+1):end]`.
* Update-stack offsets of fronts in the same level do not overlap; the
  high-water mark equals the maximum over levels; `memory_estimates` sums are
  ≥ the actual allocated bytes computed from the layout.
* `adapt(CUDABackend(), symbolic, Int32)` returns `CuVector{Int32}` index
  arrays and the round trip to host is identical.
* `nlaunches` on `laplacian2d(100,100)` with AMD is reported in the Report
  with and without regime A.

### Report

- Status: [!] (done; the regime thresholds and `memory_budget` are `Options` fields but not parameter strings,
  `"G"` maps deferred to T19, see deviations)
- What was built (internal like the rest of the symbolic layer, nothing exported):
  - `src/symbolic/schedule.jl`: `Schedule`, `ScheduleGroup`, `REGIME_A/B/C`, `tree_height(parent)`,
    `build_schedule(sp, opts, T)`, `nlaunches(schedule)`, `nsubtrees`. `level` = height above the leaves of the
    supernodal tree. Regime C: `w > regime_c_width (64)` or `f > regime_c_rows (512)`, or every non-A front with
    `factorization_alg = "algo2"` (`"algo1"` sets `vendor_c = false`: C fronts use the KA tiled path). Regime A:
    a front is eligible when it is not C, all children are eligible and its subtree's serial multifrontal stack
    peak (full `f×f` front + the `m×m` blocks waiting on the stack, children visited by decreasing `peak − cb`,
    Liu's order) fits the largest budget; the maximal eligible subtrees are the regime-A subtrees, each with the
    smallest budget class (16/32/48 KiB for `T`) holding its peak. `subtree_nodes` stores that processing order.
    Regime B: bins `(wclass ∈ 8,16,32,64, fclass ∈ 64,…,512)` (extended by powers of two if the thresholds are
    raised). B/C fronts get a *schedule level* = height counting only B/C fronts (the A subtrees run first,
    step 0), split into chunks whose produced update-stack bytes stay `≤ memory_budget` (single oversized front =
    own chunk). Each (level, chunk) is a step; launch groups: A per class, then per step B per bin and one C group.
    `nlaunches` = 1 per A class + 1 per B group + per C group 1 batched assembly + per front `potrf` (+ `trsm`,
    `syrk` when it has a CB).
  - `src/symbolic/layout.jl`: `Layout`, `build_layout(sp, schedule)`. Element offsets, 1-based: panels
    contiguous col-major `f×w` in supernode order; D buffer `2n` (diagonal at `j`, 2×2 subdiagonal at `n + j`);
    contribution blocks (full `m×m`, ld `m`) on the update stack only for B/C fronts and A-subtree roots with a
    parent, alive from their step to their parent's step (inclusive), first-fit allocated over those intervals;
    `step_top` per step and `stack_len` = high-water mark.
  - `src/symbolic/maps.jl`: `Symbolic{INT,VI}` (host: `partition`, `schedule`, `layout`, structure/view/index,
    `elsize`; device maps `DEVICE_MAPS`: `perm`, `iperm`, `super_ptr`, `snparent`, solve gather lists
    `rowptr`/`rowval`, front descriptors `front_ptr`/`front_nrows`/`front_ncols`/`cb_ptr`, `child_ptr`/`child_list`
    (owner-pull order), `relind_ptr`/`relind`, `amap` (offset, `−offset` = add the conjugate, `0` = ignored by the
    view), `amap_ptr`/`amap_src` (amap grouped per supernode, sorted by destination, for deterministic owner-pull
    assembly with duplicates summed), `subtree_ptr`/`subtree_nodes`, `group_ptr`/`group_nodes`).
    `relative_indices(sp)`, `assembly_map(...)`, `Symbolic(sp, schedule, layout, rowptr, colval, n, structure;
    view, index)`, `symbolic_analysis(A::CSR, structure, view; opts, T)` (the whole host analysis),
    `Adapt.adapt(backend, S, INT)` (overflow-checked, host fields shared), `device_map_bytes`,
    `memory_estimates(S, T[, INT])` (16 slots; 1–6 follow cuDSS: permanent/peak device, permanent/peak host,
    hybrid min device, hybrid max host; 7–12: panels, D, update stack, maps, per-front stats, largest regime-A
    budget; 13–16 zero; documented in the docstring).
  - `src/options.jl`: `Options` fields `regime_c_width = 64`, `regime_c_rows = 512`,
    `subtree_budgets = [16384, 32768, 49152]` (bytes; empty disables regime A), `memory_budget = -1` (no limit),
    settable as `Options(; kw...)` keywords (validated) or by field assignment.
  - Tests: `test/test_symbolic_schedule.jl`.
- Tests: `SDS_TEST_GPU=0 julia --project=. -e 'using Pkg; Pkg.test()'` (Julia 1.13.1, ubuntu-latest, CPU):
  41716 pass / 0 fail / 0 broken (test_symbolic_schedule: 10654, 15 s). Beyond the listed checks: subtree peaks
  are recomputed by simulating the stack in `subtree_nodes` order, regime A is closed under descendants, bins
  are the tightest classes, chunk budgets hold, every B/C/A-root CB that must go through global memory has a
  stack slot with the right lifetime, child lists, the owner-pull grouping reproduces the `amap` scatter,
  complex-symmetric `"S"` as well as `"H"`, duplicated entries are summed, the reconstruction also with
  amalgamation off, tiny regime-C thresholds and a 64-byte memory budget, and `adapt` to `Int8` raises
  `InvalidValueError`. Julia 1.10 not run. CUDA/AMDGPU: pending CI on the PR (the `adapt` testset loops over
  `BACKENDS` and checks `CuVector{INT}` on CUDA).
- Measurements (ubuntu-latest; `symbolic_analysis` defaults with AMD/ND forced, `"SPD"`, `T = Float64`;
  `slev` = schedule levels, `noA` = `subtree_budgets = Int[]`, `stack/fac` = update-stack high-water mark /
  factor entries, `peak` = `memory_estimates[2]` with `Int32` maps, `t` = whole analysis incl. ordering):

  ```text
  matrix          ord   ns     A/B/C          subtrees slev launches (noA, slev) stack/fac (noA) peak MB  t
  lap2d 100²      amd   3407   3338/69/0        91      25     37    ( 89,  45)  0.62 (0.74)    4.7   13 ms
  lap2d 100²      nd    3616   3559/57/0        70      11     22    ( 72,  32)  0.62 (0.75)    4.7   36 ms
  lap2d 300²      amd  13734  13002/732/0     1203      74    117    (141,  83)  0.63 (0.68)   58.2  181 ms
  lap2d 300²      nd   20781  20241/540/0      677      32     61    (133,  48)  0.62 (0.69)   51.7  465 ms
  lap3d 12³       amd    304    268/36/0        55      15     24    ( 32,  21)  1.83 (1.44)    2.5    3 ms
  lap3d 30³       amd   4609   3932/530/147    906      91    566    (586,  96)  2.07 (2.13)  150.1   55 ms
  lap3d 30³       nd    4569   3759/693/117    953      67    457    (468,  70)  1.90 (1.84)  109.5  230 ms
  kkt 3000+1000   amd    534    341/113/80     311      87    334    (337,  89)  7.81 (8.12)  176.3   66 ms
  random_spd 2000 amd    253    164/63/26      135      45    132    (132,  47)  6.42 (5.95)   35.1    8 ms
  ```

  **`nlaunches` on `laplacian2d(100,100)` with AMD: 37 with regime A (91 subtrees in 3 classes, 34 B groups),
  89 without regime A.** On the KKT/random matrices the launch count is dominated by regime-C fronts (3 vendor
  calls each), most of them links of the `max_width` chains T06 cuts big separators into.
- Deviations from PLAN.md / this task:
  - The regime thresholds, budgets and `memory_budget` are `Options` fields (keywords of `Options(...)`) but not
    `setparam!` strings: PLAN §1.7 fixes the parameter names beyond cuDSS and `test_options` checks the tables, so
    I did not add public names. `setparam!(opts, "memory_budget", …)` raises `ArgumentError`.
  - "per level: lists per bin and per class": the regime-A subtrees are not per level (they all run first, in one
    launch per budget class), so "per class" lists are step 0. B/C fronts use the schedule level (height over
    the A subtrees), not the full-tree level, which removes the A levels from the level loop; `level` (full tree
    height) is kept and tested as asked.
  - Level chunking bounds the update-stack bytes *produced* per chunk; with everything on the device it does not
    lower the high-water mark (blocks live until their parent's step), it only splits launches. It becomes a
    memory lever with hybrid memory (M12).
  - `amap` encodes conjugation as a negative offset (one `INT` vector); conjugation depends on the permutation
    (an `'L'` entry can land in the upper triangle of `P A Pᵀ`), not only on `'U'` input. `amap_ptr`/`amap_src`
    (owner-pull grouping) were added so T09 can assemble without atomics, as PLAN §3.4 requires.
  - `"G"` (`assembly_map`) raises `NotSupportedError` until T19 adds U panels; the schedule/layout do not depend
    on the structure.
  - `factorization_alg = "algo1"` keeps large fronts in regime C with `vendor_c = false` (they cannot fit regime-B
    local memory), `"algo2"` sends every non-A front to C.
  - Contribution blocks are stored as full `m×m` squares (vendor `syrk` writes one triangle of a square), and the
    regime-A peak counts full `f×f` fronts; packing would halve both.
- Open issues / follow-ups:
  - #48: the update stack is 5–8× the factor on KKT/random matrices (chain splits of wide supernodes + full-square
    CBs); see the T06 suggestion on `max_width`.
  - `max_lu_nnz` is still not enforced (T13 owns phase error reporting; `nnz_stored` and `layout.factor_len` are
    available).
  - `memory_estimates` slot 11 (per-front stats, 6 `Int64` per supernode) and slot 3 (`Base.summarysize` of the
    host data) are estimates; T09/T13 should adjust slot 11 when the stats arrays exist. The data parameter is
    not wired to `getparam` yet (T13).
  - The regime-A test of "fits local memory" uses full `f×f` fronts; T11 must allocate `budget ÷ sizeof(T)`
    entries of `@localmem` per class and process `subtree_nodes` in order with a stack of CBs.
- Suggested plan changes:
  - PLAN §1.7 / §2.3 step 5: decide whether the regime thresholds and the memory budget become public parameter
    strings (e.g. one `"regime_params"` NamedTuple like `"amalgamation"`) or stay internal knobs.
  - PLAN §2.3 step 6: state that the B/C level loop uses levels counted above the regime-A subtrees and that
    level chunking only bounds per-chunk bytes until hybrid memory exists.

---

## T08 — CPU reference multifrontal Cholesky (the oracle)   `[!]`

**Reads**: PLAN §2.4, §3.9, §7 (oracle).

#### Deliverables

* `src/reference/cholesky.jl` (plain Julia, host arrays, BLAS allowed):
  `Numeric` allocation from the layout (`allocate_numeric(symbolic, T, backend)`
  — generic over backend, used by the GPU path later); `ref_factorize!(numeric,
  symbolic, nzval) -> info` processing supernodes in `snpost` with assembly
  through `amap`, extend-add through `relind`, `potrf`/`trsm`/`syrk` per front;
  `ref_solve!(X, symbolic, numeric, B)` forward/backward with permutation and
  multiple RHS; `extract_L(symbolic, numeric) :: SparseMatrixCSC` (permuted
  factor); `info` = first non-positive pivot in original numbering.

#### Tests that must pass (`test/test_reference_cholesky.jl`, host, `T ∈ ELTYPES`)

* `P A Pᵀ ≈ L Lᵀ` (relative Frobenius `≤ 1e-12` for Float64, `1e-5` for
  Float32) on `laplacian2d(40,40)`, `random_spd(500, 0.01)`, `random_hpd`.
* `relres ≤ tol(T)` for `nrhs ∈ {1, 5}`; solution matches CHOLMOD's within
  `1e-8` relative (Float64).
* Views `'L'`, `'U'`, `'F'` give identical factors; refactorization with new
  values on the same `Symbolic` gives the new correct factor.
* `info` equals the known column for `singular_block_matrix` made non-SPD
  (negate one diagonal entry) and `0` otherwise.
* Amalgamation on and off give the same solution.

### Report

- Status: [!] (done; minor deviations: `Numeric` type parameters and location, host contribution blocks)
- What was built (`src/reference/cholesky.jl`, internal like the symbolic layer, nothing exported):
  - `Numeric{T, VT, VS}`: `factor` (panels, `layout.factor_len`), `d` (`2n`, unused by Cholesky), `stack`
    (`layout.stack_len`), `stats` (`FRONT_STATS_FIELDS × ns` `Int64`: `npos, nneg, nzero, nperturbed, n2x2, info`
    per front; Cholesky fills `npos` and the local failed column).
  - `allocate_numeric(symbolic, T, backend = CPU())`: zero-filled `KernelAbstractions.zeros` buffers on any backend.
  - `ref_factorize!(numeric, symbolic, nzval | CSR) -> info`: multifrontal LLᵀ/LLᴴ over `snpost`, one dense `f×f`
    host front per supernode; assembly through `amap_ptr`/`amap_src`/`amap` (negative offset = conjugate),
    extend-add of the children in `child_list` order through `relind`, then `LAPACK.potrf!`, `BLAS.trsm!`,
    `BLAS.syrk!`/`herk!`; the first `w` columns go to the panel. `info` = original column (`perm[k]`) of the first
    failed pivot; the factorization stops there. Structure must be `"SPD"` (real) or `"HPD"`, else
    `InvalidValueError`; non-BLAS `T` → `NotSupportedError`.
  - `ref_solve!(X, symbolic, numeric, B)`: permute, supernodal forward (`trsm` + `gemm` scatter) and backward
    (gather + `gemm` + `trsm`) sweeps, inverse permutation; vectors or `n × nrhs`, `X === B` allowed.
  - `extract_L(symbolic, numeric)`: `SparseMatrixCSC` of `L` with `P A Pᵀ = L Lᴴ`, `P = symbolic.partition.perm`
    (supernodal numbering), every stored panel entry (`nnz == nnz_stored`, amalgamation zeros kept); a device
    `factor` is copied to the host once (for T09).
  - Tests: `test/test_reference_cholesky.jl`.
- Tests: `SDS_TEST_GPU=0 julia --project=. -e 'using Pkg; Pkg.test()'` (Julia 1.13.1, ubuntu-latest, CPU):
  42101 pass / 0 fail / 0 broken (test_reference_cholesky: 385, 22 s). Covered: allocation sizes; `‖PAPᵀ − LLᴴ‖_F/‖A‖_F`
  ≤ 1e-12 / 1e-5 on `laplacian2d(40,40)`, `random_spd(500,0.01)`, `random_hpd(400,0.02)` (complex); `relres ≤ tol(T)`
  for `nrhs ∈ {1,5}`, Float64/ComplexF64 within `1e-8` of CHOLMOD, in-place solve; views `'L'`/`'U'`/`'F'` × index
  `'O'`/`'Z'` give `==` factors; refactorization with new values (and back to the old ones, `==`); `info == j` for
  `singular_block_matrix` with `A[j,j] = -3` and `= 0`, `0` with `+3`, for default, natural and no-amalgamation
  analyses, plus a pivot that turns negative only after elimination; amalgamation on/off solutions agree within
  `tol(T)`; schedules with other regime mixes give `==` factors. Julia 1.10 not run. CUDA/AMDGPU: pending CI on the
  PR (the new tests are host-only).
- Measurements (ubuntu-latest, Float64, default analysis, warm, best of 3): lap2d 100² (3407 supernodes,
  2.6e5 stored) factor 6.1 ms / solve 1.9 ms (CHOLMOD refactor 5.4 ms); lap3d 20³ (1380, 9.6e5) 89 ms / 1.8 ms
  (CHOLMOD 19 ms); random_spd(2000, 0.002) (254, 5.7e5) 61 ms / 0.7 ms (CHOLMOD 12 ms). The oracle allocates a dense
  front per supernode and is not meant to be fast.
- Deviations from PLAN.md / this task:
  - `Numeric` has a third type parameter `VS` for the `Int64` statistics vector (PLAN §3.2 lists `Numeric{T, VT}`).
  - `Numeric` lives in `src/reference/cholesky.jl` as the task says (PLAN §4 puts it in `types.jl`); it can move
    when `src/numeric/` exists (T09).
  - The reference keeps contribution blocks in host matrices, not on `numeric.stack`: regime-A blocks have no stack
    slot in the layout, and the oracle should not depend on the schedule. It ignores regimes entirely.
  - `info` is the first failed pivot in factor-column order (supernodal numbering, = postorder processing order),
    reported as the original column; later panels stay zero. Since columns increase along `snpost`, this equals the
    minimum failed factor column, which is what a device reduction (T09) can compute.
- Open issues / follow-ups:
  - T09 needs a device-side `info` reduction matching "smallest failed factor column"; fronts above a failed front
    are computed from garbage there, but their columns are larger, so the minimum is unaffected.
  - `stats` layout (`FRONT_STATS_FIELDS` per front, column-major by front) is fixed here; T14 fills the LDLᵀ fields.
- Suggested plan changes: none.

---

## T09 — GPU multifrontal Cholesky with assembly kernels (regime C on every front)   `[!]`

**Reads**: PLAN §2.4, §3.4, §2.7.

#### Deliverables

* `src/numeric/assembly.jl`: KA kernels `zero_fronts!`, `scatter_A!` (one
  work-item per nonzero, `amap`), `extend_add!` (owner-pull: one workgroup
  per parent front, loops over its children's CB entries through `relind`,
  fixed child order, no atomics).
* `src/numeric/factorize.jl`: level driver: for each level: zero, scatter,
  extend-add, then for each front `potrf!`, `trsm!`, `syrk!` through the dense
  interface (every front treated as regime C in this task); `info` collected
  on the device and reduced once per phase; no allocation after
  `allocate_numeric`; no host synchronization inside the phase except the
  final `info` read.
* `src/numeric/extract.jl`: copy panels to host and reuse `extract_L`.

#### Tests that must pass (`test/test_numeric_cholesky_c.jl`, CPU + CUDA, `T ∈ ELTYPES`)

* Panels equal the T08 reference panels elementwise within `100·eps(real(T))·max|L|`
  on `laplacian2d(40,40)`, `random_spd(500, 0.01)`, `laplacian3d(10,10,10)`.
* Using `ref_solve!` on the copied-back factor: `relres ≤ tol(T)`.
* `info` matches the reference for the non-SPD case.
* Determinism on CUDA: two factorizations of the same values give bitwise
  identical panels (`==`).
* Refactorization with new values is correct; no allocations during
  `factorize!` on CPU (`@allocated` on the second call ≤ a small constant).

### Report

- Status: [!] (done on the CPU backend; the driver needs an analysis without regime-A subtrees, small dense-layer
  additions, `Numeric` moved and extended; CUDA from CI)
- What was built (internal, nothing exported):
  - `src/numeric/storage.jl` (moved from `src/reference/cholesky.jl`): `Numeric{T, VT, VS, VI}` gains `info`
    (`ns + 1` device `Int32`: the `potrf` status of front `s` at `s`, the reduced result at `ns + 1`) and `plan`;
    `NumericPlan` (host, built once by `allocate_numeric`: per step the contiguous range of `group_nodes`, the
    largest child count = trip count of the extend-add, a 1-entry host staging buffer for the `info` read).
  - `src/numeric/assembly.jl`: KA kernels, one 1-D workgroup (256) per front of a step, owner-pull, no atomics:
    `zero_fronts!` (panel + update-stack CB), `scatter_A!` (work items stride over the front's `amap_src`
    entries; the item holding the first entry of a run of duplicates sums the run in `nzval` order, so the
    summation order equals the reference's), `extend_add!` (children in `child_list` order, one barrier per
    child, lower triangle of each CB through `relind` into the parent's panel or CB).
  - `src/numeric/factorize.jl`: `factorize!(numeric, symbolic, nzval | CSR; impl = :auto) -> info`: per step of
    the schedule one launch each of zero/scatter/extend-add, then per front `potrf_info!`, `trsm!('R','L','C')`
    and `herk!`/`syrk!` (into the CB on the update stack) through the dense interface with `impl` passed on;
    `cholesky_stats!` (one workgroup, `@localmem` tree reduction) fills `stats` like `ref_factorize!` and reduces
    the smallest failed factor column; one `copyto!` of that `Int32` is the only host synchronization; `info` is
    returned in original numbering (`perm[k]`), as the reference.
  - `src/numeric/extract.jl`: `host_numeric(numeric)` (host copy usable by `ref_solve!`/`extract_L`),
    `panel(symbolic, numeric, s)`; `extract_L` already copies a device factor.
  - Dense layer: `potrf_info!(uplo, A, info, idx; impl)` (asynchronous potrf, LAPACK info into `info[idx]` of a
    device `Int32` vector; vendor/generic results validated on the device by the new `ka_chol_check_info!`, as
    `potrf!` does since T03); `vendor_potrf_info!` (host LAPACK; CUDA: raw `cusolverDn?potrf` with `devInfo =
    pointer(info, idx)` and the handle's cached workspace, no host read); `ka_potrf!(...; offset)`; the
    `vendor_potrf` capability probe now also checks `vendor_potrf_info!`; `HostMatrix` accepts index views of
    panels (`view(reshape(view(buf, r), f, w), i, j)`, the `F11`/`F21` blocks); `select_impl` no longer builds the
    `dense_impls` vector (it allocated 176 bytes per dense call); internal `_potrf_info_impl!`, `_trsm_impl!`,
    `_herk_impl!`/`_syrk_impl!` take an already resolved impl, and `factorize!` resolves the three impls once per
    call (`_front_impls`), so the per-front calls do no capability lookup (review round 1).
  - `memory_estimates` slot 11 includes the `Int32` status vector.
  - Tests: `test/test_numeric_cholesky_c.jl`, a `potrf_info!` testset in `test/test_dense.jl`, `panel_tol(T)` in
    `test/utils.jl`.
- Tests: `SDS_TEST_GPU=0 julia --project=. -e 'using Pkg; Pkg.test()'` (Julia 1.13.1, ubuntu-latest, CPU):
  43099 pass / 0 fail / 0 broken (test_numeric_cholesky_c: 458, 29 s; test_dense +540 for `potrf_info!`).
  Review round 1, under CI's flags (`Pkg.test(; coverage = true, julia_args = ["--check-bounds=yes"])`,
  `SDS_TEST_ONLY=test_numeric_cholesky_c`): Julia 1.10.10 458 pass / 0 fail, Julia 1.12.1 458 pass / 0 fail.
  The first CI run on this PR failed the allocation assertion (4 per job: 2176 B on Julia 1, 9696 B on 1.10);
  see the measurements below.
  Covered, per backend and `T ∈ ELTYPES`: panels within `100·eps·max|L|` of the T08 reference on
  `laplacian2d(40,40)` (Int32 and Int64 maps), `random_spd(500,0.01)`, `laplacian3d(10,10,10)`, equal `stats`,
  `extract_L`, `relres ≤ tol(T)` with `ref_solve!` on the copied-back factor (`nrhs` 1 and 5); two factorizations
  give `==` panels (all backends; the first result is copied, since `host_numeric` returns a host `Numeric` as is); every `impl` in `dense_impls(:potrf, …)` (CPU: vendor, generic,
  ka, also against a copy); views `'U'`/`'F'` × index `'O'`/`'Z'` give `==` panels; refactorization with new
  values and back (`==` against a copy of the first factor);
  `info == j` (and the front's local `info` in `stats`) for `singular_block_matrix` with `A[j,j] ∈ {-3, 0}` under
  three analyses, plus a pivot that turns negative only after elimination; regime-A analyses raise
  `NotSupportedError`; the plan covers every front once.
  CUDA/AMDGPU: pending CI on the PR.
- Measurements (ubuntu-latest CPU backend, Float64, `subtree_budgets = Int[]`, best of 3; `:auto` = host LAPACK):

  ```text
  matrix            fronts steps  ref_factorize!  factorize!(:auto)  factorize!(:ka)  @allocated(:auto)
  lap2d 100²         3407    45      5.2 ms          14.5 ms            67 ms            64 B
  lap3d 20³          1380    47     48.2 ms          63.6 ms           379 ms            64 B
  random_spd 2000     254    42     43.4 ms          68.5 ms           361 ms            64 B
  ```

  With `:auto` the CPU panels are bitwise equal to the reference (same LAPACK calls on the same data).
  `@allocated factorize!(:auto)` of the allocation test (`random_spd(300, 0.02)`, 21–26 fronts, 10–12 steps,
  over the four `T`), review round 1:

  ```text
  Julia    plain             --code-coverage --check-bounds=yes (CI)
  1.13.1   64 B              not run
  1.12.1   64 B              2176 B (constant: 2112 B = the stats kernel's @localmem MArray on the heap)
  1.10.10  9632–11488 B      11744–13600 B (80–288 B per KA CPU launch: args boxed behind KA's `__run`)
  ```

  Resolving the impls once per phase did not change these numbers (the capability lookup did not allocate
  after compilation); the allocations are inside the KA 0.9 CPU backend. The test asserts
  `≤ ka_cpu_alloc_budget(launches, localmem)` (`test/utils.jl`): 1024 B, plus 320 B per launch on Julia < 1.12
  and, under coverage, the `@localmem` bytes plus 64 B per launch; the launch count is bounded by
  `3·nsteps + nfronts + 1`. So the strict 1024 B holds on Julia ≥ 1.12 without coverage. `:generic`
  (LinearAlgebra wrappers, `cholesky!` objects) and `:ka` (fallback launches) allocate 9–39 KB per
  factorization of that matrix on Julia 1.12/1.13 (1.4–4.7 MB for `:ka` on 1.10 under coverage).
- Deviations from PLAN.md / this task:
  - Regime-A subtrees have no update-stack slots in the T07 layout (their CBs are meant to stay in local memory),
    so the regime-C driver cannot run them: `factorize!` raises `NotSupportedError` unless the analysis has no
    subtrees (`Options(subtree_budgets = Int[])`, which the tests use). Regime-B fronts go through the regime-C
    path, step by step, as asked. T11 must replace the error by the subtree launch (step 0).
  - The "level" of the driver is the schedule step (T07: level × chunk over B/C fronts), so the update-stack
    offsets of the layout are used as they are.
  - `Numeric` moved to `src/numeric/storage.jl`, with a fourth type parameter (`VI`, the `Int32` status vector) and
    a host `plan` field. PLAN §3.2 lists `Numeric{T, VT}`.
  - The dense interface got `potrf_info!` / `vendor_potrf_info!` because `potrf!` returns `info` to the host, and
    `cuSOLVER.potrf!` reads it, so each would synchronize once per front. `:generic` still synchronizes on CUDA
    inside `cholesky!` (documented); `:auto` picks `:vendor` there.
  - The extend-add loops over the full `m×m` square of each child CB and skips the upper half (simple index
    math); packing CBs (T07 note) would remove that.
- Open issues / follow-ups:
  - The CUDA `vendor_potrf_info!` binding (raw `cusolverDn?potrf` + `CUDACore.with_workspace`) could only be
    loaded, not run, here; the CI `cuda` job checks it through the capability probe (`caps.vendor_potrf`), the
    `potrf_info!` testset and the T09 tests.
  - Per-front dense calls dominate on CPU (14.5 ms vs 5.2 ms for the reference on lap2d 100²: ~3400 fronts × 3
    calls + 135 launches). On a GPU this is ~10k small vendor launches per factorization; T10/T11 fuse them.
  - `:ka` and `:generic` allocate per front on CPU, and the KA CPU backend allocates per launch on Julia 1.10
    and for `@localmem` under coverage (see measurements): issue #53 (`found-by-agent`).
- Suggested plan changes:
  - PLAN §2.6: add the asynchronous `potrf_info!` (device status, no host read) to the dense interface list;
    the same will be needed for `getrf`/`sytrf` (T15/T19).
  - PLAN §3.2: `Numeric{T, VT, VS, VI}` with the `Int32` status vector and the host launch plan.

---

## T10 — Regime B: fused per-front kernels, level-batched   `[ ]`

**Reads**: PLAN §2.2 (regime B), §2.4, §2.7.

#### Deliverables

* `src/numeric/front.jl`: `front_cholesky_kernel!` parametrized by
  `Val(W)` (W ∈ 8,16,32,64): one workgroup per front of a (level, bin) list;
  F11 in `@localmem`, unblocked Cholesky; F21 streamed in row tiles for the
  TRSM; SYRK update into the front's own CB (no conflicts); info flag per
  front. Complex support (conjugation). Driver uses regime B for fronts
  below the regime C thresholds, regime C above.
* `bench/front_bins.jl`: synthetic batches per bin, regime B vs regime C
  (vendor) timings on CUDA.

#### Tests that must pass (`test/test_numeric_cholesky_b.jl`, CPU + CUDA, `T ∈ ELTYPES`)

* Same correctness suite as T09 with regime B active (panels vs reference,
  residuals, `info`, determinism, refactorization).
* Forcing `factorization_alg = "algo2"` (regime C everywhere) and `"algo1"`
  (regime B for everything below the C thresholds) give the same solution
  within tolerance.
* Report: per-bin timing table from `bench/front_bins.jl` on the RTX 4080 and
  the chosen crossover.

---

## T11 — Regime A: fused subtree-per-workgroup kernels   `[ ]`

**Reads**: PLAN §2.2 (regime A), §2.3 step 5, §2.7.

#### Deliverables

* `src/numeric/subtree.jl`: `subtree_cholesky_kernel!` parametrized by the
  local-memory budget class: one workgroup per subtree; postorder loop over
  the subtree's supernodes; fronts and CBs in `@localmem`; assembly from `A`
  through `amap`, extend-add in local memory, finished panels written to
  global, the subtree root's CB written to the global update stack. Driver:
  level 0 = all subtrees, one launch per budget class.

#### Tests that must pass (`test/test_numeric_cholesky_a.jl`, CPU + CUDA, `T ∈ ELTYPES`)

* Same correctness suite as T09/T10 with regimes A+B+C active.
* `nlaunches` with regime A on `laplacian2d(100,100)` (AMD) is at least 3×
  smaller than with regime A disabled; both give the same solution.
* Report: factorization time on the T04 generated matrices (CUDA) with
  A+B+C vs B+C vs C only.

---

## T12 — GPU solve sweeps, multiple right-hand sides, permutations   `[ ]`

**Reads**: PLAN §2.5, §3.4.

**Owner note (issue #36)**: `Atomix.@atomic` has no complex element types on
any backend (`capabilities(backend, T).atomic_add` is `false` for `ComplexF32`
and `ComplexF64`), so the default forward sweep below cannot accumulate complex
values directly. Decide here, and record the decision in the Report: either
accumulate into a real view of the RHS with two real atomics per entry (keeps
one code path; on the CPU backend pass a plain real array, not a
`ReinterpretArray`, into the kernel), or select the atomic-free variant
whenever `atomic_add` is `false`. Close #36 in this task's PR.

#### Deliverables

* `src/solve/permute.jl`: `permute_rhs!`, `unpermute_solution!` (strided and
  matrix RHS, `transposed` layout).
* `src/solve/sweeps.jl`: forward sweep (regime A subtrees in one workgroup;
  regimes B/C level-batched TRSV/GEMV with `Atomix.@atomic` accumulation;
  deterministic variant using per-front RHS buffers on the update-stack layout
  when `deterministic_mode = 1` or the backend lacks float atomics), diagonal
  hook (identity for Cholesky), backward sweep (gather-based, conflict-free);
  `nrhs > 1` via a second ndrange dimension; vendor `trsm`/`gemm` for regime C
  fronts.

#### Tests that must pass (`test/test_solve.jl`, CPU + CUDA, `T ∈ ELTYPES`)

* `relres ≤ tol(T)` for `nrhs ∈ {1, 2, 5}` on the T09 matrices with all
  regimes active.
* Deterministic and atomic variants agree within `10·eps(real(T))·‖x‖`; the
  deterministic variant is bitwise reproducible on CUDA.
* Transposed (row-major) RHS layout gives the same solution.
* Solving with the `(n, nrhs)` matrix equals solving each column separately.

---

## T13 — Public API v0.1: `DirectSolver`, phases, generic Cholesky, ported tests   `[ ]`

**Reads**: PLAN §1.2, §1.3, §1.4, §1.5, §3.1, §3.2; `../CUDSS.jl/src/interfaces.jl`,
`generic.jl`, `test/test_cudss.jl`.

#### Deliverables

* `src/solver.jl`: `DirectSolver(A::CSR, structure::String, view::Char; index='O')`,
  constructors from raw arrays and (CUDA ext) from `CuSparseMatrixCSR`/`CSC`;
  `execute!(phase::String, solver, X, B; asynchronous=true)` for all phase
  strings of PLAN §1.2 that exist at this point (`"reordering"`,
  `"symbolic_factorization"`, `"analysis"`, `"factorization"`,
  `"refactorization"`, `"solve"`, `"solve_fwd_perm"`, `"solve_fwd"`,
  `"solve_bwd"`, `"solve_bwd_perm"`; others raise `NotSupportedError` until
  their task); `analyze!`, `factorize!`, `refactorize!`, `solve!`;
  `update!(solver, A)`/`update!(solver, rowptr, colval, nzval)`;
  `setparam!`/`getparam`/`getparam!` on the solver for the data parameters
  `info` (get/set), `lu_nnz`, `flops`, `nsuperpanels`, `memory_estimates`,
  `perm_reorder_row/col`, `perm_row/col`, `diag`, `user_perm`; phase-order
  checks (`solve` before `factorization` throws `FactorizationError`);
  `asynchronous=false` → `KernelAbstractions.synchronize(backend)`;
  `fresh_factorization` flag; `Base.show`.
* `src/generic.jl`: `cholesky`, `cholesky!`, `ldiv!` (both forms), `\`,
  `Hermitian`/`Symmetric` wrappers, `logabsdet`, `diag`, `nnz`; the
  LinearAlgebra layer defaults `ir_n_steps = 2` once T16 exists (for now the
  option is stored and ignored with a documented note).
* `test/ported/`: the SPD/HPD parts of `cudss_execution`, `cudss_generic`,
  `small_matrices`, `refactorization_cholesky`, `cudss_solver` (parameter
  get/set loop restricted to implemented parameters) from
  `../CUDSS.jl/test/test_cudss.jl`, with `CudssSolver → DirectSolver`,
  `cudss(…) → execute!(…)`, `cudss_set/get → setparam!/getparam`.

#### Tests that must pass (`test/test_api.jl`, `test/ported/*.jl`, CPU + CUDA)

* All ported tests pass for `T ∈ ELTYPES` and `INT ∈ INTTYPES`.
* MadNLP-like loop: 20 refactorizations of `kkt_matrix`'s `H` block with a
  changing diagonal, each with `info == 0` and `relres ≤ tol(T)`, with no
  allocation on the device after the first iteration (compare `CUDA.memory_status`
  or `@allocated` on CPU).
* Phase-order errors are raised; unknown phase strings raise `ArgumentError`.
* `getparam(solver, "perm_reorder_row")` is a valid permutation and
  `getparam(solver, "lu_nnz") == nnz_L`.

---

## T14 — CPU reference LDLᵀ/LDLᴴ: in-front Bunch–Kaufman, perturbation, sign policy, inertia   `[ ]`

**Reads**: PLAN §3.3, §1.7 (`pivot_sign`, `pivot_stats`); RESEARCH section 4 (pivoting, inertia).

#### Deliverables

* `src/reference/ldlt.jl`: Bunch–Kaufman (`α = (1+√17)/8`) restricted to the
  fully-summed block of each front, `pivot_threshold` acceptance (default 0.01
  of the row max), `pivot_type` `'B'` (default for `S`/`H`), `'D'` (diagonal
  only), `'N'` (no search); if no acceptable pivot exists in the block:
  perturbation `d ← sign · ε` with `ε` from `pivot_epsilon`/`pivot_epsilon_alg`
  and `sign` from `pivot_sign[row]` when given, else `sign(d)` (or `+1` if
  `d == 0`); storage: unit-lower panels, `D` diagonal, `D2` off-diagonal of
  2×2 blocks, `pivot_kind::Vector{Int8}`; `ref_solve!` with the diagonal step;
  `inertia(numeric) -> (npos, nneg)`, `pivot_stats -> (npos, nneg, nzero,
  nperturbed, n2x2)`, `npivots`.

#### Tests that must pass (`test/test_reference_ldlt.jl`, host, `T ∈ ELTYPES`, `"H"` for complex)

* `P A Pᵀ ≈ L D Lᵀ` (relative `≤ 1e-10` Float64) on `random_symindef(400, 0.01)`,
  `kkt_matrix(300, 100, δ)` for `δ ∈ {1e-8, 1e-2}`, complex Hermitian analogues.
* `inertia` equals the eigenvalue sign counts for `n ≤ 300` matrices with no
  perturbation (`nperturbed == 0`).
* `singular_block_matrix`: `nperturbed ≥ 1`, `npivots ≥ 1`, the perturbed `D`
  entry has the sign requested through `pivot_sign` (test both signs), and
  `relres` on a consistent RHS stays `≤ 1e-6` after the perturbation.
* `pivot_type = 'D'` and `'N'` produce correct factors on quasi-definite
  `kkt_matrix` (no 2×2 needed) and `n2x2 == 0`.
* MadNLP-style inertia correction: starting from `δ = 0` on an indefinite
  `H`, increase a primal regularization until `inertia == (nh, nj)`; the
  loop terminates and the final factorization has `nperturbed == 0`.

---

## T15 — GPU LDLᵀ/LDLᴴ in regimes A/B/C, diagonal solve, statistics   `[ ]`

**Reads**: PLAN §2.4, §3.3, §2.6 (sytrf), §1.4.

#### Deliverables

* Regime A/B kernels with in-block Bunch–Kaufman, perturbation and sign
  policy (local memory), `W = F21·D` handling in the update; regime C for
  `"S"` via `cuSOLVER.sytrf!` (vendor) with a device post-check of `D` against
  `ε` and a KA redo of failing blocks, `"H"` root fronts via the KA kernel;
  `solve_diag` sweep; per-front stats arrays and a reduction kernel;
  `getparam` for `inertia`, `npivots`, `pivot_stats`, `diag`; `setparam!`
  for `pivot_sign`; generic `ldlt`, `ldlt!`; `execute!("solve_diag", …)`.
* `test/ported/`: the "Symmetric -- Hermitian" parts of `cudss_execution`
  and `cudss_generic` (ldlt) from CUDSS.jl.

#### Tests that must pass (`test/test_numeric_ldlt.jl`, ported tests, CPU + CUDA)

* `D` and `pivot_kind` equal the T14 reference (same algorithm, same pivot
  sequence) on the T14 matrices; residuals `≤ tol(T)`; inertia exact.
* Perturbation and `pivot_sign` behave as in T14 on the GPU path, including a
  case where the perturbed pivot sits in a regime C root front (force small
  thresholds so the root front is regime C).
* Determinism on CUDA (bitwise `D` and panels).
* MadNLP-style loop from T14 on the GPU; the ported CUDSS.jl symmetric tests
  pass for `T ∈ ELTYPES`.

---

## T16 — Iterative refinement, solve sub-phases, `solve_mode`, interrupt, logging   `[ ]`

**Reads**: PLAN §1.2, §1.3 (`ir_*`, `solve_mode`), §1.4 (`user_host_interrupt`, `ir_n_steps` data), §2.5.

#### Deliverables

* `src/solve/refinement.jl`: KA CSR SpMV on the full-pattern map (symmetric
  expansion, conjugation for `"H"`), residual norm reduction on the device,
  IR loop honoring `ir_n_steps` and `ir_tol` (relative residual), data
  parameter `ir_n_steps` = steps performed; `execute!("solve_refinement", …)`
  and `"solve"` including refinement; the LinearAlgebra layer now defaults to
  `ir_n_steps = 2`.
* `solve_mode` (0/1/2) for `"S"`/`"H"`/`"SPD"`/`"HPD"` (transpose is a no-op
  for symmetric, adjoint conjugates the RHS/solution for `"S"` complex and is
  a no-op for `"H"`).
* `user_host_interrupt`: `Threads.Atomic{Bool}` polled between levels; raises
  `InterruptedError` and leaves the solver in a state where `factorize!` can
  be called again. Logging via `@debug`/`@info` with `SDS_LOG_LEVEL`.

#### Tests that must pass (`test/test_refinement.jl`, CPU + CUDA)

* On a badly row-scaled SPD matrix (rows scaled by `10^(±8)`), one refinement
  step reduces `relres` by at least 100× in Float64; `ir_tol = 1e-14` with
  `ir_n_steps = 10` exits early and `getparam(solver, "ir_n_steps")` reports
  the steps performed.
* Composing `solve_fwd_perm, solve_fwd, solve_diag, solve_bwd, solve_bwd_perm,
  solve_refinement` equals `"solve"` exactly.
* `solve_mode = 2` on a complex `"S"` matrix solves `Aᴴ x = b`.
* Setting the interrupt flag before `factorize!` raises `InterruptedError`;
  clearing it and calling `factorize!` again succeeds.

---

## T17 — Uniform batch (v1 cut line)   `[ ]`

**Reads**: PLAN §1.6, §3.5, §1.3 (`ubatch_*`), §1.4 (`ubatch_mask`);
`../CUDSS.jl/docs/src/uniform_batch.md`, `test/test_uniform_batch_cudss.jl`.

#### Deliverables

* Batch stride in `Numeric` (panels, `D`, update stack, stats) and in the
  RHS handling (`(n,)` strided, `(n, nb)`, `(n, nrhs, nb)`, and
  `MatrixDescriptor(T, n; nbatch)`); all kernels take the batch index as an
  extra ndrange dimension; regime C uses `gemm_strided_batched!`,
  `trsm_strided_batched!`, `potrfBatched!` (pointer arrays built once at
  analysis) with the KA strided fallbacks; `ubatch_size`, `ubatch_index`,
  `ubatch_mask`; per-member `info`, `inertia`, `pivot_stats`; generic
  auto-detection `length(nzVal) ÷ length(colVal) > 1` in `cholesky`/`ldlt`.
* `test/ported/`: `test_uniform_batch_cudss.jl` (LU parts skipped).

#### Tests that must pass (`test/test_ubatch.jl`, ported tests, CPU + CUDA)

* Uniform batches of `nb ∈ {1, 2, 3, 16, 64}` SPD and `"S"` systems with
  `nrhs ∈ {1, 2, 4}`: every member's `relres ≤ tol(T)` (this is the regression
  test for the cuDSS multi-RHS batch bug: all columns correct at `nb ≥ 16`).
* `ubatch_index = k` refactorizes/solves only member `k`; other members'
  factors are bitwise unchanged. `ubatch_mask` likewise for a subset.
* Strided `(n·nb,)` and 3-D `(n, nrhs, nb)` layouts give the same solution.
* One non-SPD member yields `info[k] == column` and `info == 0` elsewhere.
* The ported CUDSS.jl uniform-batch tests pass.

**This completes v1 (PLAN M0–M6).** The owner re-evaluates the plan before the
tasks below are refined.

---

## Post-v1 tasks (to be refined after the v1 evaluation)

Each keeps the same structure; the test criteria are the minimum.

### T18 — FGMRES-IR extension (Krylov.jl)   `[ ]`

`ext/SparseDirectSolverKrylovExt.jl`: `ir_mode = "fgmres"` runs FGMRES with
the factorization as right preconditioner; test: on an ill-conditioned
Float64 system where plain IR stalls (`relres` not improving over 5 steps),
FGMRES-IR reaches `relres ≤ 1e-12` within 20 iterations on CPU and CUDA.

### T19 — General LU (`"G"`): CPU reference + GPU   `[ ]`

Symmetric-pattern multifrontal LU with `U` panels, in-block threshold partial
pivoting (A/B) and `getrf!` + `laswp!` + post-check (C), `perm_row/col`,
`solve_mode` 1/2, `lu`/`lu!`; ported CUDSS.jl unsymmetric tests and the
uniform-batch LU tests pass; `random_general` residuals `≤ tol(T)`.

### T20 — Schur complement mode   `[ ]`

Constrained ordering, unfactorized root front, `schur_shape`, dense and CSR
`schur_matrix`, `solve_fwd_schur`/`solve_bwd_schur`; the three 5×5 examples of
`../CUDSS.jl/docs/src/schur_complement.md` pass on CPU and CUDA (`test_schur_cudss.jl` ported).

### T21 — Matching and scaling   `[ ]`

Host MC64-style matching (job 5 first), `perm_matching`, `scale_row/col`,
composition with the ordering; test: on a badly scaled unsymmetric matrix the
LU with matching has `nperturbed == 0` where the LU without has `> 0`;
inertia with matching enabled equals the eigenvalue count (the cuDSS defect).

### T22 — Non-uniform batch (`BatchedDirectSolver`)   `[ ]`

Block-diagonal packing, forest schedule; `test_nonuniform_batch_cudss.jl` ported.

### T23 — AMDGPU, oneAPI and Metal extensions   `[ ]`

Written by analogy with the CUDA extension (sparse adapters, vendor dense
bindings, capability probes). Test on this machine: a temporary environment
that adds AMDGPU and oneAPI and precompiles the extensions (no hardware
needed to load them on Linux); Metal only by review. Mark both as untested.

### T24 — ND partition-tree export/import and ordering cache   `[ ]`

`nd_partition_tree`/`user_nd_partition_tree` in the cuDSS binary-tree
encoding; test: importing the exported tree reproduces the same supernode
partition and `nnz_L`.

### T25 — Performance pass   `[ ]`

Partitioned-inverse solve (`solve_alg = "algo1"`), CUDA sync-free forward
sweep behind a capability check, CUDA graph capture of refactorize+solve,
level merging, amalgamation/bin tuning; tests: all previous suites unchanged;
Report: timing table vs the T04 cuDSS baseline for every harness matrix.

### T26 — Hybrid memory / hybrid execute   `[ ]`

Host-resident panels streamed per level with pinned memory;
`hybrid_device_memory_min`; CPU-backend execution of regime A/B levels;
test: a factorization under a device memory budget smaller than the factor
size completes with the same solution.

### T27 — Robustness extras   `[ ]`

A posteriori threshold pivoting with optional delayed pivots (host
re-analysis), `factor_precision = Float32` with Float64 refinement; tests:
delayed-pivot case where static perturbation gives `relres > 1e-6` and APTP
gives `≤ 1e-10`; mixed precision reaches Float64 accuracy with FGMRES-IR.

### External — MadNLPGPU integration (in the MadNLP repository)   `[ ]`

Add a `SparseDirectSolver`-backed `AbstractLinearSolver` next to
`CUDSSSolver` (after T15; batch support after T17); test: a pglib case solves
with the same iteration count ±2 as with cuDSS and the same final objective
to `1e-6`.
