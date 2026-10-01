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
    only, never reads device memory); `CSR(rowptr, colval, nzval[, nrows, ncols]; index, transposed)`
    (zero-copy, `index` as `'O'`/`'Z'` or `IndexBase`), `CSR(A::SparseMatrixCSC; index)` (host
    conversion), `csr_of_transpose(A)` (zero-copy, `transposed = true`), `to_backend(A, backend)` for
    `SparseMatrixCSC` and `CSR` (generic through `KernelAbstractions.allocate` + `copyto!`, no copy if
    already on `backend`), `SparseMatrixCSC(A::CSR[, k])` (host copy of batch member `k`, rebased to 1),
    `size`, `eltype`, `nnz`, `nbatch`, `get_backend`, `Adapt.adapt_structure`, `show`.
  - `MatrixDescriptor{T,A}` (mutable; `data::Union{Nothing,A}`, `nrows, ncols, nbatch, transposed`),
    `MatrixDescriptor(T, n; nbatch)`, `MatrixDescriptor(T, m, n; nbatch, transposed)`,
    `MatrixDescriptor(x; transposed)`, `update!(desc, x)` (re-points, no copy; checks eltype, array
    type, length and shape, `InvalidValueError` otherwise), `size`, `nbatch`, `get_backend`.
  - `ext/SparseDirectSolverCUDAExt.jl`: `CSR(::CuSparseMatrixCSR)` and `CSR(::CuSparseMatrixCSC)`
    (shared arrays, CSC → `transposed = true`), `CuSparseMatrixCSR(::CSR)` (shares one-based arrays,
    rebases zero-based ones into a copy, rejects batches), `to_backend(::SparseMatrixCSC, ::CUDABackend)`.
  - Exports: `CSR, csr_of_transpose, to_backend, nbatch, MatrixDescriptor, update!`.
  - `test/test_matrix.jl`.
- Tests: `SDS_TEST_GPU=0 julia --project=. -e 'using Pkg; Pkg.test()'` (Julia 1.13.1, ubuntu-latest):
  1884 pass / 0 fail / 0 broken (test_matrix alone: 962). The CUDA extension was checked to load
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
    `CuSparseMatrixCSR{Tv,Ti}(::SparseMatrixCSC)` and always uses `Cint` (issue #30).
- Open issues / follow-ups:
  - #30: `test/backends.jl` `to_device(::CUDABackend, A, Int64)` returns Int32 indices.
  - No ROCm adapters (`CSR(::ROCSparseMatrixCSR)`) until T23; on the `amdgpu` runner the generic
    `to_backend`/`MatrixDescriptor` tests run on ROCm through `KernelAbstractions.allocate`.
  - `SparseMatrixCSC(::CSR)` on a device copies arrays to the host; it is a test/debug helper only.
- Suggested plan changes:
  - PLAN §1.1 / T02: decide whether `offsetType` (separate `rowptr` eltype) is needed for v1; if so,
    give `CSR` separate `VP`/`VI` parameters.

---

## T03 — Dense-op interface, capability audit, KA fallbacks   `[ ]`

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

---

## T04 — Benchmark harness and cuDSS baselines   `[ ]`

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

---

## T05 — Symbolic I: pattern, ordering, elimination tree, column counts   `[ ]`

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

---

## T06 — Symbolic II: supernodes and GPU-tuned amalgamation   `[ ]`

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

---

## T07 — Symbolic III: schedule, static layout, device maps   `[ ]`

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

---

## T08 — CPU reference multifrontal Cholesky (the oracle)   `[ ]`

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

---

## T09 — GPU multifrontal Cholesky with assembly kernels (regime C on every front)   `[ ]`

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
