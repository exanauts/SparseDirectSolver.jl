# SparseDirectSolver.jl

A portable sparse direct solver (LLᵀ/LLᴴ, LDLᵀ/LDLᴴ, LDU) for GPUs, written in
Julia on KernelAbstractions.jl and GPUArrays.jl. It keeps the parameter names
and phases of [CUDSS.jl](https://github.com/exanauts/CUDSS.jl) so that MadNLP
and other cuDSS users can switch to it mechanically, and targets CUDA, AMDGPU,
oneAPI, Metal and the KernelAbstractions CPU backend.

**Status: v0.1 under construction.** Working today, on the CPU backend and on
CUDA:

* host symbolic analysis: AMD or nested dissection (METIS through the Metis
  extension) ordering, elimination tree, supernodes with GPU-tuned
  amalgamation, static factor layout and device assembly maps;
* GPU multifrontal Cholesky (`"SPD"`, `"HPD"`) in three regimes: fused
  subtree-per-workgroup kernels for the many small fronts at the bottom of the
  tree, fused level-batched per-front kernels for medium fronts, and dense
  `potrf`/`trsm`/`syrk` through a backend-agnostic dense interface (cuBLAS and
  cuSOLVER on CUDA, KernelAbstractions fallbacks elsewhere) for large fronts;
* GPU triangular solves with multiple right-hand sides, forward and backward
  sub-phases and permutations, allocation-free after the analysis;
* the public API: `DirectSolver`, `execute!` with cuDSS phase strings, named
  phase wrappers, `update!`, `setparam!`/`getparam`, and the `LinearAlgebra`
  layer (`cholesky`, `cholesky!`, `ldiv!`, `\`, `logabsdet`), checked by
  the test suite of CUDSS.jl ported to this package;
* a CPU reference LDLᵀ/LDLᴴ with in-front Bunch–Kaufman pivoting,
  perturbation, pivot sign policy and inertia, the oracle for the GPU LDLᵀ
  that comes next.

Not there yet: GPU LDLᵀ/LDLᴴ (`"S"`, `"H"`), iterative refinement,
`solve_mode`, LU (`"G"`), batches, Schur complements, matching and scaling,
and the AMDGPU, oneAPI and Metal extensions. Unsupported structures,
phases and parameters raise `NotSupportedError` rather than falling back
silently.

* [`PLAN.md`](PLAN.md) — design, API, milestones.
* [`TASKS.md`](TASKS.md) — implementation tasks and their reports.
* [`RESEARCH.md`](RESEARCH.md) — background and state of the art.
* [`bench/README.md`](bench/README.md) — benchmark harness and cuDSS baselines.
* [`bench/comparison/comparison.md`](bench/comparison/comparison.md) — per-feature performance comparison with cuDSS.

## Installation

The package is not registered yet. Julia 1.10 or later is required.

```julia
using Pkg
Pkg.add(url = "https://github.com/exanauts/SparseDirectSolver.jl")
```

Loading CUDA.jl enables the CUDA extension (`CuSparseMatrixCSR`/`CuSparseMatrixCSC`
constructors, cuBLAS/cuSOLVER dense kernels). Loading Metis.jl enables nested
dissection ordering; without it the ordering is AMD.

## Usage

The handle API mirrors CUDSS.jl: a `DirectSolver` is created from a CSR matrix
living on a KernelAbstractions backend, with a structure string (`"SPD"`,
`"HPD"`; `"S"`, `"H"`, `"G"` are reserved for the next milestones) and the
triangle that is read (`'L'`, `'U'` or `'F'`). Phases are run with `execute!`.

```julia
using SparseDirectSolver, SparseArrays, LinearAlgebra
using CUDA, CUDA.CUSPARSE

n = 1000
A = sprand(n, n, 0.005); A = A * A' + I               # SPD on the host
A_gpu = CuSparseMatrixCSR(tril(A))                    # lower triangle on the device
b_gpu = CuVector(rand(n)); x_gpu = similar(b_gpu)

solver = DirectSolver(A_gpu, "SPD", 'L')
setparam!(solver, "reordering_alg", "default")        # cuDSS parameter names
execute!("analysis", solver, x_gpu, b_gpu)            # reordering + symbolic factorization
execute!("factorization", solver, x_gpu, b_gpu)
execute!("solve", solver, x_gpu, b_gpu)
getparam(solver, "info") == 0 || error("factorization failed")

# new values, same pattern: reuse the analysis
update!(solver, CuSparseMatrixCSR(tril(A + I)))
execute!("refactorization", solver, x_gpu, b_gpu)
execute!("solve", solver, x_gpu, b_gpu)
```

The same code runs on the CPU backend by wrapping the host CSR arrays in a
`CSR` (or passing a `SparseMatrixCSC`) instead of a `CuSparseMatrixCSR`. The
`LinearAlgebra` layer offers the usual shortcuts:

```julia
F = cholesky(A_gpu; view = 'L')        # analysis + factorization
ldiv!(x_gpu, F, b_gpu)                 # or x_gpu = F \ b_gpu
cholesky!(F, A_gpu_new)                # refactorization with the same pattern
```

The named phase wrappers `analyze!`, `factorize!`, `refactorize!` and `solve!`
are equivalent to the corresponding `execute!` calls.

## Running the tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'                       # CPU backend + GPUs found in test/Project.toml
SDS_TEST_GPU=0 julia --project=. -e 'using Pkg; Pkg.test()'        # CPU only
SDS_TEST_CPU=0 julia --project=. -e 'using Pkg; Pkg.test()'        # GPU only
SDS_TEST_ONLY="test_symbolic_etree,test_options" julia --project=. -e 'using Pkg; Pkg.test()'
SDS_TEST_SKIP="test_aqua" julia --project=. -e 'using Pkg; Pkg.test()'
```

GPU backends are tested when their package is present in the test environment
and functional. CI adds them itself; locally, add one with
`julia --project=test -e 'using Pkg; Pkg.add("CUDA")'` and do not commit that
change to `test/Project.toml`.

## Development

The project is developed task by task; `TASKS.md` lists the tasks and the
report of every finished one. Each task lands through a pull request that CI
(CPU suite on GitHub runners, GPU suite on self-hosted `cuda` runners) and an
automated review must pass. `AGENTS.md` describes the workflow and the code
conventions.

## License

MIT, see [`LICENSE`](LICENSE).
