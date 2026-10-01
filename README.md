# SparseDirectSolver.jl

A portable sparse direct solver (LLᵀ/LLᴴ, LDLᵀ/LDLᴴ, LDU) for GPUs, written in
Julia on KernelAbstractions.jl and GPUArrays.jl. It keeps the parameter names
and phases of [CUDSS.jl](https://github.com/exanauts/CUDSS.jl) so that MadNLP
and other cuDSS users can switch to it mechanically, and runs on CUDA, AMDGPU,
oneAPI, Metal and the KernelAbstractions CPU backend.

**Status: under construction.** Only the package scaffolding and the option
tables exist so far.

* [`PLAN.md`](PLAN.md) — design, API, milestones.
* [`TASKS.md`](TASKS.md) — implementation tasks and their reports.
* [`RESEARCH.md`](RESEARCH.md) — background and state of the art.

## Running the tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'                       # CPU (+ GPUs found in test/Project.toml)
SDS_TEST_GPU=0 julia --project=. -e 'using Pkg; Pkg.test()'        # CPU only
SDS_TEST_ONLY="test_options" julia --project=. -e 'using Pkg; Pkg.test()'
```

GPU backends are tested when their package is present in the test environment
and functional. CI adds them itself; locally, add one with
`julia --project=test -e 'using Pkg; Pkg.add("CUDA")'` and do not commit that
change to `test/Project.toml`.
