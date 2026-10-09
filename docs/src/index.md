# SparseDirectSolver.jl

SparseDirectSolver.jl is a portable sparse direct solver for GPUs, written in
Julia on [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)
and [GPUArrays.jl](https://github.com/JuliaGPU/GPUArrays.jl). It factors

* symmetric and Hermitian positive definite matrices: ``A = L L^T`` / ``L L^H``
  (structures `"SPD"`, `"HPD"`),
* symmetric and Hermitian indefinite matrices: ``A = L D L^T`` / ``L D L^H`` with
  1×1 and 2×2 pivots (structures `"S"`, `"H"`),
* general matrices: ``A = L D U`` with threshold partial pivoting (structure `"G"`),

and solves with them on the GPU. It keeps the parameter names and the phases of
[CUDSS.jl](https://github.com/exanauts/CUDSS.jl), the Julia interface of NVIDIA
cuDSS, so that [MadNLP](https://github.com/MadNLP/MadNLP.jl) and other cuDSS
users can switch to it mechanically ([Migrating from CUDSS.jl](@ref)). Unlike
cuDSS it is open source and not tied to one vendor: it runs on CUDA, AMDGPU,
oneAPI, Metal and the KernelAbstractions CPU backend.

!!! warning "Status"
    Version 0.1 is under construction. The CPU backend, CUDA and AMDGPU are
    tested; the oneAPI and Metal extensions, non-uniform batches and mixed
    precision are not there yet. Unsupported structures, phases and parameters raise
    [`NotSupportedError`](@ref) rather than falling back silently.

## Features

* **Analysis on the host**: AMD or nested dissection (METIS, through the Metis
  extension) ordering, elimination tree, supernodes with GPU-tuned
  amalgamation, a static factor layout and device assembly maps. For symmetric
  indefinite matrices the analysis pairs structurally zero pivots with a
  partner so that the in-front pivoting can form the 2×2 blocks of KKT systems.
* **Multifrontal factorization on the device** in three regimes: fused
  subtree-per-workgroup kernels for the many small fronts at the bottom of the
  elimination tree, level-batched per-front kernels for medium fronts, and dense
  `potrf`/`getrf`/`trsm`/`syrk` calls (cuBLAS and cuSOLVER on CUDA,
  KernelAbstractions kernels elsewhere) for large fronts.
* **Pivoting**: Bunch–Kaufman 1×1/2×2 pivoting for LDLᵀ/LDLᴴ, threshold partial
  pivoting inside the fully-summed block of each front for LU, static
  perturbation of tiny pivots with a user-chosen sign per row (`pivot_sign`, which
  cuDSS does not have), inertia and pivot statistics.
* **Solves** with multiple right-hand sides, transposed and conjugated systems
  (`solve_mode`), iterative refinement or FGMRES refinement (with Krylov.jl), and
  every solve sub-phase of cuDSS.
* **Schur complement mode**, **uniform batches**, **matching and scaling**
  (MC64 jobs 1–5).
* A `LinearAlgebra` layer: `cholesky`, `ldlt`, `lu`, their in-place variants,
  `ldiv!`, `\`, `logabsdet`.

## Installation

The package is not registered yet. Julia 1.13 or later is required.

```julia
using Pkg
Pkg.add(url = "https://github.com/exanauts/SparseDirectSolver.jl")
```

Loading CUDA.jl enables the CUDA extension (`CuSparseMatrixCSR` and
`CuSparseMatrixCSC` inputs, cuBLAS and cuSOLVER dense kernels); loading
AMDGPU.jl enables the AMDGPU extension (`ROCSparseMatrixCSR` and
`ROCSparseMatrixCSC` inputs, rocBLAS and rocSOLVER dense kernels). Loading Metis.jl
enables nested dissection ordering; Krylov.jl enables FGMRES refinement.

## Where to go next

* [Quick start](@ref): factor and solve on the CPU backend and on CUDA.
* [Phases](man/phases.md) and [Parameters](man/parameters.md): the handle API, phase by phase.
* [Symmetric indefinite systems](@ref): KKT matrices, pivoting and inertia.
* [Performance](@ref): comparison with cuDSS.
* The design documents in the repository: [`PLAN.md`](https://github.com/exanauts/SparseDirectSolver.jl/blob/main/PLAN.md)
  (architecture and API), [`TASKS.md`](https://github.com/exanauts/SparseDirectSolver.jl/blob/main/TASKS.md)
  (implementation tasks and their reports) and
  [`RESEARCH.md`](https://github.com/exanauts/SparseDirectSolver.jl/blob/main/RESEARCH.md)
  (state of the art).
