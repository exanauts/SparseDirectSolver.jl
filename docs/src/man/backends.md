# Backends

The solver is written once in KernelAbstractions.jl. The matrix given to
[`DirectSolver`](@ref) decides the backend: the CSR arrays are wrapped without a
copy and every numeric phase runs where they live.

| backend | input | dense kernels for large fronts | status |
| :--- | :--- | :--- | :--- |
| CPU (`KernelAbstractions.CPU()`) | `SparseMatrixCSC`, [`CSR`](@ref) of `Vector`s | KernelAbstractions kernels | tested |
| CUDA (`using CUDA`) | `CuSparseMatrixCSR`, `CuSparseMatrixCSC`, [`CSR`](@ref) of `CuArray`s | cuBLAS, cuSOLVER | tested |
| AMDGPU (`using AMDGPU`) | `ROCSparseMatrixCSR`, `ROCSparseMatrixCSC`, [`CSR`](@ref) of `ROCArray`s | rocBLAS, rocSOLVER | tested (MI300X) |
| oneAPI, Metal | [`CSR`](@ref) of device arrays | KernelAbstractions kernels | planned extensions |

[`to_backend`](@ref) moves a `SparseMatrixCSC` to a backend:

```julia
using KernelAbstractions
A_dev = to_backend(A, CUDABackend())     # or CPU(), ROCBackend(), …
```

The kernels follow portability rules that let them run on every backend: 1-D
workgroups, `@localmem` reductions and no warp intrinsics, no atomics in
assembly (an atomic-free variant exists for the one atomic kernel, selected by
`deterministic_mode = 1`), and no allocation or host synchronization inside the
numeric and solve phases. The symbolic analysis runs on the host.

Element types are `Float32`, `Float64`, `ComplexF32` and `ComplexF64`; index
types are `Int32` and `Int64`.
