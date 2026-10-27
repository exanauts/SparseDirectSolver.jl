# Backends

The solver is written once in KernelAbstractions.jl. The matrix given to
[`DirectSolver`](@ref) decides the backend: the CSR arrays are wrapped without a
copy and every numeric phase runs where they live.

| backend | input | dense kernels for large fronts | status |
| :--- | :--- | :--- | :--- |
| CPU (`KernelAbstractions.CPU()`) | `SparseMatrixCSC`, [`CSR`](@ref) of `Vector`s | KernelAbstractions kernels | tested |
| CUDA (`using CUDA`) | `CuSparseMatrixCSR`, `CuSparseMatrixCSC`, [`CSR`](@ref) of `CuArray`s | cuBLAS, cuSOLVER | tested |
| AMDGPU (`using AMDGPU`) | `ROCSparseMatrixCSR`, `ROCSparseMatrixCSC`, [`CSR`](@ref) of `ROCArray`s | rocBLAS, rocSOLVER | tested (MI300X) |
| Metal (`using Metal`) | [`CSR`](@ref) of `MtlArray`s | Metal.jl / KernelAbstractions kernels | Float32, tested (M4 Max) |
| oneAPI (`using oneAPI`) | `oneSparseMatrixCSR`, `oneSparseMatrixCSC`, [`CSR`](@ref) of `oneArray`s | oneMKL BLAS, KernelAbstractions kernels for the panel factorizations | tested (Intel Data Center GPU Max 1550) |

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
types are `Int32` and `Int64`. Metal currently supports `Float32` factors;
Apple GPUs do not expose Float64 arithmetic.

On Metal, the default SPD schedule uses narrower regime-C thresholds than the
CUDA/ROCm schedule (`32` columns and `128` rows), Metal.jl's panel GEMM/SYRK,
and fused level solves for every front. Under `impl = :auto` with vendor calls
allowed (not `factorization_alg = "algo1"`), regime-C Cholesky fronts at least
64 columns wide use offset-aware MPS decomposition objects built when the
numeric storage is allocated; the MPS status is folded into `info` (MPS names no
column, so a failure it alone detects is reported at the last column of the
front). On an M4 Max, the 90,000-row `lap2d_300` warmed refactorization fell
from 142 ms to 62 ms while its 5.2 ms solve time was preserved. The narrower SPD
thresholds apply only when every schedule option has its default value.

LDLT, LU and uniform batches run every front through the regime-C path
(`factorization_alg = "algo2"`, no regime-A subtrees), because the fused kernels
bind 32 device buffers and Apple GPUs expose 31. An explicit
`factorization_alg = "algo1"` or explicit `subtree_budgets` for them raises a
[`NotSupportedError`](@ref) at analysis rather than being replaced.
