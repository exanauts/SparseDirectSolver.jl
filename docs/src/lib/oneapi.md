# oneAPI extension

Loaded with `using oneAPI`. It accepts the sparse matrices of oneMKL, routes the
level-3 BLAS operations of large fronts (`gemm`, `syrk`/`herk`, `trsm` and
their strided batches) to oneMKL, and lets the regime-A subtree kernels use the
64 KiB local-memory class on Intel GPUs. Every oneMKL call waits for its queue
and the LAPACK routines allocate a scratchpad, so the Cholesky and LU panel
factorizations stay on the KernelAbstractions kernels unless `impl = :vendor`
is requested.

```@autodocs
Modules = [Base.get_extension(SparseDirectSolver, :SparseDirectSolverOneAPIExt)]
```
