# oneAPI extension

Loaded with `using oneAPI`. It accepts the sparse matrices of oneMKL, routes the
level-3 BLAS operations of large fronts (`gemm`, `syrk`/`herk`, `trsm` and
their strided batches) to oneMKL, and lets the regime-A subtree kernels use the
64 KiB local-memory class on Intel GPUs. Every oneMKL call waits for its queue
and the LAPACK routines allocate a scratchpad, so the Cholesky and LU panel
factorizations stay on the KernelAbstractions kernels; oneMKL's `potrf` and
`getrf` are reached only through an explicit `impl = :vendor` of the internal
dense interface, which is not a solver parameter. oneMKL's GPU `potrf` reports
the failing column relative to its 32-column panel; the extension reads the column from the factor's diagonal
instead, so `potrf!`, `potrf_info!` and the solver's `info` are exact.

```@autodocs
Modules = [Base.get_extension(SparseDirectSolver, :SparseDirectSolverOneAPIExt)]
```
