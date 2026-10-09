# AMDGPU extension

Loaded with `using AMDGPU`. It accepts the sparse matrices of rocSPARSE, routes
the dense operations of large fronts to rocBLAS and rocSOLVER, and lets the
regime-A subtree kernels use the 64 KiB of local memory (LDS) per workgroup of
AMD GPUs.

```@autodocs
Modules = [Base.get_extension(SparseDirectSolver, :SparseDirectSolverAMDGPUExt)]
```
