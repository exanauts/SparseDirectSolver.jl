# CUDA extension

Loaded with `using CUDA`. It accepts the sparse matrices of cuSPARSE and routes
the dense operations of large fronts to cuBLAS and cuSOLVER.

```@autodocs
Modules = [Base.get_extension(SparseDirectSolver, :SparseDirectSolverCUDAExt)]
```
