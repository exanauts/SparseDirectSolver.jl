module SparseDirectSolverCUDAExt

# CUDA support: CSR adapters (T02) and vendor dense bindings (T03) live here.
# The extension is declared from T01 on so that loading CUDA exercises it.

using SparseDirectSolver
using CUDACore
using cuSPARSE

end # module SparseDirectSolverCUDAExt
