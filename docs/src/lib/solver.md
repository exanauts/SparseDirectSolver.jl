# Solver

```@docs
SparseDirectSolver
AbstractDirectSolver
DirectSolver
execute!
analyze!
factorize!
refactorize!
solve!
update!
```

## Parameters

```@docs
setparam!(::DirectSolver, ::AbstractString, ::Any)
getparam(::DirectSolver, ::AbstractString)
getparam!
Options
setparam!(::Options, ::AbstractString, ::Any)
getparam(::Options, ::AbstractString)
CONFIG_PARAMETERS
DATA_PARAMETERS
CUDSS08_DATA_PARAMETERS
EXTRA_PARAMETERS
default_pivot_epsilon
SparseDirectSolver.set_log_level!
```

## Errors

```@docs
SparseDirectSolverError
NotSupportedError
InvalidValueError
FactorizationError
InterruptedError
```
