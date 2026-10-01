# Exception hierarchy. Unknown parameter or phase *names* raise `ArgumentError`
# (as in CUDSS.jl); everything else raised by the package derives from
# `SparseDirectSolverError`.

"""
    SparseDirectSolverError <: Exception

Abstract supertype of the exceptions raised by SparseDirectSolver.jl.
"""
abstract type SparseDirectSolverError <: Exception end

"""
    NotSupportedError(msg)

A parameter, value or feature that SparseDirectSolver.jl does not provide
(for example `"device_count" > 1`, or global pivoting `pivot_type = 'C'`).
"""
struct NotSupportedError <: SparseDirectSolverError
    msg::String
end

"""
    InvalidValueError(msg)

A value of the wrong type or outside the admissible range for a parameter,
structure, view, index base or phase.
"""
struct InvalidValueError <: SparseDirectSolverError
    msg::String
end

"""
    FactorizationError(info, msg = "")

A factorization that could not be completed, or a phase called out of order.
`info` follows the `"info"` data parameter: `k > 0` is the first failed pivot
(1-based, original numbering); for uniform batches it is a vector with one
entry per member.
"""
struct FactorizationError{I} <: SparseDirectSolverError
    info::I
    msg::String
end

FactorizationError(info) = FactorizationError(info, "")

"""
    InterruptedError(msg = "interrupted through \"user_host_interrupt\"")

Raised when the `"user_host_interrupt"` flag is set while a phase runs. The
solver stays usable: the phase can be executed again once the flag is cleared.
"""
struct InterruptedError <: SparseDirectSolverError
    msg::String
end

InterruptedError() = InterruptedError("interrupted through \"user_host_interrupt\"")

Base.showerror(io::IO, e::NotSupportedError) = print(io, "NotSupportedError: ", e.msg)
Base.showerror(io::IO, e::InvalidValueError) = print(io, "InvalidValueError: ", e.msg)
Base.showerror(io::IO, e::InterruptedError) = print(io, "InterruptedError: ", e.msg)

function Base.showerror(io::IO, e::FactorizationError)
    print(io, "FactorizationError: info = ", e.info)
    isempty(e.msg) || print(io, ": ", e.msg)
    return nothing
end
