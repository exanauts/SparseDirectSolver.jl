"""
    SparseDirectSolverKrylovExt

FGMRES-IR for SparseDirectSolver (PLAN §2.5, `ir_mode = "fgmres"`). Loading
Krylov.jl registers [`fgmres_correction`](@ref) in
`SparseDirectSolver.FGMRES_PROVIDER`; the operators (SpMV and the
factorization as right preconditioner) and the refinement loop live in the
core (`src/solve/refinement.jl`), this extension only runs `Krylov.fgmres!`.
"""
module SparseDirectSolverKrylovExt

using SparseDirectSolver
using Krylov

"""
    fgmres_correction(cache, A, P, R, len; atol, itmax) -> (D, iterations)

Solve `A D = R[1:len]` (the first `len` entries of the device array `R`,
copied into the Krylov storage) by `Krylov.fgmres!` with right
preconditioner `P` (`mul!`), no restart, absolute tolerance `atol`, relative
tolerance 0 and at most `itmax` iterations. The `FgmresWorkspace` and the
right-hand side vector are kept in `cache` (a `Ref{Any}`) and reallocated only
when `len` or the element/array type change or `itmax` exceeds the stored
Krylov basis. `D` is the workspace's solution vector (valid until the next
call).
"""
function fgmres_correction(cache::Base.RefValue{Any}, A, P, R::AbstractArray{FC}, len::Int; atol::Real,
                           itmax::Int) where {FC}
    S = typeof(similar(R, FC, 0))
    c = cache[]
    if c isa Tuple{Krylov.FgmresWorkspace{real(FC), FC, S}, S, Int} && c[1].m == len && c[3] >= itmax
        workspace, b, _ = c
    else
        b = similar(R, FC, len)
        workspace = Krylov.FgmresWorkspace(len, len, S; memory = itmax)
        cache[] = (workspace, b, itmax)
    end
    copyto!(b, 1, R, 1, len)
    Krylov.fgmres!(workspace, A, b; N = P, ldiv = false, restart = false, atol = real(FC)(atol),
                   rtol = zero(real(FC)), itmax)
    return workspace.x, workspace.stats.niter
end

function __init__()
    SparseDirectSolver.FGMRES_PROVIDER[] = fgmres_correction
    return nothing
end

end # module SparseDirectSolverKrylovExt
