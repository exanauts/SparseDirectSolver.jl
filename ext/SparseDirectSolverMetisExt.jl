"""
    SparseDirectSolverMetisExt

Nested-dissection ordering for SparseDirectSolver (PLAN §2.3 step 2). Loading
Metis.jl activates CliqueTrees' own Metis extension; this extension only
registers the algorithm object in `SparseDirectSolver.ND_PROVIDER`.
"""
module SparseDirectSolverMetisExt

using SparseDirectSolver
using CliqueTrees
using Metis

# METIS_NodeND with `ufactor = nd_ubfactor` and `nseps = nd_nseps` (-1: METIS
# defaults; values < 1 fall back to the default). `nd_nlevels` is cuDSS'
# *minimum* number of dissection levels; METIS_NodeND dissects recursively
# until the parts are small, so the minimum holds whenever the graph is large
# enough to be split that often. CliqueTrees' level-capped `ND{S}` (AMD below the
# cap) was 4–13× slower and gave 15–80% more fill on Laplacians (T05 Report).
function nd_algorithm(nlevels::Integer, ubfactor::Integer, nseps::Integer)
    return CliqueTrees.METIS(; ufactor = ubfactor, nseps = nseps >= 1 ? nseps : -1)
end

function __init__()
    SparseDirectSolver.ND_PROVIDER[] = nd_algorithm
    return nothing
end

end # module SparseDirectSolverMetisExt
