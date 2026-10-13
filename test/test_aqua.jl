# Package quality checks (method ambiguities are excluded, as in CI's Aqua workflow).
#
# The persistent-tasks check loads the package in a fresh process and requires
# it to exit within `tmax`. Aqua's default of 30 s is too tight on the shared
# self-hosted GPU runners when several jobs precompile at once (seen on a
# CUDA job), so give it more headroom; a real persistent task still
# fails, it just takes longer to report.

# undefined_exports is excluded HERE only: SDSSolver/SDSSolverOptions are declared
# globals the MadNLP extension assigns, and MadNLP is not in the test environment.
# CI's Aqua workflow loads MadNLP and runs that check with the exports assigned.
# persistent_tasks is skipped under KernelAbstractions >= 0.10: its CPU backend keeps
# worker tasks alive at exit, so the check reports the backend, not this package
# (upstream; re-enable when KA's task pool shuts down cleanly)
Aqua.test_all(SparseDirectSolver; ambiguities = false, undefined_exports = false,
              persistent_tasks = (pkgversion(KernelAbstractions) >= v"0.10" ? false : (tmax = 120,)))
