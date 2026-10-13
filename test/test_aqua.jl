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
# persistent_tasks is skipped under KernelAbstractions >= 0.10, where it reports the
# backend, not this package: KA 0.10's CPU backend keeps worker tasks alive at exit,
# and in Aqua's fresh temporary project its pocl_standalone_jll dependency fails to
# precompile ("Cannot load SLEEF_jll into module pocl_standalone_jll", seen on CI).
# Upstream; re-enable when both are fixed. CI's Aqua workflow applies the same gate.
Aqua.test_all(SparseDirectSolver; ambiguities = false, undefined_exports = false,
              persistent_tasks = (pkgversion(KernelAbstractions) >= v"0.10" ? false : (tmax = 120,)))
