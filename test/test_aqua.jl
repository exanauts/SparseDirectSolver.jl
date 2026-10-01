# Package quality checks (method ambiguities are excluded, as in CI's Aqua workflow).
#
# The persistent-tasks check loads the package in a fresh process and requires
# it to exit within `tmax`. Aqua's default of 30 s is too tight on the shared
# self-hosted GPU runners when several jobs precompile at once (seen on the
# Julia LTS CUDA job), so give it more headroom; a real persistent task still
# fails, it just takes longer to report.

Aqua.test_all(SparseDirectSolver; ambiguities = false, persistent_tasks = (tmax = 120,))
