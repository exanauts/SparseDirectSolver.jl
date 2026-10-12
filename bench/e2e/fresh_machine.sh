#!/usr/bin/env bash
# Fresh-machine test of the MadNLP + SparseDirectSolver workflow, on the @sds named
# PROJECT environment — no ad-hoc Pkg.add: the environment's Project.toml is generated
# from bench/e2e/Project.toml with SparseDirectSolver wired to the local checkout, and
# Pkg.instantiate() provides everything from the project file.
#
#   curl -fsSL https://raw.githubusercontent.com/exanauts/SparseDirectSolver.jl/main/bench/e2e/fresh_machine.sh | bash
#
# When a terminal is attached, the workflow ends in an interactive REPL (julia -i)
# with `m`, `sol`, `sol2` and `solve()` defined; detached (nohup/sbatch, no tty) it
# just runs and exits. After one run the environment is reusable from anywhere:
#
#   julia +1.13 --startup-file=no --project=@sds bench/e2e/workflow.jl
#
# Knobs (env vars):
#   SDS_BRANCH=main              git ref to test
#   SDS_DIR=$HOME/sds-test        checkout location
#   SDS_BACKEND=cuda|amdgpu       GPU vendor override (autodetected in workflow.jl)
#   RUN_TESTS=1                   also run the package test suite first (long)
#
# Needs: git, curl, network, and a GPU + driver. NVIDIA: CUDA toolkit and cuDSS come
# as artifacts. AMD: a system ROCm installation is required.
# NO `exec < /dev/null` here: under `curl | bash` the script itself arrives on stdin,
# and redirecting it away silently discards the rest of the script (measured: bash
# executes only its first read-ahead buffer and exits 0). Children that might read
# stdin get a per-command `< /dev/null` instead.
set -euo pipefail

SDS_BRANCH="${SDS_BRANCH:-main}"
SDS_DIR="${SDS_DIR:-$HOME/sds-test}"

# --- 1. Julia 1.13 via juliaup -----------------------------------------------
if ! command -v juliaup >/dev/null 2>&1 && [ ! -x "$HOME/.juliaup/bin/juliaup" ]; then
    curl -fsSL https://install.julialang.org | sh -s -- --yes < /dev/null
fi
export PATH="$HOME/.juliaup/bin:$PATH"
juliaup add 1.13 2>/dev/null || true
JL="julia +1.13 --startup-file=no"
$JL --version

# --- 2. Repository ------------------------------------------------------------
if [ ! -d "$SDS_DIR/.git" ]; then
    git clone --branch "$SDS_BRANCH" https://github.com/exanauts/SparseDirectSolver.jl.git "$SDS_DIR"
fi
cd "$SDS_DIR"
git fetch origin "$SDS_BRANCH"
git checkout "$SDS_BRANCH"
git merge --ff-only "origin/$SDS_BRANCH" || true
echo "== testing $(git log --oneline -1)"

# --- 3. The @sds environment, generated from the project file (no Pkg.add) -----
$JL -e '
using TOML
dir = pwd()
proj = TOML.parsefile(joinpath(dir, "bench", "e2e", "Project.toml"))
get!(proj, "sources", Dict{String, Any}())["SparseDirectSolver"] = Dict("path" => dir)
env = joinpath(DEPOT_PATH[1], "environments", "sds")
mkpath(env)
open(io -> TOML.print(io, proj), joinpath(env, "Project.toml"), "w")
println("== @sds environment: SparseDirectSolver -> ", dir)
'
$JL --project=@sds -e 'using Pkg; Pkg.instantiate()'

# --- 4. Optional: the package test suite ---------------------------------------
if [ "${RUN_TESTS:-0}" = "1" ]; then
    echo "== package test suite"
    $JL --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()' < /dev/null
fi

# --- 5. The workflow: 78k-bus ACOPF, two solves, timing split ------------------
# with a terminal attached, end in a REPL (m, sol, sol2, solve() are defined)
echo "== MadNLP + SDS workflow (downloads the pglib case on first use)"
if [ -e /dev/tty ] && [ -t 1 ]; then
    exec $JL -i --project=@sds bench/e2e/workflow.jl < /dev/tty
else
    $JL --project=@sds bench/e2e/workflow.jl
    echo "== fresh-machine run complete"
fi
