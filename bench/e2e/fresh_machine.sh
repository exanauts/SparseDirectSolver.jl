#!/usr/bin/env bash
# Fresh-machine test of the MadNLP + SparseDirectSolver workflow on an NVIDIA GPU box.
# Installs Julia 1.13 (juliaup), clones the repo, runs the test suites (CPU + CUDA),
# then the end-to-end MadNLP ACOPF solve (78k-bus pglib case, SparseCondensedKKTSystem)
# with cuDSS and with SDS as the linear solver, on the same device.
#
#   bash fresh_machine.sh
#
# Knobs (env vars):
#   SDS_BRANCH=divfree-chol   git ref to test (default: the division-free kernels PR)
#   SDS_DIR=$HOME/sds-test    checkout location
#   SKIP_TESTS=1              skip the test suites, run only the MadNLP workflow
#   CPU_ONLY_TESTS=1          run only the CPU suite (no functional GPU needed for it)
#
# Needs: git, curl, an NVIDIA GPU + driver for the CUDA suite and the workflow
# (CUDA toolkit and cuDSS are downloaded by CUDA.jl/CUDSS.jl as artifacts),
# network access (pglib case data is fetched by ExaModelsPower on first use).
set -euo pipefail

SDS_BRANCH="${SDS_BRANCH:-divfree-chol}"
SDS_DIR="${SDS_DIR:-$HOME/sds-test}"

# --- 1. Julia 1.13 via juliaup -----------------------------------------------
if ! command -v juliaup >/dev/null 2>&1 && [ ! -x "$HOME/.juliaup/bin/juliaup" ]; then
    curl -fsSL https://install.julialang.org | sh -s -- --yes
fi
export PATH="$HOME/.juliaup/bin:$PATH"
juliaup add 1.13 2>/dev/null || true
JL="julia +1.13"
$JL --version

# --- 2. Repository ------------------------------------------------------------
if [ ! -d "$SDS_DIR/.git" ]; then
    git clone https://github.com/exanauts/SparseDirectSolver.jl.git "$SDS_DIR"
fi
cd "$SDS_DIR"
git fetch origin "$SDS_BRANCH"
git checkout "$SDS_BRANCH"
git pull --ff-only origin "$SDS_BRANCH" || true
echo "== testing $(git log --oneline -1)"

# --- 3. Test suites -----------------------------------------------------------
if [ "${SKIP_TESTS:-0}" != "1" ]; then
    $JL --project=. -e 'using Pkg; Pkg.instantiate()'
    echo "== CPU suite"
    SDS_TEST_GPU=0 $JL --project=. -e 'using Pkg; Pkg.test()'
    if [ "${CPU_ONLY_TESTS:-0}" != "1" ]; then
        echo "== CUDA suite"
        $JL --project=test -e 'using Pkg; Pkg.add("CUDA")'   # as .github/workflows/ci.yml does
        SDS_TEST_CPU=0 SDS_TEST_SKIP=test_aqua $JL --project=. -e 'using Pkg; Pkg.test()'
    fi
fi

# --- 4. MadNLP workflow: 78k-bus ACOPF, cuDSS vs SDS on the same GPU -----------
# bench/e2e: MadNLPSDS.jl defines the MadNLP.AbstractLinearSolver wrappers
# (SDSSolver = stock library; SDSProtoSolver = PR #107 prototype kernels);
# run_gv100.jl solves the case with MadNLPGPU.CUDSSSolver and with SDSProtoSolver
# and prints wall/solver/linear-solver times, iteration counts and objectives
# (convergence parity is the workflow test: same iterations, same objective).
echo "== MadNLP + SDS workflow (downloads the pglib case on first use)"
$JL --project=bench/e2e -e "using Pkg; Pkg.develop(path = \"$SDS_DIR\"); Pkg.instantiate()"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" $JL --project=bench/e2e bench/e2e/run_gv100.jl
echo "== fresh-machine run complete"
