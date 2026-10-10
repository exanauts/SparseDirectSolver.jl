#!/usr/bin/env bash
# Fresh-machine test of the MadNLP + SparseDirectSolver workflow on a GPU box.
# Installs Julia 1.13 (juliaup), clones the repo, runs the test suites (CPU + GPU),
# then the end-to-end MadNLP ACOPF solve (78k-bus pglib case, SparseCondensedKKTSystem).
# NVIDIA: cuDSS vs SDS on the same device. AMD: SDS stock vs SDS prototype kernels
# (no cuDSS exists there). The vendor is auto-detected (nvidia-smi / rocm-smi), or
# force it with SDS_BACKEND=cuda|amdgpu.
#
# Everything is non-interactive (nothing reads the terminal), so the intended use
# is detached from the login session (the suites + workflow run for an hour or more):
#
#   curl -fsSL https://raw.githubusercontent.com/exanauts/SparseDirectSolver.jl/divfree-chol/bench/e2e/fresh_machine.sh -o fresh_machine.sh
#   nohup bash fresh_machine.sh > sds-fresh.log 2>&1 &     # then log out; tail -f sds-fresh.log
#
# or under Slurm:  sbatch -p <gpu-partition> --gres=gpu:1 -c 8 --mem=64G -t 120 --wrap "bash fresh_machine.sh"
#
# Knobs (env vars):
#   SDS_BRANCH=divfree-chol   git ref to test (default: the division-free kernels PR)
#   SDS_DIR=$HOME/sds-test    checkout location
#   SKIP_TESTS=1              skip the test suites, run only the MadNLP workflow
#   CPU_ONLY_TESTS=1          run only the CPU suite (no functional GPU needed for it)
#   SDS_BACKEND=cuda|amdgpu   override the GPU vendor autodetection
#
# Needs: git, curl, network access (pglib case data is fetched by ExaModelsPower on
# first use), and a GPU + driver. NVIDIA: the CUDA toolkit and cuDSS are downloaded
# by CUDA.jl/CUDSS.jl as artifacts, nothing to install. AMD: a system ROCm
# installation is required (AMDGPU.jl uses the system ROCm, it is not an artifact).
set -euo pipefail
exec < /dev/null                     # never read the terminal: safe to detach

SDS_BRANCH="${SDS_BRANCH:-divfree-chol}"
SDS_DIR="${SDS_DIR:-$HOME/sds-test}"

if [ -z "${SDS_BACKEND:-}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        SDS_BACKEND=cuda
    elif command -v rocm-smi >/dev/null 2>&1; then
        SDS_BACKEND=amdgpu
    else
        echo "no GPU detected (nvidia-smi / rocm-smi); set SDS_BACKEND=cuda|amdgpu" >&2
        SDS_BACKEND=cuda
    fi
fi
echo "== backend: $SDS_BACKEND"
GPU_PKG=CUDA; RUNNER=run_gv100.jl
[ "$SDS_BACKEND" = "amdgpu" ] && GPU_PKG=AMDGPU && RUNNER=run_amd.jl

# --- 1. Julia 1.13 via juliaup -----------------------------------------------
if ! command -v juliaup >/dev/null 2>&1 && [ ! -x "$HOME/.juliaup/bin/juliaup" ]; then
    curl -fsSL https://install.julialang.org | sh -s -- --yes
fi
export PATH="$HOME/.juliaup/bin:$PATH"
juliaup add 1.13 2>/dev/null || true
JL="julia +1.13 --startup-file=no"
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
        echo "== $GPU_PKG suite"
        $JL --project=test -e "using Pkg; Pkg.add(\"$GPU_PKG\")"   # as .github/workflows/ci.yml does
        SDS_TEST_CPU=0 SDS_TEST_SKIP=test_aqua $JL --project=. -e 'using Pkg; Pkg.test()'
    fi
fi

# --- 4. MadNLP workflow: 78k-bus ACOPF, cuDSS vs SDS on the same GPU -----------
# bench/e2e: MadNLPSDS.jl defines the MadNLP.AbstractLinearSolver wrappers
# (SDSSolver = stock library; SDSProtoSolver = PR #107 prototype kernels).
# NVIDIA (run_gv100.jl): MadNLPGPU.CUDSSSolver vs SDSProtoSolver.
# AMD (run_amd.jl): SDSSolver vs SDSProtoSolver.
# Each prints wall/solver/linear-solver times, iteration counts and objectives
# (convergence parity is the workflow test: same iterations, same objective).
echo "== MadNLP + SDS workflow via $RUNNER (downloads the pglib case on first use)"
$JL --project=bench/e2e -e "using Pkg; Pkg.develop(path = \"$SDS_DIR\"); Pkg.instantiate()"
if [ "$SDS_BACKEND" = "cuda" ]; then
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" $JL --project=bench/e2e "bench/e2e/$RUNNER"
else
    $JL --project=bench/e2e "bench/e2e/$RUNNER"
fi
echo "== fresh-machine run complete"
