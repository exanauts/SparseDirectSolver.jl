# Benchmarks

Baselines every later milestone is measured against (PLAN.md §5 M0, §7). The
`bench/` environment is separate from the package and the test suite; results
(`bench/results/`) and matrix dumps (`bench/data/`) are gitignored.

| File | Purpose |
| --- | --- |
| `matrices.jl` | module `BenchMatrices`: generated Laplacians (`lap2d_300`: 300×300 grid, n = 90 000; `lap3d_40`: 40³ grid, n = 64 000), SuiteSparse matrices through MatrixDepot (`HB/bcsstk17`, `Boeing/bcsstk38`, `GHS_psdef/apache2`, `Rajat/rajat21`, `TSOPF/TSOPF_RS_b39_c7`), loader for KKT dumps `bench/data/kkt_<case>_<kind>_<iter>.{mtx,jld2}` with `kind ∈ {k2, condensed}` |
| `harness.jl` | module `BenchHarness`: `time_phases` (median per phase over `nruns` after warm-up), CHOLMOD/UMFPACK CPU reference, CSV writer |
| `cudss_baseline.jl` | runs every matrix × structure (`SPD` and `S` for symmetric matrices, `G` for unsymmetric ones) and writes `bench/results/<solver>_baseline.csv` |
| `front_bins.jl` | regime B (fused per-front kernel, one launch per batch) vs regime C (per-front `potrf`/`trsm`/`syrk` through the dense interface) on synthetic batches per size bin; runs in the package environment: `julia --project=. bench/front_bins.jl [--backend=cuda] [--T=Float64] [--nb=256]` (CUDA from an environment on the load path) |
| `regimes.jl` | `factorize!` time on the generated matrices with regimes A+B+C (default), B+C (`subtree_budgets = []`) and C only (`factorization_alg = "algo2"`), with fronts per regime and launch counts; package environment: `julia --project=. bench/regimes.jl [--backend=cuda] [--T=Float64] [--only=lap2d_300,lap3d_40]` |
| `report.jl` | prints the CSV files as Markdown tables |
| `dump_madnlp_kkt.jl` | dumps MadNLP K2 and condensed KKT matrices of pglib-opf cases |
| `features.jl` | module `BenchFeatures`: the comparison features, one per planned capability in TASKS.md order (`cholesky_f64` … `mixed_precision`), with structure, element type, matrix selector and parameters; `task_status` reads the task's marker in TASKS.md |
| `compare.jl` | cuDSS vs SparseDirectSolver.jl per feature × matrix with BenchmarkTools; one solver per run (`--solver=cudss` or `--solver=sds`), results merged into `bench/comparison/<solver>.csv` |
| `profile_phases.jl` | where SDS spends refactorization and solve time on CUDA (PERFORMANCE.md, experiment 0): schedule per matrix (regime-A subtrees, B/C fronts, launch groups) and a CUPTI trace of one warm refactorization and solve (kernels, copies, busy time, longest kernel and its grid); writes `bench/profile/phase_split.{md,csv}`: `julia --project=bench bench/profile_phases.jl [--only=m1,m2] [--structures=SPD,S]` |
| `compare_report.jl` | renders `bench/comparison/comparison.md` (overview + one table per feature) and `comparison.png` (SDS/cuDSS ratio per feature and phase, plus a panel of the PR #107 prototypes from `comparison/prototype_pr107.csv`, numbers quoted from PERFORMANCE.md); environment `bench/report/` (CairoMakie) |
| `solve_proto_78k.jl`, `solve_proto_all.jl`, `solve_nd.jl`, `final_sweep.jl` | PR #107, experiment 6: partitioned-inverse solve (per-front `L₁₁` inverses, fused dependency-counter sweeps, single-block chain kernel) on the 78k-bus condensed KKT and the SPD harness, under METIS, the imported cuDSS ordering and native ND; configuration sweeps. CUDA, SPD, one right-hand side; `bench/` environment |
| `fact_split.jl`, `fact_fused.jl` | PR #107, experiment 5: split factorization (tiled SYRK, chunked TRSM, routing by `regime_c_rows`) and the segmented dependency-counter fused factorization with private contribution blocks; each reproduces its table in one run given the KKT dump |
| `order_search.jl`, `get_cudss_perm.jl` | PR #107 ordering study: METIS knob sweep and schedule depth; extraction of cuDSS's reordering through the raw C API for replay under `user_perm` |
| `ordering_chooser.jl` | T22 (issue #108): AMD vs ND per harness matrix and dump (schedule depth, fundamental and column-etree depth, nnz(L), flops) and the choice of the T22 and T05 cost models; host only, no GPU: `julia --project=bench bench/ordering_chooser.jl` (`SDS_BENCH_SUITESPARSE=0` skips MatrixDepot) |
| `amd_proto.jl`, `amd/Project.toml` | the solve prototype on ROCm (ROCArray, raw-array constructor, `Threads.atomic_fence`); environment `bench/amd/` |
| `e2e/MadNLPSDS.jl`, `e2e/SDSProto.jl`, `e2e/run_gv100.jl`, `e2e/run_amd.jl` | MadNLP `AbstractLinearSolver` over the public API (prototype of the External task) and the prototype kernels packaged backend-portably; full ACOPF interior-point runs on CUDA and AMD; environment `bench/e2e/` (MadNLP, MadNLPGPU, ExaModels, ExaModelsPower) |
| `pivot_pairs.jl` | 2×2 pivot pairs of the `S` analysis (issue #66): `pivot_pairs` = `none`/`default`/`all` on the K2 dumps and the KKT generators, with nnz(L), zero/perturbed/2×2 pivots, max abs L and factor error of the CPU reference LDLᵀ; package environment: `julia --project=. bench/pivot_pairs.jl [--only=case118,...] [--generators=false]` |

## Commands

```bash
# once: instantiate the benchmark environment (CUDA, CUDSS, MatrixDepot, ...)
julia --project=bench -e 'using Pkg; Pkg.instantiate()'

# cuDSS baseline (needs a functional CUDA GPU); SuiteSparse matrices are
# downloaded by MatrixDepot on first use (DATADEPS_ALWAYS_ACCEPT is set)
julia --project=bench bench/cudss_baseline.jl
julia --project=bench bench/cudss_baseline.jl --no-suitesparse --nruns=5 --only=lap2d_300,lap3d_40

# CPU reference (CHOLMOD for SPD/S, UMFPACK for G), no GPU needed
julia --project=bench bench/cudss_baseline.jl --solver=cholmod

# print the results
julia --project=bench bench/report.jl                     # every CSV in bench/results/
julia --project=bench bench/report.jl bench/results/cudss_baseline.csv

# MadNLP KKT dumps (separate environment; prints install hints and exits 0 without it)
julia --project=bench/kkt -e 'using Pkg; Pkg.add(["MadNLP", "ExaModels", "ExaModelsPower"])'
julia --project=bench/kkt bench/dump_madnlp_kkt.jl pglib_opf_case118_ieee pglib_opf_case1354_pegase --iters=1,10,20
```

## cuDSS vs SparseDirectSolver.jl comparison

`bench/comparison/` is committed (unlike `bench/results/`): the two CSVs, the
Markdown table and the plot are the record of where the package stands against
cuDSS. Rerun it by hand after a task lands:

```bash
julia --project=bench -e 'using Pkg; Pkg.instantiate()'          # once
julia --project=bench/report -e 'using Pkg; Pkg.instantiate()'   # once

julia --project=bench bench/compare.jl --solver=cudss              # cuDSS side, all features
julia --project=bench bench/compare.jl --solver=sds --backend=cuda # SparseDirectSolver.jl side
julia --project=bench/report bench/compare_report.jl              # comparison.md + comparison.png

# rerun one feature after its task lands (other rows of the CSV are kept)
julia --project=bench bench/compare.jl --solver=sds --features=ldlt
```

* **Features.** Every capability planned in TASKS.md is a row of the overview
  from the start. The SDS run skips a feature until its task is marked done
  (`[x]`/`[!]`) in TASKS.md, so its cells stay blank; `--force` runs it anyway
  and records the error. The gate is the task marker, not an error, because
  some parameters (`hybrid_memory_mode`, `ir_n_steps`) are accepted and
  ignored before their task. The cuDSS side runs every feature with a cuDSS
  counterpart, so its column is filled before the SDS one.
* **Two processes.** CUDSS.jl and the package's CUDA extension define the same
  `cholesky(::CuSparseMatrixCSR)` methods, so one run loads one solver. The SDS
  run uses only the handle layer (`DirectSolver`, `setparam!`, `execute!`,
  `getparam`) and loads Metis for the default AMD/ND ordering choice. The bench
  environment gets SparseDirectSolver from `..` through `[sources]`.
* **Timing.** BenchmarkTools, `evals = 1`, 5 samples per phase (`--samples`).
  The `setup` of each sample builds a fresh solver, runs the phases the timed
  one depends on and, on a GPU, keeps the device busy for 0.2 s: otherwise the
  GPU drops to a low-clock power state during the host-side setup and the timed
  phase starts slow (lap2d_300 factorization varied between 6 and 19 ms; with
  the spin it is a stable 5.9 ms, against 6.05 ms in the T04 baseline). The
  report shows medians; the CSV also keeps the minima. Before the trials, one
  run compiles and gives `info`, nnz(L) and the residual, and a second run is
  timed; if its factorization takes longer than `--single-run-above` (5 s),
  its times are recorded instead of a trial (`samples` = 1 in the CSV, listed
  under the feature's table). This keeps slow rows, such as the T15 LDLᵀ on
  the larger matrices, from costing twenty factorizations each. The CSV is
  rewritten after every row, so an interrupted run keeps what it measured.
* **Synthetic inputs.** `ubatch8` builds 8 value sets on one pattern by scaling
  the diagonal of member k by 1 + 0.01(k−1). `schur` takes the last
  min(64, n/10) rows as the Schur block and times `solve_fwd_schur` as the
  solve (no residual). `nubatch` batches all selected condensed dumps into one
  row. The SDS calls for uniform batches assume `(n, nbatch)` right-hand sides
  and the non-uniform batch assumes `BatchedDirectSolver` (PLAN §3); adjust
  `compare.jl` when T17/T29 fix the API.
* **Adding a feature.** One `Feature(...)` entry in `features.jl` (task id,
  structure, matrix selector, parameters); `compare.jl` handles the kinds
  `:single`, `:ubatch`, `:nubatch` and `:schur`.

## What is measured

For each matrix and structure, after one warm-up run, five runs each of
analysis, factorization, refactorization (same values) and solve (`nrhs = 1`,
`b = rand(n)` with seed 666). A fresh solver is created per run (not timed);
GPU phases are bracketed by `CUDA.synchronize()`. The CSV holds the medians in
seconds, `lu_nnz`, `flops` (cuDSS `CUDSS_DATA_FLOPS`, an `Int64` read through the C API
since CUDSS.jl has no getter) and `nsuperpanels`, the relative residual of the
last solve and a status (`ok` or the error message; a failing matrix does not
stop the run). Symmetric structures pass the lower triangle (view `'L'`).

The KKT dumps store the lower triangle (MatrixMarket `symmetric`) with
explicitly stored zeros kept, so all iterations of a case share one pattern
(refactorization workload). K2 dumps run as `S`, condensed dumps as `SPD` and `S`.
The file name records the iteration of the last factorization; if MadNLP
converges before the requested iteration, the file is named after the
iteration where it stopped.
